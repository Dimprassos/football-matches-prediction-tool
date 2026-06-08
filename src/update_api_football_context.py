"""Fetch team-news context (lineups, injuries, suspensions) from API-Football.

Pulls per-fixture lineup availability and absence/injury/suspension counts and
writes them into the external ``match_context.csv``. Requires an API-Football key and
is rate-limited on the free plan (see the printed hints); entirely optional — without
it the context features fall back to neutral defaults. Driven by
``scripts/update_team_news.py``.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import sys
import time
import unicodedata
from dataclasses import dataclass
from dataclasses import field
from datetime import date, datetime, timedelta, timezone
from difflib import SequenceMatcher
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd
import requests

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.external_context import EXTERNAL_CONTEXT_VALUE_COLUMNS, MATCH_CONTEXT_FILE
from src.team_names import normalize_team_name


RAW_DIR = PROJECT_ROOT / "data" / "raw"
API_FOOTBALL_CACHE_DIR = PROJECT_ROOT / "data" / "external" / "api_football_cache"
API_FOOTBALL_BASE_URL = "https://v3.football.api-sports.io"

API_FOOTBALL_LEAGUES = {
    "england": 39,
    "spain": 140,
    "italy": 135,
    "germany": 78,
    "france": 61,
}

API_CONTEXT_COLUMNS = [
    "lineup_available",
    "home_lineup_strength",
    "away_lineup_strength",
    "team_news_available",
    "home_absence_count",
    "away_absence_count",
    "home_injury_count",
    "away_injury_count",
    "home_suspension_count",
    "away_suspension_count",
    "home_key_absence_count",
    "away_key_absence_count",
]

TRACE_COLUMNS = [
    "api_football_fixture_id",
    "api_football_home_id",
    "api_football_away_id",
    "lineup_source",
    "team_news_source",
    "api_football_updated_utc",
]


def _parse_dates(series: pd.Series) -> pd.Series:
    parsed_short = pd.to_datetime(series, format="%d/%m/%y", errors="coerce")
    parsed_long = pd.to_datetime(series, format="%d/%m/%Y", errors="coerce")
    parsed = parsed_short.fillna(parsed_long)
    missing = parsed.isna()
    if missing.any():
        parsed.loc[missing] = pd.to_datetime(series.loc[missing], dayfirst=True, errors="coerce")
    return parsed.dt.normalize()


def _date_from_iso(value: str) -> date:
    return date.fromisoformat(str(value)[:10])


def _api_season_for_date(match_date: date) -> int:
    return match_date.year if match_date.month >= 7 else match_date.year - 1


def _iter_raw_match_files(leagues: Iterable[str]) -> Iterable[tuple[str, Path]]:
    for league in leagues:
        league_dir = RAW_DIR / league
        if not league_dir.exists():
            continue
        for path in sorted(league_dir.glob("*.csv")):
            yield league, path


def _is_played(row: pd.Series) -> bool:
    home_goals = pd.to_numeric(row.get("FTHG", np.nan), errors="coerce")
    away_goals = pd.to_numeric(row.get("FTAG", np.nan), errors="coerce")
    return bool(np.isfinite(home_goals) and np.isfinite(away_goals))


def collect_local_matches(
    *,
    leagues: Iterable[str] = API_FOOTBALL_LEAGUES.keys(),
    from_date: date,
    to_date: date,
    include_played: bool = False,
) -> pd.DataFrame:
    rows = []
    for league, path in _iter_raw_match_files(leagues):
        raw = pd.read_csv(path)
        required = {"Date", "HomeTeam", "AwayTeam"}
        if not required.issubset(raw.columns):
            continue

        parsed_dates = _parse_dates(raw["Date"])
        for idx, row in raw.iterrows():
            parsed = parsed_dates.loc[idx]
            if pd.isna(parsed):
                continue
            match_date = pd.Timestamp(parsed).date()
            if match_date < from_date or match_date > to_date:
                continue
            played = _is_played(row)
            if played and not include_played:
                continue

            rows.append({
                "date": match_date.isoformat(),
                "league": league,
                "api_season": _api_season_for_date(match_date),
                "home_team": normalize_team_name(row["HomeTeam"], league),
                "away_team": normalize_team_name(row["AwayTeam"], league),
                "is_played": played,
            })

    if not rows:
        return pd.DataFrame(columns=["date", "league", "api_season", "home_team", "away_team", "is_played"])

    return (
        pd.DataFrame(rows)
        .drop_duplicates(["date", "league", "home_team", "away_team"], keep="last")
        .sort_values(["date", "league", "home_team", "away_team"])
        .reset_index(drop=True)
    )


def _cache_path(cache_dir: Path, endpoint: str, params: dict) -> Path:
    clean_endpoint = endpoint.strip("/").replace("/", "_") or "root"
    normalized_params = {str(k): str(v) for k, v in sorted(params.items()) if v is not None}
    payload = json.dumps({"endpoint": endpoint, "params": normalized_params}, sort_keys=True)
    digest = hashlib.sha1(payload.encode("utf-8")).hexdigest()
    return cache_dir / clean_endpoint / f"{digest}.json"


@dataclass
class ApiFootballClient:
    api_key: str | None
    cache_dir: Path = API_FOOTBALL_CACHE_DIR
    refresh_cache: bool = False
    cache_only: bool = False
    timeout: int = 30
    plan_errors: list[dict] = field(default_factory=list)
    rate_limit_errors: list[dict] = field(default_factory=list)

    def get(self, endpoint: str, params: dict) -> dict:
        cache_file = _cache_path(self.cache_dir, endpoint, params)
        if cache_file.exists() and not self.refresh_cache:
            return json.loads(cache_file.read_text(encoding="utf-8"))

        if self.cache_only:
            raise FileNotFoundError(f"Missing API-Football cache file for {endpoint} {params}: {cache_file}")
        if not self.api_key:
            raise RuntimeError(
                "Missing API-Football key. Set API_FOOTBALL_KEY or API_SPORTS_KEY, "
                "or rerun with --cache-only when cache files already exist."
            )

        response = requests.get(
            f"{API_FOOTBALL_BASE_URL}{endpoint}",
            headers={"x-apisports-key": self.api_key},
            params=params,
            timeout=self.timeout,
        )
        if response.status_code == 429:
            try:
                payload = response.json()
            except ValueError:
                payload = {"errors": {"rate_limit": response.text}}
            self.rate_limit_errors.append({
                "endpoint": endpoint,
                "params": dict(params),
                "status_code": response.status_code,
                "retry_after": response.headers.get("Retry-After"),
                "payload": payload,
            })
            return payload

        response.raise_for_status()
        payload = response.json()
        cache_file.parent.mkdir(parents=True, exist_ok=True)
        cache_file.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")
        return payload

    def response(self, endpoint: str, params: dict) -> list:
        payload = self.get(endpoint, params)
        errors = payload.get("errors")
        if errors:
            self._remember_plan_error(endpoint, params, errors)
            print(f"[WARN] API-Football errors for {endpoint} {params}: {errors}")
        data = payload.get("response", [])
        return data if isinstance(data, list) else []

    def _remember_plan_error(self, endpoint: str, params: dict, errors: dict) -> None:
        text = json.dumps(errors, sort_keys=True).lower()
        if "free plans do not have access to this season" not in text:
            return
        self.plan_errors.append({
            "endpoint": endpoint,
            "params": dict(params),
            "errors": errors,
        })


def _strip_accents(value: str) -> str:
    normalized = unicodedata.normalize("NFKD", value)
    return "".join(ch for ch in normalized if not unicodedata.combining(ch))


TEAM_KEY_ALIASES = {
    "manunited": "manchesterunited",
    "manutd": "manchesterunited",
    "manchesterutd": "manchesterunited",
    "mancity": "manchestercity",
    "nottmforest": "nottinghamforest",
    "tottenham": "tottenhamhotspur",
    "spurs": "tottenhamhotspur",
    "newcastle": "newcastleunited",
    "westham": "westhamunited",
    "wolves": "wolverhamptonwanderers",
    "brighton": "brightonhovealbion",
    "leeds": "leedsunited",
    "leicester": "leicestercity",
    "athbilbao": "athleticbilbao",
    "athleticclub": "athleticbilbao",
    "athmadrid": "atleticomadrid",
    "atleticomadrid": "atleticomadrid",
    "sociedad": "realsociedad",
    "betis": "realbetis",
    "vallecano": "rayovallecano",
    "espanol": "espanyol",
    "fcinter": "inter",
    "intermilan": "inter",
    "internazionale": "inter",
    "acmilan": "milan",
    "asroma": "roma",
    "sslazio": "lazio",
    "parissg": "psg",
    "parissaintgermain": "psg",
    "stetienne": "saintetienne",
}


def team_match_key(name: str, league: str) -> str:
    normalized = normalize_team_name(name, league)
    stripped = _strip_accents(str(normalized)).lower()
    stripped = stripped.replace("&", "and")
    compact = re.sub(r"[^a-z0-9]+", "", stripped)
    compact = TEAM_KEY_ALIASES.get(compact, compact)
    return compact


def _team_similarity(a: str, b: str, league: str) -> float:
    key_a = team_match_key(a, league)
    key_b = team_match_key(b, league)
    if key_a == key_b:
        return 1.0
    return SequenceMatcher(None, key_a, key_b).ratio()


def fetch_api_fixtures(
    client: ApiFootballClient,
    local_matches: pd.DataFrame,
    *,
    pause_seconds: float = 0.0,
) -> pd.DataFrame:
    rows = []
    if local_matches.empty:
        return pd.DataFrame(columns=[
            "date", "league", "api_fixture_id", "api_home_team", "api_away_team", "api_home_id", "api_away_id"
        ])

    grouped = local_matches.groupby(["league", "api_season"], dropna=False)
    for (league, api_season), group in grouped:
        league_id = API_FOOTBALL_LEAGUES.get(str(league))
        if league_id is None:
            continue
        params = {
            "league": league_id,
            "season": int(api_season),
            "from": str(group["date"].min()),
            "to": str(group["date"].max()),
            "timezone": "UTC",
        }
        api_rows = client.response("/fixtures", params)
        if client.rate_limit_errors:
            print_rate_limit_hint(client.rate_limit_errors)
            break
        for item in api_rows:
            fixture = item.get("fixture", {}) or {}
            teams = item.get("teams", {}) or {}
            home = teams.get("home", {}) or {}
            away = teams.get("away", {}) or {}
            fixture_date = fixture.get("date")
            fixture_id = fixture.get("id")
            if fixture_id is None or not fixture_date:
                continue
            rows.append({
                "date": str(fixture_date)[:10],
                "league": league,
                "api_fixture_id": int(fixture_id),
                "api_home_team": home.get("name"),
                "api_away_team": away.get("name"),
                "api_home_id": home.get("id"),
                "api_away_id": away.get("id"),
            })
        if api_rows:
            print(f"[OK] {league} {api_season}: fetched {len(api_rows)} API fixtures")
        else:
            print(f"[WARN] {league} {api_season}: no API fixtures returned")
        if pause_seconds > 0:
            time.sleep(pause_seconds)

    if not rows:
        return pd.DataFrame(columns=[
            "date", "league", "api_fixture_id", "api_home_team", "api_away_team", "api_home_id", "api_away_id"
        ])
    return pd.DataFrame(rows).drop_duplicates(["league", "api_fixture_id"], keep="last")


def print_plan_error_hint(plan_errors: list[dict]) -> None:
    if not plan_errors:
        return

    seasons = sorted({str(item["params"].get("season")) for item in plan_errors if item.get("params")})
    print("\nAPI-Football free plan blocked the requested season(s): " + ", ".join(seasons))
    print("Your free account says it can access API seasons 2022 to 2024, not the current 2025 season.")
    print("So upcoming 2025-2026 injuries/lineups need a paid API-Football plan or another source.")
    print("For free historical training data, run:")
    print("  python scripts/update_team_news.py --free-training")


def print_rate_limit_hint(rate_limit_errors: list[dict]) -> None:
    if not rate_limit_errors:
        return

    latest = rate_limit_errors[-1]
    retry_after = latest.get("retry_after")
    print("\nAPI-Football rate limit reached (HTTP 429).")
    if retry_after:
        print(f"Retry-After: {retry_after} seconds")
    print("Nothing is wrong with the model; the free API is refusing more requests right now.")
    print("Wait a bit, then continue with a smaller batch:")
    print("  python scripts/update_team_news.py --free-training --max 10")
    print("Rows already written are skipped on the next run.")


def skip_existing_context_rows(matched: pd.DataFrame, existing: pd.DataFrame) -> tuple[pd.DataFrame, int]:
    keys = ["date", "league", "home_team", "away_team"]
    if matched.empty or existing.empty or not set(keys).issubset(existing.columns):
        return matched, 0

    signal_cols = [
        col for col in ("api_football_fixture_id", "team_news_available", "lineup_available")
        if col in existing.columns
    ]
    if not signal_cols:
        return matched, 0

    done_mask = existing[signal_cols].notna().any(axis=1)
    done = existing.loc[done_mask, keys].drop_duplicates()
    if done.empty:
        return matched, 0

    flagged = matched.merge(done.assign(_already_context=1), on=keys, how="left")
    skipped = int(flagged["_already_context"].notna().sum())
    remaining = flagged[flagged["_already_context"].isna()].drop(columns=["_already_context"])
    return remaining.reset_index(drop=True), skipped


def match_api_fixtures(
    local_matches: pd.DataFrame,
    api_fixtures: pd.DataFrame,
    *,
    min_score: float = 0.72,
) -> pd.DataFrame:
    rows = []
    for _, local in local_matches.iterrows():
        candidates = api_fixtures[
            (api_fixtures["date"] == local["date"]) & (api_fixtures["league"] == local["league"])
        ]
        best = None
        best_score = -1.0
        for _, candidate in candidates.iterrows():
            home_score = _team_similarity(local["home_team"], candidate["api_home_team"], local["league"])
            away_score = _team_similarity(local["away_team"], candidate["api_away_team"], local["league"])
            score = (home_score + away_score) / 2.0
            if score > best_score:
                best = candidate
                best_score = score

        out = local.to_dict()
        out["match_score"] = round(float(best_score), 4) if best is not None else np.nan
        if best is not None and best_score >= min_score:
            out.update({
                "api_fixture_id": int(best["api_fixture_id"]),
                "api_home_id": best.get("api_home_id"),
                "api_away_id": best.get("api_away_id"),
                "api_home_team": best.get("api_home_team"),
                "api_away_team": best.get("api_away_team"),
                "api_match_status": "matched",
            })
        else:
            out.update({
                "api_fixture_id": np.nan,
                "api_home_id": np.nan,
                "api_away_id": np.nan,
                "api_home_team": np.nan,
                "api_away_team": np.nan,
                "api_match_status": "unmatched",
            })
        rows.append(out)
    return pd.DataFrame(rows)


def _team_side_by_id(team_id, home_id, away_id) -> str | None:
    try:
        team_id_int = int(team_id)
        home_id_int = int(home_id)
        away_id_int = int(away_id)
    except (TypeError, ValueError):
        return None
    if team_id_int == home_id_int:
        return "home"
    if team_id_int == away_id_int:
        return "away"
    return None


def lineup_fields(lineups: list, *, home_id, away_id) -> dict:
    fields = {
        "lineup_available": 0.0,
        "home_lineup_strength": np.nan,
        "away_lineup_strength": np.nan,
        "lineup_source": "api-football",
    }
    for item in lineups:
        team = item.get("team", {}) or {}
        side = _team_side_by_id(team.get("id"), home_id, away_id)
        if side is None:
            continue
        start_xi = item.get("startXI") or []
        starter_count = len(start_xi)
        if starter_count > 0:
            fields["lineup_available"] = 1.0
            fields[f"{side}_lineup_strength"] = float(min(starter_count, 11) / 11.0)
    return fields


def _text_contains_suspension(*values) -> bool:
    text = " ".join("" if value is None else str(value) for value in values).lower()
    return "suspend" in text or "ban" in text or "red card" in text


def injury_fields(injuries: list, *, home_id, away_id) -> dict:
    counts = {
        "home_injury_count": 0.0,
        "away_injury_count": 0.0,
        "home_suspension_count": 0.0,
        "away_suspension_count": 0.0,
        "home_key_absence_count": 0.0,
        "away_key_absence_count": 0.0,
    }
    seen = set()
    for item in injuries:
        team = item.get("team", {}) or {}
        player = item.get("player", {}) or {}
        side = _team_side_by_id(team.get("id"), home_id, away_id)
        if side is None:
            continue
        player_key = player.get("id") or player.get("name") or json.dumps(player, sort_keys=True)
        unique_key = (side, player_key)
        if unique_key in seen:
            continue
        seen.add(unique_key)
        if _text_contains_suspension(player.get("type"), player.get("reason"), item.get("type"), item.get("reason")):
            counts[f"{side}_suspension_count"] += 1.0
        else:
            counts[f"{side}_injury_count"] += 1.0

    counts["home_absence_count"] = counts["home_injury_count"] + counts["home_suspension_count"]
    counts["away_absence_count"] = counts["away_injury_count"] + counts["away_suspension_count"]
    counts["team_news_available"] = 1.0
    counts["team_news_source"] = "api-football"
    return counts


def _should_fetch_lineups(match_date: str, *, today: date, lineup_window_days: int) -> bool:
    parsed = _date_from_iso(match_date)
    return parsed <= today + timedelta(days=lineup_window_days)


def build_api_context(
    client: ApiFootballClient,
    matched: pd.DataFrame,
    *,
    fetch_lineups: bool = True,
    fetch_injuries: bool = True,
    lineup_window_days: int = 0,
    pause_seconds: float = 0.1,
    max_fixtures: int | None = None,
) -> pd.DataFrame:
    rows = []
    updated_utc = datetime.now(timezone.utc).isoformat()
    matched_rows = matched[matched["api_match_status"] == "matched"].copy()
    if max_fixtures is not None:
        matched_rows = matched_rows.head(max_fixtures)
    today = date.today()

    for _, match in matched_rows.iterrows():
        rate_errors_before = len(client.rate_limit_errors)
        fixture_id = int(match["api_fixture_id"])
        row = {
            "date": match["date"],
            "league": match["league"],
            "home_team": match["home_team"],
            "away_team": match["away_team"],
            "api_football_fixture_id": fixture_id,
            "api_football_home_id": match.get("api_home_id"),
            "api_football_away_id": match.get("api_away_id"),
            "api_football_updated_utc": updated_utc,
        }

        for col in API_CONTEXT_COLUMNS:
            row[col] = np.nan

        if fetch_lineups and _should_fetch_lineups(match["date"], today=today, lineup_window_days=lineup_window_days):
            lineups = client.response("/fixtures/lineups", {"fixture": fixture_id})
            if len(client.rate_limit_errors) > rate_errors_before:
                print_rate_limit_hint(client.rate_limit_errors)
                break
            row.update(lineup_fields(lineups, home_id=match.get("api_home_id"), away_id=match.get("api_away_id")))
        else:
            row["lineup_available"] = 0.0
            row["lineup_source"] = "api-football-skipped"

        if fetch_injuries:
            injuries = client.response("/injuries", {"fixture": fixture_id})
            if len(client.rate_limit_errors) > rate_errors_before:
                print_rate_limit_hint(client.rate_limit_errors)
                break
            row.update(injury_fields(injuries, home_id=match.get("api_home_id"), away_id=match.get("api_away_id")))
        else:
            row["team_news_available"] = 0.0
            row["team_news_source"] = "api-football-skipped"

        rows.append(row)
        print(f"[OK] {match['league']} {match['date']} {match['home_team']} - {match['away_team']}: fixture {fixture_id}")
        if pause_seconds > 0:
            time.sleep(pause_seconds)

    columns = ["date", "league", "home_team", "away_team", *API_CONTEXT_COLUMNS, *TRACE_COLUMNS]
    if not rows:
        return pd.DataFrame(columns=columns)
    return pd.DataFrame(rows, columns=columns)


def merge_match_context(existing: pd.DataFrame, incoming: pd.DataFrame) -> pd.DataFrame:
    keys = ["date", "league", "home_team", "away_team"]
    columns = [col for col in [*EXTERNAL_CONTEXT_VALUE_COLUMNS, *TRACE_COLUMNS] if col in incoming.columns]
    if existing.empty:
        return incoming.copy()
    if incoming.empty:
        return existing.copy()

    merged = existing.merge(incoming[keys + columns], on=keys, how="outer", suffixes=("", "_api"))
    for col in columns:
        api_col = f"{col}_api"
        if api_col in merged.columns:
            merged[col] = merged[api_col].combine_first(merged.get(col))
            merged = merged.drop(columns=[api_col])
    return merged.drop_duplicates(keys, keep="last").sort_values(keys).reset_index(drop=True)


def update_api_football_context(
    *,
    from_date: date,
    to_date: date,
    leagues: Iterable[str] = API_FOOTBALL_LEAGUES.keys(),
    output_file: Path = MATCH_CONTEXT_FILE,
    cache_dir: Path = API_FOOTBALL_CACHE_DIR,
    api_key: str | None = None,
    include_played: bool = False,
    refresh_cache: bool = False,
    cache_only: bool = False,
    fetch_lineups: bool = True,
    fetch_injuries: bool = True,
    lineup_window_days: int = 0,
    pause_seconds: float = 0.1,
    max_fixtures: int | None = None,
    dry_run: bool = False,
    skip_existing: bool = True,
) -> pd.DataFrame:
    local_matches = collect_local_matches(
        leagues=leagues,
        from_date=from_date,
        to_date=to_date,
        include_played=include_played,
    )
    if local_matches.empty:
        print("[WARN] No local matches found for the requested date window.")
        return pd.DataFrame()

    client = ApiFootballClient(
        api_key=api_key,
        cache_dir=cache_dir,
        refresh_cache=refresh_cache,
        cache_only=cache_only,
    )
    api_fixtures = fetch_api_fixtures(client, local_matches, pause_seconds=pause_seconds)
    if api_fixtures.empty and client.rate_limit_errors:
        return pd.DataFrame()
    if api_fixtures.empty and client.plan_errors:
        print_plan_error_hint(client.plan_errors)
        return pd.DataFrame()

    matched = match_api_fixtures(local_matches, api_fixtures)
    unmatched = matched[matched["api_match_status"] != "matched"]
    if not unmatched.empty:
        missing_path = output_file.parent / "api_football_unmatched_fixtures.csv"
        output_file.parent.mkdir(parents=True, exist_ok=True)
        unmatched.to_csv(missing_path, index=False)
        print(f"[WARN] Unmatched local fixtures: {len(unmatched)} -> {missing_path}")

    existing = pd.read_csv(output_file) if output_file.exists() else pd.DataFrame()
    if skip_existing:
        matched, skipped = skip_existing_context_rows(matched, existing)
        if skipped:
            print(f"[INFO] Skipped {skipped} fixtures already present in {output_file.name}")

    context = build_api_context(
        client,
        matched,
        fetch_lineups=fetch_lineups,
        fetch_injuries=fetch_injuries,
        lineup_window_days=lineup_window_days,
        pause_seconds=pause_seconds,
        max_fixtures=max_fixtures,
    )

    if dry_run:
        print(context.to_string(index=False) if not context.empty else "[WARN] No context rows built.")
        return context

    if context.empty and client.plan_errors:
        print_plan_error_hint(client.plan_errors)
    if context.empty and client.rate_limit_errors:
        print_rate_limit_hint(client.rate_limit_errors)
    if context.empty and not existing.empty:
        print("[INFO] No new context rows to write.")
        return existing

    merged = merge_match_context(existing, context)
    output_file.parent.mkdir(parents=True, exist_ok=True)
    merged.to_csv(output_file, index=False)
    return merged


def _default_api_key() -> str | None:
    return os.getenv("API_FOOTBALL_KEY") or os.getenv("API_SPORTS_KEY")


def parse_args() -> argparse.Namespace:
    today = date.today()
    parser = argparse.ArgumentParser(
        description="Fetch API-Football lineups/injuries into data/external/match_context.csv."
    )
    parser.add_argument("--from-date", type=_date_from_iso, default=today)
    parser.add_argument("--to-date", type=_date_from_iso, default=None)
    parser.add_argument("--days", type=int, default=7, help="Used only when --to-date is omitted.")
    parser.add_argument("--leagues", nargs="+", default=list(API_FOOTBALL_LEAGUES.keys()))
    parser.add_argument("--output", type=Path, default=MATCH_CONTEXT_FILE)
    parser.add_argument("--cache-dir", type=Path, default=API_FOOTBALL_CACHE_DIR)
    parser.add_argument("--api-key", default=None, help="Defaults to API_FOOTBALL_KEY or API_SPORTS_KEY.")
    parser.add_argument("--include-played", action="store_true", help="Also enrich already played matches in the window.")
    parser.add_argument("--refresh-cache", action="store_true", help="Ignore cached API responses and fetch again.")
    parser.add_argument("--cache-only", action="store_true", help="Read only cached API responses; do not call the API.")
    parser.add_argument("--skip-lineups", action="store_true")
    parser.add_argument("--skip-injuries", action="store_true")
    parser.add_argument(
        "--lineup-window-days",
        type=int,
        default=0,
        help="Fetch lineups only for matches up to this many days after today. Injuries are fetched for every matched fixture.",
    )
    parser.add_argument("--pause-seconds", type=float, default=0.1)
    parser.add_argument("--max-fixtures", type=int, default=None)
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    to_date = args.to_date or (args.from_date + timedelta(days=max(0, int(args.days))))
    started = datetime.now()
    df = update_api_football_context(
        from_date=args.from_date,
        to_date=to_date,
        leagues=args.leagues,
        output_file=args.output,
        cache_dir=args.cache_dir,
        api_key=args.api_key or _default_api_key(),
        include_played=args.include_played,
        refresh_cache=args.refresh_cache,
        cache_only=args.cache_only,
        fetch_lineups=not args.skip_lineups,
        fetch_injuries=not args.skip_injuries,
        lineup_window_days=args.lineup_window_days,
        pause_seconds=args.pause_seconds,
        max_fixtures=args.max_fixtures,
        dry_run=args.dry_run,
    )
    if not df.empty:
        print(f"\nSaved {len(df)} context rows -> {args.output}")
    print(f"Elapsed: {datetime.now() - started}")


if __name__ == "__main__":
    main()
