"""Refresh the local Understat expected-goals (xG) and player-context exports.

Two related datasets are produced from Understat:

* Team-match xG/npxG/xpts -> ``data/external/understat_matches.csv`` (the default),
  which :func:`src.understat_data.add_understat_xg` later merges in.
* Optional per-player per-match participation stats (``--players``) ->
  ``data/external/player_match_stats.csv`` plus a derived
  ``data/external/player_registry.csv``. These match the schemas in
  :mod:`src.player_context` and feed the leakage-safe rolling player-strength path.

Both are optional/offline-friendly. Run with ``python src/update_understat.py`` for the
team-level export, or add ``--players`` to also build the player datasets. Player-match
JSON responses are cached on disk so reruns are resumable and polite to Understat.
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd
import requests


PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

# Reuse the canonical player-context schemas so the producer cannot drift from the
# loader/validator in src.player_context.
from src.player_context import PLAYER_MATCH_STATS_COLUMNS, PLAYER_REGISTRY_COLUMNS

EXTERNAL_DIR = PROJECT_ROOT / "data" / "external"
OUTPUT_FILE = EXTERNAL_DIR / "understat_matches.csv"
PLAYER_MATCH_OUTPUT_FILE = EXTERNAL_DIR / "player_match_stats.csv"
PLAYER_REGISTRY_OUTPUT_FILE = EXTERNAL_DIR / "player_registry.csv"
MATCH_CACHE_DIR = EXTERNAL_DIR / "understat_match_cache"
UNDERSTAT_BASE_URL = "https://understat.com"

UNDERSTAT_LEAGUES = {
    "EPL": "EPL",
    "La_liga": "La liga",
    "Bundesliga": "Bundesliga",
    "Serie_A": "Serie A",
    "Ligue_1": "Ligue 1",
}

# Map Understat league codes to the pipeline's canonical league names. Used to load
# the main dataset team names for the Understat-title -> our-name bridge (Understat
# uses long club names like "Manchester City"; the pipeline uses "Man City").
UNDERSTAT_LEAGUE_TO_CANONICAL = {
    "EPL": "england",
    "La_liga": "spain",
    "Bundesliga": "germany",
    "Serie_A": "italy",
    "Ligue_1": "france",
}

OUTPUT_COLUMNS = [
    "id",
    "league",
    "season",
    "club_name",
    "home_away",
    "xG",
    "xGA",
    "npxG",
    "npxGA",
    "ppda",
    "ppda_allowed",
    "deep",
    "deep_allowed",
    "scored",
    "missed",
    "xpts",
    "result",
    "date",
    "wins",
    "draws",
    "loses",
    "pts",
    "npxGD",
]


def current_season_start_year(today: datetime | None = None) -> int:
    if today is None:
        today = datetime.now()
    return today.year if today.month >= 7 else today.year - 1


def _headers(referer: str | None = None) -> dict[str, str]:
    headers = {
        "User-Agent": (
            "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
            "AppleWebKit/537.36 (KHTML, like Gecko) "
            "Chrome/124.0 Safari/537.36"
        ),
        "Accept": "application/json, text/javascript, */*; q=0.01",
        "X-Requested-With": "XMLHttpRequest",
    }
    if referer is not None:
        headers["Referer"] = referer
    return headers


def _numeric(value) -> float:
    value = pd.to_numeric(value, errors="coerce")
    if np.isfinite(value):
        return float(value)
    return np.nan


def _ppda_ratio(value) -> float:
    if isinstance(value, dict):
        att = _numeric(value.get("att"))
        defense = _numeric(value.get("def"))
        if np.isfinite(att) and np.isfinite(defense) and defense != 0:
            return float(att / defense)
        return np.nan
    return _numeric(value)


def _history_to_rows(league_name: str, season: int, team: dict) -> list[dict]:
    rows = []
    for match in team.get("history", []):
        rows.append({
            "id": team.get("id"),
            "league": league_name,
            "season": season,
            "club_name": team.get("title"),
            "home_away": match.get("h_a"),
            "xG": _numeric(match.get("xG")),
            "xGA": _numeric(match.get("xGA")),
            "npxG": _numeric(match.get("npxG")),
            "npxGA": _numeric(match.get("npxGA")),
            "ppda": _ppda_ratio(match.get("ppda")),
            "ppda_allowed": _ppda_ratio(match.get("ppda_allowed")),
            "deep": _numeric(match.get("deep")),
            "deep_allowed": _numeric(match.get("deep_allowed")),
            "scored": _numeric(match.get("scored")),
            "missed": _numeric(match.get("missed")),
            "xpts": _numeric(match.get("xpts")),
            "result": match.get("result"),
            "date": match.get("date"),
            "wins": _numeric(match.get("wins")),
            "draws": _numeric(match.get("draws")),
            "loses": _numeric(match.get("loses")),
            "pts": _numeric(match.get("pts")),
            "npxGD": _numeric(match.get("npxGD")),
        })
    return rows


def fetch_league_season(
    session: requests.Session,
    league_code: str,
    league_name: str,
    season: int,
    *,
    timeout: int = 30,
) -> list[dict]:
    page_url = f"{UNDERSTAT_BASE_URL}/league/{league_code}/{season}"
    api_url = f"{UNDERSTAT_BASE_URL}/getLeagueData/{league_code}/{season}"

    page_response = session.get(page_url, headers=_headers(), timeout=timeout)
    page_response.raise_for_status()

    response = session.get(api_url, headers=_headers(page_url), timeout=timeout)
    response.raise_for_status()

    payload = response.json()
    teams = payload.get("teams", {})

    rows = []
    for team in teams.values():
        rows.extend(_history_to_rows(league_name, season, team))
    return rows


def build_understat_matches(
    league_codes: Iterable[str],
    seasons: Iterable[int],
    *,
    pause_seconds: float = 0.2,
) -> pd.DataFrame:
    session = requests.Session()
    rows = []

    for league_code in league_codes:
        league_name = UNDERSTAT_LEAGUES[league_code]
        for season in seasons:
            fetched = fetch_league_season(session, league_code, league_name, int(season))
            rows.extend(fetched)
            print(f"[OK] {league_name} {season}: {len(fetched)} team-match rows")
            if pause_seconds > 0:
                time.sleep(pause_seconds)

    if not rows:
        raise RuntimeError("No Understat rows were downloaded.")

    df = pd.DataFrame(rows, columns=OUTPUT_COLUMNS)
    df["date"] = pd.to_datetime(df["date"], errors="coerce")
    df = df.dropna(subset=["date", "club_name", "home_away"])
    df = df.sort_values(["league", "season", "date", "club_name", "home_away"]).reset_index(drop=True)
    df["date"] = df["date"].dt.strftime("%Y-%m-%d %H:%M:%S")
    return df


def write_understat_matches(df: pd.DataFrame, output_file: Path = OUTPUT_FILE) -> None:
    output_file.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(output_file, index=False)


def sync_understat_matches(
    *,
    start_season: int = 2014,
    end_season: int | None = None,
    output_file: Path = OUTPUT_FILE,
    pause_seconds: float = 0.2,
) -> pd.DataFrame:
    if end_season is None:
        end_season = current_season_start_year()
    if end_season < start_season:
        raise ValueError("end_season must be greater than or equal to start_season")

    seasons = range(start_season, end_season + 1)
    df = build_understat_matches(
        UNDERSTAT_LEAGUES.keys(),
        seasons,
        pause_seconds=pause_seconds,
    )
    write_understat_matches(df, output_file)
    return df


# --------------------------------------------------------------------------- #
# Per-player per-match stats (optional, --players)                            #
# --------------------------------------------------------------------------- #
def _truthy(value) -> bool:
    """Interpret Understat ``isResult`` flags (bool/str/int) as a finished match."""
    return str(value).strip().lower() in {"true", "1", "yes"}


def fetch_season_match_list(
    session: requests.Session,
    league_code: str,
    season: int,
    *,
    timeout: int = 30,
) -> list[dict]:
    """Return finished matches for a league-season as lightweight metadata dicts.

    Uses the same ``getLeagueData`` endpoint as the team-level sync, but reads its
    ``dates`` block (one entry per fixture) instead of team histories.
    """
    page_url = f"{UNDERSTAT_BASE_URL}/league/{league_code}/{season}"
    api_url = f"{UNDERSTAT_BASE_URL}/getLeagueData/{league_code}/{season}"

    session.get(page_url, headers=_headers(), timeout=timeout).raise_for_status()
    response = session.get(api_url, headers=_headers(page_url), timeout=timeout)
    response.raise_for_status()

    matches = []
    for match in response.json().get("dates", []):
        if not _truthy(match.get("isResult")):
            continue
        goals = match.get("goals") or {}
        home = match.get("h") or {}
        away = match.get("a") or {}
        parsed_date = pd.to_datetime(match.get("datetime"), errors="coerce")
        matches.append({
            "id": str(match.get("id")),
            "date": parsed_date.strftime("%Y-%m-%d") if pd.notna(parsed_date) else None,
            "home_title": home.get("title"),
            "away_title": away.get("title"),
            "home_goals": goals.get("h"),
            "away_goals": goals.get("a"),
        })
    return matches


def fetch_match_data(
    session: requests.Session,
    match_id: str,
    *,
    cache_dir: Path = MATCH_CACHE_DIR,
    pause_seconds: float = 0.2,
    timeout: int = 30,
    use_cache: bool = True,
    max_retries: int = 3,
) -> dict:
    """Fetch one match's ``getMatchData`` payload (``rosters``/``shots``), with disk cache.

    Cached responses make the bulk download resumable: only un-cached match ids hit
    the network, and the polite ``pause_seconds`` delay is applied solely after a real
    request (never on a cache hit). Transient network/JSON errors are retried a few
    times with a simple backoff before giving up.
    """
    cache_file = cache_dir / f"{match_id}.json"
    if use_cache and cache_file.exists():
        try:
            return json.loads(cache_file.read_text(encoding="utf-8"))
        except (json.JSONDecodeError, OSError):
            pass  # corrupt/partial cache -> refetch below

    url = f"{UNDERSTAT_BASE_URL}/getMatchData/{match_id}"
    referer = f"{UNDERSTAT_BASE_URL}/match/{match_id}"
    last_error: Exception | None = None
    for attempt in range(max_retries):
        try:
            response = session.get(url, headers=_headers(referer), timeout=timeout)
            response.raise_for_status()
            data = response.json()
            cache_dir.mkdir(parents=True, exist_ok=True)
            cache_file.write_text(json.dumps(data), encoding="utf-8")
            if pause_seconds > 0:
                time.sleep(pause_seconds)
            return data
        except (requests.RequestException, json.JSONDecodeError) as exc:
            last_error = exc
            time.sleep(pause_seconds * (attempt + 1) + 0.5)  # backoff before retry
    raise RuntimeError(f"failed to fetch match {match_id} after {max_retries} attempts: {last_error}")


def build_team_name_bridge(canonical_league: str | None) -> dict[str, str]:
    """Map Understat team titles to the main dataset's own team names for a league.

    Understat uses long club names ("Manchester City") while the football-data
    pipeline uses short ones ("Man City"); both collapse to the same aggressive join
    key via :func:`src.understat_data.normalize_team_name`. We key the main dataset's
    (already pipeline-normalized) team names by that join key, so an Understat title
    can be translated to the exact name the player-context path will compare against.
    Returns an empty bridge (i.e. no translation) if the league cannot be loaded.
    """
    if not canonical_league:
        return {}
    try:  # local imports avoid import-time coupling/circularity for the script
        from src.data_processing import load_league_data
        from src.understat_data import normalize_team_name as understat_join_key
    except Exception:
        return {}
    try:
        league_df = load_league_data(canonical_league)
    except Exception:
        return {}

    bridge: dict[str, str] = {}
    for col in ("home_team", "away_team"):
        if col in league_df.columns:
            for name in league_df[col].dropna():
                if isinstance(name, str) and name:
                    bridge.setdefault(understat_join_key(name), name)
    return bridge


def _bridge_team_name(title, team_name_bridge: dict[str, str] | None):
    """Translate one Understat title through the bridge, or return it unchanged."""
    if not title or not team_name_bridge:
        return title
    from src.understat_data import normalize_team_name as understat_join_key
    return team_name_bridge.get(understat_join_key(title), title)


def _match_player_observations(
    league_name: str,
    season: int,
    match_meta: dict,
    rosters: dict,
    team_name_bridge: dict[str, str] | None = None,
) -> list[dict]:
    """Flatten one match's home/away rosters into per-player observation rows.

    Understat titles are translated to the main dataset's team names via
    ``team_name_bridge`` (so the rolling-strength join lines up); the loader in
    :mod:`src.player_context` then re-normalizes them harmlessly. ``started`` is
    derived from the Understat ``position`` field (``"Sub"`` means an off-the-bench
    appearance), and ``minutes`` comes from ``time`` (``roster_in``/``roster_out`` are
    event ids, not minutes, so they are intentionally ignored).
    """
    home_title = _bridge_team_name(match_meta.get("home_title"), team_name_bridge)
    away_title = _bridge_team_name(match_meta.get("away_title"), team_name_bridge)
    home_goals = _numeric(match_meta.get("home_goals"))
    away_goals = _numeric(match_meta.get("away_goals"))
    date_str = match_meta.get("date")

    sides = (
        ("h", rosters.get("h") or {}, home_title, away_title, "home", home_goals, away_goals),
        ("a", rosters.get("a") or {}, away_title, home_title, "away", away_goals, home_goals),
    )

    rows = []
    for _side, players, team, opponent, home_away, team_goals, opponent_goals in sides:
        if not team or not opponent:
            continue
        team_goal_diff = (
            team_goals - opponent_goals
            if np.isfinite(team_goals) and np.isfinite(opponent_goals)
            else np.nan
        )
        for entry in players.values():
            player_id = str(entry.get("player_id") or "").strip()
            if not player_id:
                continue
            position = str(entry.get("position") or "").strip()
            minutes = _numeric(entry.get("time"))
            rows.append({
                "date": date_str,
                "league": league_name,
                "season": int(season),
                "team": team,
                "opponent": opponent,
                "home_away": home_away,
                "player_id": player_id,
                "player_name": str(entry.get("player") or "").strip(),
                "position": position,
                "started": 0 if position == "Sub" else 1,
                "minutes": float(minutes) if np.isfinite(minutes) else 0.0,
                "goals": _numeric(entry.get("goals")),
                "assists": _numeric(entry.get("assists")),
                "shots": _numeric(entry.get("shots")),
                "key_passes": _numeric(entry.get("key_passes")),
                "xg": _numeric(entry.get("xG")),
                "xa": _numeric(entry.get("xA")),
                "team_goals": team_goals,
                "opponent_goals": opponent_goals,
                "team_goal_diff": team_goal_diff,
                "source": "understat",
            })
    return rows


def build_player_match_observations(
    league_codes: Iterable[str],
    seasons: Iterable[int],
    *,
    cache_dir: Path = MATCH_CACHE_DIR,
    pause_seconds: float = 0.2,
    max_matches: int | None = None,
    use_cache: bool = True,
) -> pd.DataFrame:
    """Download per-player rows for every finished match in the given leagues/seasons.

    Returns a rich frame that still carries ``season``/``player_name``/``position``
    (used to derive the registry); :func:`player_match_stats_from_observations`
    selects the strict ``player_match_stats`` schema from it.
    """
    session = requests.Session()
    observations: list[dict] = []
    failed: list[str] = []
    processed = 0

    for league_code in league_codes:
        league_name = UNDERSTAT_LEAGUES[league_code]
        team_name_bridge = build_team_name_bridge(UNDERSTAT_LEAGUE_TO_CANONICAL.get(league_code))
        for season in seasons:
            if max_matches is not None and processed >= max_matches:
                break
            try:
                matches = fetch_season_match_list(session, league_code, int(season))
            except Exception as exc:  # whole season list failed -> skip, rerun later
                print(f"[WARN] could not list matches for {league_name} {season}: {exc}")
                continue
            if pause_seconds > 0:
                time.sleep(pause_seconds)
            season_matches = 0
            for meta in matches:
                if max_matches is not None and processed >= max_matches:
                    break
                try:
                    data = fetch_match_data(
                        session,
                        meta["id"],
                        cache_dir=cache_dir,
                        pause_seconds=pause_seconds,
                        use_cache=use_cache,
                    )
                except Exception as exc:  # transient failure -> skip; cache fills on rerun
                    failed.append(meta["id"])
                    print(f"[WARN] skipping match {meta['id']} ({league_name} {season}): {exc}")
                    continue
                observations.extend(
                    _match_player_observations(
                        league_name, int(season), meta, data.get("rosters") or {},
                        team_name_bridge=team_name_bridge,
                    )
                )
                season_matches += 1
                processed += 1
            print(f"[OK] {league_name} {season}: {season_matches} matches")
        if max_matches is not None and processed >= max_matches:
            break

    if failed:
        preview = ", ".join(failed[:10]) + ("..." if len(failed) > 10 else "")
        print(f"[WARN] {len(failed)} matches failed after retries; rerun to fill from cache: {preview}")
    if not observations:
        raise RuntimeError("No Understat player rows were downloaded.")

    df = pd.DataFrame(observations)
    df = df.dropna(subset=["date", "team", "opponent", "player_id"])
    df = df[df["player_id"].astype(str).str.strip() != ""]
    df = df.drop_duplicates(
        subset=["date", "league", "team", "opponent", "player_id"], keep="first"
    )
    return df.reset_index(drop=True)


def player_match_stats_from_observations(observations: pd.DataFrame) -> pd.DataFrame:
    """Select the strict ``player_match_stats.csv`` schema from raw observations."""
    df = observations.copy()
    for col in PLAYER_MATCH_STATS_COLUMNS:
        if col not in df.columns:
            df[col] = np.nan
    df = df[PLAYER_MATCH_STATS_COLUMNS]
    return df.sort_values(["date", "league", "team", "player_id"]).reset_index(drop=True)


def player_registry_from_observations(observations: pd.DataFrame) -> pd.DataFrame:
    """Derive ``player_registry.csv`` (id/name/team/season/position) from observations.

    ``importance_tier`` is deliberately left as ``"unknown"`` rather than inferred:
    it is a manual override used only as a last-resort fallback when no match stats
    exist, so the registry should not fabricate a strength label from participation.
    """
    if observations.empty:
        return pd.DataFrame(columns=PLAYER_REGISTRY_COLUMNS)

    rows = []
    for (player_id, league, season, team), group in observations.groupby(
        ["player_id", "league", "season", "team"]
    ):
        starter_positions = group.loc[group["started"] == 1, "position"]
        starter_positions = starter_positions[starter_positions.astype(str).str.strip() != ""]
        if not starter_positions.empty:
            position = starter_positions.mode().iat[0]
        else:
            any_positions = group["position"][group["position"].astype(str).str.strip() != ""]
            position = any_positions.mode().iat[0] if not any_positions.empty else ""

        names = group["player_name"].dropna()
        names = names[names.astype(str).str.strip() != ""]
        rows.append({
            "player_id": str(player_id),
            "player_name": names.iat[0] if not names.empty else str(player_id),
            "normalized_player_name": "",  # loader derives this from player_name
            "team": team,
            "league": league,
            "season": int(season),
            "position": position,
            "importance_tier": "unknown",
            "date_from": "",
            "date_to": "",
            "source": "understat",
        })

    df = pd.DataFrame(rows, columns=PLAYER_REGISTRY_COLUMNS)
    return df.sort_values(["league", "season", "team", "player_id"]).reset_index(drop=True)


def write_player_match_stats(df: pd.DataFrame, output_file: Path = PLAYER_MATCH_OUTPUT_FILE) -> None:
    output_file.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(output_file, index=False)


def write_player_registry(df: pd.DataFrame, output_file: Path = PLAYER_REGISTRY_OUTPUT_FILE) -> None:
    output_file.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(output_file, index=False)


def sync_player_match_stats(
    *,
    start_season: int = 2014,
    end_season: int | None = None,
    leagues: Iterable[str] | None = None,
    output_file: Path = PLAYER_MATCH_OUTPUT_FILE,
    registry_file: Path = PLAYER_REGISTRY_OUTPUT_FILE,
    cache_dir: Path = MATCH_CACHE_DIR,
    pause_seconds: float = 0.2,
    max_matches: int | None = None,
    use_cache: bool = True,
    build_registry: bool = True,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Download Understat per-player stats and write the player CSV (+ optional registry)."""
    if end_season is None:
        end_season = current_season_start_year()
    if end_season < start_season:
        raise ValueError("end_season must be greater than or equal to start_season")

    league_codes = list(leagues) if leagues else list(UNDERSTAT_LEAGUES.keys())
    seasons = range(start_season, end_season + 1)
    observations = build_player_match_observations(
        league_codes,
        seasons,
        cache_dir=cache_dir,
        pause_seconds=pause_seconds,
        max_matches=max_matches,
        use_cache=use_cache,
    )

    stats = player_match_stats_from_observations(observations)
    write_player_match_stats(stats, output_file)

    registry = pd.DataFrame(columns=PLAYER_REGISTRY_COLUMNS)
    if build_registry:
        registry = player_registry_from_observations(observations)
        write_player_registry(registry, registry_file)
    return stats, registry


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Download Understat team-match xG and player data.")
    parser.add_argument("--start-season", type=int, default=2014)
    parser.add_argument("--end-season", type=int, default=current_season_start_year())
    parser.add_argument("--output", type=Path, default=OUTPUT_FILE)
    parser.add_argument("--pause-seconds", type=float, default=0.2)
    # Per-player per-match stats (optional, heavier: one request per match).
    parser.add_argument("--players", action="store_true",
                        help="Also download per-player per-match stats -> player_match_stats.csv (+ registry).")
    parser.add_argument("--players-only", action="store_true",
                        help="Skip the team-level xG sync and only build the player datasets.")
    parser.add_argument("--player-output", type=Path, default=PLAYER_MATCH_OUTPUT_FILE)
    parser.add_argument("--registry-output", type=Path, default=PLAYER_REGISTRY_OUTPUT_FILE)
    parser.add_argument("--match-cache-dir", type=Path, default=MATCH_CACHE_DIR)
    parser.add_argument("--max-matches", type=int, default=None,
                        help="Cap the number of matches fetched (useful for a quick test run).")
    parser.add_argument("--no-cache", action="store_true",
                        help="Ignore the on-disk match cache and refetch every match.")
    parser.add_argument("--no-registry", action="store_true",
                        help="Do not build player_registry.csv when fetching player stats.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    if not args.players_only:
        df = sync_understat_matches(
            start_season=args.start_season,
            end_season=args.end_season,
            output_file=args.output,
            pause_seconds=args.pause_seconds,
        )
        print(f"\nSaved {len(df)} team-match rows -> {args.output}")
        print(f"Date range: {df['date'].min()} to {df['date'].max()}")

    if args.players or args.players_only:
        stats, registry = sync_player_match_stats(
            start_season=args.start_season,
            end_season=args.end_season,
            output_file=args.player_output,
            registry_file=args.registry_output,
            cache_dir=args.match_cache_dir,
            pause_seconds=args.pause_seconds,
            max_matches=args.max_matches,
            use_cache=not args.no_cache,
            build_registry=not args.no_registry,
        )
        print(f"\nSaved {len(stats)} player-match rows -> {args.player_output}")
        if not args.no_registry:
            print(f"Saved {len(registry)} registry rows -> {args.registry_output}")


if __name__ == "__main__":
    main()
