"""Refresh the local Understat expected-goals (xG) export.

Scrapes/loads Understat per-match xG/npxG/xpts for the configured leagues and writes
the consolidated ``data/external/understat_matches.csv`` that
:func:`src.understat_data.add_understat_xg` later merges in. Optional/offline-friendly;
run with ``python src/update_understat.py``.
"""
from __future__ import annotations

import argparse
import time
from datetime import datetime
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd
import requests


PROJECT_ROOT = Path(__file__).resolve().parent.parent
OUTPUT_FILE = PROJECT_ROOT / "data" / "external" / "understat_matches.csv"
UNDERSTAT_BASE_URL = "https://understat.com"

UNDERSTAT_LEAGUES = {
    "EPL": "EPL",
    "La_liga": "La liga",
    "Bundesliga": "Bundesliga",
    "Serie_A": "Serie A",
    "Ligue_1": "Ligue 1",
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


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Download Understat team-match xG data.")
    parser.add_argument("--start-season", type=int, default=2014)
    parser.add_argument("--end-season", type=int, default=current_season_start_year())
    parser.add_argument("--output", type=Path, default=OUTPUT_FILE)
    parser.add_argument("--pause-seconds", type=float, default=0.2)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    df = sync_understat_matches(
        start_season=args.start_season,
        end_season=args.end_season,
        output_file=args.output,
        pause_seconds=args.pause_seconds,
    )
    print(f"\nSaved {len(df)} rows -> {args.output}")
    print(f"Date range: {df['date'].min()} to {df['date'].max()}")


if __name__ == "__main__":
    main()
