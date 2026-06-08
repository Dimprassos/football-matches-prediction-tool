"""Fetch Open-Meteo weather context for fixtures and write it to match_context.csv.

Looks up temperature/wind/precipitation at each venue around kickoff and merges the
weather columns into the external match-context file consumed by
:mod:`src.external_context`. Optional (needs network); the pipeline runs without it.
Run with ``python src/update_weather_context.py``.
"""
from __future__ import annotations

import argparse
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

from src.external_context import MATCH_CONTEXT_FILE
from src.team_names import normalize_team_name


RAW_DIR = PROJECT_ROOT / "data" / "raw"
TEAM_LOCATIONS_FILE = PROJECT_ROOT / "data" / "external" / "team_locations.csv"
OPEN_METEO_ARCHIVE_URL = "https://archive-api.open-meteo.com/v1/archive"

LEAGUE_CODES = {
    "england": "E0",
    "spain": "SP1",
    "italy": "I1",
    "germany": "D1",
    "france": "F1",
}

WEATHER_COLUMNS = ["weather_available", "temperature_c", "wind_kph", "precipitation_mm"]


def _parse_dates(series: pd.Series) -> pd.Series:
    parsed_short = pd.to_datetime(series, format="%d/%m/%y", errors="coerce")
    parsed_long = pd.to_datetime(series, format="%d/%m/%Y", errors="coerce")
    parsed = parsed_short.fillna(parsed_long)
    missing = parsed.isna()
    if missing.any():
        parsed.loc[missing] = pd.to_datetime(series.loc[missing], dayfirst=True, errors="coerce")
    return parsed.dt.normalize()


def _parse_kickoff_hour(value) -> int:
    if pd.isna(value):
        return 15
    parts = str(value).strip().split(":")
    if not parts:
        return 15
    try:
        hour = int(parts[0])
    except ValueError:
        return 15
    return int(np.clip(hour, 0, 23))


def _season_start_from_filename(path: Path) -> int | None:
    stem = path.stem
    if "_fixtures" in stem:
        return None
    try:
        return int(stem.rsplit("_", 1)[-1])
    except ValueError:
        return None


def _iter_raw_match_files(leagues: Iterable[str]) -> Iterable[tuple[str, Path]]:
    for league in leagues:
        league_dir = RAW_DIR / league
        if not league_dir.exists():
            continue
        for path in sorted(league_dir.glob("*.csv")):
            yield league, path


def collect_matches(
    *,
    leagues: Iterable[str] = LEAGUE_CODES.keys(),
    start_season: int | None = None,
    end_season: int | None = None,
) -> pd.DataFrame:
    rows = []
    for league, path in _iter_raw_match_files(leagues):
        season_start = _season_start_from_filename(path)
        if start_season is not None and season_start is not None and season_start < start_season:
            continue
        if end_season is not None and season_start is not None and season_start > end_season:
            continue

        raw = pd.read_csv(path)
        required = {"Date", "HomeTeam", "AwayTeam"}
        if not required.issubset(raw.columns):
            continue

        dates = _parse_dates(raw["Date"])
        times = raw["Time"] if "Time" in raw.columns else pd.Series(np.nan, index=raw.index)
        for idx, row in raw.iterrows():
            match_date = dates.loc[idx]
            if pd.isna(match_date):
                continue
            rows.append({
                "date": pd.Timestamp(match_date).strftime("%Y-%m-%d"),
                "kickoff_hour": _parse_kickoff_hour(times.loc[idx]),
                "league": league,
                "home_team": normalize_team_name(row["HomeTeam"], league),
                "away_team": normalize_team_name(row["AwayTeam"], league),
            })

    out = pd.DataFrame(rows)
    if out.empty:
        return pd.DataFrame(columns=["date", "kickoff_hour", "league", "home_team", "away_team"])
    return out.drop_duplicates(["date", "league", "home_team", "away_team"], keep="last").reset_index(drop=True)


def load_team_locations(path: Path = TEAM_LOCATIONS_FILE) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(
            f"Missing team location file: {path}\n"
            "Create data/external/team_locations.csv with columns: league,team,latitude,longitude"
        )

    locations = pd.read_csv(path)
    required = {"league", "team", "latitude", "longitude"}
    missing = required - set(locations.columns)
    if missing:
        raise ValueError(f"{path} is missing required columns: {sorted(missing)}")

    locations = locations.copy()
    locations["league"] = locations["league"].astype(str).str.strip().str.lower()
    locations["team"] = [
        normalize_team_name(team, league)
        for team, league in zip(locations["team"], locations["league"])
    ]
    locations["latitude"] = pd.to_numeric(locations["latitude"], errors="coerce")
    locations["longitude"] = pd.to_numeric(locations["longitude"], errors="coerce")
    locations = locations.dropna(subset=["league", "team", "latitude", "longitude"])
    return locations.drop_duplicates(["league", "team"], keep="last")


def write_team_location_template(
    *,
    output_file: Path = TEAM_LOCATIONS_FILE,
    start_season: int | None = None,
    end_season: int | None = None,
) -> pd.DataFrame:
    matches = collect_matches(start_season=start_season, end_season=end_season)
    teams = matches[["league", "home_team"]].drop_duplicates().sort_values(["league", "home_team"])
    template = teams.rename(columns={"home_team": "team"}).reset_index(drop=True)
    template["latitude"] = ""
    template["longitude"] = ""
    output_file.parent.mkdir(parents=True, exist_ok=True)
    template.to_csv(output_file, index=False)
    return template


def matches_with_locations(matches: pd.DataFrame, locations: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    merged = matches.merge(
        locations.rename(columns={"team": "home_team"}),
        on=["league", "home_team"],
        how="left",
    )
    missing = merged[merged["latitude"].isna() | merged["longitude"].isna()][["league", "home_team"]].drop_duplicates()
    matched = merged.dropna(subset=["latitude", "longitude"]).copy()
    return matched, missing.sort_values(["league", "home_team"]).reset_index(drop=True)


def fetch_open_meteo_hourly(
    *,
    latitude: float,
    longitude: float,
    start_date: str,
    end_date: str,
    timeout: int = 30,
) -> pd.DataFrame:
    params = {
        "latitude": latitude,
        "longitude": longitude,
        "start_date": start_date,
        "end_date": end_date,
        "hourly": "temperature_2m,precipitation,wind_speed_10m",
        "timezone": "auto",
        "wind_speed_unit": "kmh",
    }
    response = requests.get(OPEN_METEO_ARCHIVE_URL, params=params, timeout=timeout)
    response.raise_for_status()
    payload = response.json()
    hourly = payload.get("hourly", {})
    if not hourly or "time" not in hourly:
        return pd.DataFrame(columns=["date", "kickoff_hour", *WEATHER_COLUMNS])

    times = hourly["time"]
    weather = pd.DataFrame({
        "time": pd.to_datetime(times, errors="coerce"),
        "temperature_c": pd.to_numeric(hourly.get("temperature_2m", [np.nan] * len(times)), errors="coerce"),
        "wind_kph": pd.to_numeric(hourly.get("wind_speed_10m", [np.nan] * len(times)), errors="coerce"),
        "precipitation_mm": pd.to_numeric(hourly.get("precipitation", [np.nan] * len(times)), errors="coerce"),
    }).dropna(subset=["time"])
    weather["date"] = weather["time"].dt.strftime("%Y-%m-%d")
    weather["kickoff_hour"] = weather["time"].dt.hour.astype(int)
    weather["weather_available"] = (
        weather[["temperature_c", "wind_kph", "precipitation_mm"]].notna().any(axis=1).astype(int)
    )
    return weather[["date", "kickoff_hour", *WEATHER_COLUMNS]]


def build_weather_context(
    matches: pd.DataFrame,
    *,
    pause_seconds: float = 0.1,
) -> pd.DataFrame:
    rows = []
    grouped = matches.groupby(["league", "home_team", "latitude", "longitude"], dropna=False)
    for (league, home_team, latitude, longitude), group in grouped:
        start_date = str(group["date"].min())
        end_date = str(group["date"].max())
        weather = fetch_open_meteo_hourly(
            latitude=float(latitude),
            longitude=float(longitude),
            start_date=start_date,
            end_date=end_date,
        )
        enriched = group.merge(weather, on=["date", "kickoff_hour"], how="left")
        for col in WEATHER_COLUMNS:
            if col not in enriched.columns:
                enriched[col] = np.nan
        rows.append(enriched[["date", "league", "home_team", "away_team", *WEATHER_COLUMNS]])
        print(f"[OK] {league} / {home_team}: {len(group)} matches")
        if pause_seconds > 0:
            time.sleep(pause_seconds)

    if not rows:
        return pd.DataFrame(columns=["date", "league", "home_team", "away_team", *WEATHER_COLUMNS])

    out = pd.concat(rows, ignore_index=True)
    out["weather_available"] = pd.to_numeric(out["weather_available"], errors="coerce").fillna(0).astype(int)
    return out


def merge_match_context(existing: pd.DataFrame, weather: pd.DataFrame) -> pd.DataFrame:
    keys = ["date", "league", "home_team", "away_team"]
    if existing.empty:
        return weather.copy()

    merged = existing.merge(weather, on=keys, how="outer", suffixes=("", "_weather"))
    for col in WEATHER_COLUMNS:
        weather_col = f"{col}_weather"
        if weather_col in merged.columns:
            merged[col] = merged[weather_col].combine_first(merged.get(col))
            merged = merged.drop(columns=[weather_col])
    return merged.drop_duplicates(keys, keep="last").sort_values(keys).reset_index(drop=True)


def update_weather_context(
    *,
    start_season: int | None = None,
    end_season: int | None = None,
    locations_file: Path = TEAM_LOCATIONS_FILE,
    output_file: Path = MATCH_CONTEXT_FILE,
    pause_seconds: float = 0.1,
) -> pd.DataFrame:
    matches = collect_matches(start_season=start_season, end_season=end_season)
    locations = load_team_locations(locations_file)
    located_matches, missing_locations = matches_with_locations(matches, locations)
    if not missing_locations.empty:
        missing_path = output_file.parent / "missing_team_locations.csv"
        missing_locations.to_csv(missing_path, index=False)
        print(f"[WARN] Missing {len(missing_locations)} team locations -> {missing_path}")

    weather = build_weather_context(located_matches, pause_seconds=pause_seconds)
    existing = pd.read_csv(output_file) if output_file.exists() else pd.DataFrame()
    merged = merge_match_context(existing, weather)
    output_file.parent.mkdir(parents=True, exist_ok=True)
    merged.to_csv(output_file, index=False)
    return merged


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Fetch historical kickoff weather into match_context.csv.")
    parser.add_argument("--start-season", type=int, default=None, help="First season start year, e.g. 2023.")
    parser.add_argument("--end-season", type=int, default=None, help="Last season start year, e.g. 2024.")
    parser.add_argument("--locations", type=Path, default=TEAM_LOCATIONS_FILE)
    parser.add_argument("--output", type=Path, default=MATCH_CONTEXT_FILE)
    parser.add_argument("--pause-seconds", type=float, default=0.1)
    parser.add_argument(
        "--write-location-template",
        action="store_true",
        help="Write a team_locations.csv skeleton from the raw match files and exit.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.write_location_template:
        template = write_team_location_template(
            output_file=args.locations,
            start_season=args.start_season,
            end_season=args.end_season,
        )
        print(f"Saved {len(template)} team rows -> {args.locations}")
        print("Fill latitude/longitude, then rerun without --write-location-template.")
        return

    started = datetime.now()
    df = update_weather_context(
        start_season=args.start_season,
        end_season=args.end_season,
        locations_file=args.locations,
        output_file=args.output,
        pause_seconds=args.pause_seconds,
    )
    weather_available = int(pd.to_numeric(df.get("weather_available", 0), errors="coerce").fillna(0).sum())
    print(f"\nSaved {len(df)} context rows -> {args.output}")
    print(f"Rows with weather_available=1: {weather_available}")
    print(f"Elapsed: {datetime.now() - started}")


if __name__ == "__main__":
    main()
