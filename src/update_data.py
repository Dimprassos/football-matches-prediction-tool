"""Download/refresh the raw football-data.co.uk league CSVs (results + fixtures).

Fetches the current-season result and fixture files for each configured league and
writes them under ``data/raw/<league>``. Optional utility — the repo already ships
with data, so this is only needed to pull newer matches. Run with
``python src/update_data.py``.
"""
import os
import re
from io import BytesIO
from pathlib import Path
from datetime import datetime
from typing import Optional

import numpy as np
import pandas as pd
import requests
from tqdm import tqdm

from src.team_names import normalize_team_name

# ============================================================
# Historical data source (football-data.co.uk)
# ============================================================
LEAGUES = {
    "E0": "england",
    "SP1": "spain",
    "I1": "italy",
    "D1": "germany",
    "F1": "france",
}

START_YEAR = 12
END_YEAR = 25  # 2025/26
BASE_URL = "https://www.football-data.co.uk/mmz4281/{season}/{league}.csv"

# ============================================================
# Fixture source (FixtureDownload)
# ============================================================
FIXTURE_SLUGS = {
    "E0": "epl",
    "SP1": "la-liga",
    "I1": "serie-a",
    "D1": "bundesliga",
    "F1": "ligue-1",
}

PROJECT_ROOT = Path(__file__).resolve().parent.parent
BASE_DIR = PROJECT_ROOT / "data" / "raw"


def current_season_start_year(today: Optional[datetime] = None) -> int:
    """
    Example:
      Mar 2026 -> 2025
      Aug 2026 -> 2026
    """
    if today is None:
        today = datetime.now()
    return today.year if today.month >= 7 else today.year - 1


def parse_result_to_goals(result_value):
    """
    Parses strings like '2 - 1' or '2-1'.
    Returns (None, None) if result is blank / missing.
    """
    if pd.isna(result_value):
        return None, None

    s = str(result_value).strip()
    if not s:
        return None, None

    m = re.match(r"^\s*(\d+)\s*-\s*(\d+)\s*$", s)
    if m:
        return int(m.group(1)), int(m.group(2))

    return None, None


def standardize_fixturedownload_csv(raw_bytes, league_code, league_folder):
    """
    Converts FixtureDownload CSV into a football-data-like schema:
      Date, HomeTeam, AwayTeam, FTHG, FTAG
    Keeps ONLY future / unplayed fixtures.
    """
    df = pd.read_csv(BytesIO(raw_bytes))

    required_cols = {"Date", "Home Team", "Away Team"}
    if not required_cols.issubset(set(df.columns)):
        raise ValueError(
            "Unexpected FixtureDownload format for {}. Columns found: {}".format(
                league_code, list(df.columns)
            )
        )

    result_col = "Result" if "Result" in df.columns else None

    home_goals = []
    away_goals = []

    for _, row in df.iterrows():
        if result_col is None:
            hg, ag = None, None
        else:
            hg, ag = parse_result_to_goals(row[result_col])

        home_goals.append(hg)
        away_goals.append(ag)

    out = pd.DataFrame({
        "Date": df["Date"],
        "HomeTeam": df["Home Team"].map(lambda x: normalize_team_name(x, league_folder)),
        "AwayTeam": df["Away Team"].map(lambda x: normalize_team_name(x, league_folder)),
        "FTHG": home_goals,
        "FTAG": away_goals,
    })

    parsed_dt = pd.to_datetime(out["Date"], dayfirst=True, errors="coerce")
    out = out[parsed_dt.notna()].copy()
    parsed_dt = parsed_dt[parsed_dt.notna()]

    # Keep ONLY future/unplayed fixtures
    future_mask = out["FTHG"].isna() & out["FTAG"].isna()
    out = out[future_mask].copy()
    parsed_dt = parsed_dt[future_mask]

    out["Date"] = parsed_dt.dt.strftime("%d/%m/%Y")

    # empty odds placeholders
    out["PSCH"] = np.nan
    out["PSCD"] = np.nan
    out["PSCA"] = np.nan

    return out.reset_index(drop=True)


def download_historical_data():
    tasks = []

    for folder_name in LEAGUES.values():
        os.makedirs(BASE_DIR / folder_name, exist_ok=True)

    for start_yr in range(START_YEAR, END_YEAR + 1):
        end_yr = start_yr + 1
        season_str = "{:02d}{:02d}".format(start_yr, end_yr)

        for league_code, folder_name in LEAGUES.items():
            url = BASE_URL.format(season=season_str, league=league_code)
            local_filename = "{}_20{:02d}.csv".format(league_code, start_yr)
            local_filepath = BASE_DIR / folder_name / local_filename
            tasks.append((url, local_filepath))

    updated_count = 0
    skipped_count = 0
    new_count = 0
    failed_count = 0

    print("Checking historical files ({})...".format(len(tasks)))

    for url, local_filepath in tqdm(tasks, desc="Historical sync", unit="files"):
        try:
            response = requests.get(url, timeout=15)
            if response.status_code != 200:
                failed_count += 1
                continue

            remote_content = response.content

            if local_filepath.exists():
                if local_filepath.read_bytes() == remote_content:
                    skipped_count += 1
                    continue
                updated_count += 1
            else:
                new_count += 1

            with open(local_filepath, "wb") as f:
                f.write(remote_content)

        except Exception:
            failed_count += 1

    print(
        "Historical sync done. New: {}, Updated: {}, Unchanged: {}, Failed: {}".format(
            new_count, updated_count, skipped_count, failed_count
        )
    )


def download_current_future_fixtures():
    """
    Downloads current season fixture CSVs from FixtureDownload
    and saves only future/unplayed fixtures per league.
    """
    season_start = current_season_start_year()
    saved = 0
    failed = 0

    print("Downloading future fixtures for current season {}/{}...".format(
        season_start, season_start + 1
    ))

    for league_code, league_folder in LEAGUES.items():
        slug = FIXTURE_SLUGS[league_code]
        url = "https://fixturedownload.com/download/{}-{}-GMTStandardTime.csv".format(
            slug, season_start
        )

        try:
            response = requests.get(url, timeout=20)
            if response.status_code != 200:
                failed += 1
                print("[WARN] Could not fetch fixtures for {} ({})".format(
                    league_folder, response.status_code
                ))
                continue

            fixtures_df = standardize_fixturedownload_csv(
                response.content, league_code, league_folder
            )

            out_dir = BASE_DIR / league_folder
            out_dir.mkdir(parents=True, exist_ok=True)

            out_path = out_dir / "{}_{}_fixtures.csv".format(league_code, season_start)
            fixtures_df.to_csv(out_path, index=False)

            print("[OK] {}: saved {} future fixtures -> {}".format(
                league_folder, len(fixtures_df), out_path.name
            ))
            saved += 1

        except Exception as e:
            failed += 1
            print("[WARN] Failed fixtures for {}: {}".format(league_folder, e))

    print("Fixture sync done. Saved: {}, Failed: {}".format(saved, failed))


def fetch_all_data():
    download_historical_data()
    download_current_future_fixtures()
    print("\nProcess completed.")


if __name__ == "__main__":
    fetch_all_data()
