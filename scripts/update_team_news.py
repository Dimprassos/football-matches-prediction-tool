"""Optional API-Football team-news updater (injuries/suspensions context).

Run from the project root with ``python scripts/update_team_news.py``. Requires
an API key; safe to skip when offline (the rest of the pipeline runs without it).
"""
from __future__ import annotations

import argparse
import os
import sys
from datetime import date, timedelta
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))  # project root on path

from src.update_api_football_context import MATCH_CONTEXT_FILE, collect_local_matches, update_api_football_context


def _parse_date(value: str) -> date:
    return date.fromisoformat(value)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Simple updater for API-Football injuries/suspensions and optional lineups."
    )
    parser.add_argument("--from-date", type=_parse_date, default=None, help="Default: today.")
    parser.add_argument("--to-date", type=_parse_date, default=None)
    parser.add_argument("--days", type=int, default=7, help="Default: next 7 days.")
    parser.add_argument("--max", dest="max_fixtures", type=int, default=40, help="Default: 40 fixtures.")
    parser.add_argument("--lineups", action="store_true", help="Also try confirmed lineups near kickoff.")
    parser.add_argument("--played", action="store_true", help="Also include already played matches in the date window.")
    parser.add_argument(
        "--free-training",
        action="store_true",
        help="Backfill API-Football free seasons 2022-2024 for training. Re-run daily to continue.",
    )
    parser.add_argument(
        "--backtest-season",
        type=int,
        default=None,
        help="Target the validation+test window for a season backtest. Example: 2023 means 2022-07-01 to 2024-06-30.",
    )
    parser.add_argument(
        "--period",
        choices=("test", "validation", "both"),
        default="test",
        help="For --backtest-season only. Default: test.",
    )
    parser.add_argument("--cache-only", action="store_true", help="Use cached API responses only.")
    parser.add_argument("--refresh-cache", action="store_true", help="Fetch again even when a cached response exists.")
    parser.add_argument("--pause", type=float, default=6.5, help="Seconds between API requests. Default is safe for free plan.")
    parser.add_argument("--dry-run", action="store_true", help="Print rows without writing match_context.csv.")
    return parser.parse_args()


def _backtest_window(season_start: int, period: str) -> tuple[date, date]:
    if period == "validation":
        return date(season_start - 1, 7, 1), date(season_start, 6, 30)
    if period == "both":
        return date(season_start - 1, 7, 1), date(season_start + 1, 6, 30)
    return date(season_start, 7, 1), date(season_start + 1, 6, 30)


def _print_backtest_coverage(df: pd.DataFrame, season_start: int) -> None:
    if df.empty:
        return
    work = df.copy()
    work["date"] = pd.to_datetime(work["date"], errors="coerce")
    val_start = pd.Timestamp(f"{season_start - 1}-07-01")
    test_start = pd.Timestamp(f"{season_start}-07-01")
    test_end = pd.Timestamp(f"{season_start + 1}-07-01")

    local = collect_local_matches(
        from_date=val_start.date(),
        to_date=(test_end - pd.Timedelta(days=1)).date(),
        include_played=True,
    )
    local["date"] = pd.to_datetime(local["date"], errors="coerce")

    for label, start, end in (
        ("validation", val_start, test_start),
        ("test", test_start, test_end),
    ):
        total = int(((local["date"] >= start) & (local["date"] < end)).sum())
        mask = (work["date"] >= start) & (work["date"] < end)
        team_news = int(pd.to_numeric(work.loc[mask, "team_news_available"], errors="coerce").fillna(0).sum())
        print(f"{label} coverage for backtest {season_start}: {team_news}/{total} team-news rows")


def main() -> None:
    args = parse_args()
    api_key = os.getenv("API_FOOTBALL_KEY") or os.getenv("API_SPORTS_KEY")
    if not api_key and not args.cache_only:
        print("Missing API key.")
        print('Run first: $env:API_FOOTBALL_KEY="your_key_here"')
        raise SystemExit(2)

    if args.backtest_season is not None:
        default_from, default_to = _backtest_window(args.backtest_season, args.period)
        from_date = args.from_date or default_from
        to_date = args.to_date or default_to
        include_played = True
    elif args.free_training:
        from_date = args.from_date or date(2022, 7, 1)
        to_date = args.to_date or date(2025, 6, 30)
        include_played = True
    else:
        from_date = args.from_date or date.today()
        to_date = args.to_date or (from_date + timedelta(days=max(0, args.days)))
        include_played = args.played

    fetch_lineups = bool(args.lineups)

    print("Updating team news context")
    if args.backtest_season is not None:
        print(f"Mode: backtest-season {args.backtest_season} context backfill")
        print(f"Period: {args.period}")
    elif args.free_training:
        print("Mode: free historical training backfill")
    print(f"Window: {from_date} to {to_date}")
    print(f"Max fixtures: {args.max_fixtures}")
    print(f"Lineups: {'on' if fetch_lineups else 'off'}")
    print(f"API pause: {args.pause}s")

    df = update_api_football_context(
        from_date=from_date,
        to_date=to_date,
        api_key=api_key,
        include_played=include_played,
        fetch_lineups=fetch_lineups,
        fetch_injuries=True,
        lineup_window_days=1 if fetch_lineups else 0,
        max_fixtures=args.max_fixtures,
        pause_seconds=args.pause,
        cache_only=args.cache_only,
        refresh_cache=args.refresh_cache,
        dry_run=args.dry_run,
    )

    if args.dry_run:
        return
    if df.empty:
        print("No rows were written.")
        return
    print(f"Done. Context file: {MATCH_CONTEXT_FILE}")
    if args.backtest_season is not None:
        _print_backtest_coverage(df, args.backtest_season)


if __name__ == "__main__":
    main()
