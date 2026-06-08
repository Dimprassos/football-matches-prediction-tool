"""Diagnostic: print Understat xG coverage per league (matched %, unmatched teams).

Run with ``python -m src.diagnose_understat`` to check how well the local Understat
export joins onto each league's played fixtures.
"""
from __future__ import annotations

from src.config import DEFAULT_CONFIG
from src.data_processing import load_league_data
from src.understat_data import understat_coverage_report


def main():
    """Print the Understat match-coverage summary for every configured league."""
    print("=== UNDERSTAT COVERAGE REPORT ===")
    for league in DEFAULT_CONFIG.leagues:
        df = load_league_data(league)
        report = understat_coverage_report(df, league)
        print(
            f"{league.upper():<8} matched={report['matched']:>4}/{report['played']:<4} "
            f"coverage={report['coverage'] * 100:>5.1f}%"
        )
        teams = report["unmatched_teams"][:30]
        if teams:
            print("  unmatched teams:", ", ".join(teams))


if __name__ == "__main__":
    main()
