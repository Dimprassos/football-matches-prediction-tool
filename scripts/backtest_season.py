"""Entry point: leakage-safe single-season betting backtest CLI.

Thin wrapper around :func:`src.cli.backtest_season_cli.main`. Run from the
project root with ``python scripts/backtest_season.py --season 2024``.
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))  # project root on path

from src.cli.backtest_season_cli import main


if __name__ == "__main__":
    main()
