"""Entry point: interactive custom-match predictor CLI.

Thin wrapper around :func:`src.cli.predict_match_cli.main`. Run from the project
root with ``python scripts/predict_match.py``.
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))  # project root on path

from src.cli.predict_match_cli import main


if __name__ == "__main__":
    main()
