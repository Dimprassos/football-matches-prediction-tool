"""Entry point: run the full training/evaluation pipeline for the canonical experiment.

``FINAL_CONFIG`` is the opening-odds, pre-match configuration whose metrics the
thesis reports. Run from the project root with ``python scripts/main.py``.
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))  # project root on path

from src.config import FINAL_CONFIG
from src.trainer import run_training_pipeline


if __name__ == "__main__":
    run_training_pipeline(FINAL_CONFIG)
