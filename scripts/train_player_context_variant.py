"""Train the *player-context-aware* variant experiment.

Like the context-aware variant, the canonical experiment ignores player context
because a market-only feature set scores best. This runner trains a separate
experiment forced to use the lineup-strength feature set (see PLAYER_CONTEXT_CONFIG),
so the chosen starting XI actually moves the prediction and can be compared, via
ablation, against the market-only model.

Prerequisite: ``data/external/match_context.csv`` must contain historical
``home_lineup_strength``/``away_lineup_strength`` generated from the Understat
starters. Build it first with::

    python scripts/build_player_lineup_context.py

It seeds the slow per-league Elo/Dixon-Coles parameters from the canonical
experiment (reused, not re-tuned) and only re-tunes/refits the learned models on the
forced feature set. The canonical artifacts are left untouched.

Usage:
    python scripts/train_player_context_variant.py [--reset]
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))  # project root on path

from src.config import FINAL_CONFIG, PLAYER_CONTEXT_CONFIG
from src.trainer import run_training_pipeline
from scripts.retrain_runner import seed_caches


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--reset", action="store_true",
                    help="re-seed caches from the canonical experiment before training")
    args = ap.parse_args()

    seed_caches(FINAL_CONFIG, PLAYER_CONTEXT_CONFIG, reset=args.reset)
    print(
        f"Training player-context variant '{PLAYER_CONTEXT_CONFIG.experiment_name}' "
        f"(canonical '{FINAL_CONFIG.experiment_name}' left untouched)...",
        flush=True,
    )
    run_training_pipeline(PLAYER_CONTEXT_CONFIG)
    print("PLAYER_CONTEXT_VARIANT_DONE", flush=True)


if __name__ == "__main__":
    main()
