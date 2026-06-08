"""Train the feature-rich *context-aware* variant experiment.

The canonical experiment (`final_opening_market_pre_match`) auto-selects a
market-only feature set because it scores best, so understat/form features never
influence its predictions. This runner trains a separate experiment that is
*forced* to use understat-xG-aware feature sets (see CONTEXT_AWARE_CONFIG), so the
pre-match features reconstructed at serve time (predictor.build_runtime_extra_features)
actually change the prediction and the user can compare it against the market-only model.

It seeds the slow per-league Elo/Dixon-Coles parameters from the canonical
experiment (reused, not re-tuned) and only re-tunes/refits the learned models
(XGBoost, MLP, blend) on the forced feature sets. The canonical artifacts are left
untouched.

Usage:
    python scripts/train_context_variant.py [--reset]
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))  # project root on path

from src.config import CONTEXT_AWARE_CONFIG, FINAL_CONFIG
from src.trainer import run_training_pipeline
from scripts.retrain_runner import seed_caches


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--reset", action="store_true",
                    help="re-seed caches from the canonical experiment before training")
    args = ap.parse_args()

    seed_caches(FINAL_CONFIG, CONTEXT_AWARE_CONFIG, reset=args.reset)
    print(
        f"Training context-aware variant '{CONTEXT_AWARE_CONFIG.experiment_name}' "
        f"(canonical '{FINAL_CONFIG.experiment_name}' left untouched)...",
        flush=True,
    )
    run_training_pipeline(CONTEXT_AWARE_CONFIG)
    print("CONTEXT_VARIANT_DONE", flush=True)


if __name__ == "__main__":
    main()
