"""Retrain the models into a *separate* experiment.

The thesis cites the metrics of the canonical experiment
(`final_opening_market_pre_match`). To let the user retrain from the tool
without ever overwriting those numbers, this runner:

  1. seeds the target experiment's tuned-hyperparameter caches from the
     canonical experiment (so retraining reuses the expensive Optuna tuning and
     only *refits* the models on the current — possibly user-extended — data);
  2. runs the standard training pipeline under the target experiment name.

The canonical artifacts are left untouched.

Usage:
    python scripts/retrain_runner.py --experiment user_retrain [--reset]
"""
from __future__ import annotations

import argparse
import json
import shutil
import sys
from dataclasses import replace
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))  # project root on path

from src.config import FINAL_CONFIG
from src.trainer import run_training_pipeline


def seed_caches(source, target, reset: bool = False) -> None:
    """Copy tuned-hyperparameter artifacts source -> target (skip if present)."""
    pairs = [
        (source.params_file, target.params_file),
        (source.meta_file, target.meta_file),
        (source.model_file, target.model_file),
        (source.mlp_meta_file, target.mlp_meta_file),
        (source.mlp_model_file, target.mlp_model_file),
        (source.logreg_meta_file, target.logreg_meta_file),
        (source.logreg_model_file, target.logreg_model_file),
        (source.blend_file, target.blend_file),
    ]
    for src_path, dst_path in pairs:
        if src_path.exists() and (reset or not dst_path.exists()):
            shutil.copyfile(src_path, dst_path)

    # The compatibility check compares the manifest's experiment_name against the
    # target config, so copy the manifest but rewrite that field — otherwise the
    # pipeline would discard the seeded caches and re-tune from scratch.
    if source.manifest_file.exists() and (reset or not target.manifest_file.exists()):
        manifest = json.loads(source.manifest_file.read_text(encoding="utf-8"))
        if isinstance(manifest.get("config"), dict):
            manifest["config"]["experiment_name"] = target.experiment_name
        target.manifest_file.write_text(json.dumps(manifest, indent=2), encoding="utf-8")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--experiment", default="user_retrain")
    ap.add_argument("--reset", action="store_true",
                    help="re-seed caches from the canonical experiment before retraining")
    args = ap.parse_args()

    target = replace(
        FINAL_CONFIG,
        experiment_name=args.experiment,
        use_cached_artifacts=True,
        allow_partial_param_cache=True,
        generate_upcoming_picks=False,  # no network calls during a tool retrain
    )
    seed_caches(FINAL_CONFIG, target, reset=args.reset)
    print(
        f"Retraining into experiment '{args.experiment}' "
        f"(canonical '{FINAL_CONFIG.experiment_name}' left untouched)...",
        flush=True,
    )
    run_training_pipeline(target)
    print("RETRAIN_DONE", flush=True)


if __name__ == "__main__":
    main()
