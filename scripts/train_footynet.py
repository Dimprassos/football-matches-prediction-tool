"""Train + evaluate the recurrent FootyNet deep-learning model (chunk DL-3/DL-4).

Pools the leakage-safe (static + sequence) datasets across leagues, standardizes the
inputs, trains :class:`src.models.footynet.FootyNet` with early stopping + temperature
scaling, and evaluates on the held-out test season against the market benchmark — in
the *same* metric format (``final_probability_quality_footynet_deep.csv``) as every
other experiment, so it shows up in the app's Evaluation page.

Examples:
    python scripts/train_footynet.py --leagues england --quick      # fast smoke
    python scripts/train_footynet.py                                 # full run, all leagues
    python scripts/train_footynet.py --optimizer sgd --k 8 --tag footynet_gru
"""
from __future__ import annotations

import argparse
import pathlib
import sys
import time
from datetime import datetime, timezone

import numpy as np

ROOT = pathlib.Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.config import FINAL_CONFIG, FOOTYNET_CONFIG
from src.footynet_data import POOL_KEYS, build_league_datasets, pool_split
from src.footynet_stack import apply_blend, learn_blend_weights
from src.bet_selection import write_probability_quality_report
from src.models.footynet import predict_proba, train_footynet
from src.sequence_data import TEAM_MATCH_FEATURES

# Sequence channels to standardize (leave the result one-hot + home flag raw).
_CATEGORICAL = {"result_win", "result_draw", "result_loss", "is_home"}
CONT_IDX = [i for i, n in enumerate(TEAM_MATCH_FEATURES) if n not in _CATEGORICAL]


# --------------------------------------------------------------------------- #
# dataset cache (build once, reuse across ablation runs)                      #
# --------------------------------------------------------------------------- #
def cache_path(k: int, leagues: list[str]) -> pathlib.Path:
    return ROOT / "artifacts" / f"footynet_cache_k{k}_{'-'.join(leagues)}.npz"


def save_cache(path: pathlib.Path, pooled: dict) -> None:
    flat = {}
    for split, d in pooled.items():
        if d is None:
            continue
        for key, arr in d.items():
            flat[f"{split}__{key}"] = arr
    np.savez_compressed(path, **flat)


def load_cache(path: pathlib.Path) -> dict:
    z = np.load(path)
    pooled = {"train": {}, "val": {}, "test": {}}
    for kk in z.files:
        split, key = kk.split("__", 1)
        pooled[split][key] = z[kk]
    return {s: (d if d else None) for s, d in pooled.items()}


def build_pooled(leagues: list[str], k: int) -> dict:
    datasets = []
    for lg in leagues:
        t0 = time.time()
        datasets.append(build_league_datasets(lg, config=FINAL_CONFIG, k=k))
        print(f"  built {lg} in {time.time() - t0:.0f}s")
    return {split: pool_split(datasets, split) for split in ("train", "val", "test")}


# --------------------------------------------------------------------------- #
# standardization (fit on train, applied to every split, padding re-zeroed)   #
# --------------------------------------------------------------------------- #
def fit_scalers(train: dict) -> dict:
    X = train["static"]
    smu, ssd = X.mean(0), X.std(0)
    ssd[ssd < 1e-8] = 1.0

    seqs = np.concatenate([train["seq_home"], train["seq_away"]], axis=0)
    masks = np.concatenate([train["mask_home"], train["mask_away"]], axis=0).reshape(-1).astype(bool)
    flat = seqs.reshape(-1, seqs.shape[-1])[masks]
    qmu = np.zeros(seqs.shape[-1]); qsd = np.ones(seqs.shape[-1])
    qmu[CONT_IDX] = flat[:, CONT_IDX].mean(0)
    sd = flat[:, CONT_IDX].std(0); sd[sd < 1e-8] = 1.0
    qsd[CONT_IDX] = sd
    return {"static_mu": smu, "static_sd": ssd, "seq_mu": qmu, "seq_sd": qsd}


def apply_scalers(split: dict, sc: dict) -> dict:
    out = {"static": (split["static"] - sc["static_mu"]) / sc["static_sd"], "y": split["y"]}
    for side in ("home", "away"):
        seq = (split[f"seq_{side}"] - sc["seq_mu"]) / sc["seq_sd"]
        out[f"seq_{side}"] = seq * split[f"mask_{side}"][..., None]  # keep padding at 0
        out[f"mask_{side}"] = split[f"mask_{side}"]
    return out


def main():
    ap = argparse.ArgumentParser(description="Train + evaluate FootyNet (deep-learning variant).")
    ap.add_argument("--leagues", default="england,spain,italy,germany,france")
    ap.add_argument("--k", type=int, default=5, help="sequence length")
    ap.add_argument("--hidden", type=int, default=32)
    ap.add_argument("--lstm-layers", type=int, default=1)
    ap.add_argument("--cell", choices=["lstm", "gru"], default="lstm")
    ap.add_argument("--dropout", type=float, default=0.3)
    ap.add_argument("--optimizer", choices=["adam", "sgd"], default="adam")
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--batch-size", type=int, default=64)
    ap.add_argument("--epochs", type=int, default=100)
    ap.add_argument("--patience", type=int, default=12)
    ap.add_argument("--label-smoothing", type=float, default=0.0)
    ap.add_argument("--class-weights", action="store_true", help="weight classes by inverse frequency (draw help)")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--tag", default="footynet_deep", help="experiment_name for artifacts/report")
    ap.add_argument("--rebuild", action="store_true", help="ignore the dataset cache")
    args = ap.parse_args()

    leagues = [s.strip() for s in args.leagues.split(",") if s.strip()]
    cpath = cache_path(args.k, leagues)
    if cpath.exists() and not args.rebuild:
        print(f"Loading cached datasets: {cpath.name}")
        pooled = load_cache(cpath)
    else:
        print(f"Building datasets for {leagues} (k={args.k}) ...")
        pooled = build_pooled(leagues, args.k)
        save_cache(cpath, pooled)
        print(f"Cached datasets -> {cpath.name}")

    train, val, test = pooled["train"], pooled["val"], pooled["test"]
    print(f"pooled sizes: train={len(train['y'])} val={len(val['y'])} test={len(test['y'])}")

    sc = fit_scalers(train)
    tr_in, va_in, te_in = apply_scalers(train, sc), apply_scalers(val, sc), apply_scalers(test, sc)

    class_weights = None
    if args.class_weights:
        counts = np.bincount(train["y"], minlength=3).astype(float)
        class_weights = (len(train["y"]) / (3.0 * np.maximum(counts, 1.0))).tolist()
        print(f"class weights (H/D/A): {[round(w, 3) for w in class_weights]}")

    print(f"Training FootyNet ({args.cell}/{args.optimizer}, hidden={args.hidden}, layers={args.lstm_layers}, "
          f"dropout={args.dropout}, k={args.k}) ...")
    res = train_footynet(
        tr_in, va_in,
        hidden=args.hidden, lstm_layers=args.lstm_layers, dropout=args.dropout, cell=args.cell,
        optimizer=args.optimizer, lr=args.lr, batch_size=args.batch_size,
        max_epochs=args.epochs, patience=args.patience,
        class_weights=class_weights, label_smoothing=args.label_smoothing, seed=args.seed,
    )
    print(f"  best val log loss: {res.best_val_logloss:.4f} | temperature: {res.temperature:.3f} "
          f"| epochs: {len(res.history)}")

    # FootyNet probabilities per split, plus a stacking blend with the market.
    foot_va = predict_proba(res.model, va_in, res.temperature)
    foot_te = predict_proba(res.model, te_in, res.temperature)

    # learn convex blend weights on validation ONLY (leakage-safe), apply to both splits
    members_va = {"footynet": foot_va, "market": val["market"]}
    stack_weights = learn_blend_weights(members_va, val["y"])
    stack_va = apply_blend(members_va, stack_weights)
    stack_te = apply_blend({"footynet": foot_te, "market": test["market"]}, stack_weights)
    print(f"stack weights (val-fit): " + ", ".join(f"{k}={v:.2f}" for k, v in stack_weights.items()))

    # evaluate vs the market benchmark, in the canonical report schema
    run_ts = datetime.now(timezone.utc).isoformat()
    split_probs = {
        "validation": {"footynet": foot_va, "market": val["market"], "footynet_stack": stack_va},
        "test": {"footynet": foot_te, "market": test["market"], "footynet_stack": stack_te},
    }
    split_y = {"validation": val["y"], "test": test["y"]}
    from dataclasses import replace
    report_config = replace(FOOTYNET_CONFIG, experiment_name=args.tag)
    rows = write_probability_quality_report(report_config, run_ts, split_probs, split_y)

    print("\n=== TEST: FootyNet vs Market ===")
    print(f"{'model':<10}{'logloss':>9}{'brier':>9}{'acc':>8}{'macro_f1':>10}{'draw_recall':>13}")
    for r in rows:
        if r["split"] != "test":
            continue
        print(f"{r['model']:<10}{r['logloss']:>9.4f}{r['brier']:>9.4f}{r['accuracy']:>8.3f}"
              f"{r['macro_f1']:>10.4f}{r['draw_recall']:>13.3f}")

    # save the trained model + scalers + arch for later serve
    import torch
    ckpt = ROOT / "artifacts" / f"{args.tag}.pt"
    torch.save({
        "state_dict": res.model.state_dict(),
        "arch": res.config,
        "temperature": res.temperature,
        "k": args.k,
        "leagues": leagues,
        "scalers": sc,
        "best_val_logloss": res.best_val_logloss,
        "stack_weights": stack_weights,
    }, ckpt)
    print(f"\nSaved model -> {ckpt.name} | report -> {report_config.final_probability_quality_file.name}")


if __name__ == "__main__":
    main()
