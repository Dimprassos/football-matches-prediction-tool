"""Per-league isotonic recalibration experiment (leakage-safe, offline).

Question: on top of the temperature scaling the pipeline already applies, does an
extra *per-league* isotonic recalibration layer reduce test log loss?

Method (no leakage, does not touch any canonical artifact):
  * Reads the test-set probability dumps `final_per_match_predictions_*.csv`.
  * For each model and each league, recalibrates with **cross-fitted** isotonic
    regression: a one-vs-rest IsotonicRegression is fit on K-1 stratified folds
    and used to predict the held-out fold, so every calibrated probability comes
    from a model that never saw that match. The three one-vs-rest outputs are
    renormalised to sum to 1.
  * Compares log loss before vs after, per season and per model.

Because the isotonic layer is fit out-of-fold, a *negative* delta (no gain, or a
loss) is itself a valid finding: it means temperature scaling already captures
the available calibration signal and isotonic only adds variance.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.isotonic import IsotonicRegression
from sklearn.model_selection import StratifiedKFold

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))  # project root on path

SEASONS = {
    "2022-23": "season_backtest_2022_2023_opening_market_opening_price_no_move",
    "2023-24": "season_backtest_2023_2024_opening_market_opening_price_no_move",
    "2024-25": "season_backtest_2024_2025_opening_market_opening_price_no_move",
}
ARTIFACTS = "artifacts"
N_SPLITS = 5
EPS = 1e-15


def model_names(df: pd.DataFrame) -> list[str]:
    return sorted({c[:-4] for c in df.columns if c.endswith("_p_h")})


def model_probs(df: pd.DataFrame, model: str) -> np.ndarray:
    return df[[f"{model}_p_h", f"{model}_p_d", f"{model}_p_a"]].to_numpy(dtype=float)


def logloss(probs: np.ndarray, y: np.ndarray) -> float:
    p = np.clip(probs[np.arange(len(y)), y], EPS, 1.0)
    return float(-np.log(p).mean())


def cross_fitted_isotonic(probs: np.ndarray, y: np.ndarray, rng_seed: int) -> np.ndarray:
    """Out-of-fold per-class isotonic recalibration, renormalised. Returns (N,3)."""
    n = len(y)
    out = probs.copy()
    # Need enough samples of every class for stratified folds; otherwise leave as-is.
    counts = np.bincount(y, minlength=3)
    n_splits = min(N_SPLITS, int(counts[counts > 0].min())) if (counts > 0).all() else 0
    if n_splits < 2:
        return out
    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=rng_seed)
    for tr, te in skf.split(probs, y):
        cal = np.zeros((len(te), 3))
        for c in range(3):
            iso = IsotonicRegression(out_of_bounds="clip", y_min=0.0, y_max=1.0)
            iso.fit(probs[tr, c], (y[tr] == c).astype(float))
            cal[:, c] = iso.predict(probs[te, c])
        s = cal.sum(axis=1, keepdims=True)
        # Fall back to the original row if all three collapse to ~0.
        bad = s[:, 0] < EPS
        cal[bad] = probs[te][bad]
        s[bad] = cal[bad].sum(axis=1, keepdims=True)
        out[te] = cal / s
    return out


def recalibrate_per_league(probs: np.ndarray, y: np.ndarray, league: np.ndarray, seed: int) -> np.ndarray:
    out = probs.copy()
    for lg in np.unique(league):
        mask = league == lg
        out[mask] = cross_fitted_isotonic(probs[mask], y[mask], seed)
    return out


def analyze_season(name: str, exp: str, seed: int):
    df = pd.read_csv(f"{ARTIFACTS}/final_per_match_predictions_{exp}.csv")
    y = df["y_true"].to_numpy(dtype=int)
    league = df["league"].to_numpy()
    # A single "pooled" group isolates whether any damage comes from per-league
    # fragmentation (few samples per fold) rather than from isotonic itself.
    pooled = np.zeros_like(y)
    models = model_names(df)

    print(f"\n================  {name}  ({len(y)} test matches, "
          f"{len(np.unique(league))} leagues, {N_SPLITS}-fold OOF isotonic)  ================")
    print(f"{'model':<14}{'logloss':>10}{'+iso(pool)':>12}{'delta':>9}{'+iso(lg)':>11}{'delta':>9}")
    for model in models:
        probs = model_probs(df, model)
        base_ll = logloss(probs, y)
        pool_ll = logloss(recalibrate_per_league(probs, y, pooled, seed), y)
        lg_ll = logloss(recalibrate_per_league(probs, y, league, seed), y)
        print(f"{model:<14}{base_ll:>10.4f}{pool_ll:>12.4f}{pool_ll - base_ll:>+9.4f}"
              f"{lg_ll:>11.4f}{lg_ll - base_ll:>+9.4f}")
    print("  delta < 0 means isotonic lowered log loss vs the pipeline's temperature-scaled "
          "probs (out-of-fold, no leakage). pool=all leagues together, lg=per league.")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seed", type=int, default=20260602)
    args = ap.parse_args()
    for name, exp in SEASONS.items():
        analyze_season(name, exp, args.seed)


if __name__ == "__main__":
    main()
