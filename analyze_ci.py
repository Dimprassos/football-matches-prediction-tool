"""Bootstrap confidence intervals for the season backtests.

Reads the per-match prediction dumps written by the pipeline
(`final_per_match_predictions_*.csv`) and computes, per season and per model:

  * test log loss, the per-match difference vs the market, and a bootstrap 95% CI
    for that difference (is the model significantly better/worse than the market?);
  * betting ROI computed with the *real* `betting_records` logic, plus a bootstrap
    95% CI for ROI (does the edge survive resampling, or is it noise?);
  * mean closing-line value (CLV) of the placed bets, the proper test of a genuine edge.

Nothing here re-derives probabilities -- it consumes the exact probabilities and odds
the pipeline used, so the numbers are reproducible and cannot drift.
"""
from __future__ import annotations

import argparse

import numpy as np
import pandas as pd

from src.evaluation import betting_records

SEASONS = {
    "2022-23": "season_backtest_2022_2023_opening_market_opening_price_no_move",
    "2023-24": "season_backtest_2023_2024_opening_market_opening_price_no_move",
    "2024-25": "season_backtest_2024_2025_opening_market_opening_price_no_move",
}
ARTIFACTS = "artifacts"
EDGE_THRESHOLD = 0.05  # matches the pipeline default


def model_names(df: pd.DataFrame) -> list[str]:
    return sorted({c[:-4] for c in df.columns if c.endswith("_p_h")})


def model_probs(df: pd.DataFrame, model: str) -> np.ndarray:
    return df[[f"{model}_p_h", f"{model}_p_d", f"{model}_p_a"]].to_numpy(dtype=float)


def per_match_nll(probs: np.ndarray, y: np.ndarray) -> np.ndarray:
    p = np.clip(probs[np.arange(len(y)), y], 1e-15, 1.0)
    return -np.log(p)


def per_match_bet_arrays(probs, raw_odds, y, match_info):
    """Return length-N stake/profit/clv arrays (0 stake for matches with no bet)."""
    n = len(y)
    stake = np.zeros(n)
    profit = np.zeros(n)
    clv = np.full(n, np.nan)
    recs = betting_records(probs, raw_odds, y, edge_threshold=EDGE_THRESHOLD, match_info=match_info)
    for r in recs.to_dict("records"):
        i = int(r["idx"])
        stake[i] = r["stake"]
        profit[i] = r["profit"]
        clv[i] = r["clv_decimal"]
    return stake, profit, clv


def roi_from(stake: np.ndarray, profit: np.ndarray) -> float:
    s = stake.sum()
    return float(profit.sum() / s * 100.0) if s > 0 else 0.0


def ci(values: np.ndarray, lo=2.5, hi=97.5) -> tuple[float, float]:
    return float(np.percentile(values, lo)), float(np.percentile(values, hi))


def analyze_season(name: str, exp: str, n_boot: int, rng: np.random.Generator):
    path = f"{ARTIFACTS}/final_per_match_predictions_{exp}.csv"
    df = pd.read_csv(path)
    y = df["y_true"].to_numpy(dtype=int)
    raw_odds = df[["odds_h", "odds_d", "odds_a"]].to_numpy(dtype=float)
    info = df.to_dict("records")
    n = len(y)
    models = model_names(df)

    mkt_nll = per_match_nll(model_probs(df, "market"), y)

    boot_idx = rng.integers(0, n, size=(n_boot, n))

    print(f"\n================  {name}  ({n} test matches, B={n_boot})  ================")
    print(f"{'model':<12}{'logloss':>9}{'d_vs_mkt':>10}{'95% CI (d)':>20}{'sig':>5}"
          f"{'ROI%':>8}{'ROI 95% CI':>20}{'sig':>5}{'bets':>6}{'CLV%':>7}")

    for model in models:
        probs = model_probs(df, model)
        nll = per_match_nll(probs, y)
        logloss = float(nll.mean())
        diff = nll - mkt_nll  # per-match logloss difference vs market (neg = better)
        d_mean = float(diff.mean())
        d_boot = diff[boot_idx].mean(axis=1)
        d_lo, d_hi = ci(d_boot)
        d_sig = "*" if (d_lo > 0 or d_hi < 0) else ""  # CI excludes 0

        stake, profit, clv = per_match_bet_arrays(probs, raw_odds, y, info)
        roi = roi_from(stake, profit)
        n_bets = int((stake > 0).sum())
        if n_bets > 0:
            s_boot = stake[boot_idx]
            p_boot = profit[boot_idx]
            ssum = s_boot.sum(axis=1)
            roi_boot = np.where(ssum > 0, p_boot.sum(axis=1) / np.where(ssum > 0, ssum, 1) * 100.0, 0.0)
            r_lo, r_hi = ci(roi_boot)
            r_sig = "*" if (r_lo > 0 or r_hi < 0) else ""
            finite_clv = clv[np.isfinite(clv)]
            clv_mean = float(finite_clv.mean()) * 100.0 if finite_clv.size else float("nan")
        else:
            r_lo = r_hi = 0.0
            r_sig = ""
            clv_mean = float("nan")

        print(f"{model:<12}{logloss:>9.4f}{d_mean:>+10.4f}"
              f"{('[%+.4f, %+.4f]' % (d_lo, d_hi)):>20}{d_sig:>5}"
              f"{roi:>+8.2f}{('[%+.1f, %+.1f]' % (r_lo, r_hi)):>20}{r_sig:>5}"
              f"{n_bets:>6}{clv_mean:>7.2f}")

    print("  d_vs_mkt < 0 means lower log loss than the market. '*' = bootstrap 95% CI excludes 0.")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--boot", type=int, default=10000)
    ap.add_argument("--seed", type=int, default=20260530)
    args = ap.parse_args()
    rng = np.random.default_rng(args.seed)
    for name, exp in SEASONS.items():
        analyze_season(name, exp, args.boot, rng)


if __name__ == "__main__":
    main()
