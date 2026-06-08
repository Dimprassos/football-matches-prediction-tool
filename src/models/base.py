"""Hyper-parameter tuning for the base (Elo + Dixon-Coles Poisson) model.

The base model combines per-team attack/defence strengths (a Poisson/Dixon-Coles
goal model) with an Elo adjustment of the scoring rates. This module grid-searches
its parameters on the validation split in a leakage-safe way and returns the best
config, which the trainer then freezes per league.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.metrics import log_loss

from src.calibration import fit_temperature, temperature_scale_probs
from src.elo import compute_elo_ratings
from src.poisson_model import apply_elo_to_lambdas, fit_team_strengths, fit_team_strengths_weighted, match_outcome_probs, predict_lambdas
from src.state_builder import streaming_block_probs_home_away


def _validation_rows_with_elo(train_fit, val, *, K, home_adv):
    """Return the validation rows with Elo ratings computed leakage-free.

    Elo is path-dependent, so the ratings going into a validation match must come
    from a single chronological pass over train+validation (a match's rating must
    reflect only earlier matches). We tag the validation rows, concatenate, sort by
    date, run Elo once, then extract the validation slice back in its original order.
    """
    train_tagged = train_fit.copy()
    train_tagged["_is_tune_validation"] = False
    train_tagged["_tune_validation_order"] = -1

    val_tagged = val.copy()
    val_tagged["_is_tune_validation"] = True
    val_tagged["_tune_validation_order"] = np.arange(len(val_tagged))

    full_tmp = (
        pd.concat([train_tagged, val_tagged], ignore_index=True)
        .sort_values(["date", "_is_tune_validation", "_tune_validation_order"], kind="mergesort")
        .reset_index(drop=True)
    )
    elo_full = compute_elo_ratings(full_tmp, K=K, home_adv=home_adv, use_margin=True)
    full_tmp["elo_home"], full_tmp["elo_away"] = zip(*elo_full)

    val_part = (
        full_tmp[full_tmp["_is_tune_validation"]]
        .sort_values("_tune_validation_order", kind="mergesort")
        .drop(columns=["_is_tune_validation", "_tune_validation_order"])
        .reset_index(drop=True)
    )
    if len(val_part) != len(val):
        raise ValueError(f"Validation tuning row alignment failed: expected {len(val)}, got {len(val_part)}")
    return val_part


def tune_league_params(train_fit, val, full_played_df):
    """Grid-search the base-model parameters for one league on its validation split.

    Searched (by validation log loss):

    * ``K`` (Elo update rate) and ``home_adv`` (Elo home edge), jointly;
    * ``beta`` (how strongly Elo shifts the Poisson scoring rates);
    * ``decay`` (exponential time-decay weighting of older matches in the strength fit);
    * finally ``rho`` (Dixon-Coles low-score correlation correction) and ``T``
      (temperature for probability calibration), tuned jointly on the leakage-safe
      streaming probabilities.

    Returns a dict of the chosen ``{K, ha, beta, decay, rho, T}``.
    """
    Ks = [40, 50, 60, 70]
    home_advs = [60, 80, 100, 110]
    betas = [0.10, 0.11, 0.12, 0.13]
    decays = [0.0005, 0.001, 0.002, 0.003]

    best = None
    print("\n--- Elo & Beta Tuning ---")

    l_avg_h, l_avg_a, att, dfn = fit_team_strengths(train_fit)
    for K in Ks:
        for ha in home_advs:
            val_part = _validation_rows_with_elo(train_fit, val, K=K, home_adv=ha)
            val_inputs = []
            y = []
            for _, row in val_part.iterrows():
                lh, la = predict_lambdas(row["home_team"], row["away_team"], l_avg_h, l_avg_a, att, dfn)
                val_inputs.append((lh, la, row["elo_home"], row["elo_away"]))
                if row["home_goals"] > row["away_goals"]:
                    y.append(0)
                elif row["home_goals"] == row["away_goals"]:
                    y.append(1)
                else:
                    y.append(2)
            y = np.array(y, dtype=int)

            for b in betas:
                probs = []
                for lh_base, la_base, elo_home, elo_away in val_inputs:
                    lh, la = apply_elo_to_lambdas(lh_base, la_base, elo_home, elo_away, beta=b)
                    probs.append(match_outcome_probs(lh, la))

                ll = log_loss(y, np.array(probs))
                if best is None or ll < best[0]:
                    best = (ll, K, ha, b)

    _, best_K, best_ha, best_beta = best
    print(f"Best Config: K={best_K}, ha={best_ha}, beta={best_beta}")

    print("--- Tuning Time Decay ---")
    best_decay, best_decay_ll = None, float("inf")
    y_val = np.array([0 if r["home_goals"] > r["away_goals"] else 1 if r["home_goals"] == r["away_goals"] else 2 for _, r in val.iterrows()], dtype=int)

    val_part = _validation_rows_with_elo(train_fit, val, K=best_K, home_adv=best_ha)

    for d in decays:
        l_avg_h, l_avg_a, att, dfn = fit_team_strengths_weighted(train_fit, decay=d)
        probs = []
        for _, row in val_part.iterrows():
            lh, la = predict_lambdas(row["home_team"], row["away_team"], l_avg_h, l_avg_a, att, dfn)
            lh, la = apply_elo_to_lambdas(lh, la, row["elo_home"], row["elo_away"], beta=best_beta)
            probs.append(match_outcome_probs(lh, la))
        ll = log_loss(y_val, np.array(probs))
        if ll < best_decay_ll:
            best_decay_ll, best_decay = ll, d

    print(f"Best Decay: {best_decay}")
    print("--- Joint Tuning rho + Temperature ---")
    rho_grid = np.arange(-0.2, 0.21, 0.02)
    best_joint = None
    for rho in rho_grid:
        print(f"  rho={rho:.2f}")
        val_probs_raw, y_val_stream, _, _, _ = streaming_block_probs_home_away(
            val, full_played_df,
            beta=best_beta, rho=float(rho), decay=best_decay,
            K=best_K, home_adv=best_ha,
        )
        T = fit_temperature(val_probs_raw, y_val_stream)
        val_probs_cal = temperature_scale_probs(val_probs_raw, T)
        ll = log_loss(y_val_stream, val_probs_cal)
        if best_joint is None or ll < best_joint[0]:
            best_joint = (ll, float(rho), float(T))

    _, best_rho, best_T = best_joint
    print(f"Best rho={best_rho}, T={best_T}")
    return {"K": int(best_K), "ha": int(best_ha), "beta": float(best_beta), "decay": float(best_decay), "rho": float(best_rho), "T": float(best_T)}
