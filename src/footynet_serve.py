"""Serve the trained FootyNet model for a single interactive fixture (chunk DL-5).

Rebuilds, for a hypothetical ``home`` vs ``away`` fixture as of a given date, the
*same* inputs FootyNet trained on — the static feature vector (base model probs +
market + aux, via the canonical league state) and the two teams' last-K match
sequences — applies the saved standardizers, and returns calibrated 1X2
probabilities. Mirrors :mod:`src.footynet_data` so train and serve stay aligned.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from src.calibration import temperature_scale_probs
from src.feature_builder import build_meta_features, market_probs_from_odds_row
from src.footynet_stack import apply_blend
from src.models.footynet import FootyNet, predict_proba
from src.predictor import build_runtime_extra_features
from src.sequence_data import build_team_sequences, last_k_before
from src.state_builder import compute_match_components


def load_footynet(ckpt_path, device: str = "cpu"):
    """Load a trained FootyNet checkpoint into a ready-to-eval model + its metadata."""
    import torch

    ck = torch.load(ckpt_path, map_location=device, weights_only=False)
    arch = ck["arch"]
    model = FootyNet(
        static_dim=arch["static_dim"], hidden=arch["hidden"],
        lstm_layers=arch["lstm_layers"], dropout=arch["dropout"],
        cell=arch.get("cell", "lstm"),
    ).to(device)
    model.load_state_dict(ck["state_dict"])
    model.eval()
    return model, ck


def _scale_seq(seq: np.ndarray, mask: np.ndarray, sc: dict) -> np.ndarray:
    out = (seq - sc["seq_mu"]) / sc["seq_sd"]
    return out * mask[..., None]  # keep padded steps at zero


def predict_footynet_fixture(
    home: str,
    away: str,
    odds_h: float,
    odds_d: float,
    odds_a: float,
    *,
    state,
    model,
    ckpt: dict,
    as_of=None,
    team_sequences: dict | None = None,
) -> dict:
    """Return ``{"footynet", "market", "base"}`` 1X2 probability arrays for one fixture.

    ``state`` is the canonical league runtime state (Elo/strengths + ``played_df``);
    ``model``/``ckpt`` come from :func:`load_footynet`. Decimal odds <= 1.0 are treated
    as "no market" (FootyNet then sees the base model probs in the market slot, exactly
    as in training when odds were missing).
    """
    played_df = getattr(state, "played_df", None)
    if played_df is None or len(played_df) == 0:
        raise ValueError("FootyNet serve needs a league state with played history.")
    if as_of is None:
        as_of = pd.to_datetime(played_df["date"]).max() + pd.Timedelta(days=1)

    # --- static branch: identical assembly to training (footynet_data) ---
    extra_aux = build_runtime_extra_features(home, away, state, odds_h, odds_d, odds_a)
    comp = compute_match_components(home, away, state, match_date=as_of, extra_aux=extra_aux)
    probs_cal = temperature_scale_probs(np.array([comp["probs"]], dtype=float), state.params["T"])[0]
    if odds_h > 1.0 and odds_d > 1.0 and odds_a > 1.0:
        market = market_probs_from_odds_row(odds_h, odds_d, odds_a)
    else:
        market = probs_cal.copy()
    static = np.asarray(
        build_meta_features(np.array([probs_cal]), np.array([market]), np.array([comp["aux"]])),
        dtype=float,
    )

    # --- sequence branch: last-K matches of each team strictly before as_of ---
    k = int(ckpt["k"])
    sc = ckpt["scalers"]
    if team_sequences is None:
        team_sequences = build_team_sequences(played_df)
    sh, mh = last_k_before(team_sequences.get(home), as_of, k)
    sa, ma = last_k_before(team_sequences.get(away), as_of, k)

    data = {
        "static": (static - sc["static_mu"]) / sc["static_sd"],
        "seq_home": _scale_seq(sh[None], mh[None], sc), "mask_home": mh[None],
        "seq_away": _scale_seq(sa[None], ma[None], sc), "mask_away": ma[None],
    }
    foot = predict_proba(model, data, float(ckpt["temperature"]))[0]
    out = {
        "footynet": np.asarray(foot, dtype=float),
        "market": np.asarray(market, dtype=float),
        "base": np.asarray(probs_cal, dtype=float),
    }
    # stacking blend (weights learned on validation at train time). When odds are
    # missing the "market" slot holds the base probs, so the stack degrades gracefully.
    weights = ckpt.get("stack_weights")
    if weights:
        members = {"footynet": out["footynet"][None], "market": out["market"][None]}
        out["footynet_stack"] = np.asarray(apply_blend(members, weights)[0], dtype=float)
    return out
