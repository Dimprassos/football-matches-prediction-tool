"""Assemble aligned (static features + match sequences + labels) for FootyNet.

The deep model needs, per fixture and *row-aligned*:
* ``static`` — the exact engineered feature vector the tabular models consume
  (``build_meta_features`` over the leakage-safe streaming pass: model logits +
  market logits + aux), so the comparison is apples-to-apples;
* ``seq_home``/``seq_away`` (+ masks) — the last-K match sequences from
  :mod:`src.sequence_data`.

Both are built from the *same* played-history dataframe with the *same* leakage
rule (matches strictly before each fixture's date). Streaming emits rows in
``predict_df.sort_values("date")`` order, so we build the sequences over that same
ordering and then assert the reconstructed labels match streaming's ``y`` — a hard
guarantee against silent row misalignment.

Base params (Elo/Poisson ``beta/rho/decay/K/ha/T``) are seeded from the canonical
experiment's cached params (``FINAL_CONFIG``) — FootyNet does not retune the base.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from src.artifact_store import load_json_if_exists
from src.calibration import temperature_scale_probs
from src.config import FINAL_CONFIG, ExperimentConfig
from src.data_processing import load_league_data
from src.feature_builder import build_meta_features, ensure_market_probs
from src.sequence_data import (
    DEFAULT_SEQUENCE_LENGTH,
    build_fixture_sequences,
    build_team_sequences,
)
from src.state_builder import streaming_block_probs_home_away

POOL_KEYS = ("seq_home", "seq_away", "mask_home", "mask_away", "static", "y", "market", "raw_odds")


def _outcome(df: pd.DataFrame) -> np.ndarray:
    """1X2 label (0=home, 1=draw, 2=away) for each played row."""
    hg = pd.to_numeric(df["home_goals"], errors="coerce").to_numpy()
    ag = pd.to_numeric(df["away_goals"], errors="coerce").to_numpy()
    return np.where(hg > ag, 0, np.where(hg == ag, 1, 2)).astype(int)


def _split_played_periods(df: pd.DataFrame, config: ExperimentConfig):
    """Same fixed date split as the trainer (inlined to avoid importing trainer)."""
    train_fit = df[df["date"] < config.train_cut].copy()
    val = df[(df["date"] >= config.train_cut) & (df["date"] < config.test_cut)].copy()
    if config.test_end is None:
        test = df[df["date"] >= config.test_cut].copy()
    else:
        test = df[(df["date"] >= config.test_cut) & (df["date"] < config.test_end)].copy()
    return train_fit, val, test


def _build_split(split_df, full_df, params, k, odds_kwargs, team_sequences) -> dict | None:
    """Build one split's aligned static + sequence tensors + labels (or None if empty)."""
    if split_df is None or len(split_df) == 0:
        return None
    probs_raw, y, mkt, aux, raw_odds = streaming_block_probs_home_away(
        split_df, full_df,
        params["beta"], params["rho"], params["decay"], params["K"], params["ha"],
        **odds_kwargs,
    )
    probs = temperature_scale_probs(probs_raw, params["T"])
    market = ensure_market_probs(probs, mkt)
    static = np.asarray(build_meta_features(probs, market, aux), dtype=float)

    ordered = split_df.sort_values("date")  # streaming's row order
    seqs = build_fixture_sequences(full_df, ordered, k=k, team_sequences=team_sequences)

    y = np.asarray(y, dtype=int)
    if not np.array_equal(_outcome(ordered), y):
        raise RuntimeError("FootyNet row alignment mismatch: sequences vs streaming labels")

    return {
        **seqs,
        "static": static,
        "y": y,
        "market": np.asarray(market, dtype=float),
        "raw_odds": np.asarray(raw_odds, dtype=float),
    }


def build_league_datasets(
    league: str,
    *,
    config: ExperimentConfig = FINAL_CONFIG,
    params: dict | None = None,
    k: int = DEFAULT_SEQUENCE_LENGTH,
) -> dict[str, dict | None]:
    """Build train/val/test FootyNet datasets for one league (leakage-safe, aligned)."""
    df_all = load_league_data(league).sort_values("date").reset_index(drop=True)
    df = df_all[df_all["is_played"] == True].copy().reset_index(drop=True)  # noqa: E712

    if params is None:
        cached = load_json_if_exists(config.params_file) or {}
        params = cached.get(league)
        if params is None:
            raise ValueError(
                f"No cached base params for {league!r} in {config.params_file}. "
                "Run `python scripts/main.py` first to tune the base model."
            )

    odds_kwargs = {
        "market_odds_source": config.market_odds_source,
        "betting_odds_source": config.betting_odds_source,
        "include_market_movement_features": config.include_market_movement_features,
    }
    train_fit, val, test = _split_played_periods(df, config)
    team_sequences = build_team_sequences(df)  # date<d filter keeps each split leakage-safe
    return {
        "train": _build_split(train_fit, df, params, k, odds_kwargs, team_sequences),
        "val": _build_split(val, df, params, k, odds_kwargs, team_sequences),
        "test": _build_split(test, df, params, k, odds_kwargs, team_sequences),
    }


def pool_split(dataset_list: list[dict], split: str) -> dict | None:
    """Concatenate one split's tensors across leagues into a single pooled dict."""
    parts = [d[split] for d in dataset_list if d.get(split) is not None]
    if not parts:
        return None
    return {key: np.concatenate([p[key] for p in parts], axis=0) for key in POOL_KEYS}
