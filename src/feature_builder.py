"""The canonical feature layout and the helpers that build feature vectors.

:data:`FEATURE_COLUMNS` is the single source of truth for the order of every
feature the models consume; the other ``*_FEATURE_COLUMNS`` lists are named subsets
(market, local stats, understat, team news, weather...) that the meta-models choose
among. :func:`feature_indices` converts a list of names to column positions, and
the ``build_*`` functions assemble the matrices fed to the models. The order here
must stay in sync with the vectors produced in :mod:`src.state_builder`.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from src.calibration import safe_logit


# Canonical feature order (index positions matter everywhere): 3 model logits,
# 3 market logits, then the base-aux block, then the extra-aux blocks.
FEATURE_COLUMNS = [
    "model_logit_home", "model_logit_draw", "model_logit_away",
    "market_logit_home", "market_logit_draw", "market_logit_away",
    "elo_diff", "total_xg", "xg_diff", "mom_home", "mom_away", "mom_diff",
    "rest_home", "rest_away", "rest_diff",
    "form_home", "form_away", "form_diff",
    "goals_for_home_5", "goals_for_away_5", "goals_for_diff_5",
    "goals_against_home_5", "goals_against_away_5", "goals_against_diff_5",
    "shots_for_home_5", "shots_for_away_5", "shots_for_diff_5",
    "shots_against_home_5", "shots_against_away_5", "shots_against_diff_5",
    "sot_for_home_5", "sot_for_away_5", "sot_for_diff_5",
    "sot_against_home_5", "sot_against_away_5", "sot_against_diff_5",
    "corners_for_home_5", "corners_for_away_5", "corners_for_diff_5",
    "corners_against_home_5", "corners_against_away_5", "corners_against_diff_5",
    "cards_home_5", "cards_away_5", "cards_diff_5",
    "market_move_home", "market_move_draw", "market_move_away",
    "ou25_over_prob", "ah_line",
    "understat_xg_for_home_5", "understat_xg_for_away_5", "understat_xg_diff_5",
    "understat_npxg_for_home_5", "understat_npxg_for_away_5", "understat_npxg_diff_5",
    "understat_xpts_home_5", "understat_xpts_away_5", "understat_xpts_diff_5",
    "lineup_available",
    "home_lineup_strength", "away_lineup_strength", "lineup_strength_diff",
    "team_news_available",
    "home_absence_count", "away_absence_count", "absence_count_diff",
    "home_injury_count", "away_injury_count", "injury_count_diff",
    "home_suspension_count", "away_suspension_count", "suspension_count_diff",
    "home_key_absence_count", "away_key_absence_count", "key_absence_count_diff",
    "home_manager_change_recent", "away_manager_change_recent", "manager_change_recent_diff",
    "weather_available",
    "temperature_c", "wind_kph", "precipitation_mm", "weather_severity",
]

MLP_DEFAULT_FEATURE_COLUMNS = [
    "model_logit_home", "model_logit_draw", "model_logit_away",
    "market_logit_home", "market_logit_draw", "market_logit_away",
    "elo_diff",
    "mom_home", "mom_away", "mom_diff",
    "form_home", "form_away", "form_diff",
]
LOCAL_STATS_FEATURE_COLUMNS = [
    "goals_for_home_5", "goals_for_away_5", "goals_for_diff_5",
    "goals_against_home_5", "goals_against_away_5", "goals_against_diff_5",
    "shots_for_home_5", "shots_for_away_5", "shots_for_diff_5",
    "shots_against_home_5", "shots_against_away_5", "shots_against_diff_5",
    "sot_for_home_5", "sot_for_away_5", "sot_for_diff_5",
    "sot_against_home_5", "sot_against_away_5", "sot_against_diff_5",
    "corners_for_home_5", "corners_for_away_5", "corners_for_diff_5",
    "corners_against_home_5", "corners_against_away_5", "corners_against_diff_5",
    "cards_home_5", "cards_away_5", "cards_diff_5",
]
ODDS_MOVEMENT_FEATURE_COLUMNS = ["market_move_home", "market_move_draw", "market_move_away"]
MARKET_CONTEXT_FEATURE_COLUMNS = ODDS_MOVEMENT_FEATURE_COLUMNS + ["ou25_over_prob", "ah_line"]
UNDERSTAT_XG_FEATURE_COLUMNS = [
    "understat_xg_for_home_5", "understat_xg_for_away_5", "understat_xg_diff_5",
    "understat_npxg_for_home_5", "understat_npxg_for_away_5", "understat_npxg_diff_5",
    "understat_xpts_home_5", "understat_xpts_away_5", "understat_xpts_diff_5",
]
TEAM_NEWS_FEATURE_COLUMNS = [
    "lineup_available",
    "home_lineup_strength", "away_lineup_strength", "lineup_strength_diff",
    "team_news_available",
    "home_absence_count", "away_absence_count", "absence_count_diff",
    "home_injury_count", "away_injury_count", "injury_count_diff",
    "home_suspension_count", "away_suspension_count", "suspension_count_diff",
    "home_key_absence_count", "away_key_absence_count", "key_absence_count_diff",
    "home_manager_change_recent", "away_manager_change_recent", "manager_change_recent_diff",
]
WEATHER_FEATURE_COLUMNS = [
    "weather_available", "temperature_c", "wind_kph", "precipitation_mm", "weather_severity",
]
EXTERNAL_CONTEXT_FEATURE_COLUMNS = TEAM_NEWS_FEATURE_COLUMNS + WEATHER_FEATURE_COLUMNS
BASE_NON_MARKET_FEATURE_COLUMNS = [
    col for col in FEATURE_COLUMNS
    if not col.startswith("market_logit_") and col not in MARKET_CONTEXT_FEATURE_COLUMNS
]
NEW_LOCAL_FEATURE_COLUMNS = LOCAL_STATS_FEATURE_COLUMNS + MARKET_CONTEXT_FEATURE_COLUMNS
NEW_DATA_FEATURE_COLUMNS = NEW_LOCAL_FEATURE_COLUMNS + UNDERSTAT_XG_FEATURE_COLUMNS + EXTERNAL_CONTEXT_FEATURE_COLUMNS


def feature_indices(feature_names: list[str] | tuple[str, ...]) -> list[int]:
    """Convert feature names to their column positions in :data:`FEATURE_COLUMNS`."""
    return [FEATURE_COLUMNS.index(col) for col in feature_names]


MLP_DEFAULT_COLS = feature_indices(MLP_DEFAULT_FEATURE_COLUMNS)
MLP_NO_REST_COLS = [i for i, col in enumerate(FEATURE_COLUMNS) if not col.startswith("rest_")]


def market_probs_from_odds_row(odds_h, odds_d, odds_a):
    """Convert decimal 1/X/2 odds to overround-removed probabilities (NaN if invalid).

    Inverts the odds and renormalises to strip the bookmaker margin. Returns NaNs
    when any odd is missing or <= 1.0 (i.e. no usable market).
    """
    if not (np.isfinite(odds_h) and np.isfinite(odds_d) and np.isfinite(odds_a)):
        return np.array([np.nan, np.nan, np.nan], dtype=float)
    if odds_h <= 1.0001 or odds_d <= 1.0001 or odds_a <= 1.0001:
        return np.array([np.nan, np.nan, np.nan], dtype=float)
    inv = np.array([1.0 / odds_h, 1.0 / odds_d, 1.0 / odds_a], dtype=float)
    s = inv.sum()
    if s <= 0:
        return np.array([np.nan, np.nan, np.nan], dtype=float)
    return inv / s


def ensure_market_probs(model_probs: np.ndarray, market_probs: np.ndarray) -> np.ndarray:
    """Fill any non-finite market rows with the model's probabilities (so features stay valid)."""
    fixed = np.asarray(market_probs, dtype=float).copy()
    model_probs = np.asarray(model_probs, dtype=float)
    for i in range(len(fixed)):
        if not np.isfinite(fixed[i]).all():
            fixed[i] = model_probs[i]
    return fixed


def build_meta_features(model_probs: np.ndarray, market_probs: np.ndarray, aux: np.ndarray) -> np.ndarray:
    """Assemble the full feature matrix: model logits, market logits, then aux features.

    Probabilities are stored as logits (via ``safe_logit``) so the linear/tree models
    see an unbounded representation. The aux block is appended per row in FEATURE_COLUMNS order.
    """
    model_probs = np.asarray(model_probs, dtype=float)
    market_fixed = ensure_market_probs(model_probs, market_probs)
    aux = np.asarray(aux, dtype=float)

    X = []
    for i in range(len(model_probs)):
        pm = model_probs[i]
        pk = market_fixed[i]
        feats = [
            safe_logit(pm[0]), safe_logit(pm[1]), safe_logit(pm[2]),
            safe_logit(pk[0]), safe_logit(pk[1]), safe_logit(pk[2]),
        ]
        feats.extend(aux[i].tolist())
        X.append(feats)
    return np.array(X, dtype=float)


def build_single_feature_vector(
    model_probs,
    market_probs,
    *,
    elo_h,
    elo_a,
    lam_h,
    lam_a,
    mom_h,
    mom_a,
    rest_h,
    rest_a,
    form_h,
    form_a,
    extra_aux=None,
):
    """Build the 1-row feature matrix for a single fixture (used at serve time).

    Constructs the 12-element base-aux block from the supplied components, appends
    ``extra_aux`` when given, and delegates to :func:`build_meta_features`.
    """
    mom_diff = mom_h - mom_a
    rest_diff = rest_h - rest_a
    form_diff = form_h - form_a
    aux = np.array([[
        (elo_h - elo_a) / 400.0,
        lam_h + lam_a,
        lam_h - lam_a,
        mom_h,
        mom_a,
        mom_diff,
        rest_h,
        rest_a,
        rest_diff,
        form_h,
        form_a,
        form_diff,
    ]], dtype=float)
    if extra_aux is not None:
        aux = np.concatenate([aux, np.asarray(extra_aux, dtype=float).reshape(1, -1)], axis=1)
    return build_meta_features(np.asarray([model_probs], dtype=float), np.asarray([market_probs], dtype=float), aux)


def time_split_val(val_df: pd.DataFrame):
    """Split the validation set chronologically into early and late halves.

    The late half is the held-out slice used to tune feature subsets, blend weights,
    and the market correction, keeping that tuning separate from model fitting.
    """
    val_sorted = val_df.sort_values("date").reset_index(drop=True)
    mid = len(val_sorted) // 2
    return val_sorted.iloc[:mid].copy(), val_sorted.iloc[mid:].copy()
