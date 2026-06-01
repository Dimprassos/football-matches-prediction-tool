from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
from xgboost import XGBClassifier

from src.artifact_store import load_json_if_exists, load_pickle_if_exists
from src.calibration import temperature_scale_probs
from src.config import DEFAULT_CONFIG, ExperimentConfig
from src.data_loader import load_league_data
from src.feature_builder import MLP_DEFAULT_COLS, build_single_feature_vector, feature_indices, market_probs_from_odds_row
from src.models.meta import apply_market_logit_correction, blend_probabilities
from src.poisson_model import top_k_scorelines_dc
from src.state_builder import build_league_state, compute_match_components, neutral_extra_features


def load_runtime_artifacts(config: ExperimentConfig = DEFAULT_CONFIG):
    params = load_json_if_exists(config.params_file)
    if params is None:
        sys.exit("Error: Parameters file not found. Run main.py first.")
    if not config.model_file.exists():
        sys.exit("Error: XGBoost model not found. Run main.py first.")

    meta_model = XGBClassifier()
    meta_model.load_model(str(config.model_file))
    meta_cfg = load_json_if_exists(config.meta_file)
    mlp_model = load_pickle_if_exists(config.mlp_model_file)
    mlp_meta = load_json_if_exists(config.mlp_meta_file)
    logreg_model = load_pickle_if_exists(config.logreg_model_file)
    logreg_meta = load_json_if_exists(config.logreg_meta_file)
    blend_cfg = load_json_if_exists(config.blend_file)
    return params, meta_model, meta_cfg, mlp_model, mlp_meta, logreg_model, logreg_meta, blend_cfg


def get_league_runtime_state(league_name: str, params: dict):
    df = load_league_data(league_name)
    df = df[df["is_played"] == True].sort_values("date").reset_index(drop=True)
    return build_league_state(df, params[league_name])


def predict_custom_match(
    home,
    away,
    odds_h,
    odds_d,
    odds_a,
    state,
    meta_model,
    meta_cfg,
    mlp_model,
    mlp_meta,
    blend_cfg,
    logreg_model=None,
    logreg_meta=None,
):
    extra_aux = neutral_extra_features()
    comp = compute_match_components(home, away, state, extra_aux=extra_aux)
    model_probs_raw = np.array([comp["probs"]], dtype=float)
    model_probs_cal = temperature_scale_probs(model_probs_raw, state.params["T"])[0]
    if odds_h > 1.0 and odds_d > 1.0 and odds_a > 1.0:
        mkt_probs = market_probs_from_odds_row(odds_h, odds_d, odds_a)
    else:
        mkt_probs = model_probs_cal.copy()

    X = build_single_feature_vector(
        model_probs_cal,
        mkt_probs,
        elo_h=comp["elo_home"],
        elo_a=comp["elo_away"],
        lam_h=comp["lam_home"],
        lam_a=comp["lam_away"],
        mom_h=comp["mom_home"],
        mom_a=comp["mom_away"],
        rest_h=comp["rest_home"],
        rest_a=comp["rest_away"],
        form_h=comp["form_home"],
        form_a=comp["form_away"],
        extra_aux=extra_aux,
    )
    xgb_cols = feature_indices(meta_cfg.get("feature_columns", [])) if meta_cfg is not None and meta_cfg.get("feature_columns") else list(range(X.shape[1]))
    meta_probs = meta_model.predict_proba(X[:, xgb_cols])[0]
    if mlp_model is not None:
        mlp_cols = feature_indices(mlp_meta.get("feature_columns", [])) if mlp_meta is not None and mlp_meta.get("feature_columns") else MLP_DEFAULT_COLS
        mlp_probs_raw = mlp_model.predict_proba(X[:, mlp_cols])
        if mlp_meta is not None and "temperature" in mlp_meta:
            mlp_probs = temperature_scale_probs(mlp_probs_raw, float(mlp_meta["temperature"]))[0]
        else:
            mlp_probs = mlp_probs_raw[0]
    else:
        mlp_probs = meta_probs.copy()
    if logreg_model is not None and logreg_meta is not None:
        logreg_cols = feature_indices(logreg_meta.get("feature_columns", [])) if logreg_meta.get("feature_columns") else list(range(X.shape[1]))
        logreg_probs_raw = logreg_model.predict_proba(X[:, logreg_cols])
        logreg_probs = temperature_scale_probs(logreg_probs_raw, float(logreg_meta.get("temperature", 1.0)))[0]
    else:
        logreg_probs = meta_probs.copy()

    if blend_cfg is not None:
        ens_probs = blend_probabilities(blend_cfg["weights"], {
            "base": model_probs_cal.reshape(1, -1),
            "market": mkt_probs.reshape(1, -1),
            "xgb": meta_probs.reshape(1, -1),
            "mlp": mlp_probs.reshape(1, -1),
        })[0]
        market_corr_probs = apply_market_logit_correction(
            mkt_probs.reshape(1, -1),
            {
                "base": model_probs_cal.reshape(1, -1),
                "xgb": meta_probs.reshape(1, -1),
                "mlp": mlp_probs.reshape(1, -1),
            },
            blend_cfg.get("market_correction"),
        )[0]
    else:
        ens_probs = meta_probs.copy()
        market_corr_probs = mkt_probs.copy()

    top_scores = top_k_scorelines_dc(comp["lam_home"], comp["lam_away"], state.params["rho"], k=3)
    return {
        "base": model_probs_cal,
        "market": mkt_probs,
        "market_corr": market_corr_probs,
        "meta": meta_probs,
        "logreg": logreg_probs,
        "mlp": mlp_probs,
        "ensemble": ens_probs,
        "elo": (comp["elo_home"], comp["elo_away"]),
        "xg": (comp["lam_home"], comp["lam_away"]),
        "scores": top_scores,
    }
