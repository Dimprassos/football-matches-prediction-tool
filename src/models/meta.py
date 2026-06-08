"""Learned meta-models, the ensemble blend, and the market-logit correction.

On top of the base (Elo + Dixon-Coles) model this module provides the learned
layers and the ways they are combined:

* **XGBoost / MLP / Logistic Regression** meta-models, each with its own
  hyper-parameter tuning, fitting, and validation-locked feature-subset selection.
* **Ensemble blend** (:func:`blend_probabilities` / :func:`tune_blend_weights`):
  a convex combination of the base/market/xgb/mlp probabilities whose weights are
  grid-searched on late validation.
* **Market-logit correction** (:func:`market_logit_correction_probs`): a
  conservative, log-space nudge of the opening market by a model, gated so it can
  only be accepted if it beats the market by a margin on late validation.

All tuning is done on a *late* slice of the training data (held-out validation),
never on the test set, so reported metrics are leakage-safe.
"""
from __future__ import annotations

import numpy as np
try:
    import optuna
except ImportError:  # fallback for environments without optuna
    optuna = None
from sklearn.metrics import log_loss
from sklearn.linear_model import LogisticRegression
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from xgboost import XGBClassifier

from src.calibration import fit_temperature, temperature_scale_probs

if optuna is not None:
    optuna.logging.set_verbosity(optuna.logging.WARNING)


def make_mlp_pipeline(best_mlp_cfg):
    """Build a StandardScaler + MLPClassifier pipeline from a tuned config dict.

    Uses early stopping on an internal validation fraction so the network does not
    overfit; the scaler is required because the raw features are on very different scales.
    """
    return make_pipeline(
        StandardScaler(),
        MLPClassifier(
            hidden_layer_sizes=tuple(best_mlp_cfg["hidden_layer_sizes"]),
            activation="relu",
            solver="adam",
            alpha=float(best_mlp_cfg["alpha"]),
            learning_rate_init=float(best_mlp_cfg["learning_rate_init"]),
            batch_size=64,
            max_iter=1500,
            random_state=42,
            early_stopping=True,
            validation_fraction=0.1,
            n_iter_no_change=20,
        ),
    )


def probs_from_meta_features(X_meta, start_idx):
    """Recover a normalised 1/X/2 probability matrix from 3 logit columns.

    The feature matrix stores each model's contribution as logits; this applies a
    sigmoid to the three columns starting at ``start_idx`` and renormalises to sum to 1.
    """
    logits = X_meta[:, start_idx:start_idx + 3]
    probs = 1.0 / (1.0 + np.exp(-logits))
    row_sums = probs.sum(axis=1, keepdims=True)
    row_sums[row_sums == 0] = 1.0
    return probs / row_sums


def blend_probabilities(weight_dict, probs_dict):
    """Weighted average of available probability matrices, renormalised per row.

    Only keys present (and non-None) in ``probs_dict`` contribute; weights are
    normalised by their realised total. If the total weight is zero, falls back to
    the XGBoost probabilities.
    """
    out = np.zeros_like(probs_dict["base"], dtype=float)
    total_w = 0.0
    for key, w in weight_dict.items():
        if key in probs_dict and probs_dict[key] is not None:
            out += float(w) * probs_dict[key]
            total_w += float(w)
    if total_w <= 0:
        out = probs_dict["xgb"].copy()
    else:
        out /= total_w
    row_sums = out.sum(axis=1, keepdims=True)
    row_sums[row_sums == 0] = 1.0
    return out / row_sums


def _normalize_prob_matrix(probs: np.ndarray, eps: float = 1e-12) -> np.ndarray:
    """Sanitise a probability matrix: replace NaN/inf, clip to [eps, 1], renormalise rows."""
    arr = np.asarray(probs, dtype=float)
    arr = np.nan_to_num(arr, nan=eps, posinf=1.0, neginf=eps)
    arr = np.clip(arr, eps, 1.0)
    row_sums = arr.sum(axis=1, keepdims=True)
    row_sums[~np.isfinite(row_sums) | (row_sums <= 0)] = 1.0
    return arr / row_sums


def market_logit_correction_probs(
    market_probs: np.ndarray,
    correction_probs: np.ndarray | None,
    alpha: float,
) -> np.ndarray:
    """
    Conservative market-aware correction:
    p = normalize(market ** (1 - alpha) * correction ** alpha)

    alpha=0 returns the market. Small alpha values allow a model to nudge the
    opening market without replacing it.
    """
    market = _normalize_prob_matrix(market_probs)
    if correction_probs is None or float(alpha) <= 0.0:
        return market.copy()

    correction = _normalize_prob_matrix(correction_probs)
    alpha = float(np.clip(alpha, 0.0, 1.0))
    logits = (1.0 - alpha) * np.log(market) + alpha * np.log(correction)
    logits = logits - np.max(logits, axis=1, keepdims=True)
    exp_logits = np.exp(logits)
    return exp_logits / exp_logits.sum(axis=1, keepdims=True)


def apply_market_logit_correction(
    market_probs: np.ndarray,
    source_probs: dict[str, np.ndarray],
    cfg: dict | None,
) -> np.ndarray:
    """Apply a tuned correction config to the market probabilities at serve/eval time.

    ``cfg`` (from :func:`tune_market_logit_correction`) names the source model and
    ``alpha``. If the config is empty or selects "market", the market is returned unchanged.
    """
    if not cfg:
        return _normalize_prob_matrix(market_probs)
    source_model = str(cfg.get("source_model", "market"))
    if source_model == "market":
        return _normalize_prob_matrix(market_probs)
    return market_logit_correction_probs(
        market_probs,
        source_probs.get(source_model),
        float(cfg.get("alpha", 0.0)),
    )


def tune_market_logit_correction(
    y_late: np.ndarray,
    market_probs: np.ndarray,
    candidate_probs: dict[str, np.ndarray | None],
    *,
    alpha_grid: list[float] | np.ndarray | None = None,
    min_improvement: float = 0.0005,
    fold_masks: list[np.ndarray] | None = None,
    min_fold_improvement: float = 0.0,
) -> dict:
    """
    Tune a small log-space correction over the market on late validation.

    The gate is deliberately conservative: if no source beats the market by at
    least `min_improvement`, the correction collapses to the market.
    """
    if alpha_grid is None:
        alpha_grid = [0.0, 0.005, 0.01, 0.02, 0.03, 0.05, 0.075, 0.10, 0.15, 0.20, 0.30, 0.40, 0.50, 0.75, 1.0]

    market_fixed = _normalize_prob_matrix(market_probs)
    market_ll = float(log_loss(y_late, market_fixed))
    best = {
        "source_model": "market",
        "alpha": 0.0,
        "late_val_logloss": market_ll,
        "late_market_logloss": market_ll,
        "accepted": False,
        "min_improvement": float(min_improvement),
        "min_fold_improvement": float(min_fold_improvement),
    }
    source_scores = []

    for name, probs in candidate_probs.items():
        if probs is None:
            continue
        source_ll = float(log_loss(y_late, _normalize_prob_matrix(probs)))
        source_best = {
            "source_model": name,
            "source_logloss": source_ll,
            "best_alpha": 0.0,
            "best_logloss": market_ll,
        }
        for alpha in alpha_grid:
            corrected = market_logit_correction_probs(market_fixed, probs, float(alpha))
            ll = float(log_loss(y_late, corrected))
            if ll < source_best["best_logloss"]:
                source_best["best_alpha"] = float(alpha)
                source_best["best_logloss"] = ll
            if ll < best["late_val_logloss"]:
                best.update({
                    "source_model": name,
                    "alpha": float(alpha),
                    "late_val_logloss": ll,
                    "source_logloss": source_ll,
                })
        source_scores.append(source_best)

    best["source_scores"] = source_scores
    if best["late_val_logloss"] < market_ll - float(min_improvement):
        corrected = apply_market_logit_correction(market_fixed, candidate_probs, best)
        fold_checks = []
        for fold_idx, fold_mask in enumerate(fold_masks or []):
            fold_mask = np.asarray(fold_mask, dtype=bool)
            if fold_mask.shape[0] != len(y_late) or not np.any(fold_mask):
                continue
            fold_market_ll = float(log_loss(y_late[fold_mask], market_fixed[fold_mask]))
            fold_corr_ll = float(log_loss(y_late[fold_mask], corrected[fold_mask]))
            fold_delta = fold_corr_ll - fold_market_ll
            fold_checks.append({
                "fold": int(fold_idx),
                "market_logloss": fold_market_ll,
                "corrected_logloss": fold_corr_ll,
                "delta_logloss": fold_delta,
                "passed": bool(fold_corr_ll <= fold_market_ll - float(min_fold_improvement)),
            })

        best["fold_checks"] = fold_checks
        if fold_checks and not all(row["passed"] for row in fold_checks):
            best.update({
                "source_model": "market",
                "alpha": 0.0,
                "late_val_logloss": market_ll,
                "accepted": False,
                "rejection_reason": "failed_chronological_fold_gate",
            })
            return best

        best["accepted"] = True
        return best

    best.update({
        "source_model": "market",
        "alpha": 0.0,
        "late_val_logloss": market_ll,
        "accepted": False,
        "rejection_reason": "insufficient_late_validation_improvement",
    })
    return best


def tune_xgb_hyperparams(X_early, y_early, X_late, y_late, n_trials=20):
    """Tune XGBoost (learning_rate, max_depth, n_estimators) by late-validation log loss.

    Fits on the early slice and scores on the late slice. Uses Optuna when available,
    otherwise falls back to a deterministic grid search. Returns the best config dict.
    """
    print("Tuning XGBoost Hyperparameters. Please wait...")

    def score_cfg(lr, md, ne):
        meta = XGBClassifier(
            n_estimators=ne, learning_rate=lr, max_depth=md,
            objective="multi:softprob", eval_metric="mlogloss",
            random_state=42, n_jobs=-1,
        )
        meta.fit(X_early, y_early)
        return log_loss(y_late, meta.predict_proba(X_late))

    if optuna is not None:
        def objective(trial):
            lr = trial.suggest_float("learning_rate", 0.005, 0.2, log=True)
            md = trial.suggest_int("max_depth", 2, 6)
            ne = trial.suggest_int("n_estimators", 50, 600)
            return score_cfg(lr, md, ne)

        study = optuna.create_study(direction="minimize")
        study.optimize(objective, n_trials=n_trials)
        return {
            "learning_rate": float(study.best_params["learning_rate"]),
            "max_depth": int(study.best_params["max_depth"]),
            "n_estimators": int(study.best_params["n_estimators"]),
            "late_val_logloss": float(study.best_value),
        }

    grid = [(lr, md, ne) for lr in [0.01, 0.02, 0.05, 0.1] for md in [2, 3, 4, 5] for ne in [50, 100, 200, 400]]
    best = None
    for lr, md, ne in grid[:max(8, min(len(grid), n_trials * 3))]:
        ll = score_cfg(lr, md, ne)
        if best is None or ll < best[0]:
            best = (ll, lr, md, ne)
    return {"learning_rate": float(best[1]), "max_depth": int(best[2]), "n_estimators": int(best[3]), "late_val_logloss": float(best[0])}


def fit_xgb_model(X_train, y_train, cfg):
    """Fit the final XGBoost classifier on the full training data using a tuned config."""
    model = XGBClassifier(
        n_estimators=int(cfg["n_estimators"]),
        learning_rate=float(cfg["learning_rate"]),
        max_depth=int(cfg["max_depth"]),
        objective="multi:softprob",
        eval_metric="mlogloss",
        random_state=42,
        n_jobs=-1,
    )
    model.fit(X_train, y_train)
    return model


def make_logreg_pipeline(C: float = 1.0):
    """Build a StandardScaler + multinomial LogisticRegression pipeline (the simple baseline)."""
    return make_pipeline(
        StandardScaler(),
        LogisticRegression(
            C=float(C),
            solver="lbfgs",
            max_iter=2000,
            random_state=42,
        ),
    )


def tune_logreg_hyperparams(X_early, y_early, X_late, y_late):
    """Tune the LogReg regularisation ``C`` (with temperature calibration) by late log loss."""
    print("Tuning Logistic Regression baseline...")
    best = None
    for C in [0.01, 0.03, 0.1, 0.3, 1.0, 3.0, 10.0]:
        model = make_logreg_pipeline(C)
        model.fit(X_early, y_early)
        late_probs_raw = model.predict_proba(X_late)
        T = fit_temperature(late_probs_raw, y_late)
        late_probs = temperature_scale_probs(late_probs_raw, T)
        ll = log_loss(y_late, late_probs)
        if best is None or ll < best["late_val_logloss"]:
            best = {"C": float(C), "temperature": float(T), "late_val_logloss": float(ll)}
    return best


def tune_feature_subset(model_factory, X_early, y_early, X_late, y_late, candidate_feature_sets, *, temperature_scale=False):
    """Pick the feature subset (by column indices) with the best late-validation log loss.

    For each named candidate subset, fits a fresh model (from ``model_factory``) on
    those columns and scores it on late validation. This is where the canonical XGB
    ends up market-only — the market subset simply scores best. Returns
    ``(best_row, all_rows)``.
    """
    best = None
    rows = []
    for name, cols in candidate_feature_sets.items():
        model = model_factory()
        model.fit(X_early[:, cols], y_early)
        late_probs_raw = model.predict_proba(X_late[:, cols])
        if temperature_scale:
            T = fit_temperature(late_probs_raw, y_late)
            late_probs = temperature_scale_probs(late_probs_raw, T)
        else:
            T = 1.0
            late_probs = late_probs_raw
        ll = log_loss(y_late, late_probs)
        row = {
            "name": name,
            "cols": list(cols),
            "late_val_logloss": float(ll),
            "temperature": float(T),
        }
        rows.append(row)
        if best is None or ll < best["late_val_logloss"]:
            best = row
    return best, rows


def tune_mlp_hyperparams(X_early, y_early, X_late, y_late, n_trials=15):
    """Tune MLP architecture/regularisation (with temperature calibration) by late log loss.

    Searches 1-2 hidden layers, layer widths, ``alpha`` and initial learning rate.
    Uses Optuna when available, else a grid fallback. Returns the best config dict.
    """
    print("Tuning MLP Hyperparameters. Please wait...")

    def score_cfg(cfg):
        model = make_mlp_pipeline(cfg)
        model.fit(X_early, y_early)
        late_probs_raw = model.predict_proba(X_late)
        T_mlp = fit_temperature(late_probs_raw, y_late)
        late_probs_cal = temperature_scale_probs(late_probs_raw, T_mlp)
        return log_loss(y_late, late_probs_cal), T_mlp

    if optuna is not None:
        def objective(trial):
            n_layers = trial.suggest_int("n_layers", 1, 2)
            layers = [trial.suggest_categorical(f"n_units_l{i}", [32, 64, 128]) for i in range(n_layers)]
            alpha = trial.suggest_float("alpha", 1e-5, 1e-2, log=True)
            lr_init = trial.suggest_float("learning_rate_init", 1e-4, 5e-3, log=True)
            cfg = {"hidden_layer_sizes": tuple(layers), "alpha": alpha, "learning_rate_init": lr_init}
            ll, _ = score_cfg(cfg)
            return ll

        study = optuna.create_study(direction="minimize")
        study.optimize(objective, n_trials=n_trials)
        best_layers = [study.best_params[f"n_units_l{i}"] for i in range(study.best_params["n_layers"])]
        best_cfg = {
            "hidden_layer_sizes": best_layers,
            "alpha": float(study.best_params["alpha"]),
            "learning_rate_init": float(study.best_params["learning_rate_init"]),
        }
        best_ll, best_T = score_cfg(best_cfg)
        best_cfg["late_val_logloss"] = float(best_ll)
        best_cfg["temperature"] = float(best_T)
        return best_cfg

    grid = []
    for layers in ([32], [64], [128], [64, 32], [128, 64]):
        for alpha in [1e-5, 1e-4, 1e-3]:
            for lr in [1e-4, 5e-4, 1e-3, 2e-3]:
                grid.append({"hidden_layer_sizes": layers, "alpha": alpha, "learning_rate_init": lr})
    best = None
    for cfg in grid[:max(8, min(len(grid), n_trials * 3))]:
        ll, T = score_cfg(cfg)
        if best is None or ll < best[0]:
            best = (ll, T, cfg)
    out = dict(best[2])
    out["late_val_logloss"] = float(best[0])
    out["temperature"] = float(best[1])
    return out


def tune_mlp_feature_subset(X_early, y_early, X_late, y_late, base_cfg, candidate_feature_sets):
    """Select the MLP feature subset with the best late-validation log loss (temperature-scaled)."""
    print("Selecting MLP feature subset on late validation...")
    best = None
    rows = []
    for name, cols in candidate_feature_sets.items():
        model = make_mlp_pipeline(base_cfg)
        model.fit(X_early[:, cols], y_early)
        late_probs_raw = model.predict_proba(X_late[:, cols])
        T_mlp = fit_temperature(late_probs_raw, y_late)
        late_probs = temperature_scale_probs(late_probs_raw, T_mlp)
        ll = log_loss(y_late, late_probs)
        row = {
            "name": name,
            "cols": list(cols),
            "late_val_logloss": float(ll),
            "temperature": float(T_mlp),
        }
        rows.append(row)
        if best is None or ll < best["late_val_logloss"]:
            best = row

    print("\nMLP feature subset scores:")
    for row in sorted(rows, key=lambda r: r["late_val_logloss"]):
        print(f"  {row['name']:<22} {row['late_val_logloss']:.4f}")
    return best, rows


def tune_blend_weights(y_late, probs_base, probs_market, probs_xgb, probs_mlp, step=0.1):
    """Exhaustively grid-search ensemble blend weights over the available models.

    Recursively tries every weight combination (on a ``step`` grid) for the
    non-None probability sources and keeps the one with the lowest late-validation
    log loss. Returns ``{"weights": {...}, "late_val_logloss": ...}``.
    """
    weight_grid = np.arange(0.0, 1.0 + 1e-9, step)
    best = None
    print("Tuning blend weights on late validation. Please wait...")
    candidates = {
        "base": probs_base,
        "market": probs_market,
        "xgb": probs_xgb,
        "mlp": probs_mlp,
    }
    active_names = [name for name, probs in candidates.items() if probs is not None]

    def search(idx, weights):
        nonlocal best
        if idx == len(active_names):
            if sum(weights.values()) == 0:
                return
            full_weights = {name: float(weights.get(name, 0.0)) for name in candidates}
            probs_blend = blend_probabilities(full_weights, candidates)
            ll = log_loss(y_late, probs_blend)
            if best is None or ll < best["late_val_logloss"]:
                best = {"weights": full_weights, "late_val_logloss": float(ll)}
            return

        name = active_names[idx]
        for w in weight_grid:
            weights[name] = float(w)
            search(idx + 1, weights)

    search(0, {})
    return best
