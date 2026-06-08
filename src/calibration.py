"""Probability calibration: logit transform, softmax, and temperature scaling.

Temperature scaling is the project's calibration method: :func:`fit_temperature`
finds the single ``T`` that minimises validation NLL, and :func:`temperature_scale_probs`
applies it (T>1 softens an overconfident model, T<1 sharpens an underconfident one).
:func:`safe_logit` is the feature transform used when packing probabilities into the
meta-feature matrix.
"""
import numpy as np
from typing import Optional

def safe_logit(p: np.ndarray, eps: float = 1e-12) -> np.ndarray:
    """
    One-vs-rest logit transformation: log(p / (1 - p)).

    This is useful as a feature transform for individual class
    probabilities. It is not used for multiclass temperature scaling,
    because softmax temperature scaling operates on log probabilities
    when only probabilities, not raw model logits, are available.
    """
    p = np.clip(p, eps, 1.0 - eps)
    return np.log(p) - np.log(1.0 - p)

def softmax(z: np.ndarray) -> np.ndarray:
    z = z - np.max(z, axis=1, keepdims=True)
    e = np.exp(z)
    return e / np.sum(e, axis=1, keepdims=True)

def temperature_scale_probs(probs: np.ndarray, T: float) -> np.ndarray:
    """
    Calibrate multiclass probability vectors with temperature scaling.

    We only have probabilities here, not raw model logits. For a softmax
    probability vector, log(p) is equivalent to the logits up to an additive
    constant, so softmax(log(p) / T) gives standard temperature behavior:

    - T = 1 keeps the original probabilities unchanged.
    - T > 1 softens the distribution.
    - T < 1 sharpens the distribution.
    """
    probs = np.asarray(probs, dtype=float)
    probs = np.nan_to_num(probs, nan=1e-12, posinf=1.0, neginf=1e-12)
    probs = np.clip(probs, 1e-12, 1.0)
    probs = probs / probs.sum(axis=1, keepdims=True)
    logits = np.log(probs)
    scaled = logits / max(T, 1e-6)
    return softmax(scaled)

def _nll(T_arr: np.ndarray, probs_val: np.ndarray, y_val: np.ndarray) -> float:
    """Negative log-likelihood as a scalar function of T (for scipy minimize)."""
    T = float(T_arr[0])
    p = temperature_scale_probs(probs_val, T)
    return float(-np.mean(np.log(np.clip(p[np.arange(len(y_val)), y_val], 1e-12, 1.0))))


def fit_temperature(
    probs_val: np.ndarray,
    y_val: np.ndarray,
    T_grid: Optional[np.ndarray] = None,
    T_min: float = 0.1,
    T_max: float = 10.0,
) -> float:
    """
    Finds the temperature T in [T_min, T_max] that minimises NLL on validation.

    Strategy (two-stage):
      1. Coarse grid search over [T_min, T_max] to find a good starting point
         and detect whether the optimum sits near a boundary.
      2. scipy.optimize.minimize_scalar refines the answer to high precision
         within the same bounds.

    A warning is printed if the grid-search optimum lands at either boundary,
    which would indicate the model is extremely over- or under-confident and
    the T range may need extending.

    T > 1 -> softens probabilities (model is overconfident)
    T < 1 -> sharpens probabilities (model is underconfident)
    T = 1 -> no change (identity)
    """
    from scipy.optimize import minimize_scalar

    if T_grid is None:
        # Finer grid, wider range; safe for proper logit-space scaling
        T_grid = np.concatenate([
            np.arange(T_min, 1.0,  0.05),
            np.arange(1.0,  T_max + 0.01, 0.1),
        ])

    best_T = 1.0
    best_nll = float("inf")
    for T in T_grid:
        nll = _nll(np.array([T]), probs_val, y_val)
        if nll < best_nll:
            best_nll = nll
            best_T = float(T)

    # Boundary warning — signals the search range is too narrow
    if best_T <= T_grid[1]:
        print(
            f"[calibration WARNING] Optimal T={best_T:.2f} is at the LOWER boundary "
            f"(T_min={T_min}). Model may be underconfident. Consider lowering T_min."
        )
    if best_T >= T_grid[-2]:
        print(
            f"[calibration WARNING] Optimal T={best_T:.2f} is at the UPPER boundary "
            f"(T_max={T_max}). Model is very overconfident. Consider raising T_max."
        )

    # Refine with scipy around the coarse best
    result = minimize_scalar(
        lambda t: _nll(np.array([t]), probs_val, y_val),
        bounds=(T_min, T_max),
        method="bounded",
        options={"xatol": 1e-4},
    )
    if result.success and result.fun < best_nll:
        best_T = float(result.x)

    return best_T
