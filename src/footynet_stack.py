"""Stacking blend for FootyNet (chunk ST-1).

Learns a convex combination of member probability matrices (e.g. FootyNet + market)
on the *validation* split by minimizing multiclass log loss, then applies the learned
weights to any split. Weights live on the probability simplex (non-negative, sum to
one), so the blend is itself a valid distribution and never extrapolates beyond its
members. Fitting only on validation keeps the held-out test evaluation leakage-free.

The learner is a deterministic simplex grid search — no extra dependency, fully
reproducible, and exact enough for the 2-3 members we combine. It is written for an
arbitrary number of members so a third source (base model / XGBoost meta) can be
added later without touching the call sites.
"""
from __future__ import annotations

import numpy as np

_EPS = 1e-12


def _logloss(probs: np.ndarray, y: np.ndarray) -> float:
    """Multiclass log loss of a (renormalized) probability matrix against integer labels."""
    p = np.clip(np.asarray(probs, dtype=float), _EPS, None)
    p = p / p.sum(axis=1, keepdims=True)
    y = np.asarray(y, dtype=int)
    return float(-np.log(p[np.arange(len(y)), y]).mean())


def apply_blend(members: dict[str, np.ndarray], weights: dict[str, float]) -> np.ndarray:
    """Convex combination of member probability matrices, renormalized to sum to one.

    ``weights`` keys must be a subset of ``members``; only weighted members contribute.
    """
    blended = sum(
        float(weights[name]) * np.asarray(members[name], dtype=float)
        for name in weights
    )
    blended = np.clip(blended, _EPS, None)
    return blended / blended.sum(axis=1, keepdims=True)


def _compositions(total: int, parts: int):
    """Yield every way to write ``total`` as an ordered sum of ``parts`` non-negatives."""
    if parts == 1:
        yield (total,)
        return
    for first in range(total + 1):
        for rest in _compositions(total - first, parts - 1):
            yield (first,) + rest


def learn_blend_weights(
    members: dict[str, np.ndarray],
    y: np.ndarray,
    *,
    step: float = 0.01,
) -> dict[str, float]:
    """Learn simplex weights over ``members`` that minimize log loss on ``(members, y)``.

    Searches the probability simplex at resolution ``step`` (e.g. 0.01 -> grid of
    1/100). Members are weighted in dict-insertion order; the returned dict maps each
    member name to its learned weight (summing to one). Intended to be fit on the
    validation split, then applied to other splits via :func:`apply_blend`.
    """
    names = list(members.keys())
    mats = [np.asarray(members[n], dtype=float) for n in names]
    y = np.asarray(y, dtype=int)
    levels = int(round(1.0 / step))

    best_weights: tuple[float, ...] | None = None
    best_ll = np.inf
    for comp in _compositions(levels, len(names)):
        w = np.asarray(comp, dtype=float) / levels
        blended = sum(wi * m for wi, m in zip(w, mats))
        ll = _logloss(blended, y)
        if ll < best_ll:
            best_ll, best_weights = ll, tuple(float(x) for x in w)

    return {name: best_weights[i] for i, name in enumerate(names)}
