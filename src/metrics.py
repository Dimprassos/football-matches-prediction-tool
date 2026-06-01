import numpy as np


CLASS_PREFIXES = ("home", "draw", "away")


def multiclass_brier(probs: np.ndarray, y: np.ndarray) -> float:
    """
    Multiclass Brier score for 3-class problem.
    probs: shape (N, 3)
    y: shape (N,)
    """
    y_onehot = np.zeros_like(probs)
    y_onehot[np.arange(len(y)), y] = 1.0
    return float(np.mean(np.sum((probs - y_onehot) ** 2, axis=1)))


def top_label_ece(probs: np.ndarray, y: np.ndarray, n_bins: int = 10) -> float:
    """
    Top-label Expected Calibration Error.
    Compares confidence of predicted class vs empirical accuracy.
    """
    conf = np.max(probs, axis=1)
    pred = np.argmax(probs, axis=1)
    acc = (pred == y).astype(float)

    bins = np.linspace(0, 1, n_bins + 1)
    ece = 0.0
    N = len(y)

    for i in range(n_bins):
        lo = bins[i]
        hi = bins[i + 1]

        if i < n_bins - 1:
            mask = (conf >= lo) & (conf < hi)
        else:
            mask = (conf >= lo) & (conf <= hi)

        n = int(np.sum(mask))
        if n > 0:
            ece += (n / N) * abs(np.mean(acc[mask]) - np.mean(conf[mask]))

    return float(ece)


def class_metric_summary(probs: np.ndarray, y: np.ndarray) -> dict[str, float]:
    pred = np.argmax(probs, axis=1)
    out: dict[str, float] = {}
    f1_values = []
    for cls, prefix in enumerate(CLASS_PREFIXES):
        true_pos = float(np.sum((pred == cls) & (y == cls)))
        pred_pos = float(np.sum(pred == cls))
        actual_pos = float(np.sum(y == cls))
        precision = true_pos / pred_pos if pred_pos > 0 else 0.0
        recall = true_pos / actual_pos if actual_pos > 0 else 0.0
        f1 = (2.0 * precision * recall / (precision + recall)) if precision + recall > 0 else 0.0
        out[f"{prefix}_precision"] = float(precision)
        out[f"{prefix}_recall"] = float(recall)
        out[f"{prefix}_f1"] = float(f1)
        f1_values.append(f1)
    out["macro_f1"] = float(np.mean(f1_values))
    return out
