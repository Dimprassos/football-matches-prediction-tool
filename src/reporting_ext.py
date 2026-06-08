"""Console reports for a model's probability quality and confusion matrix."""
from __future__ import annotations

import numpy as np
from sklearn.metrics import accuracy_score, confusion_matrix, log_loss

from src.metrics import multiclass_brier, top_label_ece


def print_prob_report(name, probs, y_true):
    """Print log loss, Brier, ECE, and accuracy for one model's predictions."""
    preds = np.argmax(probs, axis=1)
    print(f"\n{name}:")
    print("LogLoss:", round(log_loss(y_true, probs), 4))
    print("Brier:", round(multiclass_brier(probs, y_true), 4))
    print("ECE:", round(top_label_ece(probs, y_true), 4))
    print("Accuracy:", round(accuracy_score(y_true, preds), 4))


def print_confusion(name, probs, y_true):
    """Print the [Home, Draw, Away] confusion matrix for one model's predictions."""
    preds = np.argmax(probs, axis=1)
    cm = confusion_matrix(y_true, preds, labels=[0, 1, 2])
    print(f"\n{name} confusion matrix [H,D,A]:")
    print(cm)
