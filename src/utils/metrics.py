"""
Evaluation metrics for entity matching.

All functions operate on flat lists/arrays of integer labels (0 = non-match, 1 = match).
"""

from sklearn.metrics import (
    precision_score,
    recall_score,
    f1_score,
    classification_report,
    confusion_matrix,
)


def compute_metrics(labels: list[int], predictions: list[int]) -> dict:
    """Return precision, recall, and F1 for the positive (match) class."""
    return {
        "precision": precision_score(labels, predictions, zero_division=0),
        "recall": recall_score(labels, predictions, zero_division=0),
        "f1": f1_score(labels, predictions, zero_division=0),
    }


def full_report(labels: list[int], predictions: list[int]) -> str:
    """Full sklearn classification report as a string."""
    return classification_report(labels, predictions, target_names=["non-match", "match"])


def confusion(labels: list[int], predictions: list[int]) -> dict:
    """Return TP, FP, FN, TN counts."""
    cm = confusion_matrix(labels, predictions, labels=[0, 1])
    tn, fp, fn, tp = cm.ravel()
    return {"tp": int(tp), "fp": int(fp), "fn": int(fn), "tn": int(tn)}
