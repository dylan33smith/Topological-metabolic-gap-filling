from __future__ import annotations

import numpy as np
from sklearn.metrics import average_precision_score, precision_recall_curve, roc_auc_score


def binary_ranking_metrics(y_true: list[int], y_score: list[float], k_values=(10, 50, 100)) -> dict:
    y_true_array = np.asarray(y_true)
    y_score_array = np.asarray(y_score)

    metrics = {
        "average_precision": float(average_precision_score(y_true_array, y_score_array)),
    }
    if len(np.unique(y_true_array)) == 2:
        metrics["auroc"] = float(roc_auc_score(y_true_array, y_score_array))

    for k in k_values:
        if k <= len(y_true_array):
            metrics[f"precision_at_{k}"] = precision_at_k(y_true_array, y_score_array, k)
            metrics[f"recall_at_{k}"] = recall_at_k(y_true_array, y_score_array, k)

    return metrics


def precision_at_k(y_true: np.ndarray, y_score: np.ndarray, k: int) -> float:
    order = np.argsort(y_score)[::-1][:k]
    return float(y_true[order].sum() / k)


def recall_at_k(y_true: np.ndarray, y_score: np.ndarray, k: int) -> float:
    positives = y_true.sum()
    if positives == 0:
        return 0.0
    order = np.argsort(y_score)[::-1][:k]
    return float(y_true[order].sum() / positives)


def pr_curve(y_true: list[int], y_score: list[float]):
    precision, recall, thresholds = precision_recall_curve(y_true, y_score)
    return precision, recall, thresholds
