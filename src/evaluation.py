"""
evaluation.py — detection metrics.

Accuracy is computed but never led with. NF-UNSW-NB15-v2 is 96.22% benign,
so an always-benign classifier scores 96.22% accuracy while detecting nothing.
Macro-F1 leads every table.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.metrics import (
    f1_score, recall_score, precision_score, accuracy_score,
    average_precision_score, confusion_matrix, classification_report,
)
from sklearn.preprocessing import label_binarize


def false_alarm_rate(y_true, y_pred, benign_label=0) -> float:
    """Fraction of benign flows wrongly flagged as attack.

    This is the number a SOC analyst actually cares about: at 1.9M benign
    flows, a 1% FAR is 19,000 false alerts.
    """
    benign_mask = (y_true == benign_label)
    if benign_mask.sum() == 0:
        return float("nan")
    return float((y_pred[benign_mask] != benign_label).mean())


def pr_auc(y_true, y_proba, classes) -> float:
    """Macro-averaged average-precision. Robust to imbalance in a way ROC-AUC
    is not — with 3.78% positives, ROC-AUC flatters every model."""
    if len(classes) == 2:
        return float(average_precision_score(y_true, y_proba[:, 1]))
    y_bin = label_binarize(y_true, classes=classes)
    return float(average_precision_score(y_bin, y_proba, average="macro"))


def evaluate(y_true, y_pred, y_proba, classes,
             benign_label=0, name="") -> dict:
    return {
        "model": name,
        "macro_f1": float(f1_score(y_true, y_pred, average="macro", zero_division=0)),
        "weighted_f1": float(f1_score(y_true, y_pred, average="weighted", zero_division=0)),
        "macro_recall": float(recall_score(y_true, y_pred, average="macro", zero_division=0)),
        "macro_precision": float(precision_score(y_true, y_pred, average="macro", zero_division=0)),
        "pr_auc": pr_auc(y_true, y_proba, classes),
        "false_alarm_rate": false_alarm_rate(y_true, y_pred, benign_label),
        "accuracy": float(accuracy_score(y_true, y_pred)),   # reported in passing only
    }


def per_class_recall(y_true, y_pred, class_names) -> pd.DataFrame:
    """Per-class recall with support. Support matters: Worms has 164 flows
    total, so its recall is estimated from ~33 test instances. Any claim about
    Worms must carry that caveat."""
    rep = classification_report(
        y_true, y_pred, target_names=class_names,
        output_dict=True, zero_division=0,
    )
    rows = []
    for cls in class_names:
        if cls in rep:
            rows.append({
                "class": cls,
                "recall": rep[cls]["recall"],
                "precision": rep[cls]["precision"],
                "f1": rep[cls]["f1-score"],
                "support": int(rep[cls]["support"]),
            })
    return pd.DataFrame(rows).sort_values("support", ascending=False)
