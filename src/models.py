"""
models.py — detection layer.

Two things worth knowing:

1. SVM is wrapped in a Pipeline with StandardScaler. The scaler is fit inside
   the pipeline, so it sees training data only — scaling before the split
   would leak test statistics into training. Tree models need no scaling and
   are passed raw.

2. Soft voting is computed manually rather than with VotingClassifier, which
   refits every base estimator from scratch. Averaging predict_proba is the
   same mathematics at zero extra cost.
"""

from __future__ import annotations

import numpy as np
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from xgboost import XGBClassifier

import config


def build_models(task: str, n_classes: int, seed: int = config.SEED,
                 include_svm: bool = True) -> dict:
    common_tree = dict(random_state=seed, class_weight="balanced")

    models = {
        "DecisionTree": DecisionTreeClassifier(
            min_samples_leaf=5, max_depth=30, **common_tree,
        ),
        "RandomForest": RandomForestClassifier(
            n_estimators=100, min_samples_leaf=5, max_depth=30,
            n_jobs=config.N_JOBS, **common_tree,
        ),
        "XGBoost": XGBClassifier(
            n_estimators=200, max_depth=8, learning_rate=0.1,
            subsample=0.8, colsample_bytree=0.8,
            n_jobs=config.N_JOBS, random_state=seed,
            tree_method="hist",
            objective="binary:logistic" if task == "binary" else "multi:softprob",
            num_class=None if task == "binary" else n_classes,
            eval_metric="logloss",
        ),
    }

    if include_svm:
        models["SVM"] = Pipeline([
            ("scale", StandardScaler()),
            ("svc", SVC(
                kernel="rbf",
                C=1.0,
                gamma="scale",
                class_weight="balanced",
                probability=True,      # required for soft voting; ~5x cost
                cache_size=1000,       # MB; larger = fewer kernel recomputes
                random_state=seed,
            )),
        ])

    return models


def soft_vote(probas: list[np.ndarray]) -> np.ndarray:
    """Mean of predict_proba matrices. All must share column order."""
    return np.mean(np.stack(probas, axis=0), axis=0)
