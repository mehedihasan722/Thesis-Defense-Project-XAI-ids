"""
agreement.py — RQ4 metrics, plus TOPSIS.

TOPSIS is not an explainability method. It ranks features from the DATA
alone — variance, correlation, entropy — with no reference to the model. That
is exactly what makes it useful here: if a supposedly model-explaining method
correlates more strongly with TOPSIS than with the model's measured
behaviour, it is describing the dataset rather than the classifier.
"""

from __future__ import annotations

import itertools
import numpy as np
import pandas as pd
from scipy.stats import spearmanr, entropy


def topsis_feature_ranking(X: pd.DataFrame) -> np.ndarray:
    """Multi-criteria feature ranking from three data-intrinsic criteria.

    Criteria (all benefit-type — higher is better):
      variance    — a constant feature carries no information
      entropy     — spread of the value distribution
      uniqueness  — mean absolute correlation with other features, INVERTED,
                    so a feature redundant with many others scores lower

    Returns feature indices ordered best-first.
    """
    Xv = X.values.astype(float)
    n_feat = Xv.shape[1]

    var = Xv.var(axis=0)

    ent = np.zeros(n_feat)
    for j in range(n_feat):
        hist, _ = np.histogram(Xv[:, j], bins=20)
        p = hist / hist.sum() if hist.sum() else np.ones(20) / 20
        ent[j] = entropy(p + 1e-12)

    corr = np.abs(np.corrcoef(Xv, rowvar=False))
    np.fill_diagonal(corr, 0.0)
    corr = np.nan_to_num(corr)
    uniqueness = 1.0 - corr.mean(axis=1)

    M = np.vstack([var, ent, uniqueness]).T          # (features, criteria)

    # vector normalisation, then equal weights
    norms = np.sqrt((M ** 2).sum(axis=0))
    norms[norms == 0] = 1.0
    N = M / norms
    W = N * (1.0 / M.shape[1])

    ideal_best = W.max(axis=0)
    ideal_worst = W.min(axis=0)

    d_best = np.sqrt(((W - ideal_best) ** 2).sum(axis=1))
    d_worst = np.sqrt(((W - ideal_worst) ** 2).sum(axis=1))

    closeness = d_worst / (d_best + d_worst + 1e-12)
    return np.argsort(-closeness)


def ranks_from_order(order, n_features) -> np.ndarray:
    """Convert an ordered index list into a position vector."""
    r = np.empty(n_features, dtype=float)
    for pos, feat in enumerate(order):
        r[feat] = pos
    return r


def agreement_matrix(rankings: dict, n_features: int) -> pd.DataFrame:
    """Pairwise Spearman rho between every pair of named rankings."""
    names = list(rankings)
    M = pd.DataFrame(np.eye(len(names)), index=names, columns=names)
    for a, b in itertools.combinations(names, 2):
        ra = ranks_from_order(rankings[a], n_features)
        rb = ranks_from_order(rankings[b], n_features)
        rho, _ = spearmanr(ra, rb)
        M.loc[a, b] = M.loc[b, a] = round(float(rho), 4)
    return M


def jaccard_matrix(rankings: dict, k: int) -> pd.DataFrame:
    """Pairwise Jaccard overlap of the top-k features.

    Reported alongside the Spearman matrix because rank correlation over a
    full ordering is dominated by the tail. When a majority of features carry
    importance indistinguishable from zero, their relative order is arbitrary
    under every method, and rho collapses toward zero even where the methods
    agree on what actually matters.
    """
    names = list(rankings)
    M = pd.DataFrame(np.eye(len(names)), index=names, columns=names)
    for a, b in itertools.combinations(names, 2):
        sa, sb = set(rankings[a][:k]), set(rankings[b][:k])
        j = len(sa & sb) / len(sa | sb) if (sa | sb) else 1.0
        M.loc[a, b] = M.loc[b, a] = round(j, 4)
    return M


def importance_noise_report(importances: dict, stds: dict,
                            feature_names: list) -> pd.DataFrame:
    """How much of each ranking is distinguishable from zero?

    A method whose tail is noise cannot be meaningfully rank-correlated over
    its full ordering. This table is the justification for reporting top-k
    overlap as well as Spearman rho.
    """
    rows = []
    for name, imp in importances.items():
        sd = stds.get(name)
        if sd is None:
            rows.append({"method": name, "n_features": len(imp),
                         "max_abs": float(np.abs(imp).max()),
                         "n_within_noise": None,
                         "pct_within_noise": None})
        else:
            noisy = int((sd > np.abs(imp)).sum())
            rows.append({"method": name, "n_features": len(imp),
                         "max_abs": round(float(np.abs(imp).max()), 6),
                         "n_within_noise": noisy,
                         "pct_within_noise": round(100 * noisy / len(imp), 1)})
    return pd.DataFrame(rows)
