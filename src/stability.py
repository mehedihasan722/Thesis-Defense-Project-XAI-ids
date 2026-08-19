"""
stability.py — RQ1 metrics.

Jaccard@k measures top-k set overlap; Kendall's tau measures agreement over
the complete ranking. Both are needed: two explanations can share the same
top-5 set while ordering it differently, and can agree on ordering while
disagreeing about which features make the cut.
"""

from __future__ import annotations

import itertools
import numpy as np
from scipy.stats import kendalltau


def jaccard_at_k(rankings: list[list[int]], k: int) -> float:
    """Mean pairwise Jaccard similarity of top-k sets across runs.

    1.0 = every run selected the same k features. 0.0 = no overlap at all.
    """
    tops = [set(r[:k]) for r in rankings]
    sims = []
    for a, b in itertools.combinations(tops, 2):
        union = a | b
        sims.append(len(a & b) / len(union) if union else 1.0)
    return float(np.mean(sims)) if sims else 1.0


def mean_kendall_tau(rankings: list[list[int]], n_features: int) -> float:
    """Mean pairwise Kendall's tau over the full feature ordering.

    Rankings are converted to position vectors so that features absent from a
    truncated explanation still contribute.
    """
    pos = []
    for r in rankings:
        p = np.full(n_features, n_features, dtype=float)
        for rank, feat in enumerate(r):
            p[feat] = rank
        pos.append(p)

    taus = []
    for a, b in itertools.combinations(pos, 2):
        t, _ = kendalltau(a, b)
        if not np.isnan(t):
            taus.append(t)
    return float(np.mean(taus)) if taus else 1.0


def rank_stability(rankings: list[list[int]], n_features: int) -> dict:
    return {
        "jaccard_at_5": jaccard_at_k(rankings, 5),
        "jaccard_at_10": jaccard_at_k(rankings, 10),
        "kendall_tau": mean_kendall_tau(rankings, n_features),
        "n_runs": len(rankings),
    }
