"""
check_granularity.py — the last mechanism candidate for RQ1.

Three hypotheses have already been falsified against six models:

  surface smoothness  — LogisticRegression (globally linear) sits mid-table
  model accuracy      — GaussianNB (F1 0.121) and LogisticRegression (0.329)
                        have near-identical stability
  model family        — trees occupy both the best and worst positions

Remaining candidate, stated BEFORE looking: LIME fits a local linear surrogate
by regressing perturbed inputs against predicted probabilities. If the model
emits only a few distinct probability values in that neighbourhood, the
regression has little to fit and the resulting weights are close to arbitrary.

PREDICTION: the count of distinct predicted probabilities should rise in the
same order as Jaccard@5, i.e. DecisionTree lowest and RandomForest highest.

If it does not, stop hunting. With six models, any further variable that
happens to fit is curve-fitting, and the honest result is that LIME stability
varies by model in ways no simple property predicts.
"""

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "src"))

import joblib
import numpy as np
import pandas as pd
from scipy.stats import spearmanr

import config

# Jaccard@5 measured in RQ1, for the correlation at the end
JACCARD5 = {
    "DecisionTree": 0.3373,
    "GaussianNB": 0.5443,
    "LogisticRegression": 0.5505,
    "XGBoost": 0.6381,
    "MLP": 0.8080,
    "RandomForest": 0.8684,
}

N_SAMPLE = 20_000
TASK = "multiclass"
SEED = 42

meta = json.loads(
    (config.RESULTS_MODELS / f"split_{TASK}_seed{SEED}.json").read_text()
)
df = pd.read_parquet(config.DATA_PROCESSED / "unsw_clean.parquet")
X = df[meta["feature_names"]].loc[meta["test_index"]].sample(
    N_SAMPLE, random_state=SEED
)

rows = []
for m in JACCARD5:
    path = config.RESULTS_MODELS / f"{m}_{TASK}_seed{SEED}.joblib"
    if not path.exists():
        print(f"skip {m}: no model file")
        continue
    clf = joblib.load(path)
    P = clf.predict_proba(X)
    top = P.max(axis=1)

    rows.append({
        "model": m,
        "jaccard@5": JACCARD5[m],
        # distinct values the model can emit — the regression target's resolution
        "unique_top_proba": int(len(np.unique(np.round(top, 4)))),
        "unique_all_proba": int(len(np.unique(np.round(P, 4)))),
        "mean_confidence": round(float(top.mean()), 4),
        "std_confidence": round(float(top.std()), 4),
    })

t = pd.DataFrame(rows).sort_values("jaccard@5")
out = config.RESULTS_TABLES / "rq1_granularity.csv"
t.to_csv(out, index=False)

print("=" * 82)
print("Probability granularity vs LIME stability (sorted by Jaccard@5)")
print("=" * 82)
print(t.to_string(index=False))

if len(t) >= 4:
    for col in ("unique_top_proba", "unique_all_proba", "mean_confidence",
                "std_confidence"):
        rho, p = spearmanr(t["jaccard@5"], t[col])
        verdict = "SUPPORTS" if (rho > 0.7 and p < 0.10) else "does not support"
        print(f"\nSpearman(jaccard@5, {col}) = {rho:+.3f}  p={p:.3f}   -> {verdict}")

    print("\nWith n=6 models, treat any single correlation as suggestive only.")
    print("If nothing clears rho>0.7, report the variation itself as the")
    print("finding: LIME stability is strongly model-dependent and not")
    print("predicted by smoothness, accuracy, or model family.")

print(f"\nwrote {out}")
