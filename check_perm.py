"""
check_perm.py — is permutation importance signal or noise?

RQ4 returned near-zero Spearman correlation between permutation importance and
every other ranking, including TOPSIS (which never sees the model). That
pattern is what an uninformative ranking looks like. This script checks
directly: if the importances are ~1e-4 with std larger than the mean, the
ordering is noise and must not be reported as disagreement.

Scored two ways:
  f1_macro — the RQ4 setting. On a 96% Benign test set, shuffling a feature
             may barely move macro-F1.
  accuracy — sanity check.
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
from sklearn.inspection import permutation_importance

import config

MODEL = "RandomForest"
TASK = "multiclass"
SEED = 42
N_SUB = 30_000

clf = joblib.load(config.RESULTS_MODELS / f"{MODEL}_{TASK}_seed{SEED}.joblib")
meta = json.loads((config.RESULTS_MODELS / f"split_{TASK}_seed{SEED}.json").read_text())

df = pd.read_parquet(config.DATA_PROCESSED / "unsw_clean.parquet")
names = meta["feature_names"]
X = df[names].loc[meta["test_index"]]
y = df[config.LABEL_MULTI].loc[meta["test_index"]]

sub = X.sample(N_SUB, random_state=SEED)
ys = y.loc[sub.index]
codes = pd.Categorical(ys, categories=meta["class_names"]).codes

for scoring in ("f1_macro", "accuracy"):
    print("\n" + "=" * 62)
    print(f"permutation importance — scoring = {scoring}")
    print("=" * 62)
    r = permutation_importance(
        clf, sub, codes, n_repeats=5, random_state=SEED,
        n_jobs=config.N_JOBS, scoring=scoring,
    )
    order = np.argsort(-r.importances_mean)[:10]
    for i in order:
        m, s = r.importances_mean[i], r.importances_std[i]
        flag = "  <- std > mean" if s > abs(m) else ""
        print(f"  {names[i]:32s} {m:+.6f} +/- {s:.6f}{flag}")

    mx = np.abs(r.importances_mean).max()
    n_noisy = int((r.importances_std > np.abs(r.importances_mean)).sum())
    print(f"\n  max |importance| : {mx:.6f}")
    print(f"  features where std > |mean| : {n_noisy}/{len(names)}")
    if mx < 1e-3:
        print("  VERDICT: importances are negligible — ranking is noise.")
    elif n_noisy > len(names) // 2:
        print("  VERDICT: majority of features are within noise. Unreliable.")
    else:
        print("  VERDICT: ranking carries signal.")
