"""
run_rq4_agreement.py — RQ4: do different explainers agree?

    python run_rq4_agreement.py --model RandomForest --n-instances 300

Builds four global feature rankings for the same frozen model:
  LIME                   aggregated over instances
  TreeSHAP               aggregated over instances (exact)
  Permutation importance model-agnostic, measured on held-out data
  TOPSIS                 data-only, no model involved

then reports the pairwise Spearman matrix. Where two established explainers
disagree about the same model, at least one is misleading an analyst. The
TOPSIS row is the diagnostic: high correlation with TOPSIS and low correlation
with the model-behaviour methods means the explanation is describing the
dataset, not the classifier.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
import warnings

warnings.filterwarnings("ignore", message=".*valid feature names.*")

from pathlib import Path

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "src"))

import joblib
import numpy as np
import pandas as pd
import shap
from lime.lime_tabular import LimeTabularExplainer
from sklearn.inspection import permutation_importance

import config
from explainers import get_shap_explainer
from agreement import (agreement_matrix, jaccard_matrix,
                       importance_noise_report, topsis_feature_ranking)

BACKGROUND_N = 20_000
PERM_SUBSET = 30_000        # permutation importance is O(features x repeats)
LIME_SEED = 101


def main(model_name: str, task: str, seed: int, n_instances: int,
         num_samples: int):
    t_all = time.time()

    clf = joblib.load(config.RESULTS_MODELS / f"{model_name}_{task}_seed{seed}.joblib")
    meta = json.loads((config.RESULTS_MODELS / f"split_{task}_seed{seed}.json").read_text())

    df = pd.read_parquet(config.DATA_PROCESSED / "unsw_clean.parquet")
    feature_names = meta["feature_names"]
    class_names = [str(c) for c in meta["class_names"]]
    n_features = len(feature_names)

    X = df[feature_names]
    y_raw = df[config.LABEL_BINARY if task == "binary" else config.LABEL_MULTI]
    test_idx = meta["test_index"]
    X_te, y_te = X.loc[test_idx], y_raw.loc[test_idx]
    X_tr = X.drop(index=test_idx)

    print(f"model={model_name}  task={task}  features={n_features}")

    per_class = max(1, n_instances // y_te.nunique())
    picks = [g.sample(min(per_class, len(g)), random_state=seed)
             for _, g in y_te.groupby(y_te)]
    sample_idx = pd.concat(picks).index
    print(f"sampled {len(sample_idx)} instances")

    bg = X_tr.sample(min(BACKGROUND_N, len(X_tr)), random_state=seed)

    # ---- LIME: mean |weight| across instances -----------------------------
    print("LIME...", flush=True)
    expl = LimeTabularExplainer(
        bg.values, feature_names=feature_names, class_names=class_names,
        discretize_continuous=True, random_state=LIME_SEED,
        mode="classification",
    )
    lime_imp = np.zeros(n_features)
    for idx in sample_idx:
        x = X_te.loc[idx].values.astype(float)
        label = int(clf.predict(x.reshape(1, -1))[0])
        e = expl.explain_instance(x, clf.predict_proba,
                                  num_features=n_features,
                                  num_samples=num_samples, labels=(label,))
        for f, w in e.as_map()[label]:
            lime_imp[int(f)] += abs(w)
    lime_imp /= len(sample_idx)

    # ---- TreeSHAP: mean |shap value| --------------------------------------
    tree_expl, shap_kind = get_shap_explainer(clf, bg.values)
    print(f"{shap_kind}...", flush=True)
    Xs = X_te.loc[sample_idx]
    if shap_kind == "LinearSHAP":
        sv = tree_expl.shap_values(clf.steps[0][1].transform(Xs))
    else:
        sv = tree_expl.shap_values(Xs)
    arr = np.array(sv)
    if arr.ndim == 3:
        shap_imp = np.abs(arr).mean(axis=(0, 2)) if arr.shape[0] == len(Xs) \
                   else np.abs(arr).mean(axis=(0, 1))
    else:
        shap_imp = np.abs(arr).mean(axis=0)
    shap_imp = np.asarray(shap_imp).ravel()[:n_features]

    # ---- Permutation importance -------------------------------------------
    print("permutation importance...", flush=True)
    sub = X_te.sample(min(PERM_SUBSET, len(X_te)), random_state=seed)
    ysub = y_te.loc[sub.index]
    codes = pd.Categorical(ysub, categories=class_names).codes \
            if task == "multiclass" else ysub.values
    perm = permutation_importance(
        clf, sub, codes, n_repeats=5, random_state=seed,
        n_jobs=config.N_JOBS, scoring="f1_macro",
    )
    perm_imp = perm.importances_mean
    perm_std = perm.importances_std

    # ---- TOPSIS (no model) -------------------------------------------------
    print("TOPSIS...", flush=True)
    topsis_order = topsis_feature_ranking(X_tr.sample(100_000, random_state=seed))

    rankings = {
        "LIME": list(np.argsort(-lime_imp)),
        shap_kind: list(np.argsort(-shap_imp)),
        "Permutation": list(np.argsort(-perm_imp)),
        "TOPSIS": list(topsis_order),
    }

    M = agreement_matrix(rankings, n_features)
    out = config.RESULTS_TABLES / f"rq4_agreement_{model_name}_{task}.csv"
    M.to_csv(out)

    print("\n" + "=" * 60)
    print("RQ4a — pairwise Spearman rho over the FULL ranking")
    print("=" * 60)
    print(M.to_string())

    # ---- top-k overlap ----------------------------------------------------
    # Spearman over 39 features is dominated by the tail. These answer the
    # question an analyst actually has: do the methods agree on what matters?
    for k in (5, 10):
        J = jaccard_matrix(rankings, k)
        j_out = config.RESULTS_TABLES / f"rq4_jaccard{k}_{model_name}_{task}.csv"
        J.to_csv(j_out)
        print(f"\n{'=' * 60}")
        print(f"RQ4b — pairwise Jaccard@{k} (top-{k} set overlap)")
        print("=" * 60)
        print(J.to_string())

    # ---- how much of each ranking is above noise? -------------------------
    noise = importance_noise_report(
        {"LIME": lime_imp, shap_kind: shap_imp, "Permutation": perm_imp},
        {"Permutation": perm_std},
        feature_names,
    )
    n_out = config.RESULTS_TABLES / f"rq4_noise_{model_name}_{task}.csv"
    noise.to_csv(n_out, index=False)
    print("\n" + "=" * 60)
    print("RQ4c — how much of each ranking exceeds its own noise floor")
    print("=" * 60)
    print(noise.to_string(index=False))
    print("\nNote: where a majority of features sit within noise, their")
    print("relative order is arbitrary under every method, and Spearman rho")
    print("over the full ordering understates real agreement at the head.")

    # top-10 side by side, for the results chapter
    top = pd.DataFrame({
        k: [feature_names[i] for i in v[:10]] for k, v in rankings.items()
    }, index=[f"#{i+1}" for i in range(10)])
    t_out = config.RESULTS_TABLES / f"rq4_top10_{model_name}_{task}.csv"
    top.to_csv(t_out)
    print("\nTop-10 features by each method:")
    print(top.to_string())

    print(f"\nwall clock {(time.time()-t_all)/60:.1f} min")
    print(f"wrote {out}\nwrote {t_out}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", dest="model_name", default="RandomForest")
    ap.add_argument("--task", choices=["binary", "multiclass"],
                    default="multiclass")
    ap.add_argument("--seed", type=int, default=config.SEED)
    ap.add_argument("--n-instances", type=int, default=300)
    ap.add_argument("--num-samples", type=int, default=5000)
    main(**vars(ap.parse_args()))
