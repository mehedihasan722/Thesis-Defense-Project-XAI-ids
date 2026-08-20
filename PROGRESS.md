# Thesis project state — XAI reliability in ensemble IDS

Mehedi Hasan (C213061) · Sazzadul Islam (C213066R)
Supervisor: Mr. Md. Mahiuddin · CSE, IIUC

Last updated: 20 August 2026

---

## Environment

- Windows 11, i3-10100 (4c/8t), 16 GB RAM, ~7 GB free
- Project at `E:\Thesis\Project\thesis-xai-ids`
- Python **3.11.9** in `.venv` (system Python is 3.14.5 — incompatible with pinned wheels)
- Versions pinned in `requirements.txt`: numpy 1.26.4, pandas 2.2.2,
  scikit-learn 1.4.2, shap 0.45.1, lime 0.2.0.1, xgboost 2.0.3
- `config.py`: `SEED = 42`, `N_JOBS = 4`, `TEST_SIZE = 0.2`
- VS Code: `source.organizeImports` set to `never` — it reorders imports above
  the `sys.path.insert` calls and breaks every script

## Dataset

`NF-UNSW-NB15-V2.parquet` from Kaggle `dhoogla/nfunswnb15v2`, licence CC-BY-NC-SA-4.0.

| | value | note |
|---|---|---|
| flows | **1,986,745** | NOT 2.39M — dhoogla release is deduplicated |
| attack rate | **3.78%** | NOT 3.98% — same reason |
| raw columns | 43 | IPs/timestamps already stripped upstream |
| after ID drop | **39 features** | dropped `l4_src_port`, `l4_dst_port` |
| classes | 10 | Benign + 9 attack families |

Test-set support (20% split): Benign 382,333 · Exploits 5,981 · Fuzzers 4,129 ·
Reconnaissance 2,234 · Generic 1,198 · DoS 835 · Shellcode 285 · Backdoor 167 ·
Analysis 154 · **Worms 33**

IDS2018 **not downloaded** — RQ3 not started, and likely out of scope.

---

## Decisions made (with reasons)

**SVM excluded from the full-scale ensemble.** Measured RBF-SVM training time:
0.1s (5K), 0.2s (10K), 0.7s (20K), 2.6s (40K), 8.5s (80K), **196.0s (160K)**.
Growth exponent exceeds 4 beyond 80K as the kernel cache overflows. Projected
training time at 1.59M rows is infeasible. Report the scaling curve as
justification. Ensemble = DT + RF + XGBoost.

**Soft voting computed manually** (mean of `predict_proba`) rather than
sklearn's `VotingClassifier`, which refits every base estimator. Identical
mathematics, half the compute.

**Trees constrained**: `max_depth=30`, `min_samples_leaf=5`. Caps memory on
2M rows. Goes in Threats to Validity as an untuned-hyperparameter limitation.

**`class_weight="balanced"`** on all tree models. Trades precision for recall
on rare classes — visible in the per-class table. Stage 3 ablation not yet run.

---

## Results so far

### Stage 2 — detection baselines (seed 42)

Binary (macro-F1): DT 0.9742 · RF 0.9718 · **XGB 0.9799** · Ensemble 0.9751
Multiclass (macro-F1): DT 0.6389 · RF 0.6629 · **XGB 0.6731** · Ensemble 0.6687

Two things to report honestly:
- **XGBoost alone beats the soft-voting ensemble** on both tasks.
- Multiclass macro-F1 collapses ~30 points vs binary. Dragged down by
  Analysis (F1 0.087) and Backdoor (F1 0.185). Failure is *precision*, not
  recall — Worms recall 0.939 but precision 0.574.

### RQ1 — LIME stability (933 instances, 10 runs each, 9,330 explanations/model)

| model | Jaccard@5 | Jaccard@10 | Kendall τ | s/expl |
|---|---|---|---|---|
| DecisionTree | **0.337** | 0.285 | **0.158** | 0.090 |
| XGBoost | 0.638 | 0.468 | 0.545 | 0.155 |
| RandomForest | 0.868 | 0.802 | 0.653 | 0.102 |

**LIME is least stable on the simplest, intrinsically interpretable model.**
Proposed mechanism: a DT's `predict_proba` is piecewise-constant with sharp
axis-aligned boundaries, so LIME's local linear surrogate has almost no
gradient to fit and the weights are near-arbitrary. RF averages 100 trees into
a smooth surface, giving the surrogate real signal.

Benign has the worst Kendall τ in all three models. The Jaccard/Kendall
*inversion* (high top-k overlap, low rank correlation) is **RandomForest-specific** —
do not generalise it.

Timing note: RF measured on an idle machine (0.102s). DT and XGB were measured
under concurrent load — state this or re-time.

### RQ2 — faithfulness (483 instances, frozen model, mean-imputation masking)

Comprehensiveness AUC, LIME vs Random control:

| model | LIME | TreeSHAP | Random | LIME ratio | Wilcoxon p |
|---|---|---|---|---|---|
| DecisionTree | 0.8015 | 0.7687 | 0.3497 | **2.29×** | 2.2e-64 |
| XGBoost | 0.5402 | 0.5474 | 0.3123 | 1.73× | 3.5e-47 |
| RandomForest | 0.5521 | 0.4738 | 0.3173 | 1.74× | 3.1e-76 |

**The dissociation is the headline: the least stable explanations are the most
faithful.** DT has the worst RQ1 numbers and the best RQ2 ratio. Consistent
with the mechanism above — many redundant feature subsets are each individually
sufficient to break the prediction, so different runs find different valid ones.

Conclusion to argue: stability and faithfulness measure different things and
can move in opposite directions. Validating an explainer on one alone is
insufficient. No IDS paper measures either.

Secondary findings:
- **Sufficiency is uninformative.** All explainers retain only 12–36% of the
  original probability from the top-10 of 39 features, and on XGBoost Random
  (0.364) beats LIME (0.315) with std 0.97 > mean. Report with the std.
- LIME > TreeSHAP on RF (1.74 vs 1.49) but they converge on XGB (1.73 vs 1.75).
  Likely because LIME's surrogate is fitted in the same perturbation regime the
  masking test uses. Do not conclude TreeSHAP is "worse".

**Known gap:** RQ2 used a single LIME seed, but RQ1 shows the ranking is a
random variable. LIME's faithfulness is therefore itself stochastic and
unmeasured. Multi-seed variant not yet run.

---

### RQ4 — explainer agreement (300 instances, global rankings)

Pairwise Spearman rho over the full 39-feature ranking:

| pair | RandomForest | DecisionTree | XGBoost |
|---|---|---|---|
| LIME ↔ TreeSHAP | **0.676** | 0.277 | 0.481 |
| TreeSHAP ↔ Permutation | 0.017 | **0.896** | 0.578 |
| LIME ↔ Permutation | −0.057 | 0.163 | **0.563** |
| any ↔ TOPSIS | 0.015–0.267 | −0.045–0.145 | 0.074–0.160 |

Jaccard@5 (top-5 set overlap), same pairs: RF 0.25 / 0.25 / 0.00 ·
DT 0.25 / **0.667** / 0.111 · XGB 0.25 / 0.429 / **0.667**

**Headline: which explainers agree depends on the model, not on the
explainers.** A different pair is closest in each of the three models and the
ordering reshuffles completely. A practitioner cannot read "LIME and SHAP
broadly concur" from one paper and carry it to their own system.

Supporting findings:

- **LIME is the outlier on DecisionTree.** TreeSHAP and Permutation — two
  methodologically independent methods — converge at rho 0.896 / Jaccard@5
  0.667, while LIME sits at 0.277 / 0.163. This is exactly the model where RQ1
  found LIME least stable (tau 0.158). LIME's unique top-10 picks there include
  `dns_query_id`, `dns_query_type`, `dns_ttl_answer` — a query nonce has no
  intrinsic security meaning, which is what a surrogate fitting noise looks
  like.
- **Unresolved tension:** RQ2 rates LIME as MORE faithful than TreeSHAP on
  DecisionTree (2.29x vs 2.20x), yet RQ4 shows it disagreeing with everything.
  Candidate explanation: masking a feature that happens to route this instance
  through the tree breaks the prediction locally even if the model does not
  rely on it generally — local faithfulness without global validity. State as
  an open question; do not resolve more confidently than the data allows.
- **`min_ttl` is ranked #1 by LIME, TreeSHAP and Permutation on both
  DecisionTree and XGBoost.** Where the methods converge, they converge on TTL.
  Hernandez et al. also found TTL dominant in their SHAP rankings — name this
  point of contact in the discussion.
- **TOPSIS correlates weakly with everything** (max 0.267, sometimes negative).
  The reassuring result: explainers are tracking model behaviour, not merely
  dataset structure.

**Metric-choice finding, common to RQ1 and RQ4.** In RQ1, Jaccard@5 and
Kendall tau ranked the classes differently. In RQ4, Spearman and Jaccard@5 give
different pictures of the same pair (LIME↔TreeSHAP on RF: rho 0.676 but only
2 of 5 shared). Permutation importance has 18/39 features within its own noise
floor, so rank correlation over a full ordering is partly correlating noise.
Argue: agreement and stability are not single quantities, and the metric
determines the conclusion. Report both, always.

Diagnostic run (`check_perm.py`): permutation importance under `f1_macro` has
max importance 0.085 and 18/39 within noise — usable. Under `accuracy` it is
0.002 with 22/39 within noise — unusable, because shuffling barely moves a
metric dominated by 96% Benign. `f1_macro` was the correct choice.

---

## Next steps

**Experiments are sufficient for a thesis. Writing is now the binding
constraint.** RQ1, RQ2 and RQ4 are complete on three models. Do not start new
experiments until a draft exists.

Priority order if time is short:

1. **Write.** Methodology and three results chapters. Every number needed is in
   `results/tables/`.
2. **Figures.** RQ1 box plots by class; RQ2 comprehensiveness curves with the
   random baseline; RQ4 heatmaps. Three plotting scripts, ~1 hour.
3. **Threats to Validity.** Candidates listed below.

Optional, only if the draft is done:

- Multi-seed RQ2 (closes the known gap: RQ2 used one LIME seed, but RQ1 proves
  the ranking is stochastic).
- Stage 3 class-imbalance ablation (SMOTE vs class_weight vs undersampling vs
  none).
- Extra seeds for Stage 2 → mean ± std (defect #4).
- **RQ3 is out of scope** unless the deadline moves. Needs a 3 GB download plus
  a full retrain cycle. Scope it as future work with the design specified —
  that is a legitimate, documentable limitation.

## Writing order (roadmap Section 08)

Methodology → Results (one chapter per RQ) → Related Work → Discussion →
Threats to Validity → Conclusion → Introduction → Abstract.

Related Work must name Patil et al. 2022, Kalutharage et al. 2023, and
Hernandez et al. 2025 explicitly, stating what each did and what differs here.
Omitting them is the largest examination risk.

Threats to Validity candidates: benchmark is synthetic; single seed for the
split; mean-imputation masking creates off-manifold inputs; hyperparameters
untuned per model; TOPSIS criteria chosen by us and not canonical; Worms
metrics rest on n=33.

## Files

```
config.py  verify.py  run_baselines.py  run_rq1_stability.py
run_rq2_faithfulness.py  run_rq4_agreement.py
src/  data_loader.py  models.py  evaluation.py  stability.py
      faithfulness.py  agreement.py
results/tables/   all CSVs — tracked in git
results/models/   .joblib + split_*.json — gitignored, regenerable
```

`split_{task}_seed{seed}.json` holds the exact test indices. RQ1/RQ2/RQ4 all
load it so every experiment uses the identical split and the identical frozen
model. Do not re-run Stage 2 with a different seed mid-stream.
