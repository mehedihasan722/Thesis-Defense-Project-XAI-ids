# Evaluating the Reliability of Explainable AI in Ensemble-Based Intrusion Detection

**Mehedi Hasan** (C213061) · **Sazzadul Islam** (C213066R)
Supervisor: Mr. Md. Mahiuddin, Associate Professor
Department of Computer Science & Engineering, International Islamic University Chittagong

---

## What this project asks

Explainable AI is routinely applied to intrusion detection, but the explanations
themselves are almost never validated. Papers show a LIME plot and move on.

This project treats **the explanation as the object of evaluation rather than
the output**. Given a trained IDS and a flow it has flagged, it asks four
questions:

| | Question | Metric |
|---|---|---|
| **RQ1** | Does LIME return the same explanation twice? | Jaccard@k, Kendall τ |
| **RQ2** | Do the features it names actually drive the prediction? | Comprehensiveness / sufficiency vs a random control |
| **RQ4** | Do different explainers agree with each other? | Pairwise Spearman ρ, Jaccard@k |
| **RQ3** | Do explanations survive a change of network? | *Not implemented — see Scope* |

The unifying claim: **an analyst cannot trust an explanation that has not been
measured, and the measurement you choose determines the answer you get.**

---

## Headline results

All on NF-UNSW-NB15-v2, multiclass, 1,986,745 flows, 39 features.

**RQ1 — LIME is least stable on the simplest model.**

| model | Jaccard@5 | Kendall τ |
|---|---|---|
| DecisionTree | 0.337 | 0.158 |
| XGBoost | 0.638 | 0.545 |
| RandomForest | 0.868 | 0.653 |

Ten runs of LIME on an *identical* frozen model and an *identical* input share
barely a third of their top-5 features on a Decision Tree. Proposed mechanism:
a tree's `predict_proba` is piecewise-constant with sharp boundaries, so the
local linear surrogate has no gradient to fit. Averaging 100 trees smooths the
surface and gives the surrogate real signal.

**RQ2 — but the least stable explanations are the most faithful.**

Comprehensiveness AUC on a frozen model, mean-imputation masking:

| model | LIME | TreeSHAP | Random | LIME ratio |
|---|---|---|---|---|
| DecisionTree | 0.802 | 0.769 | 0.350 | **2.29×** |
| XGBoost | 0.540 | 0.547 | 0.312 | 1.73× |
| RandomForest | 0.552 | 0.474 | 0.317 | 1.74× |

All highly significant (Wilcoxon p < 1e-46). **Stability and faithfulness move
in opposite directions**, so validating an explainer on either alone is
insufficient.

**RQ4 — which explainers agree depends on the model, not the explainers.**

Spearman ρ, closest pair in bold:

| pair | RandomForest | DecisionTree | XGBoost |
|---|---|---|---|
| LIME ↔ TreeSHAP | **0.676** | 0.277 | 0.481 |
| TreeSHAP ↔ Permutation | 0.017 | **0.896** | 0.578 |
| LIME ↔ Permutation | −0.057 | 0.163 | **0.563** |

A different pair is closest every time. Where they *do* converge, they converge
on `min_ttl` — ranked #1 by three independent methods on two models.

---

## Quick start

Requires **Python 3.11** (3.13+ has no wheels for the pinned versions).

```bash
git clone <repo> && cd thesis-xai-ids
py -3.11 -m venv .venv
.\.venv\Scripts\Activate.ps1          # Windows
pip install -r requirements.txt

# Kaggle credentials, then:
kaggle datasets download -d dhoogla/nfunswnb15v2 -p data/raw/unsw --unzip

python verify.py                       # gate: schema + identifier check
python run_baselines.py --task multiclass --no-svm
python run_rq1_stability.py   --model RandomForest --n-instances 1000
python run_rq2_faithfulness.py --model RandomForest --n-instances 500
python run_rq4_agreement.py    --model RandomForest --n-instances 300
```

Total runtime on an i3-10100 / 16 GB: roughly 90 minutes for the full set.

---

## Repository layout

```
config.py                    seeds, paths, identifier list, N_JOBS
verify.py                    Stage 1 gate — run this first
run_baselines.py             Stage 2 — detection models
run_rq1_stability.py         RQ1 — LIME repeatability
run_rq2_faithfulness.py      RQ2 — masking vs random control
run_rq4_agreement.py         RQ4 — four-way ranking agreement
check_perm.py                diagnostic: is permutation importance signal?

src/
  data_loader.py             single entry point for all data
  models.py                  model definitions, manual soft voting
  evaluation.py              macro-F1, PR-AUC, false alarm rate
  stability.py               Jaccard@k, Kendall τ
  faithfulness.py            comprehensiveness, sufficiency, masking
  agreement.py               Spearman/Jaccard matrices, TOPSIS

results/tables/              all outputs — tracked in git
results/models/              .joblib + split_*.json — gitignored
data/                        parquet — gitignored
PROGRESS.md                  full state, decisions, and open questions
```

---

## Design decisions worth defending

**Identifier columns are dropped, non-negotiably.** A model that memorises an
attacker's IP or port reports inflated accuracy and produces explanations
citing the IP as the most important feature. The dhoogla release already
strips IPs and timestamps; `data_loader.py` additionally drops `l4_src_port`
and `l4_dst_port`, leaving 39 features from 43.

**Macro-F1 leads every table, never accuracy.** The dataset is 96.22% benign,
so an always-benign classifier scores 96.22% accuracy while detecting nothing.
Accuracy is reported in passing only.

**SVM is excluded, with measurements rather than assertion.** RBF-SVM training
time was measured at 0.1s (5K), 0.2s (10K), 0.7s (20K), 2.6s (40K), 8.5s (80K),
**196.0s (160K)**. The growth exponent exceeds 4 beyond 80K as the kernel cache
overflows, projecting an infeasible time at 1.59M training rows. The ensemble
is DT + RF + XGBoost.

**Soft voting is computed manually** as the mean of `predict_proba`, because
sklearn's `VotingClassifier` refits every base estimator. Identical
mathematics, half the compute.

**The model is frozen for RQ2.** Retraining on a top-k subset measures whether
those features *suffice to build a model* — it does not measure whether *this
model used them*, because a retrained model recovers performance from
correlated substitutes. Masking on a frozen model is the only test that
answers the actual question. `split_*.json` stores the exact test indices so
RQ1, RQ2 and RQ4 all operate on the same instances and the same fitted model.

**The random-feature control is the experiment, not a robustness check.**
Masking 10 random features already drops the predicted probability by ~0.32.
Without that baseline, LIME's 0.55 would look impressive in isolation.

---

## Known limitations

Carried into the Threats to Validity chapter:

- **Mean-imputation masking creates off-manifold inputs.** A flow with average
  byte counts but original TCP flags may be physically impossible. Unavoidable
  with this protocol; a second masking strategy would test robustness.
- **RQ2 used a single LIME seed**, but RQ1 proves the ranking is stochastic —
  so LIME's measured faithfulness is itself a random variable, unquantified.
- **Sufficiency is uninformative here.** All explainers retain only 12–36% of
  the original probability from the top-10 of 39, and on XGBoost the random
  control *beats* LIME with a std larger than the mean.
- **Permutation importance has 18/39 features inside its own noise floor**, so
  Spearman ρ over its full ranking is partly correlating noise. This is why
  Jaccard@k is reported alongside.
- **TOPSIS criteria (variance, entropy, inverted mean correlation) are our
  choice**, not canonical; Shuwandy et al. used different ones.
- **Rare-class results rest on tiny samples.** Worms has 33 test flows; a 95%
  interval on its 0.939 recall spans roughly 0.80–0.98.
- **Hyperparameters are not tuned per model** (`max_depth=30`,
  `min_samples_leaf=5` throughout), chosen to cap memory on 2M rows.
- **Benchmark datasets are synthetic**, and a single subsampling seed is used.

---

## Scope

**RQ3 (cross-network drift) is not implemented.** It requires
NF-CSE-CIC-IDS2018-v2 (~3 GB) and a full retrain cycle in both directions.
Specified as future work: train on one infrastructure, test on the other,
then quantify importance drift by Spearman ρ, Kendall τ and Jaccard@10 *per
attack class* — which is the quantitative step Hernandez et al. (2025) did not
take, having compared two rankings qualitatively on merged background traffic.

---

## Reproducibility

Fixed seed 42 throughout; LIME seeds 101–1010 for the ten stability runs.
Versions pinned in `requirements.txt` (numpy 1.26.4, scikit-learn 1.4.2,
shap 0.45.1, lime 0.2.0.1, xgboost 2.0.3). Reruns reproduce results to four
decimal places — verified by two independent executions of the binary
baselines.

## Data

NF-UNSW-NB15-v2 via Kaggle `dhoogla/nfunswnb15v2`, licence **CC-BY-NC-SA-4.0**.
Note this release is **deduplicated**: 1,986,745 flows at a 3.78% attack rate,
not the 2.39M / 3.98% quoted in the original literature. Cite the figures this
pipeline reports.

Original feature schema: Sarhan, M., Layeghy, S. & Portmann, M., *Towards a
Standard Feature Set for Network Intrusion Detection System Datasets*,
arXiv:2101.11315.
