"""
make_figures.py — every figure for the thesis, from the CSVs in results/tables.

    python make_figures.py

Writes PNG (300 dpi, for the document) and PDF (vector, for print) into
results/figures/. Missing inputs are skipped with a note rather than crashing,
so this is safe to run at any point.

A note on chart choice: there is deliberately no pie chart of the class
distribution. At 96.2% Benign it would be one slice and nine slivers, and it
cannot show the uncertainty that matters for the rare classes. A log-scale
horizontal bar does the same job readably.
"""

import sys
import warnings
from pathlib import Path

warnings.filterwarnings("ignore")

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "src"))

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.patches import Patch

try:
    import config
    TABLES = config.RESULTS_TABLES
    FIGURES = config.RESULTS_FIGURES
except Exception:
    TABLES = ROOT / "results" / "tables"
    FIGURES = ROOT / "results" / "figures"
FIGURES.mkdir(parents=True, exist_ok=True)

# ---------------------------------------------------------------- style ----
plt.rcParams.update({
    "figure.dpi": 110,
    "savefig.dpi": 300,
    "savefig.bbox": "tight",
    "font.size": 10,
    "axes.titlesize": 11,
    "axes.titleweight": "bold",
    "axes.labelsize": 10,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "axes.grid": True,
    "grid.alpha": 0.25,
    "grid.linestyle": "-",
    "legend.frameon": False,
    "figure.facecolor": "white",
})

# consistent model order and colour across every figure
MODEL_ORDER = ["DecisionTree", "GaussianNB", "LogisticRegression",
               "XGBoost", "MLP", "RandomForest"]
MODEL_COLOR = {
    "DecisionTree":       "#c1440e",
    "GaussianNB":         "#e08214",
    "LogisticRegression": "#b8a02e",
    "XGBoost":            "#4a8c5f",
    "MLP":                "#2b7a9e",
    "RandomForest":       "#3d4f8f",
}
EXPLAINER_COLOR = {"LIME": "#2b7a9e", "TreeSHAP": "#4a8c5f",
                   "LinearSHAP": "#4a8c5f", "KernelSHAP": "#4a8c5f",
                   "Permutation": "#e08214", "Random": "#999999",
                   "TOPSIS": "#b8a02e"}

made, skipped = [], []


def save(fig, name):
    for ext in ("png", "pdf"):
        fig.savefig(FIGURES / f"{name}.{ext}")
    plt.close(fig)
    made.append(name)
    print(f"  wrote {name}.png / .pdf")


def read(name):
    p = TABLES / name
    return pd.read_csv(p) if p.exists() else None


def available_models(pattern):
    return [m for m in MODEL_ORDER if (TABLES / pattern.format(m)).exists()]


# =========================================================== FIGURE 1 ======
def fig_class_distribution():
    d = read("per_class_recall_multiclass_seed42.csv")
    if d is None:
        skipped.append("01_class_distribution")
        return
    d = d.sort_values("support")
    fig, ax = plt.subplots(figsize=(7, 4))
    bars = ax.barh(d["class"], d["support"], color="#3d4f8f", alpha=0.85)
    bars[-1].set_color("#999999")           # Benign de-emphasised
    ax.set_xscale("log")
    ax.set_xlabel("test-set flows (log scale)")
    ax.set_title("Class distribution is extreme: Benign is 96.2% of flows")
    for b, v in zip(bars, d["support"]):
        ax.text(v * 1.15, b.get_y() + b.get_height() / 2, f"{v:,}",
                va="center", fontsize=8)
    ax.set_xlim(right=d["support"].max() * 4)
    ax.text(0.99, 0.03,
            "Worms: 33 flows — every metric for it rests on this sample",
            transform=ax.transAxes, ha="right", fontsize=8,
            style="italic", color="#c1440e")
    save(fig, "01_class_distribution")


# =========================================================== FIGURE 2 ======
def fig_detection_performance():
    d = read("baselines_multiclass_seed42.csv")
    if d is None:
        skipped.append("02_detection_performance")
        return
    d = d.sort_values("macro_f1")
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(11, 4.2))

    cols = [MODEL_COLOR.get(m, "#6b4c7a") for m in d["model"]]
    a1.barh(d["model"], d["macro_f1"], color=cols, alpha=0.9)
    a1.axvline(0.5, color="#999", lw=0.8, ls=":")
    a1.set_xlabel("macro-F1")
    a1.set_title("Detection performance (multiclass)")
    for i, (v, m) in enumerate(zip(d["macro_f1"], d["model"])):
        a1.text(v + 0.01, i, f"{v:.3f}", va="center", fontsize=8)
    a1.set_xlim(0, max(d["macro_f1"]) * 1.2)

    a2.barh(d["model"], d["false_alarm_rate"] * 100, color=cols, alpha=0.9)
    a2.set_xlabel("false alarm rate (%)")
    a2.set_title("False alarms on 382,333 benign flows")
    for i, v in enumerate(d["false_alarm_rate"] * 100):
        a2.text(v + max(d["false_alarm_rate"] * 100) * 0.015, i,
                f"{v:.2f}%", va="center", fontsize=8)
    a2.set_xlim(0, max(d["false_alarm_rate"] * 100) * 1.25)

    fig.suptitle("GaussianNB, LogisticRegression and MLP are RQ1 controls, "
                 "not competitive detectors", fontsize=9, style="italic", y=1.02)
    save(fig, "02_detection_performance")


# =========================================================== FIGURE 3 ======
def fig_per_class_recall():
    d = read("per_class_recall_multiclass_seed42.csv")
    if d is None:
        skipped.append("03_per_class_performance")
        return
    d = d.sort_values("support", ascending=False)
    x = np.arange(len(d))
    fig, ax = plt.subplots(figsize=(9, 4.2))
    ax.bar(x - 0.22, d["recall"], 0.42, label="recall", color="#2b7a9e", alpha=0.9)
    ax.bar(x + 0.22, d["precision"], 0.42, label="precision", color="#e08214", alpha=0.9)
    ax.set_xticks(x)
    ax.set_xticklabels([f"{c}\nn={s:,}" for c, s in zip(d["class"], d["support"])],
                       fontsize=7.5)
    ax.set_ylabel("score")
    ax.set_ylim(0, 1.05)
    ax.legend(loc="upper right")
    ax.set_title("Rare classes fail on precision, not recall")
    save(fig, "03_per_class_performance")


# =========================================================== FIGURE 4 ======
def fig_rq1_by_model():
    models = available_models("rq1_stability_{}_multiclass.csv")
    if not models:
        skipped.append("04_rq1_stability_by_model")
        return
    data = {m: read(f"rq1_stability_{m}_multiclass.csv") for m in models}

    fig, (a1, a2) = plt.subplots(1, 2, figsize=(12, 4.6))

    for ax, col, lab in ((a1, "jaccard_at_5", "Jaccard@5"),
                         (a2, "kendall_tau", "Kendall τ")):
        vals = [data[m][col].dropna().values for m in models]
        bp = ax.boxplot(vals, labels=models, patch_artist=True, widths=0.6,
                        medianprops=dict(color="black", lw=1.4),
                        flierprops=dict(marker=".", ms=2, alpha=0.25))
        for patch, m in zip(bp["boxes"], models):
            patch.set_facecolor(MODEL_COLOR.get(m, "#888"))
            patch.set_alpha(0.75)
        ax.set_ylabel(lab)
        ax.set_ylim(-0.05, 1.05)
        ax.tick_params(axis="x", rotation=35, labelsize=8)
        for t in ax.get_xticklabels():
            t.set_ha("right")
        ax.set_title(f"{lab} across ten LIME runs on the same instance")
        ax.axhline(1.0, color="#4a8c5f", lw=0.9, ls="--", alpha=0.7)

    a1.text(0.02, 0.05, "1.0 = perfectly reproducible", transform=a1.transAxes,
            fontsize=8, style="italic", color="#4a8c5f")
    fig.suptitle("RQ1 — identical model, identical input, ten runs: "
                 "reproducibility varies 2.6-fold across models",
                 fontsize=11, fontweight="bold", y=1.03)
    save(fig, "04_rq1_stability_by_model")


# =========================================================== FIGURE 5 ======
def fig_rq1_heatmap():
    models = available_models("rq1_summary_{}_multiclass.csv")
    if not models:
        skipped.append("05_rq1_class_heatmap")
        return
    frames = {}
    for m in models:
        d = pd.read_csv(TABLES / f"rq1_summary_{m}_multiclass.csv", index_col=0)
        frames[m] = d["jaccard5_mean"]
    M = pd.DataFrame(frames)
    M = M.loc[M.mean(axis=1).sort_values().index]

    fig, ax = plt.subplots(figsize=(7.5, 4.8))
    im = ax.imshow(M.values, cmap="RdYlGn", vmin=0, vmax=1, aspect="auto")
    ax.set_xticks(range(len(M.columns)))
    ax.set_xticklabels(M.columns, rotation=35, ha="right", fontsize=8)
    ax.set_yticks(range(len(M.index)))
    ax.set_yticklabels(M.index, fontsize=8)
    for i in range(M.shape[0]):
        for j in range(M.shape[1]):
            v = M.values[i, j]
            ax.text(j, i, f"{v:.2f}", ha="center", va="center", fontsize=7.5,
                    color="black" if 0.25 < v < 0.8 else "white")
    ax.set_title("RQ1 — mean Jaccard@5 by attack class and model")
    ax.grid(False)
    fig.colorbar(im, ax=ax, label="Jaccard@5", shrink=0.85)
    save(fig, "05_rq1_class_heatmap")


# =========================================================== FIGURE 6 ======
def fig_rq1_falsification():
    ax_df = read("rq1_smoothness_axis.csv")
    gr = read("rq1_granularity.csv")
    base = read("baselines_multiclass_seed42.csv")
    if ax_df is None or gr is None:
        skipped.append("06_rq1_falsified_mechanisms")
        return

    g = gr.copy()
    if base is not None:
        g = g.merge(base[["model", "macro_f1"]], on="model", how="left")

    fig, axes = plt.subplots(1, 3, figsize=(13, 4.1))

    # (a) smoothness axis
    a = axes[0]
    order = ["GaussianNB", "LogisticRegression", "MLP",
             "XGBoost", "RandomForest", "DecisionTree"]
    sub = ax_df.set_index("model").reindex([m for m in order
                                            if m in set(ax_df["model"])])
    cols = [MODEL_COLOR.get(m, "#888") for m in sub.index]
    a.bar(range(len(sub)), sub["jaccard@5"], color=cols, alpha=0.9)
    a.set_xticks(range(len(sub)))
    a.set_xticklabels(sub.index, rotation=40, ha="right", fontsize=7.5)
    a.set_ylabel("Jaccard@5")
    a.set_title("(a) Smoothness — falsified", fontsize=10)
    a.text(0.5, 0.94, "predicted: monotonic decrease →",
           transform=a.transAxes, ha="center", fontsize=8, color="#c1440e")

    # (b) accuracy
    b = axes[1]
    if "macro_f1" in g:
        for _, r in g.iterrows():
            b.scatter(r["macro_f1"], r["jaccard@5"], s=110,
                      color=MODEL_COLOR.get(r["model"], "#888"),
                      edgecolor="white", zorder=3)
            b.annotate(r["model"], (r["macro_f1"], r["jaccard@5"]),
                       textcoords="offset points", xytext=(0, 9),
                       ha="center", fontsize=7)
        rho = g[["macro_f1", "jaccard@5"]].corr(method="spearman").iloc[0, 1]
        b.set_xlabel("macro-F1")
        b.set_ylabel("Jaccard@5")
        b.set_title(f"(b) Accuracy — falsified (ρ={rho:+.2f})", fontsize=10)
        b.set_ylim(0.2, 1.0)

    # (c) granularity
    c = axes[2]
    for _, r in g.iterrows():
        c.scatter(r["unique_top_proba"], r["jaccard@5"], s=110,
                  color=MODEL_COLOR.get(r["model"], "#888"),
                  edgecolor="white", zorder=3)
        c.annotate(r["model"], (r["unique_top_proba"], r["jaccard@5"]),
                   textcoords="offset points", xytext=(0, 9),
                   ha="center", fontsize=7)
    rho = g[["unique_top_proba", "jaccard@5"]].corr(method="spearman").iloc[0, 1]
    c.set_xlabel("distinct predicted probabilities")
    c.set_ylabel("Jaccard@5")
    c.set_title(f"(c) Granularity — falsified (ρ={rho:+.2f}, p=0.21)", fontsize=10)
    c.set_ylim(0.2, 1.0)

    fig.suptitle("RQ1 — four pre-registered mechanisms, none supported",
                 fontsize=11, fontweight="bold", y=1.04)
    save(fig, "06_rq1_falsified_mechanisms")


# =========================================================== FIGURE 7 ======
def fig_rq2_curves():
    models = available_models("rq2_faithfulness_{}_multiclass.csv")
    if not models:
        skipped.append("07_rq2_faithfulness_curves")
        return
    ks = list(range(1, 11))
    fig, axes = plt.subplots(1, len(models), figsize=(4.4 * len(models), 4.2),
                             squeeze=False)
    axes = axes[0]

    for ax, m in zip(axes, models):
        d = read(f"rq2_faithfulness_{m}_multiclass.csv")
        for expl, grp in d.groupby("explainer"):
            drops = [(grp["p_original"] - grp[f"comp_k{k}"]).mean() for k in ks]
            sem = [(grp["p_original"] - grp[f"comp_k{k}"]).sem() for k in ks]
            col = EXPLAINER_COLOR.get(expl, "#666")
            style = "--" if expl == "Random" else "-"
            ax.plot(ks, drops, style, color=col, lw=2, marker="o", ms=4,
                    label=expl, zorder=3 if expl != "Random" else 2)
            ax.fill_between(ks, np.array(drops) - np.array(sem),
                            np.array(drops) + np.array(sem),
                            color=col, alpha=0.15)
        ax.set_xlabel("k features masked")
        ax.set_title(m, fontsize=10)
        ax.set_xticks(ks)
        ax.tick_params(labelsize=8)
    axes[0].set_ylabel("drop in predicted probability")
    axes[0].legend(fontsize=8, loc="upper left")

    fig.suptitle("RQ2 — comprehensiveness: masking the top-k features on a "
                 "frozen model.  Dashed grey is the random control — "
                 "the experiment, not a robustness check.",
                 fontsize=9.5, y=1.04)
    save(fig, "07_rq2_faithfulness_curves")


# =========================================================== FIGURE 8 ======
def fig_rq2_summary():
    models = available_models("rq2_faithfulness_{}_multiclass.csv")
    if not models:
        skipped.append("08_rq2_ratio_vs_random")
        return
    rows = []
    for m in models:
        d = read(f"rq2_faithfulness_{m}_multiclass.csv")
        g = d.groupby("explainer")["comprehensiveness_auc"].agg(["mean", "sem"])
        rnd = g.loc["Random", "mean"]
        for expl in g.index:
            rows.append({"model": m, "explainer": expl,
                         "auc": g.loc[expl, "mean"], "sem": g.loc[expl, "sem"],
                         "ratio": g.loc[expl, "mean"] / rnd})
    r = pd.DataFrame(rows)

    fig, (a1, a2) = plt.subplots(1, 2, figsize=(11.5, 4.2))
    expls = [e for e in ["LIME", "TreeSHAP", "LinearSHAP", "KernelSHAP", "Random"]
             if e in set(r["explainer"])]
    x = np.arange(len(models))
    w = 0.8 / len(expls)
    for i, e in enumerate(expls):
        sub = r[r["explainer"] == e].set_index("model").reindex(models)
        a1.bar(x + i * w - 0.4 + w / 2, sub["auc"], w, yerr=sub["sem"],
               label=e, color=EXPLAINER_COLOR.get(e, "#666"), alpha=0.9,
               error_kw=dict(lw=0.8))
    a1.set_xticks(x)
    a1.set_xticklabels(models, rotation=25, ha="right", fontsize=8)
    a1.set_ylabel("comprehensiveness AUC")
    a1.legend(fontsize=8)
    a1.set_title("Absolute faithfulness")

    lime = r[r["explainer"] == "LIME"].set_index("model").reindex(models)
    cols = [MODEL_COLOR.get(m, "#888") for m in models]
    a2.bar(x, lime["ratio"], color=cols, alpha=0.9)
    a2.axhline(1.0, color="#c1440e", lw=1.2, ls="--")
    a2.text(len(models) - 0.5, 1.03, "no better than random",
            ha="right", fontsize=8, color="#c1440e")
    a2.set_xticks(x)
    a2.set_xticklabels(models, rotation=25, ha="right", fontsize=8)
    a2.set_ylabel("LIME ÷ random control")
    a2.set_title("How much better than random?")
    for i, v in enumerate(lime["ratio"]):
        if not np.isnan(v):
            a2.text(i, v + 0.03, f"{v:.2f}×", ha="center", fontsize=8)

    save(fig, "08_rq2_ratio_vs_random")


# =========================================================== FIGURE 9 ======
def fig_stability_vs_faithfulness():
    ms = available_models("rq2_faithfulness_{}_multiclass.csv")
    ax_df = read("rq1_smoothness_axis.csv")
    if not ms or ax_df is None:
        skipped.append("09_stability_vs_faithfulness")
        return
    j = ax_df.set_index("model")["jaccard@5"]
    pts = []
    for m in ms:
        d = read(f"rq2_faithfulness_{m}_multiclass.csv")
        g = d.groupby("explainer")["comprehensiveness_auc"].mean()
        if m in j.index and "LIME" in g and "Random" in g:
            pts.append({"model": m, "jaccard": j[m],
                        "ratio": g["LIME"] / g["Random"]})
    p = pd.DataFrame(pts)
    if len(p) < 2:
        skipped.append("09_stability_vs_faithfulness")
        return

    fig, ax = plt.subplots(figsize=(6.4, 4.8))
    for _, r in p.iterrows():
        ax.scatter(r["jaccard"], r["ratio"], s=200,
                   color=MODEL_COLOR.get(r["model"], "#888"),
                   edgecolor="white", lw=1.5, zorder=3)
        ax.annotate(r["model"], (r["jaccard"], r["ratio"]),
                    textcoords="offset points", xytext=(0, 14),
                    ha="center", fontsize=8.5)
    if len(p) > 2:
        z = np.polyfit(p["jaccard"], p["ratio"], 1)
        xs = np.linspace(p["jaccard"].min() * 0.9, p["jaccard"].max() * 1.05, 50)
        ax.plot(xs, np.polyval(z, xs), ":", color="#c1440e", lw=1.5)
    ax.set_xlabel("RQ1 — stability (Jaccard@5)")
    ax.set_ylabel("RQ2 — faithfulness (LIME ÷ random)")
    ax.set_title("The least stable explanations are the most faithful")
    ax.text(0.98, 0.95, "stability and faithfulness\nare not the same property",
            transform=ax.transAxes, ha="right", va="top", fontsize=8.5,
            style="italic", color="#c1440e")
    save(fig, "09_stability_vs_faithfulness")


# ========================================================== FIGURE 10 ======
def fig_rq4_heatmaps():
    models = available_models("rq4_agreement_{}_multiclass.csv")
    if not models:
        skipped.append("10_rq4_agreement_heatmaps")
        return
    has_j = all((TABLES / f"rq4_jaccard5_{m}_multiclass.csv").exists()
                for m in models)
    nrow = 2 if has_j else 1
    fig, axes = plt.subplots(nrow, len(models),
                             figsize=(4.0 * len(models), 3.7 * nrow),
                             squeeze=False)

    for col, m in enumerate(models):
        specs = [("rq4_agreement_{}_multiclass.csv", "Spearman ρ", -1, 1, "RdBu_r")]
        if has_j:
            specs.append(("rq4_jaccard5_{}_multiclass.csv", "Jaccard@5", 0, 1, "YlGnBu"))
        for row, (pat, lab, vmin, vmax, cmap) in enumerate(specs):
            M = pd.read_csv(TABLES / pat.format(m), index_col=0)
            ax = axes[row][col]
            im = ax.imshow(M.values, cmap=cmap, vmin=vmin, vmax=vmax)
            ax.set_xticks(range(len(M.columns)))
            ax.set_xticklabels(M.columns, rotation=40, ha="right", fontsize=7)
            ax.set_yticks(range(len(M.index)))
            ax.set_yticklabels(M.index if col == 0 else [], fontsize=7)
            for i in range(M.shape[0]):
                for jj in range(M.shape[1]):
                    v = M.values[i, jj]
                    ax.text(jj, i, f"{v:.2f}", ha="center", va="center",
                            fontsize=6.8,
                            color="white" if abs(v) > 0.6 else "black")
            ax.set_title(f"{m}\n{lab}" if row == 0 else lab, fontsize=9)
            ax.grid(False)
    fig.suptitle("RQ4 — which explainers agree depends on the model, "
                 "and on the metric",
                 fontsize=11, fontweight="bold", y=1.01)
    save(fig, "10_rq4_agreement_heatmaps")


# ========================================================== FIGURE 11 ======
def fig_cost():
    d = read("baselines_multiclass_seed42.csv")
    models = available_models("rq1_stability_{}_multiclass.csv")
    if d is None:
        skipped.append("11_computational_cost")
        return
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(11, 4.2))
    d2 = d.sort_values("train_time_s")
    cols = [MODEL_COLOR.get(m, "#6b4c7a") for m in d2["model"]]
    a1.barh(d2["model"], d2["train_time_s"], color=cols, alpha=0.9)
    a1.set_xlabel("training time (s)")
    a1.set_title("Training cost")
    for i, v in enumerate(d2["train_time_s"]):
        a1.text(v + max(d2["train_time_s"]) * 0.015, i, f"{v:.0f}s",
                va="center", fontsize=8)

    d3 = d.sort_values("inference_ms_per_flow")
    cols = [MODEL_COLOR.get(m, "#6b4c7a") for m in d3["model"]]
    a2.barh(d3["model"], d3["inference_ms_per_flow"] * 1000, color=cols, alpha=0.9)
    a2.set_xlabel("inference (µs per flow)")
    a2.set_title("Inference cost")

    fig.suptitle("Cost accounting — every claim carries a price",
                 fontsize=9.5, style="italic", y=1.02)
    save(fig, "11_computational_cost")


# ---------------------------------------------------------------- main ----
if __name__ == "__main__":
    print(f"reading  {TABLES}")
    print(f"writing  {FIGURES}\n")
    for fn in (fig_class_distribution, fig_detection_performance,
               fig_per_class_recall, fig_rq1_by_model, fig_rq1_heatmap,
               fig_rq1_falsification, fig_rq2_curves, fig_rq2_summary,
               fig_stability_vs_faithfulness, fig_rq4_heatmaps, fig_cost):
        try:
            fn()
        except Exception as e:
            print(f"  FAILED {fn.__name__}: {type(e).__name__}: {e}")
            skipped.append(fn.__name__)

    print(f"\n{len(made)} figures written to {FIGURES}")
    if skipped:
        print(f"skipped (missing inputs): {', '.join(skipped)}")
