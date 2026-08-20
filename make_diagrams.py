"""
make_diagrams.py — schematic figures for Chapters I, II and III.

    python make_diagrams.py

make_figures.py produces the data plots (Chapter IV). This produces the
conceptual and methodological diagrams the IIUC format expects earlier in the
document: the trust gap that motivates the work, the position against prior
art, the system architecture, and the protocol for each research question.

Named by chapter to match the LIST OF ILLUSTRATIONS convention:
Figure 1.1, 2.1, 3.1 ... Output goes to results/figures/.
"""

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "src"))

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch, Rectangle

try:
    import config
    FIGURES = config.RESULTS_FIGURES
except Exception:
    FIGURES = ROOT / "results" / "figures"
FIGURES.mkdir(parents=True, exist_ok=True)

plt.rcParams.update({
    "figure.dpi": 110, "savefig.dpi": 300, "savefig.bbox": "tight",
    "font.size": 9.5, "figure.facecolor": "white",
})

INK = "#1a1a2e"
BLUE = "#2b7a9e"
GREEN = "#4a8c5f"
ORANGE = "#e08214"
RED = "#c1440e"
GREY = "#8a8a8a"
PALE = "#eef2f6"

made = []


def save(fig, name):
    for ext in ("png", "pdf"):
        fig.savefig(FIGURES / f"{name}.{ext}")
    plt.close(fig)
    made.append(name)
    print(f"  wrote {name}.png / .pdf")


def box(ax, x, y, w, h, text, fc=PALE, ec=INK, fs=9, weight="normal",
        tc=INK, lw=1.2, style="round,pad=0.02"):
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle=style,
                                facecolor=fc, edgecolor=ec, linewidth=lw))
    ax.text(x + w / 2, y + h / 2, text, ha="center", va="center",
            fontsize=fs, color=tc, fontweight=weight, linespacing=1.5)


def arrow(ax, x1, y1, x2, y2, color=INK, lw=1.4, style="-|>", ls="-"):
    ax.add_patch(FancyArrowPatch((x1, y1), (x2, y2), arrowstyle=style,
                                 mutation_scale=13, color=color,
                                 linewidth=lw, linestyle=ls,
                                 shrinkA=2, shrinkB=2))


def blank(figsize):
    fig, ax = plt.subplots(figsize=figsize)
    ax.set_xlim(0, 100); ax.set_ylim(0, 100)
    ax.axis("off")
    return fig, ax


# ====================================================== FIGURE 1.1 ========
def fig_1_1_trust_gap():
    """Chapter I — the problem this thesis addresses."""
    fig, ax = blank((9, 4.6))

    box(ax, 2, 62, 20, 20, "Network flow\n(43 features)", fc="#ffffff")
    arrow(ax, 22, 72, 30, 72)
    box(ax, 30, 62, 20, 20, "Ensemble IDS\nDT · RF · XGBoost", fc=PALE)
    arrow(ax, 50, 72, 58, 72)
    box(ax, 58, 62, 20, 20, 'Prediction:\n"Exploits"\np = 0.97', fc="#e8f0e8",
        ec=GREEN)

    arrow(ax, 68, 62, 68, 48, color=BLUE)
    box(ax, 52, 26, 32, 22,
        "LIME explanation\n\n1. min_ttl\n2. max_ttl\n3. tcp_flags",
        fc="#e8f2f6", ec=BLUE)

    ax.text(88, 37, "?", fontsize=54, color=RED, ha="center",
            va="center", fontweight="bold")

    qs = ["Would a second run give the same answer?",
          "Do those features actually drive the prediction?",
          "Would SHAP name the same three?",
          "What does producing this cost?"]
    for i, q in enumerate(qs):
        ax.text(4, 18 - i * 5, f"·  {q}", fontsize=9.5, color=RED, va="center")

    ax.plot([2, 96], [22.5, 22.5], color=GREY, lw=0.8, ls=":")
    ax.text(2, 92,
            "Standard practice stops at the explanation. "
            "This thesis measures whether it can be trusted.",
            fontsize=10, fontweight="bold", color=INK)
    save(fig, "fig_1_1_trust_gap")


# ====================================================== FIGURE 2.1 ========
def fig_2_1_prior_art():
    """Chapter II — position against the three closest papers."""
    fig, ax = blank((9.5, 5.2))

    works = [
        ("Patil et al. (2022)\nElectronics 11(19)",
         "Applies LIME to IDS.\nNo validation of the\nexplanation itself.", 3),
        ("Kalutharage et al. (2023)\nComputers 12(2)",
         "Reports 6, 8, 7, 10, 4 'most\ninfluential' features across five\n"
         "instances. Attributes the spread\nto the data, not the explainer.", 27),
        ("Hernandez et al. (2025)\nElectronics 14(13)",
         "Retrains on top-k SHAP features\n(feature selection). Compares two\n"
         "rankings by eye on merged traffic.", 51),
    ]
    for title, body, x in works:
        box(ax, x, 56, 22, 12, title, fc="#ffffff", fs=8.5, weight="bold")
        ax.text(x + 11, 50, body, ha="center", va="top", fontsize=7.6,
                color="#444")

    box(ax, 75, 56, 22, 12, "This thesis", fc="#e8f2f6", ec=BLUE, fs=9,
        weight="bold", tc=BLUE)

    for x in (14, 38, 62):
        arrow(ax, x, 55, x + 4, 30, color=GREY, lw=1.0, ls=":")
    arrow(ax, 86, 55, 86, 32, color=BLUE, lw=1.6)

    deltas = [
        "Measures explanation stability directly — no prior art",
        "Masks features on a FROZEN model, against a random control,\n"
        "  rather than retraining on a subset",
        "Quantifies four-way explainer agreement with rank correlation\n"
        "  AND top-k overlap, not visual comparison",
        "Reports the computational cost of every explanation",
    ]
    box(ax, 3, 4, 94, 26, "", fc="#f7fafc", ec=BLUE, lw=1.4)
    ax.text(6, 26, "What this thesis does differently", fontsize=9.5,
            fontweight="bold", color=BLUE)
    for i, d in enumerate(deltas):
        ax.text(6, 20.5 - i * 4.6, f"{i+1}.  {d}", fontsize=8.4, va="top",
                color=INK)

    ax.text(3, 92,
            "All three neighbours are DDoS-specific; this thesis is general "
            "multiclass intrusion detection.",
            fontsize=8.6, style="italic", color="#555")
    save(fig, "fig_2_1_prior_art_positioning")


# ====================================================== FIGURE 3.1 ========
def fig_3_1_architecture():
    """Chapter III — the system architecture. The core methodology figure."""
    fig, ax = blank((10, 7.6))

    # data layer
    box(ax, 4, 88, 30, 9, "NF-UNSW-NB15-v2\n1,986,745 flows · 43 columns",
        fc="#ffffff", fs=8.6)
    arrow(ax, 19, 88, 19, 82)
    box(ax, 4, 72, 30, 9,
        "Drop identifiers\nl4_src_port · l4_dst_port  →  39 features",
        fc="#fdf0e8", ec=ORANGE, fs=8.2)
    arrow(ax, 19, 72, 19, 66)
    box(ax, 4, 57, 30, 8, "Stratified split  80 / 20\nseed 42, frozen",
        fc=PALE, fs=8.4)

    # detection layer
    ax.text(50, 96, "DETECTION LAYER", fontsize=8.5, fontweight="bold",
            color=GREY, ha="center")
    trees = [("DecisionTree", 40), ("RandomForest", 57), ("XGBoost", 74)]
    for nm, x in trees:
        box(ax, x, 84, 16, 7, nm, fc="#e8f0e8", ec=GREEN, fs=7.8)
    ctrl = [("GaussianNB", 40), ("LogisticReg.", 57), ("MLP", 74)]
    for nm, x in ctrl:
        box(ax, x, 74, 16, 7, nm, fc="#f4f0f8", ec="#6b4c7a", fs=7.8)
    ax.text(91, 77.5, "RQ1\ncontrols", fontsize=7.4, color="#6b4c7a",
            ha="center", va="center", style="italic")

    arrow(ax, 34, 61, 40, 79, color=GREY, lw=1.0)
    for x in (48, 65, 82):
        arrow(ax, x, 84, x, 81.5, color=GREY, lw=0.9)
        arrow(ax, x, 74, x, 68, color=GREY, lw=0.9)
    box(ax, 40, 60, 50, 7, "Soft-voting ensemble   (mean of predict_proba)",
        fc="#e8f0e8", ec=GREEN, fs=8.4, weight="bold")

    arrow(ax, 65, 60, 65, 54, lw=1.6)
    box(ax, 40, 46, 50, 7, "FROZEN MODEL  →  prediction + probability",
        fc="#fff4e8", ec=RED, fs=8.6, weight="bold")
    ax.text(93, 49.5, "never\nretrained", fontsize=7.4, color=RED,
            ha="center", va="center", style="italic")

    # explanation layer
    ax.text(50, 41, "EXPLANATION LAYER", fontsize=8.5, fontweight="bold",
            color=GREY, ha="center")
    expl = [("LIME\nlocal, surrogate", 6, BLUE),
            ("SHAP\nTree / Linear / Kernel", 30, GREEN),
            ("Permutation\nglobal, model-agnostic", 54, ORANGE),
            ("TOPSIS\nno model at all", 78, "#b8a02e")]
    for nm, x, c in expl:
        box(ax, x, 30, 20, 8, nm, fc="#ffffff", ec=c, fs=7.6)
        arrow(ax, x + 10, 30, x + 10, 24, color=c, lw=1.1)
    arrow(ax, 65, 46, 16, 38.5, color=GREY, lw=1.0, ls=":")

    # evaluation layer
    ax.text(50, 20, "EVALUATION — the contribution", fontsize=8.5,
            fontweight="bold", color=GREY, ha="center")
    evals = [("RQ1  Stability\nJaccard@k · Kendall τ", 4, 26),
             ("RQ2  Faithfulness\ncomp. / suff. vs random", 33, 30),
             ("RQ4  Agreement\nSpearman ρ · Jaccard@k", 66, 30)]
    for nm, x, w in evals:
        box(ax, x, 9, w, 8, nm, fc="#e8f2f6", ec=BLUE, fs=7.8)

    box(ax, 4, 1, 92, 6,
        "Can an analyst trust what the IDS says about its own decision?",
        fc="#f7fafc", ec=INK, fs=9.4, weight="bold")
    save(fig, "fig_3_1_system_architecture")


# ====================================================== FIGURE 3.2 ========
def fig_3_2_rq1_protocol():
    """Chapter III — how RQ1 is measured."""
    fig, ax = blank((9.5, 4.8))

    box(ax, 3, 72, 22, 12, "One test flow\n(fixed)", fc="#ffffff")
    box(ax, 3, 54, 22, 12, "One trained model\n(frozen)", fc="#fff4e8", ec=RED)
    ax.text(14, 47, "nothing changes between runs", fontsize=8,
            style="italic", color=RED, ha="center")

    for i in range(5):
        y = 78 - i * 15
        arrow(ax, 25, 70, 36, y + 4, color=GREY, lw=0.9)
        lbl = f"LIME  seed {[101,202,303,404,1010][i]}"
        box(ax, 36, y, 26, 8, lbl, fc="#e8f2f6", ec=BLUE, fs=8)
        ax.text(66, y + 4, "→  ranking", fontsize=8, va="center", color="#555")
    ax.text(49, 20, "⋮", fontsize=16, ha="center", color=GREY)
    ax.text(49, 12, "ten runs in total", fontsize=8.4, ha="center",
            style="italic", color="#555")

    box(ax, 78, 46, 20, 30,
        "Compare the\nten rankings\n\nJaccard@5\nJaccard@10\nKendall τ",
        fc="#f7fafc", ec=BLUE, fs=8.4)
    arrow(ax, 74, 60, 78, 60, color=BLUE)

    ax.text(3, 94,
            "If LIME were deterministic every run would return the same "
            "ranking. It fits a surrogate\nto a random perturbation sample, "
            "so the ranking is a random variable.",
            fontsize=8.8, color=INK, va="top")
    save(fig, "fig_3_2_rq1_protocol")


# ====================================================== FIGURE 3.3 ========
def fig_3_3_rq2_protocol():
    """Chapter III — comprehensiveness, sufficiency, and the control."""
    fig, ax = blank((10, 5.8))

    ax.text(3, 99,
            "The model is never retrained. Retraining on a subset would test "
            "whether those features\nSUFFICE to build a model — not whether "
            "THIS model used them.",
            fontsize=8.8, va="top", color=INK)

    box(ax, 3, 76, 26, 9, "Frozen model\np(Exploits) = 0.97",
        fc="#fff4e8", ec=RED, fs=8.6)
    ax.text(38, 80.5, "explanation names top-k features", fontsize=8.4,
            va="center", color="#555")

    rows = [
        ("COMPREHENSIVENESS", "mask the top-k  →  probability should FALL",
         GREEN, 54, ["0.97", "0.81", "0.64", "0.41"]),
        ("SUFFICIENCY", "keep ONLY the top-k  →  probability should HOLD",
         BLUE, 31, ["0.97", "0.42", "0.61", "0.89"]),
        ("RANDOM CONTROL", "mask k RANDOM features  —  the actual experiment",
         RED, 8, ["0.97", "0.83", "0.69", "0.52"]),
    ]
    for title, sub, col, y, vals in rows:
        ax.text(3, y + 15, title, fontsize=8.8, fontweight="bold", color=col)
        ax.text(3, y + 11, sub, fontsize=8.2, color="#555")
        for i, v in enumerate(vals):
            x = 3 + i * 17
            lab = ["original", "k = 1", "k = 2", "k = 3"][i]
            box(ax, x, y, 14, 8, f"{lab}\n{v}", fc="#ffffff", ec=col, fs=7.8)
            if i < 3:
                arrow(ax, x + 14, y + 4, x + 17, y + 4, color=col, lw=1.0)
        ax.text(74, y + 4, "…  k = 10", fontsize=8, va="center", color="#777")

    box(ax, 84, 4, 14, 72,
        "Curve\n+\nAUC\n\nper\nexplainer",
        fc="#f7fafc", ec=INK, fs=8.2)

    save(fig, "fig_3_3_rq2_protocol")


# ====================================================== FIGURE 3.4 ========
def fig_3_4_experimental_design():
    """Chapter III — how the research questions connect."""
    fig, ax = blank((9, 4.4))

    box(ax, 35, 80, 30, 12, "Frozen trained model\n+ fixed test instances",
        fc="#fff4e8", ec=RED, fs=8.8, weight="bold")

    qs = [("RQ1\nStability", 4, BLUE,
           "Same model, same input,\nten LIME runs.\nDoes it repeat?"),
          ("RQ2\nFaithfulness", 36, GREEN,
           "Mask the named features.\nDoes the prediction move\nmore than random?"),
          ("RQ4\nAgreement", 68, ORANGE,
           "Four explainers,\none model.\nDo they concur?")]
    for nm, x, c, body in qs:
        arrow(ax, 50, 80, x + 14, 62, color=GREY, lw=1.0)
        box(ax, x, 50, 28, 11, nm, fc="#ffffff", ec=c, fs=9.2, weight="bold",
            tc=c)
        ax.text(x + 14, 46, body, ha="center", va="top", fontsize=7.8,
                color="#444")

    arrow(ax, 64, 55, 68, 55, color=INK, lw=1.3, style="<|-|>")
    ax.text(66, 66, "RQ2 adjudicates\nwhere RQ4 finds\ndisagreement",
            fontsize=7.4, ha="center", color=INK, style="italic")

    box(ax, 4, 6, 92, 14,
        "RQ3 — cross-network drift (NF-CSE-CIC-IDS2018-v2):  "
        "specified, not executed.  See Future Work.",
        fc="#f4f4f4", ec=GREY, fs=8.6, lw=1.0, style="round,pad=0.02")
    save(fig, "fig_3_4_experimental_design")


# ====================================================== FIGURE 3.5 ========
def fig_3_5_preprocessing():
    """Chapter III — the data pipeline, and where leakage is avoided."""
    fig, ax = blank((9.5, 4.2))

    steps = [
        ("Raw parquet\n1,986,745 × 43", "#ffffff", INK),
        ("Normalise\ncolumn names", PALE, INK),
        ("Drop identifiers\n→ 39 features", "#fdf0e8", ORANGE),
        ("Clean\ninf / NaN → 0", PALE, INK),
        ("Stratified split\n80 / 20, seed 42", "#e8f2f6", BLUE),
        ("Scale\n(non-tree only)", "#f4f0f8", "#6b4c7a"),
        ("Train", "#e8f0e8", GREEN),
    ]
    w, gap = 12, 1.6
    for i, (txt, fc, ec) in enumerate(steps):
        x = 2 + i * (w + gap)
        box(ax, x, 52, w, 16, txt, fc=fc, ec=ec, fs=7.4)
        if i < len(steps) - 1:
            arrow(ax, x + w, 60, x + w + gap, 60, lw=1.1)

    ax.plot([57, 57], [46, 72], color=RED, lw=1.4, ls="--")
    ax.text(57, 75, "leakage boundary", fontsize=8, color=RED, ha="center",
            fontweight="bold")
    ax.text(57, 42, "everything to the right is fitted on TRAINING data only",
            fontsize=8, color=RED, ha="center", va="top")

    box(ax, 4, 12, 92, 20, "", fc="#fff7f4", ec=RED, lw=1.2)
    ax.text(7, 28, "Why the identifier drop is non-negotiable",
            fontsize=9, fontweight="bold", color=RED)
    ax.text(7, 23,
            "A model that memorises a source port reports inflated accuracy, and the "
            "explainer then\nreports that port as the most important feature. "
            "The explanation is fluent and useless.\n"
            "The dhoogla release strips IPs and timestamps; this pipeline additionally "
            "drops both port columns.",
            fontsize=8, va="top", color=INK)
    save(fig, "fig_3_5_preprocessing_pipeline")


if __name__ == "__main__":
    print(f"writing {FIGURES}\n")
    for fn in (fig_1_1_trust_gap, fig_2_1_prior_art, fig_3_1_architecture,
               fig_3_2_rq1_protocol, fig_3_3_rq2_protocol,
               fig_3_4_experimental_design, fig_3_5_preprocessing):
        try:
            fn()
        except Exception as e:
            print(f"  FAILED {fn.__name__}: {type(e).__name__}: {e}")
    print(f"\n{len(made)} diagrams written")
