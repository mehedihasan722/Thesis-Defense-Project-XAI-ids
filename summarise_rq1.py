"""
summarise_rq1.py — collect RQ1 across all models onto the smoothness axis.

    python summarise_rq1.py

RQ1 claims LIME stability tracks the smoothness of the model's probability
surface. Three tree models cannot test that claim, because they vary only
ensemble size within one family. This table places every model on the axis and
lets the claim be checked — or falsified.
"""

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "src"))

import pandas as pd
import config
from models import SURFACE_SMOOTHNESS

ORDER = ["GaussianNB", "LogisticRegression", "MLP",
         "XGBoost", "RandomForest", "DecisionTree"]

rows = []
for m in ORDER:
    f = config.RESULTS_TABLES / f"rq1_stability_{m}_multiclass.csv"
    if not f.exists():
        continue
    d = pd.read_csv(f)
    rows.append({
        "model": m,
        "surface": SURFACE_SMOOTHNESS.get(m, ""),
        "n": len(d),
        "jaccard@5": round(d["jaccard_at_5"].mean(), 4),
        "jaccard@10": round(d["jaccard_at_10"].mean(), 4),
        "kendall_tau": round(d["kendall_tau"].mean(), 4),
    })

if not rows:
    raise SystemExit("no rq1_stability_*.csv found — run RQ1 first")

t = pd.DataFrame(rows)
out = config.RESULTS_TABLES / "rq1_smoothness_axis.csv"
t.to_csv(out, index=False)

print("=" * 78)
print("RQ1 across the smoothness axis (smoothest first)")
print("=" * 78)
print(t.to_string(index=False))
print()

if len(t) >= 3:
    j = t["jaccard@5"].tolist()
    mono = all(j[i] >= j[i + 1] for i in range(len(j) - 1))
    print("Jaccard@5 decreases monotonically along the axis:",
          "YES — mechanism supported" if mono else "NO — see below")
    if not mono:
        for i in range(len(j) - 1):
            if j[i] < j[i + 1]:
                print(f"  inversion: {t['model'][i]} ({j[i]}) < "
                      f"{t['model'][i+1]} ({j[i+1]})")
        print("  The smoothness hypothesis does not hold as stated. Report the")
        print("  inversion honestly and revise the mechanism — a falsified")
        print("  prediction is a stronger result than an untested one.")
print(f"\nwrote {out}")
