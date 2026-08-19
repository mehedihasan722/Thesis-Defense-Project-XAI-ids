"""
verify.py — Stage 1 gate.

Run this before writing any modelling code:

    python verify.py

Nothing else proceeds until this prints a sensible attack rate and class list.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent / "src"))

import pandas as pd
import config
from data_loader import load_dataset, align_schemas


def main():
    unsw = load_dataset("unsw")

    print("\n" + "=" * 60)
    print("NF-UNSW-NB15-v2")
    print("=" * 60)
    print(f"flows       : {len(unsw):,}")
    print(f"features    : {unsw.shape[1] - 2}")

    print("\nfeature columns retained:")
    feats = [c for c in unsw.columns
             if c not in (config.LABEL_BINARY, config.LABEL_MULTI)]
    for i, c in enumerate(feats, 1):
        print(f"  {i:2d}. {c}")

    print("\nbinary label distribution:")
    print(unsw[config.LABEL_BINARY].value_counts(normalize=True).round(4))

    print("\nattack classes:")
    print(unsw[config.LABEL_MULTI].value_counts())

    attack_rate = unsw[config.LABEL_BINARY].mean()
    print(f"\nattack rate : {attack_rate:.2%}   (roadmap expects ~3.98%)")
    if abs(attack_rate - 0.0398) > 0.01:
        print("  NOTE: differs from the roadmap figure. The dhoogla release is")
        print("  deduplicated, so this is expected — cite YOUR number, not the")
        print("  roadmap's, and say why in the methodology chapter.")

    # ---- sanity: no identifier leaked through -----------------------------
    suspicious = [c for c in feats
                  if any(t in c for t in ("addr", "port", "ip", "time", "id"))]
    if suspicious:
        print(f"\n!! POSSIBLE IDENTIFIER LEAK: {suspicious}")
        print("   Check these against config.IDENTIFIER_COLUMNS before training.")
    else:
        print("\nidentifier check: clean")

    # ---- statistics table for the methodology chapter ---------------------
    stats = pd.DataFrame({
        "flows": [len(unsw)],
        "features": [len(feats)],
        "attack_rate": [f"{attack_rate:.2%}"],
        "attack_classes": [unsw[config.LABEL_MULTI].nunique()],
    }, index=["NF-UNSW-NB15-v2"])
    out = config.RESULTS_TABLES / "dataset_statistics.csv"
    stats.to_csv(out)
    print(f"\nwrote {out}")

    # ---- checkpoint -------------------------------------------------------
    ckpt = config.DATA_PROCESSED / "unsw_clean.parquet"
    unsw.to_parquet(ckpt)
    print(f"wrote {ckpt}  (load this from now on, never the raw files)")

    # ---- optional: RQ3 schema gate ----------------------------------------
    if (config.DATA_RAW / "ids2018").exists() and \
       any((config.DATA_RAW / "ids2018").glob("*.parquet")):
        print("\n" + "=" * 60)
        print("RQ3 schema alignment")
        print("=" * 60)
        ids = load_dataset("ids2018")
        _, _, shared = align_schemas(unsw, ids)
        print(f"\nRQ3 viable with {len(shared)} shared features.")
        if len(shared) < 30:
            print("WARNING: small intersection. RQ3 may not be defensible.")
    else:
        print("\nids2018 not downloaded — skipping RQ3 schema check.")


if __name__ == "__main__":
    main()
