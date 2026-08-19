"""
data_loader.py — returns identical-schema DataFrames for either dataset.

This is the single entry point for all data in the thesis. Every notebook
calls load_dataset(); nothing reads a parquet file directly. That is what
makes RQ3 (cross-network transfer) possible: both datasets come out of this
function with the same columns in the same order.
"""

from __future__ import annotations

import re
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

import sys
sys.path.append("..")
from config import (
    DATA_RAW, DATASETS, IDENTIFIER_COLUMNS,
    LABEL_BINARY, LABEL_MULTI, SEED, TEST_SIZE,
)


# --------------------------------------------------------------------------
# Column-name normalisation
# --------------------------------------------------------------------------
def normalise(name: str) -> str:
    """'IPV4_SRC_ADDR' -> 'ipv4_src_addr';  'Flow Duration' -> 'flow_duration'."""
    name = name.strip().lower()
    name = re.sub(r"[\s\-/]+", "_", name)
    name = re.sub(r"[^\w]", "", name)
    return name


def _read_raw(key: str) -> pd.DataFrame:
    spec = DATASETS[key]
    folder = DATA_RAW / key
    files = sorted(folder.glob(spec["pattern"]))
    if not files:
        raise FileNotFoundError(
            f"No files matching {spec['pattern']} in {folder}. "
            f"Download with: kaggle datasets download -d {spec['kaggle']}"
        )
    df = pd.concat((pd.read_parquet(f) for f in files), ignore_index=True)
    df.columns = [normalise(c) for c in df.columns]
    return df


# --------------------------------------------------------------------------
# Cleaning
# --------------------------------------------------------------------------
def drop_identifiers(df: pd.DataFrame, verbose: bool = True) -> pd.DataFrame:
    dropped = [c for c in df.columns if c in IDENTIFIER_COLUMNS]
    if verbose:
        print(f"  dropping {len(dropped)} identifier columns: {dropped}")
    return df.drop(columns=dropped)


def clean_numeric(df: pd.DataFrame) -> pd.DataFrame:
    """Replace inf with NaN, then NaN with 0. Report what was touched."""
    feat = df.drop(columns=[c for c in (LABEL_BINARY, LABEL_MULTI) if c in df])
    n_inf = int(np.isinf(feat.select_dtypes(include=[np.number])).sum().sum())
    n_nan = int(feat.isna().sum().sum())
    if n_inf or n_nan:
        print(f"  cleaning: {n_inf} inf, {n_nan} NaN -> 0")
    df = df.replace([np.inf, -np.inf], np.nan).fillna(0)
    return df


def stratified_subsample(df: pd.DataFrame, n: int, seed: int = SEED) -> pd.DataFrame:
    """Preserve class proportions exactly. Documented in methodology chapter."""
    if n is None or len(df) <= n:
        return df
    frac = n / len(df)
    out = (
        df.groupby(LABEL_MULTI, group_keys=False)
          .apply(lambda g: g.sample(max(1, int(round(len(g) * frac))), random_state=seed))
          .reset_index(drop=True)
    )
    print(f"  subsampled {len(df):,} -> {len(out):,} flows (stratified, seed={seed})")
    return out


# --------------------------------------------------------------------------
# Public API
# --------------------------------------------------------------------------
def load_dataset(key: str, subsample: int | None = "default",
                 seed: int = SEED) -> pd.DataFrame:
    """key is 'unsw' or 'ids2018'. Returns a cleaned, identifier-free frame."""
    print(f"[{key}] loading...")
    df = _read_raw(key)
    print(f"  raw shape: {df.shape}")
    df = drop_identifiers(df)
    df = clean_numeric(df)
    if subsample == "default":
        subsample = DATASETS[key]["subsample"]
    df = stratified_subsample(df, subsample, seed)
    print(f"  final shape: {df.shape}")
    return df


def align_schemas(df_a: pd.DataFrame, df_b: pd.DataFrame):
    """
    Reduce both frames to their shared feature columns, in identical order.
    RQ3 depends entirely on this. If the intersection is smaller than either
    input, that reduction MUST be documented in the methodology chapter.
    """
    labels = [c for c in (LABEL_BINARY, LABEL_MULTI) if c in df_a and c in df_b]
    fa = [c for c in df_a.columns if c not in labels]
    fb = [c for c in df_b.columns if c not in labels]
    shared = sorted(set(fa) & set(fb))

    print(f"schema alignment: A={len(fa)}  B={len(fb)}  shared={len(shared)}")
    if set(fa) - set(shared):
        print(f"  only in A: {sorted(set(fa) - set(shared))}")
    if set(fb) - set(shared):
        print(f"  only in B: {sorted(set(fb) - set(shared))}")

    cols = shared + labels
    return df_a[cols].copy(), df_b[cols].copy(), shared


def split_xy(df: pd.DataFrame, task: str = "binary", seed: int = SEED):
    """task='binary' -> label column; task='multiclass' -> attack column."""
    target = LABEL_BINARY if task == "binary" else LABEL_MULTI
    drop = [c for c in (LABEL_BINARY, LABEL_MULTI) if c in df]
    X = df.drop(columns=drop)
    y = df[target]
    return train_test_split(
        X, y, test_size=TEST_SIZE, random_state=seed, stratify=y
    )
