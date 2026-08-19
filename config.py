"""
config.py — central configuration for the thesis project.
Everything that could vary between runs lives here, so the methodology
chapter can be written directly from this file.
"""

from pathlib import Path

# --------------------------------------------------------------------------
# Reproducibility
# --------------------------------------------------------------------------
SEED = 42
SEEDS = [42, 7, 1337, 2024, 99]      # use SEEDS[:1] if short on time

# Windows/16 GB: use 4, not -1. Each worker copies the data.
N_JOBS = 4

# --------------------------------------------------------------------------
# Paths — resolved relative to this file, so the project is portable
# --------------------------------------------------------------------------
ROOT = Path(__file__).resolve().parent
DATA_RAW = ROOT / "data" / "raw"
DATA_PROCESSED = ROOT / "data" / "processed"
RESULTS = ROOT / "results"
RESULTS_TABLES = RESULTS / "tables"
RESULTS_FIGURES = RESULTS / "figures"
RESULTS_MODELS = RESULTS / "models"

for _p in (DATA_RAW, DATA_PROCESSED, RESULTS_TABLES, RESULTS_FIGURES, RESULTS_MODELS):
    _p.mkdir(parents=True, exist_ok=True)

# --------------------------------------------------------------------------
# Datasets
# --------------------------------------------------------------------------
DATASETS = {
    "unsw": {
        "kaggle": "dhoogla/nfunswnb15v2",
        "pattern": "*.parquet",
        "subsample": None,          # ~2.39M flows, fits in Colab
    },
    "ids2018": {
        "kaggle": "dhoogla/nfcsecicids2018v2",
        "pattern": "*.parquet",
        "subsample": 1_500_000,     # ~18.9M flows -> stratified subsample
    },
}

# --------------------------------------------------------------------------
# Identifier columns — NON-NEGOTIABLE removal (roadmap Section 04)
# A model that memorises an attacker IP reports inflated accuracy and
# produces explanations that cite the IP as the top feature.
# Matched case-insensitively against a normalised column name.
# --------------------------------------------------------------------------
IDENTIFIER_COLUMNS = {
    "flow_id", "ipv4_src_addr", "ipv4_dst_addr", "src_ip", "dst_ip",
    "source_ip", "destination_ip", "l4_src_port", "l4_dst_port",
    "src_port", "dst_port", "source_port", "destination_port",
    "timestamp", "flow_start_milliseconds", "flow_end_milliseconds",
    "start_time", "end_time", "date", "unnamed:_0",
}

LABEL_BINARY = "label"        # 0 = benign, 1 = attack
LABEL_MULTI = "attack"        # attack family name

# --------------------------------------------------------------------------
# Split
# --------------------------------------------------------------------------
TEST_SIZE = 0.2
