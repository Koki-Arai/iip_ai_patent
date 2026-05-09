"""
Configuration for the replication package.

Modify DATA_ROOT to point to your local copy of the IIP Patent Database.
The default assumes Google Colab with Drive mounted.
"""

import os
from pathlib import Path

# ----------------------------------------------------------------------
# Paths
# ----------------------------------------------------------------------
# Override via environment variable: IIP_DATA_ROOT=/path/to/data python script.py
DATA_ROOT = Path(os.environ.get(
    "IIP_DATA_ROOT",
    "/content/drive/MyDrive/iip_patent_ai/data/"
))

# Repository paths
REPO_ROOT    = Path(__file__).resolve().parent.parent
RESULTS_DIR  = REPO_ROOT / "results"
TABLES_DIR   = RESULTS_DIR / "tables"
FIGURES_DIR  = RESULTS_DIR / "figures"
INTERIM_DIR  = REPO_ROOT / "interim"  # cached intermediate parquet files

for d in (TABLES_DIR, FIGURES_DIR, INTERIM_DIR):
    d.mkdir(parents=True, exist_ok=True)

# ----------------------------------------------------------------------
# Sample restrictions
# ----------------------------------------------------------------------
SAMPLE_YEAR_MIN = 2010
SAMPLE_YEAR_MAX = 2018

DENSITY_YEAR_MIN = 2000
DENSITY_YEAR_MAX = 2021

CONCENTRATION_YEAR_MIN = 1990
CONCENTRATION_YEAR_MAX = 2023

# ----------------------------------------------------------------------
# AI definitions
# ----------------------------------------------------------------------
G06N = "G06N"
SOFTWARE_CLASSES  = ["G06F", "G06Q", "G06T", "G10L", "H04N"]
BROAD_AI_CLASSES  = [G06N] + SOFTWARE_CLASSES

# ----------------------------------------------------------------------
# Citation reason codes (IIP convention)
# ----------------------------------------------------------------------
REJECT_REASON_CODES      = [19, 89]
POST_GRANT_REASON_CODES  = [31]
TRIAL_APPEAL_REASON_CODES = [21, 23, 24, 25, 26]   # confirm with IIP documentation

# ----------------------------------------------------------------------
# Style
# ----------------------------------------------------------------------
PLOT_NAVY = "#1f3a5f"
PLOT_GRAY = "#7f7f7f"

print(f"[config] DATA_ROOT    = {DATA_ROOT}")
print(f"[config] RESULTS_DIR  = {RESULTS_DIR}")
print(f"[config] Sample years: {SAMPLE_YEAR_MIN}–{SAMPLE_YEAR_MAX}")
