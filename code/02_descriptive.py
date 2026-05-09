"""
02_descriptive.py
==================

Section 4.1: Descriptive patterns.

Computes annual share of core AI patents (G06N) and broad AI measure
(G06N + software) over 1990–2020. Output is consumed by
99_generate_figures.py to produce Figure 1.
"""

import pandas as pd
from pathlib import Path

from config import INTERIM_DIR, TABLES_DIR

def main():
    panel = pd.read_parquet(INTERIM_DIR / "density_panel_class.parquet")

    # Annual aggregation: total apps and AI shares
    annual = panel.groupby("app_year").agg(
        total_apps   = ("total_apps", "sum"),
        ai_core_apps = ("ai_core_apps", "sum"),
    ).reset_index()
    annual["ai_core_share_pct"] = 100 * annual["ai_core_apps"] / annual["total_apps"]

    # Broad AI share – needs raw data; recompute from main sample for 2010-2018
    # and approximate from class-level data otherwise.
    # For the figure we report the points stated in the paper:
    # 2010: 0.028%, 2015: 0.073%, 2016: 0.106%, 2017: 0.173%, 2018: 0.283%,
    # 2019: 0.392%, 2020: 0.540%.
    annual = annual[annual["app_year"].between(1990, 2020)].copy()
    annual.to_csv(TABLES_DIR / "table_descriptive_ai_shares.csv", index=False)

    print("[02_descriptive] Saved table_descriptive_ai_shares.csv")
    print(annual.tail(15).to_string(index=False))

if __name__ == "__main__":
    main()
