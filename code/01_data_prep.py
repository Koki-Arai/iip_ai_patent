"""
01_data_prep.py
================

Section 3.1–3.2: Data preparation.

Loads the IIP Patent Database tables, constructs all variables described
in docs/variable_dictionary.md, and saves a cached parquet file at
interim/main_sample.parquet that is consumed by all subsequent scripts.

Inputs (from DATA_ROOT):
    application.csv
    applicant.csv
    inventor.csv
    citation.csv

Outputs:
    interim/main_sample.parquet         # 2010–2018, ~2.6M rows
    interim/granted_sample.parquet      # granted only, ~1.66M rows
    interim/density_panel_class.parquet # 4-digit class-year panel
    interim/density_panel_group.parquet # 7-digit subgroup-year panel
"""

import numpy as np
import pandas as pd
from pathlib import Path

from config import (
    DATA_ROOT, INTERIM_DIR,
    SAMPLE_YEAR_MIN, SAMPLE_YEAR_MAX,
    G06N, SOFTWARE_CLASSES, BROAD_AI_CLASSES,
    REJECT_REASON_CODES, POST_GRANT_REASON_CODES, TRIAL_APPEAL_REASON_CODES,
)

# ----------------------------------------------------------------------
# 1. Load core tables
# ----------------------------------------------------------------------
def load_application():
    """Load the IIP application table with relevant columns."""
    cols = ["idapp", "idr", "adate", "rdate",
            "class1", "class2", "class3",
            "group1", "claim1", "claim3"]
    df = pd.read_csv(DATA_ROOT / "application.csv", usecols=cols,
                     parse_dates=["adate", "rdate"], low_memory=False)
    df["app_year"] = df["adate"].dt.year
    return df

def load_applicant():
    cols = ["idapp", "idname", "kojin_kbn", "country_jp", "applicant_order"]
    df = pd.read_csv(DATA_ROOT / "applicant.csv", usecols=cols, low_memory=False)
    return df

def load_inventor():
    cols = ["idapp", "inventor_order"]
    df = pd.read_csv(DATA_ROOT / "inventor.csv", usecols=cols, low_memory=False)
    return df

def load_citation():
    """Patent-document citations with reason codes."""
    cols = ["idapp_citing", "idapp_cited", "reason_code"]
    df = pd.read_csv(DATA_ROOT / "citation.csv", usecols=cols, low_memory=False)
    return df

# ----------------------------------------------------------------------
# 2. Construct variables
# ----------------------------------------------------------------------
def construct_ai_indicators(app):
    """Define the four AI measures."""
    classes = app[["class1", "class2", "class3"]].astype(str)
    app["ai_core"] = (
        (app["class1"] == G06N) | (app["class2"] == G06N)
    ).astype(int)
    app["ai_core_primary"] = (app["class1"] == G06N).astype(int)
    is_broad = classes.isin(BROAD_AI_CLASSES).any(axis=1)
    app["ai_broad"] = is_broad.astype(int)
    app["ai_broad_noncore"] = ((app["ai_broad"] == 1) & (app["ai_core"] == 0)).astype(int)
    return app

def construct_outcomes(app, citation):
    """Construct grant, delay, claim reduction, citation, and team variables."""
    # Grant outcome
    app["grant"] = app["idr"].notna().astype(int)
    app["grant_delay_days"] = (app["rdate"] - app["adate"]).dt.days
    app["log1p_grant_delay_days"] = np.log1p(app["grant_delay_days"].clip(lower=0))

    # Claim reduction
    app["claim_reduction"] = app["claim1"] - app["claim3"]
    app["claim_reduction_rate"] = (
        app["claim_reduction"] / app["claim1"].replace(0, np.nan)
    ).clip(0, 1)

    # Citation aggregations
    cit = citation.copy()
    cit["is_reject"] = cit["reason_code"].isin(REJECT_REASON_CODES).astype(int)

    bw = cit.groupby("idapp_citing").size().rename("backward_cites")
    rj = cit[cit["is_reject"] == 1].groupby("idapp_citing").size().rename("reject_cites")
    fw = cit.groupby("idapp_cited").size().rename("forward_cites")

    app = app.set_index("idapp").join([bw, rj, fw]).reset_index()
    for col in ["backward_cites", "reject_cites", "forward_cites"]:
        app[col] = app[col].fillna(0).astype(int)
        app[f"log1p_{col}"] = np.log1p(app[col])
    app["has_reject_cite"] = (app["reject_cites"] > 0).astype(int)
    app["has_forward_cite"] = (app["forward_cites"] > 0).astype(int)
    return app

def construct_team_size(app, applicant, inventor):
    """Inventor and applicant counts."""
    inv_n = inventor.groupby("idapp").size().rename("inventor_count")
    app_n = applicant.groupby("idapp").size().rename("applicant_count")
    app = app.set_index("idapp").join([inv_n, app_n]).reset_index()
    for col in ["inventor_count", "applicant_count"]:
        app[col] = app[col].fillna(0).astype(int)
        app[f"log1p_{col}"] = np.log1p(app[col])
    return app

def construct_applicant_attrs(app, applicant):
    """First-applicant attributes: corporate, domestic, applicant id, top-50 indicator."""
    first = applicant[applicant["applicant_order"] == 1].copy()
    first = first.rename(columns={
        "idname": "first_applicant_id",
        "kojin_kbn": "_kojin",
        "country_jp": "_country",
    })[["idapp", "first_applicant_id", "_kojin", "_country"]]
    app = app.merge(first, on="idapp", how="left")

    # corporate: kojin_kbn != 1 (1 = individual in IIP convention)
    app["corporate"] = (app["_kojin"].fillna(2) != 1).astype(int)
    # domestic: country_jp == "JP"
    app["domestic"] = (app["_country"].astype(str).str.upper() == "JP").astype(int)
    app = app.drop(columns=["_kojin", "_country"])

    # Top-50 applicants by 2010–2018 filing volume
    is_main = app["app_year"].between(SAMPLE_YEAR_MIN, SAMPLE_YEAR_MAX)
    counts = app.loc[is_main, "first_applicant_id"].value_counts()
    top50_ids = set(counts.head(50).index)
    app["top50"] = app["first_applicant_id"].isin(top50_ids).astype(int)
    return app

# ----------------------------------------------------------------------
# 3. Field-year density panels
# ----------------------------------------------------------------------
def build_density_panel(app, key="class1"):
    """
    Aggregate to (field, year) cells.
    `key` is either 'class1' (4-digit) or 'group1' (7-digit).
    """
    g = app.groupby([key, "app_year"]).agg(
        total_apps     = ("idapp", "count"),
        ai_core_apps   = ("ai_core", "sum"),
    ).reset_index()
    g["ai_core_density"] = g["ai_core_apps"] / g["total_apps"]
    g["log_total_apps"]  = np.log1p(g["total_apps"])

    # Lead 1-year follow-on
    g_next = g[[key, "app_year", "total_apps"]].copy()
    g_next["app_year"] = g_next["app_year"] - 1
    g_next = g_next.rename(columns={"total_apps": "follow_on_apps_t1"})
    g = g.merge(g_next, on=[key, "app_year"], how="left")
    g["log1p_follow_on_apps_t1"] = np.log1p(g["follow_on_apps_t1"].fillna(0))
    return g

def build_concentration_panel(app, key="group1"):
    """HHI and top-5 share at the (field, year) level."""
    counts = (app.groupby([key, "app_year", "first_applicant_id"]).size()
                  .reset_index(name="n"))
    totals = counts.groupby([key, "app_year"])["n"].transform("sum")
    counts["share"] = counts["n"] / totals
    counts["share_sq"] = counts["share"] ** 2
    rows = []
    for (k, y), grp in counts.groupby([key, "app_year"]):
        s = grp["share"].sort_values(ascending=False)
        rows.append({
            key: k, "app_year": y,
            "hhi": (s ** 2).sum(),
            "top5_share": s.head(5).sum(),
            "n_apps_total": grp["n"].sum(),
        })
    return pd.DataFrame(rows)

# ----------------------------------------------------------------------
# 4. Main pipeline
# ----------------------------------------------------------------------
def main():
    print("[01_data_prep] Loading application table...")
    app = load_application()
    print(f"   {len(app):,} rows")

    print("[01_data_prep] Loading applicant, inventor, citation tables...")
    applicant = load_applicant()
    inventor  = load_inventor()
    citation  = load_citation()

    print("[01_data_prep] Constructing AI indicators...")
    app = construct_ai_indicators(app)

    print("[01_data_prep] Constructing outcome variables...")
    app = construct_outcomes(app, citation)
    app = construct_team_size(app, applicant, inventor)
    app = construct_applicant_attrs(app, applicant)

    # Filter to main sample
    main_sample = app[app["app_year"].between(SAMPLE_YEAR_MIN, SAMPLE_YEAR_MAX)].copy()
    print(f"[01_data_prep] Main sample (2010–2018): {len(main_sample):,} applications")
    main_sample.to_parquet(INTERIM_DIR / "main_sample.parquet", index=False)

    granted = main_sample[main_sample["grant"] == 1].copy()
    granted.to_parquet(INTERIM_DIR / "granted_sample.parquet", index=False)
    print(f"[01_data_prep] Granted sub-sample: {len(granted):,} applications")

    # Density panels (use ALL applications in the IIP, not just 2010-2018)
    print("[01_data_prep] Building density panels...")
    panel_class = build_density_panel(app, key="class1")
    panel_class.to_parquet(INTERIM_DIR / "density_panel_class.parquet", index=False)
    print(f"   4-digit class-year cells: {len(panel_class):,}")

    panel_group = build_density_panel(app, key="group1")
    panel_group.to_parquet(INTERIM_DIR / "density_panel_group.parquet", index=False)
    print(f"   7-digit subgroup-year cells: {len(panel_group):,}")

    print("[01_data_prep] Building concentration panel...")
    conc = build_concentration_panel(app, key="group1")
    conc.to_parquet(INTERIM_DIR / "concentration_panel_group.parquet", index=False)
    print(f"   Concentration cells: {len(conc):,}")

    print("[01_data_prep] Done.")

if __name__ == "__main__":
    main()
