"""
03_main_regressions.py
=======================

Section 4.2–4.3: Main application-level regressions.

Estimates 9 main outcomes regressed on AI_core indicator with
class1 + app_year fixed effects and SE clustered by class1.

Outputs Table 1.
"""

import numpy as np
import pandas as pd
import pyfixest as pf

from config import INTERIM_DIR, TABLES_DIR

# ----------------------------------------------------------------------
# Specifications
# ----------------------------------------------------------------------
# Standard control set Z (drop the corresponding control when the outcome
# itself is one of these counts/sizes)
Z_BASE = ["log1p_claim1", "log1p_inventor_count", "log1p_applicant_count",
          "corporate", "domestic"]

OUTCOMES = [
    # (variable, label, drop-from-Z)
    ("grant",                    "Grant",                              None),
    ("log1p_grant_delay_days",   "log1p_grant_delay_days",             None),
    ("claim_reduction_rate",     "ClaimReductionRate",                 "log1p_claim1"),
    ("log1p_backward_cites",     "log1p_backward_cites",               None),
    ("log1p_reject_cites",       "log1p_reject_cites",                 None),
    ("has_reject_cite",          "has_reject_cite",                    None),
    ("log1p_inventor_count",     "log1p_inventor_count",               "log1p_inventor_count"),
    ("log1p_applicant_count",    "log1p_applicant_count",              "log1p_applicant_count"),
    ("log1p_forward_cites",      "log1p_forward_cites",                None),
]

def run_main_regressions(df):
    rows = []
    for var, label, drop in OUTCOMES:
        Z = [z for z in Z_BASE if z != drop] if drop else Z_BASE
        formula = f"{var} ~ ai_core + " + " + ".join(Z) + " | class1 + app_year"

        # Some outcomes (claim_reduction_rate, grant_delay) are conditional on grant
        sample = df.copy()
        if var in ("claim_reduction_rate", "log1p_grant_delay_days"):
            sample = sample[sample["grant"] == 1].copy()

        # Drop rows with NaN in the outcome
        sample = sample.dropna(subset=[var] + Z)

        fit = pf.feols(formula, data=sample, vcov={"CRV1": "class1"})
        coef = fit.coef()["ai_core"]
        se   = fit.se()["ai_core"]
        pval = fit.pvalue()["ai_core"]
        n    = fit.N
        rows.append({
            "outcome": label,
            "coef":    round(coef, 4),
            "se":      round(se, 4),
            "p":       round(pval, 4),
            "N":       n,
            "sign":    "+" if coef > 0 else ("-" if coef < 0 else "0"),
        })
        print(f"   {label:30s}: β={coef:+.4f}  (SE={se:.4f}, p={pval:.4f}, N={n:,})")
    return pd.DataFrame(rows)

def main():
    print("[03_main_regressions] Loading main sample...")
    df = pd.read_parquet(INTERIM_DIR / "main_sample.parquet")
    df["log1p_claim1"] = np.log1p(df["claim1"].fillna(0))
    print(f"   {len(df):,} rows")

    print("[03_main_regressions] Running 9 main regressions...")
    out = run_main_regressions(df)
    out.to_csv(TABLES_DIR / "table01_main_results.csv", index=False)
    print("[03_main_regressions] Saved table01_main_results.csv")

if __name__ == "__main__":
    main()
