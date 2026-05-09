"""
05_heterogeneity.py
====================

Section 4.5: Heterogeneity by applicant type (sub-sample estimates).

Estimates AI_core effect on three outcomes (grant, log1p_forward_cites,
log1p_inventor_count) for four sub-samples (corporate, individual, domestic,
foreign).

Outputs Table 3.

Also reports Top-50 interaction (mentioned in Section 4.5 text):
    AI_j × Top50_j on log1p_forward_cites.
"""

import numpy as np
import pandas as pd
import pyfixest as pf

from config import INTERIM_DIR, TABLES_DIR

Z = ["log1p_claim1", "log1p_inventor_count", "log1p_applicant_count",
     "corporate", "domestic"]

OUTCOMES = ["grant", "log1p_forward_cites", "log1p_inventor_count"]

SUBSAMPLES = [
    ("corporate",   lambda df: df["corporate"] == 1),
    ("individual",  lambda df: df["corporate"] == 0),
    ("domestic",    lambda df: df["domestic"]  == 1),
    ("foreign",     lambda df: df["domestic"]  == 0),
]

def fit_one(df, outcome):
    Z_use = [z for z in Z if z != outcome.replace("log1p_", "")]
    Z_use = [z for z in Z_use if z != outcome]
    if outcome == "log1p_inventor_count":
        Z_use = [z for z in Z if z != "log1p_inventor_count"]
    formula = f"{outcome} ~ ai_core + " + " + ".join(Z_use) + " | class1 + app_year"
    fit = pf.feols(formula, data=df, vcov={"CRV1": "class1"})
    return fit.coef()["ai_core"], fit.se()["ai_core"], fit.pvalue()["ai_core"], fit.N

def main():
    print("[05_heterogeneity] Loading main sample...")
    df = pd.read_parquet(INTERIM_DIR / "main_sample.parquet")
    df["log1p_claim1"] = np.log1p(df["claim1"].fillna(0))

    rows = []
    for sub_name, sub_filter in SUBSAMPLES:
        sub = df[sub_filter(df)].copy()
        for out in OUTCOMES:
            beta, se, p, n = fit_one(sub, out)
            rows.append({
                "subsample": sub_name,
                "outcome":   out,
                "coef":      round(beta, 4),
                "se":        round(se, 4),
                "p":         round(p, 4),
                "N":         n,
            })
            print(f"   {sub_name:12s} × {out:25s}: β={beta:+.4f} (p={p:.4f}, N={n:,})")

    out = pd.DataFrame(rows)
    out.to_csv(TABLES_DIR / "table03_heterogeneity.csv", index=False)

    # Top-50 interaction (in-text)
    print("[05_heterogeneity] Top-50 × AI interaction on forward citations...")
    df["ai_x_top50"] = df["ai_core"] * df["top50"]
    formula = ("log1p_forward_cites ~ ai_core + top50 + ai_x_top50 + "
               + " + ".join(Z) + " | class1 + app_year")
    fit = pf.feols(formula, data=df, vcov={"CRV1": "class1"})
    print(f"   AI × Top50: β={fit.coef()['ai_x_top50']:+.4f} "
          f"(p={fit.pvalue()['ai_x_top50']:.4f})")
    interaction_row = pd.DataFrame([{
        "subsample": "interaction (AI × Top50)",
        "outcome":   "log1p_forward_cites",
        "coef":      round(fit.coef()["ai_x_top50"], 4),
        "se":        round(fit.se()["ai_x_top50"], 4),
        "p":         round(fit.pvalue()["ai_x_top50"], 4),
        "N":         fit.N,
    }])
    interaction_row.to_csv(TABLES_DIR / "table03_top50_interaction.csv", index=False)

    print("[05_heterogeneity] Saved table03_heterogeneity.csv")

if __name__ == "__main__":
    main()
