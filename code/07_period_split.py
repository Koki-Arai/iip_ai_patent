"""
07_period_split.py
===================

Section 5.2: Pre-/post-AI-boom period split.

Estimates each baseline regression separately for 2010–2014 and 2015–2018
to check temporal stability.

Outputs Table 4.
"""

import numpy as np
import pandas as pd
import pyfixest as pf

from config import INTERIM_DIR, TABLES_DIR

OUTCOMES = ["grant", "log1p_grant_delay_days",
            "log1p_backward_cites", "log1p_reject_cites", "log1p_forward_cites",
            "log1p_inventor_count", "log1p_applicant_count"]

def Z_for(outcome):
    base = ["log1p_claim1", "log1p_inventor_count", "log1p_applicant_count",
            "corporate", "domestic"]
    if outcome in base:
        base = [z for z in base if z != outcome]
    return base

def fit(df, outcome):
    Z = Z_for(outcome)
    formula = f"{outcome} ~ ai_core + " + " + ".join(Z) + " | class1 + app_year"
    return pf.feols(formula, data=df, vcov={"CRV1": "class1"})

def main():
    df = pd.read_parquet(INTERIM_DIR / "main_sample.parquet")
    df["log1p_claim1"] = np.log1p(df["claim1"].fillna(0))

    rows = []
    for out in OUTCOMES:
        sub_pre  = df[df["app_year"].between(2010, 2014)].copy()
        sub_post = df[df["app_year"].between(2015, 2018)].copy()
        if out == "log1p_grant_delay_days":
            sub_pre  = sub_pre[sub_pre["grant"] == 1]
            sub_post = sub_post[sub_post["grant"] == 1]
        f_pre  = fit(sub_pre,  out)
        f_post = fit(sub_post, out)
        b_pre,  p_pre  = f_pre.coef()["ai_core"],  f_pre.pvalue()["ai_core"]
        b_post, p_post = f_post.coef()["ai_core"], f_post.pvalue()["ai_core"]
        rows.append({
            "outcome":      out,
            "coef_2010_14": round(b_pre,  4),
            "p_2010_14":    round(p_pre,  4),
            "coef_2015_18": round(b_post, 4),
            "p_2015_18":    round(p_post, 4),
            "delta_post_pre": round(b_post - b_pre, 4),
            "both_p<0.05":  (p_pre < 0.05) and (p_post < 0.05),
            "N_pre":        f_pre.N,
            "N_post":       f_post.N,
        })
        print(f"   {out:30s}: pre={b_pre:+.4f} (p={p_pre:.3f})  "
              f"post={b_post:+.4f} (p={p_post:.3f})")

    out = pd.DataFrame(rows)
    out.to_csv(TABLES_DIR / "table04_period_split.csv", index=False)
    print("[07_period_split] Saved table04_period_split.csv")

if __name__ == "__main__":
    main()
