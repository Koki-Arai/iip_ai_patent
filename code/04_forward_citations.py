"""
04_forward_citations.py
========================

Section 4.4: Forward citations as patent value.

Estimates AI_core effect on forward citations under multiple specifications:
- Full sample, log1p OLS
- Granted-only, log1p OLS
- Full sample, Poisson PML
- Has-forward-cite, LPM
- Period 2010–2014, log1p OLS
- Period 2015–2018, log1p OLS

Outputs Table 2.
"""

import numpy as np
import pandas as pd
import pyfixest as pf

from config import INTERIM_DIR, TABLES_DIR

Z = ["log1p_claim1", "log1p_inventor_count", "log1p_applicant_count",
     "corporate", "domestic"]

def fit_log_ols(df, fe="class1 + app_year"):
    formula = f"log1p_forward_cites ~ ai_core + " + " + ".join(Z) + f" | {fe}"
    return pf.feols(formula, data=df, vcov={"CRV1": "class1"})

def fit_lpm_has_forward(df):
    formula = f"has_forward_cite ~ ai_core + " + " + ".join(Z) + " | class1 + app_year"
    return pf.feols(formula, data=df, vcov={"CRV1": "class1"})

def fit_ppml(df):
    formula = f"forward_cites ~ ai_core + " + " + ".join(Z) + " | class1 + app_year"
    return pf.fepois(formula, data=df, vcov={"CRV1": "class1"})

def report(name, fit):
    coef = fit.coef()["ai_core"]
    se   = fit.se()["ai_core"]
    p    = fit.pvalue()["ai_core"]
    n    = fit.N
    return {"specification": name,
            "coef": round(coef, 4),
            "se":   round(se, 4),
            "p":    round(p, 4),
            "N":    n}

def main():
    print("[04_forward_citations] Loading main sample...")
    df = pd.read_parquet(INTERIM_DIR / "main_sample.parquet")
    df["log1p_claim1"] = np.log1p(df["claim1"].fillna(0))
    df = df.dropna(subset=Z)

    rows = []

    print("   Full sample, log1p OLS...")
    rows.append(report("Full sample, log1p OLS", fit_log_ols(df)))

    print("   Granted only, log1p OLS...")
    rows.append(report("Granted only, log1p OLS",
                       fit_log_ols(df[df["grant"] == 1])))

    print("   Full sample, Poisson PML (slow)...")
    rows.append(report("Full sample, Poisson PML", fit_ppml(df)))

    print("   Has-forward-cite, LPM...")
    rows.append(report("Has-forward-cite, LPM", fit_lpm_has_forward(df)))

    print("   Period 2010–2014, log1p OLS...")
    rows.append(report("Period 2010–2014, log1p OLS",
                       fit_log_ols(df[df["app_year"].between(2010, 2014)])))

    print("   Period 2015–2018, log1p OLS...")
    rows.append(report("Period 2015–2018, log1p OLS",
                       fit_log_ols(df[df["app_year"].between(2015, 2018)])))

    out = pd.DataFrame(rows)
    out.to_csv(TABLES_DIR / "table02_forward_citations.csv", index=False)
    print("[04_forward_citations] Saved table02_forward_citations.csv")
    print(out.to_string(index=False))

if __name__ == "__main__":
    main()
