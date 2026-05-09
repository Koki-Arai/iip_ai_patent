"""
09_robustness.py
=================

Sections 5.4–5.5: Functional form, multi-way clustering, multiple-hypothesis
correction, and Oster (2019) sensitivity analysis.

Outputs Tables 6 (functional form) and 7 (Oster δ).

Section 5.8 multi-way clustering result is in-text only (not a separate table).
Section 5.9 multiple-hypothesis correction is also in-text.
"""

import numpy as np
import pandas as pd
import pyfixest as pf
import statsmodels.api as sm
from statsmodels.stats.multitest import multipletests

from config import INTERIM_DIR, TABLES_DIR, SOFTWARE_CLASSES

Z = ["log1p_claim1", "log1p_inventor_count", "log1p_applicant_count",
     "corporate", "domestic"]

# ----------------------------------------------------------------------
# Table 6: Functional form (LPM vs Probit, log1p OLS vs PPML)
# ----------------------------------------------------------------------
def table_functional_form(df):
    rows = []

    # Software-only sub-sample for Probit comparability
    is_software = (df["ai_core"] == 1) | df["class1"].isin(SOFTWARE_CLASSES)
    sw = df[is_software].copy()

    # Grant: LPM vs Probit
    print("   Grant LPM (software)...")
    f_lpm = pf.feols("grant ~ ai_core + " + " + ".join(Z) + " | class1 + app_year",
                     data=sw, vcov={"CRV1": "class1"})
    rows.append({"outcome": "Grant", "spec": "LPM, full FE",
                 "coef": round(f_lpm.coef()["ai_core"], 4),
                 "p":    round(f_lpm.pvalue()["ai_core"], 4), "sample": "software"})

    print("   Grant Probit APE (software)...")
    # Probit using statsmodels (may be slow on full software sample)
    sw_small = sw.dropna(subset=Z + ["grant"]).sample(n=min(200_000, len(sw)),
                                                       random_state=42)
    X = sm.add_constant(sw_small[["ai_core"] + Z])
    probit = sm.Probit(sw_small["grant"], X).fit(disp=0)
    # Average partial effect for ai_core
    ape = probit.get_margeff(at="overall").summary_frame().loc["ai_core", "dy/dx"]
    p_probit = probit.get_margeff(at="overall").pvalues[1]  # row 1 = ai_core
    rows.append({"outcome": "Grant", "spec": "Probit (APE)",
                 "coef": round(ape, 4), "p": round(p_probit, 4),
                 "sample": "software (200k)"})

    # HasRejectCite LPM and Probit (similar approach)
    print("   has_reject_cite LPM...")
    f_lpm = pf.feols("has_reject_cite ~ ai_core + " + " + ".join(Z) +
                     " | class1 + app_year",
                     data=sw, vcov={"CRV1": "class1"})
    rows.append({"outcome": "HasRejectCite", "spec": "LPM, full FE",
                 "coef": round(f_lpm.coef()["ai_core"], 4),
                 "p":    round(f_lpm.pvalue()["ai_core"], 4), "sample": "software"})

    print("   has_reject_cite Probit APE...")
    X = sm.add_constant(sw_small[["ai_core"] + Z])
    probit = sm.Probit(sw_small["has_reject_cite"], X).fit(disp=0)
    ape = probit.get_margeff(at="overall").summary_frame().loc["ai_core", "dy/dx"]
    p_probit = probit.get_margeff(at="overall").pvalues[1]
    rows.append({"outcome": "HasRejectCite", "spec": "Probit (APE)",
                 "coef": round(ape, 4), "p": round(p_probit, 4),
                 "sample": "software (200k)"})

    # log1p OLS vs Poisson PML for count outcomes
    for var, label in [("reject_cites",  "reject_cites"),
                       ("forward_cites", "forward_cites")]:
        print(f"   {label} log1p OLS...")
        f_log = pf.feols(f"log1p_{var} ~ ai_core + " + " + ".join(Z) +
                         " | class1 + app_year",
                         data=df, vcov={"CRV1": "class1"})
        rows.append({"outcome": label, "spec": "log1p OLS",
                     "coef": round(f_log.coef()["ai_core"], 4),
                     "p":    round(f_log.pvalue()["ai_core"], 4),
                     "sample": "full"})

        print(f"   {label} Poisson PML (slow)...")
        f_pml = pf.fepois(f"{var} ~ ai_core + " + " + ".join(Z) +
                          " | class1 + app_year",
                          data=df, vcov={"CRV1": "class1"})
        rows.append({"outcome": label, "spec": "Poisson PML",
                     "coef": round(f_pml.coef()["ai_core"], 4),
                     "p":    round(f_pml.pvalue()["ai_core"], 4),
                     "sample": "full"})

    return pd.DataFrame(rows)

# ----------------------------------------------------------------------
# Table 7: Oster δ
# ----------------------------------------------------------------------
def oster_delta(beta_full, R2_full, beta_restr, R2_restr, R_max):
    """Oster (2019) δ_zero (selection on unobservables relative to observables)."""
    if abs(beta_full - beta_restr) < 1e-9:
        return np.inf
    return ((beta_full - beta_restr) * (R_max - R2_full) /
            ((beta_restr - beta_full) * (R2_full - R2_restr))) * \
           (R2_full - R2_restr) / (R_max - R2_full)
    # Note: simplified; some implementations adopt different sign conventions.

def fit_pair(df, outcome):
    Z_use = [z for z in Z if z != outcome]
    sample = df.copy()
    if outcome == "log1p_grant_delay_days":
        sample = sample[sample["grant"] == 1]
    f_restr = pf.feols(f"{outcome} ~ ai_core | class1 + app_year",
                       data=sample, vcov={"CRV1": "class1"})
    f_full  = pf.feols(f"{outcome} ~ ai_core + " + " + ".join(Z_use) +
                       " | class1 + app_year",
                       data=sample, vcov={"CRV1": "class1"})
    return f_restr, f_full

def table_oster(df):
    rows = []
    outcomes = ["grant", "log1p_grant_delay_days",
                "log1p_backward_cites", "log1p_reject_cites",
                "log1p_forward_cites", "log1p_inventor_count",
                "log1p_applicant_count", "has_reject_cite"]
    for out in outcomes:
        print(f"   Oster δ for {out}...")
        f_restr, f_full = fit_pair(df, out)
        b_restr = f_restr.coef()["ai_core"]
        b_full  = f_full.coef()["ai_core"]
        # Need R² values; pyfixest exposes via fit.r2 or fit.r2_within
        try:
            r2_restr = f_restr.r2
            r2_full  = f_full.r2
        except AttributeError:
            r2_restr = getattr(f_restr, "_r2", np.nan)
            r2_full  = getattr(f_full,  "_r2", np.nan)
        r_max = min(1.0, 1.3 * r2_full) if not np.isnan(r2_full) else np.nan
        delta = oster_delta(b_full, r2_full, b_restr, r2_restr, r_max)
        rows.append({
            "outcome":      out,
            "beta_full":    round(b_full, 4),
            "r2_full":      round(r2_full, 4) if not np.isnan(r2_full) else np.nan,
            "abs_delta":    round(abs(delta), 2) if np.isfinite(delta) else np.inf,
            "robust_to_d>1": (abs(delta) > 1) if np.isfinite(delta) else True,
        })
    return pd.DataFrame(rows)

# ----------------------------------------------------------------------
# In-text: Multi-way clustering (Section 5.4) and multiple-hypothesis (5.4)
# ----------------------------------------------------------------------
def multiway_clustering_check(df):
    """Reports SE differences for one-way vs two-way clustering."""
    print("[09_robustness] Multi-way clustering check...")
    rows = []
    for var in ["grant", "log1p_forward_cites", "log1p_inventor_count"]:
        Z_use = [z for z in Z if z != var]
        formula = f"{var} ~ ai_core + " + " + ".join(Z_use) + " | class1 + app_year"
        f_one = pf.feols(formula, data=df, vcov={"CRV1": "class1"})
        f_two = pf.feols(formula, data=df,
                         vcov={"CRV3x1": ["class1", "app_year"]})
        rows.append({
            "outcome":  var,
            "coef":     round(f_one.coef()["ai_core"], 4),
            "se_1way":  round(f_one.se()["ai_core"], 4),
            "se_2way":  round(f_two.se()["ai_core"], 4),
        })
    out = pd.DataFrame(rows)
    out.to_csv(TABLES_DIR / "table_multiway_clustering.csv", index=False)
    return out

def multiple_hypothesis_correction(table1):
    """Apply Holm-Bonferroni and Benjamini-Hochberg to the 9 main outcomes."""
    pvals = table1["p"].values
    holm  = multipletests(pvals, alpha=0.05, method="holm")[1]
    bh    = multipletests(pvals, alpha=0.05, method="fdr_bh")[1]
    out = table1[["outcome", "p"]].copy()
    out["p_holm"] = holm.round(4)
    out["p_bh"]   = bh.round(4)
    out["sig_holm_5%"] = holm < 0.05
    out["sig_bh_5%"]   = bh < 0.05
    out.to_csv(TABLES_DIR / "table_multiple_hypothesis.csv", index=False)
    return out

# ----------------------------------------------------------------------
# Main
# ----------------------------------------------------------------------
def main():
    df = pd.read_parquet(INTERIM_DIR / "main_sample.parquet")
    df["log1p_claim1"] = np.log1p(df["claim1"].fillna(0))

    print("[09_robustness] Table 6: functional form...")
    t6 = table_functional_form(df)
    t6.to_csv(TABLES_DIR / "table06_functional_form.csv", index=False)

    print("[09_robustness] Table 7: Oster δ...")
    t7 = table_oster(df)
    t7.to_csv(TABLES_DIR / "table07_oster_delta.csv", index=False)

    print("[09_robustness] Multi-way clustering (in-text)...")
    multiway_clustering_check(df)

    print("[09_robustness] Multiple-hypothesis correction (in-text)...")
    t1 = pd.read_csv(TABLES_DIR / "table01_main_results.csv")
    multiple_hypothesis_correction(t1)

    print("[09_robustness] Done.")

if __name__ == "__main__":
    main()
