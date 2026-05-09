"""
10_citation_decomp_applicant_fe.py
====================================

Section 5.6: Citation decomposition (reason codes, self/external) and
applicant fixed effects.

Outputs Table 8 (citation decomposition) and Table 9 (applicant FE).
"""

import numpy as np
import pandas as pd
import pyfixest as pf

from config import (INTERIM_DIR, TABLES_DIR, DATA_ROOT,
                    REJECT_REASON_CODES, POST_GRANT_REASON_CODES,
                    TRIAL_APPEAL_REASON_CODES)

Z = ["log1p_claim1", "log1p_inventor_count", "log1p_applicant_count",
     "corporate", "domestic"]

# ----------------------------------------------------------------------
# Table 8: Citation decomposition
# ----------------------------------------------------------------------
def build_decomposition(df, citation):
    """Aggregate citations by reason-code group and self/external source."""
    cit = citation.copy()
    cit["is_reject"] = cit["reason_code"].isin(REJECT_REASON_CODES).astype(int)
    cit["is_postgrant"] = cit["reason_code"].isin(POST_GRANT_REASON_CODES).astype(int)
    cit["is_trial"] = cit["reason_code"].isin(TRIAL_APPEAL_REASON_CODES).astype(int)

    # Per-citing-application aggregations by reason
    grp = cit.groupby("idapp_citing")
    out = pd.DataFrame({
        "refusal_related_cites":   grp["is_reject"].sum(),
        "post_grant_ref_cites":    grp["is_postgrant"].sum(),
        "trial_appeal_cites":      grp["is_trial"].sum(),
    }).reset_index().rename(columns={"idapp_citing": "idapp"})

    # Self vs external citations: join cited application's first applicant
    citing_app = df[["idapp", "first_applicant_id"]].rename(
        columns={"idapp": "idapp_citing", "first_applicant_id": "applicant_citing"})
    cited_app  = df[["idapp", "first_applicant_id"]].rename(
        columns={"idapp": "idapp_cited", "first_applicant_id": "applicant_cited"})

    cit_with = (cit.merge(citing_app, on="idapp_citing", how="left")
                   .merge(cited_app,  on="idapp_cited",  how="left"))
    cit_with["is_self"] = (
        cit_with["applicant_citing"] == cit_with["applicant_cited"]
    ).astype(int)
    grp = cit_with.groupby("idapp_citing")
    self_ext = pd.DataFrame({
        "self_cites":     grp["is_self"].sum(),
        "external_cites": (1 - grp["is_self"]).sum(),
    }).reset_index().rename(columns={"idapp_citing": "idapp"})

    out = out.merge(self_ext, on="idapp", how="left")
    for c in ["refusal_related_cites", "post_grant_ref_cites", "trial_appeal_cites",
              "self_cites", "external_cites"]:
        out[c] = out[c].fillna(0).astype(int)
        out[f"log1p_{c}"] = np.log1p(out[c])

    out["self_cite_share"] = out["self_cites"] / (
        out["self_cites"] + out["external_cites"]).replace(0, np.nan)
    return out

def fit_decomp(df, outcome):
    formula = f"{outcome} ~ ai_core + " + " + ".join(Z) + " | class1 + app_year"
    return pf.feols(formula, data=df, vcov={"CRV1": "class1"})

def table_8(df, citation):
    print("   Building citation decomposition variables...")
    decomp = build_decomposition(df, citation)
    df = df.merge(decomp, on="idapp", how="left")

    rows = []
    for label, var in [
        ("Reason: refusal-related",  "log1p_refusal_related_cites"),
        ("Reason: post-grant ref",   "log1p_post_grant_ref_cites"),
        ("Reason: trial/appeal",     "log1p_trial_appeal_cites"),
        ("Source: self-citations",   "log1p_self_cites"),
        ("Source: external citations","log1p_external_cites"),
        ("Source: self-cite share",  "self_cite_share"),
    ]:
        sample = df.copy()
        if var == "self_cite_share":
            sample = sample.dropna(subset=[var])
        fit = fit_decomp(sample, var)
        rows.append({
            "citation_type": label,
            "coef":  round(fit.coef()["ai_core"], 4),
            "se":    round(fit.se()["ai_core"], 4),
            "p":     round(fit.pvalue()["ai_core"], 4),
            "N":     fit.N,
        })
        print(f"   {label:30s}: β={fit.coef()['ai_core']:+.4f} "
              f"(p={fit.pvalue()['ai_core']:.4f})")
    return pd.DataFrame(rows)

# ----------------------------------------------------------------------
# Table 9: Applicant fixed effects
# ----------------------------------------------------------------------
def table_9(df):
    rows = []
    outcomes = [
        ("grant",                    "Grant"),
        ("log1p_grant_delay_days",   "log1p_grant_delay_days"),
        ("log1p_backward_cites",     "log1p_backward_cites"),
        ("log1p_reject_cites",       "log1p_reject_cites"),
        ("log1p_forward_cites",      "log1p_forward_cites"),
        ("log1p_inventor_count",     "log1p_inventor_count"),
        ("log1p_applicant_count",    "log1p_applicant_count"),
        ("has_reject_cite",          "has_reject_cite"),
    ]
    # Restrict to applicants with ≥2 filings to avoid singletons
    counts = df["first_applicant_id"].value_counts()
    keep_apps = set(counts[counts >= 2].index)
    sample = df[df["first_applicant_id"].isin(keep_apps)].copy()

    for var, label in outcomes:
        Z_use = [z for z in Z if z != var]
        s = sample.copy()
        if var == "log1p_grant_delay_days":
            s = s[s["grant"] == 1]

        # Baseline (class1 + app_year FE)
        f_base = pf.feols(f"{var} ~ ai_core + " + " + ".join(Z_use) +
                          " | class1 + app_year",
                          data=s, vcov={"CRV1": "class1"})
        # + applicant FE
        f_full = pf.feols(f"{var} ~ ai_core + " + " + ".join(Z_use) +
                          " | class1 + app_year + first_applicant_id",
                          data=s, vcov={"CRV1": "class1"})
        b_base, p_base = f_base.coef()["ai_core"], f_base.pvalue()["ai_core"]
        b_full, p_full = f_full.coef()["ai_core"], f_full.pvalue()["ai_core"]
        rows.append({
            "outcome":  label,
            "coef_baseline":      round(b_base, 4),
            "p_baseline":         round(p_base, 4),
            "coef_with_applicant_fe": round(b_full, 4),
            "p_with_applicant_fe":    round(p_full, 4),
            "delta": round(b_full - b_base, 4),
            "survives_5%":  p_full < 0.05,
        })
        print(f"   {label:30s}: base={b_base:+.4f} → with FE={b_full:+.4f}")
    return pd.DataFrame(rows)

# ----------------------------------------------------------------------
# Main
# ----------------------------------------------------------------------
def main():
    df = pd.read_parquet(INTERIM_DIR / "main_sample.parquet")
    df["log1p_claim1"] = np.log1p(df["claim1"].fillna(0))
    citation = pd.read_csv(DATA_ROOT / "citation.csv",
                           usecols=["idapp_citing", "idapp_cited", "reason_code"])

    print("[10_citation_decomp_applicant_fe] Table 8...")
    t8 = table_8(df, citation)
    t8.to_csv(TABLES_DIR / "table08_citation_decomp.csv", index=False)

    print("[10_citation_decomp_applicant_fe] Table 9 (applicant FE)...")
    t9 = table_9(df)
    t9.to_csv(TABLES_DIR / "table09_applicant_fe.csv", index=False)

    print("[10_citation_decomp_applicant_fe] Done.")

if __name__ == "__main__":
    main()
