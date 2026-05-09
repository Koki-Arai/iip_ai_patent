"""
08_did_2018.py
===============

Section 5.3: Difference-in-differences around the 2018 JPO AI guideline update.

Two-period DiD on a balanced 2014–2018 window:
    Y = α + δ1 AI + δ3 (AI × Post2018) + X'γ + class1 FE + year FE + ε

The coefficient of interest is δ3 (interaction).

Also runs an event-study specification with year dummies to plot
year-by-year coefficients (Figure 2).

Outputs Table 5 and event_study_master.csv (consumed by 99_generate_figures.py).

Note: Per Section 5.3 in the paper, this is positioned as a *supplementary*
institutional-timing check, not a primary identification strategy.
Pre-period coefficients are not all close to zero, so parallel trends are
only partially supported.
"""

import numpy as np
import pandas as pd
import pyfixest as pf

from config import INTERIM_DIR, TABLES_DIR

OUTCOMES = [
    ("grant",                    "Grant"),
    ("log1p_grant_delay_days",   "log1p_grant_delay_days"),
    ("log1p_backward_cites",     "log1p_backward_cites"),
    ("log1p_reject_cites",       "log1p_reject_cites"),
    ("has_reject_cite",          "has_reject_cite"),
    ("log1p_inventor_count",     "log1p_inventor_count"),
    ("log1p_applicant_count",    "log1p_applicant_count"),
    ("log1p_forward_cites",      "log1p_forward_cites"),
]

Z = ["log1p_claim1", "log1p_inventor_count", "log1p_applicant_count",
     "corporate", "domestic"]

def fit_did(df, outcome):
    Z_use = [z for z in Z if z != outcome]
    df = df.copy()
    df["ai_x_post"] = df["ai_core"] * df["post2018"]
    formula = (f"{outcome} ~ ai_core + ai_x_post + "
               + " + ".join(Z_use) + " | class1 + app_year")
    sample = df.copy()
    if outcome == "log1p_grant_delay_days":
        sample = sample[sample["grant"] == 1]
    return pf.feols(formula, data=sample, vcov={"CRV1": "class1"})

def fit_event_study(df, outcome, control_label="all"):
    """Event study with year dummies and 2014 as the omitted base."""
    df = df.copy()
    sample = df.copy()
    if outcome == "log1p_grant_delay_days":
        sample = sample[sample["grant"] == 1]

    rows = []
    base_year = 2014
    Z_use = [z for z in Z if z != outcome]
    for y in range(2010, 2019):
        if y == base_year:
            rows.append({
                "year": y, "estimate": 0.0,
                "std_error": np.nan, "p_value": np.nan,
                "ci_lo": np.nan, "ci_hi": np.nan,
                "is_base": True,
                "outcome": outcome,
                "control_group": control_label,
            })
            continue
        sample[f"ai_x_y{y}"] = ((sample["ai_core"] == 1) &
                                (sample["app_year"] == y)).astype(int)

    interaction_terms = " + ".join(
        [f"ai_x_y{y}" for y in range(2010, 2019) if y != base_year]
    )
    formula = (f"{outcome} ~ ai_core + {interaction_terms} + "
               + " + ".join(Z_use) + " | class1 + app_year")
    fit = pf.feols(formula, data=sample, vcov={"CRV1": "class1"})

    for y in range(2010, 2019):
        if y == base_year:
            continue
        v = f"ai_x_y{y}"
        rows.append({
            "year": y,
            "estimate":  fit.coef()[v],
            "std_error": fit.se()[v],
            "p_value":   fit.pvalue()[v],
            "ci_lo":     fit.confint()[v][0],
            "ci_hi":     fit.confint()[v][1],
            "is_base":   False,
            "outcome":   outcome,
            "control_group": control_label,
        })
    return pd.DataFrame(rows)

def main():
    df = pd.read_parquet(INTERIM_DIR / "main_sample.parquet")
    df["log1p_claim1"] = np.log1p(df["claim1"].fillna(0))
    df["post2018"] = (df["app_year"] >= 2018).astype(int)

    # ------------------------------------------------------------------
    # DiD coefficients (Table 5)
    # ------------------------------------------------------------------
    rows = []
    for var, label in OUTCOMES:
        fit = fit_did(df, var)
        b  = fit.coef()["ai_x_post"]
        se = fit.se()["ai_x_post"]
        p  = fit.pvalue()["ai_x_post"]
        rows.append({
            "outcome":   label,
            "ai_x_post": round(b, 4),
            "se":        round(se, 4),
            "p":         round(p, 4),
            "N":         fit.N,
        })
        print(f"   {label:30s}: δ3={b:+.4f} (p={p:.4f}, N={fit.N:,})")

    pd.DataFrame(rows).to_csv(TABLES_DIR / "table05_did_2018.csv", index=False)

    # ------------------------------------------------------------------
    # Event study (saved for Figure 2)
    # ------------------------------------------------------------------
    print("[08_did_2018] Running event studies...")
    parts = []
    for var, _ in [("grant", None), ("log1p_forward_cites", None)]:
        for control_label, sub_filter in [
            ("all", None),
            ("software", lambda d: (d["ai_core"] == 1) |
                                    d["class1"].isin(["G06F", "G06Q", "G06T", "G10L", "H04N"])),
        ]:
            sample = df if sub_filter is None else df[sub_filter(df)].copy()
            es = fit_event_study(sample, var, control_label)
            parts.append(es)
    es_master = pd.concat(parts, ignore_index=True)
    es_master.to_csv(TABLES_DIR / "event_studies_master.csv", index=False)
    print("[08_did_2018] Saved table05_did_2018.csv and event_studies_master.csv")

if __name__ == "__main__":
    main()
