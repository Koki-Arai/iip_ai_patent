"""
06_density_followon.py
=======================

Section 4.5 (4-digit IPC class-year level): Density and follow-on innovation.

Estimates linear and quadratic specifications of follow-on innovation
on AI density at the 4-digit IPC class-year level.

In the paper this is reported in the body of Section 4.5 (no separate table);
the result is null (linear coef p=0.958, quadratic coef p=0.475).

The granular 7-digit subgroup analysis is in 11_subgroup_density.py.
"""

import numpy as np
import pandas as pd
import pyfixest as pf

from config import INTERIM_DIR, TABLES_DIR

def main():
    print("[06_density_followon] Loading 4-digit class-year panel...")
    p = pd.read_parquet(INTERIM_DIR / "density_panel_class.parquet")

    # Restrict to cells with ≥5 applications and 2000–2021 (pre/post deep learning)
    p = p[(p["total_apps"] >= 5) & p["app_year"].between(2000, 2021)].copy()
    p = p.dropna(subset=["follow_on_apps_t1"])
    p["ai_density_sq"] = p["ai_core_density"] ** 2

    print(f"   N (class-year cells): {len(p):,}")

    # Linear specification
    f_lin = pf.feols("log1p_follow_on_apps_t1 ~ ai_core_density + log_total_apps "
                     "| class1 + app_year",
                     data=p, vcov={"CRV1": "class1"})
    # Quadratic specification
    f_quad = pf.feols("log1p_follow_on_apps_t1 ~ ai_core_density + ai_density_sq + "
                      "log_total_apps | class1 + app_year",
                      data=p, vcov={"CRV1": "class1"})

    rows = []
    for spec_name, fit, vars_ in [
        ("linear",    f_lin,  ["ai_core_density"]),
        ("quadratic", f_quad, ["ai_core_density", "ai_density_sq"]),
    ]:
        for v in vars_:
            rows.append({
                "level": "class1 (4-digit)",
                "spec":  spec_name,
                "var":   v,
                "coef":  round(fit.coef()[v], 4),
                "se":    round(fit.se()[v], 4),
                "p":     round(fit.pvalue()[v], 4),
                "N":     fit.N,
            })

    out = pd.DataFrame(rows)
    out.to_csv(TABLES_DIR / "table_class_density.csv", index=False)
    print("[06_density_followon] Result (in-text in Section 4.5):")
    print(out.to_string(index=False))
    print("[06_density_followon] Saved table_class_density.csv")

if __name__ == "__main__":
    main()
