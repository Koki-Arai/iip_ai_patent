"""
11_subgroup_density.py
=======================

Section 5.7: Subgroup-level density and concentration.

The 4-digit IPC class-year analysis (Section 4.5) found null effects.
This script re-estimates at the granular 7-digit IPC subgroup (group1) level,
where an inverted-U pattern emerges (linear coef +1.136, squared coef -1.282,
turning point ≈ 0.443).

Also computes HHI and top-5 share at the (group1, year) level.

Outputs Table 10 and the data underlying Figure 3.

Per Section 5.7 in the paper, this is positioned as a granular analysis
that is consistent with prospect-like spillovers at low density and
anti-commons-like attenuation at high density, rather than a definitive
identification of either mechanism.
"""

import numpy as np
import pandas as pd
import pyfixest as pf

from config import (INTERIM_DIR, TABLES_DIR,
                    DENSITY_YEAR_MIN, DENSITY_YEAR_MAX,
                    CONCENTRATION_YEAR_MIN, CONCENTRATION_YEAR_MAX)

def main():
    panel = pd.read_parquet(INTERIM_DIR / "density_panel_group.parquet")

    # ---- Follow-on innovation, 7-digit subgroup ----
    p = panel[(panel["total_apps"] >= 5) &
              panel["app_year"].between(DENSITY_YEAR_MIN, DENSITY_YEAR_MAX)].copy()
    p = p.dropna(subset=["follow_on_apps_t1"])
    p["ai_density_sq"] = p["ai_core_density"] ** 2
    print(f"[11_subgroup_density] Subgroup-year cells: {len(p):,}")

    rows = []

    # Linear specification
    f_lin = pf.feols("log1p_follow_on_apps_t1 ~ ai_core_density + log_total_apps "
                     "| group1 + app_year",
                     data=p, vcov={"CRV1": "group1"})
    rows.append({
        "specification": "FollowOn (linear): AI density",
        "coef":  round(f_lin.coef()["ai_core_density"], 4),
        "se":    round(f_lin.se()["ai_core_density"], 4),
        "p":     round(f_lin.pvalue()["ai_core_density"], 4),
        "N":     f_lin.N,
    })
    print(f"   Linear:    β1 = {f_lin.coef()['ai_core_density']:+.4f} "
          f"(p = {f_lin.pvalue()['ai_core_density']:.4f})")

    # Quadratic specification (the inverted-U)
    f_quad = pf.feols("log1p_follow_on_apps_t1 ~ ai_core_density + ai_density_sq + "
                      "log_total_apps | group1 + app_year",
                      data=p, vcov={"CRV1": "group1"})
    b1 = f_quad.coef()["ai_core_density"]
    b2 = f_quad.coef()["ai_density_sq"]
    p1 = f_quad.pvalue()["ai_core_density"]
    p2 = f_quad.pvalue()["ai_density_sq"]
    rows.append({
        "specification": "FollowOn (quadratic): AI density",
        "coef": round(b1, 4),
        "se":   round(f_quad.se()["ai_core_density"], 4),
        "p":    round(p1, 4),
        "N":    f_quad.N,
    })
    rows.append({
        "specification": "FollowOn (quadratic): AI density²",
        "coef": round(b2, 4),
        "se":   round(f_quad.se()["ai_density_sq"], 4),
        "p":    round(p2, 4),
        "N":    f_quad.N,
    })
    if b2 < 0:
        turning_point = -b1 / (2 * b2)
        print(f"   Quadratic: β1 = {b1:+.4f} (p={p1:.4f}), "
              f"β2 = {b2:+.4f} (p={p2:.4f})")
        print(f"   Implied turning point: density* ≈ {turning_point:.4f}")

    # ---- HHI and Top-5 share, 7-digit subgroup ----
    conc = pd.read_parquet(INTERIM_DIR / "concentration_panel_group.parquet")
    # Merge density variable for regression
    density_lookup = panel[["group1", "app_year", "ai_core_density"]]
    conc = conc.merge(density_lookup, on=["group1", "app_year"], how="left")
    conc = conc.dropna(subset=["ai_core_density"])

    print("[11_subgroup_density] HHI and top-5 share...")
    f_hhi = pf.feols("hhi ~ ai_core_density | group1 + app_year",
                     data=conc, vcov={"CRV1": "group1"})
    rows.append({
        "specification": "HHI ~ AI density",
        "coef": round(f_hhi.coef()["ai_core_density"], 4),
        "se":   round(f_hhi.se()["ai_core_density"], 4),
        "p":    round(f_hhi.pvalue()["ai_core_density"], 4),
        "N":    f_hhi.N,
    })
    print(f"   HHI:        β = {f_hhi.coef()['ai_core_density']:+.4f} "
          f"(p = {f_hhi.pvalue()['ai_core_density']:.4f})")

    f_top5 = pf.feols("top5_share ~ ai_core_density | group1 + app_year",
                      data=conc, vcov={"CRV1": "group1"})
    rows.append({
        "specification": "Top-5 share ~ AI density",
        "coef": round(f_top5.coef()["ai_core_density"], 4),
        "se":   round(f_top5.se()["ai_core_density"], 4),
        "p":    round(f_top5.pvalue()["ai_core_density"], 4),
        "N":    f_top5.N,
    })
    print(f"   Top-5 share: β = {f_top5.coef()['ai_core_density']:+.4f} "
          f"(p = {f_top5.pvalue()['ai_core_density']:.4f})")

    out = pd.DataFrame(rows)
    out.to_csv(TABLES_DIR / "table10_subgroup_density.csv", index=False)

    # Also save for figure
    p[["group1", "app_year", "ai_core_density",
       "log1p_follow_on_apps_t1"]].to_csv(
        TABLES_DIR / "subgroup_density_data.csv", index=False)

    print("[11_subgroup_density] Saved table10_subgroup_density.csv")
    print(out.to_string(index=False))

if __name__ == "__main__":
    main()
