import numpy as np, pandas as pd
from .config import ROBUST_MODEL_DIR, ROBUST_TABLE_DIR
from .utils import get_checkpoint, ensure_pyfixest, save_model_summary, extract_key_rows, safe_to_csv

def run_robustness(max_n=500000):
    ensure_pyfixest()
    import pyfixest as pf
    df = get_checkpoint("analysis_dataset_main_2010_2018.parquet")
    if max_n is not None and len(df) > max_n:
        df = df.sample(max_n, random_state=123).copy()
    df["ai_core_primary"] = df["class1"].astype(str).str.startswith("G06N").astype(int)
    df["ai_broad_noncore"] = ((df["ai_broad"]==1)&(df["ai_core"]==0)).astype(int)
    df["period_2015_2018"] = ((df["app_year"]>=2015)&(df["app_year"]<=2018)).astype(int)
    df["ai_core_x_2015_2018"] = df["ai_core"] * df["period_2015_2018"]
    controls = "np.log1p(claim1) + np.log1p(inventor_count) + np.log1p(applicant_count) + domestic_first_applicant + first_is_corporation"
    fe = "| class1 + app_year"
    def run(formula, data, fn):
        print("[run]", fn)
        m = pf.feols(formula, data=data, vcov={"CRV1":"class1"})
        save_model_summary(m, ROBUST_MODEL_DIR / fn)
    for v in ["ai_core","ai_core_primary","ai_broad","ai_broad_noncore"]:
        run(f"grant ~ {v} + {controls} {fe}", df, f"R1_grant_{v}.txt")
        run(f"np.log1p(backward_cites) ~ {v} + {controls} {fe}", df, f"R1_backward_{v}.txt")
        run(f"np.log1p(reject_cites) ~ {v} + {controls} {fe}", df, f"R1_reject_{v}.txt")
    soft = df[df["software_related"]==1]
    run(f"grant ~ ai_core + {controls} {fe}", soft, "R3_grant_software_subset.txt")
    run(f"np.log1p(backward_cites) ~ ai_core + {controls} {fe}", soft, "R3_backward_software_subset.txt")
    run(f"np.log1p(reject_cites) ~ ai_core + {controls} {fe}", soft, "R3_reject_software_subset.txt")
    granted = df[df["grant"]==1]
    run(f"np.log1p(backward_cites) ~ ai_core + {controls} {fe}", granted, "R4_backward_granted_only.txt")
    run(f"np.log1p(reject_cites) ~ ai_core + {controls} {fe}", granted, "R4_reject_granted_only.txt")
    run(f"grant ~ ai_core + period_2015_2018 + ai_core_x_2015_2018 + {controls} {fe}", df, "R5_grant_ai_interaction_2015_2018.txt")
    kt = get_checkpoint("class_year_panel.parquet")
    kt = kt[(kt["app_year"]>=2000)&(kt["app_year"]<=2021)&(kt["followon_apps_t1"].notna())&(kt["total_apps"]>=5)]
    run("log_followon_apps_t1 ~ ai_core_density + ai_core_density_sq + log_total_apps | class1 + app_year", kt, "R6_followon_core_density.txt")
    rows = []
    for f in sorted(ROBUST_MODEL_DIR.glob("*.txt")):
        rows.extend(extract_key_rows(f))
    safe_to_csv(pd.DataFrame(rows), ROBUST_TABLE_DIR / "robustness_summary.csv")
