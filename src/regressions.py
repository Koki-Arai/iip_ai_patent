import numpy as np
from .config import MODEL_DIR
from .utils import get_checkpoint, ensure_pyfixest, save_model_summary

def prep(df):
    df = df.copy()
    df = df[df["class1"].notna()]
    for c in ["domestic_first_applicant","first_is_corporation","claim1","claim3","claim_reduction","claim_reduction_rate","inventor_count","applicant_count","grant","grant_delay_days","backward_cites","reject_cites","has_reject_cite"]:
        if c not in df.columns: df[c] = np.nan
        df[c] = df[c].fillna(0)
    return df

def run_main_regressions():
    ensure_pyfixest()
    import pyfixest as pf
    main = prep(get_checkpoint("analysis_dataset_main_2010_2018.parquet"))
    controls = "np.log1p(claim1) + np.log1p(inventor_count) + np.log1p(applicant_count) + domestic_first_applicant + first_is_corporation"
    fe = "| class1 + app_year"
    def run(formula, data, fn):
        print("[run]", fn)
        m = pf.feols(formula, data=data, vcov={"CRV1":"class1"})
        save_model_summary(m, MODEL_DIR / fn)
        print(m.summary())
    run(f"grant ~ ai_core + {controls} {fe}", main, "H1_grant_ai_core.txt")
    run(f"np.log1p(grant_delay_days) ~ ai_core + {controls} {fe}", main[(main['grant']==1)&(main['grant_delay_days']>=0)], "H1b_grant_delay_ai_core.txt")
    claims = main[(main["grant"]==1)&(main["claim1"]>0)&(main["claim_reduction_rate"]>-5)&(main["claim_reduction_rate"]<5)]
    run(f"claim_reduction ~ ai_core + {controls} {fe}", claims, "H2_claim_reduction_ai_core.txt")
    run(f"claim_reduction_rate ~ ai_core + {controls} {fe}", claims, "H2_claim_reduction_rate_ai_core.txt")
    run(f"np.log1p(backward_cites) ~ ai_core + {controls} {fe}", main, "H3_backward_cites_ai_core.txt")
    run(f"np.log1p(reject_cites) ~ ai_core + {controls} {fe}", main, "H4_reject_cites_ai_core.txt")
    run(f"has_reject_cite ~ ai_core + {controls} {fe}", main, "H4_has_reject_cite_ai_core.txt")
    run("np.log1p(inventor_count) ~ ai_core + np.log1p(claim1) + domestic_first_applicant + first_is_corporation | class1 + app_year", main, "H5_inventor_count_ai_core.txt")
    run(f"np.log1p(applicant_count) ~ ai_core + {controls} {fe}", main, "H5_applicant_count_ai_core.txt")
    kt = get_checkpoint("class_year_panel.parquet")
    kt = kt[(kt["app_year"]>=2000)&(kt["app_year"]<=2021)&(kt["followon_apps_t1"].notna())&(kt["total_apps"]>=5)]
    run("log_followon_apps_t1 ~ ai_core_density + ai_core_density_sq + log_total_apps | class1 + app_year", kt, "H6_followon_density_nonlinear.txt")
