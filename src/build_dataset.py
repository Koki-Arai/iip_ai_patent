import numpy as np
import pandas as pd
from .config import *
from .utils import *

def compact_numeric(df):
    for c in ["grant","exam_request","ai_core","ai_broad","software_related","applicant_count","inventor_count","backward_cites","forward_cites","reject_cites","grant_reason_cites","has_reject_cite","first_is_corporation","first_is_individual","first_is_government"]:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce").fillna(0)
            if df[c].max(skipna=True) < 2147483647:
                df[c] = df[c].astype("int32")
    for c in ["claim1","claim2","claim3","grant_delay_days","claim_reduction","claim_reduction_rate","domestic_first_applicant"]:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce").astype("float32")
    if "app_year" in df.columns:
        df["app_year"] = pd.to_numeric(df["app_year"], errors="coerce").astype("Int16")
    return df

def build_ap_decade(dec):
    ck = f"ap_{dec}.parquet"
    if has_checkpoint(ck) and not FORCE_REBUILD:
        return get_checkpoint(ck)
    p = file_path("ap", dec)
    df = read_iip_table(p, expected_cols=AP_COLS, usecols=AP_COLS)
    df["decade"] = dec
    for c in ["adate","sdate","rdate","tdate"]:
        df[c] = pd.to_datetime(df[c], errors="coerce")
    for c in ["claim1","claim2","claim3"]:
        df[c] = pd.to_numeric(df[c], errors="coerce")
    df["app_year"] = df["adate"].dt.year
    df["grant"] = (df["idr"].notna() & (df["idr"].astype(str).str.strip() != "")).astype(int)
    df["exam_request"] = df["sdate"].notna().astype(int)
    df["grant_delay_days"] = (df["rdate"] - df["adate"]).dt.days
    df["claim_reduction"] = df["claim1"] - df["claim3"]
    df["claim_reduction_rate"] = df["claim_reduction"] / df["claim1"]
    df.loc[df["claim1"] <= 0, "claim_reduction_rate"] = np.nan
    for c in ["class1","class2","group1","group2"]:
        df[c] = df[c].astype(str).str.strip()
        df.loc[df[c].str.lower().isin(["nan","none",""]), c] = np.nan
    df["ai_core"] = (df["class1"].apply(lambda x: starts_with_any(x, AI_CORE_CLASSES)) | df["class2"].apply(lambda x: starts_with_any(x, AI_CORE_CLASSES))).astype(int)
    df["ai_broad"] = (df["class1"].apply(lambda x: starts_with_any(x, AI_BROAD_CLASSES)) | df["class2"].apply(lambda x: starts_with_any(x, AI_BROAD_CLASSES))).astype(int)
    df["software_related"] = (df["class1"].apply(lambda x: starts_with_any(x, SOFTWARE_CLASSES)) | df["class2"].apply(lambda x: starts_with_any(x, SOFTWARE_CLASSES))).astype(int)
    put_checkpoint(compact_numeric(df), ck)
    return df

def build_applicant_agg_decade(dec):
    ck = f"applicant_agg_{dec}.parquet"
    if has_checkpoint(ck) and not FORCE_REBUILD:
        return get_checkpoint(ck)
    reader = read_iip_table(file_path("applicant", dec), expected_cols=APPLICANT_COLS, usecols=APPLICANT_COLS, chunksize=CHUNKSIZE)
    count_parts, first_parts = [], []
    for i, chunk in enumerate(reader):
        chunk["seq_num"] = pd.to_numeric(chunk["seq"], errors="coerce")
        chunk = chunk.sort_values(["ida","seq_num"])
        count_parts.append(chunk.groupby("ida").size().rename("applicant_count"))
        first_parts.append(chunk.groupby("ida").first()[["name","idname","country_pref","kohokan"]])
        if i % 10 == 0: print(" applicant chunk", i)
    counts = pd.concat(count_parts).groupby(level=0).sum()
    first = pd.concat(first_parts).reset_index().drop_duplicates("ida", keep="first").set_index("ida")
    first = first.rename(columns={"name":"first_applicant_name","idname":"first_applicant_id","country_pref":"first_country_pref","kohokan":"first_kohokan"})
    agg = pd.concat([counts, first], axis=1).reset_index()
    def is_domestic(x):
        if pd.isna(x): return np.nan
        x = normalize_digits(x)
        if x == "JP": return 1
        try:
            v = int(x); return 1 if 1 <= v <= 47 else 0
        except Exception: return 0
    agg["domestic_first_applicant"] = agg["first_country_pref"].apply(is_domestic)
    k = agg["first_kohokan"].astype(str).str.strip().apply(normalize_digits)
    agg["first_is_corporation"] = (k == "2").astype(int)
    agg["first_is_individual"] = (k == "1").astype(int)
    agg["first_is_government"] = (k == "3").astype(int)
    put_checkpoint(compact_numeric(agg), ck)
    return agg

def build_inventor_agg_decade(dec):
    ck = f"inventor_agg_{dec}.parquet"
    if has_checkpoint(ck) and not FORCE_REBUILD:
        return get_checkpoint(ck)
    parts = []
    reader = read_iip_table(file_path("inventor", dec), expected_cols=INVENTOR_COLS, usecols=INVENTOR_COLS, chunksize=CHUNKSIZE)
    for i, chunk in enumerate(reader):
        parts.append(chunk.groupby("ida").size().rename("inventor_count"))
        if i % 10 == 0: print(" inventor chunk", i)
    out = pd.concat(parts).groupby(level=0).sum().reset_index()
    put_checkpoint(compact_numeric(out), ck)
    return out

def build_cc_agg_decade(dec):
    ck = f"cc_agg_{dec}.parquet"
    if has_checkpoint(ck) and not FORCE_REBUILD:
        return get_checkpoint(ck)
    b_parts, f_parts, r_parts, g_parts = [], [], [], []
    reader = read_iip_table(file_path("cc", dec), expected_cols=CC_COLS, usecols=CC_COLS, chunksize=CHUNKSIZE)
    for i, chunk in enumerate(reader):
        chunk["reason"] = chunk["reason"].apply(normalize_digits)
        b_parts.append(chunk.groupby("citing").size().rename("backward_cites"))
        f_parts.append(chunk.groupby("cited").size().rename("forward_cites"))
        r_parts.append(chunk[chunk["reason"].isin(REJECT_REASON_CODES)].groupby("citing").size().rename("reject_cites"))
        g_parts.append(chunk[chunk["reason"].isin(GRANT_REASON_CODES)].groupby("citing").size().rename("grant_reason_cites"))
        if i % 5 == 0: print(" cc chunk", i)
    def combine(parts, name):
        if not parts: return pd.Series(dtype="float64", name=name)
        s = pd.concat(parts).groupby(level=0).sum(); s.name = name; return s
    out = pd.concat([combine(b_parts,"backward_cites"), combine(f_parts,"forward_cites"), combine(r_parts,"reject_cites"), combine(g_parts,"grant_reason_cites")], axis=1).fillna(0).reset_index().rename(columns={"index":"ida","citing":"ida","cited":"ida"})
    put_checkpoint(compact_numeric(out), ck)
    return out

def add_period_logs(df):
    def period(y):
        if pd.isna(y): return np.nan
        y = int(y)
        if 1990 <= y <= 1999: return "1990s"
        if 2000 <= y <= 2009: return "2000s"
        if 2010 <= y <= 2019: return "2010s"
        if 2020 <= y <= 2029: return "2020s"
        return np.nan
    df["period"] = df["app_year"].apply(period)
    for c in ["claim1","claim3","claim_reduction","backward_cites","forward_cites","reject_cites","inventor_count","applicant_count"]:
        if c in df.columns: df[f"log1p_{c}"] = np.log1p(pd.to_numeric(df[c], errors="coerce").fillna(0)).astype("float32")
    return df

def build_decade_analysis(dec):
    out_name = f"analysis_dataset_{dec}.parquet"
    if has_checkpoint(out_name) and not FORCE_REBUILD:
        return get_checkpoint(out_name)
    ap = build_ap_decade(dec)
    applicant = build_applicant_agg_decade(dec).drop_duplicates("ida", keep="first")
    inventor = build_inventor_agg_decade(dec).groupby("ida", as_index=False)["inventor_count"].sum()
    cc = build_cc_agg_decade(dec).groupby("ida", as_index=False)[["backward_cites","forward_cites","reject_cites","grant_reason_cites"]].sum()
    df = ap.merge(applicant, on="ida", how="left").merge(inventor, on="ida", how="left").merge(cc, on="ida", how="left")
    for c in ["applicant_count","inventor_count","backward_cites","forward_cites","reject_cites","grant_reason_cites"]:
        df[c] = pd.to_numeric(df[c], errors="coerce").fillna(0)
    df["has_reject_cite"] = (df["reject_cites"] > 0).astype(int)
    df = compact_numeric(add_period_logs(df))
    put_checkpoint(df, out_name)
    return df

def build_panel_piece(df):
    panel = df[df["app_year"].notna()].copy()
    panel = panel[panel["class1"].notna()]
    kt = panel.groupby(["class1","app_year"]).agg(
        total_apps=("ida","count"), ai_core_apps=("ai_core","sum"), ai_broad_apps=("ai_broad","sum"), software_apps=("software_related","sum"),
        grant_rate=("grant","mean"), avg_claim1=("claim1","mean"), avg_claim3=("claim3","mean"), avg_claim_reduction=("claim_reduction","mean"),
        avg_backward=("backward_cites","mean"), avg_forward=("forward_cites","mean"), avg_reject=("reject_cites","mean"), avg_inventors=("inventor_count","mean"), avg_applicants=("applicant_count","mean")
    ).reset_index()
    return kt

def build_all_memory_safe():
    print_file_status(DECADES)
    panel_parts, main_parts, sw_parts = [], [], []
    for dec in DECADES:
        df = build_decade_analysis(dec)
        panel_parts.append(build_panel_piece(df))
        main = df[(df["app_year"] >= 2010) & (df["app_year"] <= 2018)].copy()
        if len(main): main_parts.append(main)
        sw = df[(df["app_year"] >= 1998) & (df["app_year"] <= 2008)].copy()
        if len(sw): sw_parts.append(sw)
        del df; clean_memory()
    if main_parts:
        put_checkpoint(compact_numeric(pd.concat(main_parts, ignore_index=True)), "analysis_dataset_main_2010_2018.parquet")
    if sw_parts:
        put_checkpoint(compact_numeric(pd.concat(sw_parts, ignore_index=True)), "analysis_dataset_1998_2008.parquet")
    kt = pd.concat(panel_parts, ignore_index=True).groupby(["class1","app_year"], as_index=False).sum(numeric_only=True)
    kt["ai_core_density"] = kt["ai_core_apps"] / kt["total_apps"]
    kt["ai_broad_density"] = kt["ai_broad_apps"] / kt["total_apps"]
    kt = kt.sort_values(["class1","app_year"])
    kt["followon_apps_t1"] = kt.groupby("class1")["total_apps"].shift(-1)
    kt["ai_core_density_sq"] = kt["ai_core_density"] ** 2
    kt["ai_broad_density_sq"] = kt["ai_broad_density"] ** 2
    kt["log_total_apps"] = np.log1p(kt["total_apps"])
    kt["log_followon_apps_t1"] = np.log1p(kt["followon_apps_t1"])
    put_checkpoint(kt, "class_year_panel.parquet")
    safe_to_csv(kt, TABLE_DIR / "class_year_panel.csv")
