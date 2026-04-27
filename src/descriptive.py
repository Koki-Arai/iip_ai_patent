import pandas as pd
import matplotlib.pyplot as plt
from .config import DECADES, TABLE_DIR, FIGURE_DIR
from .utils import get_checkpoint, has_checkpoint, safe_to_csv, clean_memory

def create_descriptive_outputs():
    parts = []
    for dec in DECADES:
        name = f"analysis_dataset_{dec}.parquet"
        if not has_checkpoint(name): continue
        df = get_checkpoint(name)
        y = df.groupby("app_year").agg(
            total_apps=("ida","count"), ai_core_apps=("ai_core","sum"), ai_broad_apps=("ai_broad","sum"),
            grant_rate=("grant","mean"), avg_claim1=("claim1","mean"), avg_claim3=("claim3","mean")
        ).reset_index()
        parts.append(y)
        del df; clean_memory()
    yearly = pd.concat(parts).groupby("app_year", as_index=False).sum(numeric_only=True)
    yearly["ai_core_share"] = yearly["ai_core_apps"] / yearly["total_apps"]
    yearly["ai_broad_share"] = yearly["ai_broad_apps"] / yearly["total_apps"]
    safe_to_csv(yearly, TABLE_DIR / "yearly_ai_patent_trends.csv")
    plt.figure(figsize=(10,5))
    plt.plot(yearly["app_year"], yearly["ai_core_share"], marker="o", label="Core AI: G06N")
    plt.plot(yearly["app_year"], yearly["ai_broad_share"], marker="o", label="Broad AI")
    plt.legend(); plt.grid(True); plt.tight_layout()
    plt.savefig(FIGURE_DIR / "ai_share_trend.png", dpi=200); plt.close()
    plt.figure(figsize=(10,5))
    plt.plot(yearly["app_year"], yearly["avg_claim1"], marker="o", label="Claims at filing")
    plt.plot(yearly["app_year"], yearly["avg_claim3"], marker="o", label="Claims at grant")
    plt.legend(); plt.grid(True); plt.tight_layout()
    plt.savefig(FIGURE_DIR / "claims_trend.png", dpi=200); plt.close()
