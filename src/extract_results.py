import pandas as pd
from .config import MODEL_DIR, TABLE_DIR, ROBUST_MODEL_DIR, ROBUST_TABLE_DIR
from .utils import extract_key_rows, safe_to_csv

def extract_main_results():
    rows = []
    for f in sorted(MODEL_DIR.glob("H*.txt")):
        rows.extend(extract_key_rows(f))
    out = pd.DataFrame(rows)
    safe_to_csv(out, TABLE_DIR / "main_regression_summary.csv")
    return out

def extract_robustness_results():
    rows = []
    for f in sorted(ROBUST_MODEL_DIR.glob("*.txt")):
        rows.extend(extract_key_rows(f))
    out = pd.DataFrame(rows)
    safe_to_csv(out, ROBUST_TABLE_DIR / "robustness_summary.csv")
    return out
