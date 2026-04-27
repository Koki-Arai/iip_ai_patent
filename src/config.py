from pathlib import Path
import os

PROJECT_DIR = Path(os.environ.get("IIP_PROJECT_DIR", Path.cwd()))
DATA_DIR = Path(os.environ.get("IIP_DATA_DIR", PROJECT_DIR / "data"))
OUTPUT_DIR = Path(os.environ.get("IIP_OUTPUT_DIR", PROJECT_DIR / "output"))

CHECKPOINT_DIR = OUTPUT_DIR / "checkpoints"
TABLE_DIR = OUTPUT_DIR / "tables"
FIGURE_DIR = OUTPUT_DIR / "figures"
MODEL_DIR = OUTPUT_DIR / "models"
ROBUST_MODEL_DIR = MODEL_DIR / "robustness"
ROBUST_TABLE_DIR = TABLE_DIR / "robustness"

for p in [OUTPUT_DIR, CHECKPOINT_DIR, TABLE_DIR, FIGURE_DIR, MODEL_DIR, ROBUST_MODEL_DIR, ROBUST_TABLE_DIR]:
    p.mkdir(parents=True, exist_ok=True)

DECADES = [x.strip() for x in os.environ.get("IIP_DECADES", "1990s,2000s,2010s,2020s").split(",") if x.strip()]
CHUNKSIZE = int(os.environ.get("IIP_CHUNKSIZE", "500000"))
FORCE_REBUILD = os.environ.get("IIP_FORCE_REBUILD", "0") == "1"

AI_CORE_CLASSES = ["G06N"]
AI_BROAD_CLASSES = ["G06N", "G06T", "G10L", "H04N", "G06F", "G06Q"]
SOFTWARE_CLASSES = ["G06F", "G06Q", "G06N", "G06T", "G10L", "H04N"]

REJECT_REASON_CODES = {"19", "89"}
GRANT_REASON_CODES = {"31"}

AP_COLS = ["ida", "adate", "sdate", "idr", "rdate", "tdate", "class1", "group1", "class2", "group2", "claim1", "claim2", "claim3"]
APPLICANT_COLS = ["ida", "seq", "ida_seq", "name", "address", "idname", "country_pref", "kohokan"]
INVENTOR_COLS = ["ida", "seq", "ida_seq", "name", "address"]
CC_COLS = ["citing", "cited", "cited_doc_num", "reason", "reason_date"]
