from pathlib import Path
import gc
import subprocess
import sys
import pandas as pd
from .config import DATA_DIR, CHECKPOINT_DIR

FULLWIDTH_TRANS = str.maketrans("０１２３４５６７８９", "0123456789")

def normalize_digits(x):
    if pd.isna(x):
        return x
    return str(x).strip().translate(FULLWIDTH_TRANS)

def starts_with_any(x, prefixes):
    if pd.isna(x):
        return False
    return any(str(x).strip().startswith(p) for p in prefixes)

def file_path(table, decade):
    return DATA_DIR / f"{table}_{decade}.txt"

def checkpoint_path(name):
    return CHECKPOINT_DIR / name

def save_df(df, path):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    try:
        if path.suffix != ".parquet":
            path = path.with_suffix(".parquet")
        df.to_parquet(path, index=False)
        return path
    except Exception:
        pkl = path.with_suffix(".pkl")
        df.to_pickle(pkl)
        return pkl

def load_df(path):
    path = Path(path)
    if path.exists():
        if path.suffix == ".parquet":
            return pd.read_parquet(path)
        if path.suffix == ".pkl":
            return pd.read_pickle(path)
    if path.with_suffix(".parquet").exists():
        return pd.read_parquet(path.with_suffix(".parquet"))
    if path.with_suffix(".pkl").exists():
        return pd.read_pickle(path.with_suffix(".pkl"))
    raise FileNotFoundError(path)

def has_checkpoint(name):
    p = checkpoint_path(name)
    return p.exists() or p.with_suffix(".parquet").exists() or p.with_suffix(".pkl").exists()

def get_checkpoint(name):
    return load_df(checkpoint_path(name))

def put_checkpoint(df, name):
    return save_df(df, checkpoint_path(name))

def safe_to_csv(df, path):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, index=False, encoding="utf-8-sig")

def clean_memory():
    gc.collect()

def print_file_status(decades):
    print("DATA_DIR:", DATA_DIR)
    for dec in decades:
        for table in ["ap", "applicant", "inventor", "cc"]:
            p = file_path(table, dec)
            print(f"{dec:5s} {table:10s} {str(p.exists()):5s} {p}")

def read_iip_table(path, expected_cols=None, usecols=None, chunksize=None):
    path = Path(path)
    if chunksize is not None:
        try:
            test = pd.read_csv(path, sep="\t", encoding="utf-8", dtype=str, nrows=5, low_memory=False)
            headerless = expected_cols is not None and not set(expected_cols[:2]).issubset(set(test.columns))
        except Exception:
            headerless = True
        return pd.read_csv(path, sep="\t", encoding="utf-8", dtype=str,
                           names=expected_cols if headerless else None,
                           header=None if headerless else 0,
                           usecols=usecols, chunksize=chunksize, low_memory=False)
    try:
        df = pd.read_csv(path, sep="\t", encoding="utf-8", dtype=str, usecols=usecols, low_memory=False)
        if expected_cols is not None and not set(expected_cols[:2]).issubset(set(df.columns)):
            df = pd.read_csv(path, sep="\t", encoding="utf-8", dtype=str, names=expected_cols, header=None, usecols=usecols, low_memory=False)
        return df
    except ValueError:
        return pd.read_csv(path, sep="\t", encoding="utf-8", dtype=str, names=expected_cols, header=None, usecols=usecols, low_memory=False)

def ensure_pyfixest():
    try:
        import pyfixest  # noqa
        return True
    except Exception:
        subprocess.check_call([sys.executable, "-m", "pip", "install", "-q", "pyfixest"])
        import pyfixest  # noqa
        return True

def save_model_summary(model, path):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        f.write(str(model.summary()))

def extract_key_rows(model_file):
    rows = []
    text = Path(model_file).read_text(encoding="utf-8")
    targets = {"ai_core", "ai_broad", "ai_core_primary", "ai_broad_noncore", "ai_core_x_2015_2018", "ai_core_density", "ai_core_density_sq", "ai_broad_density", "ai_broad_density_sq"}
    for line in text.splitlines():
        s = line.strip()
        if not s.startswith("|"):
            continue
        parts = [p.strip() for p in s.strip("|").split("|")]
        if len(parts) >= 7 and parts[0] in targets:
            rows.append({"file": Path(model_file).name, "variable": parts[0], "estimate": parts[1], "std_error": parts[2], "t_value": parts[3], "p_value": parts[4], "ci_2_5": parts[5], "ci_97_5": parts[6]})
    return rows
