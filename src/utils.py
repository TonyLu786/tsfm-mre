import os
import json
import yaml
import math
import hashlib
import warnings
from pathlib import Path
from typing import Dict, Any, Optional
import numpy as np
import pandas as pd
from pandas.tseries.offsets import BDay

RNG = np.random.default_rng(42)

def set_seed(seed: int = 42):
    global RNG
    RNG = np.random.default_rng(seed)

def ensure_dir(path: str):
    Path(path).mkdir(parents=True, exist_ok=True)

def sha256_of_bytes(b: bytes) -> str:
    return hashlib.sha256(b).hexdigest()

def sha256_of_file(path: str) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(8192), b""):
            h.update(chunk)
    return h.hexdigest()

def sha256_of_dataframe(df: pd.DataFrame) -> str:
    # stable hash: sort columns and index, convert to csv bytes
    c = sorted(df.columns)
    tmp = df[c].copy()
    tmp = tmp.sort_index()
    b = tmp.to_csv(index=True).encode("utf-8")
    return sha256_of_bytes(b)

def load_yaml(path: str) -> Dict[str, Any]:
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)

def save_json(obj: Any, path: str):
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, ensure_ascii=False, indent=2)

def business_days(start: str, end: str) -> pd.DatetimeIndex:
    return pd.bdate_range(start=start, end=end, freq="C")

def robust_zscore_fit(x: pd.Series) -> Dict[str, float]:
    med = np.nanmedian(x.values)
    iqr = np.nanpercentile(x.values, 75) - np.nanpercentile(x.values, 25)
    scale = iqr if iqr > 1e-8 else (np.nanstd(x.values) + 1e-8)
    return {"median": float(med), "scale": float(scale)}

def robust_zscore_transform(x: pd.Series, params: Dict[str, float]) -> pd.Series:
    return (x - params["median"]) / (params["scale"] + 1e-8)

def inverse_robust_zscore(x: pd.Series, params: Dict[str, float]) -> pd.Series:
    return x * (params["scale"] + 1e-8) + params["median"]

def sliding_windows(dates: pd.DatetimeIndex, window: int, step: int=1):
    for i in range(window, len(dates), step):
        yield (dates[i-1], dates[i-window:i])

def make_logger():
    import logging
    logger = logging.getLogger("tsfm_mre")
    if not logger.handlers:
        logger.setLevel(logging.INFO)
        ch = logging.StreamHandler()
        ch.setLevel(logging.INFO)
        fmt = logging.Formatter("[%(asctime)s] %(levelname)s - %(message)s")
        ch.setFormatter(fmt)
        logger.addHandler(ch)
    return logger

LOGGER = make_logger()
