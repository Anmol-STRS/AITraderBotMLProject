# backend/utils.py
from typing import List, Dict, Any
import pandas as pd
from datetime import datetime, timezone
import numpy as np

def safe_float(value, default=None):
    try:
        if pd.isna(value):
            return default
        return float(value)
    except Exception:
        return default

def safe_int(value, default=0):
    try:
        if pd.isna(value):
            return default
        return int(value)
    except Exception:
        return default

def json_now_iso() -> str:
    """Return an ISO-8601 timestamp with UTC timezone."""
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")

def df_to_json_records(df: pd.DataFrame) -> List[Dict[str, Any]]:
    """
    Convert DataFrame into a list of JSON-serializable dicts:
      - replace NaN/inf with None
      - convert numpy types to native Python types
      - convert Timestamps/datetimes to ISO strings
    """
    if df is None or df.empty:
        return []

    # Make a shallow copy to avoid mutating caller's df
    df2 = df.copy()

    # Convert datetimes -> ISO strings
    for col in df2.select_dtypes(include=["datetime64[ns]", "datetimetz"]).columns:
        df2[col] = df2[col].apply(lambda x: x.isoformat() if pd.notna(x) else None)

    # Replace numpy/pandas NaN/infinite with None
    df2 = df2.replace({np.nan: None, np.inf: None, -np.inf: None})

    # Convert numpy scalar types to Python types for each cell
    records = []
    for row in df2.to_dict(orient="records"):
        clean = {}
        for k, v in row.items():
            if v is None:
                clean[k] = None
                continue
            # pandas/Numpy integer/float/bool -> native
            if isinstance(v, (np.integer,)):
                clean[k] = int(v)
            elif isinstance(v, (np.floating,)):
                clean[k] = float(v)
            elif isinstance(v, (np.bool_,)):
                clean[k] = bool(v)
            # datetimes may be present as pandas.Timestamp objects
            elif isinstance(v, (pd.Timestamp, datetime)):
                clean[k] = v.isoformat()
            else:
                clean[k] = v
        records.append(clean)

    return records
