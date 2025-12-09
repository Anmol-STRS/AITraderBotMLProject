from __future__ import annotations

import json
import sqlite3
from typing import Dict, Iterable, Optional, Tuple

import numpy as np
import pandas as pd

from ..config import MODEL_RESULTS_DB
from ..extensions import log

ALLOWED_MODEL_TYPES = ("xgboost", "xgb")
ALLOWED_MODES = {"returns", "returns_improved", "log_returns"}
DEFAULT_MODE = "returns_improved"


def _parse_additional_metrics(value: Optional[str]) -> Dict:
    if not value:
        return {}
    if isinstance(value, dict):
        return value
    try:
        parsed = json.loads(value)
        return parsed if isinstance(parsed, dict) else {}
    except (ValueError, TypeError):
        return {}


def _infer_mode(row: pd.Series) -> str:
    payload = _parse_additional_metrics(row.get("additional_metrics"))
    mode = str(payload.get("mode") or "").strip().lower()
    if not mode:
        name = str(row.get("model_name") or "").lower()
        if "return" in name:
            mode = "returns"
    if not mode:
        mode = DEFAULT_MODE
    return mode


def _sanitize_numeric(value):
    if value is None:
        return None
    if isinstance(value, (np.generic,)):
        value = value.item()
    try:
        if isinstance(value, float) and (np.isnan(value) or np.isinf(value)):
            return None
    except Exception:
        return value
    return value


def load_filtered_model_rows(
    *,
    allowed_modes: Optional[Iterable[str]] = None,
    allowed_model_types: Tuple[str, ...] = ALLOWED_MODEL_TYPES,
    horizon: int = 1,
) -> pd.DataFrame:
    """
    Load the latest model rows per symbol, filtered according to dashboard rules.
    """
    with sqlite3.connect(MODEL_RESULTS_DB) as conn:
        df = pd.read_sql_query(
            """
            SELECT
                m.model_id,
                m.symbol,
                m.model_name,
                m.model_type,
                m.horizon,
                m.created_at,
                tr.trained_at,
                tr.test_rmse,
                tr.test_mae,
                tr.test_r2,
                tr.test_mape,
                tr.test_direction_accuracy,
                tr.train_rmse,
                tr.train_mae,
                tr.train_r2,
                tr.additional_metrics
            FROM models m
            JOIN training_results tr ON m.model_id = tr.model_id
            """,
            conn,
        )

    raw_count = len(df)

    if df.empty:
        log.info("model_metrics: no model rows available in database")
        df.attrs["raw_count"] = raw_count
        return df

    df["created_at"] = pd.to_datetime(df["created_at"], errors="coerce")
    df["trained_at"] = pd.to_datetime(df["trained_at"], errors="coerce")
    df = df.sort_values(
        ["trained_at", "created_at", "model_id"],
        ascending=False,
        na_position="last",
    )
    df = df.drop_duplicates(subset=["model_id"], keep="first")

    df["mode"] = df.apply(_infer_mode, axis=1)
    extras = df["additional_metrics"].apply(_parse_additional_metrics)
    for column in ("train_samples", "val_samples", "test_samples"):
        df[column] = extras.apply(lambda payload, key=column: payload.get(key))

    allowed_modes = set(m.lower() for m in (allowed_modes or ALLOWED_MODES))

    df = df[
        df["mode"].str.lower().isin(allowed_modes)
        & df["model_type"]
        .astype(str)
        .str.lower()
        .str.contains("|".join(allowed_model_types))
        & (df["horizon"].fillna(0).astype(int) == horizon)
    ]

    metric_fields = ["test_r2", "test_rmse", "test_direction_accuracy"]
    df = df.dropna(subset=metric_fields)

    df = df.sort_values(
        ["created_at", "trained_at", "model_id"],
        ascending=[False, False, False],
        na_position="last",
    )
    df = df.drop_duplicates(subset=["symbol"], keep="first")
    df = df.reset_index(drop=True)

    log.info(
        "model_metrics: filtered rows -> %s symbols (from %s models)",
        len(df),
        df["model_id"].nunique(),
    )

    df = df.set_axis(df.columns, axis=1)
    df.attrs["raw_count"] = raw_count
    return df


def dataframe_to_symbol_payload(df: pd.DataFrame) -> Dict[str, Dict]:
    """
    Convert filtered DataFrame into a dict keyed by symbol with sanitized values.
    """
    payload: Dict[str, Dict] = {}
    if df.empty:
        return payload

    for _, row in df.iterrows():
        symbol = str(row["symbol"])
        payload[symbol] = {
            "symbol": symbol,
            "model_id": _sanitize_numeric(row.get("model_id")),
            "model_name": row.get("model_name"),
            "model_type": row.get("model_type"),
            "mode": row.get("mode"),
            "horizon": _sanitize_numeric(row.get("horizon")),
            "test_r2": _sanitize_numeric(row.get("test_r2")),
            "test_rmse": _sanitize_numeric(row.get("test_rmse")),
            "test_mae": _sanitize_numeric(row.get("test_mae")),
            "test_mape": _sanitize_numeric(row.get("test_mape")),
            "test_direction_accuracy": _sanitize_numeric(row.get("test_direction_accuracy")),
            "train_samples": _sanitize_numeric(row.get("train_samples")),
            "val_samples": _sanitize_numeric(row.get("val_samples")),
            "test_samples": _sanitize_numeric(row.get("test_samples")),
            "created_at": row.get("created_at").isoformat() if pd.notna(row.get("created_at")) else None,
        }
    return payload
