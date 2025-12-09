# backend/routes/symbols.py
from flask import Blueprint, jsonify, request
import sqlite3
import pandas as pd
import numpy as np
from ..utils import safe_float, safe_int, json_now_iso, df_to_json_records
from ..extensions import log
from .model_metrics import dataframe_to_symbol_payload, load_filtered_model_rows
from src.database.model_results_db import ModelResultsDB
from ..config import MODEL_RESULTS_DB

bp = Blueprint("symbols", __name__)

def _sanitize(v):
    """Return plain Python types safe for JSON."""
    if v is None:
        return None
    # pandas Timestamp / datetime
    if isinstance(v, (pd.Timestamp,)) or hasattr(v, "isoformat"):
        try:
            return v.isoformat()
        except Exception:
            return str(v)
    # numpy scalars -> native
    if isinstance(v, (np.integer,)):
        return int(v)
    if isinstance(v, (np.floating,)):
        return float(v)
    if isinstance(v, (np.bool_,)):
        return bool(v)
    # catch NaN/inf
    try:
        if isinstance(v, float) and (pd.isna(v) or np.isinf(v)):
            return None
    except Exception:
        pass
    return v

@bp.route("/symbols", methods=["GET"])
def get_symbols():
    """Get all symbols with their latest metrics (sanitized)."""
    conn = None
    try:
        conn = sqlite3.connect("data/storage/stocks.db")
        # Use parameterized queries where possible
        query = """
            SELECT
                symbol,
                MAX(ts) as last_updated,
                COUNT(*) as total_candles
            FROM candle
            GROUP BY symbol
            ORDER BY symbol
        """
        symbols_df = pd.read_sql_query(query, conn)
        symbols_list = []

        filtered_models = load_filtered_model_rows()
        model_lookup = dataframe_to_symbol_payload(filtered_models)

        for _, row in symbols_df.iterrows():
            symbol = str(row['symbol'])

            price_q = "SELECT close, volume FROM candle WHERE symbol = ? ORDER BY ts DESC LIMIT 1"
            latest = pd.read_sql_query(price_q, conn, params=(symbol,))

            if latest.empty:
                continue

            entry = {
                "symbol": symbol,
                "price": _sanitize(safe_float(latest['close'].iloc[0])),
                "volume": _sanitize(safe_int(latest['volume'].iloc[0])),
                "last_updated": _sanitize(row['last_updated']),
                "total_candles": _sanitize(int(row['total_candles']))
            }

            # Merge latest filtered model metrics if available
            model_entry = model_lookup.get(symbol)
            if model_entry:
                entry.update(
                    {
                        "model_id": _sanitize(model_entry.get("model_id")),
                        "model_name": _sanitize(model_entry.get("model_name")),
                        "model_type": _sanitize(model_entry.get("model_type")),
                        "mode": _sanitize(model_entry.get("mode")),
                        "test_r2": _sanitize(model_entry.get("test_r2")),
                        "test_rmse": _sanitize(model_entry.get("test_rmse")),
                        "test_mae": _sanitize(model_entry.get("test_mae")),
                        "test_mape": _sanitize(model_entry.get("test_mape")),
                        "test_direction_accuracy": _sanitize(model_entry.get("test_direction_accuracy")),
                        "train_samples": _sanitize(model_entry.get("train_samples")),
                        "val_samples": _sanitize(model_entry.get("val_samples")),
                        "test_samples": _sanitize(model_entry.get("test_samples")),
                        "model_created_at": model_entry.get("created_at"),
                    }
                )

            symbols_list.append(entry)

        log.info(
            "symbols endpoint: %d symbols with %d filtered model rows",
            len(symbols_list),
            len(model_lookup),
        )
        return jsonify(symbols_list)

    except Exception as e:
        log.exception("Error getting symbols")
        return jsonify({"error": str(e)}), 500

    finally:
        if conn:
            conn.close()

@bp.route("/model/summary", methods=["GET"])
def model_summary():
    """
    Returns a summary of models. Optional query param: ?symbol=XXX
    """
    try:
        symbol = request.args.get("symbol")
        with ModelResultsDB(MODEL_RESULTS_DB) as db:
            df = db.get_all_models()

        if symbol:
            symbol = symbol.upper()
            df = df[df['symbol'] == symbol]
            if df.empty:
                return jsonify({"error": "No model found for symbol"}), 404

        return jsonify(df_to_json_records(df))
    except Exception as e:
        log.exception("Error getting model summary")
        return jsonify({"error": str(e)}), 500
