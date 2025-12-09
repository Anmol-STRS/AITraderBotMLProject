# backend/routes/models_api.py
from flask import Blueprint, jsonify, request
from pathlib import Path
from typing import List, Optional, Tuple
import pandas as pd
import math
import json

from ..extensions import log
from ..config import MODEL_RESULTS_DB
from ..utils import df_to_json_records

from src.database.model_results_db import ModelResultsDB
from src.agents.custom_agent.agentVD import UnifiedStockTrainer, get_sector_parameters
from src.agents.agentsconfig.io import save_xgb_model, load_xgb_model

bp = Blueprint("models_api", __name__)

# Absolute project root: project/backend/routes/models_api.py -> parents[2] == project root
PROJECT_ROOT = Path(__file__).resolve().parents[2]
MODELS_DIR = PROJECT_ROOT / "models"


# ---------------------------
# Helpers
# ---------------------------

def _safe_int(value, default: int) -> int:
    try:
        return int(value)
    except Exception:
        return default


def _pick_latest_xgb_row(models_df: pd.DataFrame, symbol: str, horizon: int) -> Optional[pd.Series]:
    if models_df is None or models_df.empty:
        return None

    df = models_df.copy()

    # Filter horizon if column exists
    if "horizon" in df.columns:
        df = df[df["horizon"].astype("Int64") == horizon]

    # Prefer XGBoost rows if model_type exists
    if "model_type" in df.columns:
        df = df[df["model_type"].astype(str).str.lower().str.contains("xgb|xgboost")]

    if df.empty:
        return None

    # Prefer newest by created_at if available, else highest model_id
    if "created_at" in df.columns:
        df = df.sort_values("created_at", ascending=False)
    elif "model_id" in df.columns:
        df = df.sort_values("model_id", ascending=False)

    return df.iloc[0]


def _resolve_model_path(symbol: str, horizon: int, model_path_hint: Optional[str]) -> Optional[Path]:
    safe = symbol.replace(".", "_")

    candidates: List[Path] = []

    if model_path_hint:
        hint = Path(model_path_hint)
        # If hint is relative, treat it as relative to project root
        candidates.append(hint if hint.is_absolute() else (PROJECT_ROOT / hint))

    candidates.extend([
        MODELS_DIR / f"{safe}_xgb_h{horizon}.json",
        MODELS_DIR / f"{symbol}_xgb_h{horizon}.json",  # if someone saved with dot
    ])

    for p in candidates:
        if p.exists():
            return p

    return None


def _decode_additional_metrics(raw) -> Optional[dict]:
    """Safely decode the JSON blob stored inside training_results.additional_metrics."""
    if raw is None:
        return None
    if isinstance(raw, float) and math.isnan(raw):
        return None
    if isinstance(raw, bytes):
        raw = raw.decode("utf-8", errors="ignore")
    if isinstance(raw, str):
        text = raw.strip()
        if not text:
            return None
        try:
            return json.loads(text)
        except json.JSONDecodeError:
            log.warning("Failed to parse additional_metrics JSON: %s", text[:120])
            return None
    if isinstance(raw, dict):
        return raw
    return None


def _merge_additional_fields(payload: dict, additional: Optional[dict]) -> None:
    """Enrich API payload with validation metrics, sample counts, and LLM context."""
    if not isinstance(additional, dict):
        return

    val_metrics = {
        key.replace("val_", "", 1): val
        for key, val in additional.items()
        if key.startswith("val_")
    }
    if val_metrics:
        payload["val_metrics"] = val_metrics

    sample_counts = {
        key: additional[key]
        for key in ("train_samples", "val_samples", "test_samples")
        if additional.get(key) is not None
    }
    if sample_counts:
        payload["sample_counts"] = sample_counts

    for metric_key in ("train_mdae", "train_ramape", "test_mdae", "test_ramape", "mode"):
        if metric_key in additional and additional[metric_key] is not None:
            payload[metric_key] = additional[metric_key]

    if "insights" in additional and "insights" not in payload:
        payload["insights"] = additional["insights"]
    if "instrument_profile" in additional and "instrument_profile" not in payload:
        payload["instrument_profile"] = additional["instrument_profile"]
    if "split" in additional and "split" not in payload:
        payload["split"] = additional["split"]


def _importance_df_from_scores(scores: dict, feature_names: List[str]) -> pd.DataFrame:
    """Map XGBoost booster scores (f0, f1, ...) to human feature names."""
    if not feature_names:
        return pd.DataFrame(columns=["feature", "importance"])

    features, importances = [], []
    for idx, name in enumerate(feature_names):
        features.append(name)
        importances.append(float(scores.get(f"f{idx}", 0.0)))

    df = pd.DataFrame({"feature": features, "importance": importances})
    df = df.sort_values("importance", ascending=False).reset_index(drop=True)
    return df


def _store_feature_importance(model_id: int, importance_df: pd.DataFrame) -> None:
    """Persist computed feature importance back to the DB for caching."""
    if importance_df is None or importance_df.empty:
        return

    # support either "importance" or "gain"
    if "importance" in importance_df.columns:
        score_col = "importance"
    elif "gain" in importance_df.columns:
        score_col = "gain"
    else:
        score_col = importance_df.columns[-1]

    importance_dict = {
        str(row["feature"]): float(row[score_col])
        for _, row in importance_df.iterrows()
    }

    with ModelResultsDB(MODEL_RESULTS_DB) as db:
        cur = db.conn.cursor()
        cur.execute("DELETE FROM feature_importance WHERE model_id = ?", (model_id,))
        db.add_feature_importance(model_id, importance_dict)
        db.conn.commit()


def _normalize_feature_df(df: pd.DataFrame) -> pd.DataFrame:
    """
    Normalize whatever DB returns into:
      columns: feature_name, importance_score, rank
    """
    if df is None or df.empty:
        return pd.DataFrame(columns=["feature_name", "importance_score", "rank"])

    try:
        out = df.copy()

        # Common variants
        if "feature" in out.columns and "feature_name" not in out.columns:
            out = out.rename(columns={"feature": "feature_name"})
        if "importance" in out.columns and "importance_score" not in out.columns:
            out = out.rename(columns={"importance": "importance_score"})
        if "gain" in out.columns and "importance_score" not in out.columns:
            out = out.rename(columns={"gain": "importance_score"})

        if "rank" not in out.columns:
            # Create rank based on sorted importance
            out = out.sort_values("importance_score", ascending=False).reset_index(drop=True)
            out["rank"] = list(range(1, len(out) + 1))

        return out
    except Exception as e:
        log.error(f"Error normalizing feature DataFrame: {e}, columns: {df.columns.tolist() if df is not None else 'None'}")
        return pd.DataFrame(columns=["feature_name", "importance_score", "rank"])


def _auto_train_symbol(symbol: str, horizon: int = 1, mode: str = "returns") -> Optional[Path]:
    """
    Train an XGBoost model for the given symbol if none exists yet.
    Stores the trained model (with meta) and persists metrics into ModelResultsDB.
    """
    try:
        symbol = symbol.upper()

        # If DB already has a usable model, don't retrain
        with ModelResultsDB(MODEL_RESULTS_DB) as db:
            models_df = db.get_models_by_symbol(symbol)
            row = _pick_latest_xgb_row(models_df, symbol, horizon)
            if row is not None:
                hint = row.get("model_path")
                existing_path = _resolve_model_path(symbol, horizon, hint)
                if existing_path and existing_path.exists():
                    return existing_path

        trainer = UnifiedStockTrainer(symbol, mode=mode, params=get_sector_parameters(symbol))
        df = trainer.load_data(years=5)
        if df is None or df.empty:
            log.warning("No data to auto-train symbol %s", symbol)
            return None

        X, y, ts = trainer.prepare_data(df, "close", forecast_horizon=horizon)
        if X is None or X.empty:
            log.warning("Insufficient features to auto-train %s", symbol)
            return None

        (
            X_train, X_val, X_test,
            y_train, y_val, y_test,
            *_,
        ) = trainer.temporal_split(X, y, ts)

        if X_train.empty or X_val.empty or X_test.empty:
            log.warning("Need more samples to auto-train %s", symbol)
            return None

        trainer.train(X_train, y_train, X_val, y_val, early_stopping_rounds=20)
        metrics = trainer.evaluate(X_train, y_train, X_val, y_val, X_test, y_test)

        booster = trainer.model.get_booster()
        scores = booster.get_score(importance_type="gain")
        importance = _importance_df_from_scores(scores, trainer.feature_names).rename(columns={"importance": "gain"})

        MODELS_DIR.mkdir(parents=True, exist_ok=True)
        model_name = f"XGBoost_{symbol}_h{horizon}"
        model_path = MODELS_DIR / f"{symbol.replace('.', '_')}_xgb_h{horizon}.json"

        save_xgb_model(
            trainer.model,
            model_path,
            {
                "symbol": symbol,
                "mode": mode,
                "horizon": horizon,
                "feature_names": trainer.feature_names,
            },
        )

        trainer.save_results(
            metrics,
            importance,
            horizon=horizon,
            train_samples=len(X_train),
            val_samples=len(X_val),
            test_samples=len(X_test),
            model_path=str(model_path),
            model_name=model_name,
        )

        log.info("Auto-trained model for %s saved to %s", symbol, model_path)
        return model_path

    except Exception:
        log.exception("Auto-training failed for %s", symbol)
        return None


# ---------------------------
# Routes
# ---------------------------

@bp.route("/models", methods=["GET"])
def get_all_models():
    try:
        with ModelResultsDB(MODEL_RESULTS_DB) as db:
            df = db.get_all_models()
            return jsonify(df_to_json_records(df))
    except Exception as e:
        log.exception("Error fetching models")
        return jsonify({"error": str(e)}), 500


@bp.route("/metrics/<symbol>", methods=["GET"])
def get_metrics(symbol: str):
    """
    Returns:
      - model + metrics (from DB)
      - top feature importance (cached in DB if available, else computed from model file)
    Query params:
      - horizon: int (default 1)
      - auto_train: 1 to allow auto-train if model missing (default 0)
    """
    symbol = symbol.upper()
    horizon = _safe_int(request.args.get("horizon", 1), 1)
    auto_train = str(request.args.get("auto_train", "0")).lower() in ("1", "true", "yes")

    try:
        model_row = None
        feature_df = None
        additional_data = None

        # QUICK FIX: Bypass feature importance to unblock demo
        with ModelResultsDB(MODEL_RESULTS_DB) as db:
            models_df = db.get_models_by_symbol(symbol)
            model_row = _pick_latest_xgb_row(models_df, symbol, horizon)
            if model_row is not None:
                model_id = int(model_row["model_id"])
                # SKIP feature importance for now - just return metrics
                feature_df = None
                additional_data = _decode_additional_metrics(model_row.get("additional_metrics"))

        # If cached importance exists, return fast
        if model_row is not None and feature_df is not None and not feature_df.empty:
            top_features = [
                {"name": r["feature_name"], "importance": float(r["importance_score"])}
                for _, r in feature_df.iterrows()
            ]

            payload = {
                "symbol": symbol,
                "horizon": horizon,
                "model_id": int(model_row["model_id"]),
                "model_name": model_row.get("model_name"),
                "model_type": model_row.get("model_type"),
                "top_features": top_features,
                "total_features": int(model_row.get("feature_count") or len(feature_df)),
                "_source": "cached_db",
            }

            # include common metric fields if present
            for k in ["test_rmse", "test_mae", "test_r2", "test_mape", "test_direction_accuracy", "total_predictions"]:
                if k in model_row.index:
                    val = model_row.get(k)
                    # Convert pandas/numpy types to Python native types and handle NaN/Inf
                    if pd.isna(val):
                        payload[k] = None
                    elif isinstance(val, (int, float)):
                        if math.isinf(val):
                            payload[k] = None
                        else:
                            payload[k] = float(val)
                    else:
                        payload[k] = val

            _merge_additional_fields(payload, additional_data)

            return jsonify(payload)

        # Need model file to compute importance if not cached
        model_path_hint = model_row.get("model_path") if model_row is not None else None
        model_path = _resolve_model_path(symbol, horizon, model_path_hint)

        # Optionally auto-train if missing
        if model_path is None and auto_train:
            model_path = _auto_train_symbol(symbol, horizon=horizon)
            if model_path:
                # re-fetch model row after training
                with ModelResultsDB(MODEL_RESULTS_DB) as db:
                    models_df = db.get_models_by_symbol(symbol)
                    model_row = _pick_latest_xgb_row(models_df, symbol, horizon)
                    if model_row is not None:
                        additional_data = _decode_additional_metrics(model_row.get("additional_metrics"))

        if model_path is None:
            return jsonify({
                "error": "Model not found",
                "symbol": symbol,
                "horizon": horizon,
                "hint": "Train the model first or call /metrics/<symbol>?horizon=X&auto_train=1"
            }), 404

        # Compute importance from model meta (NO candle DB query, NO prepare_data)
        model, meta = load_xgb_model(model_path)
        feature_order = meta.get("feature_names") or []
        if not feature_order:
            # fallback: at least return something without crashing
            log.warning("No feature_names in model meta for %s (h%d)", symbol, horizon)

        booster = model.get_booster()
        scores = booster.get_score(importance_type="gain")
        full_importance = _importance_df_from_scores(scores, feature_order)

        # Cache computed importance if we have model_id
        if model_row is not None and not full_importance.empty:
            try:
                _store_feature_importance(int(model_row["model_id"]), full_importance)
            except Exception:
                log.warning("Failed to store feature importance for %s", symbol, exc_info=True)

        top = full_importance.head(10)
        top_features = [{"name": r["feature"], "importance": float(r["importance"])} for _, r in top.iterrows()]

        payload = {
            "symbol": symbol,
            "horizon": horizon,
            "top_features": top_features,
            "total_features": len(feature_order) if feature_order else len(full_importance),
            "_source": "computed_model_file",
        }

        if model_row is not None:
            payload.update({
                "model_id": int(model_row["model_id"]),
                "model_name": model_row.get("model_name"),
                "model_type": model_row.get("model_type"),
            })
            for k in ["test_rmse", "test_mae", "test_r2", "test_mape", "test_direction_accuracy", "total_predictions"]:
                if k in model_row.index:
                    val = model_row.get(k)
                    # Convert pandas/numpy types to Python native types and handle NaN/Inf
                    if pd.isna(val):
                        payload[k] = None
                    elif isinstance(val, (int, float)):
                        if math.isinf(val):
                            payload[k] = None
                        else:
                            payload[k] = float(val)
                    else:
                        payload[k] = val
            _merge_additional_fields(payload, additional_data)

        return jsonify(payload)

    except Exception as e:
        import traceback
        log.exception("Error getting metrics for %s (h%d)", symbol, horizon)
        error_details = {
            "error": str(e),
            "type": type(e).__name__,
            "traceback": traceback.format_exc()
        }
        log.error(f"Full error details: {error_details}")
        return jsonify({"error": str(e), "type": type(e).__name__}), 500


@bp.route("/llm/<symbol>", methods=["GET"])
def get_llm_insights(symbol: str):
    """Return the latest stored GPT/Claude summaries for a symbol."""
    symbol = symbol.upper()
    limit = max(1, _safe_int(request.args.get("limit", 4), 4))
    try:
        with ModelResultsDB(MODEL_RESULTS_DB) as db:
            df = db.get_models_by_symbol(symbol)

        if df is None or df.empty:
            return jsonify([])

        llm_df = df[df["model_type"].astype(str).str.contains("LLM", case=False, na=False)]
        if llm_df.empty:
            return jsonify([])

        llm_df = llm_df.sort_values("created_at", ascending=False).head(limit)
        payloads = []
        for _, row in llm_df.iterrows():
            entry = {
                "symbol": symbol,
                "model_id": int(row["model_id"]),
                "model_name": row.get("model_name"),
                "model_type": row.get("model_type"),
                "horizon": int(row.get("horizon") or 1),
                "created_at": row.get("created_at"),
            }
            description = row.get("description")
            if isinstance(description, str) and description.strip():
                try:
                    entry["summary"] = json.loads(description)
                except json.JSONDecodeError:
                    entry["summary"] = {"text": description}

            additional = _decode_additional_metrics(row.get("additional_metrics"))
            _merge_additional_fields(entry, additional)
            payloads.append(entry)

        return jsonify(payloads)
    except Exception as exc:
        log.exception("Error fetching LLM insights for %s", symbol)
        return jsonify({"error": str(exc)}), 500
