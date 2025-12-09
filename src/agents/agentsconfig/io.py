# src/models/io.py
from pathlib import Path
import json
import logging
from typing import Tuple, Dict, Any

import xgboost as xgb

logger = logging.getLogger("models.io")


def save_xgb_model(model: xgb.XGBRegressor, model_path: Path, meta: Dict[str, Any]) -> None:
    """
    Save XGBoost model and a companion JSON meta file.
    model_path example: models/BMO_TO_xgb_h1.json
    meta will be saved as models/BMO_TO_xgb_h1.json.meta.json
    """
    model_path.parent.mkdir(parents=True, exist_ok=True)
    logger.info("Saving XGBoost model to %s", model_path)
    model.save_model(str(model_path))
    meta_path = model_path.with_suffix(model_path.suffix + ".meta.json")
    meta_path.write_text(json.dumps(meta, indent=2))
    logger.info("Saved model meta to %s (keys=%s)", meta_path, list(meta.keys()))


def load_xgb_model(model_path: Path) -> Tuple[xgb.XGBRegressor, Dict[str, Any]]:
    """
    Load an XGBoost model and its meta JSON.
    Returns (model, meta_dict). Raises FileNotFoundError if model missing.
    """
    if not model_path.exists():
        raise FileNotFoundError(f"Model not found: {model_path}")

    logger.info("Loading XGBoost model from %s", model_path)
    model = xgb.XGBRegressor()
    model.load_model(str(model_path))

    meta_path = model_path.with_suffix(model_path.suffix + ".meta.json")
    meta: Dict[str, Any] = {}
    if meta_path.exists():
        try:
            meta = json.loads(meta_path.read_text())
            logger.info("Loaded meta for %s (keys=%s)", model_path, list(meta.keys()))
        except Exception as exc:
            logger.warning("Could not parse meta file %s: %s", meta_path, exc)
    else:
        logger.debug("No meta file found for %s", model_path)

    return model, meta
