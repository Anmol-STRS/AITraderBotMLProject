# src/backend/config.py
from pathlib import Path
from typing import Optional
import os

# PROJECT_ROOT = repo root (two levels up from this file: src/backend -> src -> repo-root)
PROJECT_ROOT: Path = Path(__file__).resolve().parents[2]

# Useful runtime directories (absolute)
MODELS_DIR: Path = PROJECT_ROOT / "models"
PREDICTIONS_DIR: Path = PROJECT_ROOT / "predictions"
LOGS_DIR: Path = PROJECT_ROOT / "logs"
DATA_DIR: Path = PROJECT_ROOT / "data"
STORAGE_DIR: Path = DATA_DIR / "storage"
MODEL_RESULTS_DB: Path = PROJECT_ROOT / "model_results.db"

# Ensure directories exist (safe to call at import time)
for p in (MODELS_DIR, PREDICTIONS_DIR, LOGS_DIR, STORAGE_DIR):
    try:
        p.mkdir(parents=True, exist_ok=True)
    except Exception:
        # don't crash on permission issues; caller can handle
        pass

# Optional adapter to your project's Config class (if you have one).
# If you don't want to import src.config.config here, remove the function.
def get_src_config():
    """
    Return an instance of your project's Config if available.
    If your project doesn't expose a Config class, this will raise ImportError.
    Use try/except when calling this.
    """
    try:
        from src.config.config import Config as SrcConfig
        return SrcConfig()
    except Exception:
        return None


# helper to get absolute path for runtime files
def project_path(*parts: str) -> Path:
    return PROJECT_ROOT.joinpath(*parts).resolve()
