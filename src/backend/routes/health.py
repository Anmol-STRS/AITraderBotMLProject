# backend/routes/health.py
from flask import Blueprint, jsonify
from datetime import datetime
from ..extensions import log
from ..agents_manager import agent_manager
import sqlite3
from pathlib import Path

bp = Blueprint("health", __name__)

@bp.route("/health", methods=["GET"])
def health_check():
    # Check database connectivity
    db_healthy = False
    try:
        db_path = Path("data/storage/stocks.db")
        if db_path.exists():
            conn = sqlite3.connect(str(db_path))
            cursor = conn.execute("SELECT 1")
            cursor.fetchone()
            conn.close()
            db_healthy = True
    except Exception as e:
        log.error(f"Database health check failed: {e}")

    return jsonify({
        "status": "healthy",
        "timestamp": datetime.now().isoformat(),
        "database": db_healthy,
        "agents": list(agent_manager.agents.keys()) if hasattr(agent_manager, "agents") else []
    })
