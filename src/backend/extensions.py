# backend/extensions.py
from flask_cors import CORS
from flask_socketio import SocketIO
import logging
from pathlib import Path
from .config import PROJECT_ROOT, LOGS_DIR

# Logger
LOGS_DIR.mkdir(parents=True, exist_ok=True)
def get_default_logger(name="DashboardAPI"):
    logger = logging.getLogger(name)
    if not logger.handlers:
        handler = logging.StreamHandler()
        fmt = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
        handler.setFormatter(logging.Formatter(fmt))
        logger.addHandler(handler)
        logger.setLevel(logging.INFO)
    return logger

log = get_default_logger("DashboardAPI")

# SocketIO (use eventlet or gevent in production)
socketio = SocketIO(cors_allowed_origins="*", logger=False, engineio_logger=False)

def init_extensions(app):
    # enable CORS with specific configuration
    CORS(app, resources={
        r"/api/*": {
            "origins": ["http://localhost:3000", "http://127.0.0.1:3000"],
            "methods": ["GET", "POST", "PUT", "DELETE", "OPTIONS"],
            "allow_headers": ["Content-Type", "Authorization", "X-Request-ID"],
            "supports_credentials": True
        }
    })
    # attach socketio to app for later use
    socketio.init_app(app)
    log.info("Extensions initialized")
