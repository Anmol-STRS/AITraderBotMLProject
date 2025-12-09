# backend/__init__.py
from flask import Flask
from .config import PROJECT_ROOT
from .extensions import init_extensions, socketio, log
from .routes import register_routes

def create_app():
    app = Flask(__name__, instance_relative_config=False)
    app.config['SECRET_KEY'] = 'trading-arena-secret-key'

    # init extensions (socketio, cors, logger)
    init_extensions(app)

    # register blueprints
    register_routes(app)


    # ensure default agentVD models exist before serving requests
    try:
        from .agents_manager import agent_manager
        from .routes.models_api import _auto_train_symbol

        for symbol in getattr(agent_manager, "symbols", []):
            _auto_train_symbol(symbol.upper())
    except Exception:
        log.exception("Bootstrap auto-training failed")

    return app
