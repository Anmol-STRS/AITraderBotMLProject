# backend/routes/__init__.py

def register_routes(app):
    from .health import bp as health_bp
    from .symbol import bp as symbols_bp
    from .models_api import bp as models_bp
    from .agents_api import bp as agents_bp
    from .analytics import bp as analytics_bp
    from .mock_api import bp as mock_bp

    app.register_blueprint(health_bp, url_prefix="/api")
    app.register_blueprint(symbols_bp, url_prefix="/api")
    app.register_blueprint(models_bp, url_prefix="/api")
    app.register_blueprint(agents_bp, url_prefix="/api")
    app.register_blueprint(analytics_bp, url_prefix="/api")
    app.register_blueprint(mock_bp, url_prefix="/api")
