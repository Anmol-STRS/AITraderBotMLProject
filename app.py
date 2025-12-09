# app.py
from src.backend import create_app, socketio
from src.backend.extensions import log

app = create_app()

if __name__ == "__main__":
    log.info("="*70)
    log.info("AI Trading Dashboard API")
    log.info("="*70)
    socketio.run(app, host="0.0.0.0", port=8000, debug=True, use_reloader=True)
