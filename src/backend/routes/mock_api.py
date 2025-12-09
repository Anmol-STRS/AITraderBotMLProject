# backend/routes/mock_api.py
"""
Mock API endpoints for dashboard testing
Use /api/mock/* endpoints to get fake data
"""
from flask import Blueprint, jsonify, request
from ..mock_data import (
    generate_mock_summary,
    generate_mock_symbols,
    generate_mock_predictions,
    generate_mock_training_metrics,
    generate_mock_live_predictions
)
from ..extensions import log

bp = Blueprint("mock_api", __name__)

@bp.route("/mock/summary", methods=["GET"])
def get_mock_summary():
    """Get mock summary statistics"""
    try:
        return jsonify(generate_mock_summary())
    except Exception as e:
        log.exception("Error generating mock summary")
        return jsonify({"error": str(e)}), 500

@bp.route("/mock/symbols", methods=["GET"])
def get_mock_symbols():
    """Get mock symbols with metrics"""
    try:
        return jsonify(generate_mock_symbols())
    except Exception as e:
        log.exception("Error generating mock symbols")
        return jsonify({"error": str(e)}), 500

@bp.route("/mock/predictions/compare", methods=["GET"])
def get_mock_predictions():
    """Get mock predictions for a symbol"""
    try:
        symbol = request.args.get('symbol', 'BMO.TO')
        limit = int(request.args.get('limit', 100))

        predictions = generate_mock_predictions(symbol, limit)

        # Format as multi-model response
        return jsonify({
            'symbol': symbol,
            'models': [
                {
                    'model_name': 'XGBoost',
                    'model_type': 'agentVD',
                    'predictions': [
                        {
                            'timestamp': p['date'],
                            'actual': p['actual'],
                            'predicted': p['agentVD']
                        } for p in predictions
                    ]
                }
            ]
        })
    except Exception as e:
        log.exception("Error generating mock predictions")
        return jsonify({"error": str(e)}), 500

@bp.route("/mock/metrics/<symbol>", methods=["GET"])
def get_mock_metrics(symbol: str):
    """Get mock training metrics for a symbol"""
    try:
        return jsonify(generate_mock_training_metrics(symbol))
    except Exception as e:
        log.exception("Error generating mock metrics")
        return jsonify({"error": str(e)}), 500

@bp.route("/mock/live/<symbol>", methods=["GET"])
def get_mock_live(symbol: str):
    """Get mock live predictions"""
    try:
        return jsonify(generate_mock_live_predictions(symbol))
    except Exception as e:
        log.exception("Error generating mock live predictions")
        return jsonify({"error": str(e)}), 500
