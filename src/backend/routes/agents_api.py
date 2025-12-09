# backend/routes/agents_api.py
from flask import Blueprint, jsonify, request
from ..extensions import log
from ..agents_manager import agent_manager
from ..utils import json_now_iso
from src.agents.custom_agent.agentVD import UnifiedStockTrainer
from src.agents.agentsconfig.io import load_xgb_model
from pathlib import Path

bp = Blueprint("agents_api", __name__)

@bp.route("/analyze", methods=["POST"])
def analyze_request():
    data = request.get_json() or {}
    symbol = data.get("symbol")
    agent = data.get("agent", "agentVD")
    if not symbol:
        return jsonify({"error": "symbol required"}), 400

    # For agentVD: load model and predict
    if agent == "agentVD":
        try:
            # load recent data
            import sqlite3, pandas as pd
            conn = sqlite3.connect("data/storage/stocks.db")
            df = pd.read_sql_query(f"SELECT * FROM candle WHERE symbol = '{symbol}' ORDER BY ts DESC LIMIT 200", conn)
            conn.close()
            if df.empty:
                return jsonify({"error": "no data for symbol"}), 404

            model_path = Path(f"models/{symbol.replace('.', '_')}_xgb_h1.json")
            if not model_path.exists():
                return jsonify({"error": "model not found"}), 404

            model, meta = load_xgb_model(model_path)
            mode = meta.get("mode", "returns")
            horizon = int(meta.get("horizon", 1))

            trainer = UnifiedStockTrainer(symbol, mode=mode)
            X, _, _ = trainer.prepare_data(df, "close", forecast_horizon=horizon)
            if X.shape[0] == 0:
                return jsonify({"error": "insufficient feature rows"}), 400

            # align features
            feature_order = meta.get("feature_names", trainer.feature_names)
            for f in feature_order:
                if f not in X.columns:
                    X[f] = 0.0
            ordered = X[feature_order].iloc[-1:].fillna(0.0)

            pred_raw = model.predict(ordered)[0]
            current_price = float(df['close'].iloc[-1])
            if mode == "returns":
                predicted_price = current_price * (1.0 + float(pred_raw))
            else:
                predicted_price = float(pred_raw)
            change_pct = (predicted_price - current_price) / current_price * 100.0

            signal = "neutral"
            action = "hold"
            if change_pct > 2.0:
                signal, action = "strong_buy", "buy"
            elif change_pct > 0.5:
                signal, action = "buy", "buy"
            elif change_pct < -2.0:
                signal, action = "strong_sell", "sell"
            elif change_pct < -0.5:
                signal, action = "sell", "sell"

            result = {
                "symbol": symbol,
                "agent": "agentVD",
                "timestamp": json_now_iso(),
                "current_price": current_price,
                "predicted_price": float(predicted_price),
                "change_pct": float(change_pct),
                "signal": signal,
                "action": action,
                "confidence": min(abs(change_pct) / 5.0, 1.0)
            }

            # optionally store result in agent_results table (left for you)
            return jsonify(result)

        except Exception as e:
            log.exception("agentVD analysis error")
            return jsonify({"error": str(e)}), 500

    # For LLM agents, delegate to agent_manager (if available)
    else:
        try:
            res = agent_manager.analyze_symbol(agent, symbol)
            return jsonify(res)
        except Exception as e:
            log.exception("LLM agent analyze error")
            return jsonify({"error": str(e)}), 500
