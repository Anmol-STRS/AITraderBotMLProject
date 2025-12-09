import sqlite3
import pandas as pd
from typing import Dict, Any
from .extensions import log

class AgentManager:
    def __init__(self):
        self.agents = {"agentVD": {"instance": None, "status": "active", "last_analysis": None, "total_analyses": 0}}
        self.symbols = ["BMO.TO", "RY.TO", "TD.TO"]

    def get_market_data(self, symbol: str, days: int = 100) -> pd.DataFrame:
        try:
            conn = sqlite3.connect("data/storage/stocks.db")
            df = pd.read_sql_query(f"SELECT * FROM candle WHERE symbol = '{symbol}' ORDER BY ts DESC LIMIT {days}", conn)
            conn.close()
            if not df.empty:
                df = df.sort_values("ts").reset_index(drop=True)
                df['ts'] = pd.to_datetime(df['ts'])
            return df
        except Exception as e:
            log.error("get_market_data error: %s", e)
            return pd.DataFrame()

    def analyze_symbol(self, agent_name: str, symbol: str) -> Dict[str, Any]:
        """Analyze a symbol with the specified agent."""
        try:
            # For agentVD, perform direct model prediction
            if agent_name == "agentVD":
                from pathlib import Path
                from src.agents.custom_agent.agentVD import UnifiedStockTrainer
                from src.agents.agentsconfig.io import load_xgb_model
                from .utils import json_now_iso

                # Load recent data
                df = self.get_market_data(symbol, days=200)
                if df.empty:
                    return {"error": "no data for symbol", "agent": agent_name, "symbol": symbol}

                model_path = Path(f"models/{symbol.replace('.', '_')}_xgb_h1.json")
                if not model_path.exists():
                    return {"error": "model not found", "agent": agent_name, "symbol": symbol}

                model, meta = load_xgb_model(model_path)
                mode = meta.get("mode", "returns")
                horizon = int(meta.get("horizon", 1))

                trainer = UnifiedStockTrainer(symbol, mode=mode)
                X, _, _ = trainer.prepare_data(df, "close", forecast_horizon=horizon)
                if X.shape[0] == 0:
                    return {"error": "insufficient feature rows", "agent": agent_name, "symbol": symbol}

                # Align features
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

                return {
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

            # For other agents, return a placeholder
            log.warning(f"Agent {agent_name} analysis not fully implemented")
            return {"error": f"Agent {agent_name} not fully implemented", "agent": agent_name, "symbol": symbol}

        except Exception as e:
            log.error(f"Error analyzing {symbol} with {agent_name}: {e}")
            return {"error": str(e), "agent": agent_name, "symbol": symbol}

agent_manager = AgentManager()