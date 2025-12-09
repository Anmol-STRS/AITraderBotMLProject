# backend/routes/analytics.py
from flask import Blueprint, jsonify, request
import sqlite3
import pandas as pd
from ..extensions import log
from ..agents_manager import agent_manager
from .model_metrics import load_filtered_model_rows

bp = Blueprint("analytics", __name__)

def get_training_summary_stats():
    """Aggregate statistics from database."""
    try:
        filtered = load_filtered_model_rows()
        raw_count = filtered.attrs.get("raw_count", 0)
        log.info(
            "dashboard summary: %s total model rows -> %s filtered rows",
            raw_count,
            len(filtered),
        )

        if filtered.empty:
            return {
                'total_models': 0,
                'avg_r2': 0.0,
                'avg_rmse': 0.0,
                'avg_mae': 0.0,
                'avg_direction': 0.0,
                'avg_mape': 0.0,
                'rmse_unit': 'mixed',
            }

        selection_debug = filtered[["symbol", "created_at", "model_id", "mode", "test_rmse", "test_r2"]]
        log.debug("dashboard summary rows:\n%s", selection_debug.to_string(index=False))

        def _safe_mean(series):
            value = series.dropna().mean()
            return float(value) if pd.notna(value) else 0.0

        mode_labels = {
            str(m).lower()
            for m in filtered['mode'].dropna().unique()
            if isinstance(m, str) and m
        }
        if not mode_labels:
            rmse_unit = 'mixed'
        elif all('return' in label for label in mode_labels):
            rmse_unit = 'percent'
        elif all('return' not in label for label in mode_labels):
            rmse_unit = 'absolute'
        else:
            rmse_unit = 'mixed'

        stats = {
            'total_models': int(len(filtered)),
            'avg_r2': _safe_mean(filtered['test_r2']),
            'avg_rmse': _safe_mean(filtered['test_rmse']),
            'avg_mae': _safe_mean(filtered['test_mae']),
            'avg_direction': _safe_mean(filtered['test_direction_accuracy']),
            'avg_mape': _safe_mean(filtered['test_mape']),
            'rmse_unit': rmse_unit,
        }

        return stats
    except Exception as e:
        log.error(f"Error getting training summary stats: {e}")
        return {
            'total_models': 0,
            'avg_r2': 0.0,
            'avg_rmse': 0.0,
            'avg_mae': 0.0,
            'avg_direction': 0.0,
            'avg_mape': 0.0,
            'rmse_unit': 'mixed',
        }

@bp.route("/summary", methods=["GET"])
def get_summary():
    """Get overall summary statistics"""
    try:
        conn = sqlite3.connect("data/storage/stocks.db")

        # Count symbols
        symbols_query = "SELECT COUNT(DISTINCT symbol) as count FROM candle"
        symbols_count = pd.read_sql_query(symbols_query, conn)['count'].iloc[0]

        # Count total records
        records_query = "SELECT COUNT(*) as count FROM candle"
        records_count = pd.read_sql_query(records_query, conn)['count'].iloc[0]

        # Get date range
        date_query = "SELECT MIN(ts) as min_date, MAX(ts) as max_date FROM candle"
        date_range = pd.read_sql_query(date_query, conn)

        conn.close()

        # Get agent stats and training summary
        agent_stats = {}
        for agent_name, agent_info in agent_manager.agents.items():
            agent_stats[agent_name] = {
                'status': agent_info.get('status', 'active'),
                'last_analysis': agent_info.get('last_analysis'),
                'total_analyses': agent_info.get('total_analyses', 0)
            }

        training_stats = get_training_summary_stats()

        return jsonify({
            **training_stats,
            'symbols_count': int(symbols_count),
            'total_records': int(records_count),
            'date_range': {
                'start': date_range['min_date'].iloc[0],
                'end': date_range['max_date'].iloc[0]
            },
            'agents': agent_stats,
            'data_source': 'database'
        })

    except Exception as e:
        log.error(f"Error getting summary: {e}")
        return jsonify({'error': str(e)}), 500

@bp.route("/live/<symbol>", methods=["GET"])
def get_live_predictions(symbol):
    """Get live predictions from all agents"""
    try:
        results = []

        # Only analyze with agentVD for now (LLM agents may timeout)
        for agent_name in ["agentVD"]:
            if agent_name in agent_manager.agents:
                try:
                    analysis = agent_manager.analyze_symbol(agent_name, symbol)

                    if analysis and 'error' not in analysis:
                        results.append(analysis)
                    else:
                        log.warning(f"Analysis error for {symbol} with {agent_name}: {analysis.get('error')}")
                except Exception as e:
                    log.error(f"Error analyzing {symbol} with {agent_name}: {e}")
                    continue

        return jsonify(results)

    except Exception as e:
        log.error(f"Error getting live predictions: {e}")
        return jsonify({'error': str(e)}), 500
