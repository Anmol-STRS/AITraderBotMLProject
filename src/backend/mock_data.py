# backend/mock_data.py
"""
Mock data generator for dashboard testing - uses real data from database
"""
import random
import sqlite3
from typing import List, Dict, Any
from datetime import datetime
import os

SYMBOLS = ['BMO.TO', 'BNS.TO', 'CM.TO', 'CNQ.TO', 'ENB.TO', 'RY.TO', 'SHOP.TO', 'SU.TO', 'TD.TO', 'TRP.TO']

# Database path
DB_PATH = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(__file__))), 'model_results.db')

def get_db_connection():
    """Get database connection"""
    return sqlite3.connect(DB_PATH)

def get_real_predictions_from_db(symbol: str, limit: int = 100, year: str = None) -> List[Dict[str, Any]]:
    """Get real predictions from database"""
    try:
        conn = get_db_connection()
        cursor = conn.cursor()

        # Get model_id for the symbol
        cursor.execute('SELECT model_id FROM models WHERE symbol = ? LIMIT 1', (symbol,))
        model_result = cursor.fetchone()

        if not model_result:
            conn.close()
            return []

        model_id = model_result[0]

        # Build query with optional year filter
        if year and year != 'all':
            query = '''
                SELECT timestamp, actual_value, predicted_value
                FROM predictions
                WHERE model_id = ? AND strftime('%Y', timestamp) = ?
                ORDER BY timestamp DESC
                LIMIT ?
            '''
            cursor.execute(query, (model_id, year, limit))
        else:
            query = '''
                SELECT timestamp, actual_value, predicted_value
                FROM predictions
                WHERE model_id = ?
                ORDER BY timestamp DESC
                LIMIT ?
            '''
            cursor.execute(query, (model_id, limit))

        predictions = cursor.fetchall()
        conn.close()

        return predictions
    except Exception as e:
        print(f"Error fetching predictions from DB: {e}")
        return []

def generate_mock_summary() -> Dict[str, Any]:
    """Generate mock summary statistics from real database"""
    try:
        conn = get_db_connection()
        cursor = conn.cursor()

        # Get actual statistics from database
        cursor.execute('SELECT COUNT(DISTINCT model_id) FROM models')
        total_models = cursor.fetchone()[0] or 10

        cursor.execute('SELECT COUNT(*) FROM predictions')
        total_records = cursor.fetchone()[0] or 12550

        cursor.execute('SELECT MIN(timestamp), MAX(timestamp) FROM predictions')
        date_range = cursor.fetchone()

        conn.close()

        return {
            'total_models': total_models,
            'avg_r2': 0.156,
            'avg_rmse': 2.34,
            'avg_mae': 1.82,
            'avg_direction': 52.4,
            'avg_mape': 3.2,
            'rmse_unit': 'percent',
            'symbols_count': len(SYMBOLS),
            'total_records': total_records,
            'data_source': 'database',
            'date_range': {
                'start': date_range[0][:10] if date_range[0] else '2019-11-27',
                'end': date_range[1][:10] if date_range[1] else '2025-12-07'
            },
            'agents': {
                'agentVD': {
                    'status': 'active',
                    'last_analysis': '2025-12-08T07:17:03.725286',
                    'total_analyses': 668
                },
                'gpt': {
                    'status': 'active',
                    'last_analysis': '2025-12-08T07:17:51.877306',
                    'total_analyses': 673
                },
                'claude': {
                    'status': 'active',
                    'last_analysis': '2025-12-08T07:17:51.967293',
                    'total_analyses': 673
                },
                'deepseek': {
                    'status': 'active',
                    'last_analysis': '2025-12-08T07:17:01.662266',
                    'total_analyses': 668
                }
            }
        }
    except Exception as e:
        print(f"Error generating summary: {e}")
        # Fallback to static mock data
        return {
            'total_models': 10,
            'avg_r2': 0.156,
            'avg_rmse': 2.34,
            'avg_mae': 1.82,
            'avg_direction': 52.4,
            'avg_mape': 3.2,
            'rmse_unit': 'percent',
            'symbols_count': len(SYMBOLS),
            'total_records': 12550,
            'data_source': 'mock_data',
            'date_range': {
                'start': '2019-11-27',
                'end': '2025-12-07'
            }
        }

def generate_mock_symbol_metrics(symbol: str) -> Dict[str, Any]:
    """Generate mock metrics for a single symbol from database"""
    try:
        conn = get_db_connection()
        cursor = conn.cursor()

        # Get model info
        cursor.execute('''
            SELECT m.model_id, m.model_name, m.model_type,
                   tr.test_r2, tr.test_rmse, tr.test_mae, tr.test_mape,
                   tr.test_direction_accuracy
            FROM models m
            LEFT JOIN training_results tr ON m.model_id = tr.model_id
            WHERE m.symbol = ?
            LIMIT 1
        ''', (symbol,))

        result = cursor.fetchone()

        if result:
            # Get latest prediction for price
            cursor.execute('''
                SELECT actual_value
                FROM predictions
                WHERE model_id = ?
                ORDER BY timestamp DESC
                LIMIT 1
            ''', (result[0],))
            price_result = cursor.fetchone()
            price = price_result[0] if price_result else random.uniform(50, 200)

            conn.close()

            return {
                'symbol': symbol,
                'price': round(price, 2),
                'volume': random.randint(1000000, 5000000),
                'last_updated': '2025-12-07',
                'total_candles': 1255,
                'test_r2': round(result[3], 6) if result[3] else round(random.uniform(-0.1, 0.3), 6),
                'test_rmse': round(result[4], 6) if result[4] else round(random.uniform(0.008, 0.015), 6),
                'test_mae': round(result[5], 6) if result[5] else round(random.uniform(0.006, 0.012), 6),
                'test_mape': round(result[6], 2) if result[6] else round(random.uniform(100, 160), 2),
                'test_direction_accuracy': round(result[7], 2) if result[7] else round(random.uniform(48, 55), 2),
                'model_name': result[1] or f'XGBoostImproved_{symbol}_ret_h1_20251208',
                'model_type': result[2] or 'XGBoost',
                'mode': 'returns_improved'
            }

        conn.close()
    except Exception as e:
        print(f"Error fetching symbol metrics for {symbol}: {e}")

    # Fallback to random mock data
    random.seed(hash(symbol))
    base_price = random.uniform(50, 200)
    r2 = random.uniform(-0.1, 0.3)
    rmse = random.uniform(0.008, 0.015)

    return {
        'symbol': symbol,
        'price': round(base_price, 2),
        'volume': random.randint(1000000, 5000000),
        'last_updated': '2025-12-07',
        'total_candles': 1255,
        'test_r2': round(r2, 6),
        'test_rmse': round(rmse, 6),
        'test_mae': round(rmse * 0.75, 6),
        'test_mape': round(random.uniform(100, 160), 2),
        'test_direction_accuracy': round(random.uniform(48, 55), 2),
        'model_name': f'XGBoostImproved_{symbol}_ret_h1_20251208',
        'model_type': 'XGBoost',
        'mode': 'returns_improved'
    }

def generate_mock_symbols() -> List[Dict[str, Any]]:
    """Generate mock data for all symbols"""
    return [generate_mock_symbol_metrics(symbol) for symbol in SYMBOLS]

def generate_mock_predictions(symbol: str, limit: int = 100, year: str = None) -> List[Dict[str, Any]]:
    """Generate mock prediction data for charts from real database"""
    # Try to get real predictions from database
    real_predictions = get_real_predictions_from_db(symbol, limit, year)

    if real_predictions:
        # Use real data and add mock predictions for other agents
        predictions = []
        random.seed(hash(symbol))

        for i, (timestamp, actual, predicted) in enumerate(real_predictions):
            # Parse date from timestamp
            date = timestamp[:10] if len(timestamp) > 10 else timestamp

            # Generate variations for other agents (±2-4 difference from XGBoost)
            predictions.append({
                'date': date,
                'actual': round(actual, 2),
                'agentVD': round(predicted, 2),  # XGBoost prediction
                'gpt': round(predicted + random.uniform(-4, 4), 2),
                'claude': round(predicted + random.uniform(-3.5, 3.5), 2),
                'deepseek': round(predicted + random.uniform(-4.5, 4.5), 2),
            })

        # Return in chronological order
        return predictions[::-1]

    # Fallback to fully random mock data if no real data available
    random.seed(hash(symbol))
    predictions = []
    base_price = random.uniform(50, 200)

    for i in range(limit):
        date = f"2025-{12 - (i // 30):02d}-{(i % 30) + 1:02d}"
        actual = base_price + random.uniform(-5, 5)
        predicted = actual + random.uniform(-2, 2)

        predictions.append({
            'date': date,
            'actual': round(actual, 2),
            'agentVD': round(predicted, 2),
            'gpt': round(predicted + random.uniform(-4, 4), 2),
            'claude': round(predicted + random.uniform(-3.5, 3.5), 2),
            'deepseek': round(predicted + random.uniform(-4.5, 4.5), 2),
        })

    return predictions[::-1]

def generate_mock_training_metrics(symbol: str) -> Dict[str, Any]:
    """Generate mock training metrics"""
    try:
        conn = get_db_connection()
        cursor = conn.cursor()

        cursor.execute('''
            SELECT m.model_id, m.model_name, m.model_type,
                   tr.test_r2, tr.test_rmse, tr.test_mae, tr.test_mape,
                   tr.test_direction_accuracy
            FROM models m
            LEFT JOIN training_results tr ON m.model_id = tr.model_id
            WHERE m.symbol = ?
            LIMIT 1
        ''', (symbol,))

        result = cursor.fetchone()

        if result:
            # Get top features
            cursor.execute('''
                SELECT feature_name, importance
                FROM feature_importance
                WHERE model_id = ?
                ORDER BY importance DESC
                LIMIT 10
            ''', (result[0],))
            features = cursor.fetchall()

            conn.close()

            return {
                'symbol': symbol,
                'horizon': 1,
                'model_id': result[0],
                'model_name': result[1] or f'XGBoostImproved_{symbol}_ret_h1_20251208',
                'model_type': result[2] or 'XGBoost',
                'test_r2': round(result[3], 6) if result[3] else 0.156,
                'test_rmse': round(result[4], 6) if result[4] else 2.34,
                'test_mae': round(result[5], 6) if result[5] else 1.82,
                'test_mape': round(result[6], 2) if result[6] else 125.0,
                'test_direction_accuracy': round(result[7], 2) if result[7] else 52.4,
                'total_features': len(features) if features else 38,
                'top_features': [
                    {'name': feat[0], 'importance': round(feat[1], 4)}
                    for feat in features
                ] if features else [
                    {'name': f'feature_{i}', 'importance': round(random.uniform(0.01, 0.15), 4)}
                    for i in range(10)
                ]
            }

        conn.close()
    except Exception as e:
        print(f"Error fetching training metrics for {symbol}: {e}")

    # Fallback to random mock
    random.seed(hash(symbol))
    r2 = random.uniform(-0.1, 0.3)
    rmse = random.uniform(0.008, 0.015)

    return {
        'symbol': symbol,
        'horizon': 1,
        'model_id': random.randint(1, 100),
        'model_name': f'XGBoostImproved_{symbol}_ret_h1_20251208',
        'model_type': 'XGBoost',
        'test_r2': round(r2, 6),
        'test_rmse': round(rmse, 6),
        'test_mae': round(rmse * 0.75, 6),
        'test_mape': round(random.uniform(100, 160), 2),
        'test_direction_accuracy': round(random.uniform(48, 55), 2),
        'total_features': 38,
        'top_features': [
            {'name': f'feature_{i}', 'importance': round(random.uniform(0.01, 0.15), 4)}
            for i in range(10)
        ]
    }

def generate_mock_live_predictions(symbol: str) -> List[Dict[str, Any]]:
    """Generate mock live predictions"""
    try:
        conn = get_db_connection()
        cursor = conn.cursor()

        # Get latest actual price from database
        cursor.execute('''
            SELECT p.actual_value
            FROM predictions p
            JOIN models m ON p.model_id = m.model_id
            WHERE m.symbol = ?
            ORDER BY p.timestamp DESC
            LIMIT 1
        ''', (symbol,))

        result = cursor.fetchone()
        current_price = result[0] if result else random.uniform(50, 200)

        conn.close()
    except Exception as e:
        print(f"Error fetching live predictions for {symbol}: {e}")
        random.seed(hash(symbol))
        current_price = random.uniform(50, 200)

    random.seed(hash(symbol))
    agents = ['agentVD', 'gpt', 'claude', 'deepseek']
    predictions = []

    for agent in agents:
        # Different prediction accuracy per agent
        if agent == 'agentVD':
            predicted = current_price + random.uniform(-2, 2)  # Best accuracy
        elif agent == 'gpt':
            predicted = current_price + random.uniform(-4, 4)
        elif agent == 'claude':
            predicted = current_price + random.uniform(-3.5, 3.5)
        else:  # deepseek
            predicted = current_price + random.uniform(-4.5, 4.5)

        change = ((predicted - current_price) / current_price) * 100

        predictions.append({
            'agent': agent,
            'symbol': symbol,
            'timestamp': '2025-12-08T07:17:00.000000',
            'current_price': round(current_price, 2),
            'predicted_price': round(predicted, 2),
            'predicted': round(predicted, 2),
            'actual': round(current_price, 2),
            'change_pct': round(change, 2),
            'change': round(change, 2),
            'confidence': round(random.uniform(0.6, 0.9), 2),
            'action': random.choice(['BUY', 'HOLD', 'SELL']),
            'signal': random.choice(['bullish', 'neutral', 'bearish']),
            'date': '2025-12-08T07:17:00.000000'
        })

    return predictions
