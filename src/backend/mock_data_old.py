# backend/mock_data.py
"""
Mock data generator for dashboard testing
"""
import random
from typing import List, Dict, Any

SYMBOLS = ['BMO.TO', 'BNS.TO', 'CM.TO', 'CNQ.TO', 'ENB.TO', 'RY.TO', 'SHOP.TO', 'SU.TO', 'TD.TO', 'TRP.TO']

def generate_mock_summary() -> Dict[str, Any]:
    """Generate mock summary statistics"""
    return {
        'total_models': 10,
        'avg_r2': 0.156,
        'avg_rmse': 2.34,
        'avg_mae': 1.82,
        'avg_direction': 52.4,
        'avg_mape': 3.2,
        'symbols_count': len(SYMBOLS),
        'total_records': 12550,
        'data_source': 'mock_data',
        'date_range': {
            'start': '2020-12-07',
            'end': '2025-12-05'
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

def generate_mock_symbol_metrics(symbol: str) -> Dict[str, Any]:
    """Generate mock metrics for a single symbol"""
    # Use symbol hash for consistent random values
    random.seed(hash(symbol))

    base_price = random.uniform(50, 200)
    r2 = random.uniform(-0.1, 0.3)
    rmse = random.uniform(0.008, 0.015)

    return {
        'symbol': symbol,
        'price': round(base_price, 2),
        'volume': random.randint(1000000, 5000000),
        'last_updated': '2025-12-05',
        'total_candles': 1255,
        'test_r2': round(r2, 6),
        'test_rmse': round(rmse, 6),
        'test_mae': round(rmse * 0.75, 6),
        'test_mape': round(random.uniform(100, 160), 2),
        'test_direction_accuracy': round(random.uniform(48, 55), 2),
        'model_name': f'XGBoostImproved_{symbol}_ret_h1_20251208',
        'model_type': 'XGBoost'
    }

def generate_mock_symbols() -> List[Dict[str, Any]]:
    """Generate mock data for all symbols"""
    return [generate_mock_symbol_metrics(symbol) for symbol in SYMBOLS]

def generate_mock_predictions(symbol: str, limit: int = 100) -> List[Dict[str, Any]]:
    """Generate mock prediction data for charts"""
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
            'gpt': round(predicted + random.uniform(-1, 1), 2) if i % 3 == 0 else None,
            'claude': round(predicted + random.uniform(-1, 1), 2) if i % 3 == 1 else None,
            'deepseek': round(predicted + random.uniform(-1, 1), 2) if i % 3 == 2 else None,
        })

    return predictions[::-1]  # Reverse to get chronological order

def generate_mock_training_metrics(symbol: str) -> Dict[str, Any]:
    """Generate mock training metrics"""
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
    random.seed(hash(symbol))

    base_price = random.uniform(50, 200)
    current_price = base_price + random.uniform(-2, 2)

    agents = ['agentVD', 'gpt', 'claude', 'deepseek']
    predictions = []

    for agent in agents:
        predicted = current_price + random.uniform(-3, 3)
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
