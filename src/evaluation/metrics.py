"""
Evaluation metrics for stock price prediction models
"""

import numpy as np
from typing import Dict, Optional
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score


def calculate_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    set_name: str = "test"
) -> Dict[str, float]:
    """
    Calculate comprehensive evaluation metrics for predictions.

    Args:
        y_true: Actual values
        y_pred: Predicted values
        set_name: Name of the dataset (train/val/test) for metric keys

    Returns:
        Dictionary with evaluation metrics:
        - {set_name}_rmse: Root Mean Squared Error
        - {set_name}_mae: Mean Absolute Error
        - {set_name}_r2: R-squared score
        - {set_name}_mape: Mean Absolute Percentage Error
        - {set_name}_direction_accuracy: Percentage of correct direction predictions
    """
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)

    # Basic regression metrics
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    mae = mean_absolute_error(y_true, y_pred)
    r2 = r2_score(y_true, y_pred)

    # Mean Absolute Percentage Error
    epsilon = 1e-6
    denom = np.clip(np.abs(y_true), epsilon, None)
    mape_errors = np.abs(y_true - y_pred) / denom
    mape = float(np.mean(np.nan_to_num(mape_errors, nan=0.0, posinf=1e6))) * 100

    # Direction accuracy (did we predict up/down correctly?)
    # Compare if actual and predicted both increased/decreased from previous value
    direction_correct = 0
    total_directions = 0

    for i in range(1, len(y_true)):
        actual_direction = 1 if y_true[i] > y_true[i-1] else 0
        pred_direction = 1 if y_pred[i] > y_pred[i-1] else 0

        if actual_direction == pred_direction:
            direction_correct += 1
        total_directions += 1

    direction_accuracy = (direction_correct / total_directions * 100) if total_directions > 0 else 0.0

    ramape_errors = np.abs(y_true - y_pred) / (np.abs(y_true) + np.abs(y_pred) + epsilon)
    ramape = float(np.mean(np.nan_to_num(ramape_errors, nan=0.0, posinf=1e6))) * 100
    mdae = np.median(np.abs(y_true - y_pred))

    return {
        f'{set_name}_rmse': rmse,
        f'{set_name}_mae': mae,
        f'{set_name}_mdae': mdae,
        f'{set_name}_r2': r2,
        f'{set_name}_mape': mape,
        f'{set_name}_ramape': ramape,
        f'{set_name}_direction_accuracy': direction_accuracy
    }


def calculate_all_metrics(
    y_train: np.ndarray,
    y_train_pred: np.ndarray,
    y_val: np.ndarray,
    y_val_pred: np.ndarray,
    y_test: np.ndarray,
    y_test_pred: np.ndarray
) -> Dict[str, float]:
    """
    Calculate metrics for train, validation, and test sets.

    Args:
        y_train: Training set actual values
        y_train_pred: Training set predictions
        y_val: Validation set actual values
        y_val_pred: Validation set predictions
        y_test: Test set actual values
        y_test_pred: Test set predictions

    Returns:
        Dictionary with all metrics for all sets
    """
    metrics = {}

    metrics.update(calculate_metrics(y_train, y_train_pred, 'train'))
    metrics.update(calculate_metrics(y_val, y_val_pred, 'val'))
    metrics.update(calculate_metrics(y_test, y_test_pred, 'test'))

    return metrics
