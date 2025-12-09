"""
Model Results Database
Stores training inputs, predictions, and metadata for all ML models
"""

import sqlite3
import pandas as pd
import json
from pathlib import Path
from datetime import datetime
from typing import Dict, Any, List, Optional, Tuple
import logging

logger = logging.getLogger(__name__)


class ModelResultsDB:
    """Database for storing and retrieving ML model results."""

    def __init__(self, db_path: str = "model_results.db"):
        """
        Initialize the model results database.

        Args:
            db_path: Path to the SQLite database file
        """
        self.db_path = str(db_path)
        self.conn = None
        self._init_database()

    def _init_database(self):
        """Initialize database schema."""
        self.conn = sqlite3.connect(self.db_path)
        cursor = self.conn.cursor()

        # Table 1: Models - Stores model metadata
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS models (
                model_id INTEGER PRIMARY KEY AUTOINCREMENT,
                model_name TEXT NOT NULL,
                model_type TEXT NOT NULL,
                symbol TEXT,
                horizon INTEGER,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                model_path TEXT,
                hyperparameters TEXT,
                feature_count INTEGER,
                description TEXT,
                UNIQUE(model_name, symbol, horizon)
            )
        """)

        # Table 2: Training Results - Stores training metrics
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS training_results (
                result_id INTEGER PRIMARY KEY AUTOINCREMENT,
                model_id INTEGER NOT NULL,
                train_samples INTEGER,
                test_samples INTEGER,
                train_rmse REAL,
                train_mae REAL,
                train_r2 REAL,
                test_rmse REAL,
                test_mae REAL,
                test_r2 REAL,
                test_mape REAL,
                test_direction_accuracy REAL,
                top_feature TEXT,
                training_duration REAL,
                trained_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                additional_metrics TEXT,
                FOREIGN KEY (model_id) REFERENCES models(model_id)
            )
        """)

        # Table 3: Predictions - Stores individual predictions
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS predictions (
                prediction_id INTEGER PRIMARY KEY AUTOINCREMENT,
                model_id INTEGER NOT NULL,
                timestamp TEXT NOT NULL,
                actual_value REAL,
                predicted_value REAL NOT NULL,
                error REAL,
                error_pct REAL,
                prediction_type TEXT DEFAULT 'test',
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                FOREIGN KEY (model_id) REFERENCES models(model_id)
            )
        """)

        # Table 4: Input Features - Stores input data used for predictions
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS input_features (
                input_id INTEGER PRIMARY KEY AUTOINCREMENT,
                prediction_id INTEGER NOT NULL,
                feature_name TEXT NOT NULL,
                feature_value REAL NOT NULL,
                FOREIGN KEY (prediction_id) REFERENCES predictions(prediction_id)
            )
        """)

        # Table 5: Feature Importance - Stores feature importance scores
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS feature_importance (
                importance_id INTEGER PRIMARY KEY AUTOINCREMENT,
                model_id INTEGER NOT NULL,
                feature_name TEXT NOT NULL,
                importance_score REAL NOT NULL,
                rank INTEGER,
                FOREIGN KEY (model_id) REFERENCES models(model_id)
            )
        """)

        # Create indexes for faster queries
        cursor.execute("""
            CREATE INDEX IF NOT EXISTS idx_model_symbol
            ON models(symbol)
        """)

        cursor.execute("""
            CREATE INDEX IF NOT EXISTS idx_predictions_model
            ON predictions(model_id)
        """)

        cursor.execute("""
            CREATE INDEX IF NOT EXISTS idx_predictions_timestamp
            ON predictions(timestamp)
        """)

        cursor.execute("""
            CREATE INDEX IF NOT EXISTS idx_input_features_prediction
            ON input_features(prediction_id)
        """)

        self.conn.commit()
        logger.info(f"Model results database initialized at {self.db_path}")

    def add_model(self, model_name: str, model_type: str, symbol: str = None,
                  horizon: int = 1, model_path: str = None,
                  hyperparameters: Dict = None, feature_count: int = None,
                  description: str = None) -> int:
        """
        Add a new model to the database.

        Args:
            model_name: Name of the model
            model_type: Type of model (e.g., 'XGBoost', 'LSTM', 'RandomForest')
            symbol: Stock symbol (if applicable)
            horizon: Prediction horizon
            model_path: Path to saved model file
            hyperparameters: Dictionary of hyperparameters
            feature_count: Number of features used
            description: Model description

        Returns:
            model_id: ID of the inserted/updated model
        """
        cursor = self.conn.cursor()

        hyperparams_json = json.dumps(hyperparameters) if hyperparameters else None

        cursor.execute("""
            INSERT INTO models
            (model_name, model_type, symbol, horizon, model_path,
             hyperparameters, feature_count, description)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?)
            ON CONFLICT(model_name, symbol, horizon)
            DO UPDATE SET
                model_type = excluded.model_type,
                model_path = excluded.model_path,
                hyperparameters = excluded.hyperparameters,
                feature_count = excluded.feature_count,
                description = excluded.description,
                created_at = CURRENT_TIMESTAMP
        """, (model_name, model_type, symbol, horizon, model_path,
              hyperparams_json, feature_count, description))

        self.conn.commit()

        # Get the model_id
        cursor.execute("""
            SELECT model_id FROM models
            WHERE model_name = ? AND symbol = ? AND horizon = ?
        """, (model_name, symbol, horizon))

        model_id = cursor.fetchone()[0]
        logger.info(f"Model added/updated: {model_name} (ID: {model_id})")
        return model_id

    def add_training_result(self, model_id: int, train_samples: int,
                           test_samples: int, metrics: Dict[str, float],
                           top_feature: str = None, training_duration: float = None,
                           additional_metrics: Dict = None) -> int:
        """
        Add training results for a model.

        Args:
            model_id: ID of the model
            train_samples: Number of training samples
            test_samples: Number of test samples
            metrics: Dictionary containing metrics (train_rmse, test_rmse, etc.)
            top_feature: Most important feature
            training_duration: Training time in seconds
            additional_metrics: Additional metrics as dictionary

        Returns:
            result_id: ID of the inserted result
        """
        cursor = self.conn.cursor()

        additional_json = json.dumps(additional_metrics) if additional_metrics else None

        cursor.execute("""
            INSERT INTO training_results
            (model_id, train_samples, test_samples, train_rmse, train_mae, train_r2,
             test_rmse, test_mae, test_r2, test_mape, test_direction_accuracy,
             top_feature, training_duration, additional_metrics)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """, (
            model_id, train_samples, test_samples,
            metrics.get('train_rmse'), metrics.get('train_mae'), metrics.get('train_r2'),
            metrics.get('test_rmse'), metrics.get('test_mae'), metrics.get('test_r2'),
            metrics.get('test_mape'), metrics.get('test_direction_accuracy'),
            top_feature, training_duration, additional_json
        ))

        self.conn.commit()
        result_id = cursor.lastrowid
        logger.info(f"Training result added for model_id {model_id} (result_id: {result_id})")
        return result_id

    def add_predictions(self, model_id: int, predictions_df: pd.DataFrame,
                       prediction_type: str = 'test') -> List[int]:
        """
        Add predictions to the database.

        Args:
            model_id: ID of the model
            predictions_df: DataFrame with columns: ts, actual, predicted, error, error_pct
            prediction_type: Type of prediction ('test', 'validation', 'production')

        Returns:
            List of prediction_ids
        """
        cursor = self.conn.cursor()
        prediction_ids = []

        for _, row in predictions_df.iterrows():
            cursor.execute("""
                INSERT INTO predictions
                (model_id, timestamp, actual_value, predicted_value,
                 error, error_pct, prediction_type)
                VALUES (?, ?, ?, ?, ?, ?, ?)
            """, (
                model_id, str(row['ts']),
                row.get('actual'), row['predicted'],
                row.get('error'), row.get('error_pct'),
                prediction_type
            ))
            prediction_ids.append(cursor.lastrowid)

        self.conn.commit()
        logger.info(f"Added {len(prediction_ids)} predictions for model_id {model_id}")
        return prediction_ids

    def add_input_features(self, prediction_id: int, features: Dict[str, float]):
        """
        Add input features for a specific prediction.

        Args:
            prediction_id: ID of the prediction
            features: Dictionary of feature_name: feature_value
        """
        cursor = self.conn.cursor()

        for feature_name, feature_value in features.items():
            cursor.execute("""
                INSERT INTO input_features (prediction_id, feature_name, feature_value)
                VALUES (?, ?, ?)
            """, (prediction_id, feature_name, feature_value))

        self.conn.commit()
        logger.info(f"Added {len(features)} features for prediction_id {prediction_id}")

    def add_feature_importance(self, model_id: int, importance_dict: Dict[str, float]):
        """
        Add feature importance scores for a model.

        Args:
            model_id: ID of the model
            importance_dict: Dictionary of feature_name: importance_score
        """
        cursor = self.conn.cursor()

        # Sort by importance and add rank
        sorted_features = sorted(importance_dict.items(),
                                key=lambda x: x[1], reverse=True)

        for rank, (feature_name, score) in enumerate(sorted_features, 1):
            cursor.execute("""
                INSERT INTO feature_importance
                (model_id, feature_name, importance_score, rank)
                VALUES (?, ?, ?, ?)
            """, (model_id, feature_name, score, rank))

        self.conn.commit()
        logger.info(f"Added feature importance for model_id {model_id}")

    def save_model_results(
        self,
        symbol: str,
        mode: str,
        params: Optional[Dict[str, Any]] = None,
        metrics: Optional[Dict[str, float]] = None,
        feature_importance: Optional[pd.DataFrame] = None,
        feature_names: Optional[List[str]] = None,
        horizon: int = 1,
        train_samples: Optional[int] = None,
        val_samples: Optional[int] = None,
        test_samples: Optional[int] = None,
        model_name: Optional[str] = None,
        model_path: Optional[str] = None,
    ) -> int:
        """
        Persist model metadata, training metrics, and feature importance.

        Returns:
            model_id of the upserted model.
        """
        metrics = metrics or {}
        params = params or {}

        cursor = self.conn.cursor()
        resolved_model_name = model_name or f"XGBoost_{symbol}_{mode}_h{horizon}"
        feature_count = len(feature_names) if feature_names else None
        hyperparams_json = json.dumps(params) if params else None

        cursor.execute(
            """
            INSERT INTO models
            (model_name, model_type, symbol, horizon, model_path,
             hyperparameters, feature_count, description)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?)
            ON CONFLICT(model_name, symbol, horizon)
            DO UPDATE SET
                model_type = excluded.model_type,
                model_path = excluded.model_path,
                hyperparameters = excluded.hyperparameters,
                feature_count = excluded.feature_count,
                description = excluded.description,
                created_at = CURRENT_TIMESTAMP
            """,
            (
                resolved_model_name,
                "XGBoost",
                symbol,
                horizon,
                model_path,
                hyperparams_json,
                feature_count,
                f"{mode} model trained via UnifiedStockTrainer",
            ),
        )
        self.conn.commit()

        cursor.execute(
            """
            SELECT model_id FROM models
            WHERE model_name = ? AND symbol = ? AND horizon = ?
            """,
            (resolved_model_name, symbol, horizon),
        )
        row = cursor.fetchone()
        if not row:
            raise RuntimeError(f"Failed to retrieve model_id for {resolved_model_name}")
        model_id = row[0]

        training_metrics = {
            "train_rmse": metrics.get("train_rmse"),
            "train_mae": metrics.get("train_mae"),
            "train_r2": metrics.get("train_r2"),
            "test_rmse": metrics.get("test_rmse"),
            "test_mae": metrics.get("test_mae"),
            "test_r2": metrics.get("test_r2"),
            "test_mape": metrics.get("test_mape"),
            "test_direction_accuracy": metrics.get("test_direction_accuracy"),
        }
        val_metrics = {k: v for k, v in metrics.items() if k.startswith("val_")}
        additional_metrics = {
            **val_metrics,
            "train_mdae": metrics.get("train_mdae"),
            "train_ramape": metrics.get("train_ramape"),
            "test_mdae": metrics.get("test_mdae"),
            "test_ramape": metrics.get("test_ramape"),
            "mode": mode,
            "train_samples": train_samples,
            "val_samples": val_samples,
            "test_samples": test_samples,
        }
        additional_metrics = {
            k: v for k, v in additional_metrics.items() if v is not None
        }

        top_feature = None
        if feature_importance is not None and not feature_importance.empty:
            top_feature = str(feature_importance.iloc[0]["feature"])

        self.add_training_result(
            model_id=model_id,
            train_samples=train_samples,
            test_samples=test_samples,
            metrics=training_metrics,
            top_feature=top_feature,
            additional_metrics=additional_metrics or None,
        )

        if feature_importance is not None and not feature_importance.empty:
            cursor.execute(
                "DELETE FROM feature_importance WHERE model_id = ?", (model_id,)
            )
            importance_dict = {
                str(row["feature"]): float(row.get("gain", row.get("importance", 0.0)))
                for _, row in feature_importance.iterrows()
            }
            self.add_feature_importance(model_id, importance_dict)

        logger.info(
            "Saved model %s (id=%s, mode=%s, horizon=%s)", resolved_model_name, model_id, mode, horizon
        )
        return model_id

    def get_model_by_name(self, model_name: str, symbol: str = None,
                         horizon: int = 1) -> Optional[Dict]:
        """Get model information by name."""
        cursor = self.conn.cursor()
        cursor.execute("""
            SELECT * FROM models
            WHERE model_name = ? AND symbol = ? AND horizon = ?
        """, (model_name, symbol, horizon))

        row = cursor.fetchone()
        if row:
            columns = [desc[0] for desc in cursor.description]
            return dict(zip(columns, row))
        return None

    def get_all_models(self) -> pd.DataFrame:
        """Get all models as DataFrame."""
        return pd.read_sql_query("SELECT * FROM models ORDER BY created_at DESC", self.conn)

    def get_model_results(self, model_id: int) -> pd.DataFrame:
        """Get all training results for a model."""
        return pd.read_sql_query("""
            SELECT * FROM training_results
            WHERE model_id = ?
            ORDER BY trained_at DESC
        """, self.conn, params=(model_id,))

    def get_predictions(self, model_id: int, limit: int = None) -> pd.DataFrame:
        """Get predictions for a model."""
        query = """
            SELECT * FROM predictions
            WHERE model_id = ?
            ORDER BY timestamp DESC
        """
        if limit:
            query += f" LIMIT {limit}"

        return pd.read_sql_query(query, self.conn, params=(model_id,))

    def get_feature_importance(self, model_id: int) -> pd.DataFrame:
        """Get feature importance for a model."""
        return pd.read_sql_query("""
            SELECT * FROM feature_importance
            WHERE model_id = ?
            ORDER BY rank
        """, self.conn, params=(model_id,))

    def get_models_by_symbol(self, symbol: str) -> pd.DataFrame:
        """Get all models for a specific symbol."""
        return pd.read_sql_query("""
            SELECT m.*,
                   tr.test_rmse,
                   tr.test_mae,
                   tr.test_r2,
                   tr.test_mape,
                   tr.test_direction_accuracy,
                   tr.additional_metrics
            FROM models m
            LEFT JOIN (
                SELECT model_id, test_rmse, test_mae, test_r2, test_mape, test_direction_accuracy, additional_metrics
                FROM training_results
                WHERE (model_id, trained_at) IN (
                    SELECT model_id, MAX(trained_at)
                    FROM training_results
                    GROUP BY model_id
                )
            ) tr ON m.model_id = tr.model_id
            WHERE m.symbol = ?
            ORDER BY m.created_at DESC
        """, self.conn, params=(symbol,))

    def get_model_summary(self) -> pd.DataFrame:
        """Get summary of all models with their latest valid metrics."""
        return pd.read_sql_query("""
            SELECT
                m.model_id,
                m.model_name,
                m.model_type,
                m.symbol,
                m.horizon,
                m.feature_count,
                tr.test_rmse,
                tr.test_mae,
                tr.test_r2,
                tr.test_mape,
                tr.test_direction_accuracy,
                tr.top_feature,
                m.created_at,
                COUNT(DISTINCT p.prediction_id) as total_predictions
            FROM models m
            INNER JOIN (
                SELECT model_id, test_rmse, test_mae, test_r2, test_mape, test_direction_accuracy, top_feature
                FROM training_results
                WHERE test_r2 IS NOT NULL
                  AND (model_id, trained_at) IN (
                    SELECT model_id, MAX(trained_at)
                    FROM training_results
                    WHERE test_r2 IS NOT NULL
                    GROUP BY model_id
                )
            ) tr ON m.model_id = tr.model_id
            LEFT JOIN predictions p ON m.model_id = p.model_id
            GROUP BY m.model_id
            ORDER BY m.created_at DESC
        """, self.conn)

    def compare_models(self, symbol: str = None) -> pd.DataFrame:
        """
        Compare models by their performance metrics.

        Args:
            symbol: Filter by symbol (optional)

        Returns:
            DataFrame with model comparison
        """
        query = """
            SELECT
                m.model_name,
                m.model_type,
                m.symbol,
                tr.test_rmse,
                tr.test_mae,
                tr.test_r2,
                tr.test_mape,
                tr.test_direction_accuracy,
                tr.train_samples,
                tr.test_samples,
                m.feature_count
            FROM models m
            JOIN training_results tr ON m.model_id = tr.model_id
        """

        if symbol:
            query += f" WHERE m.symbol = '{symbol}'"

        query += " ORDER BY tr.test_r2 DESC"

        return pd.read_sql_query(query, self.conn)

    def export_model_data(self, model_id: int, output_dir: str = "exports"):
        """
        Export all data for a specific model.

        Args:
            model_id: ID of the model
            output_dir: Directory to save exports
        """
        output_path = Path(output_dir)
        output_path.mkdir(exist_ok=True)

        # Get model info
        model_info = pd.read_sql_query(
            "SELECT * FROM models WHERE model_id = ?",
            self.conn, params=(model_id,)
        )

        if model_info.empty:
            logger.error(f"Model {model_id} not found")
            return

        model_name = model_info.iloc[0]['model_name']
        symbol = model_info.iloc[0]['symbol']

        # Export model info
        model_info.to_csv(output_path / f"{model_name}_{symbol}_info.csv", index=False)

        # Export training results
        results = self.get_model_results(model_id)
        results.to_csv(output_path / f"{model_name}_{symbol}_results.csv", index=False)

        # Export predictions
        predictions = self.get_predictions(model_id)
        predictions.to_csv(output_path / f"{model_name}_{symbol}_predictions.csv", index=False)

        # Export feature importance
        importance = self.get_feature_importance(model_id)
        importance.to_csv(output_path / f"{model_name}_{symbol}_importance.csv", index=False)

        logger.info(f"Exported model {model_id} data to {output_path}")
        print(f"✅ Exported model data to {output_path}")

    def delete_model(self, model_id: int):
        """Delete a model and all associated data."""
        cursor = self.conn.cursor()

        # Delete in order due to foreign key constraints
        cursor.execute("DELETE FROM input_features WHERE prediction_id IN (SELECT prediction_id FROM predictions WHERE model_id = ?)", (model_id,))
        cursor.execute("DELETE FROM predictions WHERE model_id = ?", (model_id,))
        cursor.execute("DELETE FROM feature_importance WHERE model_id = ?", (model_id,))
        cursor.execute("DELETE FROM training_results WHERE model_id = ?", (model_id,))
        cursor.execute("DELETE FROM models WHERE model_id = ?", (model_id,))

        self.conn.commit()
        logger.info(f"Deleted model {model_id} and all associated data")

    def close(self):
        """Close database connection."""
        if self.conn:
            self.conn.close()
            logger.info("Database connection closed")

    def __enter__(self):
        """Context manager entry."""
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit."""
        self.close()
