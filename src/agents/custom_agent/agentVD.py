from __future__ import annotations

import sys
import sqlite3
import logging
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import xgboost as xgb

from src.config.config import Config
from src.evaluation.metrics import calculate_all_metrics
from src.database.model_results_db import ModelResultsDB

PROJECT_ROOT = Path(__file__).resolve().parents[3]
DEFAULT_STOCKS_DB = PROJECT_ROOT / "data" / "storage" / "stocks.db"
DEFAULT_MODEL_RESULTS_DB = PROJECT_ROOT / "model_results.db"

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger("agentVD")


@dataclass
class TrainerParams:
    """Hyperparameters for XGBoost (tuned for noisy returns)."""

    n_estimators: int = 1500
    max_depth: int = 4
    learning_rate: float = 0.02
    subsample: float = 0.85
    colsample_bytree: float = 0.85
    min_child_weight: int = 6
    gamma: float = 0.2
    reg_alpha: float = 0.3
    reg_lambda: float = 1.2
    objective: str = "reg:squarederror"
    random_state: int = 42
    n_jobs: int = -1

    def to_dict(self) -> Dict:
        return self.__dict__.copy()


SYMBOL_SECTORS: Dict[str, str] = {
    "BMO.TO": "banks",
    "BNS.TO": "banks",
    "TD.TO": "banks",
    "RY.TO": "banks",
    "CM.TO": "banks",
    "ENB.TO": "energy",
    "TRP.TO": "energy",
    "CNQ.TO": "energy",
    "SU.TO": "energy",
    "SHOP.TO": "technology",
}

SECTOR_PARAMS: Dict[str, TrainerParams] = {
    "banks": TrainerParams(
        n_estimators=1800,
        max_depth=4,
        learning_rate=0.02,
        subsample=0.9,
        colsample_bytree=0.9,
        min_child_weight=6,
        gamma=0.2,
        reg_alpha=0.3,
        reg_lambda=1.1,
    ),
    "energy": TrainerParams(
        n_estimators=2200,
        max_depth=5,
        learning_rate=0.018,
        subsample=0.85,
        colsample_bytree=0.85,
        min_child_weight=4,
        gamma=0.1,
        reg_alpha=0.2,
        reg_lambda=1.0,
    ),
    "technology": TrainerParams(
        n_estimators=2000,
        max_depth=5,
        learning_rate=0.025,
        subsample=0.8,
        colsample_bytree=0.95,
        min_child_weight=4,
        gamma=0.15,
        reg_alpha=0.25,
        reg_lambda=1.0,
    ),
    "default": TrainerParams(),
}


def get_symbol_sector(symbol: str) -> str:
    """Return canonical sector label for a ticker."""
    if not symbol:
        return "default"
    return SYMBOL_SECTORS.get(symbol.upper(), "default")


def get_sector_parameters(symbol: str) -> TrainerParams:
    """Return a copy of the tuned hyperparameters for the symbol's sector."""
    sector = get_symbol_sector(symbol)
    base = SECTOR_PARAMS.get(sector, SECTOR_PARAMS["default"])
    # dataclasses.replace returns a shallow copy so downstream mutation is safe
    return replace(base)


def _ema(s: pd.Series, span: int) -> pd.Series:
    return s.ewm(span=span, adjust=False).mean()

def _rsi(close: pd.Series, period: int = 14) -> pd.Series:
    delta = close.diff()
    gain = delta.clip(lower=0)
    loss = (-delta).clip(lower=0)
    avg_gain = gain.rolling(period).mean()
    avg_loss = loss.rolling(period).mean()
    rs = avg_gain / (avg_loss.replace(0, np.nan))
    return 100 - (100 / (1 + rs))

def _atr(high: pd.Series, low: pd.Series, close: pd.Series, period: int = 14) -> pd.Series:
    prev_close = close.shift(1)
    tr = pd.concat([
        (high - low).abs(),
        (high - prev_close).abs(),
        (low - prev_close).abs(),
    ], axis=1).max(axis=1)
    return tr.rolling(period).mean()

def _obv(close: pd.Series, volume: pd.Series) -> pd.Series:
    direction = np.sign(close.diff()).fillna(0.0)
    return (direction * volume.fillna(0.0)).cumsum()



class UnifiedStockTrainer:
    """
    Improved (returns-only) XGBoost trainer.

    - Predicts log-returns over horizon
    - Leakage-safe feature shifting (default shift=1)
    - 70/15/15 temporal split
    - Early stopping on validation
    """

    def __init__(
        self,
        symbol: str,
        mode: str = "returns",  # kept for compatibility; blocked if not returns
        db_path: Optional[str] = None,
        model_results_db_path: Optional[str] = None,
        params: Optional[TrainerParams | Dict[str, float]] = None,
        feature_shift: int = 1,           # key: prevents accidental leakage
        clip_target_q: float = 0.995,     # clip extreme returns
        use_log_returns: bool = True,
    ) -> None:
        if mode != "returns":
            raise ValueError("This trainer is returns-only (Improved). Remove prices mode.")

        self.symbol = symbol
        self.feature_shift = int(feature_shift)
        self.clip_target_q = float(clip_target_q)
        self.use_log_returns = bool(use_log_returns)

        resolved_params = params or get_sector_parameters(symbol)
        if isinstance(resolved_params, TrainerParams):
            resolved_params = resolved_params.to_dict()
        else:
            resolved_params = dict(resolved_params)

        self.params = resolved_params
        self.model: Optional[xgb.XGBRegressor] = None
        self.feature_names: List[str] = []

        config = Config()
        db_cfg = config.database or {}
        default_db_path = db_cfg.get("path") or str(DEFAULT_STOCKS_DB)
        self.db_path = str(db_path or default_db_path)

        model_db_cfg = getattr(config, "model_results_db", {}) or {}
        default_model_results_db = model_db_cfg.get("path") or str(DEFAULT_MODEL_RESULTS_DB)
        self.model_results_db_path = str(model_results_db_path or default_model_results_db)
        self.results_db = ModelResultsDB(self.model_results_db_path)

        logger.info("Initialized Improved Returns Trainer for %s", symbol)

    def load_data(self, years: int) -> pd.DataFrame:
        db_path = Path(self.db_path)
        if not db_path.exists():
            raise FileNotFoundError(f"Database not found: {db_path}")

        conn = sqlite3.connect(str(db_path))
        query = """
            SELECT *
            FROM candle
            WHERE symbol = ?
              AND ts >= ?
            ORDER BY ts ASC
        """
        cutoff = (pd.Timestamp.utcnow() - pd.DateOffset(years=years)).strftime("%Y-%m-%d")
        df = pd.read_sql_query(query, conn, params=(self.symbol, cutoff))
        conn.close()

        if df.empty:
            raise ValueError(f"No data for {self.symbol} in last {years} years")

        df["ts"] = pd.to_datetime(df["ts"])
        # Basic cleanup
        df = df.drop_duplicates(subset=["symbol", "ts"]).sort_values("ts").reset_index(drop=True)

        logger.info(
            "Dataset for %s: %d rows (%s to %s)",
            self.symbol, len(df), df["ts"].min(), df["ts"].max()
        )
        return df

    def prepare_data(
        self,
        df: pd.DataFrame,
        target_col: str = "close",
        forecast_horizon: int = 1,
    ) -> Tuple[pd.DataFrame, pd.Series, pd.Series]:
        df = df.sort_values("ts").reset_index(drop=True)

        close = df[target_col].astype(float)
        high = df.get("high", close).astype(float)
        low = df.get("low", close).astype(float)
        volume = (
            df.get("volume", pd.Series(index=df.index, dtype=float))
            .astype(float)
            .fillna(0.0)
        )

        future = close.shift(-forecast_horizon)
        if self.use_log_returns:
            y = np.log(future / close)
        else:
            y = (future - close) / close

        y = y.replace([np.inf, -np.inf], np.nan)

        q = self.clip_target_q
        if 0.5 < q < 1.0:
            lo = y.quantile(1 - q)
            hi = y.quantile(q)
            y = y.clip(lo, hi)

        ret1 = close.pct_change(1)
        ret3 = close.pct_change(3)
        ret5 = close.pct_change(5)
        ret10 = close.pct_change(10)
        ret20 = close.pct_change(20)

        ema12 = _ema(close, 12)
        ema26 = _ema(close, 26)
        macd = ema12 - ema26
        macd_signal = _ema(macd, 9)
        macd_hist = macd - macd_signal

        feats = pd.DataFrame(index=df.index)
        feats["ret_1"] = ret1
        feats["ret_3"] = ret3
        feats["ret_5"] = ret5
        feats["ret_10"] = ret10
        feats["ret_20"] = ret20
        feats["momentum_20"] = close / close.shift(20) - 1.0

        feats["vol_5"] = ret1.rolling(5).std()
        feats["vol_20"] = ret1.rolling(20).std()
        feats["vol_60"] = ret1.rolling(60).std()

        feats["sma_5"] = close.rolling(5).mean()
        feats["sma_20"] = close.rolling(20).mean()
        feats["sma_60"] = close.rolling(60).mean()
        feats["std_20"] = close.rolling(20).std()
        feats["price_sma5_ratio"] = close / (feats["sma_5"] + 1e-9)
        feats["price_sma20_ratio"] = close / (feats["sma_20"] + 1e-9)
        feats["price_sma60_ratio"] = close / (feats["sma_60"] + 1e-9)

        feats["ema_12"] = ema12
        feats["ema_26"] = ema26
        feats["macd"] = macd
        feats["macd_signal"] = macd_signal
        feats["macd_hist"] = macd_hist
        feats["ema_ratio"] = ema12 / (ema26 + 1e-9)

        feats["rsi_7"] = _rsi(close, 7)
        feats["rsi_14"] = _rsi(close, 14)
        feats["atr_14"] = _atr(high, low, close, 14)
        feats["atr_norm"] = feats["atr_14"] / (close + 1e-9)

        feats["vol_mean_20"] = volume.rolling(20).mean()
        feats["vol_std_20"] = volume.rolling(20).std()
        feats["vol_chg_1"] = volume.pct_change(1)
        feats["vol_spike"] = volume / (feats["vol_mean_20"] + 1e-9)
        feats["vol_zscore"] = (volume - feats["vol_mean_20"]) / (feats["vol_std_20"] + 1e-9)
        feats["obv"] = _obv(close, volume)

        rolling_high_20 = high.rolling(20).max()
        rolling_low_20 = low.rolling(20).min()
        feats["pct_from_high_20"] = close / (rolling_high_20 + 1e-9) - 1.0
        feats["pct_from_low_20"] = close / (rolling_low_20 + 1e-9) - 1.0

        bollinger_width = 2 * feats["std_20"]
        feats["bb_width"] = bollinger_width / (feats["sma_20"] + 1e-9)
        feats["bb_percent_b"] = (close - feats["sma_20"]) / (bollinger_width + 1e-9)

        feats["dow"] = df["ts"].dt.dayofweek.astype(float)
        feats["month"] = df["ts"].dt.month.astype(float)

        if self.feature_shift > 0:
            feats = feats.shift(self.feature_shift)

        valid = feats.notna().all(axis=1) & y.notna()
        X = feats.loc[valid].copy()
        y = y.loc[valid].copy()
        timestamps = df.loc[valid, "ts"].copy()

        self.feature_names = list(X.columns)
        logger.info("Prepared data: %d samples, %d features", len(X), len(self.feature_names))
        return X, y, timestamps

    @staticmethod
    def temporal_split(
        X: pd.DataFrame,
        y: pd.Series,
        timestamps: pd.Series,
        train_ratio: float = 0.7,
        val_ratio: float = 0.15,
    ):
        n = len(X)
        train_size = int(n * train_ratio)
        val_size = int(n * val_ratio)

        X_train = X.iloc[:train_size]
        X_val = X.iloc[train_size : train_size + val_size]
        X_test = X.iloc[train_size + val_size :]

        y_train = y.iloc[:train_size]
        y_val = y.iloc[train_size : train_size + val_size]
        y_test = y.iloc[train_size + val_size :]

        ts_train = timestamps.iloc[:train_size]
        ts_val = timestamps.iloc[train_size : train_size + val_size]
        ts_test = timestamps.iloc[train_size + val_size :]

        logger.info(
            "Split: Train=%d, Val=%d, Test=%d",
            len(X_train), len(X_val), len(X_test)
        )
        return X_train, X_val, X_test, y_train, y_val, y_test, ts_train, ts_val, ts_test

    def train(
        self,
        X_train: pd.DataFrame,
        y_train: pd.Series,
        X_val: pd.DataFrame,
        y_val: pd.Series,
        early_stopping_rounds: int = 50,
    ) -> None:
        logger.info("Training XGBoost (Improved returns)...")

        self.model = xgb.XGBRegressor(
            **self.params,
             tree_method="hist",   # GPU-enabled histogram method
             device="cuda",        # or "cuda:0" for GPU 0
             eval_metric="rmse",
        )

        fit_kwargs = {
            "X": X_train,
            "y": y_train,
            "eval_set": [(X_val, y_val)],
            "verbose": False,
        }

        trained = False
        if early_stopping_rounds:
            try:
                self.model.fit(**fit_kwargs, early_stopping_rounds=early_stopping_rounds)
                trained = True
            except TypeError:
                callbacks = [
                    xgb.callback.EarlyStopping(
                        rounds=early_stopping_rounds,
                        metric_name="rmse",
                        data_name="validation_0",
                        save_best=True,
                    )
                ]
                try:
                    self.model.fit(**fit_kwargs, callbacks=callbacks)
                    trained = True
                except TypeError:
                    trained = False

        if not trained:
            self.model.fit(**fit_kwargs)

        best_iter = getattr(self.model, "best_iteration", None)
        logger.info("Training complete (best_iteration=%s)", best_iter)

    def evaluate(
        self,
        X_train: pd.DataFrame,
        y_train: pd.Series,
        X_val: pd.DataFrame,
        y_val: pd.Series,
        X_test: pd.DataFrame,
        y_test: pd.Series,
    ) -> Dict[str, float]:
        if self.model is None:
            raise RuntimeError("Model not trained")

        y_train_pred = self.model.predict(X_train)
        y_val_pred = self.model.predict(X_val)
        y_test_pred = self.model.predict(X_test)

        metrics = calculate_all_metrics(
            y_train=y_train.values,
            y_train_pred=y_train_pred,
            y_val=y_val.values,
            y_val_pred=y_val_pred,
            y_test=y_test.values,
            y_test_pred=y_test_pred,
        )
        return metrics

    def save_results(
        self,
        metrics: Dict[str, float],
        feature_importance: Optional[pd.DataFrame] = None,
        *,
        horizon: int = 1,
        train_samples: Optional[int] = None,
        val_samples: Optional[int] = None,
        test_samples: Optional[int] = None,
        model_path: Optional[str] = None,
        model_name: Optional[str] = None,
    ) -> None:
        if self.model is None:
            return

        if model_name is None:
            run_id = pd.Timestamp.utcnow().strftime("%Y%m%d_%H%M%S")
            model_name = f"XGBoostImproved_{self.symbol}_ret_h{horizon}_{run_id}"

        if feature_importance is None:
            booster = self.model.get_booster()
            scores = booster.get_score(importance_type="gain")
            feature_importance = (
                pd.DataFrame({"feature": list(scores.keys()), "gain": list(scores.values())})
                .sort_values("gain", ascending=False)
                .reset_index(drop=True)
            )

        self.results_db.save_model_results(
            symbol=self.symbol,
            mode="returns_improved",
            params=self.params,
            metrics=metrics,
            feature_importance=feature_importance,
            feature_names=self.feature_names,
            horizon=horizon,
            train_samples=train_samples,
            val_samples=val_samples,
            test_samples=test_samples,
            model_path=model_path,
            model_name=model_name,
        )
        logger.info("Saved results: %s", model_name)


class UnifiedMultiSymbolTrainer:
    """Batch trainer using the improved returns-only trainer."""

    def __init__(
        self,
        db_path: Optional[str] = None,
        model_results_db_path: Optional[str] = None,
    ) -> None:
        self.config = Config()
        db_cfg = self.config.database or {}
        default_db_path = db_cfg.get("path") or str(DEFAULT_STOCKS_DB)
        self.db_path = str(db_path or default_db_path)

        model_db_cfg = getattr(self.config, "model_results_db", {}) or {}
        default_model_results_db = model_db_cfg.get("path") or str(DEFAULT_MODEL_RESULTS_DB)
        self.model_results_db_path = str(model_results_db_path or default_model_results_db)

        self.errors: List[Dict[str, str]] = []
        self.sector_params = SECTOR_PARAMS
        self.symbol_sectors = SYMBOL_SECTORS

    def list_symbols(self) -> List[str]:
        conn = sqlite3.connect(self.db_path)
        query = "SELECT DISTINCT symbol FROM candle ORDER BY symbol"
        symbols = [row[0] for row in conn.execute(query)]
        conn.close()
        return symbols

    def train_symbol(
        self,
        symbol: str,
        years: int,
        forecast_horizon: int,
        save_results: bool = True,
    ) -> Optional[Dict]:
        try:
            logger.info("=== Training %s (Improved returns) ===", symbol)

            params = get_sector_parameters(symbol)

            trainer = UnifiedStockTrainer(
                symbol=symbol,
                mode="returns",
                db_path=self.db_path,
                model_results_db_path=self.model_results_db_path,
                feature_shift=1,
                use_log_returns=True,
                params=params,
            )

            df = trainer.load_data(years=years)
            X, y, ts = trainer.prepare_data(df, target_col="close", forecast_horizon=forecast_horizon)

            X_train, X_val, X_test, y_train, y_val, y_test, *_ = trainer.temporal_split(X, y, ts)

            trainer.train(X_train, y_train, X_val, y_val, early_stopping_rounds=50)
            metrics = trainer.evaluate(X_train, y_train, X_val, y_val, X_test, y_test)

            if save_results:
                booster = trainer.model.get_booster()
                scores = booster.get_score(importance_type="gain")
                importance = (
                    pd.DataFrame({"feature": list(scores.keys()), "gain": list(scores.values())})
                    .sort_values("gain", ascending=False)
                    .reset_index(drop=True)
                )

                trainer.save_results(
                    metrics,
                    importance,
                    horizon=forecast_horizon,
                    train_samples=len(X_train),
                    val_samples=len(X_val),
                    test_samples=len(X_test),
                )

            result = {
                "symbol": symbol,
                "features": len(trainer.feature_names),
                **{k: v for k, v in metrics.items() if k.startswith("val_")},
                **{k: v for k, v in metrics.items() if k.startswith("test_")},
            }
            logger.info("SUCCESS: %s", symbol)
            return result

        except Exception as exc:
            logger.error("FAILED: %s - %s", symbol, exc)
            self.errors.append({"symbol": symbol, "error": str(exc)})
            return None

    def train_all(
        self,
        years: int = 10,
        forecast_horizon: int = 1,
        save_results: bool = True,
    ) -> List[Dict]:
        symbols = self.list_symbols()
        logger.info("Training %d symbols (Improved returns)...", len(symbols))

        results: List[Dict] = []
        for symbol in symbols:
            res = self.train_symbol(symbol, years=years, forecast_horizon=forecast_horizon, save_results=save_results)
            if res is not None:
                results.append(res)

        logger.info("Finished: %d success, %d errors", len(results), len(self.errors))
        return results


def parse_args(argv: List[str]) -> Dict:
    args = {"years": 10, "horizon": 1, "all": False, "symbol": None}

    if "--years" in argv:
        args["years"] = int(argv[argv.index("--years") + 1])
    if "--horizon" in argv:
        args["horizon"] = int(argv[argv.index("--horizon") + 1])
    if "--all" in argv:
        args["all"] = True
    if "--symbol" in argv:
        args["symbol"] = argv[argv.index("--symbol") + 1]

    # Hard block old flag if someone tries it
    if "--mode" in argv:
        raise ValueError("This trainer is returns-only. Remove --mode entirely.")

    return args


def main() -> None:

    args = parse_args(sys.argv[1:])
    years = args["years"]
    horizon = args["horizon"]

    if args["all"]:
        multi_trainer = UnifiedMultiSymbolTrainer()
        results = multi_trainer.train_all(years=years, forecast_horizon=horizon, save_results=True)

        print("\nSummary:")
        for r in results:
            print(f"{r['symbol']}: test_RMSE={r.get('test_rmse')}, test_R2={r.get('test_r2')}")

        if multi_trainer.errors:
            print("\nErrors:")
            for e in multi_trainer.errors:
                print(f"{e['symbol']}: {e['error']}")

    elif args["symbol"]:
        multi_trainer = UnifiedMultiSymbolTrainer()
        res = multi_trainer.train_symbol(symbol=args["symbol"], years=years, forecast_horizon=horizon, save_results=True)
        print("\nResult:")
        print(res)
    else:
        print("You must pass either --all or --symbol SYMBOL")
        sys.exit(1)


if __name__ == "__main__":
    main()
