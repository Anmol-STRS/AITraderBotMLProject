"""
Pipeline runner that ingests TSX data, cleans/structures it, runs agentVD,
and produces mock GPT/Claude analyses that get persisted to model_results.db.
"""

from __future__ import annotations

import json
import logging
import sqlite3
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
import yfinance as yf

from src.config.config import Config
from src.processing.procesingpip import DataPipeline
from src.agents.custom_agent.agentVD import UnifiedMultiSymbolTrainer


TSX_TOP_10 = [
    "RY.TO",
    "TD.TO",
    "BNS.TO",
    "BMO.TO",
    "CM.TO",
    "ENB.TO",
    "CNQ.TO",
    "SHOP.TO",
    "SU.TO",
    "TRP.TO",
]


LOG_FILE = Path("runner.log")


@dataclass
class IngestionResult:
    symbol: str
    rows_inserted: int
    start: str
    end: str


def reset_stocks_storage(db_path: Path) -> None:
    """Drop historical data so ingestion starts from a clean slate."""
    logger = logging.getLogger("runner")
    db_path = Path(db_path)
    db_path.parent.mkdir(parents=True, exist_ok=True)

    conn = sqlite3.connect(db_path)
    ensure_tables(conn)
    cur = conn.cursor()
    for table in ("candle", "structured_data"):
        cur.execute(f"DELETE FROM {table}")
    conn.commit()
    conn.close()
    logger.info("Cleared candle and structured_data tables at %s", db_path)


def ensure_tables(conn: sqlite3.Connection) -> None:
    """Create required tables when they do not yet exist."""
    logger = logging.getLogger("runner")
    cur = conn.cursor()
    cur.execute(
        """
        CREATE TABLE IF NOT EXISTS candle (
            symbol TEXT NOT NULL,
            ts TEXT NOT NULL,
            open REAL,
            high REAL,
            low REAL,
            close REAL,
            volume REAL,
            timeframe TEXT NOT NULL,
            PRIMARY KEY (symbol, ts)
        )
    """
    )
    cur.execute(
        """
        CREATE TABLE IF NOT EXISTS structured_data (
            symbol TEXT NOT NULL,
            ts TEXT NOT NULL,
            close REAL NOT NULL,
            return_1d REAL,
            return_5d REAL,
            sma_5 REAL,
            sma_20 REAL,
            ema_12 REAL,
            ema_26 REAL,
            macd REAL,
            rsi_14 REAL,
            volatility_20d REAL,
            volume_avg_20d REAL,
            PRIMARY KEY (symbol, ts)
        )
    """
    )
    conn.commit()
    logger.debug("Ensured candle and structured_data tables exist.")


def fetch_symbol_history(symbol: str, period: str = "5y") -> pd.DataFrame:
    """Download historical candles for a single TSX symbol."""
    logger = logging.getLogger("runner")
    logger.info("Fetching history for %s (period=%s)", symbol, period)
    df = yf.download(
        symbol,
        period=period,
        interval="1d",
        auto_adjust=False,
        progress=False,
        threads=False,
    )
    if df.empty:
        return df
    df = df.reset_index().rename(
        columns={
            "Date": "ts",
            "Open": "open",
            "High": "high",
            "Low": "low",
            "Close": "close",
            "Adj Close": "adj_close",
            "Volume": "volume",
        }
    )
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = [col[0] for col in df.columns]
    df["symbol"] = symbol
    df["timeframe"] = "1d"
    return df[["symbol", "ts", "open", "high", "low", "close", "volume", "timeframe"]]


def ingest_top_tsx_symbols(db_path: Path) -> List[IngestionResult]:
    """Fetch top TSX symbols and store them in the stocks DB."""
    logger = logging.getLogger("runner")
    logger.info("Starting TSX ingestion into %s", db_path)
    conn = sqlite3.connect(db_path)
    ensure_tables(conn)
    cur = conn.cursor()

    results: List[IngestionResult] = []
    for symbol in TSX_TOP_10:
        df = fetch_symbol_history(symbol)
        if df.empty:
            logger.warning("No data returned for %s", symbol)
            continue
        df["ts"] = pd.to_datetime(df["ts"]).dt.tz_localize(None).astype(str)
        payload = [
            (
                row.symbol,
                row.ts,
                float(row.open),
                float(row.high),
                float(row.low),
                float(row.close),
                float(row.volume),
                row.timeframe,
            )
            for row in df.itertuples(index=False)
        ]
        cur.executemany(
            """
            INSERT INTO candle (symbol, ts, open, high, low, close, volume, timeframe)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?)
            ON CONFLICT(symbol, ts) DO UPDATE SET
                open=excluded.open,
                high=excluded.high,
                low=excluded.low,
                close=excluded.close,
                volume=excluded.volume,
                timeframe=excluded.timeframe
        """,
            payload,
        )
        conn.commit()
        results.append(
            IngestionResult(
                symbol=symbol,
                rows_inserted=len(payload),
                start=df["ts"].min(),
                end=df["ts"].max(),
            )
        )
    conn.close()
    logger.info("Completed ingestion for %d symbols", len(results))
    return results


def compute_rsi(series: pd.Series, window: int = 14) -> pd.Series:
    """Standard RSI implementation for feature engineering."""
    delta = series.diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)
    avg_gain = gain.rolling(window=window).mean()
    avg_loss = loss.rolling(window=window).mean()
    rs = avg_gain / avg_loss.replace(0, np.nan)
    rsi = 100 - (100 / (1 + rs))
    return rsi


def build_structured_dataset(clean_df: pd.DataFrame) -> pd.DataFrame:
    """Create engineered features for models + LLMs."""
    logger = logging.getLogger("runner")
    if clean_df is None or clean_df.empty:
        logger.warning("No cleaned data available; skipping structuring.")
        return pd.DataFrame()

    logger.info("Building structured dataset from %d rows", len(clean_df))
    df = clean_df.sort_values(["symbol", "ts"]).copy()
    df["return_1d"] = df.groupby("symbol")["close"].pct_change()
    df["return_5d"] = df.groupby("symbol")["close"].pct_change(5)
    df["sma_5"] = df.groupby("symbol")["close"].rolling(5).mean().reset_index(level=0, drop=True)
    df["sma_20"] = df.groupby("symbol")["close"].rolling(20).mean().reset_index(level=0, drop=True)
    df["ema_12"] = (
        df.groupby("symbol")["close"].transform(lambda x: x.ewm(span=12, adjust=False).mean())
    )
    df["ema_26"] = (
        df.groupby("symbol")["close"].transform(lambda x: x.ewm(span=26, adjust=False).mean())
    )
    df["macd"] = df["ema_12"] - df["ema_26"]
    df["rsi_14"] = df.groupby("symbol")["close"].transform(compute_rsi)
    df["volatility_20d"] = (
        df.groupby("symbol")["return_1d"]
        .transform(lambda x: x.rolling(20).std() * np.sqrt(252))
    )
    df["volume_avg_20d"] = df.groupby("symbol")["volume"].transform(lambda x: x.rolling(20).mean())

    features = df[
        [
            "symbol",
            "ts",
            "close",
            "return_1d",
            "return_5d",
            "sma_5",
            "sma_20",
            "ema_12",
            "ema_26",
            "macd",
            "rsi_14",
            "volatility_20d",
            "volume_avg_20d",
        ]
    ].dropna()

    features["ts"] = pd.to_datetime(features["ts"]).dt.tz_localize(None).astype(str)
    logger.info("Structured dataset ready with %d rows", len(features))
    return features


def store_structured_data(structured_df: pd.DataFrame, db_path: Path) -> None:
    """Persist structured dataset into SQLite."""
    if structured_df.empty:
        logging.getLogger("runner").warning("Structured dataframe empty; nothing stored.")
        return

    logger = logging.getLogger("runner")
    logger.info("Persisting structured data (%d rows) into %s", len(structured_df), db_path)
    conn = sqlite3.connect(db_path)
    ensure_tables(conn)
    cur = conn.cursor()
    payload = [
        (
            row.symbol,
            row.ts,
            float(row.close),
            float(row.return_1d),
            float(row.return_5d),
            float(row.sma_5),
            float(row.sma_20),
            float(row.ema_12),
            float(row.ema_26),
            float(row.macd),
            float(row.rsi_14),
            float(row.volatility_20d),
            float(row.volume_avg_20d),
        )
        for row in structured_df.itertuples(index=False)
    ]
    cur.executemany(
        """
        INSERT INTO structured_data (
            symbol, ts, close, return_1d, return_5d, sma_5, sma_20,
            ema_12, ema_26, macd, rsi_14, volatility_20d, volume_avg_20d
        )
        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        ON CONFLICT(symbol, ts) DO UPDATE SET
            close=excluded.close,
            return_1d=excluded.return_1d,
            return_5d=excluded.return_5d,
            sma_5=excluded.sma_5,
            sma_20=excluded.sma_20,
            ema_12=excluded.ema_12,
            ema_26=excluded.ema_26,
            macd=excluded.macd,
            rsi_14=excluded.rsi_14,
            volatility_20d=excluded.volatility_20d,
            volume_avg_20d=excluded.volume_avg_20d
    """,
        payload,
    )
    conn.commit()
    conn.close()
    logger.info("Structured data persisted successfully.")


def split_structured_data(structured_df: pd.DataFrame) -> Dict[str, Dict[str, pd.DataFrame]]:
    """Split structured data into 75/15/5 windows per symbol (remaining rows go to holdout)."""
    logger = logging.getLogger("runner")
    splits: Dict[str, Dict[str, pd.DataFrame]] = {}
    for symbol, group in structured_df.groupby("symbol"):
        group = group.sort_values("ts")
        n = len(group)
        if n < 40:
            continue
        idx_train = int(n * 0.75)
        idx_val = idx_train + int(n * 0.15)
        idx_test = idx_val + int(n * 0.05)
        splits[symbol] = {
            "train": group.iloc[:idx_train],
            "val": group.iloc[idx_train:idx_val],
            "test": group.iloc[idx_val:idx_test],
            "holdout": group.iloc[idx_test:],
        }
    logger.info("Generated splits for %d symbols", len(splits))
    return splits


def summarize_split(df: pd.DataFrame) -> Dict[str, float]:
    """Summaries that we will supply to the mock LLMs."""
    if df.empty:
        return {"rows": 0}
    mean_return = df["return_1d"].mean()
    annualized_return = float(mean_return * 252)
    volatility = float(df["volatility_20d"].mean())
    rsi_latest = float(df["rsi_14"].iloc[-1])
    momentum = float(df["macd"].iloc[-1])
    return {
        "rows": len(df),
        "annualized_return": annualized_return,
        "volatility": volatility,
        "latest_rsi": rsi_latest,
        "latest_macd": momentum,
    }


def _safe_series(series: pd.Series) -> pd.Series:
    if series is None:
        return pd.Series(dtype=float)
    out = pd.to_numeric(series, errors="coerce")
    return out


def _safe_mean(series: pd.Series) -> float:
    if series is None or series.empty:
        return 0.0
    if series.notna().sum() == 0:
        return 0.0
    return float(np.nanmean(series))


def _sigmoid_score(value: float, scale: float) -> float:
    return float(np.clip(50 + 50 * np.tanh(value / (scale + 1e-9)), 0, 100))


def aggregate_factor_scores(splits: Dict[str, pd.DataFrame]) -> Dict[str, float]:
    """Aggregate factor-style context so LLM output is richer than simple RSI/MACD."""
    train = splits.get("train", pd.DataFrame())
    val = splits.get("val", pd.DataFrame())
    test = splits.get("test", pd.DataFrame())

    returns_5d = _safe_series(train.get("return_5d"))
    momentum = _safe_mean(returns_5d.tail(120)) * (252 / 5) if not returns_5d.empty else 0.0
    volatility = _safe_mean(_safe_series(train.get("volatility_20d")))
    volume_avg = _safe_mean(_safe_series(train.get("volume_avg_20d")))
    rsi_test_series = _safe_series(test.get("rsi_14"))
    rsi_val_series = _safe_series(val.get("rsi_14"))
    rsi_test = _safe_mean(rsi_test_series.tail(5)) or 50.0
    rsi_val = _safe_mean(rsi_val_series.tail(5)) or 50.0
    returns_1d = _safe_series(train.get("return_1d"))
    breadth = float((returns_1d > 0).mean()) if not returns_1d.empty else 0.0

    factor_scores = {
        "trend_strength": _sigmoid_score(momentum / (volatility + 1e-6), scale=3.0),
        "mean_reversion": _sigmoid_score((50 - abs(rsi_test - 50)) / 25.0, scale=1.0),
        "volatility_regime": _sigmoid_score((0.2 - volatility), scale=0.2),
        "liquidity_pulse": _sigmoid_score(np.log10(volume_avg + 1.0), scale=1.5),
        "breadth_score": _sigmoid_score((breadth - 0.5) * 2.0, scale=1.0),
    }
    factor_scores["breadth_pct"] = float(max(0.0, breadth) * 100.0)
    factor_scores["rsi_alignment"] = _sigmoid_score((50 - abs(rsi_val - rsi_test)) / 25.0, scale=1.0)
    return factor_scores


def mock_llm_analysis(agent: str, symbol: str, splits: Dict[str, pd.DataFrame]) -> Dict:
    """Produce deterministic summaries to simulate GPT/Claude JSON payloads."""
    logger = logging.getLogger("runner")
    train_stats = summarize_split(splits["train"])
    val_stats = summarize_split(splits["val"])
    test_stats = summarize_split(splits["test"])
    short_term = summarize_split(splits["train"].tail(60)) if len(splits["train"]) > 60 else train_stats

    instrument_profile = {
        "train_stats": train_stats,
        "val_stats": val_stats,
        "test_stats": test_stats,
        "momentum_short": short_term,
        "vol_cluster": {
            "train_vol": train_stats.get("volatility"),
            "test_vol": test_stats.get("volatility"),
        },
        "rsi_triangle": {
            "train": train_stats.get("latest_rsi"),
            "val": val_stats.get("latest_rsi"),
            "test": test_stats.get("latest_rsi"),
        },
    }
    factor_scores = aggregate_factor_scores(splits)
    instrument_profile["factor_scores"] = factor_scores

    trend = train_stats.get("annualized_return", 0.0)
    volatility = train_stats.get("volatility", 0.0)
    rsi = test_stats.get("latest_rsi", 50.0)

    if rsi > 65 and trend > 0:
        recommendation = "trim"
    elif rsi < 35 and trend > 0:
        recommendation = "accumulate"
    elif trend < 0 and volatility > 0.25:
        recommendation = "avoid"
    else:
        recommendation = "hold"

    confidence = max(
        0.1,
        min(
            0.95,
            0.5 + 0.5 * np.tanh((trend - volatility) * 2),
        ),
    )

    insights = {
        "train_stats": train_stats,
        "val_stats": val_stats,
        "test_stats": test_stats,
        "recommendation": recommendation,
        "confidence": float(confidence),
    }
    insights["factor_drivers"] = [
        {"factor": name, "score": score}
        for name, score in sorted(factor_scores.items(), key=lambda kv: kv[1], reverse=True)
    ]
    insights["context"] = {
        "breadth_positive_pct": factor_scores.get("breadth_pct"),
        "volatility_regime_score": factor_scores.get("volatility_regime"),
        "trend_vs_vol": trend - volatility if None not in (trend, volatility) else None,
    }
    logger.debug("LLM summary created for %s via %s", symbol, agent)
    return {
        "agent": agent,
        "symbol": symbol,
        "split": {"train": train_stats["rows"], "val": val_stats["rows"], "test": test_stats["rows"]},
        "insights": insights,
        "instrument_profile": instrument_profile,
    }


def save_llm_results(summaries: List[Dict], model_results_db: Path) -> None:
    """Persist GPT/Claude summaries into model_results DB."""
    logger = logging.getLogger("runner")
    conn = sqlite3.connect(model_results_db)
    cur = conn.cursor()
    for summary in summaries:
        model_name = f"{summary['agent']}_{summary['symbol']}_analysis"
        hyperparams = json.dumps({"split": "75/15/5"})
        payload = {
            "agent": summary["agent"],
            "symbol": summary["symbol"],
            "split": summary["split"],
            "insights": summary["insights"],
            "instrument_profile": summary["instrument_profile"],
        }
        top_driver = None
        if summary["insights"].get("factor_drivers"):
            top_driver = max(summary["insights"]["factor_drivers"], key=lambda d: d.get("score", 0.0))
        description_obj = {
            "recommendation": summary["insights"].get("recommendation"),
            "confidence": summary["insights"].get("confidence"),
            "top_factor": top_driver,
        }
        description = json.dumps(description_obj)
        additional_metrics_json = json.dumps(payload)
        cur.execute(
            """
            SELECT model_id FROM models
            WHERE model_name = ? AND symbol = ? AND horizon = ?
        """,
            (model_name, summary["symbol"], 1),
        )
        existing = cur.fetchone()
        if existing:
            model_id = existing[0]
            cur.execute("DELETE FROM training_results WHERE model_id = ?", (model_id,))
            cur.execute("DELETE FROM feature_importance WHERE model_id = ?", (model_id,))
            cur.execute("DELETE FROM models WHERE model_id = ?", (model_id,))
        cur.execute(
            """
            INSERT INTO models (
                model_name, model_type, symbol, horizon, model_path,
                hyperparameters, feature_count, description
            )
            VALUES (?, ?, ?, ?, ?, ?, ?, ?)
        """,
            (
                model_name,
                summary["agent"],
                summary["symbol"],
                1,
                None,
                hyperparams,
                0,
                description,
            ),
        )
        model_id = cur.lastrowid
        cur.execute(
            """
            INSERT INTO training_results (
                model_id, train_samples, test_samples, additional_metrics
            )
            VALUES (?, ?, ?, ?)
        """,
            (
                model_id,
                summary["split"]["train"],
                summary["split"]["test"],
                additional_metrics_json,
            ),
        )
    conn.commit()
    conn.close()
    logger.info("Stored %d LLM summary models into %s", len(summaries), model_results_db)


def serialize(obj):
    """JSON serializer for numpy/pandas objects."""
    if isinstance(obj, (np.integer, int)):
        return int(obj)
    if isinstance(obj, (np.floating, float)):
        return float(obj)
    if isinstance(obj, (pd.Timestamp,)):
        return obj.isoformat()
    if isinstance(obj, Path):
        return str(obj)
    return str(obj)


def main() -> None:
    logger = logging.getLogger("runner")
    config = Config()
    db_path = Path(config.database.get("path", "data/storage/stocks.db"))
    model_results_path = Path("model_results.db")

    reset_stocks_storage(db_path)
    ingestion_summary = ingest_top_tsx_symbols(db_path)
    summary_payload = [
        {
            "symbol": r.symbol,
            "rows": r.rows_inserted,
            "start": r.start,
            "end": r.end,
        }
        for r in ingestion_summary
    ]
    print("=== Ingestion Summary ===")
    print(json.dumps(summary_payload, indent=2))
    logger.info("Ingestion summary:\n%s", json.dumps(summary_payload, indent=2, default=serialize))
    log_file = Path("logs") / "tsx_ingestion_summary.txt"
    log_file.parent.mkdir(exist_ok=True)
    log_file.write_text(json.dumps(summary_payload, indent=2))

    pipeline = DataPipeline()
    clean_df = pipeline.loadAndClean()
    pipeline.close()

    structured_df = build_structured_dataset(clean_df)
    store_structured_data(structured_df, db_path)

    multi_trainer = UnifiedMultiSymbolTrainer(db_path=str(db_path))
    agent_results = multi_trainer.train_all(
        years=5,
        forecast_horizon=1,
        save_results=True,
    )
    logger.info("agentVD training complete: %d results", len(agent_results))

    splits = split_structured_data(structured_df)
    llm_summaries: List[Dict] = []
    for symbol, symbol_splits in splits.items():
        if symbol_splits["train"].empty or symbol_splits["test"].empty:
            continue
        llm_summaries.append(mock_llm_analysis("LLM-GPT", symbol, symbol_splits))
        llm_summaries.append(mock_llm_analysis("LLM-CLAUDE", symbol, symbol_splits))
    if llm_summaries:
        save_llm_results(llm_summaries, model_results_path)
    logger.info("LLM summaries generated for %d symbols", len(llm_summaries) // 2 if llm_summaries else 0)

    final_summary = {
        "ingestion": summary_payload,
        "structured_rows": len(structured_df),
        "agentvd_models": agent_results,
        "llm_models": llm_summaries,
    }
    print("=== Final Pipeline Summary ===")
    print(json.dumps(final_summary, indent=2, default=serialize))
    logger.info("Final summary:\n%s", json.dumps(final_summary, indent=2, default=serialize))


if __name__ == "__main__":
    logger = logging.getLogger("runner")
    logger.setLevel(logging.INFO)
    formatter = logging.Formatter("%(asctime)s - %(levelname)s - %(message)s")

    # Reset handlers to avoid duplicates when rerunning.
    if logger.handlers:
        logger.handlers.clear()

    LOG_FILE.parent.mkdir(parents=True, exist_ok=True)
    file_handler = logging.FileHandler(LOG_FILE, encoding="utf-8")
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)

    stream_handler = logging.StreamHandler()
    stream_handler.setFormatter(formatter)
    logger.addHandler(stream_handler)

    main()
