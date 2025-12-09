import os
import sqlite3
import pandas as pd
import numpy as np

from pathlib import Path
from datetime import datetime, timedelta
from src.config.config import Config
from src.util.logging import get_logger


class DataPipeline:
    """Data Pipeline for database operations and data cleaning."""

    def __init__(self):
        """Initialize data pipeline."""
        self.log = get_logger(name="DATAPIPELINE", enable_console=False)
        self.config = Config()
        self.conn = None
        self.db_path = None
        self.tables = {}

        self.log.info("DataPipeline initialized")

    def loadDBAndCheckDB(self):
        """
        Load database and check structure.

        Returns:
            dict with database information
        """
        self.log.info("Loading database...")

        # Get database path from config
        self.db_path = self.config.database['path']

        # Check if database exists
        if not Path(self.db_path).exists():
            self.log.error(f"Database not found: {self.db_path}")
            return {
                'success': False,
                'error': 'Database file not found',
                'path': self.db_path
            }

        try:
            self.conn = sqlite3.connect(self.db_path)
            self.log.info(f"Connected to database: {self.db_path}")
            cursor = self.conn.cursor()
            cursor.execute("SELECT name FROM sqlite_master WHERE type='table';")
            table_names = [row[0] for row in cursor.fetchall()]

            self.log.info(f"Found {len(table_names)} tables: {', '.join(table_names)}")

            db_info = {
                'success': True,
                'path': self.db_path,
                'tables': {}
            }

            for table_name in table_names:
                # Get column info
                cursor.execute(f"PRAGMA table_info({table_name})")
                columns = cursor.fetchall()
                # Get row count
                cursor.execute(f"SELECT COUNT(*) FROM {table_name}")
                row_count = cursor.fetchone()[0]

                # Store table info
                table_info = {
                    'columns': [],
                    'column_types': {},
                    'row_count': row_count
                }

                for col in columns:
                    col_name = col[1]
                    col_type = col[2]
                    table_info['columns'].append(col_name)
                    table_info['column_types'][col_name] = col_type

                db_info['tables'][table_name] = table_info
                self.tables[table_name] = table_info

                self.log.info(f"Table '{table_name}': {len(table_info['columns'])} columns, {row_count:,} rows")

            return db_info

        except Exception as e:
            self.log.error(f"Error loading database: {e}")
            return {
                'success': False,
                'error': str(e),
                'path': self.db_path
            }

    def load_symbol_data(self, symbol: str, table_name: str = 'candle', isAll : bool = False) -> pd.DataFrame:
        """
        Load all data for a specific symbol.

        Args:
            symbol: Stock symbol
            table_name: Name of the table

        Returns:
            DataFrame with symbol data
        """
        if self.conn is None:
            self.log.error("Database not connected")
            return None
        try:
            if isAll == True:
                query = f"SELECT * FROM {table_name} ORDER BY symbol, ts"
                df = pd.read_sql_query(query, self.conn)
                self.log.info(f"Loaded {len(df):,} rows for all the symbols")
                return df
            query = f"SELECT * FROM {table_name} WHERE symbol='{symbol}' ORDER BY ts"
            df = pd.read_sql_query(query, self.conn)
            self.log.info(f"Loaded {len(df):,} rows for {symbol}")
            return df

        except Exception as e:
            self.log.error(f"Error loading data for {symbol}: {e}")
            return None


    # ========================================================================
    # DATA CLEANING LAYER
    # ========================================================================

    def clean_data(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Main data cleaning pipeline.

        Applies all cleaning steps in order:
        1. Handle missing values
        2. Remove duplicates
        3. Fix data types
        4. Handle outliers
        5. Validate OHLCV relationships
        6. Fill gaps in time series

        Args:
            df: DataFrame to clean

        Returns:
            Cleaned DataFrame
        """
        original_rows = len(df)

        # Make a copy to avoid modifying original
        df_clean = df.copy()

        # Apply cleaning steps
        df_clean = self._fix_data_types(df_clean)
        df_clean = self._handle_missing_values(df_clean)
        df_clean = self._remove_duplicates(df_clean)
        df_clean = self._handle_outliers(df_clean)
        df_clean = self._validate_ohlcv(df_clean)
        df_clean = self._fill_time_gaps(df_clean)

        cleaned_rows = len(df_clean)
        removed_rows = original_rows - cleaned_rows
        return df_clean

    def _fix_data_types(self, df: pd.DataFrame) -> pd.DataFrame:
        """Fix data types for all columns."""
        self.log.info("Fixing data types...")

        df = df.copy()

        # Convert timestamp to datetime
        if 'ts' in df.columns:
            df['ts'] = pd.to_datetime(df['ts'], errors='coerce')

        # Convert price columns to float
        price_cols = ['open', 'high', 'low', 'close']
        for col in price_cols:
            if col in df.columns:
                df[col] = pd.to_numeric(df[col], errors='coerce')

        # Convert volume to integer
        if 'volume' in df.columns:
            df['volume'] = pd.to_numeric(df['volume'], errors='coerce')
            df['volume'] = df['volume'].fillna(0).astype(np.int64)

        self.log.info(f"Data types fixed: {len(df)} rows")
        return df

    def _handle_missing_values(self, df: pd.DataFrame) -> pd.DataFrame:
        """Handle missing values in the data."""
        self.log.info("Handling missing values...")

        df = df.copy()

        # Count missing values
        missing_before = df.isnull().sum().sum()

        if missing_before > 0:
            self.log.warning(f"Found {missing_before} missing values")

            # Drop rows with missing timestamps
            if 'ts' in df.columns:
                before = len(df)
                df = df.dropna(subset=['ts'])
                dropped = before - len(df)
                if dropped > 0:
                    self.log.warning(f"Dropped {dropped} rows with missing timestamps")

            # Drop rows with missing OHLC values
            price_cols = ['open', 'high', 'low', 'close']
            price_cols_present = [col for col in price_cols if col in df.columns]

            if price_cols_present:
                before = len(df)
                df = df.dropna(subset=price_cols_present)
                dropped = before - len(df)
                if dropped > 0:
                    self.log.warning(f"Dropped {dropped} rows with missing prices")

            # Fill missing volume with 0
            if 'volume' in df.columns:
                df['volume'] = df['volume'].fillna(0)

        missing_after = df.isnull().sum().sum()
        self.log.info(f"Missing values: {missing_before} → {missing_after}")

        return df

    def _remove_duplicates(self, df: pd.DataFrame) -> pd.DataFrame:
        """Remove duplicate rows."""
        self.log.info("Removing duplicates...")

        before = len(df)

        # Remove duplicates based on symbol and timestamp
        if 'symbol' in df.columns and 'ts' in df.columns:
            df = df.drop_duplicates(subset=['symbol', 'ts'], keep='first')
        elif 'ts' in df.columns:
            df = df.drop_duplicates(subset=['ts'], keep='first')
        else:
            df = df.drop_duplicates()

        removed = before - len(df)

        if removed > 0:
            self.log.warning(f"Removed {removed} duplicate rows")
        else:
            self.log.info("No duplicates found")

        return df

    def _handle_outliers(self, df: pd.DataFrame) -> pd.DataFrame:
        """Handle outliers in price and volume data."""
        self.log.info("Handling outliers...")

        df = df.copy()
        outliers_removed = 0

        # Check for negative prices
        price_cols = ['open', 'high', 'low', 'close']
        for col in price_cols:
            if col in df.columns:
                negative_mask = df[col] <= 0
                negative_count = negative_mask.sum()
                if negative_count > 0:
                    self.log.warning(f"Found {negative_count} negative/zero {col} values")
                    df = df[~negative_mask]
                    outliers_removed += negative_count

        # Check for extreme price changes (>50% in one day)
        if 'close' in df.columns:
            df = df.sort_values('ts')
            price_change = df['close'].pct_change().abs()
            extreme_mask = price_change > 0.5  # 50% change
            extreme_count = extreme_mask.sum()

            if extreme_count > 0:
                self.log.warning(f"Found {extreme_count} extreme price changes (>50%)")
                # Don't remove these automatically - could be stock splits or real volatility
                # Just log them for manual review

        # Check for zero volume on trading days
        if 'volume' in df.columns:
            zero_volume = (df['volume'] == 0).sum()
            if zero_volume > 0:
                self.log.info(f"Found {zero_volume} days with zero volume (possibly holidays)")

        if outliers_removed > 0:
            self.log.info(f"Removed {outliers_removed} outlier rows")
        else:
            self.log.info("No outliers removed")

        return df

    def _validate_ohlcv(self, df: pd.DataFrame) -> pd.DataFrame:
        """Validate OHLCV relationships (High >= Low, etc.)."""
        self.log.info("Validating OHLCV relationships...")

        df = df.copy()
        invalid_count = 0

        # Check if all OHLC columns exist
        required_cols = ['open', 'high', 'low', 'close']
        if not all(col in df.columns for col in required_cols):
            self.log.warning("Not all OHLC columns present, skipping validation")
            return df

        # Rule 1: High should be >= Low
        invalid_high_low = df['high'] < df['low']
        if invalid_high_low.any():
            count = invalid_high_low.sum()
            self.log.warning(f"Found {count} rows where high < low")
            df = df[~invalid_high_low]
            invalid_count += count

        # Rule 2: High should be >= Open and Close
        invalid_high_open = df['high'] < df['open']
        if invalid_high_open.any():
            count = invalid_high_open.sum()
            self.log.warning(f"Found {count} rows where high < open")
            df = df[~invalid_high_open]
            invalid_count += count

        invalid_high_close = df['high'] < df['close']
        if invalid_high_close.any():
            count = invalid_high_close.sum()
            self.log.warning(f"Found {count} rows where high < close")
            df = df[~invalid_high_close]
            invalid_count += count

        # Rule 3: Low should be <= Open and Close
        invalid_low_open = df['low'] > df['open']
        if invalid_low_open.any():
            count = invalid_low_open.sum()
            self.log.warning(f"Found {count} rows where low > open")
            df = df[~invalid_low_open]
            invalid_count += count

        invalid_low_close = df['low'] > df['close']
        if invalid_low_close.any():
            count = invalid_low_close.sum()
            self.log.warning(f"Found {count} rows where low > close")
            df = df[~invalid_low_close]
            invalid_count += count

        if invalid_count > 0:
            self.log.info(f"Removed {invalid_count} rows with invalid OHLCV relationships")
        else:
            self.log.info("All OHLCV relationships valid")

        return df

    def _fill_time_gaps(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Identify and optionally fill gaps in time series.
        Note: Does not fill by default, just identifies gaps.
        """
        self.log.info("Checking for time gaps...")

        if 'ts' not in df.columns:
            self.log.warning("No timestamp column found")
            return df

        df = df.copy()
        df = df.sort_values('ts')

        # Calculate time differences
        time_diff = df['ts'].diff()

        # For daily data, gaps > 4 days are suspicious (accounting for weekends)
        large_gaps = time_diff > timedelta(days=4)
        gap_count = large_gaps.sum()

        if gap_count > 0:
            self.log.warning(f"Found {gap_count} time gaps > 4 days")
            # Log the gaps
            gap_indices = df[large_gaps].index
            for idx in gap_indices[:5]:  # Show first 5 gaps
                if idx > 0:
                    prev_date = df.loc[idx-1, 'ts']
                    curr_date = df.loc[idx, 'ts']
                    gap_days = (curr_date - prev_date).days
                    self.log.info(f"Gap: {prev_date} → {curr_date} ({gap_days} days)")
        else:
            self.log.info("No significant time gaps found")

        return df

    def get_data_quality_report(self, df: pd.DataFrame, symbol: str = None) -> dict:
        """
        Generate a data quality report.

        Args:
            df: DataFrame to analyze
            symbol: Optional symbol name

        Returns:
            dict with quality metrics
        """
        symbol_str = f" for {symbol}" if symbol else ""
        self.log.info(f"Generating data quality report{symbol_str}")

        report = {
            'symbol': symbol,
            'total_rows': len(df),
            'date_range': None,
            'missing_values': {},
            'duplicates': 0,
            'outliers': {},
            'data_quality_score': 0
        }

        # Date range
        if 'ts' in df.columns:
            df['ts'] = pd.to_datetime(df['ts'])
            report['date_range'] = {
                'min': str(df['ts'].min()),
                'max': str(df['ts'].max()),
                'days': (df['ts'].max() - df['ts'].min()).days
            }

        # Missing values
        for col in df.columns:
            missing = df[col].isnull().sum()
            if missing > 0:
                report['missing_values'][col] = {
                    'count': int(missing),
                    'percentage': round(missing / len(df) * 100, 2)
                }

        # Duplicates
        if 'ts' in df.columns:
            report['duplicates'] = df.duplicated(subset=['ts']).sum()

        # Outliers
        if 'close' in df.columns:
            # Negative prices
            report['outliers']['negative_prices'] = (df['close'] <= 0).sum()

            # Extreme changes
            price_change = df['close'].pct_change().abs()
            report['outliers']['extreme_changes'] = (price_change > 0.5).sum()

        # Calculate data quality score (0-100)
        score = 100

        # Deduct for missing values
        total_missing = sum(v['count'] for v in report['missing_values'].values())
        missing_penalty = min(30, total_missing / len(df) * 100)
        score -= missing_penalty

        # Deduct for duplicates
        dup_penalty = min(20, report['duplicates'] / len(df) * 100)
        score -= dup_penalty

        # Deduct for outliers
        total_outliers = sum(report['outliers'].values())
        outlier_penalty = min(20, total_outliers / len(df) * 100)
        score -= outlier_penalty

        report['data_quality_score'] = max(0, round(score, 2))

        return report
    def close(self):
        """Close database connection."""
        if self.conn:
            self.conn.close()
            self.log.info("Database connection closed")
            self.conn = None

    def loadAndClean(self):
        data_db = self.loadDBAndCheckDB()
        if data_db['success']:
            df = self.load_symbol_data(symbol= None, table_name='candle', isAll= True)
            clean_df = self.clean_data(df)
            self.close()
            self.log.info("DB CLEANING COMPLETE")
            return clean_df
        else:
            self.log.error("CHECK THE LOAD DB FUNC UNC!!!")
