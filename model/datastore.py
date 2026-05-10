"""DataStore — shared data loading layer for all essays."""

import logging

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


class DataStore:
    """
    Loads research data from AWS (primary) or SQLite (fallback).

    All DataFrames are available as attributes after initialization.

    Attributes
    ----------
    events : pd.DataFrame
        Culture war company events with tickers, dates, political leanings.
    stock_returns : pd.DataFrame
        Daily returns indexed by (DATE, TICKER).
    ff3 : pd.DataFrame
        Fama-French 3-factor model data (MKT_RF, SMB, HML, RF).
    ff5 : pd.DataFrame
        Fama-French 5-factor model data (+ RMW, CMA).
    momentum : pd.DataFrame
        Momentum factor (MOM).
    vix : pd.DataFrame
        VIX volatility index.
    inflation : pd.DataFrame
        Inflation measures (CPI, PCE, PPI YoY changes).
    rates : pd.DataFrame
        Treasury yields, policy rates, credit spreads, curve metrics.
    employment : pd.DataFrame
        Employment, unemployment, JOLTS, wages data.
    gdp : pd.DataFrame
        GDP headline, components, growth rates.
    macro : pd.DataFrame
        Additional macro (sentiment, housing, dollar index).
    backend : str
        Which database backend was used ('aws' or 'sqlite').
    """

    def __init__(self, backend=None, aws_database=None, sqlite_path=None):
        """
        Initialize the DataStore by connecting to the database.

        Parameters
        ----------
        backend : str, optional
            Force a specific backend: 'aws' or 'sqlite'.
            If None, tries AWS first, falls back to SQLite.
        aws_database : str, optional
            Override Glue database name.
        sqlite_path : str, optional
            Override SQLite database path.
        """
        self.backend = backend
        self._conn = None
        self._loader = None

        self._connect(backend, aws_database, sqlite_path)
        self._load_all()

    def _connect(self, backend, aws_database, sqlite_path):
        """Establish database connection with AWS -> SQLite fallback."""
        if backend == 'sqlite':
            self._connect_sqlite(sqlite_path)
            return

        if backend == 'aws':
            self._connect_aws(aws_database)
            return

        # Auto-detect: try AWS first, verify reads work
        try:
            self._connect_aws(aws_database)
            test_df = self._loader.read_table('CULTURE_WAR_COMPANIES', limit=1)
            if test_df.empty:
                raise RuntimeError("AWS reads return empty results")
            logger.info("AWS read verified")
            return
        except Exception as e:
            logger.warning("AWS unusable (%s). Falling back to SQLite.", e)
            if self._loader:
                try:
                    self._loader.close()
                except Exception:
                    pass

        self._connect_sqlite(sqlite_path)

    def _connect_aws(self, database=None):
        """Connect to AWS (S3 + Glue + Athena)."""
        from Database import AthenaLoader
        loader = AthenaLoader(database=database)
        loader.connect()
        self._loader = loader
        self.backend = 'aws'
        logger.info("DataStore connected to AWS")

    def _connect_sqlite(self, db_path):
        """Connect to SQLite."""
        from Database import SQLiteLoader
        loader = SQLiteLoader(db_path=db_path)
        loader.connect()
        self._loader = loader
        self._conn = loader.conn
        self.backend = 'sqlite'
        logger.info("DataStore connected to SQLite")

    def _read(self, table_name, parse_dates=None):
        """Read a table from whichever backend is connected (internal)."""
        try:
            df = self._loader.read_table(table_name)
            if parse_dates and len(df) > 0:
                for col in parse_dates:
                    if col in df.columns:
                        df[col] = pd.to_datetime(df[col], errors='coerce')
            return df
        except Exception as e:
            logger.warning("Could not load %s: %s", table_name, e)
            return pd.DataFrame()

    def read_table(self, table_name: str, parse_dates=None) -> 'pd.DataFrame':
        """
        Read a table from the database (public API).

        Parameters
        ----------
        table_name : str
            Table name to read.
        parse_dates : list of str, optional
            Columns to parse as datetime.

        Returns
        -------
        pd.DataFrame (empty if table not found).
        """
        return self._read(table_name, parse_dates=parse_dates)

    def _load_all(self):
        """Load all tables into DataStore attributes."""
        logger.info("Loading datasets from %s...", self.backend)

        # --- Core Events ---
        self.events = self._read('CULTURE_WAR_COMPANIES', parse_dates=['EVENT_DATE'])

        # --- Stock Data & Returns ---
        raw_stocks = self._read('STOCK_DATA')
        raw_stocks = self._reshape_stock_data(raw_stocks)
        self.stocks = raw_stocks
        self.stock_returns = self._compute_returns(raw_stocks)

        # --- Factor Models ---
        self.ff3 = self._read('FF3_FACTORS', parse_dates=['DATE'])
        self.ff5 = self._read('FF5_FACTORS', parse_dates=['DATE'])
        self.momentum = self._read('MOMENTUM_FACTORS', parse_dates=['DATE'])

        # Merge all factors into one frame
        self.factors = self._merge_factors()

        # --- VIX ---
        self.vix = self._read('VIX_DATA', parse_dates=['DATE'])

        # --- Macro Data ---
        self.inflation = self._read('INFLATION_COMPREHENSIVE', parse_dates=['DATE'])
        self.rates = self._read('RATES_COMPREHENSIVE', parse_dates=['DATE'])
        self.employment = self._read('EMPLOYMENT_COMPREHENSIVE', parse_dates=['DATE'])
        self.gdp = self._read('GDP_COMPREHENSIVE', parse_dates=['DATE'])
        self.macro = self._read('ADDITIONAL_MACRO', parse_dates=['DATE'])
        self.money = self._read('M2_COMPREHENSIVE', parse_dates=['DATE'])
        self.ip = self._read('IP_COMPREHENSIVE', parse_dates=['DATE'])

        # --- Political Events & Exposure (Essay 3 reframe) ---
        self.political_events = self._read('POLITICAL_EVENTS', parse_dates=['EVENT_DATE'])
        self.political_exposure = self._read('POLITICAL_EXPOSURE')

        logger.info("DataStore ready (%s backend, %d events, %d political events, %d tickers)",
                     self.backend, len(self.events), len(self.political_events),
                     self.stock_returns['TICKER'].nunique() if 'TICKER' in self.stock_returns.columns else 0)

    def _reshape_stock_data(self, df):
        """
        Reshape stock data from wide MultiIndex format to long format.

        yfinance returns MultiIndex columns like ('Open', 'NKE'). When saved
        to SQLite via Database.py, these become mangled strings like
        "('OPEN',_'NKE')". This method detects that format and reshapes into
        a clean long-format table: DATE, TICKER, OPEN, HIGH, LOW, CLOSE,
        VOLUME, ADJ_CLOSE.
        """
        if df.empty:
            return df

        cols = df.columns.tolist()

        # Already in long format (has clean TICKER and DATE columns)
        if 'TICKER' in cols and 'DATE' in cols and 'ADJ_CLOSE' in cols:
            df['DATE'] = pd.to_datetime(df['DATE'], errors='coerce')
            return df

        # Detect mangled MultiIndex columns from SQLite
        # Pattern: "('METRIC', 'TICKER')" or "('METRIC',_'TICKER')"
        import re
        ticker_col = None
        date_col = None
        metric_map = {}  # {ticker: {metric: col_name}}

        for c in cols:
            m = re.match(r"\('(\w+)',[\s_]*'([^']*)'\)", c)
            if m:
                metric, ticker = m.group(1), m.group(2)
                if ticker == '' and metric == 'TICKER':
                    ticker_col = c
                elif ticker == '' and metric == 'DATE':
                    date_col = c
                elif ticker:
                    metric_map.setdefault(ticker, {})[metric] = c

        if not ticker_col or not date_col or not metric_map:
            logger.warning("Stock data has unrecognized column format")
            return df

        # Build long-format table: for each row, extract the ticker's columns
        records = []
        metrics = ['OPEN', 'HIGH', 'LOW', 'CLOSE', 'VOLUME', 'ADJ_CLOSE']

        for ticker, col_map in metric_map.items():
            # Get rows for this ticker
            mask = df[ticker_col] == ticker
            sub = df.loc[mask, [date_col] + [col_map[m] for m in metrics if m in col_map]].copy()
            sub.columns = ['DATE'] + [m for m in metrics if m in col_map]
            sub['TICKER'] = ticker
            sub = sub.dropna(subset=['OPEN', 'CLOSE'], how='all')
            records.append(sub)

        if not records:
            return pd.DataFrame()

        result = pd.concat(records, ignore_index=True)
        result['DATE'] = pd.to_datetime(result['DATE'], errors='coerce')

        # Convert price columns to float
        for m in metrics:
            if m in result.columns:
                result[m] = pd.to_numeric(result[m], errors='coerce')

        logger.info("Reshaped stock data: %d rows, %d tickers",
                     len(result), result['TICKER'].nunique())
        return result

    def _compute_returns(self, stocks):
        """Compute daily simple and log returns from price data."""
        if stocks.empty or 'ADJ_CLOSE' not in stocks.columns:
            logger.warning("No ADJ_CLOSE column in stock data; skipping return computation")
            return stocks

        df = stocks.sort_values(['TICKER', 'DATE']).copy()
        df['RETURN'] = df.groupby('TICKER')['ADJ_CLOSE'].pct_change()
        df['LOG_RETURN'] = np.log1p(df['RETURN'])
        return df

    def _merge_factors(self):
        """Merge FF5 + Momentum into a single factor DataFrame."""
        if self.ff5.empty:
            return self.ff3.copy() if not self.ff3.empty else pd.DataFrame()

        factors = self.ff5.copy()
        if not self.momentum.empty and 'DATE' in self.momentum.columns:
            factors = factors.merge(self.momentum, on='DATE', how='left')
        return factors

    def get_event_tickers(self):
        """Get list of unique treatment tickers."""
        if self.events.empty:
            return []
        return self.events['TICKER'].unique().tolist()

    def get_control_tickers(self):
        """Get list of unique control tickers."""
        if self.events.empty:
            return []
        return self.events['CONTROL_TICKER'].dropna().unique().tolist()

    def get_ticker_returns(self, ticker):
        """Get daily returns for a single ticker, sorted by date."""
        if self.stock_returns.empty:
            return pd.DataFrame()
        mask = self.stock_returns['TICKER'] == ticker
        return self.stock_returns.loc[mask].sort_values('DATE').copy()

    def get_event_info(self, ticker):
        """Get event details for a ticker."""
        mask = self.events['TICKER'] == ticker
        return self.events.loc[mask].copy()

    # -----------------------------------------------------------------
    # Result persistence
    # -----------------------------------------------------------------
    def save_results(
        self,
        event_results: pd.DataFrame = None,
        did_results: pd.DataFrame = None,
        cross_sectional: pd.DataFrame = None,
        model_name: str = 'FF5',
    ) -> dict:
        """
        Save model results back to the database (AWS or SQLite).

        Parameters
        ----------
        event_results : pd.DataFrame, optional
            Output from event_study_all().
        did_results : pd.DataFrame, optional
            Output from diff_in_diff_all().
        cross_sectional : pd.DataFrame, optional
            Output from cross_sectional_car().
        model_name : str
            Factor model used (for metadata).

        Returns
        -------
        dict mapping table name to write result.
        """
        if self._loader is None:
            raise RuntimeError("DataStore is closed; cannot save results.")

        results = {}
        timestamp = pd.Timestamp.now().isoformat()

        if event_results is not None and not event_results.empty:
            df = event_results.copy()
            df['MODEL_NAME'] = model_name
            df['RUN_TIMESTAMP'] = timestamp
            res = self._loader.write_table(df, 'EVENT_STUDY_RESULTS', replace=True)
            results['EVENT_STUDY_RESULTS'] = res
            logger.info("Saved EVENT_STUDY_RESULTS: %d rows", res['rows'])

        if did_results is not None and not did_results.empty:
            df = did_results.copy()
            df['MODEL_NAME'] = model_name
            df['RUN_TIMESTAMP'] = timestamp
            res = self._loader.write_table(df, 'DID_RESULTS', replace=True)
            results['DID_RESULTS'] = res
            logger.info("Saved DID_RESULTS: %d rows", res['rows'])

        if cross_sectional is not None and not cross_sectional.empty:
            df = cross_sectional.copy()
            df['MODEL_NAME'] = model_name
            df['RUN_TIMESTAMP'] = timestamp
            res = self._loader.write_table(df, 'CROSS_SECTIONAL_CAR', replace=True)
            results['CROSS_SECTIONAL_CAR'] = res
            logger.info("Saved CROSS_SECTIONAL_CAR: %d rows", res['rows'])

        # Save a run summary row
        summary_row = {
            'RUN_TIMESTAMP': timestamp,
            'MODEL_NAME': model_name,
            'BACKEND': self.backend,
            'N_EVENTS': len(self.events),
            'N_TICKERS': self.stock_returns['TICKER'].nunique() if 'TICKER' in self.stock_returns.columns else 0,
        }
        if event_results is not None and not event_results.empty:
            ok = event_results[event_results['STATUS'] == 'OK']
            summary_row['N_EVENT_STUDIES'] = len(ok)
            summary_row['N_SIGNIFICANT_CAR_005'] = int((ok['CAR_P'] < 0.05).sum()) if 'CAR_P' in ok.columns else 0
            summary_row['N_SIGNIFICANT_CAR_010'] = int((ok['CAR_P'] < 0.10).sum()) if 'CAR_P' in ok.columns else 0
            summary_row['AVG_CAR'] = ok['CAR'].mean() if 'CAR' in ok.columns else None
            summary_row['MEDIAN_CAR'] = ok['CAR'].median() if 'CAR' in ok.columns else None
        if did_results is not None and not did_results.empty:
            ok = did_results[did_results['STATUS'] == 'OK']
            summary_row['N_DID'] = len(ok)
            summary_row['N_SIGNIFICANT_DID_005'] = int((ok['DID_P'] < 0.05).sum()) if 'DID_P' in ok.columns else 0
            summary_row['AVG_DID_COEF'] = ok['DID_COEF'].mean() if 'DID_COEF' in ok.columns else None

        summary_df = pd.DataFrame([summary_row])
        res = self._loader.write_table(summary_df, 'MODEL_RUN_SUMMARY', replace=False)
        results['MODEL_RUN_SUMMARY'] = res
        logger.info("Saved MODEL_RUN_SUMMARY")

        return results

    def write_table(self, df: pd.DataFrame, table_name: str, replace: bool = True) -> dict:
        """
        Write a DataFrame to the database.

        Parameters
        ----------
        df : pd.DataFrame
        table_name : str
        replace : bool
            If True, replace existing table. Otherwise append.

        Returns
        -------
        dict with write status.
        """
        if self._loader is None:
            raise RuntimeError("DataStore is closed; cannot write.")
        return self._loader.write_table(df, table_name, replace=replace)

    def close(self):
        """Close the database connection."""
        if self._loader:
            self._loader.close()
            self._loader = None
