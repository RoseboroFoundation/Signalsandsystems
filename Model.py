"""
Modeling module for Signals & Systems dissertation research.

Provides a DataStore that loads from Snowflake (with SQLite fallback) and
implements the core models for three interconnected essays:

  Essay 1 - Event Study: Abnormal returns around culture war events
  Essay 2 - Behavioral Factors: News sentiment and market reactions
  Essay 3 - Systematic Risk: Macro regime effects on event impacts

Usage:
    from Model import DataStore, event_study, factor_model

    # Load all data
    store = DataStore()

    # Run event study for a single company
    result = event_study(store, ticker='NKE')

    # Run event study for all companies
    results = event_study_all(store)

    # Factor model estimation
    betas = factor_model(store, ticker='NKE', model='FF5')

    # Difference-in-differences
    did = diff_in_diff(store)
"""

import logging
import warnings
from dataclasses import dataclass, field
from typing import Optional

import numpy as np
import pandas as pd
import statsmodels.api as sm
from scipy import stats

logger = logging.getLogger(__name__)


# =============================================================================
# DATA STORE
# =============================================================================
class DataStore:
    """
    Loads research data from Snowflake (primary) or SQLite (fallback).

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
        Which database backend was used ('snowflake' or 'sqlite').
    """

    def __init__(self, backend=None, snowflake_schema=None, sqlite_path=None):
        """
        Initialize the DataStore by connecting to the database.

        Parameters
        ----------
        backend : str, optional
            Force a specific backend: 'snowflake' or 'sqlite'.
            If None, tries Snowflake first, falls back to SQLite.
        snowflake_schema : str, optional
            Override Snowflake schema (default: PUBLIC).
        sqlite_path : str, optional
            Override SQLite database path.
        """
        self.backend = backend
        self._conn = None
        self._loader = None

        self._connect(backend, snowflake_schema, sqlite_path)
        self._load_all()

    def _connect(self, backend, snowflake_schema, sqlite_path):
        """Establish database connection with fallback logic."""
        if backend == 'sqlite':
            self._connect_sqlite(sqlite_path)
            return

        if backend == 'snowflake':
            self._connect_snowflake(snowflake_schema)
            return

        # Auto-detect: try Snowflake first, verify reads work
        try:
            self._connect_snowflake(snowflake_schema)
            # Verify reads actually work (account may connect but be suspended)
            test_df = self._loader.read_table('CULTURE_WAR_COMPANIES')
            if test_df.empty:
                raise RuntimeError("Snowflake reads return empty results")
            logger.info("Snowflake read verified (%d rows from CULTURE_WAR_COMPANIES)", len(test_df))
            return
        except Exception as e:
            logger.warning("Snowflake unusable (%s). Falling back to SQLite.", e)
            if self._loader:
                try:
                    self._loader.close()
                except Exception:
                    pass

        self._connect_sqlite(sqlite_path)

    def _connect_snowflake(self, schema):
        """Connect to Snowflake."""
        from Database import SnowflakeLoader
        loader = SnowflakeLoader(schema=schema)
        loader.connect()
        if not loader._writable:
            logger.warning("Snowflake account suspended. Checking if reads work...")
        self._loader = loader
        self._conn = loader.conn
        self.backend = 'snowflake'
        logger.info("DataStore connected to Snowflake")

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
        """Read a table from whichever backend is connected."""
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

        logger.info("DataStore ready (%s backend, %d events, %d tickers)",
                     self.backend, len(self.events),
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
        Save model results back to the database (Snowflake or SQLite).

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

    def close(self):
        """Close the database connection."""
        if self._loader:
            self._loader.close()
            self._loader = None


# =============================================================================
# FACTOR MODEL ESTIMATION
# =============================================================================
@dataclass
class FactorModelResult:
    """Result from a factor model regression."""
    ticker: str
    model_name: str
    alpha: float
    alpha_t: float
    alpha_p: float
    betas: dict
    r_squared: float
    adj_r_squared: float
    n_obs: int
    residuals: pd.Series = field(repr=False)
    summary: object = field(repr=False, default=None)


def factor_model(
    store: DataStore,
    ticker: str,
    model: str = 'FF5',
    start_date: str = None,
    end_date: str = None,
) -> Optional[FactorModelResult]:
    """
    Estimate a Fama-French factor model for a given ticker.

    Parameters
    ----------
    store : DataStore
        Initialized DataStore.
    ticker : str
        Stock ticker symbol.
    model : str
        'FF3' (Market, Size, Value), 'FF5' (+ Profitability, Investment),
        or 'FF5+MOM' (+ Momentum).
    start_date, end_date : str, optional
        Date range for estimation window (YYYY-MM-DD).

    Returns
    -------
    FactorModelResult or None
    """
    returns = store.get_ticker_returns(ticker)
    if returns.empty or 'RETURN' not in returns.columns:
        logger.warning("No return data for %s", ticker)
        return None

    # Select factor columns
    if model == 'FF3':
        factor_cols = ['MKT_RF', 'SMB', 'HML']
        factors = store.ff3.copy()
    elif model == 'FF5':
        factor_cols = ['MKT_RF', 'SMB', 'HML', 'RMW', 'CMA']
        factors = store.ff5.copy()
    elif model == 'FF5+MOM':
        factor_cols = ['MKT_RF', 'SMB', 'HML', 'RMW', 'CMA', 'MOM']
        factors = store.factors.copy()
    else:
        raise ValueError(f"Unknown model: {model}. Use 'FF3', 'FF5', or 'FF5+MOM'.")

    if factors.empty:
        logger.warning("No factor data available for model %s", model)
        return None

    # Fama-French returns are in percent; convert to decimal
    for col in factor_cols + ['RF']:
        if col in factors.columns:
            if factors[col].abs().max() > 1:
                factors[col] = factors[col] / 100

    # Merge returns with factors
    merged = returns[['DATE', 'RETURN']].merge(factors, on='DATE', how='inner')

    # Apply date filter
    if start_date:
        merged = merged[merged['DATE'] >= pd.Timestamp(start_date)]
    if end_date:
        merged = merged[merged['DATE'] <= pd.Timestamp(end_date)]

    # Drop missing
    merged = merged.dropna(subset=['RETURN'] + factor_cols + ['RF'])
    if len(merged) < 30:
        logger.warning("Insufficient observations for %s (%d)", ticker, len(merged))
        return None

    # Excess return = return - risk-free rate
    merged['EXCESS_RETURN'] = merged['RETURN'] - merged['RF']

    # OLS regression: R_excess = alpha + beta * factors + epsilon
    y = merged['EXCESS_RETURN']
    X = sm.add_constant(merged[factor_cols])

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        result = sm.OLS(y, X).fit(cov_type='HC1')

    betas = {col: result.params[col] for col in factor_cols}

    return FactorModelResult(
        ticker=ticker,
        model_name=model,
        alpha=result.params['const'],
        alpha_t=result.tvalues['const'],
        alpha_p=result.pvalues['const'],
        betas=betas,
        r_squared=result.rsquared,
        adj_r_squared=result.rsquared_adj,
        n_obs=result.nobs,
        residuals=pd.Series(result.resid.values, index=merged['DATE'].values),
        summary=result,
    )


# =============================================================================
# EVENT STUDY
# =============================================================================
@dataclass
class EventStudyResult:
    """Result from a single-event event study."""
    ticker: str
    event_date: pd.Timestamp
    event_description: str
    political_leaning: str
    estimation_window: tuple
    event_window: tuple
    ar: pd.DataFrame          # Daily abnormal returns
    car: float                # Cumulative abnormal return over event window
    car_t: float              # t-statistic for CAR
    car_p: float              # p-value for CAR
    bhar: float               # Buy-and-hold abnormal return
    factor_model: FactorModelResult = field(repr=False, default=None)


def event_study(
    store: DataStore,
    ticker: str = None,
    event_date: str = None,
    estimation_days: int = 252,
    pre_event_days: int = 30,
    post_event_days: int = 30,
    gap_days: int = 10,
    model: str = 'FF5',
) -> Optional[EventStudyResult]:
    """
    Run a standard event study for a culture war event.

    Methodology:
      1. Estimate factor model over the estimation window
         (ending `gap_days` before the event).
      2. Predict expected returns over the event window.
      3. Abnormal return = actual - expected.
      4. Cumulative abnormal return (CAR) and significance test.

    Parameters
    ----------
    store : DataStore
        Initialized DataStore.
    ticker : str
        Stock ticker. If None, uses first event.
    event_date : str, optional
        Override event date (YYYY-MM-DD). If None, uses date from events table.
    estimation_days : int
        Trading days in the estimation window (default 252 = 1 year).
    pre_event_days : int
        Trading days before the event in the event window.
    post_event_days : int
        Trading days after the event in the event window.
    gap_days : int
        Gap between estimation window end and event window start.
    model : str
        Factor model to use ('FF3', 'FF5', 'FF5+MOM').

    Returns
    -------
    EventStudyResult or None
    """
    # Get event info
    if ticker is None:
        if store.events.empty:
            logger.error("No events loaded")
            return None
        row = store.events.iloc[0]
        ticker = row['TICKER']
    else:
        event_info = store.get_event_info(ticker)
        if event_info.empty:
            logger.warning("No event found for ticker %s", ticker)
            return None
        row = event_info.iloc[0]

    evt_date = pd.Timestamp(event_date) if event_date else pd.Timestamp(row['EVENT_DATE'])
    evt_desc = row.get('CULTURE_WAR_EVENT', '')
    evt_leaning = row.get('ESTIMATED_POLITICAL_LEANING', '')

    # Get returns
    returns = store.get_ticker_returns(ticker)
    if returns.empty or 'RETURN' not in returns.columns:
        logger.warning("No return data for %s", ticker)
        return None

    returns = returns.set_index('DATE').sort_index()

    # Find the event date in the trading calendar (or nearest)
    trading_dates = returns.index
    if evt_date not in trading_dates:
        mask = trading_dates >= evt_date
        if mask.any():
            evt_date = trading_dates[mask][0]
        else:
            logger.warning("Event date %s is after all available data for %s", evt_date, ticker)
            return None

    evt_idx = trading_dates.get_loc(evt_date)

    # Define windows
    est_end_idx = max(0, evt_idx - gap_days)
    est_start_idx = max(0, est_end_idx - estimation_days)
    evt_start_idx = max(0, evt_idx - pre_event_days)
    evt_end_idx = min(len(trading_dates) - 1, evt_idx + post_event_days)

    est_start = trading_dates[est_start_idx]
    est_end = trading_dates[est_end_idx]
    evt_start = trading_dates[evt_start_idx]
    evt_end = trading_dates[evt_end_idx]

    # Step 1: Estimate factor model on estimation window
    fm = factor_model(
        store, ticker, model=model,
        start_date=str(est_start.date()),
        end_date=str(est_end.date()),
    )
    if fm is None:
        logger.warning("Factor model estimation failed for %s", ticker)
        return None

    # Step 2: Predict expected returns over the event window
    evt_returns = returns.loc[evt_start:evt_end].copy()

    # Get factors for event window
    if model == 'FF3':
        factor_cols = ['MKT_RF', 'SMB', 'HML']
        factors = store.ff3.copy()
    elif model == 'FF5':
        factor_cols = ['MKT_RF', 'SMB', 'HML', 'RMW', 'CMA']
        factors = store.ff5.copy()
    else:
        factor_cols = ['MKT_RF', 'SMB', 'HML', 'RMW', 'CMA', 'MOM']
        factors = store.factors.copy()

    # Convert factor returns to decimal if needed
    for col in factor_cols + ['RF']:
        if col in factors.columns and factors[col].abs().max() > 1:
            factors[col] = factors[col] / 100

    factors = factors.set_index('DATE')
    evt_factors = factors.loc[evt_start:evt_end].reindex(evt_returns.index)

    # Drop days where factors are missing
    valid = evt_factors[factor_cols + ['RF']].dropna().index
    evt_returns = evt_returns.loc[valid]
    evt_factors = evt_factors.loc[valid]

    if len(evt_returns) == 0:
        logger.warning("No overlapping factor data for event window of %s", ticker)
        return None

    # Expected return = alpha + sum(beta_i * factor_i) + RF
    expected = (
        fm.alpha
        + sum(fm.betas[col] * evt_factors[col] for col in factor_cols)
        + evt_factors['RF']
    )

    # Step 3: Abnormal returns
    actual = evt_returns['RETURN']
    ar = actual - expected
    ar.name = 'AR'

    car_value = ar.sum()

    # Step 4: Significance test (Patell test)
    sigma = fm.residuals.std()
    n_event = len(ar)
    if sigma > 0 and n_event > 0:
        car_t = car_value / (sigma * np.sqrt(n_event))
        car_p = 2 * (1 - stats.t.cdf(abs(car_t), df=fm.n_obs - len(factor_cols) - 1))
    else:
        car_t = 0.0
        car_p = 1.0

    # Buy-and-hold abnormal return
    bhar = (1 + actual).prod() - (1 + expected).prod()

    # Build result DataFrame
    ar_df = pd.DataFrame({
        'DATE': ar.index,
        'ACTUAL': actual.values,
        'EXPECTED': expected.values,
        'AR': ar.values,
        'CAR': ar.cumsum().values,
    })
    ar_df['DAY'] = range(-len(ar_df[ar_df['DATE'] <= evt_date]) + 1,
                          len(ar_df) - len(ar_df[ar_df['DATE'] <= evt_date]) + 1)

    return EventStudyResult(
        ticker=ticker,
        event_date=evt_date,
        event_description=evt_desc,
        political_leaning=evt_leaning,
        estimation_window=(est_start, est_end),
        event_window=(evt_start, evt_end),
        ar=ar_df,
        car=car_value,
        car_t=car_t,
        car_p=car_p,
        bhar=bhar,
        factor_model=fm,
    )


def event_study_all(
    store: DataStore,
    model: str = 'FF5',
    estimation_days: int = 252,
    pre_event_days: int = 30,
    post_event_days: int = 30,
) -> pd.DataFrame:
    """
    Run event studies for all culture war events.

    Returns a summary DataFrame with one row per event.
    """
    rows = []
    for _, evt in store.events.iterrows():
        ticker = evt['TICKER']
        result = event_study(
            store, ticker=ticker,
            model=model,
            estimation_days=estimation_days,
            pre_event_days=pre_event_days,
            post_event_days=post_event_days,
        )
        if result is None:
            rows.append({
                'TICKER': ticker,
                'COMPANY': evt.get('COMPANY', ''),
                'EVENT_DATE': evt.get('EVENT_DATE'),
                'POLITICAL_LEANING': evt.get('ESTIMATED_POLITICAL_LEANING', ''),
                'CONTROL_TICKER': evt.get('CONTROL_TICKER', ''),
                'CAR': np.nan,
                'CAR_T': np.nan,
                'CAR_P': np.nan,
                'BHAR': np.nan,
                'ALPHA': np.nan,
                'R_SQUARED': np.nan,
                'N_OBS': 0,
                'STATUS': 'FAILED',
            })
            continue

        rows.append({
            'TICKER': ticker,
            'COMPANY': evt.get('COMPANY', ''),
            'EVENT_DATE': result.event_date,
            'EVENT_DESCRIPTION': result.event_description,
            'POLITICAL_LEANING': result.political_leaning,
            'CONTROL_TICKER': evt.get('CONTROL_TICKER', ''),
            'CAR': result.car,
            'CAR_T': result.car_t,
            'CAR_P': result.car_p,
            'BHAR': result.bhar,
            'ALPHA': result.factor_model.alpha,
            'MKT_BETA': result.factor_model.betas.get('MKT_RF', np.nan),
            'R_SQUARED': result.factor_model.r_squared,
            'N_OBS': result.factor_model.n_obs,
            'STATUS': 'OK',
        })

    return pd.DataFrame(rows)


# =============================================================================
# DIFFERENCE-IN-DIFFERENCES
# =============================================================================
@dataclass
class DiffInDiffResult:
    """Result from a difference-in-differences regression."""
    coefficient: float      # Treatment × Post interaction
    t_stat: float
    p_value: float
    ci_lower: float
    ci_upper: float
    n_obs: int
    r_squared: float
    summary: object = field(repr=False, default=None)
    data: pd.DataFrame = field(repr=False, default=None)


def diff_in_diff(
    store: DataStore,
    ticker: str = None,
    pre_days: int = 30,
    post_days: int = 30,
    model: str = 'FF5',
) -> Optional[DiffInDiffResult]:
    """
    Difference-in-differences: treatment firm vs. matched control.

    Regression:
        ExcessReturn = β₀ + β₁·Treated + β₂·Post + β₃·Treated×Post
                       + γ·Factors + ε

    β₃ is the DiD estimator (causal effect of the culture war event).

    Parameters
    ----------
    store : DataStore
    ticker : str, optional
        Treatment ticker. If None, uses first event.
    pre_days, post_days : int
        Days before/after event for the window.
    model : str
        Factor model for controls ('FF3', 'FF5').

    Returns
    -------
    DiffInDiffResult or None
    """
    if ticker is None:
        if store.events.empty:
            return None
        row = store.events.iloc[0]
        ticker = row['TICKER']
    else:
        info = store.get_event_info(ticker)
        if info.empty:
            return None
        row = info.iloc[0]

    control_ticker = row.get('CONTROL_TICKER')
    if pd.isna(control_ticker) or not control_ticker:
        logger.warning("No control ticker for %s", ticker)
        return None

    evt_date = pd.Timestamp(row['EVENT_DATE'])

    # Factor columns
    if model == 'FF3':
        factor_cols = ['MKT_RF', 'SMB', 'HML']
        factors = store.ff3.copy()
    else:
        factor_cols = ['MKT_RF', 'SMB', 'HML', 'RMW', 'CMA']
        factors = store.ff5.copy()

    for col in factor_cols + ['RF']:
        if col in factors.columns and factors[col].abs().max() > 1:
            factors[col] = factors[col] / 100

    # Build panel for treatment and control
    panels = []
    for tkr, treated_flag in [(ticker, 1), (control_ticker, 0)]:
        ret = store.get_ticker_returns(tkr)
        if ret.empty or 'RETURN' not in ret.columns:
            continue

        ret = ret[['DATE', 'RETURN']].copy()
        ret['TICKER'] = tkr
        ret['TREATED'] = treated_flag
        panels.append(ret)

    if len(panels) < 2:
        logger.warning("Missing data for treatment or control (%s / %s)", ticker, control_ticker)
        return None

    panel = pd.concat(panels, ignore_index=True)
    panel = panel.merge(factors, on='DATE', how='inner')
    panel = panel.dropna(subset=['RETURN'] + factor_cols + ['RF'])

    # Filter to event window
    panel = panel[
        (panel['DATE'] >= evt_date - pd.Timedelta(days=pre_days * 2)) &
        (panel['DATE'] <= evt_date + pd.Timedelta(days=post_days * 2))
    ].copy()

    # Use trading-day proximity
    trading_dates = sorted(panel['DATE'].unique())
    if evt_date not in trading_dates:
        future = [d for d in trading_dates if d >= evt_date]
        if future:
            evt_date = future[0]

    panel['POST'] = (panel['DATE'] >= evt_date).astype(int)
    panel['TREATED_POST'] = panel['TREATED'] * panel['POST']
    panel['EXCESS_RETURN'] = panel['RETURN'] - panel['RF']

    if len(panel) < 20:
        logger.warning("Insufficient DiD observations for %s (%d)", ticker, len(panel))
        return None

    # DiD regression
    regressors = ['TREATED', 'POST', 'TREATED_POST'] + factor_cols
    y = panel['EXCESS_RETURN']
    X = sm.add_constant(panel[regressors])

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        result = sm.OLS(y, X).fit(cov_type='HC1')

    coef = result.params['TREATED_POST']
    ci = result.conf_int().loc['TREATED_POST']

    return DiffInDiffResult(
        coefficient=coef,
        t_stat=result.tvalues['TREATED_POST'],
        p_value=result.pvalues['TREATED_POST'],
        ci_lower=ci[0],
        ci_upper=ci[1],
        n_obs=int(result.nobs),
        r_squared=result.rsquared,
        summary=result,
        data=panel,
    )


def diff_in_diff_all(store: DataStore, model: str = 'FF5') -> pd.DataFrame:
    """Run DiD for all events with matched controls. Returns summary DataFrame."""
    rows = []
    for _, evt in store.events.iterrows():
        ticker = evt['TICKER']
        control = evt.get('CONTROL_TICKER', '')
        if pd.isna(control) or not control:
            continue

        result = diff_in_diff(store, ticker=ticker, model=model)
        if result is None:
            rows.append({
                'TICKER': ticker,
                'CONTROL_TICKER': control,
                'COMPANY': evt.get('COMPANY', ''),
                'POLITICAL_LEANING': evt.get('ESTIMATED_POLITICAL_LEANING', ''),
                'DID_COEF': np.nan,
                'DID_T': np.nan,
                'DID_P': np.nan,
                'STATUS': 'FAILED',
            })
            continue

        rows.append({
            'TICKER': ticker,
            'CONTROL_TICKER': control,
            'COMPANY': evt.get('COMPANY', ''),
            'POLITICAL_LEANING': evt.get('ESTIMATED_POLITICAL_LEANING', ''),
            'DID_COEF': result.coefficient,
            'DID_T': result.t_stat,
            'DID_P': result.p_value,
            'CI_LOWER': result.ci_lower,
            'CI_UPPER': result.ci_upper,
            'R_SQUARED': result.r_squared,
            'N_OBS': result.n_obs,
            'STATUS': 'OK',
        })

    return pd.DataFrame(rows)


# =============================================================================
# CROSS-SECTIONAL ANALYSIS
# =============================================================================
def cross_sectional_car(
    event_results: pd.DataFrame,
    group_by: str = 'POLITICAL_LEANING',
) -> pd.DataFrame:
    """
    Analyze CARs across groups (e.g., political leaning).

    Parameters
    ----------
    event_results : pd.DataFrame
        Output from event_study_all().
    group_by : str
        Column to group by.

    Returns
    -------
    pd.DataFrame with group means, t-tests, and counts.
    """
    valid = event_results.dropna(subset=['CAR']).copy()
    if valid.empty or group_by not in valid.columns:
        return pd.DataFrame()

    groups = valid.groupby(group_by)['CAR']
    summary = groups.agg(['mean', 'std', 'count']).reset_index()
    summary.columns = [group_by, 'MEAN_CAR', 'STD_CAR', 'N']

    # t-test: H0: mean CAR = 0
    for idx, row in summary.iterrows():
        grp = valid[valid[group_by] == row[group_by]]['CAR']
        if len(grp) > 1:
            t_stat, p_val = stats.ttest_1samp(grp, 0)
            summary.loc[idx, 'T_STAT'] = t_stat
            summary.loc[idx, 'P_VALUE'] = p_val

    return summary


# =============================================================================
# REGIME ANALYSIS
# =============================================================================
def classify_vix_regime(store: DataStore, low=15, high=25):
    """
    Classify market dates into VIX regimes.

    Returns a DataFrame with DATE and VIX_REGIME columns.
    """
    if store.vix.empty:
        return pd.DataFrame()

    vix = store.vix[['DATE', 'VIX']].copy()
    vix['VIX_REGIME'] = pd.cut(
        vix['VIX'],
        bins=[-np.inf, low, high, np.inf],
        labels=['Low Volatility', 'Normal', 'High Volatility'],
    )
    return vix


def classify_inflation_regime(store: DataStore, low=2.0, high=4.0):
    """
    Classify months into inflation regimes using Core PCE YoY.

    Returns a DataFrame with DATE and INFLATION_REGIME columns.
    """
    if store.inflation.empty:
        return pd.DataFrame()

    col = None
    for candidate in ['CORE_PCE_YOY', 'CORE_CPI_YOY', 'CPI_YOY']:
        if candidate in store.inflation.columns:
            col = candidate
            break

    if col is None:
        logger.warning("No inflation YoY column found")
        return pd.DataFrame()

    inf = store.inflation[['DATE', col]].dropna().copy()
    inf['INFLATION_REGIME'] = pd.cut(
        inf[col],
        bins=[-np.inf, low, high, np.inf],
        labels=['Low', 'Moderate', 'High'],
    )
    return inf[['DATE', 'INFLATION_REGIME']]


def event_study_by_regime(
    store: DataStore,
    regime_df: pd.DataFrame,
    regime_col: str,
    model: str = 'FF5',
) -> pd.DataFrame:
    """
    Run event studies and group results by macro regime at the event date.

    Parameters
    ----------
    store : DataStore
    regime_df : pd.DataFrame
        Must have DATE and the regime column.
    regime_col : str
        Column name in regime_df that contains the regime classification.
    model : str
        Factor model.

    Returns
    -------
    pd.DataFrame
        Event study results augmented with regime at event date.
    """
    results = event_study_all(store, model=model)

    # Merge regime at event date
    if regime_df.empty or regime_col not in regime_df.columns:
        results[regime_col] = np.nan
        return results

    regime_df = regime_df.sort_values('DATE')
    results['EVENT_DATE'] = pd.to_datetime(results['EVENT_DATE'])

    # For each event, find the nearest regime date
    regimes = []
    for _, row in results.iterrows():
        if pd.isna(row['EVENT_DATE']):
            regimes.append(np.nan)
            continue
        mask = regime_df['DATE'] <= row['EVENT_DATE']
        if mask.any():
            regimes.append(regime_df.loc[mask.values, regime_col].iloc[-1])
        else:
            regimes.append(np.nan)

    results[regime_col] = regimes
    return results


# =============================================================================
# SUMMARY & REPORTING
# =============================================================================
def summary_statistics(store: DataStore) -> dict:
    """
    Generate summary statistics for the loaded data.

    Returns a dict of DataFrames with descriptive stats.
    """
    stats_dict = {}

    # Event summary
    if not store.events.empty:
        evt = store.events.copy()
        stats_dict['events'] = pd.DataFrame({
            'Total Events': [len(evt)],
            'Unique Tickers': [evt['TICKER'].nunique()],
            'Liberal': [(evt['ESTIMATED_POLITICAL_LEANING'] == 'Liberal').sum()],
            'Conservative': [(evt['ESTIMATED_POLITICAL_LEANING'] == 'Conservative').sum()],
            'Mixed': [(evt['ESTIMATED_POLITICAL_LEANING'] == 'Mixed').sum()],
            'Date Range': [f"{evt['EVENT_DATE'].min()} to {evt['EVENT_DATE'].max()}"],
            'Industries': [evt['INDUSTRY'].nunique()],
        })

    # Stock returns summary
    if not store.stock_returns.empty and 'RETURN' in store.stock_returns.columns:
        rets = store.stock_returns['RETURN'].dropna()
        stats_dict['returns'] = pd.DataFrame({
            'Obs': [len(rets)],
            'Mean Daily': [f"{rets.mean():.4f}"],
            'Std Daily': [f"{rets.std():.4f}"],
            'Min': [f"{rets.min():.4f}"],
            'Max': [f"{rets.max():.4f}"],
            'Tickers': [store.stock_returns['TICKER'].nunique()],
        })

    # Factor summary
    if not store.ff5.empty:
        factor_cols = ['MKT_RF', 'SMB', 'HML', 'RMW', 'CMA']
        available = [c for c in factor_cols if c in store.ff5.columns]
        stats_dict['factors'] = store.ff5[available].describe().round(4)

    return stats_dict


# =============================================================================
# CONVENIENCE: RUN ALL & SAVE
# =============================================================================
def run_and_save(backend=None, model='FF5', save=True):
    """
    Run all models (event study, DiD, cross-sectional) and save to the database.

    Parameters
    ----------
    backend : str, optional
        Force 'snowflake' or 'sqlite'. Default: auto-detect.
    model : str
        Factor model ('FF3', 'FF5', 'FF5+MOM').
    save : bool
        If True, write results back to the database.

    Returns
    -------
    dict with keys: store, event_results, did_results, cross_sectional, save_results.
    """
    store = DataStore(backend=backend)

    print("=" * 60)
    print(f"  Running all models ({model}) on {store.backend} backend")
    print(f"  {len(store.events)} events, {store.stock_returns['TICKER'].nunique() if 'TICKER' in store.stock_returns.columns else 0} tickers")
    print("=" * 60)

    # Event studies for all events
    print("\nRunning event studies for all events...")
    event_results = event_study_all(store, model=model)
    ok_events = event_results[event_results['STATUS'] == 'OK']
    print(f"  Completed: {len(ok_events)}/{len(event_results)} events")
    if not ok_events.empty:
        sig_005 = (ok_events['CAR_P'] < 0.05).sum()
        sig_010 = (ok_events['CAR_P'] < 0.10).sum()
        print(f"  Significant CAR (p<0.05): {sig_005}")
        print(f"  Significant CAR (p<0.10): {sig_010}")
        print(f"  Mean CAR: {ok_events['CAR'].mean():.4f}")
        print(f"  Median CAR: {ok_events['CAR'].median():.4f}")

    # Diff-in-diff for all events with matched controls
    print("\nRunning difference-in-differences for all events...")
    did_results = diff_in_diff_all(store, model=model)
    ok_did = did_results[did_results['STATUS'] == 'OK']
    print(f"  Completed: {len(ok_did)}/{len(did_results)} pairs")
    if not ok_did.empty:
        sig_005 = (ok_did['DID_P'] < 0.05).sum()
        print(f"  Significant DiD (p<0.05): {sig_005}")
        print(f"  Mean DiD coef: {ok_did['DID_COEF'].mean():.4f}")

    # Cross-sectional analysis by political leaning
    print("\nRunning cross-sectional CAR analysis...")
    cross_sectional = cross_sectional_car(event_results, group_by='POLITICAL_LEANING')
    if not cross_sectional.empty:
        for _, row in cross_sectional.iterrows():
            sig = "*" if row.get('P_VALUE', 1) < 0.05 else ""
            print(f"  {row['POLITICAL_LEANING']:15s}  CAR={row['MEAN_CAR']:+.4f}  "
                  f"(N={row['N']:.0f}, t={row.get('T_STAT', 0):.2f}){sig}")

    # Save results back to database
    save_output = {}
    if save:
        print("\nSaving results to database...")
        save_output = store.save_results(
            event_results=event_results,
            did_results=did_results,
            cross_sectional=cross_sectional,
            model_name=model,
        )
        saved = sum(1 for v in save_output.values() if v['status'] == 'SUCCESS')
        print(f"  Saved {saved}/{len(save_output)} tables")

    store.close()
    print("\n" + "=" * 60)

    return {
        'store': store,
        'event_results': event_results,
        'did_results': did_results,
        'cross_sectional': cross_sectional,
        'save_results': save_output,
    }


# =============================================================================
# MAIN
# =============================================================================
if __name__ == '__main__':
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
    )

    output = run_and_save()
