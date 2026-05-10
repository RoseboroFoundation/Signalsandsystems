"""
Database module for loading ETL data into AWS (Athena/Glue/S3) or SQLite.

Supports two backends:
  - AWS (cloud) via Athena/Glue catalog with Parquet on S3
  - SQLite (local fallback) for offline use

Usage:
    # Load into AWS knowledge graph (Athena/Glue/S3)
    from Database import load_to_athena
    load_to_athena()

    # Load into SQLite (local fallback)
    from Database import load_to_sqlite
    load_to_sqlite()                         # -> ./data/signals_systems.db
    load_to_sqlite(db_path='custom.db')

    # Auto-select: tries AWS, falls back to SQLite
    from Database import load_to_database
    load_to_database()

    # Query Athena after loading
    from Database import AthenaLoader
    with AthenaLoader() as db:
        df = db.read_table('INFLATION_DATA')
        df = db.run_query('SELECT * FROM TREASURY_YIELDS WHERE DATE > DATE "2020-01-01"')

    # Query SQLite after loading
    from Database import SQLiteLoader
    with SQLiteLoader() as db:
        df = db.read_table('INFLATION_DATA')
"""

import logging
import os
import sqlite3
import time
from datetime import datetime, timezone
from io import BytesIO

import pandas as pd
from dotenv import load_dotenv

logger = logging.getLogger(__name__)

# =============================================================================
# CONFIGURATION
# =============================================================================
_LOCAL_ENV = os.path.join(os.path.dirname(os.path.abspath(__file__)), '.env')
load_dotenv(_LOCAL_ENV)

# AWS configuration
AWS_REGION = os.getenv('AWS_REGION', 'us-east-1')
AWS_PROFILE = os.getenv('AWS_PROFILE', None)  # None = use default credential chain

# Athena / Glue / S3 configuration
GLUE_DATABASE = os.getenv('GLUE_DATABASE', 'roseboro_research')
ATHENA_WORKGROUP = os.getenv('ATHENA_WORKGROUP', 'roseboro')
ATHENA_RESULTS_BUCKET = os.getenv('ATHENA_RESULTS_BUCKET', 's3://roseboro-athena-results/')
S3_DATA_BUCKET = os.getenv('S3_DATA_BUCKET', 'roseboro-snowflake-export')  # legacy name; set S3_DATA_BUCKET to override
S3_DATA_PREFIX = os.getenv('S3_DATA_PREFIX', 'signals_systems')

_VALID_EXTRACT_MODES = {'dataframe', 'combined', 'concat', 'multi', 'nested'}

# =============================================================================
# TABLE MAPPING
# =============================================================================
# Maps ETL data_dict keys to table names and extraction strategy.
#
# extract_mode:
#   'dataframe'  - value is a DataFrame directly
#   'combined'   - value is a dict, use the 'combined' key
#   'concat'     - value is a dict of DataFrames, concat them all
#   'multi'      - value is a dict, write multiple sub-tables
#   'nested'     - value is a nested dict, extract 'combined' from sub-dicts

TABLE_MAP = {
    # --- Market Data ---
    'culturewardata': {
        'table': 'CULTURE_WAR_COMPANIES',
        'extract_mode': 'dataframe',
        'description': 'Culture war company events (160 companies)',
    },
    'stockdata': {
        'table': 'STOCK_DATA',
        'extract_mode': 'concat',
        'description': 'Historical stock prices for culture war companies',
    },
    'vixdata': {
        'table': 'VIX_DATA',
        'extract_mode': 'dataframe',
        'description': 'CBOE Volatility Index daily values',
    },
    'ff_factors': {
        'table': 'FAMA_FRENCH_FACTORS',
        'extract_mode': 'multi',
        'sub_tables': {
            'FF3': 'FF3_FACTORS',
            'FF5': 'FF5_FACTORS',
            'MOM': 'MOMENTUM_FACTORS',
        },
        'description': 'Fama-French 3-factor, 5-factor, and Momentum',
    },
    'newsdata': {
        'table': 'NEWS_DATA',
        'extract_mode': 'dataframe',
        'description': 'News articles from Guardian, NYT, Reddit',
    },

    # --- Inflation ---
    'inflationdata': {
        'table': 'INFLATION_DATA',
        'extract_mode': 'combined',
        'description': 'Core inflation measures (CPI, PCE, PPI) with YoY/MoM',
    },
    'inflation_expectations': {
        'table': 'INFLATION_EXPECTATIONS',
        'extract_mode': 'combined',
        'description': 'Breakeven inflation, survey expectations, Fed measures',
    },
    'comprehensive_inflation': {
        'table': 'INFLATION_COMPREHENSIVE',
        'extract_mode': 'combined',
        'description': 'All inflation measures combined with component-level CPI',
    },

    # --- Interest Rates ---
    'treasury_yields': {
        'table': 'TREASURY_YIELDS',
        'extract_mode': 'combined',
        'description': 'Treasury yield curve (1M-30Y) and TIPS real yields',
    },
    'policy_rates': {
        'table': 'POLICY_RATES',
        'extract_mode': 'combined',
        'description': 'Fed Funds, SOFR, Prime, Discount rates',
    },
    'credit_spreads': {
        'table': 'CREDIT_SPREADS',
        'extract_mode': 'combined',
        'description': 'Corporate yields, credit spreads, mortgage rates',
    },
    'comprehensive_rates': {
        'table': 'RATES_COMPREHENSIVE',
        'extract_mode': 'combined',
        'description': 'All rates with yield curve metrics',
    },

    # --- Industrial Production ---
    'industrial_production': {
        'table': 'INDUSTRIAL_PRODUCTION',
        'extract_mode': 'combined',
        'description': 'IP indices, sector production, capacity utilization',
    },
    'ip_growth': {
        'table': 'IP_GROWTH_RATES',
        'extract_mode': 'combined',
        'description': 'IP growth rates (YoY, MoM) and diffusion indices',
    },
    'comprehensive_ip': {
        'table': 'IP_COMPREHENSIVE',
        'extract_mode': 'combined',
        'description': 'All industrial production measures combined',
    },

    # --- Money Supply ---
    'money_supply': {
        'table': 'MONEY_SUPPLY',
        'extract_mode': 'combined',
        'description': 'M1, M2, monetary base, and components',
    },
    'money_velocity': {
        'table': 'MONEY_VELOCITY',
        'extract_mode': 'combined',
        'description': 'M1 and M2 velocity of money',
    },
    'fed_balance_sheet': {
        'table': 'FED_BALANCE_SHEET',
        'extract_mode': 'combined',
        'description': 'Fed total assets, Treasury/MBS holdings, reserves',
    },
    'comprehensive_m2': {
        'table': 'M2_COMPREHENSIVE',
        'extract_mode': 'combined',
        'description': 'All money supply measures with growth rates',
    },

    # --- GDP ---
    'gdp_data': {
        'table': 'GDP_DATA',
        'extract_mode': 'combined',
        'description': 'Nominal/Real GDP, growth rates, per capita',
    },
    'gdp_components': {
        'table': 'GDP_COMPONENTS',
        'extract_mode': 'combined',
        'description': 'GDP expenditure components (C + I + G + NX)',
    },
    'gdp_industry': {
        'table': 'GDP_INDUSTRY',
        'extract_mode': 'combined',
        'description': 'GDP by industry/sector (value added)',
    },
    'comprehensive_gdp': {
        'table': 'GDP_COMPREHENSIVE',
        'extract_mode': 'combined',
        'description': 'All GDP measures combined',
    },

    # --- Employment ---
    'employment_data': {
        'table': 'EMPLOYMENT_DATA',
        'extract_mode': 'combined',
        'description': 'Payrolls, unemployment rates, labor force',
    },
    'jobless_claims': {
        'table': 'JOBLESS_CLAIMS',
        'extract_mode': 'combined',
        'description': 'Initial/continuing claims, insured unemployment rate',
    },
    'wages_hours': {
        'table': 'WAGES_HOURS',
        'extract_mode': 'combined',
        'description': 'Average earnings, weekly hours, ECI, unit labor costs',
    },
    'jolts_data': {
        'table': 'JOLTS_DATA',
        'extract_mode': 'combined',
        'description': 'Job openings, hires, quits, layoffs',
    },
    'comprehensive_employment': {
        'table': 'EMPLOYMENT_COMPREHENSIVE',
        'extract_mode': 'combined',
        'description': 'All employment measures combined',
    },

    # --- Additional Macro ---
    'additional_macro': {
        'table': 'ADDITIONAL_MACRO',
        'extract_mode': 'dataframe',
        'description': 'Consumer Sentiment, Housing Starts, Home Prices, Dollar Index',
    },

    # --- Political Events & Exposure ---
    'political_events': {
        'table': 'POLITICAL_EVENTS',
        'extract_mode': 'dataframe',
        'description': 'Political fundamental events (votes, EOs, court decisions)',
    },
    'political_exposure': {
        'table': 'POLITICAL_EXPOSURE',
        'extract_mode': 'dataframe',
        'description': 'Firm-level political exposure (lobbying, PAC)',
    },

    # --- SEC Filings ---
    'form4data': {
        'table': 'FORM4_TRANSACTIONS',
        'extract_mode': 'dataframe',
        'description': 'SEC Form 4 insider trading filings for culture war companies',
    },
    'controlcompanies': {
        'table': 'CONTROL_COMPANIES',
        'extract_mode': 'dataframe',
        'description': 'Industry-matched control companies for treatment firms',
    },

    # --- Essay 1 — Volatility Regimes & FF5 ---
    'essay1_regime_summary': {
        'table': 'ESSAY1_REGIME_SUMMARY',
        'extract_mode': 'dataframe',
        'description': 'VIX regime summary statistics (means, durations, counts)',
    },
    'essay1_model_selection': {
        'table': 'ESSAY1_MODEL_SELECTION',
        'extract_mode': 'dataframe',
        'description': 'Markov-switching model selection (BIC, AIC by n_regimes)',
    },
    'essay1_transition_matrix': {
        'table': 'ESSAY1_TRANSITION_MATRIX',
        'extract_mode': 'dataframe',
        'description': 'Regime transition probabilities',
    },
    'essay1_lr_tests': {
        'table': 'ESSAY1_LR_TESTS',
        'extract_mode': 'dataframe',
        'description': 'Likelihood-ratio tests for number of regimes',
    },
    'essay1_coefficients': {
        'table': 'ESSAY1_COEFFICIENTS',
        'extract_mode': 'dataframe',
        'description': 'Regime-conditional factor model coefficients',
    },
    'essay1_interaction': {
        'table': 'ESSAY1_INTERACTION',
        'extract_mode': 'dataframe',
        'description': 'Regime × factor interaction terms',
    },
    'essay1_ff5_coefficients': {
        'table': 'ESSAY1_FF5_COEFFICIENTS',
        'extract_mode': 'dataframe',
        'description': 'FF5 model coefficients by regime',
    },
    'essay1_factor_premia': {
        'table': 'ESSAY1_FACTOR_PREMIA',
        'extract_mode': 'dataframe',
        'description': 'Factor premia comparison across regimes',
    },
    'essay1_chow_test': {
        'table': 'ESSAY1_CHOW_TEST',
        'extract_mode': 'dataframe',
        'description': 'Chow structural break test for regime shifts',
    },
    'essay1_cw_stock_results': {
        'table': 'ESSAY1_CW_STOCK_RESULTS',
        'extract_mode': 'dataframe',
        'description': 'Culture war stock-level regime analysis results',
    },
    'essay1_cw_regime_aggregates': {
        'table': 'ESSAY1_CW_REGIME_AGGREGATES',
        'extract_mode': 'dataframe',
        'description': 'Culture war aggregate statistics by regime',
    },
    'essay1_cw_vs_market': {
        'table': 'ESSAY1_CW_VS_MARKET',
        'extract_mode': 'dataframe',
        'description': 'Culture war portfolio vs market comparison by regime',
    },
    'essay1_ctrl_coefficients': {
        'table': 'ESSAY1_CTRL_COEFFICIENTS',
        'extract_mode': 'dataframe',
        'description': 'Control firm regime-conditional coefficients',
    },
    'essay1_ctrl_cw_stock_results': {
        'table': 'ESSAY1_CTRL_CW_STOCK_RESULTS',
        'extract_mode': 'dataframe',
        'description': 'Control firm stock-level regime results',
    },
    'essay1_ctrl_cw_aggregates': {
        'table': 'ESSAY1_CTRL_CW_AGGREGATES',
        'extract_mode': 'dataframe',
        'description': 'Control firm aggregate statistics by regime',
    },
    'essay1_ctrl_interaction': {
        'table': 'ESSAY1_CTRL_INTERACTION',
        'extract_mode': 'dataframe',
        'description': 'Control firm regime × factor interaction terms',
    },
    'essay1_sentiment_daily': {
        'table': 'ESSAY1_SENTIMENT_DAILY',
        'extract_mode': 'dataframe',
        'description': 'Daily sentiment scores by regime',
    },
    'essay1_fomo_by_regime': {
        'table': 'ESSAY1_FOMO_BY_REGIME',
        'extract_mode': 'dataframe',
        'description': 'Fear-of-missing-out metric by regime',
    },
    'essay1_matched_coverage': {
        'table': 'ESSAY1_MATCHED_COVERAGE',
        'extract_mode': 'dataframe',
        'description': 'Matched control analysis coverage per stock',
    },
    'essay1_matched_ttest': {
        'table': 'ESSAY1_MATCHED_TTEST',
        'extract_mode': 'dataframe',
        'description': 'Matched control paired t-tests by regime × factor',
    },
    'essay1_matched_deltas': {
        'table': 'ESSAY1_MATCHED_DELTAS',
        'extract_mode': 'dataframe',
        'description': 'Treatment minus control deltas per stock × regime',
    },
    'essay1_matched_sign': {
        'table': 'ESSAY1_MATCHED_SIGN',
        'extract_mode': 'dataframe',
        'description': 'Binomial sign test for matched pair direction',
    },
    'essay1_matched_amplification': {
        'table': 'ESSAY1_MATCHED_AMPLIFICATION',
        'extract_mode': 'dataframe',
        'description': 'Treatment amplification ratio vs matched controls',
    },

    # --- Essay 2 — NLP, Political Alignment & Event DiD ---
    'essay2_news_sentiment': {
        'table': 'ESSAY2_NEWS_SENTIMENT',
        'extract_mode': 'dataframe',
        'description': 'FinBERT-scored news sentiment per article',
    },
    'essay2_filing_sentiment': {
        'table': 'ESSAY2_FILING_SENTIMENT',
        'extract_mode': 'dataframe',
        'description': 'SEC filing MDA/risk section sentiment scores',
    },
    'essay2_event_nlp': {
        'table': 'ESSAY2_EVENT_NLP',
        'extract_mode': 'dataframe',
        'description': 'Event-level NLP panel (news + filing sentiment around events)',
    },
    'essay2_distinctive_phrases': {
        'table': 'ESSAY2_DISTINCTIVE_PHRASES',
        'extract_mode': 'dataframe',
        'description': 'Partisan distinctive phrases from party platforms',
    },
    'essay2_political_alignment': {
        'table': 'ESSAY2_POLITICAL_ALIGNMENT',
        'extract_mode': 'dataframe',
        'description': 'Computed political alignment scores per firm',
    },
    'essay2_event_alignment': {
        'table': 'ESSAY2_EVENT_ALIGNMENT',
        'extract_mode': 'dataframe',
        'description': 'Event-level political alignment classification',
    },
    'essay2_alignment_validation': {
        'table': 'ESSAY2_ALIGNMENT_VALIDATION',
        'extract_mode': 'dataframe',
        'description': 'Political alignment validation metrics',
    },
    'essay2_car_panel': {
        'table': 'ESSAY2_CAR_PANEL',
        'extract_mode': 'dataframe',
        'description': 'Cumulative abnormal returns panel per event',
    },
    'essay2_did_coefficients': {
        'table': 'ESSAY2_DID_COEFFICIENTS',
        'extract_mode': 'dataframe',
        'description': 'Difference-in-differences regression coefficients',
    },
    'essay2_parallel_trends': {
        'table': 'ESSAY2_PARALLEL_TRENDS',
        'extract_mode': 'dataframe',
        'description': 'Parallel trends pre-treatment test results',
    },
    'essay2_peer_parallel_trends': {
        'table': 'ESSAY2_PEER_PARALLEL_TRENDS',
        'extract_mode': 'dataframe',
        'description': 'Peer firm parallel trends test results',
    },
    'essay2_multi_window_panel': {
        'table': 'ESSAY2_MULTI_WINDOW_PANEL',
        'extract_mode': 'dataframe',
        'description': 'Multi-window CAR panel (1d to 60d)',
    },
    'essay2_multi_window_summary': {
        'table': 'ESSAY2_MULTI_WINDOW_SUMMARY',
        'extract_mode': 'dataframe',
        'description': 'Multi-window CAR summary statistics',
    },
    'essay2_multi_window_by_lean': {
        'table': 'ESSAY2_MULTI_WINDOW_BY_LEAN',
        'extract_mode': 'dataframe',
        'description': 'Multi-window CARs by political leaning',
    },
    'essay2_multi_window_treat_vs_ctrl': {
        'table': 'ESSAY2_MULTI_WINDOW_TREAT_VS_CTRL',
        'extract_mode': 'dataframe',
        'description': 'Multi-window treatment vs control comparison',
    },
    'essay2_contagion_panel': {
        'table': 'ESSAY2_CONTAGION_PANEL',
        'extract_mode': 'dataframe',
        'description': 'Industry contagion spillover panel',
    },
    'essay2_contagion_summary': {
        'table': 'ESSAY2_CONTAGION_SUMMARY',
        'extract_mode': 'dataframe',
        'description': 'Contagion effect summary by window',
    },
    'essay2_contagion_by_lean': {
        'table': 'ESSAY2_CONTAGION_BY_LEAN',
        'extract_mode': 'dataframe',
        'description': 'Contagion effects by political leaning',
    },
    'essay2_contagion_by_facing': {
        'table': 'ESSAY2_CONTAGION_BY_FACING',
        'extract_mode': 'dataframe',
        'description': 'Contagion effects by consumer-facing status',
    },
    'essay2_contagion_peer_vs_nonpeer': {
        'table': 'ESSAY2_CONTAGION_PEER_VS_NONPEER',
        'extract_mode': 'dataframe',
        'description': 'Contagion peer vs non-peer comparison',
    },
    'essay2_contagion_cons_vs_b2b': {
        'table': 'ESSAY2_CONTAGION_CONS_VS_B2B',
        'extract_mode': 'dataframe',
        'description': 'Contagion consumer vs B2B comparison',
    },
    'essay2_contagion_lean_pairwise': {
        'table': 'ESSAY2_CONTAGION_LEAN_PAIRWISE',
        'extract_mode': 'dataframe',
        'description': 'Contagion pairwise leaning comparisons',
    },
    'essay2_contagion_lean_mech': {
        'table': 'ESSAY2_CONTAGION_LEAN_MECH',
        'extract_mode': 'dataframe',
        'description': 'Contagion leaning mechanism analysis',
    },
    'essay2_contagion_tight_diff': {
        'table': 'ESSAY2_CONTAGION_TIGHT_DIFF',
        'extract_mode': 'dataframe',
        'description': 'Contagion tight vs diffuse event comparison',
    },
    'essay2_enhanced_contagion_panel': {
        'table': 'ESSAY2_ENHANCED_CONTAGION_PANEL',
        'extract_mode': 'dataframe',
        'description': 'Enhanced contagion panel with additional controls',
    },

    # --- Essay 3 — Insider Trading & Political Controversies ---
    'essay3_insider_panel': {
        'table': 'ESSAY3_INSIDER_PANEL',
        'extract_mode': 'dataframe',
        'description': 'Event-insider panel with Form 4 metrics per window',
    },
    'essay3_window_summary': {
        'table': 'ESSAY3_WINDOW_SUMMARY',
        'extract_mode': 'dataframe',
        'description': 'Aggregate insider trading stats per window',
    },
    'essay3_abnormal_selling': {
        'table': 'ESSAY3_ABNORMAL_SELLING',
        'extract_mode': 'dataframe',
        'description': 'Pre-event vs benchmark abnormal selling tests',
    },
    'essay3_car_insider_regression': {
        'table': 'ESSAY3_CAR_INSIDER_REGRESSION',
        'extract_mode': 'dataframe',
        'description': 'Cross-sectional CAR ~ insider selling regression',
    },
    'essay3_leaning_analysis': {
        'table': 'ESSAY3_LEANING_ANALYSIS',
        'extract_mode': 'dataframe',
        'description': 'Insider trading by political alignment group',
    },
    'essay3_treatment_vs_control': {
        'table': 'ESSAY3_TREATMENT_VS_CONTROL',
        'extract_mode': 'dataframe',
        'description': 'DiD: insider selling in treatment vs control firms',
    },
    'essay3_routine_vs_opportunistic': {
        'table': 'ESSAY3_ROUTINE_VS_OPPORTUNISTIC',
        'extract_mode': 'dataframe',
        'description': 'Cohen et al routine vs opportunistic trade decomposition',
    },
    'essay3_regime_interaction': {
        'table': 'ESSAY3_REGIME_INTERACTION',
        'extract_mode': 'dataframe',
        'description': 'VIX regime × insider trading interaction',
    },
    'essay3_placebo_test': {
        'table': 'ESSAY3_PLACEBO_TEST',
        'extract_mode': 'dataframe',
        'description': 'Placebo permutation test with random pseudo-event dates',
    },
    'essay3_acceleration_test': {
        'table': 'ESSAY3_ACCELERATION_TEST',
        'extract_mode': 'dataframe',
        'description': 'Jonckheere-Terpstra trend test across pre-event windows',
    },
    'essay3_information_gradient': {
        'table': 'ESSAY3_INFORMATION_GRADIENT',
        'extract_mode': 'dataframe',
        'description': 'CAR magnitude × insider selling information gradient',
    },
}


# =============================================================================
# SHARED UTILITIES
# =============================================================================
def _prepare_dataframe(df):
    """
    Prepare a DataFrame for database ingestion (shared by all backends).

    - Resets DatetimeIndex to a DATE column
    - Uppercases column names
    - Deduplicates column names
    - Strips timezone info (converts to naive UTC)
    - Converts PeriodIndex columns to timestamps
    """
    write_df = df.copy()

    # If index is a DatetimeIndex, reset it to a column named DATE
    if isinstance(write_df.index, pd.DatetimeIndex):
        write_df.index.name = write_df.index.name or 'DATE'
        write_df = write_df.reset_index()
    elif write_df.index.name and write_df.index.name != 'index':
        write_df = write_df.reset_index()

    # Uppercase column names
    write_df.columns = [str(c).upper().replace(' ', '_').replace('-', '_') for c in write_df.columns]

    # Deduplicate column names (original kept clean, first duplicate gets _2)
    seen = {}
    new_cols = []
    for col in write_df.columns:
        if col in seen:
            seen[col] += 1
            new_cols.append(f"{col}_{seen[col]}")
        else:
            seen[col] = 1
            new_cols.append(col)
    write_df.columns = new_cols

    # Convert timezone-aware datetimes to naive UTC
    for col in write_df.columns:
        if pd.api.types.is_datetime64_any_dtype(write_df[col]):
            if write_df[col].dt.tz is not None:
                write_df[col] = write_df[col].dt.tz_convert('UTC').dt.tz_localize(None)

    # Convert PeriodIndex columns to timestamps
    for col in write_df.columns:
        if hasattr(write_df[col], 'dtype') and str(write_df[col].dtype).startswith('period'):
            write_df[col] = write_df[col].dt.to_timestamp()

    return write_df


def _extract_dataframe(etl_key, data, extract_mode):
    """Extract a DataFrame from an ETL data entry (shared by all backends)."""
    if extract_mode not in _VALID_EXTRACT_MODES:
        logger.warning(
            "Unknown extract_mode '%s' for key '%s' — skipping",
            extract_mode, etl_key,
        )
        return None

    if extract_mode == 'dataframe':
        if isinstance(data, pd.DataFrame):
            return data
        return None

    if extract_mode == 'combined':
        if isinstance(data, dict):
            combined = data.get('combined')
            if isinstance(combined, pd.DataFrame):
                return combined
            for v in data.values():
                if isinstance(v, pd.DataFrame):
                    return v
        if isinstance(data, pd.DataFrame):
            return data
        return None

    if extract_mode == 'concat':
        if isinstance(data, dict):
            frames = []
            for v in data.values():
                if isinstance(v, pd.DataFrame) and not v.empty:
                    frames.append(v)
            if frames:
                return pd.concat(frames, ignore_index=True)
        return None

    if extract_mode == 'nested':
        if isinstance(data, dict):
            combined = data.get('combined')
            if isinstance(combined, pd.DataFrame):
                return combined
        return None

    return None


def _retry(func, max_attempts=3, base_delay=1.0, retryable_codes=None):
    """Simple exponential-backoff retry for transient AWS errors."""
    if max_attempts < 1:
        raise ValueError("max_attempts must be >= 1")
    retryable_codes = retryable_codes or {
        'ThrottlingException', 'RequestLimitExceeded', 'TooManyRequestsException',
        'ServiceUnavailable', 'InternalServerError', 'SlowDown',
    }
    last_exc = None
    for attempt in range(max_attempts):
        try:
            return func()
        except Exception as e:
            error_code = getattr(e, 'response', {}).get('Error', {}).get('Code', '')
            if error_code not in retryable_codes:
                raise
            last_exc = e
            delay = base_delay * (2 ** attempt)
            logger.warning(
                "Retryable error (%s), attempt %d/%d — waiting %.1fs",
                error_code, attempt + 1, max_attempts, delay,
            )
            time.sleep(delay)
    raise last_exc


# =============================================================================
# BASE LOADER
# =============================================================================
class BaseLoader:
    """Shared load_etl loop and extraction logic for all backends.

    Subclasses must set ``_backend_name`` and implement ``_write_table``,
    ``_print_header``, ``connect``, and ``close``.
    """

    _backend_name = '<unnamed backend>'

    def load_etl(self, data_dict, replace=True, verbose=True):
        """
        Load all datasets from an ETL data dictionary.

        Parameters
        ----------
        data_dict : dict
            Output from ETL.run_etl().
        replace : bool
            If True, drop and recreate tables. If False, append.
        verbose : bool
            If True, print progress and summary.

        Returns
        -------
        dict
            Load results: {table_name: {'rows': int, 'status': str, 'duration': float}}
        """
        results = {}
        total_start = time.time()

        if verbose:
            self._print_header(replace)

        for etl_key, data in data_dict.items():
            if data is None:
                if verbose:
                    logger.info("  [SKIP] %-30s (no data)", etl_key)
                continue

            mapping = TABLE_MAP.get(etl_key)
            if mapping is None:
                if verbose:
                    logger.info("  [SKIP] %-30s (no table mapping)", etl_key)
                continue

            extract_mode = mapping['extract_mode']

            if extract_mode == 'multi':
                sub_tables = mapping.get('sub_tables', {})
                for sub_key, sub_table in sub_tables.items():
                    sub_df = data.get(sub_key) if isinstance(data, dict) else None
                    if sub_df is not None and isinstance(sub_df, pd.DataFrame) and not sub_df.empty:
                        result = self._write_table(
                            sub_df, sub_table, replace, etl_key, verbose
                        )
                        results[sub_table] = result
                    elif verbose:
                        logger.info("  [SKIP] %-30s (sub-key '%s' empty)", sub_table, sub_key)
            else:
                df = _extract_dataframe(etl_key, data, extract_mode)
                if df is not None and not df.empty:
                    table_name = mapping['table']
                    result = self._write_table(
                        df, table_name, replace, etl_key, verbose
                    )
                    results[table_name] = result
                elif verbose:
                    logger.info("  [SKIP] %-30s (extracted empty)", etl_key)

        total_elapsed = time.time() - total_start

        self._after_load(results)

        if verbose:
            self._print_summary(results, total_elapsed)

        return results

    def _write_table(self, df, table_name, replace, etl_key, verbose):
        raise NotImplementedError

    def _print_header(self, replace):
        raise NotImplementedError

    def _after_load(self, results):
        """Hook for post-load actions (e.g. logging). Override in subclasses."""
        pass

    def _print_summary(self, results, total_elapsed):
        """Print load summary (shared structure, backend-specific header)."""
        succeeded = {k: v for k, v in results.items() if v['status'] == 'SUCCESS'}
        failed = {k: v for k, v in results.items() if v['status'] != 'SUCCESS'}
        total_rows = sum(v['rows'] for v in succeeded.values())

        logger.info("")
        logger.info("=" * 60)
        logger.info("  %s Load Summary", self._backend_name)
        logger.info("=" * 60)
        self._print_summary_details()
        logger.info("  Tables loaded:  %d", len(succeeded))
        logger.info("  Tables failed:  %d", len(failed))
        logger.info("  Total rows:     %s", f"{total_rows:,}")
        logger.info("  Total time:     %.1fs", total_elapsed)
        logger.info("")

        if succeeded:
            logger.info("  --- Loaded Tables ---")
            for table, info in sorted(succeeded.items()):
                logger.info(
                    "    %-35s %6d rows  (%.1fs)",
                    table, info['rows'], info['duration'],
                )

        if failed:
            logger.info("")
            logger.info("  --- Failed Tables ---")
            for table, info in sorted(failed.items()):
                logger.info("    %-35s %s", table, info.get('error', 'unknown'))

        logger.info("=" * 60)

    def _print_summary_details(self):
        """Print backend-specific summary lines. Override in subclasses."""
        pass

    @property
    def _is_connected(self):
        """Check if the loader has an active connection. Override in subclasses."""
        return False

    # -----------------------------------------------------------------
    # Public write interface
    # -----------------------------------------------------------------
    def write_table(self, df, table_name, replace=True):
        """
        Write a DataFrame to a table. Must be called after connect() or
        within a context manager.

        Parameters
        ----------
        df : pd.DataFrame
        table_name : str
            Target table name (will be uppercased).
        replace : bool
            If True, drop and recreate. If False, append.

        Returns
        -------
        dict with keys: rows, status, duration, error.
        """
        if not self._is_connected:
            raise RuntimeError(
                "Not connected. Call connect() or use as a context manager "
                "before calling write_table()."
            )
        table_name = table_name.upper()
        return self._write_table(df, table_name, replace, etl_key=table_name.lower(), verbose=True)


# =============================================================================
# ATHENA / GLUE / S3 LOADER
# =============================================================================
class AthenaLoader(BaseLoader):
    """Loads ETL pipeline data into AWS via S3 (Parquet) + Glue catalog + Athena."""

    _backend_name = 'AWS (Athena/Glue/S3)'

    def __init__(self, database=None, workgroup=None, s3_bucket=None, s3_prefix=None):
        self.database = database or GLUE_DATABASE
        self.workgroup = workgroup or ATHENA_WORKGROUP
        self.s3_bucket = s3_bucket or S3_DATA_BUCKET
        self.s3_prefix = s3_prefix or S3_DATA_PREFIX
        self._s3 = None
        self._glue = None
        self._athena = None
        self._connected = False

    def connect(self):
        """Establish AWS connections via boto3."""
        import boto3

        session_kwargs = {'region_name': AWS_REGION}
        if AWS_PROFILE:
            session_kwargs['profile_name'] = AWS_PROFILE

        session = boto3.Session(**session_kwargs)
        self._s3 = session.client('s3')
        self._glue = session.client('glue')
        self._athena = session.client('athena')
        self._connected = True

        try:
            self._check_access()
        except Exception:
            self.close()
            raise

        logger.info(
            "Connected to AWS: Glue DB=%s, S3=s3://%s/%s/, workgroup=%s",
            self.database, self.s3_bucket, self.s3_prefix, self.workgroup,
        )
        return self

    def _check_access(self):
        """Verify we can reach S3 and Glue."""
        try:
            _retry(lambda: self._s3.head_bucket(Bucket=self.s3_bucket))
        except Exception as e:
            raise ConnectionError(
                f"Cannot access S3 bucket s3://{self.s3_bucket}: {e}"
            ) from e

        try:
            self._glue.get_database(Name=self.database)
        except self._glue.exceptions.EntityNotFoundException:
            logger.info("Glue database '%s' not found — will create it.", self.database)
            _retry(lambda: self._glue.create_database(
                DatabaseInput={'Name': self.database, 'Description': 'Signals & Systems research data'}
            ))
        except Exception as e:
            raise ConnectionError(
                f"Cannot access Glue database '{self.database}': {e}"
            ) from e

    def close(self):
        """Close AWS connections. Safe to call multiple times."""
        if not self._connected:
            return
        self._s3 = None
        self._glue = None
        self._athena = None
        self._connected = False
        logger.info("AWS connections closed")

    @property
    def _is_connected(self):
        return self._connected

    def __enter__(self):
        self.connect()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close()

    def _print_header(self, replace):
        logger.info("=" * 60)
        logger.info("  Loading ETL data into AWS (S3 + Glue + Athena)")
        logger.info("  Glue DB:    %s", self.database)
        logger.info("  S3 path:    s3://%s/%s/", self.s3_bucket, self.s3_prefix)
        logger.info("  Workgroup:  %s", self.workgroup)
        logger.info("  Mode:       %s", "REPLACE" if replace else "APPEND")
        logger.info("=" * 60)

    def _print_summary_details(self):
        logger.info("  Glue DB:        %s", self.database)
        logger.info("  S3 location:    s3://%s/%s/", self.s3_bucket, self.s3_prefix)

    def _write_table(self, df, table_name, replace, etl_key, verbose):
        """Write a DataFrame as Parquet to S3 and register in Glue catalog."""
        t0 = time.time()
        status = 'SUCCESS'
        error_msg = None
        rows = len(df)
        s3_uploaded = False
        glue_registered = False

        try:
            write_df = _prepare_dataframe(df)

            # Convert object columns with mixed types to string for Parquet
            for col in write_df.columns:
                if write_df[col].dtype == object:
                    write_df[col] = write_df[col].astype(str)

            glue_table = table_name.lower()

            # In replace mode, clear any stale files from prior appends
            if replace:
                self._clear_s3_prefix(glue_table)

            # Generate a timestamped key for append mode, flat key for replace
            if replace:
                s3_key = f"{self.s3_prefix}/{glue_table}/data.parquet"
            else:
                ts = datetime.now(timezone.utc).strftime('%Y%m%d_%H%M%S')
                s3_key = f"{self.s3_prefix}/{glue_table}/data_{ts}.parquet"

            # Write Parquet to S3 via managed upload (handles multipart for large files)
            buf = BytesIO()
            write_df.to_parquet(buf, index=False, engine='pyarrow')
            buf.seek(0)

            parquet_bytes = buf.getvalue()
            buf.close()

            _retry(lambda: self._s3.upload_fileobj(
                BytesIO(parquet_bytes), self.s3_bucket, s3_key,
            ))
            s3_uploaded = True

            # Register/update Glue table (skip on append if schema unchanged)
            should_register = replace
            if not replace:
                should_register = not self._glue_schema_matches(glue_table, write_df)
            if should_register:
                _retry(lambda: self._register_glue_table(glue_table, write_df, etl_key))
            glue_registered = True

        except Exception as e:
            status = 'FAILED'
            error_msg = str(e)[:500]
            rows = 0
            if s3_uploaded and not glue_registered:
                logger.error(
                    "  [FAIL] %s: %s (S3 data orphaned — not registered in Glue)",
                    table_name, e,
                )
            else:
                logger.error("  [FAIL] %s: %s", table_name, e)

        elapsed = time.time() - t0

        if verbose and status == 'SUCCESS':
            logger.info(
                "  [OK]   %-30s %6d rows  (%4.1fs)",
                table_name, rows, elapsed,
            )

        return {
            'rows': rows,
            'status': status,
            'duration': elapsed,
            'error': error_msg,
            'etl_key': etl_key,
        }

    def _clear_s3_prefix(self, glue_table):
        """Delete all existing objects under a table's S3 prefix before replace."""
        prefix = f"{self.s3_prefix}/{glue_table}/"
        total_deleted = 0
        continuation_token = None

        while True:
            list_kwargs = {'Bucket': self.s3_bucket, 'Prefix': prefix}
            if continuation_token:
                list_kwargs['ContinuationToken'] = continuation_token

            resp = self._s3.list_objects_v2(**list_kwargs)
            objects = resp.get('Contents', [])
            if not objects:
                break

            delete_keys = [{'Key': obj['Key']} for obj in objects]
            self._s3.delete_objects(
                Bucket=self.s3_bucket,
                Delete={'Objects': delete_keys, 'Quiet': True},
            )
            total_deleted += len(delete_keys)

            if not resp.get('IsTruncated'):
                break
            continuation_token = resp.get('NextContinuationToken')

        if total_deleted:
            logger.debug("Cleared %d stale S3 objects under %s", total_deleted, prefix)

    def _glue_schema_matches(self, table_name, df):
        """Check if the Glue table already exists with matching columns."""
        try:
            resp = self._glue.get_table(DatabaseName=self.database, Name=table_name)
            existing_cols = {
                c['Name']: c['Type']
                for c in resp['Table']['StorageDescriptor']['Columns']
            }
            new_cols = {
                col.lower(): self._pandas_dtype_to_glue(df[col].dtype)
                for col in df.columns
            }
            return existing_cols == new_cols
        except Exception as e:
            logger.debug("Could not check Glue schema for '%s': %s", table_name, e)
            return False

    def _pandas_dtype_to_glue(self, dtype):
        """Map pandas dtype to Glue/Athena column type."""
        dtype_str = str(dtype)
        if 'int' in dtype_str:
            return 'bigint'
        if 'float' in dtype_str:
            return 'double'
        if 'datetime' in dtype_str:
            return 'timestamp'
        if 'bool' in dtype_str:
            return 'boolean'
        return 'string'

    def _register_glue_table(self, table_name, df, etl_key):
        """Create or update a Glue catalog table pointing to S3 Parquet data."""
        s3_location = f"s3://{self.s3_bucket}/{self.s3_prefix}/{table_name}/"

        columns = [
            {'Name': col.lower(), 'Type': self._pandas_dtype_to_glue(df[col].dtype)}
            for col in df.columns
        ]

        mapping = TABLE_MAP.get(etl_key, {})
        description = mapping.get('description', f'Signals & Systems: {table_name}')

        table_input = {
            'Name': table_name,
            'Description': description,
            'StorageDescriptor': {
                'Columns': columns,
                'Location': s3_location,
                'InputFormat': 'org.apache.hadoop.hive.ql.io.parquet.MapredParquetInputFormat',
                'OutputFormat': 'org.apache.hadoop.hive.ql.io.parquet.MapredParquetOutputFormat',
                'SerdeInfo': {
                    'SerializationLibrary': 'org.apache.hadoop.hive.ql.io.parquet.serde.ParquetHiveSerDe',
                    'Parameters': {'serialization.format': '1'},
                },
                'Compressed': False,
            },
            'TableType': 'EXTERNAL_TABLE',
            'Parameters': {
                'classification': 'parquet',
                'has_encrypted_data': 'false',
                'ss_etl_key': etl_key,
                'ss_last_updated': datetime.now(timezone.utc).isoformat(),
            },
        }

        try:
            self._glue.update_table(
                DatabaseName=self.database,
                TableInput=table_input,
            )
        except self._glue.exceptions.EntityNotFoundException:
            self._glue.create_table(
                DatabaseName=self.database,
                TableInput=table_input,
            )

    # -----------------------------------------------------------------
    # Query helpers
    # -----------------------------------------------------------------
    def _wait_for_query(self, execution_id, poll_interval=0.5, max_wait=300):
        """Poll Athena until query completes or fails."""
        start = time.monotonic()
        while (time.monotonic() - start) < max_wait:
            resp = self._athena.get_query_execution(QueryExecutionId=execution_id)
            state = resp['QueryExecution']['Status']['State']
            if state == 'SUCCEEDED':
                return resp
            if state in ('FAILED', 'CANCELLED'):
                reason = resp['QueryExecution']['Status'].get('StateChangeReason', 'unknown')
                raise RuntimeError(f"Athena query {state}: {reason}")
            time.sleep(poll_interval)
            poll_interval = min(poll_interval * 1.5, 10.0)
        raise TimeoutError(f"Athena query timed out after {max_wait}s")

    def run_query(self, sql):
        """Run a SQL query via Athena and return results as a DataFrame."""
        resp = self._athena.start_query_execution(
            QueryString=sql,
            QueryExecutionContext={'Database': self.database},
            WorkGroup=self.workgroup,
        )
        execution_id = resp['QueryExecutionId']

        final_resp = self._wait_for_query(execution_id)

        # Get results location from the already-fetched response
        result_location = final_resp['QueryExecution']['ResultConfiguration']['OutputLocation']

        if not result_location.startswith('s3://'):
            raise ValueError(f"Unexpected result location format: {result_location!r}")

        # Parse s3://bucket/key from the output location
        parts = result_location[5:].split('/', 1)
        bucket = parts[0]
        key = parts[1]

        obj = self._s3.get_object(Bucket=bucket, Key=key)
        return pd.read_csv(obj['Body'])

    def read_table(self, table_name, limit=None):
        """Read a table into a pandas DataFrame via Athena.

        Unlike run_query(), this method coerces columns to their Glue catalog
        types so that timestamps come back as datetime64 and numerics as
        int/float — matching the behaviour of SQLiteLoader.read_table().
        """
        glue_table = table_name.lower()
        query = f'SELECT * FROM "{glue_table}"'
        if limit:
            query += f" LIMIT {limit}"
        df = self.run_query(query)
        return self._coerce_types(df, glue_table)

    def _coerce_types(self, df, glue_table):
        """Cast CSV-typed columns to their Glue catalog types."""
        try:
            resp = self._glue.get_table(DatabaseName=self.database, Name=glue_table)
            columns = resp['Table']['StorageDescriptor']['Columns']
        except Exception:
            return df

        glue_types = {c['Name']: c['Type'] for c in columns}

        for col in df.columns:
            glue_type = glue_types.get(col.lower())
            if glue_type is None:
                continue
            try:
                if glue_type == 'timestamp':
                    df[col] = pd.to_datetime(df[col], errors='coerce')
                elif glue_type == 'bigint':
                    df[col] = pd.to_numeric(df[col], errors='coerce').astype('Int64')
                elif glue_type == 'double':
                    df[col] = pd.to_numeric(df[col], errors='coerce')
                elif glue_type == 'boolean':
                    df[col] = df[col].map({'true': True, 'false': False, 'True': True, 'False': False})
            except Exception:
                pass  # leave column as-is if coercion fails

        return df

    def list_tables(self):
        """List Signals & Systems tables in the Glue catalog."""
        tables = []
        paginator = self._glue.get_paginator('get_tables')
        for page in paginator.paginate(DatabaseName=self.database):
            for t in page['TableList']:
                location = t.get('StorageDescriptor', {}).get('Location', '')
                params = t.get('Parameters', {})
                # Only include tables under our S3 prefix with our tag
                if f'/{self.s3_prefix}/' in location and params.get('ss_etl_key'):
                    tables.append({
                        'name': t['Name'],
                        'description': t.get('Description', ''),
                        'location': location,
                    })
        return tables

    def get_table_info(self, table_name):
        """Get column info for a table from Glue catalog."""
        glue_table = table_name.lower()
        resp = self._glue.get_table(DatabaseName=self.database, Name=glue_table)
        columns = resp['Table']['StorageDescriptor']['Columns']
        return [
            {'name': c['Name'], 'type': c['Type'], 'nullable': True}
            for c in columns
        ]


# =============================================================================
# SQLITE LOADER
# =============================================================================
SQLITE_DEFAULT_PATH = os.path.join(
    os.path.dirname(os.path.abspath(__file__)), 'data', 'signals_systems.db'
)


class SQLiteLoader(BaseLoader):
    """Loads ETL pipeline data into a local SQLite database."""

    _backend_name = 'SQLite'

    def __init__(self, db_path=None):
        self.db_path = db_path or SQLITE_DEFAULT_PATH
        self.conn = None

    def connect(self):
        """Open (or create) the SQLite database."""
        parent = os.path.dirname(self.db_path)
        if parent:
            os.makedirs(parent, exist_ok=True)
        self.conn = sqlite3.connect(self.db_path)
        self.conn.execute("PRAGMA journal_mode=WAL")
        logger.info("Connected to SQLite: %s", self.db_path)
        return self

    def close(self):
        """Close the SQLite connection."""
        if self.conn:
            self.conn.close()
            self.conn = None
            logger.info("SQLite connection closed")

    @property
    def _is_connected(self):
        return self.conn is not None

    def __enter__(self):
        self.connect()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close()

    def _print_header(self, replace):
        logger.info("=" * 60)
        logger.info("  Loading ETL data into SQLite")
        logger.info("  Database: %s", self.db_path)
        logger.info("  Mode: %s", "REPLACE" if replace else "APPEND")
        logger.info("=" * 60)

    def _print_summary_details(self):
        logger.info("  Database:       %s", self.db_path)
        if os.path.exists(self.db_path):
            db_size_mb = os.path.getsize(self.db_path) / 1024 / 1024
            logger.info("  Database size:  %.1f MB", db_size_mb)

    def _after_load(self, results):
        """Log load metadata to ETL_LOAD_LOG table."""
        self._create_load_log()
        for table_name, result in results.items():
            self._log_load(
                table_name, result['rows'], result['status'],
                result.get('error'), result['duration'],
            )

    def _write_table(self, df, table_name, replace, etl_key, verbose):
        """Write a DataFrame to a SQLite table."""
        t0 = time.time()
        status = 'SUCCESS'
        error_msg = None
        rows = len(df)

        try:
            write_df = _prepare_dataframe(df)
            if_exists = 'replace' if replace else 'append'
            write_df.to_sql(table_name, self.conn, if_exists=if_exists, index=False)
        except Exception as e:
            status = 'FAILED'
            error_msg = str(e)[:500]
            rows = 0
            logger.error("  [FAIL] %s: %s", table_name, e)

        elapsed = time.time() - t0

        if verbose and status == 'SUCCESS':
            logger.info(
                "  [OK]   %-30s %6d rows  (%4.1fs)",
                table_name, rows, elapsed,
            )

        return {
            'rows': rows,
            'status': status,
            'duration': elapsed,
            'error': error_msg,
            'etl_key': etl_key,
        }

    def _create_load_log(self):
        """Create the ETL load log table if it doesn't exist."""
        self.conn.execute("""
            CREATE TABLE IF NOT EXISTS ETL_LOAD_LOG (
                LOAD_ID INTEGER PRIMARY KEY AUTOINCREMENT,
                LOAD_TIMESTAMP TEXT DEFAULT (datetime('now')),
                TABLE_NAME TEXT,
                ROWS_LOADED INTEGER,
                STATUS TEXT,
                ERROR_MESSAGE TEXT,
                DURATION_SECONDS REAL
            )
        """)

    def _log_load(self, table_name, rows, status, error_msg, duration):
        """Log a load operation."""
        try:
            self.conn.execute(
                """
                INSERT INTO ETL_LOAD_LOG
                    (LOAD_TIMESTAMP, TABLE_NAME, ROWS_LOADED, STATUS, ERROR_MESSAGE, DURATION_SECONDS)
                VALUES (datetime('now'), ?, ?, ?, ?, ?)
                """,
                (table_name, rows, status, error_msg, round(duration, 2)),
            )
            self.conn.commit()
        except Exception as e:
            logger.warning("Failed to log ETL load for %s: %s", table_name, e)

    # -----------------------------------------------------------------
    # Query helpers
    # -----------------------------------------------------------------
    def list_tables(self):
        """List all tables in the database."""
        cur = self.conn.execute(
            "SELECT name FROM sqlite_master WHERE type='table' "
            "AND name NOT LIKE 'sqlite_%' ORDER BY name"
        )
        tables = []
        for (name,) in cur.fetchall():
            count = self.conn.execute(f'SELECT COUNT(*) FROM "{name}"').fetchone()[0]
            tables.append({'name': name, 'rows': count})
        return tables

    def read_table(self, table_name, limit=None):
        """Read a table into a pandas DataFrame."""
        query = f'SELECT * FROM "{table_name}"'
        if limit:
            query += f" LIMIT {limit}"
        return pd.read_sql(query, self.conn)

    def get_table_info(self, table_name):
        """Get column info for a table."""
        cur = self.conn.execute(f'PRAGMA table_info("{table_name}")')
        return [
            {'name': r[1], 'type': r[2], 'nullable': not r[3]}
            for r in cur.fetchall()
        ]

    def run_query(self, sql):
        """Run an arbitrary SQL query and return results as a DataFrame."""
        return pd.read_sql(sql, self.conn)


# =============================================================================
# CONVENIENCE FUNCTIONS
# =============================================================================
def load_to_athena(
    categories=None,
    keys=None,
    force_refresh=False,
    replace=True,
    clean=True,
    database=None,
):
    """
    Run the full pipeline: ETL extract -> clean -> load into AWS (Athena/Glue/S3).

    Parameters
    ----------
    categories : list[str], optional
        ETL categories to load. Defaults to all.
    keys : list[str], optional
        Specific ETL keys to load (overrides categories).
    force_refresh : bool
        Bypass ETL cache and re-download from sources.
    replace : bool
        If True, replace existing tables. If False, append.
    clean : bool
        Apply data cleaning before loading.
    database : str, optional
        Override Glue database name (default: roseboro_research).

    Returns
    -------
    dict
        Load results per table.
    """
    from ETL import run_etl

    logger.info("Step 1/2: Running ETL pipeline...")
    data = run_etl(
        categories=categories,
        keys=keys,
        force_refresh=force_refresh,
        clean=clean,
    )

    logger.info("Step 2/2: Loading into AWS (S3 + Glue + Athena)...")
    with AthenaLoader(database=database) as loader:
        results = loader.load_etl(data, replace=replace)

    return results


def load_to_sqlite(
    categories=None,
    keys=None,
    force_refresh=False,
    replace=True,
    clean=True,
    db_path=None,
):
    """
    Run the full pipeline: ETL extract -> clean -> load into SQLite.

    Parameters
    ----------
    categories : list[str], optional
        ETL categories to load. Defaults to all.
    keys : list[str], optional
        Specific ETL keys to load (overrides categories).
    force_refresh : bool
        Bypass ETL cache and re-download from sources.
    replace : bool
        If True, replace existing tables. If False, append.
    clean : bool
        Apply data cleaning before loading.
    db_path : str, optional
        Path to SQLite database file (default: ./data/signals_systems.db).

    Returns
    -------
    dict
        Load results per table.
    """
    from ETL import run_etl

    logger.info("Step 1/2: Running ETL pipeline...")
    data = run_etl(
        categories=categories,
        keys=keys,
        force_refresh=force_refresh,
        clean=clean,
    )

    logger.info("Step 2/2: Loading into SQLite...")
    with SQLiteLoader(db_path=db_path) as loader:
        results = loader.load_etl(data, replace=replace)

    return results


def load_to_database(
    categories=None,
    keys=None,
    force_refresh=False,
    replace=True,
    clean=True,
):
    """
    Auto-select backend: try AWS (Athena) first, fall back to SQLite.

    Probes AWS connectivity before running ETL to avoid wasting time
    on data extraction if credentials are invalid. Reuses the probed
    connection for the actual load to avoid double-auth.

    Returns
    -------
    tuple(str, dict)
        ('athena' or 'sqlite', load results)
    """
    from ETL import run_etl

    # Probe AWS credentials before running ETL
    athena_loader = None
    try:
        athena_loader = AthenaLoader()
        athena_loader.connect()
    except ImportError:
        logger.warning("boto3 not installed. Will use SQLite.")
        if athena_loader:
            athena_loader.close()
        athena_loader = None
    except Exception as e:
        logger.warning("AWS not available: %s. Will use SQLite.", e)
        if athena_loader:
            athena_loader.close()
        athena_loader = None

    backend = 'athena' if athena_loader else 'sqlite'

    logger.info("Step 1/2: Running ETL pipeline...")
    data = run_etl(
        categories=categories,
        keys=keys,
        force_refresh=force_refresh,
        clean=clean,
    )

    logger.info("Step 2/2: Loading into %s...", backend)
    if athena_loader:
        try:
            results = athena_loader.load_etl(data, replace=replace)
            return 'athena', results
        except Exception as e:
            logger.warning("AWS load failed: %s. Falling back to SQLite.", e)
        finally:
            athena_loader.close()

    with SQLiteLoader() as loader:
        results = loader.load_etl(data, replace=replace)

    return 'sqlite', results


def list_tables(backend='athena', database=None, db_path=None):
    """List all Signals & Systems tables.

    Parameters
    ----------
    backend : str
        'athena' or 'sqlite'.
    database : str, optional
        Glue database name (for athena backend).
    db_path : str, optional
        SQLite database path (for sqlite backend).
    """
    if backend == 'athena':
        with AthenaLoader(database=database) as loader:
            tables = loader.list_tables()
    else:
        with SQLiteLoader(db_path=db_path) as loader:
            tables = loader.list_tables()

    logger.info("Tables (%s):", backend)
    logger.info("-" * 50)
    for t in tables:
        logger.info("  %s", t['name'])
    return tables


# =============================================================================
# MAIN
# =============================================================================
if __name__ == '__main__':
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
    )

    print("Signals & Systems - Database Loader")
    print(f"Started at {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print()

    backend, results = load_to_database()

    print(f"\nBackend used: {backend}")
    print(f"Completed at {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
