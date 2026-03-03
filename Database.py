"""
Database module for loading ETL data into Snowflake or SQLite.

Supports two backends:
  - Snowflake (cloud) via key-pair authentication
  - SQLite (local fallback) for offline / suspended-account use

Usage:
    # Load into Snowflake
    from Database import load_to_snowflake
    load_to_snowflake()

    # Load into SQLite (local fallback)
    from Database import load_to_sqlite
    load_to_sqlite()                         # -> ./data/signals_systems.db
    load_to_sqlite(db_path='custom.db')

    # Auto-select: tries Snowflake, falls back to SQLite
    from Database import load_to_database
    load_to_database()

    # Query SQLite after loading
    from Database import SQLiteLoader
    with SQLiteLoader() as db:
        df = db.read_table('INFLATION_DATA')
        df = db.run_query('SELECT * FROM TREASURY_YIELDS WHERE DATE > "2020-01-01"')
"""

import logging
import os
import sqlite3
import time
from datetime import datetime

import pandas as pd
from cryptography.hazmat.backends import default_backend
from cryptography.hazmat.primitives import serialization
import snowflake.connector
from snowflake.connector.pandas_tools import write_pandas
from dotenv import load_dotenv

logger = logging.getLogger(__name__)

# =============================================================================
# CONFIGURATION
# =============================================================================
# Load env from roseboro-backend (where Snowflake creds live)
_BACKEND_ENV = '/Users/administrator/Services/roseboro-backend/.env'
_LOCAL_ENV = os.path.join(os.path.dirname(os.path.abspath(__file__)), '.env')

# Load local first, then backend for Snowflake vars
load_dotenv(_LOCAL_ENV)
load_dotenv(_BACKEND_ENV, override=False)

SNOWFLAKE_USER = os.getenv('SNOWFLAKE_USER', 'ROSEBOROFOUNDATION')
SNOWFLAKE_ACCOUNT = os.getenv('SNOWFLAKE_ACCOUNT', 'FXMECDJ-MJC30157')
SNOWFLAKE_WAREHOUSE = os.getenv('SNOWFLAKE_WAREHOUSE', 'ROSEBORO_FOUNDATION')
SNOWFLAKE_DATABASE = os.getenv('SNOWFLAKE_DATABASE', 'ROSEBORO_FOUNDATION_RESEARCH')
SNOWFLAKE_ROLE = os.getenv('SNOWFLAKE_ROLE', 'ACCOUNTADMIN')
SNOWFLAKE_PRIVATE_KEY_PATH = os.getenv(
    'SNOWFLAKE_PRIVATE_KEY_PATH', '/Users/administrator/.snowflake_key'
)

TARGET_SCHEMA = 'PUBLIC'

# =============================================================================
# TABLE MAPPING
# =============================================================================
# Maps ETL data_dict keys to Snowflake table names and extraction strategy.
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
}


# =============================================================================
# SNOWFLAKE LOADER
# =============================================================================
class SnowflakeLoader:
    """Loads ETL pipeline data into Snowflake."""

    def __init__(self, schema=None):
        self.schema = schema or TARGET_SCHEMA
        self.conn = None

    def connect(self):
        """Establish Snowflake connection using key-pair authentication."""
        with open(SNOWFLAKE_PRIVATE_KEY_PATH, 'rb') as f:
            p_key = serialization.load_pem_private_key(
                f.read(), password=None, backend=default_backend()
            )

        private_key_bytes = p_key.private_bytes(
            encoding=serialization.Encoding.DER,
            format=serialization.PrivateFormat.PKCS8,
            encryption_algorithm=serialization.NoEncryption(),
        )

        self.conn = snowflake.connector.connect(
            user=SNOWFLAKE_USER,
            account=SNOWFLAKE_ACCOUNT,
            private_key=private_key_bytes,
            warehouse=SNOWFLAKE_WAREHOUSE,
            database=SNOWFLAKE_DATABASE,
            schema='PUBLIC',
            role=SNOWFLAKE_ROLE,
        )
        logger.info("Connected to Snowflake: %s.%s", SNOWFLAKE_DATABASE, self.schema)

        # Test that write operations are available
        self._check_write_access()

        return self

    def _check_write_access(self):
        """Verify the account can perform write operations."""
        cur = self.conn.cursor()
        try:
            cur.execute("CREATE TEMPORARY TABLE _ss_write_test (id NUMBER)")
            cur.execute("DROP TABLE IF EXISTS _ss_write_test")
            self._writable = True
        except snowflake.connector.errors.ProgrammingError as e:
            if '000666' in str(e):
                self._writable = False
                logger.warning(
                    "Snowflake account is SUSPENDED (no payment method). "
                    "Read-only mode: tables cannot be created or written."
                )
            else:
                raise
        finally:
            cur.close()

    def close(self):
        """Close the Snowflake connection."""
        if self.conn:
            self.conn.close()
            self.conn = None
            logger.info("Snowflake connection closed")

    def __enter__(self):
        self.connect()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close()

    def _ensure_schema(self):
        """Create the target schema if it doesn't exist, or fall back to PUBLIC."""
        cur = self.conn.cursor()
        try:
            if self.schema != 'PUBLIC':
                try:
                    cur.execute(f"CREATE SCHEMA IF NOT EXISTS {self.schema}")
                except snowflake.connector.errors.ProgrammingError as e:
                    if '000666' in str(e):
                        logger.warning(
                            "Cannot create schema %s (account suspended). "
                            "Falling back to PUBLIC schema.",
                            self.schema,
                        )
                        self.schema = 'PUBLIC'
                    else:
                        raise
            cur.execute(f"USE SCHEMA {self.schema}")
            logger.info("Using schema: %s", self.schema)
        finally:
            cur.close()

    def load_etl(self, data_dict, replace=True, verbose=True):
        """
        Load all datasets from an ETL data dictionary into Snowflake.

        Parameters
        ----------
        data_dict : dict
            Output from ETL.run_etl() or clean.load_data().
        replace : bool
            If True, drop and recreate tables. If False, append.
        verbose : bool
            If True, print progress and summary.

        Returns
        -------
        dict
            Load results: {table_name: {'rows': int, 'status': str, 'duration': float}}
        """
        if not self._writable:
            logger.error(
                "Snowflake account is suspended. Cannot load data. "
                "Please add a payment method at https://app.snowflake.com/ "
                "and try again."
            )
            return {}

        self._ensure_schema()

        results = {}
        total_start = time.time()

        if verbose:
            logger.info("=" * 60)
            logger.info("  Loading ETL data into Snowflake")
            logger.info("  Schema: %s.%s", SNOWFLAKE_DATABASE, self.schema)
            logger.info("  Mode: %s", "REPLACE" if replace else "APPEND")
            logger.info("=" * 60)

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
                # Write multiple sub-tables
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
                # Extract single DataFrame
                df = self._extract_dataframe(etl_key, data, extract_mode)
                if df is not None and not df.empty:
                    table_name = mapping['table']
                    result = self._write_table(
                        df, table_name, replace, etl_key, verbose
                    )
                    results[table_name] = result
                elif verbose:
                    logger.info("  [SKIP] %-30s (extracted empty)", etl_key)

        total_elapsed = time.time() - total_start

        # Update data dictionary
        self._update_data_dictionary(results)

        if verbose:
            self._print_summary(results, total_elapsed)

        return results

    def _extract_dataframe(self, etl_key, data, extract_mode):
        """Extract a DataFrame from the ETL data entry."""
        if extract_mode == 'dataframe':
            if isinstance(data, pd.DataFrame):
                return data
            return None

        if extract_mode == 'combined':
            if isinstance(data, dict):
                combined = data.get('combined')
                if isinstance(combined, pd.DataFrame):
                    return combined
                # Fallback: try to find any DataFrame value
                for v in data.values():
                    if isinstance(v, pd.DataFrame):
                        return v
            if isinstance(data, pd.DataFrame):
                return data
            return None

        if extract_mode == 'concat':
            if isinstance(data, dict):
                frames = []
                for k, v in data.items():
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

    def _write_table(self, df, table_name, replace, etl_key, verbose):
        """Write a DataFrame to a Snowflake table."""
        t0 = time.time()
        status = 'SUCCESS'
        error_msg = None
        rows = len(df)

        try:
            # Prepare DataFrame for Snowflake
            write_df = self._prepare_dataframe(df, table_name)

            if replace:
                self._drop_table(table_name)

            # Create table and write data
            success, num_chunks, num_rows, output = write_pandas(
                self.conn,
                write_df,
                table_name,
                schema=self.schema,
                database=SNOWFLAKE_DATABASE,
                auto_create_table=True,
                overwrite=replace,
                quote_identifiers=False,
            )

            if not success:
                status = 'FAILED'
                error_msg = 'write_pandas returned failure'
                logger.error("  [FAIL] %s: write_pandas failed", table_name)

        except Exception as e:
            status = 'FAILED'
            error_msg = str(e)[:500]
            rows = 0
            logger.error("  [FAIL] %s: %s", table_name, e)

        elapsed = time.time() - t0

        # Log the load
        self._log_load(table_name, rows, status, error_msg, elapsed)

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

    def _prepare_dataframe(self, df, table_name):
        """
        Prepare a DataFrame for Snowflake ingestion.

        - Resets index to make DatetimeIndex a column
        - Converts column names to uppercase
        - Handles problematic dtypes
        """
        write_df = df.copy()

        # If index is a DatetimeIndex, reset it to a column named DATE
        if isinstance(write_df.index, pd.DatetimeIndex):
            write_df.index.name = write_df.index.name or 'DATE'
            write_df = write_df.reset_index()
        elif write_df.index.name and write_df.index.name != 'index':
            write_df = write_df.reset_index()

        # Uppercase column names (Snowflake convention)
        write_df.columns = [str(c).upper().replace(' ', '_').replace('-', '_') for c in write_df.columns]

        # Handle duplicate column names
        seen = {}
        new_cols = []
        for col in write_df.columns:
            if col in seen:
                seen[col] += 1
                new_cols.append(f"{col}_{seen[col]}")
            else:
                seen[col] = 0
                new_cols.append(col)
        write_df.columns = new_cols

        # Convert timezone-aware datetimes to naive UTC
        for col in write_df.columns:
            if pd.api.types.is_datetime64_any_dtype(write_df[col]):
                if write_df[col].dt.tz is not None:
                    write_df[col] = write_df[col].dt.tz_convert('UTC').dt.tz_localize(None)

        # Convert PeriodIndex columns (from Fama-French) to timestamps
        for col in write_df.columns:
            if hasattr(write_df[col], 'dtype') and str(write_df[col].dtype).startswith('period'):
                write_df[col] = write_df[col].dt.to_timestamp()

        return write_df

    def _drop_table(self, table_name):
        """Drop a table if it exists."""
        cur = self.conn.cursor()
        try:
            cur.execute(f"DROP TABLE IF EXISTS {self.schema}.{table_name}")
        finally:
            cur.close()

    def _log_load(self, table_name, rows, status, error_msg, duration):
        """Log the load operation to ETL_LOAD_LOG in PUBLIC schema."""
        cur = self.conn.cursor()
        try:
            cur.execute(
                """
                INSERT INTO PUBLIC.ETL_LOAD_LOG
                    (LOAD_TIMESTAMP, TABLE_NAME, ROWS_LOADED, STATUS, ERROR_MESSAGE, DURATION_SECONDS)
                VALUES
                    (CURRENT_TIMESTAMP(), %s, %s, %s, %s, %s)
                """,
                (
                    f"{self.schema}.{table_name}",
                    rows,
                    status,
                    error_msg,
                    round(duration, 2),
                ),
            )
        except Exception as e:
            logger.warning("Failed to log ETL load for %s: %s", table_name, e)
        finally:
            cur.close()

    def _update_data_dictionary(self, results):
        """Update the DATA_DICTIONARY table with column metadata."""
        cur = self.conn.cursor()
        try:
            for table_name, result in results.items():
                if result['status'] != 'SUCCESS':
                    continue

                etl_key = result.get('etl_key', '')
                mapping = TABLE_MAP.get(etl_key, {})
                source_desc = mapping.get('description', '')

                # Get column info from Snowflake
                try:
                    cur.execute(
                        f"DESCRIBE TABLE {self.schema}.{table_name}"
                    )
                    columns = cur.fetchall()

                    for col_row in columns:
                        col_name = col_row[0]
                        col_type = col_row[1]

                        cur.execute(
                            """
                            MERGE INTO PUBLIC.DATA_DICTIONARY AS tgt
                            USING (SELECT %s AS TABLE_NAME, %s AS COLUMN_NAME) AS src
                            ON tgt.TABLE_NAME = src.TABLE_NAME
                               AND tgt.COLUMN_NAME = src.COLUMN_NAME
                            WHEN MATCHED THEN UPDATE SET
                                DATA_TYPE = %s,
                                SOURCE = %s,
                                LAST_UPDATED = CURRENT_TIMESTAMP()
                            WHEN NOT MATCHED THEN INSERT
                                (TABLE_NAME, COLUMN_NAME, DATA_TYPE, DESCRIPTION, SOURCE, LAST_UPDATED)
                            VALUES (%s, %s, %s, %s, %s, CURRENT_TIMESTAMP())
                            """,
                            (
                                f"{self.schema}.{table_name}",
                                col_name,
                                col_type,
                                source_desc,
                                f"{self.schema}.{table_name}",
                                col_name,
                                col_type,
                                '',
                                source_desc,
                            ),
                        )
                except Exception as e:
                    logger.warning(
                        "Failed to update data dictionary for %s: %s",
                        table_name, e,
                    )
        finally:
            cur.close()

    def _print_summary(self, results, total_elapsed):
        """Print load summary."""
        succeeded = {k: v for k, v in results.items() if v['status'] == 'SUCCESS'}
        failed = {k: v for k, v in results.items() if v['status'] != 'SUCCESS'}
        total_rows = sum(v['rows'] for v in succeeded.values())

        logger.info("")
        logger.info("=" * 60)
        logger.info("  Snowflake Load Summary")
        logger.info("=" * 60)
        logger.info("  Database: %s", SNOWFLAKE_DATABASE)
        logger.info("  Schema:   %s", self.schema)
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

    # -----------------------------------------------------------------
    # Public write interface
    # -----------------------------------------------------------------
    def write_table(self, df, table_name, replace=True):
        """
        Write a DataFrame to a Snowflake table.

        Parameters
        ----------
        df : pd.DataFrame
            Data to write.
        table_name : str
            Target table name (will be uppercased).
        replace : bool
            If True, drop and recreate. If False, append.

        Returns
        -------
        dict with keys: rows, status, duration, error.
        """
        if not self._writable:
            raise RuntimeError(
                "Snowflake account is suspended. Cannot write tables."
            )
        table_name = table_name.upper()
        return self._write_table(df, table_name, replace, etl_key='model_results', verbose=True)

    # -----------------------------------------------------------------
    # Query helpers
    # -----------------------------------------------------------------
    def list_tables(self):
        """List all tables in the signals & systems schema."""
        cur = self.conn.cursor()
        try:
            cur.execute(f"SHOW TABLES IN SCHEMA {self.schema}")
            rows = cur.fetchall()
            tables = []
            for row in rows:
                tables.append({
                    'name': row[1],
                    'rows': row[5] if len(row) > 5 else None,
                })
            return tables
        finally:
            cur.close()

    def read_table(self, table_name, limit=None):
        """Read a table into a pandas DataFrame."""
        query = f"SELECT * FROM {self.schema}.{table_name}"
        if limit:
            query += f" LIMIT {limit}"
        return pd.read_sql(query, self.conn)

    def get_table_info(self, table_name):
        """Get column info for a table."""
        cur = self.conn.cursor()
        try:
            cur.execute(f"DESCRIBE TABLE {self.schema}.{table_name}")
            return [
                {'name': r[0], 'type': r[1], 'nullable': r[3]}
                for r in cur.fetchall()
            ]
        finally:
            cur.close()

    def run_query(self, sql):
        """Run an arbitrary SQL query and return results as a DataFrame."""
        return pd.read_sql(sql, self.conn)


# =============================================================================
# CONVENIENCE FUNCTIONS
# =============================================================================
def load_to_snowflake(
    categories=None,
    keys=None,
    force_refresh=False,
    replace=True,
    clean=True,
    schema=None,
):
    """
    Run the full pipeline: ETL extract -> clean -> load into Snowflake.

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
    schema : str, optional
        Override target Snowflake schema (default: SIGNALS_SYSTEMS).

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

    logger.info("Step 2/2: Loading into Snowflake...")
    with SnowflakeLoader(schema=schema) as loader:
        results = loader.load_etl(data, replace=replace)

    return results


def list_tables(schema=None):
    """List all tables in the Snowflake schema."""
    with SnowflakeLoader(schema=schema) as loader:
        tables = loader.list_tables()
        print(f"\nTables in {loader.schema}:")
        print("-" * 50)
        for t in tables:
            print(f"  {t['name']}")
        return tables


# =============================================================================
# SQLITE LOADER
# =============================================================================
SQLITE_DEFAULT_PATH = os.path.join(
    os.path.dirname(os.path.abspath(__file__)), 'data', 'signals_systems.db'
)


class SQLiteLoader:
    """Loads ETL pipeline data into a local SQLite database."""

    def __init__(self, db_path=None):
        self.db_path = db_path or SQLITE_DEFAULT_PATH
        self.conn = None

    def connect(self):
        """Open (or create) the SQLite database."""
        os.makedirs(os.path.dirname(self.db_path), exist_ok=True)
        self.conn = sqlite3.connect(self.db_path)
        # Enable WAL mode for better concurrent read performance
        self.conn.execute("PRAGMA journal_mode=WAL")
        logger.info("Connected to SQLite: %s", self.db_path)
        return self

    def close(self):
        """Close the SQLite connection."""
        if self.conn:
            self.conn.close()
            self.conn = None
            logger.info("SQLite connection closed")

    def __enter__(self):
        self.connect()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close()

    def load_etl(self, data_dict, replace=True, verbose=True):
        """
        Load all datasets from an ETL data dictionary into SQLite.

        Parameters
        ----------
        data_dict : dict
            Output from ETL.run_etl() or clean.load_data().
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
            logger.info("=" * 60)
            logger.info("  Loading ETL data into SQLite")
            logger.info("  Database: %s", self.db_path)
            logger.info("  Mode: %s", "REPLACE" if replace else "APPEND")
            logger.info("=" * 60)

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
                df = self._extract_dataframe(etl_key, data, extract_mode)
                if df is not None and not df.empty:
                    table_name = mapping['table']
                    result = self._write_table(
                        df, table_name, replace, etl_key, verbose
                    )
                    results[table_name] = result
                elif verbose:
                    logger.info("  [SKIP] %-30s (extracted empty)", etl_key)

        total_elapsed = time.time() - total_start

        # Log load metadata
        self._create_load_log()
        for table_name, result in results.items():
            self._log_load(
                table_name, result['rows'], result['status'],
                result.get('error'), result['duration'],
            )

        if verbose:
            self._print_summary(results, total_elapsed)

        return results

    def _extract_dataframe(self, etl_key, data, extract_mode):
        """Extract a DataFrame from the ETL data entry (same logic as Snowflake)."""
        if extract_mode == 'dataframe':
            return data if isinstance(data, pd.DataFrame) else None

        if extract_mode == 'combined':
            if isinstance(data, dict):
                combined = data.get('combined')
                if isinstance(combined, pd.DataFrame):
                    return combined
                for v in data.values():
                    if isinstance(v, pd.DataFrame):
                        return v
            return data if isinstance(data, pd.DataFrame) else None

        if extract_mode == 'concat':
            if isinstance(data, dict):
                frames = [v for v in data.values()
                          if isinstance(v, pd.DataFrame) and not v.empty]
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

    def _write_table(self, df, table_name, replace, etl_key, verbose):
        """Write a DataFrame to a SQLite table."""
        t0 = time.time()
        status = 'SUCCESS'
        error_msg = None
        rows = len(df)

        try:
            write_df = self._prepare_dataframe(df)
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

    def _prepare_dataframe(self, df):
        """Prepare a DataFrame for SQLite ingestion."""
        write_df = df.copy()

        # Reset DatetimeIndex to a DATE column
        if isinstance(write_df.index, pd.DatetimeIndex):
            write_df.index.name = write_df.index.name or 'DATE'
            write_df = write_df.reset_index()
        elif write_df.index.name and write_df.index.name != 'index':
            write_df = write_df.reset_index()

        # Uppercase column names for consistency with Snowflake
        write_df.columns = [
            str(c).upper().replace(' ', '_').replace('-', '_')
            for c in write_df.columns
        ]

        # Handle duplicate column names
        seen = {}
        new_cols = []
        for col in write_df.columns:
            if col in seen:
                seen[col] += 1
                new_cols.append(f"{col}_{seen[col]}")
            else:
                seen[col] = 0
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

    def _print_summary(self, results, total_elapsed):
        """Print load summary."""
        succeeded = {k: v for k, v in results.items() if v['status'] == 'SUCCESS'}
        failed = {k: v for k, v in results.items() if v['status'] != 'SUCCESS'}
        total_rows = sum(v['rows'] for v in succeeded.values())

        # Get file size
        db_size_mb = os.path.getsize(self.db_path) / 1024 / 1024

        logger.info("")
        logger.info("=" * 60)
        logger.info("  SQLite Load Summary")
        logger.info("=" * 60)
        logger.info("  Database:       %s", self.db_path)
        logger.info("  Database size:  %.1f MB", db_size_mb)
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

    # -----------------------------------------------------------------
    # Public write interface
    # -----------------------------------------------------------------
    def write_table(self, df, table_name, replace=True):
        """
        Write a DataFrame to a SQLite table.

        Parameters
        ----------
        df : pd.DataFrame
            Data to write.
        table_name : str
            Target table name (will be uppercased).
        replace : bool
            If True, drop and recreate. If False, append.

        Returns
        -------
        dict with keys: rows, status, duration, error.
        """
        table_name = table_name.upper()
        return self._write_table(df, table_name, replace, etl_key='model_results', verbose=True)

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
# SQLITE CONVENIENCE FUNCTIONS
# =============================================================================
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
    Auto-select backend: try Snowflake first, fall back to SQLite.

    Returns
    -------
    tuple(str, dict)
        ('snowflake' or 'sqlite', load results)
    """
    from ETL import run_etl

    logger.info("Step 1/2: Running ETL pipeline...")
    data = run_etl(
        categories=categories,
        keys=keys,
        force_refresh=force_refresh,
        clean=clean,
    )

    # Try Snowflake first
    logger.info("Step 2/2: Attempting Snowflake load...")
    try:
        with SnowflakeLoader() as loader:
            if loader._writable:
                results = loader.load_etl(data, replace=replace)
                return 'snowflake', results
            else:
                logger.warning("Snowflake not writable. Falling back to SQLite.")
    except Exception as e:
        logger.warning("Snowflake connection failed: %s. Falling back to SQLite.", e)

    # Fall back to SQLite
    logger.info("Loading into SQLite instead...")
    with SQLiteLoader() as loader:
        results = loader.load_etl(data, replace=replace)

    return 'sqlite', results


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
