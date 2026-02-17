"""FRED macroeconomic data loaders: inflation, rates, IP, M2, GDP, employment."""

import os
import shutil
from datetime import datetime

import pandas as pd

from .config import _download_fred_series, _validate_fred_api_key, logger, API_KEY
from .cache import _save_cache, _load_cache

def load_inflation_data(
    start_date='2000-01-01',
    end_date=None,
    cache_path='./data/fred'
):
    """
    Load inflation data from FRED (Federal Reserve Economic Data).

    Provides multiple inflation measures:
    - CPI: Consumer Price Index (All Urban Consumers)
    - Core CPI: CPI excluding food and energy
    - PCE: Personal Consumption Expenditures Price Index
    - Core PCE: PCE excluding food and energy (Fed's preferred measure)
    - PPI: Producer Price Index

    Parameters:
    -----------
    start_date : str
        Start date in 'YYYY-MM-DD' format
    end_date : str
        End date in 'YYYY-MM-DD' format (defaults to today)
    cache_path : str
        Directory to cache downloaded data

    Returns:
    --------
    dict : Dictionary containing:
        - 'raw': Raw index values
        - 'yoy': Year-over-year percent changes
        - 'mom': Month-over-month percent changes
        - 'combined': All measures in one DataFrame
    """
    if end_date is None:
        end_date = datetime.today().strftime('%Y-%m-%d')

    os.makedirs(cache_path, exist_ok=True)
    cache_dir = os.path.join(cache_path, f'inflation_data_{start_date}_{end_date}')

    if os.path.exists(cache_dir):
        logger.info("Loading cached inflation data from %s", cache_dir)
        return _load_cache(cache_dir)

    logger.info("Downloading inflation data from FRED...")

    series = {
        'CPI': 'CPIAUCSL',
        'Core_CPI': 'CPILFESL',
        'PCE': 'PCEPI',
        'Core_PCE': 'PCEPILFE',
        'PPI': 'PPIACO',
        'GDP_Deflator': 'GDPDEF',
    }

    try:
        inflation_raw = _download_fred_series(series, start_date, end_date)

        logger.info("Calculating year-over-year changes...")
        inflation_yoy = inflation_raw.pct_change(periods=12) * 100
        inflation_yoy.columns = [f'{col}_YoY' for col in inflation_yoy.columns]

        logger.info("Calculating month-over-month changes...")
        inflation_mom = inflation_raw.pct_change() * 100 * 12
        inflation_mom.columns = [f'{col}_MoM' for col in inflation_mom.columns]

        inflation_combined = pd.concat([
            inflation_raw,
            inflation_yoy,
            inflation_mom
        ], axis=1)

        result = {
            'raw': inflation_raw,
            'yoy': inflation_yoy,
            'mom': inflation_mom,
            'combined': inflation_combined
        }

        _save_cache(result, cache_dir)
        logger.info("Cached inflation data to %s", cache_dir)

        logger.info("=== Inflation Data Summary ===")
        logger.info("Date range: %s to %s", inflation_raw.index.min(), inflation_raw.index.max())
        logger.info("Observations: %d", len(inflation_raw))
        logger.info("Latest values (Year-over-Year %%):\n%s", inflation_yoy.iloc[-1])

        return result

    except Exception as e:
        logger.error("downloading inflation data: %s", e)
        return None


def load_inflation_expectations_data(
    start_date='2000-01-01',
    end_date=None,
    cache_path='./data/fred'
):
    """
    Load inflation expectations and breakeven inflation data from FRED.

    Provides forward-looking inflation measures:
    - Breakeven Inflation: Market-implied inflation from TIPS spreads
    - University of Michigan Inflation Expectations
    - Cleveland Fed Inflation Expectations
    - NY Fed Survey of Consumer Expectations

    Parameters:
    -----------
    start_date : str
        Start date in 'YYYY-MM-DD' format
    end_date : str
        End date in 'YYYY-MM-DD' format (defaults to today)
    cache_path : str
        Directory to cache downloaded data

    Returns:
    --------
    dict : Dictionary containing:
        - 'breakeven': Breakeven inflation rates (market-based)
        - 'survey': Survey-based inflation expectations
        - 'fed_measures': Federal Reserve inflation measures
        - 'combined': All expectations in one DataFrame
    """
    if end_date is None:
        end_date = datetime.today().strftime('%Y-%m-%d')

    os.makedirs(cache_path, exist_ok=True)
    cache_dir = os.path.join(
        cache_path,
        f'inflation_expectations_{start_date}_{end_date}'
    )

    if os.path.exists(cache_dir):
        logger.info("Loading cached inflation expectations from %s", cache_dir)
        return _load_cache(cache_dir)

    logger.info("Downloading inflation expectations data from FRED...")

    # Breakeven inflation rates (TIPS spreads)
    breakeven_series = {
        'Breakeven_5Y': 'T5YIE',       # 5-Year Breakeven Inflation Rate
        'Breakeven_10Y': 'T10YIE',     # 10-Year Breakeven Inflation Rate
        'Breakeven_5Y5Y': 'T5YIFR',    # 5-Year, 5-Year Forward Inflation Rate
    }

    # Survey-based expectations
    survey_series = {
        'UMich_Inflation_1Y': 'MICH',           # U of Michigan 1-Year Inflation Expectations
        'UMich_Inflation_5Y': 'UMCSENT5',       # U of Michigan 5-Year Inflation Expectations (if available)
    }

    # Federal Reserve measures
    fed_series = {
        'Trimmed_Mean_PCE': 'PCETRIM12M159SFRBDAL',  # Dallas Fed Trimmed Mean PCE
        'Sticky_Price_CPI': 'CORESTICKM159SFRBATL',  # Atlanta Fed Sticky Price CPI
        'Flexible_Price_CPI': 'FLEXCPIM159SFRBATL',  # Atlanta Fed Flexible Price CPI
        'Median_CPI': 'MEDCPIM158SFRBCLE',           # Cleveland Fed Median CPI
        'CPI_Trimmed_Mean_16': 'TRMMEANCPIM158SFRBCLE',  # Cleveland Fed 16% Trimmed Mean CPI
    }

    try:
        # Download breakeven inflation
        logger.info("--- Breakeven Inflation (Market-Based) ---")
        breakeven_df = _download_fred_series(breakeven_series, start_date, end_date)

        # Download survey expectations
        logger.info("--- Survey-Based Expectations ---")
        survey_df = _download_fred_series(survey_series, start_date, end_date)

        # Download Fed measures
        logger.info("--- Federal Reserve Inflation Measures ---")
        fed_df = _download_fred_series(fed_series, start_date, end_date)

        # Combine all data
        combined_df = pd.concat([breakeven_df, survey_df, fed_df], axis=1)

        result = {
            'breakeven': breakeven_df,
            'survey': survey_df,
            'fed_measures': fed_df,
            'combined': combined_df
        }

        _save_cache(result, cache_dir)
        logger.info("Cached inflation expectations to %s", cache_dir)

        # Print summary
        logger.info("=" * 60)
        logger.info("=== Inflation Expectations Summary ===")
        logger.info("=" * 60)

        if len(breakeven_df) > 0:
            logger.info("Breakeven Inflation (Market-Based):")
            logger.info("  Date range: %s to %s", breakeven_df.index.min(), breakeven_df.index.max())
            logger.info("  Series: %s", list(breakeven_df.columns))
            logger.info("  Latest values:")
            for col in breakeven_df.columns:
                latest = breakeven_df[col].dropna().iloc[-1] if len(breakeven_df[col].dropna()) > 0 else 'N/A'
                if isinstance(latest, float):
                    logger.info("    %s: %.2f%%", col, latest)
                else:
                    logger.info("    %s: %s", col, latest)

        if len(survey_df) > 0:
            logger.info("Survey-Based Expectations:")
            logger.info("  Date range: %s to %s", survey_df.index.min(), survey_df.index.max())
            logger.info("  Series: %s", list(survey_df.columns))
            logger.info("  Latest values:")
            for col in survey_df.columns:
                latest = survey_df[col].dropna().iloc[-1] if len(survey_df[col].dropna()) > 0 else 'N/A'
                if isinstance(latest, float):
                    logger.info("    %s: %.2f%%", col, latest)
                else:
                    logger.info("    %s: %s", col, latest)

        if len(fed_df) > 0:
            logger.info("Federal Reserve Measures:")
            logger.info("  Date range: %s to %s", fed_df.index.min(), fed_df.index.max())
            logger.info("  Series: %s", list(fed_df.columns))
            logger.info("  Latest values:")
            for col in fed_df.columns:
                latest = fed_df[col].dropna().iloc[-1] if len(fed_df[col].dropna()) > 0 else 'N/A'
                if isinstance(latest, float):
                    logger.info("    %s: %.2f%%", col, latest)
                else:
                    logger.info("    %s: %s", col, latest)

        logger.info("=" * 60)
        logger.info("Citation:")
        logger.info("Federal Reserve Economic Data (FRED), Federal Reserve Bank of St. Louis")
        logger.info("https://fred.stlouisfed.org/")
        logger.info("=" * 60)

        return result

    except Exception as e:
        logger.error("downloading inflation expectations data: %s", e)
        return None


def load_comprehensive_inflation_data(
    start_date='2000-01-01',
    end_date=None,
    cache_path='./data/fred',
    force_refresh=False
):
    """
    Load comprehensive inflation dataset combining all inflation measures.

    This function aggregates:
    - Core inflation measures (CPI, PCE, PPI)
    - Breakeven inflation rates (TIPS-based)
    - Survey-based inflation expectations
    - Federal Reserve alternative measures
    - Component-level inflation data

    Parameters:
    -----------
    start_date : str
        Start date in 'YYYY-MM-DD' format
    end_date : str
        End date in 'YYYY-MM-DD' format (defaults to today)
    cache_path : str
        Directory to cache downloaded data

    Returns:
    --------
    dict : Dictionary containing:
        - 'core': Core inflation measures (CPI, PCE, PPI)
        - 'expectations': Breakeven and survey expectations
        - 'components': Component-level inflation
        - 'combined': All measures merged on date
        - 'summary_stats': Summary statistics for all series
    """
    if end_date is None:
        end_date = datetime.today().strftime('%Y-%m-%d')

    os.makedirs(cache_path, exist_ok=True)
    cache_dir = os.path.join(
        cache_path,
        f'comprehensive_inflation_{start_date}_{end_date}'
    )

    if force_refresh and os.path.exists(cache_dir):
        shutil.rmtree(cache_dir)
        logger.info("Force refresh: removed %s", cache_dir)

    if os.path.exists(cache_dir):
        logger.info("Loading cached comprehensive inflation data from %s", cache_dir)
        return _load_cache(cache_dir)
    logger.info("=" * 60)
    logger.info("Loading Comprehensive Inflation Data (2000-2025)")
    logger.info("=" * 60)

    # Load core inflation data
    logger.info("[1/3] Loading core inflation measures...")
    core_data = load_inflation_data(start_date, end_date, cache_path)

    # Load expectations data
    logger.info("[2/3] Loading inflation expectations...")
    expectations_data = load_inflation_expectations_data(start_date, end_date, cache_path)

    # Load component-level inflation
    logger.info("[3/3] Loading component-level inflation...")
    component_series = {
        'CPI_Food': 'CPIUFDSL',              # CPI Food
        'CPI_Energy': 'CPIENGSL',            # CPI Energy
        'CPI_Shelter': 'CUSR0000SAH1',       # CPI Shelter
        'CPI_Medical': 'CPIMEDSL',           # CPI Medical Care
        'CPI_Transportation': 'CPITRNSL',    # CPI Transportation
        'CPI_Apparel': 'CPIAPPSL',           # CPI Apparel
        'CPI_Education': 'CUSR0000SAE1',     # CPI Education
        'CPI_Services': 'CUSR0000SAS',       # CPI Services
        'CPI_Commodities': 'CUSR0000SAC',    # CPI Commodities less food & energy
        'Import_Prices': 'IR',               # Import Price Index
        'Export_Prices': 'IQ',               # Export Price Index
    }

    components_df = _download_fred_series(component_series, start_date, end_date)

    # Calculate YoY changes for components
    components_yoy = components_df.pct_change(periods=12) * 100
    components_yoy.columns = [f'{col}_YoY' for col in components_yoy.columns]

    # Combine all data
    combined_dfs = []

    if core_data and 'combined' in core_data:
        combined_dfs.append(core_data['combined'])

    if expectations_data and 'combined' in expectations_data:
        combined_dfs.append(expectations_data['combined'])

    if len(components_yoy) > 0:
        combined_dfs.append(components_yoy)

    if combined_dfs:
        combined_df = pd.concat(combined_dfs, axis=1)
        # Remove duplicate columns if any
        combined_df = combined_df.loc[:, ~combined_df.columns.duplicated()]
    else:
        combined_df = pd.DataFrame()

    # Calculate summary statistics
    summary_stats = {}
    if len(combined_df) > 0:
        for col in combined_df.columns:
            series = combined_df[col].dropna()
            if len(series) > 0:
                summary_stats[col] = {
                    'count': len(series),
                    'mean': series.mean(),
                    'std': series.std(),
                    'min': series.min(),
                    'max': series.max(),
                    'latest': series.iloc[-1],
                    'start_date': series.index.min(),
                    'end_date': series.index.max()
                }

    result = {
        'core': core_data,
        'expectations': expectations_data,
        'components': {
            'raw': components_df,
            'yoy': components_yoy
        },
        'combined': combined_df,
        'summary_stats': summary_stats
    }

    _save_cache(result, cache_dir)
    logger.info("Cached comprehensive inflation data to %s", cache_dir)

    # Print final summary
    logger.info("=" * 60)
    logger.info("=== Comprehensive Inflation Data Summary ===")
    logger.info("=" * 60)
    logger.info("Total series loaded: %d", len(combined_df.columns))
    logger.info("Date range: %s to %s", combined_df.index.min(), combined_df.index.max())
    logger.info("Total observations: %d", len(combined_df))

    logger.info("--- Series Categories ---")
    logger.info("Core inflation measures: %d", len(core_data['combined'].columns) if core_data else 0)
    logger.info("Expectations measures: %d", len(expectations_data['combined'].columns) if expectations_data else 0)
    logger.info("Component measures: %d", len(components_yoy.columns))

    return result


# =============================================================================
# RATES DATA
# =============================================================================
def load_treasury_yields(
    start_date='2000-01-01',
    end_date=None,
    cache_path='./data/fred'
):
    """
    Load Treasury yield curve data from FRED (2000-2025).

    Provides the complete Treasury yield curve:
    - Short-term: 1M, 3M, 6M
    - Medium-term: 1Y, 2Y, 3Y, 5Y, 7Y
    - Long-term: 10Y, 20Y, 30Y
    - Inflation-indexed: 5Y TIPS, 10Y TIPS, 20Y TIPS, 30Y TIPS

    Parameters:
    -----------
    start_date : str
        Start date in 'YYYY-MM-DD' format
    end_date : str
        End date in 'YYYY-MM-DD' format (defaults to today)
    cache_path : str
        Directory to cache downloaded data

    Returns:
    --------
    dict : Dictionary containing:
        - 'nominal': Nominal Treasury yields
        - 'real': TIPS (real) yields
        - 'combined': All yields in one DataFrame
    """
    if end_date is None:
        end_date = datetime.today().strftime('%Y-%m-%d')

    os.makedirs(cache_path, exist_ok=True)
    cache_dir = os.path.join(cache_path, f'treasury_yields_{start_date}_{end_date}')

    if os.path.exists(cache_dir):
        logger.info("Loading cached Treasury yields from %s", cache_dir)
        return _load_cache(cache_dir)

    logger.info("Downloading Treasury yield data from FRED...")

    # Nominal Treasury yields
    nominal_series = {
        'Treasury_1M': 'DGS1MO',    # 1-Month Treasury
        'Treasury_3M': 'DGS3MO',    # 3-Month Treasury
        'Treasury_6M': 'DGS6MO',    # 6-Month Treasury
        'Treasury_1Y': 'DGS1',      # 1-Year Treasury
        'Treasury_2Y': 'DGS2',      # 2-Year Treasury
        'Treasury_3Y': 'DGS3',      # 3-Year Treasury
        'Treasury_5Y': 'DGS5',      # 5-Year Treasury
        'Treasury_7Y': 'DGS7',      # 7-Year Treasury
        'Treasury_10Y': 'DGS10',    # 10-Year Treasury
        'Treasury_20Y': 'DGS20',    # 20-Year Treasury
        'Treasury_30Y': 'DGS30',    # 30-Year Treasury
    }

    # TIPS (Treasury Inflation-Protected Securities)
    tips_series = {
        'TIPS_5Y': 'DFII5',         # 5-Year TIPS
        'TIPS_10Y': 'DFII10',       # 10-Year TIPS
        'TIPS_20Y': 'DFII20',       # 20-Year TIPS
        'TIPS_30Y': 'DFII30',       # 30-Year TIPS
    }

    try:
        # Download nominal yields
        logger.info("--- Nominal Treasury Yields ---")
        nominal_df = _download_fred_series(nominal_series, start_date, end_date)

        # Download TIPS yields
        logger.info("--- TIPS (Real) Yields ---")
        tips_df = _download_fred_series(tips_series, start_date, end_date)

        # Combine all yields
        combined_df = pd.concat([nominal_df, tips_df], axis=1)

        result = {
            'nominal': nominal_df,
            'real': tips_df,
            'combined': combined_df
        }

        _save_cache(result, cache_dir)
        logger.info("Cached Treasury yields to %s", cache_dir)

        # Print summary
        logger.info("=" * 60)
        logger.info("=== Treasury Yields Summary ===")
        logger.info("=" * 60)
        logger.info("Date range: %s to %s", combined_df.index.min(), combined_df.index.max())
        logger.info("Observations: %d", len(combined_df))
        logger.info("Nominal series: %d", len(nominal_df.columns))
        logger.info("TIPS series: %d", len(tips_df.columns))

        logger.info("Latest yields (%%):")
        for col in combined_df.columns:
            latest = combined_df[col].dropna().iloc[-1] if len(combined_df[col].dropna()) > 0 else 'N/A'
            if isinstance(latest, float):
                logger.info("  %s: %.2f%%", col, latest)
            else:
                logger.info("  %s: %s", col, latest)

        return result

    except Exception as e:
        logger.error("downloading Treasury yields: %s", e)
        return None


def load_policy_rates(
    start_date='2000-01-01',
    end_date=None,
    cache_path='./data/fred'
):
    """
    Load Federal Reserve policy rates and money market rates from FRED.

    Includes:
    - Federal Funds Rate (effective and target)
    - Discount Rate
    - SOFR (Secured Overnight Financing Rate)
    - Prime Rate
    - Reserve Balances

    Parameters:
    -----------
    start_date : str
        Start date in 'YYYY-MM-DD' format
    end_date : str
        End date in 'YYYY-MM-DD' format (defaults to today)
    cache_path : str
        Directory to cache downloaded data

    Returns:
    --------
    dict : Dictionary containing:
        - 'fed_funds': Federal Funds rates
        - 'money_market': Money market rates
        - 'combined': All policy rates
    """
    if end_date is None:
        end_date = datetime.today().strftime('%Y-%m-%d')

    os.makedirs(cache_path, exist_ok=True)
    cache_dir = os.path.join(cache_path, f'policy_rates_{start_date}_{end_date}')

    if os.path.exists(cache_dir):
        logger.info("Loading cached policy rates from %s", cache_dir)
        return _load_cache(cache_dir)

    logger.info("Downloading policy rates from FRED...")

    # Federal Funds and discount rates
    fed_series = {
        'Fed_Funds_Effective': 'DFF',           # Daily Effective Federal Funds Rate
        'Fed_Funds_Target_Upper': 'DFEDTARU',   # Fed Funds Target Range Upper
        'Fed_Funds_Target_Lower': 'DFEDTARL',   # Fed Funds Target Range Lower
        'Discount_Rate': 'INTDSRUSM193N',       # Discount Rate
    }

    # Money market rates
    money_market_series = {
        'SOFR': 'SOFR',                         # Secured Overnight Financing Rate
        'Prime_Rate': 'DPRIME',                 # Bank Prime Loan Rate
        'Overnight_Bank_Funding': 'OBFR',       # Overnight Bank Funding Rate
        'EFFR': 'EFFR',                         # Effective Federal Funds Rate (daily)
    }

    try:
        # Download Fed Funds rates
        logger.info("--- Federal Funds & Discount Rates ---")
        fed_df = _download_fred_series(fed_series, start_date, end_date)

        # Download money market rates
        logger.info("--- Money Market Rates ---")
        mm_df = _download_fred_series(money_market_series, start_date, end_date)

        # Combine all rates
        combined_df = pd.concat([fed_df, mm_df], axis=1)

        result = {
            'fed_funds': fed_df,
            'money_market': mm_df,
            'combined': combined_df
        }

        _save_cache(result, cache_dir)
        logger.info("Cached policy rates to %s", cache_dir)

        # Print summary
        logger.info("=" * 60)
        logger.info("=== Policy Rates Summary ===")
        logger.info("=" * 60)
        logger.info("Date range: %s to %s", combined_df.index.min(), combined_df.index.max())
        logger.info("Observations: %d", len(combined_df))

        logger.info("Latest rates (%%):")
        for col in combined_df.columns:
            latest = combined_df[col].dropna().iloc[-1] if len(combined_df[col].dropna()) > 0 else 'N/A'
            if isinstance(latest, float):
                logger.info("  %s: %.2f%%", col, latest)
            else:
                logger.info("  %s: %s", col, latest)

        return result

    except Exception as e:
        logger.error("downloading policy rates: %s", e)
        return None


def load_credit_spreads(
    start_date='2000-01-01',
    end_date=None,
    cache_path='./data/fred'
):
    """
    Load credit spreads and corporate bond yields from FRED.

    Includes:
    - Investment Grade: AAA, AA, A, BBB corporate yields
    - High Yield: BB, B, CCC corporate yields
    - Credit Spreads: Investment grade and high yield spreads
    - Mortgage rates: 30Y and 15Y fixed

    Parameters:
    -----------
    start_date : str
        Start date in 'YYYY-MM-DD' format
    end_date : str
        End date in 'YYYY-MM-DD' format (defaults to today)
    cache_path : str
        Directory to cache downloaded data

    Returns:
    --------
    dict : Dictionary containing:
        - 'corporate': Corporate bond yields
        - 'spreads': Credit spreads
        - 'mortgage': Mortgage rates
        - 'combined': All credit data
    """
    if end_date is None:
        end_date = datetime.today().strftime('%Y-%m-%d')

    os.makedirs(cache_path, exist_ok=True)
    cache_dir = os.path.join(cache_path, f'credit_spreads_{start_date}_{end_date}')

    if os.path.exists(cache_dir):
        logger.info("Loading cached credit spreads from %s", cache_dir)
        return _load_cache(cache_dir)

    logger.info("Downloading credit spreads and corporate yields from FRED...")

    # Corporate bond yields
    corporate_series = {
        'Moodys_AAA': 'AAA',                    # Moody's AAA Corporate Bond Yield
        'Moodys_BAA': 'BAA',                    # Moody's BAA Corporate Bond Yield
        'ICE_BofA_AAA': 'BAMLC0A1CAAAEY',       # ICE BofA AAA Corporate Index Yield
        'ICE_BofA_AA': 'BAMLC0A2CAAEY',         # ICE BofA AA Corporate Index Yield
        'ICE_BofA_A': 'BAMLC0A3CAEY',           # ICE BofA A Corporate Index Yield
        'ICE_BofA_BBB': 'BAMLC0A4CBBBEY',       # ICE BofA BBB Corporate Index Yield
        'ICE_BofA_HighYield': 'BAMLH0A0HYM2EY', # ICE BofA High Yield Index Yield
    }

    # Credit spreads
    spread_series = {
        'BAA_10Y_Spread': 'BAA10Y',             # BAA - 10Y Treasury Spread
        'AAA_10Y_Spread': 'AAA10Y',             # AAA - 10Y Treasury Spread
        'IG_Spread': 'BAMLC0A0CM',              # Investment Grade Corporate Spread
        'HY_Spread': 'BAMLH0A0HYM2',            # High Yield Corporate Spread
        'TED_Spread': 'TEDRATE',                # TED Spread (3M LIBOR - 3M T-Bill)
    }

    # Mortgage rates
    mortgage_series = {
        'Mortgage_30Y': 'MORTGAGE30US',         # 30-Year Fixed Mortgage Rate
        'Mortgage_15Y': 'MORTGAGE15US',         # 15-Year Fixed Mortgage Rate
        'Mortgage_5Y_ARM': 'MORTGAGE5US',       # 5/1-Year ARM Rate
    }

    try:
        # Download corporate yields
        logger.info("--- Corporate Bond Yields ---")
        corp_df = _download_fred_series(corporate_series, start_date, end_date)

        # Download credit spreads
        logger.info("--- Credit Spreads ---")
        spread_df = _download_fred_series(spread_series, start_date, end_date)

        # Download mortgage rates
        logger.info("--- Mortgage Rates ---")
        mortgage_df = _download_fred_series(mortgage_series, start_date, end_date)

        # Combine all data
        combined_df = pd.concat([corp_df, spread_df, mortgage_df], axis=1)

        result = {
            'corporate': corp_df,
            'spreads': spread_df,
            'mortgage': mortgage_df,
            'combined': combined_df
        }

        _save_cache(result, cache_dir)
        logger.info("Cached credit spreads to %s", cache_dir)

        # Print summary
        logger.info("=" * 60)
        logger.info("=== Credit Spreads Summary ===")
        logger.info("=" * 60)
        logger.info("Date range: %s to %s", combined_df.index.min(), combined_df.index.max())
        logger.info("Observations: %d", len(combined_df))
        logger.info("Corporate yields: %d", len(corp_df.columns))
        logger.info("Credit spreads: %d", len(spread_df.columns))
        logger.info("Mortgage rates: %d", len(mortgage_df.columns))

        logger.info("Latest values:")
        for col in combined_df.columns:
            latest = combined_df[col].dropna().iloc[-1] if len(combined_df[col].dropna()) > 0 else 'N/A'
            if isinstance(latest, float):
                unit = "bps" if "Spread" in col and latest > 10 else "%"
                logger.info("  %s: %.2f%s", col, latest, unit)
            else:
                logger.info("  %s: %s", col, latest)

        return result

    except Exception as e:
        logger.error("downloading credit spreads: %s", e)
        return None


def calculate_yield_curve_metrics(treasury_data):
    """
    Calculate yield curve metrics from Treasury data.

    Metrics calculated:
    - Yield curve slope (10Y - 2Y, 10Y - 3M)
    - Yield curve curvature (butterfly spread)
    - Inversion indicators
    - Term premium proxies

    Parameters:
    -----------
    treasury_data : dict
        Output from load_treasury_yields()

    Returns:
    --------
    pd.DataFrame : Yield curve metrics
    """
    if treasury_data is None or 'nominal' not in treasury_data:
        logger.info("Treasury data not available")
        return None

    nominal = treasury_data['nominal']
    metrics = pd.DataFrame(index=nominal.index)

    logger.info("Calculating yield curve metrics...")

    # Yield curve slopes
    if 'Treasury_10Y' in nominal.columns and 'Treasury_2Y' in nominal.columns:
        metrics['Slope_10Y_2Y'] = nominal['Treasury_10Y'] - nominal['Treasury_2Y']
        logger.info("  Calculated 10Y-2Y slope")

    if 'Treasury_10Y' in nominal.columns and 'Treasury_3M' in nominal.columns:
        metrics['Slope_10Y_3M'] = nominal['Treasury_10Y'] - nominal['Treasury_3M']
        logger.info("  Calculated 10Y-3M slope")

    if 'Treasury_30Y' in nominal.columns and 'Treasury_5Y' in nominal.columns:
        metrics['Slope_30Y_5Y'] = nominal['Treasury_30Y'] - nominal['Treasury_5Y']
        logger.info("  Calculated 30Y-5Y slope")

    if 'Treasury_2Y' in nominal.columns and 'Treasury_3M' in nominal.columns:
        metrics['Slope_2Y_3M'] = nominal['Treasury_2Y'] - nominal['Treasury_3M']
        logger.info("  Calculated 2Y-3M slope")

    # Yield curve curvature (butterfly spread)
    if all(col in nominal.columns for col in ['Treasury_2Y', 'Treasury_5Y', 'Treasury_10Y']):
        metrics['Curvature_2_5_10'] = (
            2 * nominal['Treasury_5Y'] -
            nominal['Treasury_2Y'] -
            nominal['Treasury_10Y']
        )
        logger.info("  Calculated 2-5-10 curvature (butterfly)")

    if all(col in nominal.columns for col in ['Treasury_3M', 'Treasury_2Y', 'Treasury_10Y']):
        metrics['Curvature_3M_2Y_10Y'] = (
            2 * nominal['Treasury_2Y'] -
            nominal['Treasury_3M'] -
            nominal['Treasury_10Y']
        )
        logger.info("  Calculated 3M-2Y-10Y curvature")

    # Inversion indicators
    if 'Slope_10Y_2Y' in metrics.columns:
        metrics['Inverted_10Y_2Y'] = (metrics['Slope_10Y_2Y'] < 0).astype(int)
        logger.info("  Calculated 10Y-2Y inversion indicator")

    if 'Slope_10Y_3M' in metrics.columns:
        metrics['Inverted_10Y_3M'] = (metrics['Slope_10Y_3M'] < 0).astype(int)
        logger.info("  Calculated 10Y-3M inversion indicator")

    # Near-term forward spread (recession predictor)
    if 'Treasury_3M' in nominal.columns and 'Treasury_1Y' in nominal.columns:
        # 18-month forward 3-month rate minus current 3-month rate (approximation)
        metrics['Near_Term_Forward_Spread'] = nominal['Treasury_1Y'] - nominal['Treasury_3M']
        logger.info("  Calculated near-term forward spread")

    # Level (average of key tenors)
    if all(col in nominal.columns for col in ['Treasury_2Y', 'Treasury_5Y', 'Treasury_10Y']):
        metrics['Curve_Level'] = (
            nominal['Treasury_2Y'] +
            nominal['Treasury_5Y'] +
            nominal['Treasury_10Y']
        ) / 3
        logger.info("  Calculated curve level")

    # Print summary
    logger.info("=" * 60)
    logger.info("=== Yield Curve Metrics Summary ===")
    logger.info("=" * 60)
    logger.info("Metrics calculated: %d", len(metrics.columns))
    logger.info("Date range: %s to %s", metrics.index.min(), metrics.index.max())

    logger.info("Latest values:")
    for col in metrics.columns:
        latest = metrics[col].dropna().iloc[-1] if len(metrics[col].dropna()) > 0 else 'N/A'
        if isinstance(latest, (int, float)):
            if 'Inverted' in col:
                logger.info("  %s: %s", col, 'Yes' if latest == 1 else 'No')
            else:
                logger.info("  %s: %.2f%%", col, latest)
        else:
            logger.info("  %s: %s", col, latest)

    # Inversion statistics
    if 'Inverted_10Y_2Y' in metrics.columns:
        inversion_pct = metrics['Inverted_10Y_2Y'].mean() * 100
        logger.info("10Y-2Y Inversion frequency: %.1f%% of observations", inversion_pct)

    if 'Inverted_10Y_3M' in metrics.columns:
        inversion_pct = metrics['Inverted_10Y_3M'].mean() * 100
        logger.info("10Y-3M Inversion frequency: %.1f%% of observations", inversion_pct)

    return metrics


def load_comprehensive_rates_data(
    start_date='2000-01-01',
    end_date=None,
    cache_path='./data/fred',
    force_refresh=False
):
    """
    Load comprehensive interest rates dataset from FRED (2000-2025).

    This function aggregates:
    - Treasury yield curve (1M to 30Y)
    - TIPS (real) yields
    - Federal Reserve policy rates
    - Money market rates (SOFR, Prime)
    - Corporate bond yields (IG and HY)
    - Credit spreads
    - Mortgage rates
    - Yield curve metrics (slope, curvature, inversion)

    Parameters:
    -----------
    start_date : str
        Start date in 'YYYY-MM-DD' format
    end_date : str
        End date in 'YYYY-MM-DD' format (defaults to today)
    cache_path : str
        Directory to cache downloaded data

    Returns:
    --------
    dict : Dictionary containing:
        - 'treasury': Treasury yields (nominal and TIPS)
        - 'policy': Fed Funds and money market rates
        - 'credit': Corporate yields, spreads, mortgages
        - 'curve_metrics': Yield curve slope, curvature, inversion
        - 'combined': All rates merged on date
        - 'summary_stats': Summary statistics for all series
    """
    if end_date is None:
        end_date = datetime.today().strftime('%Y-%m-%d')

    os.makedirs(cache_path, exist_ok=True)
    cache_dir = os.path.join(
        cache_path,
        f'comprehensive_rates_{start_date}_{end_date}'
    )

    if force_refresh and os.path.exists(cache_dir):
        shutil.rmtree(cache_dir)
        logger.info("Force refresh: removed %s", cache_dir)

    if os.path.exists(cache_dir):
        logger.info("Loading cached comprehensive rates data from %s", cache_dir)
        return _load_cache(cache_dir)
    logger.info("=" * 60)
    logger.info("Loading Comprehensive Rates Data (2000-2025)")
    logger.info("=" * 60)

    # Load Treasury yields
    logger.info("[1/4] Loading Treasury yields...")
    treasury_data = load_treasury_yields(start_date, end_date, cache_path)

    # Load policy rates
    logger.info("[2/4] Loading policy rates...")
    policy_data = load_policy_rates(start_date, end_date, cache_path)

    # Load credit spreads
    logger.info("[3/4] Loading credit spreads...")
    credit_data = load_credit_spreads(start_date, end_date, cache_path)

    # Calculate yield curve metrics
    logger.info("[4/4] Calculating yield curve metrics...")
    curve_metrics = calculate_yield_curve_metrics(treasury_data)

    # Combine all data
    combined_dfs = []

    if treasury_data and 'combined' in treasury_data:
        combined_dfs.append(treasury_data['combined'])

    if policy_data and 'combined' in policy_data:
        combined_dfs.append(policy_data['combined'])

    if credit_data and 'combined' in credit_data:
        combined_dfs.append(credit_data['combined'])

    if curve_metrics is not None and len(curve_metrics) > 0:
        combined_dfs.append(curve_metrics)

    if combined_dfs:
        combined_df = pd.concat(combined_dfs, axis=1)
        # Remove duplicate columns if any
        combined_df = combined_df.loc[:, ~combined_df.columns.duplicated()]
    else:
        combined_df = pd.DataFrame()

    # Calculate summary statistics
    summary_stats = {}
    if len(combined_df) > 0:
        for col in combined_df.columns:
            series = combined_df[col].dropna()
            if len(series) > 0:
                summary_stats[col] = {
                    'count': len(series),
                    'mean': series.mean(),
                    'std': series.std(),
                    'min': series.min(),
                    'max': series.max(),
                    'latest': series.iloc[-1],
                    'start_date': series.index.min(),
                    'end_date': series.index.max()
                }

    result = {
        'treasury': treasury_data,
        'policy': policy_data,
        'credit': credit_data,
        'curve_metrics': curve_metrics,
        'combined': combined_df,
        'summary_stats': summary_stats
    }

    _save_cache(result, cache_dir)
    logger.info("Cached comprehensive rates data to %s", cache_dir)

    # Print final summary
    logger.info("=" * 60)
    logger.info("=== Comprehensive Rates Data Summary ===")
    logger.info("=" * 60)
    logger.info("Total series loaded: %d", len(combined_df.columns))
    logger.info("Date range: %s to %s", combined_df.index.min(), combined_df.index.max())
    logger.info("Total observations: %d", len(combined_df))

    logger.info("--- Series Categories ---")
    logger.info("Treasury yields: %d", len(treasury_data['combined'].columns) if treasury_data else 0)
    logger.info("Policy rates: %d", len(policy_data['combined'].columns) if policy_data else 0)
    logger.info("Credit/Mortgage: %d", len(credit_data['combined'].columns) if credit_data else 0)
    logger.info("Curve metrics: %d", len(curve_metrics.columns) if curve_metrics is not None else 0)

    logger.info("=" * 60)
    logger.info("Citation:")
    logger.info("Federal Reserve Economic Data (FRED), Federal Reserve Bank of St. Louis")
    logger.info("https://fred.stlouisfed.org/")
    logger.info("=" * 60)

    return result


# =============================================================================
# INDUSTRIAL PRODUCTION DATA
# =============================================================================
def load_industrial_production_data(
    start_date='2000-01-01',
    end_date=None,
    cache_path='./data/fred'
):
    """
    Load Industrial Production (IP) data from FRED (2000-2025).

    Provides comprehensive industrial production measures:
    - Total Industrial Production Index
    - Manufacturing production
    - Mining production
    - Utilities production
    - Capacity Utilization rates

    Parameters:
    -----------
    start_date : str
        Start date in 'YYYY-MM-DD' format
    end_date : str
        End date in 'YYYY-MM-DD' format (defaults to today)
    cache_path : str
        Directory to cache downloaded data

    Returns:
    --------
    dict : Dictionary containing:
        - 'total': Total industrial production index
        - 'sectors': Sector-level production indices
        - 'capacity': Capacity utilization rates
        - 'combined': All IP measures in one DataFrame
    """
    if end_date is None:
        end_date = datetime.today().strftime('%Y-%m-%d')

    os.makedirs(cache_path, exist_ok=True)
    cache_dir = os.path.join(cache_path, f'industrial_production_{start_date}_{end_date}')

    if os.path.exists(cache_dir):
        logger.info("Loading cached industrial production data from %s", cache_dir)
        return _load_cache(cache_dir)

    logger.info("Downloading Industrial Production data from FRED...")

    # Total Industrial Production
    total_series = {
        'IP_Total': 'INDPRO',                   # Industrial Production: Total Index
        'IP_Total_ExcludeHighTech': 'IPXHTE',   # IP excluding High-Tech
    }

    # Sector-level production
    sector_series = {
        'IP_Manufacturing': 'IPMAN',            # Manufacturing
        'IP_Mining': 'IPMINE',                  # Mining
        'IP_Utilities': 'IPUTIL',               # Utilities
        'IP_Durable_Goods': 'IPDMAN',           # Durable Goods Manufacturing
        'IP_Nondurable_Goods': 'IPNMAN',        # Nondurable Goods Manufacturing
        'IP_Consumer_Goods': 'IPCONGD',         # Consumer Goods
        'IP_Business_Equipment': 'IPBUSEQ',     # Business Equipment
        'IP_Materials': 'IPMAT',                # Materials
        'IP_Final_Products': 'IPFINAL',         # Final Products
        'IP_Motor_Vehicles': 'IPG3361T3S',      # Motor Vehicles and Parts
        'IP_Computers_Electronics': 'IPG334S',  # Computer and Electronic Products
        'IP_Chemicals': 'IPG325S',              # Chemicals
        'IP_Primary_Metals': 'IPG331S',         # Primary Metals
        'IP_Food_Beverage': 'IPG311A2S',        # Food, Beverage, and Tobacco
        'IP_Petroleum_Coal': 'IPG324S',         # Petroleum and Coal Products
        'IP_Machinery': 'IPG333S',              # Machinery
    }

    # Capacity Utilization
    capacity_series = {
        'CapUtil_Total': 'TCU',                 # Total Capacity Utilization
        'CapUtil_Manufacturing': 'MCUMFN',      # Manufacturing Capacity Utilization
        'CapUtil_Mining': 'CAPUTLG21S',         # Mining Capacity Utilization
        'CapUtil_Utilities': 'CAPUTLG2211S',    # Utilities Capacity Utilization
        'CapUtil_Durable_Goods': 'CAPUTLDGMFG', # Durable Goods Capacity Utilization
        'CapUtil_Nondurable_Goods': 'CAPUTLNDMFG',  # Nondurable Goods Capacity Utilization
        'CapUtil_HighTech': 'CAPUTLHTI',        # High-Tech Industries Capacity Utilization
    }

    try:
        # Download total IP
        logger.info("--- Total Industrial Production ---")
        total_df = _download_fred_series(total_series, start_date, end_date)

        # Download sector production
        logger.info("--- Sector-Level Production ---")
        sector_df = _download_fred_series(sector_series, start_date, end_date)

        # Download capacity utilization
        logger.info("--- Capacity Utilization ---")
        capacity_df = _download_fred_series(capacity_series, start_date, end_date)

        # Combine all data
        combined_df = pd.concat([total_df, sector_df, capacity_df], axis=1)

        result = {
            'total': total_df,
            'sectors': sector_df,
            'capacity': capacity_df,
            'combined': combined_df
        }

        _save_cache(result, cache_dir)
        logger.info("Cached industrial production data to %s", cache_dir)

        # Print summary
        logger.info("=" * 60)
        logger.info("=== Industrial Production Summary ===")
        logger.info("=" * 60)
        logger.info("Date range: %s to %s", combined_df.index.min(), combined_df.index.max())
        logger.info("Observations: %d", len(combined_df))
        logger.info("Total IP series: %d", len(total_df.columns))
        logger.info("Sector series: %d", len(sector_df.columns))
        logger.info("Capacity utilization series: %d", len(capacity_df.columns))

        logger.info("Latest values:")
        for col in combined_df.columns:
            latest = combined_df[col].dropna().iloc[-1] if len(combined_df[col].dropna()) > 0 else 'N/A'
            if isinstance(latest, float):
                if 'CapUtil' in col:
                    logger.info("  %s: %.1f%%", col, latest)
                else:
                    logger.info("  %s: %.2f", col, latest)
            else:
                logger.info("  %s: %s", col, latest)

        return result

    except Exception as e:
        logger.error("downloading industrial production data: %s", e)
        return None


def load_ip_growth_rates(
    start_date='2000-01-01',
    end_date=None,
    cache_path='./data/fred'
):
    """
    Load Industrial Production growth rates from FRED.

    Provides year-over-year and month-over-month growth rates
    for industrial production indices.

    Parameters:
    -----------
    start_date : str
        Start date in 'YYYY-MM-DD' format
    end_date : str
        End date in 'YYYY-MM-DD' format (defaults to today)
    cache_path : str
        Directory to cache downloaded data

    Returns:
    --------
    dict : Dictionary containing:
        - 'yoy': Year-over-year growth rates
        - 'mom': Month-over-month growth rates
        - 'diffusion': Diffusion indices
        - 'combined': All growth measures
    """
    if end_date is None:
        end_date = datetime.today().strftime('%Y-%m-%d')

    os.makedirs(cache_path, exist_ok=True)
    cache_dir = os.path.join(cache_path, f'ip_growth_{start_date}_{end_date}')

    if os.path.exists(cache_dir):
        logger.info("Loading cached IP growth data from %s", cache_dir)
        return _load_cache(cache_dir)

    logger.info("Downloading IP growth rates from FRED...")

    # Base IP series for growth calculation (names are final column names)
    growth_series = {
        'IP': 'INDPRO',
        'IP_Manufacturing': 'IPMAN',
    }

    # Diffusion indices (percent of industries expanding)
    diffusion_series = {
        'IP_Diffusion_1M': 'IPDIFF1M',          # 1-Month Diffusion Index
        'IP_Diffusion_3M': 'IPDIFF3M',          # 3-Month Diffusion Index
        'IP_Diffusion_6M': 'IPDIFF6M',          # 6-Month Diffusion Index
    }

    try:
        # Download base IP series for growth calculation
        logger.info("--- Industrial Production Indices ---")
        ip_df = _download_fred_series(growth_series, start_date, end_date)

        # Calculate YoY growth rates
        logger.info("Calculating year-over-year growth rates...")
        yoy_df = ip_df.pct_change(periods=12) * 100
        yoy_df.columns = [f'{col}_YoY' for col in yoy_df.columns]

        # Calculate MoM growth rates
        logger.info("Calculating month-over-month growth rates...")
        mom_df = ip_df.pct_change() * 100
        mom_df.columns = [f'{col}_MoM' for col in mom_df.columns]

        # Download diffusion indices
        logger.info("--- Diffusion Indices ---")
        diffusion_df = _download_fred_series(diffusion_series, start_date, end_date)

        # Combine all data
        combined_df = pd.concat([yoy_df, mom_df, diffusion_df], axis=1)

        result = {
            'raw': ip_df,
            'yoy': yoy_df,
            'mom': mom_df,
            'diffusion': diffusion_df,
            'combined': combined_df
        }

        _save_cache(result, cache_dir)
        logger.info("Cached IP growth data to %s", cache_dir)

        # Print summary
        logger.info("=" * 60)
        logger.info("=== IP Growth Rates Summary ===")
        logger.info("=" * 60)
        logger.info("Date range: %s to %s", combined_df.index.min(), combined_df.index.max())
        logger.info("Observations: %d", len(combined_df))

        logger.info("Latest growth rates:")
        for col in yoy_df.columns:
            latest = yoy_df[col].dropna().iloc[-1] if len(yoy_df[col].dropna()) > 0 else 'N/A'
            if isinstance(latest, float):
                logger.info("  %s: %.2f%%", col, latest)
            else:
                logger.info("  %s: %s", col, latest)

        if len(diffusion_df) > 0:
            logger.info("Latest diffusion indices:")
            for col in diffusion_df.columns:
                latest = diffusion_df[col].dropna().iloc[-1] if len(diffusion_df[col].dropna()) > 0 else 'N/A'
                if isinstance(latest, float):
                    logger.info("  %s: %.1f", col, latest)
                else:
                    logger.info("  %s: %s", col, latest)

        return result

    except Exception as e:
        logger.error("downloading IP growth data: %s", e)
        return None


def load_comprehensive_ip_data(
    start_date='2000-01-01',
    end_date=None,
    cache_path='./data/fred',
    force_refresh=False
):
    """
    Load comprehensive Industrial Production dataset from FRED (2000-2025).

    This function aggregates:
    - Total Industrial Production indices
    - Sector-level production (Manufacturing, Mining, Utilities)
    - Industry-specific production (Motor Vehicles, Chemicals, etc.)
    - Capacity Utilization rates
    - Growth rates (YoY, MoM)
    - Diffusion indices

    Parameters:
    -----------
    start_date : str
        Start date in 'YYYY-MM-DD' format
    end_date : str
        End date in 'YYYY-MM-DD' format (defaults to today)
    cache_path : str
        Directory to cache downloaded data

    Returns:
    --------
    dict : Dictionary containing:
        - 'production': IP indices and sector data
        - 'growth': Growth rates and diffusion indices
        - 'combined': All IP measures merged on date
        - 'summary_stats': Summary statistics for all series
    """
    if end_date is None:
        end_date = datetime.today().strftime('%Y-%m-%d')

    os.makedirs(cache_path, exist_ok=True)
    cache_dir = os.path.join(
        cache_path,
        f'comprehensive_ip_{start_date}_{end_date}'
    )

    if force_refresh and os.path.exists(cache_dir):
        shutil.rmtree(cache_dir)
        logger.info("Force refresh: removed %s", cache_dir)

    if os.path.exists(cache_dir):
        logger.info("Loading cached comprehensive IP data from %s", cache_dir)
        return _load_cache(cache_dir)
    logger.info("=" * 60)
    logger.info("Loading Comprehensive Industrial Production Data (2000-2025)")
    logger.info("=" * 60)

    # Load production data
    logger.info("[1/2] Loading industrial production indices...")
    production_data = load_industrial_production_data(start_date, end_date, cache_path)

    # Load growth rates
    logger.info("[2/2] Loading growth rates and diffusion indices...")
    growth_data = load_ip_growth_rates(start_date, end_date, cache_path)

    # Combine all data
    combined_dfs = []

    if production_data and 'combined' in production_data:
        combined_dfs.append(production_data['combined'])

    if growth_data and 'combined' in growth_data:
        combined_dfs.append(growth_data['combined'])

    if combined_dfs:
        combined_df = pd.concat(combined_dfs, axis=1)
        # Remove duplicate columns if any
        combined_df = combined_df.loc[:, ~combined_df.columns.duplicated()]
    else:
        combined_df = pd.DataFrame()

    # Calculate summary statistics
    summary_stats = {}
    if len(combined_df) > 0:
        for col in combined_df.columns:
            series = combined_df[col].dropna()
            if len(series) > 0:
                summary_stats[col] = {
                    'count': len(series),
                    'mean': series.mean(),
                    'std': series.std(),
                    'min': series.min(),
                    'max': series.max(),
                    'latest': series.iloc[-1],
                    'start_date': series.index.min(),
                    'end_date': series.index.max()
                }

    result = {
        'production': production_data,
        'growth': growth_data,
        'combined': combined_df,
        'summary_stats': summary_stats
    }

    _save_cache(result, cache_dir)
    logger.info("Cached comprehensive IP data to %s", cache_dir)

    # Print final summary
    logger.info("=" * 60)
    logger.info("=== Comprehensive Industrial Production Summary ===")
    logger.info("=" * 60)
    logger.info("Total series loaded: %d", len(combined_df.columns))
    logger.info("Date range: %s to %s", combined_df.index.min(), combined_df.index.max())
    logger.info("Total observations: %d", len(combined_df))

    logger.info("--- Series Categories ---")
    if production_data:
        logger.info("Production indices: %d", len(production_data['combined'].columns))
    if growth_data:
        logger.info("Growth rates & diffusion: %d", len(growth_data['combined'].columns))

    # Key metrics
    if production_data and 'total' in production_data:
        total_df = production_data['total']
        if 'IP_Total' in total_df.columns:
            latest_ip = total_df['IP_Total'].dropna().iloc[-1]
            logger.info("Latest Total IP Index: %.2f", latest_ip)

    if production_data and 'capacity' in production_data:
        cap_df = production_data['capacity']
        if 'CapUtil_Total' in cap_df.columns:
            latest_cap = cap_df['CapUtil_Total'].dropna().iloc[-1]
            logger.info("Latest Capacity Utilization: %.1f%%", latest_cap)

    if growth_data and 'yoy' in growth_data:
        yoy_df = growth_data['yoy']
        if 'IP_Total_YoY' in yoy_df.columns or 'IP_YoY' in yoy_df.columns:
            col = 'IP_Total_YoY' if 'IP_Total_YoY' in yoy_df.columns else 'IP_YoY'
            latest_growth = yoy_df[col].dropna().iloc[-1] if len(yoy_df[col].dropna()) > 0 else None
            if latest_growth is not None:
                logger.info("Latest IP YoY Growth: %.2f%%", latest_growth)

    logger.info("=" * 60)
    logger.info("Citation:")
    logger.info("Board of Governors of the Federal Reserve System (US)")
    logger.info("Federal Reserve Economic Data (FRED), Federal Reserve Bank of St. Louis")
    logger.info("https://fred.stlouisfed.org/")
    logger.info("=" * 60)

    return result


# =============================================================================
# M2 MONEY SUPPLY DATA
# =============================================================================
def load_money_supply_data(
    start_date='2000-01-01',
    end_date=None,
    cache_path='./data/fred'
):
    """
    Load Money Supply (M1, M2) data from FRED (2000-2025).

    Provides comprehensive money supply measures:
    - M1: Currency + demand deposits + other checkable deposits
    - M2: M1 + savings deposits + small time deposits + retail money funds
    - Monetary Base
    - Currency in Circulation

    Parameters:
    -----------
    start_date : str
        Start date in 'YYYY-MM-DD' format
    end_date : str
        End date in 'YYYY-MM-DD' format (defaults to today)
    cache_path : str
        Directory to cache downloaded data

    Returns:
    --------
    dict : Dictionary containing:
        - 'aggregates': M1, M2, and monetary base
        - 'components': Components of money supply
        - 'combined': All money supply measures in one DataFrame
    """
    if end_date is None:
        end_date = datetime.today().strftime('%Y-%m-%d')

    os.makedirs(cache_path, exist_ok=True)
    cache_dir = os.path.join(cache_path, f'money_supply_{start_date}_{end_date}')

    if os.path.exists(cache_dir):
        logger.info("Loading cached money supply data from %s", cache_dir)
        return _load_cache(cache_dir)

    logger.info("Downloading Money Supply data from FRED...")

    # Money supply aggregates
    aggregate_series = {
        'M1': 'M1SL',                           # M1 Money Stock
        'M2': 'M2SL',                           # M2 Money Stock
        'Monetary_Base': 'BOGMBASE',            # Monetary Base; Total
        'Monetary_Base_Adjusted': 'BOGMBASEW',  # Monetary Base (Weekly)
    }

    # Money supply components
    component_series = {
        'Currency_Circulation': 'CURRSL',       # Currency in Circulation
        'Demand_Deposits': 'DEMDEPSL',          # Demand Deposits
        'Savings_Deposits': 'SAVINGSL',         # Savings Deposits
        'Retail_Money_Funds': 'RMFSL',          # Retail Money Market Funds
        'Small_Time_Deposits': 'STDSL',         # Small Time Deposits
        'Checkable_Deposits': 'TCDSL',          # Total Checkable Deposits
        'Travelers_Checks': 'TVCKSSL',          # Travelers Checks Outstanding
    }

    try:
        # Download money supply aggregates
        logger.info("--- Money Supply Aggregates ---")
        aggregate_df = _download_fred_series(aggregate_series, start_date, end_date)

        # Download money supply components
        logger.info("--- Money Supply Components ---")
        component_df = _download_fred_series(component_series, start_date, end_date)

        # Combine all data
        combined_df = pd.concat([aggregate_df, component_df], axis=1)

        result = {
            'aggregates': aggregate_df,
            'components': component_df,
            'combined': combined_df
        }

        _save_cache(result, cache_dir)
        logger.info("Cached money supply data to %s", cache_dir)

        # Print summary
        logger.info("=" * 60)
        logger.info("=== Money Supply Summary ===")
        logger.info("=" * 60)
        logger.info("Date range: %s to %s", combined_df.index.min(), combined_df.index.max())
        logger.info("Observations: %d", len(combined_df))
        logger.info("Aggregate series: %d", len(aggregate_df.columns))
        logger.info("Component series: %d", len(component_df.columns))

        logger.info("Latest values (Billions USD):")
        for col in combined_df.columns:
            latest = combined_df[col].dropna().iloc[-1] if len(combined_df[col].dropna()) > 0 else 'N/A'
            if isinstance(latest, float):
                logger.info("  %s: $%.1fB", col, latest)
            else:
                logger.info("  %s: %s", col, latest)

        return result

    except Exception as e:
        logger.error("downloading money supply data: %s", e)
        return None


def load_money_velocity_data(
    start_date='2000-01-01',
    end_date=None,
    cache_path='./data/fred'
):
    """
    Load Money Velocity data from FRED.

    Velocity measures how quickly money circulates in the economy.
    V = GDP / Money Stock

    Parameters:
    -----------
    start_date : str
        Start date in 'YYYY-MM-DD' format
    end_date : str
        End date in 'YYYY-MM-DD' format (defaults to today)
    cache_path : str
        Directory to cache downloaded data

    Returns:
    --------
    dict : Dictionary containing:
        - 'velocity': M1 and M2 velocity measures
        - 'combined': All velocity measures
    """
    if end_date is None:
        end_date = datetime.today().strftime('%Y-%m-%d')

    os.makedirs(cache_path, exist_ok=True)
    cache_dir = os.path.join(cache_path, f'money_velocity_{start_date}_{end_date}')

    if os.path.exists(cache_dir):
        logger.info("Loading cached money velocity data from %s", cache_dir)
        return _load_cache(cache_dir)

    logger.info("Downloading Money Velocity data from FRED...")

    velocity_series = {
        'M1_Velocity': 'M1V',                   # Velocity of M1 Money Stock
        'M2_Velocity': 'M2V',                   # Velocity of M2 Money Stock
    }

    try:
        logger.info("--- Money Velocity ---")
        velocity_df = _download_fred_series(velocity_series, start_date, end_date)

        result = {
            'velocity': velocity_df,
            'combined': velocity_df
        }

        _save_cache(result, cache_dir)
        logger.info("Cached money velocity data to %s", cache_dir)

        # Print summary
        logger.info("=" * 60)
        logger.info("=== Money Velocity Summary ===")
        logger.info("=" * 60)
        logger.info("Date range: %s to %s", velocity_df.index.min(), velocity_df.index.max())
        logger.info("Observations: %d", len(velocity_df))

        logger.info("Latest values:")
        for col in velocity_df.columns:
            latest = velocity_df[col].dropna().iloc[-1] if len(velocity_df[col].dropna()) > 0 else 'N/A'
            if isinstance(latest, float):
                logger.info("  %s: %.2f", col, latest)
            else:
                logger.info("  %s: %s", col, latest)

        return result

    except Exception as e:
        logger.error("downloading money velocity data: %s", e)
        return None


def load_fed_balance_sheet_data(
    start_date='2000-01-01',
    end_date=None,
    cache_path='./data/fred'
):
    """
    Load Federal Reserve Balance Sheet data from FRED.

    Provides Fed assets, liabilities, and reserve measures:
    - Total Assets
    - Treasury Holdings
    - MBS Holdings
    - Reserve Balances
    - Excess Reserves

    Parameters:
    -----------
    start_date : str
        Start date in 'YYYY-MM-DD' format
    end_date : str
        End date in 'YYYY-MM-DD' format (defaults to today)
    cache_path : str
        Directory to cache downloaded data

    Returns:
    --------
    dict : Dictionary containing:
        - 'assets': Fed asset holdings
        - 'reserves': Bank reserve measures
        - 'combined': All balance sheet data
    """
    if end_date is None:
        end_date = datetime.today().strftime('%Y-%m-%d')

    os.makedirs(cache_path, exist_ok=True)
    cache_dir = os.path.join(cache_path, f'fed_balance_sheet_{start_date}_{end_date}')

    if os.path.exists(cache_dir):
        logger.info("Loading cached Fed balance sheet data from %s", cache_dir)
        return _load_cache(cache_dir)

    logger.info("Downloading Federal Reserve Balance Sheet data from FRED...")

    # Fed assets
    asset_series = {
        'Fed_Total_Assets': 'WALCL',            # Total Assets
        'Fed_Treasury_Holdings': 'TREAST',      # Treasury Securities Held
        'Fed_MBS_Holdings': 'WSHOMCB',          # Mortgage-Backed Securities Held
        'Fed_Agency_Debt': 'WSHOFDSL',          # Federal Agency Debt Securities
    }

    # Reserve measures
    reserve_series = {
        'Reserve_Balances': 'WRESBAL',          # Reserve Balances with Fed
        'Required_Reserves': 'REQRESNS',        # Required Reserves
        'Excess_Reserves': 'EXCSRESNS',         # Excess Reserves
        'Total_Reserves': 'TOTRESNS',           # Total Reserves
    }

    try:
        # Download Fed assets
        logger.info("--- Federal Reserve Assets ---")
        asset_df = _download_fred_series(asset_series, start_date, end_date)

        # Download reserve measures
        logger.info("--- Bank Reserves ---")
        reserve_df = _download_fred_series(reserve_series, start_date, end_date)

        # Combine all data
        combined_df = pd.concat([asset_df, reserve_df], axis=1)

        result = {
            'assets': asset_df,
            'reserves': reserve_df,
            'combined': combined_df
        }

        _save_cache(result, cache_dir)
        logger.info("Cached Fed balance sheet data to %s", cache_dir)

        # Print summary
        logger.info("=" * 60)
        logger.info("=== Fed Balance Sheet Summary ===")
        logger.info("=" * 60)
        logger.info("Date range: %s to %s", combined_df.index.min(), combined_df.index.max())
        logger.info("Observations: %d", len(combined_df))
        logger.info("Asset series: %d", len(asset_df.columns))
        logger.info("Reserve series: %d", len(reserve_df.columns))

        logger.info("Latest values (Millions/Billions USD):")
        for col in combined_df.columns:
            latest = combined_df[col].dropna().iloc[-1] if len(combined_df[col].dropna()) > 0 else 'N/A'
            if isinstance(latest, float):
                if latest > 1000000:
                    logger.info("  %s: $%.2fT", col, latest/1000000)
                elif latest > 1000:
                    logger.info("  %s: $%.1fB", col, latest/1000)
                else:
                    logger.info("  %s: $%.1fM", col, latest)
            else:
                logger.info("  %s: %s", col, latest)

        return result

    except Exception as e:
        logger.error("downloading Fed balance sheet data: %s", e)
        return None


def load_m2_growth_rates(money_supply_data):
    """
    Calculate M2 growth rates from money supply data.

    Parameters:
    -----------
    money_supply_data : dict
        Output from load_money_supply_data()

    Returns:
    --------
    pd.DataFrame : M2 growth rates (YoY, MoM)
    """
    if money_supply_data is None or 'aggregates' not in money_supply_data:
        logger.info("Money supply data not available")
        return None

    aggregates = money_supply_data['aggregates']
    growth = pd.DataFrame(index=aggregates.index)

    logger.info("Calculating M2 growth rates...")

    # Year-over-Year growth
    if 'M2' in aggregates.columns:
        growth['M2_YoY'] = aggregates['M2'].pct_change(periods=12) * 100
        logger.info("  Calculated M2 YoY growth")

    if 'M1' in aggregates.columns:
        growth['M1_YoY'] = aggregates['M1'].pct_change(periods=12) * 100
        logger.info("  Calculated M1 YoY growth")

    # Month-over-Month growth (annualized)
    if 'M2' in aggregates.columns:
        growth['M2_MoM'] = aggregates['M2'].pct_change() * 100
        growth['M2_MoM_Annualized'] = aggregates['M2'].pct_change() * 100 * 12
        logger.info("  Calculated M2 MoM growth")

    if 'M1' in aggregates.columns:
        growth['M1_MoM'] = aggregates['M1'].pct_change() * 100
        growth['M1_MoM_Annualized'] = aggregates['M1'].pct_change() * 100 * 12
        logger.info("  Calculated M1 MoM growth")

    # 3-month and 6-month annualized growth
    if 'M2' in aggregates.columns:
        growth['M2_3M_Annualized'] = aggregates['M2'].pct_change(periods=3) * 100 * 4
        growth['M2_6M_Annualized'] = aggregates['M2'].pct_change(periods=6) * 100 * 2
        logger.info("  Calculated M2 3M and 6M annualized growth")

    # Print summary
    logger.info("=" * 60)
    logger.info("=== M2 Growth Rates Summary ===")
    logger.info("=" * 60)
    logger.info("Date range: %s to %s", growth.index.min(), growth.index.max())

    logger.info("Latest growth rates:")
    for col in growth.columns:
        latest = growth[col].dropna().iloc[-1] if len(growth[col].dropna()) > 0 else 'N/A'
        if isinstance(latest, float):
            logger.info("  %s: %.2f%%", col, latest)
        else:
            logger.info("  %s: %s", col, latest)

    return growth


def load_comprehensive_m2_data(
    start_date='2000-01-01',
    end_date=None,
    cache_path='./data/fred',
    force_refresh=False
):
    """
    Load comprehensive M2 Money Supply dataset from FRED (2000-2025).

    This function aggregates:
    - Money supply aggregates (M1, M2, Monetary Base)
    - Money supply components (Currency, Deposits, etc.)
    - Money velocity (M1V, M2V)
    - Federal Reserve balance sheet data
    - Growth rates (YoY, MoM, annualized)

    Parameters:
    -----------
    start_date : str
        Start date in 'YYYY-MM-DD' format
    end_date : str
        End date in 'YYYY-MM-DD' format (defaults to today)
    cache_path : str
        Directory to cache downloaded data

    Returns:
    --------
    dict : Dictionary containing:
        - 'money_supply': M1, M2, components
        - 'velocity': Money velocity measures
        - 'fed_balance_sheet': Fed assets and reserves
        - 'growth_rates': M2 growth calculations
        - 'combined': All M2 measures merged on date
        - 'summary_stats': Summary statistics for all series
    """
    if end_date is None:
        end_date = datetime.today().strftime('%Y-%m-%d')

    os.makedirs(cache_path, exist_ok=True)
    cache_dir = os.path.join(
        cache_path,
        f'comprehensive_m2_{start_date}_{end_date}'
    )

    if force_refresh and os.path.exists(cache_dir):
        shutil.rmtree(cache_dir)
        logger.info("Force refresh: removed %s", cache_dir)

    if os.path.exists(cache_dir):
        logger.info("Loading cached comprehensive M2 data from %s", cache_dir)
        return _load_cache(cache_dir)
    logger.info("=" * 60)
    logger.info("Loading Comprehensive M2 Money Supply Data (2000-2025)")
    logger.info("=" * 60)

    # Load money supply data
    logger.info("[1/4] Loading money supply aggregates and components...")
    money_supply_data = load_money_supply_data(start_date, end_date, cache_path)

    # Load velocity data
    logger.info("[2/4] Loading money velocity...")
    velocity_data = load_money_velocity_data(start_date, end_date, cache_path)

    # Load Fed balance sheet
    logger.info("[3/4] Loading Fed balance sheet...")
    fed_data = load_fed_balance_sheet_data(start_date, end_date, cache_path)

    # Calculate growth rates
    logger.info("[4/4] Calculating M2 growth rates...")
    growth_rates = load_m2_growth_rates(money_supply_data)

    # Combine all data
    combined_dfs = []

    if money_supply_data and 'combined' in money_supply_data:
        combined_dfs.append(money_supply_data['combined'])

    if velocity_data and 'combined' in velocity_data:
        combined_dfs.append(velocity_data['combined'])

    if fed_data and 'combined' in fed_data:
        combined_dfs.append(fed_data['combined'])

    if growth_rates is not None and len(growth_rates) > 0:
        combined_dfs.append(growth_rates)

    if combined_dfs:
        combined_df = pd.concat(combined_dfs, axis=1)
        # Remove duplicate columns if any
        combined_df = combined_df.loc[:, ~combined_df.columns.duplicated()]
    else:
        combined_df = pd.DataFrame()

    # Calculate summary statistics
    summary_stats = {}
    if len(combined_df) > 0:
        for col in combined_df.columns:
            series = combined_df[col].dropna()
            if len(series) > 0:
                summary_stats[col] = {
                    'count': len(series),
                    'mean': series.mean(),
                    'std': series.std(),
                    'min': series.min(),
                    'max': series.max(),
                    'latest': series.iloc[-1],
                    'start_date': series.index.min(),
                    'end_date': series.index.max()
                }

    result = {
        'money_supply': money_supply_data,
        'velocity': velocity_data,
        'fed_balance_sheet': fed_data,
        'growth_rates': growth_rates,
        'combined': combined_df,
        'summary_stats': summary_stats
    }

    _save_cache(result, cache_dir)
    logger.info("Cached comprehensive M2 data to %s", cache_dir)

    # Print final summary
    logger.info("=" * 60)
    logger.info("=== Comprehensive M2 Money Supply Summary ===")
    logger.info("=" * 60)
    logger.info("Total series loaded: %d", len(combined_df.columns))
    logger.info("Date range: %s to %s", combined_df.index.min(), combined_df.index.max())
    logger.info("Total observations: %d", len(combined_df))

    logger.info("--- Series Categories ---")
    if money_supply_data:
        logger.info("Money supply measures: %d", len(money_supply_data['combined'].columns))
    if velocity_data:
        logger.info("Velocity measures: %d", len(velocity_data['combined'].columns))
    if fed_data:
        logger.info("Fed balance sheet: %d", len(fed_data['combined'].columns))
    if growth_rates is not None:
        logger.info("Growth rates: %d", len(growth_rates.columns))

    # Key metrics
    if money_supply_data and 'aggregates' in money_supply_data:
        agg_df = money_supply_data['aggregates']
        if 'M2' in agg_df.columns:
            latest_m2 = agg_df['M2'].dropna().iloc[-1]
            logger.info("Latest M2 Money Stock: $%.1fB", latest_m2)

    if growth_rates is not None and 'M2_YoY' in growth_rates.columns:
        latest_growth = growth_rates['M2_YoY'].dropna().iloc[-1]
        logger.info("Latest M2 YoY Growth: %.2f%%", latest_growth)

    if velocity_data and 'velocity' in velocity_data:
        vel_df = velocity_data['velocity']
        if 'M2_Velocity' in vel_df.columns:
            latest_vel = vel_df['M2_Velocity'].dropna().iloc[-1]
            logger.info("Latest M2 Velocity: %.2f", latest_vel)

    logger.info("=" * 60)
    logger.info("Citation:")
    logger.info("Board of Governors of the Federal Reserve System (US)")
    logger.info("Federal Reserve Economic Data (FRED), Federal Reserve Bank of St. Louis")
    logger.info("https://fred.stlouisfed.org/")
    logger.info("=" * 60)

    return result


# =============================================================================
# GDP DATA
# =============================================================================
def load_gdp_data(
    start_date='2000-01-01',
    end_date=None,
    cache_path='./data/fred'
):
    """
    Load GDP data from FRED (2000-2025).

    Provides comprehensive GDP measures:
    - Nominal GDP
    - Real GDP (inflation-adjusted)
    - GDP growth rates
    - Per capita GDP

    Parameters:
    -----------
    start_date : str
        Start date in 'YYYY-MM-DD' format
    end_date : str
        End date in 'YYYY-MM-DD' format (defaults to today)
    cache_path : str
        Directory to cache downloaded data

    Returns:
    --------
    dict : Dictionary containing:
        - 'headline': Nominal and Real GDP
        - 'growth': GDP growth rates
        - 'per_capita': Per capita measures
        - 'combined': All GDP measures in one DataFrame
    """
    if end_date is None:
        end_date = datetime.today().strftime('%Y-%m-%d')

    os.makedirs(cache_path, exist_ok=True)
    cache_dir = os.path.join(cache_path, f'gdp_data_{start_date}_{end_date}')

    if os.path.exists(cache_dir):
        logger.info("Loading cached GDP data from %s", cache_dir)
        return _load_cache(cache_dir)

    logger.info("Downloading GDP data from FRED...")

    # Headline GDP measures
    headline_series = {
        'GDP_Nominal': 'GDP',                   # Gross Domestic Product (Nominal)
        'GDP_Real': 'GDPC1',                    # Real Gross Domestic Product
        'GNP_Nominal': 'GNP',                   # Gross National Product
        'GNP_Real': 'GNPC96',                   # Real Gross National Product
    }

    # GDP growth rates
    growth_series = {
        'GDP_Growth_QoQ': 'A191RL1Q225SBEA',    # Real GDP Growth Rate (QoQ annualized)
        'GDP_Growth_Pct_Change': 'A191RO1Q156NBEA',  # Real GDP Percent Change
    }

    # Per capita measures
    per_capita_series = {
        'GDP_Per_Capita_Nominal': 'A939RC0Q052SBEA',  # GDP Per Capita
        'GDP_Per_Capita_Real': 'A939RX0Q048SBEA',     # Real GDP Per Capita
    }

    try:
        # Download headline GDP
        logger.info("--- Headline GDP ---")
        headline_df = _download_fred_series(headline_series, start_date, end_date)

        # Download growth rates
        logger.info("--- GDP Growth Rates ---")
        growth_df = _download_fred_series(growth_series, start_date, end_date)

        # Download per capita measures
        logger.info("--- Per Capita GDP ---")
        per_capita_df = _download_fred_series(per_capita_series, start_date, end_date)

        # Combine all data
        combined_df = pd.concat([headline_df, growth_df, per_capita_df], axis=1)

        result = {
            'headline': headline_df,
            'growth': growth_df,
            'per_capita': per_capita_df,
            'combined': combined_df
        }

        _save_cache(result, cache_dir)
        logger.info("Cached GDP data to %s", cache_dir)

        # Print summary
        logger.info("=" * 60)
        logger.info("=== GDP Summary ===")
        logger.info("=" * 60)
        logger.info("Date range: %s to %s", combined_df.index.min(), combined_df.index.max())
        logger.info("Observations: %d", len(combined_df))
        logger.info("Headline series: %d", len(headline_df.columns))
        logger.info("Growth series: %d", len(growth_df.columns))
        logger.info("Per capita series: %d", len(per_capita_df.columns))

        logger.info("Latest values:")
        for col in combined_df.columns:
            latest = combined_df[col].dropna().iloc[-1] if len(combined_df[col].dropna()) > 0 else 'N/A'
            if isinstance(latest, float):
                if 'Growth' in col or 'Pct' in col:
                    logger.info("  %s: %.2f%%", col, latest)
                elif latest > 1000:
                    logger.info("  %s: $%.1fB", col, latest)
                else:
                    logger.info("  %s: $%.2f", col, latest)
            else:
                logger.info("  %s: %s", col, latest)

        return result

    except Exception as e:
        logger.error("downloading GDP data: %s", e)
        return None


def load_gdp_components_data(
    start_date='2000-01-01',
    end_date=None,
    cache_path='./data/fred'
):
    """
    Load GDP components data from FRED.

    GDP = C + I + G + (X - M)
    - C: Personal Consumption Expenditures
    - I: Gross Private Domestic Investment
    - G: Government Consumption & Investment
    - X-M: Net Exports (Exports - Imports)

    Parameters:
    -----------
    start_date : str
        Start date in 'YYYY-MM-DD' format
    end_date : str
        End date in 'YYYY-MM-DD' format (defaults to today)
    cache_path : str
        Directory to cache downloaded data

    Returns:
    --------
    dict : Dictionary containing:
        - 'consumption': Personal consumption expenditures
        - 'investment': Private investment
        - 'government': Government spending
        - 'trade': Exports, imports, net exports
        - 'combined': All components in one DataFrame
    """
    if end_date is None:
        end_date = datetime.today().strftime('%Y-%m-%d')

    os.makedirs(cache_path, exist_ok=True)
    cache_dir = os.path.join(cache_path, f'gdp_components_{start_date}_{end_date}')

    if os.path.exists(cache_dir):
        logger.info("Loading cached GDP components from %s", cache_dir)
        return _load_cache(cache_dir)

    logger.info("Downloading GDP components from FRED...")

    # Personal Consumption Expenditures (C)
    consumption_series = {
        'PCE_Total': 'PCE',                     # Personal Consumption Expenditures
        'PCE_Real': 'PCEC96',                   # Real PCE
        'PCE_Goods': 'DGDSRC1',                 # PCE: Goods
        'PCE_Durable_Goods': 'PCDG',            # PCE: Durable Goods
        'PCE_Nondurable_Goods': 'PCND',         # PCE: Nondurable Goods
        'PCE_Services': 'PCESV',                # PCE: Services
    }

    # Gross Private Domestic Investment (I)
    investment_series = {
        'Investment_Total': 'GPDI',             # Gross Private Domestic Investment
        'Investment_Fixed': 'FPI',              # Fixed Private Investment
        'Investment_Nonresidential': 'PNFI',    # Private Nonresidential Fixed Investment
        'Investment_Residential': 'PRFI',       # Private Residential Fixed Investment
        'Investment_Inventories': 'CBI',        # Change in Private Inventories
    }

    # Government Consumption & Investment (G)
    government_series = {
        'Govt_Total': 'GCE',                    # Government Consumption & Investment
        'Govt_Federal': 'FGCE',                 # Federal Government
        'Govt_Defense': 'FDEFX',                # Federal Defense
        'Govt_Nondefense': 'FNDEX',             # Federal Nondefense
        'Govt_State_Local': 'SLCE',             # State and Local Government
    }

    # Net Exports (X - M)
    trade_series = {
        'Exports_Total': 'EXPGS',               # Exports of Goods and Services
        'Exports_Goods': 'EXPGSC1',             # Exports of Goods
        'Exports_Services': 'EXPGSCA',          # Exports of Services
        'Imports_Total': 'IMPGS',               # Imports of Goods and Services
        'Imports_Goods': 'IMPGSC1',             # Imports of Goods
        'Imports_Services': 'IMPGSCA',          # Imports of Services
        'Net_Exports': 'NETEXP',                # Net Exports
    }

    try:
        # Download consumption
        logger.info("--- Personal Consumption Expenditures (C) ---")
        consumption_df = _download_fred_series(consumption_series, start_date, end_date)

        # Download investment
        logger.info("--- Gross Private Domestic Investment (I) ---")
        investment_df = _download_fred_series(investment_series, start_date, end_date)

        # Download government spending
        logger.info("--- Government Consumption & Investment (G) ---")
        government_df = _download_fred_series(government_series, start_date, end_date)

        # Download trade data
        logger.info("--- Net Exports (X - M) ---")
        trade_df = _download_fred_series(trade_series, start_date, end_date)

        # Combine all data
        combined_df = pd.concat([consumption_df, investment_df, government_df, trade_df], axis=1)

        result = {
            'consumption': consumption_df,
            'investment': investment_df,
            'government': government_df,
            'trade': trade_df,
            'combined': combined_df
        }

        _save_cache(result, cache_dir)
        logger.info("Cached GDP components to %s", cache_dir)

        # Print summary
        logger.info("=" * 60)
        logger.info("=== GDP Components Summary ===")
        logger.info("=" * 60)
        logger.info("Date range: %s to %s", combined_df.index.min(), combined_df.index.max())
        logger.info("Observations: %d", len(combined_df))
        logger.info("Consumption (C): %d series", len(consumption_df.columns))
        logger.info("Investment (I): %d series", len(investment_df.columns))
        logger.info("Government (G): %d series", len(government_df.columns))
        logger.info("Trade (X-M): %d series", len(trade_df.columns))

        return result

    except Exception as e:
        logger.error("downloading GDP components: %s", e)
        return None


def load_gdp_by_industry_data(
    start_date='2000-01-01',
    end_date=None,
    cache_path='./data/fred'
):
    """
    Load GDP by industry/sector data from FRED.

    Provides value added by major industry sectors.

    Parameters:
    -----------
    start_date : str
        Start date in 'YYYY-MM-DD' format
    end_date : str
        End date in 'YYYY-MM-DD' format (defaults to today)
    cache_path : str
        Directory to cache downloaded data

    Returns:
    --------
    dict : Dictionary containing:
        - 'industries': GDP by industry
        - 'combined': All industry data
    """
    if end_date is None:
        end_date = datetime.today().strftime('%Y-%m-%d')

    os.makedirs(cache_path, exist_ok=True)
    cache_dir = os.path.join(cache_path, f'gdp_industry_{start_date}_{end_date}')

    if os.path.exists(cache_dir):
        logger.info("Loading cached GDP by industry from %s", cache_dir)
        return _load_cache(cache_dir)

    logger.info("Downloading GDP by industry from FRED...")

    # GDP by industry (Value Added)
    industry_series = {
        'VA_Private_Industries': 'VAPGDP',      # Private Industries Value Added
        'VA_Agriculture': 'VAGDPAG',            # Agriculture, Forestry, Fishing
        'VA_Mining': 'VAGDPMI',                 # Mining
        'VA_Utilities': 'VAGDPUT',              # Utilities
        'VA_Construction': 'VAGDPCO',           # Construction
        'VA_Manufacturing': 'VAGDPMF',          # Manufacturing
        'VA_Durable_Manufacturing': 'VAGDPDG',  # Durable Goods Manufacturing
        'VA_Nondurable_Manufacturing': 'VAGDPND',  # Nondurable Goods Manufacturing
        'VA_Wholesale_Trade': 'VAGDPWT',        # Wholesale Trade
        'VA_Retail_Trade': 'VAGDPRT',           # Retail Trade
        'VA_Transportation': 'VAGDPTW',         # Transportation and Warehousing
        'VA_Information': 'VAGDPIF',            # Information
        'VA_Finance_Insurance': 'VAGDPFI',      # Finance and Insurance
        'VA_Real_Estate': 'VAGDPRE',            # Real Estate
        'VA_Professional_Services': 'VAGDPPS',  # Professional and Business Services
        'VA_Education_Health': 'VAGDPEH',       # Educational Services, Health Care
        'VA_Arts_Entertainment': 'VAGDPAR',     # Arts, Entertainment, Recreation
        'VA_Government': 'VAGDPGV',             # Government
    }

    try:
        logger.info("--- GDP by Industry (Value Added) ---")
        industry_df = _download_fred_series(industry_series, start_date, end_date)

        result = {
            'industries': industry_df,
            'combined': industry_df
        }

        _save_cache(result, cache_dir)
        logger.info("Cached GDP by industry to %s", cache_dir)

        # Print summary
        logger.info("=" * 60)
        logger.info("=== GDP by Industry Summary ===")
        logger.info("=" * 60)
        logger.info("Date range: %s to %s", industry_df.index.min(), industry_df.index.max())
        logger.info("Observations: %d", len(industry_df))
        logger.info("Industry sectors: %d", len(industry_df.columns))

        return result

    except Exception as e:
        logger.error("downloading GDP by industry: %s", e)
        return None


def load_comprehensive_gdp_data(
    start_date='2000-01-01',
    end_date=None,
    cache_path='./data/fred',
    force_refresh=False
):
    """
    Load comprehensive GDP dataset from FRED (2000-2025).

    This function aggregates:
    - Headline GDP (Nominal, Real, GNP)
    - GDP growth rates
    - Per capita GDP
    - GDP components (C + I + G + NX)
    - GDP by industry/sector

    Parameters:
    -----------
    start_date : str
        Start date in 'YYYY-MM-DD' format
    end_date : str
        End date in 'YYYY-MM-DD' format (defaults to today)
    cache_path : str
        Directory to cache downloaded data

    Returns:
    --------
    dict : Dictionary containing:
        - 'headline': GDP headline measures
        - 'components': GDP expenditure components
        - 'industries': GDP by industry
        - 'combined': All GDP measures merged on date
        - 'summary_stats': Summary statistics for all series
    """
    if end_date is None:
        end_date = datetime.today().strftime('%Y-%m-%d')

    os.makedirs(cache_path, exist_ok=True)
    cache_dir = os.path.join(
        cache_path,
        f'comprehensive_gdp_{start_date}_{end_date}'
    )

    if force_refresh and os.path.exists(cache_dir):
        shutil.rmtree(cache_dir)
        logger.info("Force refresh: removed %s", cache_dir)

    if os.path.exists(cache_dir):
        logger.info("Loading cached comprehensive GDP data from %s", cache_dir)
        return _load_cache(cache_dir)
    logger.info("=" * 60)
    logger.info("Loading Comprehensive GDP Data (2000-2025)")
    logger.info("=" * 60)

    # Load headline GDP
    logger.info("[1/3] Loading headline GDP measures...")
    headline_data = load_gdp_data(start_date, end_date, cache_path)

    # Load GDP components
    logger.info("[2/3] Loading GDP components (C + I + G + NX)...")
    components_data = load_gdp_components_data(start_date, end_date, cache_path)

    # Load GDP by industry
    logger.info("[3/3] Loading GDP by industry...")
    industry_data = load_gdp_by_industry_data(start_date, end_date, cache_path)

    # Combine all data
    combined_dfs = []

    if headline_data and 'combined' in headline_data:
        combined_dfs.append(headline_data['combined'])

    if components_data and 'combined' in components_data:
        combined_dfs.append(components_data['combined'])

    if industry_data and 'combined' in industry_data:
        combined_dfs.append(industry_data['combined'])

    if combined_dfs:
        combined_df = pd.concat(combined_dfs, axis=1)
        # Remove duplicate columns if any
        combined_df = combined_df.loc[:, ~combined_df.columns.duplicated()]
    else:
        combined_df = pd.DataFrame()

    # Calculate summary statistics
    summary_stats = {}
    if len(combined_df) > 0:
        for col in combined_df.columns:
            series = combined_df[col].dropna()
            if len(series) > 0:
                summary_stats[col] = {
                    'count': len(series),
                    'mean': series.mean(),
                    'std': series.std(),
                    'min': series.min(),
                    'max': series.max(),
                    'latest': series.iloc[-1],
                    'start_date': series.index.min(),
                    'end_date': series.index.max()
                }

    result = {
        'headline': headline_data,
        'components': components_data,
        'industries': industry_data,
        'combined': combined_df,
        'summary_stats': summary_stats
    }

    _save_cache(result, cache_dir)
    logger.info("Cached comprehensive GDP data to %s", cache_dir)

    # Print final summary
    logger.info("=" * 60)
    logger.info("=== Comprehensive GDP Summary ===")
    logger.info("=" * 60)
    logger.info("Total series loaded: %d", len(combined_df.columns))
    logger.info("Date range: %s to %s", combined_df.index.min(), combined_df.index.max())
    logger.info("Total observations: %d", len(combined_df))

    logger.info("--- Series Categories ---")
    if headline_data:
        logger.info("Headline GDP: %d series", len(headline_data['combined'].columns))
    if components_data:
        logger.info("GDP components: %d series", len(components_data['combined'].columns))
    if industry_data:
        logger.info("GDP by industry: %d series", len(industry_data['combined'].columns))

    # Key metrics
    if headline_data and 'headline' in headline_data:
        hdl_df = headline_data['headline']
        if 'GDP_Real' in hdl_df.columns:
            latest_gdp = hdl_df['GDP_Real'].dropna().iloc[-1]
            logger.info("Latest Real GDP: $%.1fB", latest_gdp)

    if headline_data and 'growth' in headline_data:
        growth_df = headline_data['growth']
        if 'GDP_Growth_QoQ' in growth_df.columns:
            latest_growth = growth_df['GDP_Growth_QoQ'].dropna().iloc[-1]
            logger.info("Latest GDP Growth (QoQ Ann.): %.1f%%", latest_growth)

    logger.info("=" * 60)
    logger.info("Citation:")
    logger.info("U.S. Bureau of Economic Analysis (BEA)")
    logger.info("Federal Reserve Economic Data (FRED), Federal Reserve Bank of St. Louis")
    logger.info("https://fred.stlouisfed.org/")
    logger.info("=" * 60)

    return result


# =============================================================================
# EMPLOYMENT DATA
# =============================================================================
def load_employment_data(
    start_date='2000-01-01',
    end_date=None,
    cache_path='./data/fred'
):
    """
    Load employment data from FRED (2000-2025).

    Provides comprehensive employment measures:
    - Nonfarm Payrolls
    - Unemployment rates (U-3, U-6)
    - Labor force participation
    - Employment-population ratio

    Parameters:
    -----------
    start_date : str
        Start date in 'YYYY-MM-DD' format
    end_date : str
        End date in 'YYYY-MM-DD' format (defaults to today)
    cache_path : str
        Directory to cache downloaded data

    Returns:
    --------
    dict : Dictionary containing:
        - 'payrolls': Nonfarm payroll data
        - 'unemployment': Unemployment rates
        - 'labor_force': Labor force measures
        - 'combined': All employment measures in one DataFrame
    """
    if end_date is None:
        end_date = datetime.today().strftime('%Y-%m-%d')

    os.makedirs(cache_path, exist_ok=True)
    cache_dir = os.path.join(cache_path, f'employment_data_{start_date}_{end_date}')

    if os.path.exists(cache_dir):
        logger.info("Loading cached employment data from %s", cache_dir)
        return _load_cache(cache_dir)

    logger.info("Downloading employment data from FRED...")

    # Nonfarm Payrolls
    payroll_series = {
        'Nonfarm_Payrolls': 'PAYEMS',           # Total Nonfarm Payrolls (thousands)
        'Private_Payrolls': 'USPRIV',           # Private Sector Payrolls
        'Govt_Payrolls': 'USGOVT',              # Government Payrolls
        'Manufacturing_Employment': 'MANEMP',   # Manufacturing Employment
        'Service_Employment': 'SRVPRD',         # Service-Providing Industries
    }

    # Unemployment rates
    unemployment_series = {
        'Unemployment_Rate': 'UNRATE',          # U-3 Unemployment Rate
        'U6_Rate': 'U6RATE',                    # U-6 Unemployment Rate (broader)
        'Natural_Unemployment': 'NROU',         # Natural Rate of Unemployment
        'Long_Term_Unemployment': 'LNS13025703',  # 27+ weeks unemployed (%)
    }

    # Labor force measures
    labor_force_series = {
        'Labor_Force_Participation': 'CIVPART',  # Civilian Labor Force Participation
        'Employment_Population_Ratio': 'EMRATIO',  # Employment-Population Ratio
        'Labor_Force_Level': 'CLF16OV',         # Civilian Labor Force Level
        'Employed_Level': 'CE16OV',             # Civilian Employment Level
        'Unemployed_Level': 'UNEMPLOY',         # Unemployed Level
        'Prime_Age_LFPR': 'LNS11300060',        # Prime Age (25-54) LFPR
    }

    try:
        # Download payroll data
        logger.info("--- Nonfarm Payrolls ---")
        payroll_df = _download_fred_series(payroll_series, start_date, end_date)

        # Download unemployment rates
        logger.info("--- Unemployment Rates ---")
        unemployment_df = _download_fred_series(unemployment_series, start_date, end_date)

        # Download labor force measures
        logger.info("--- Labor Force Measures ---")
        labor_force_df = _download_fred_series(labor_force_series, start_date, end_date)

        # Combine all data
        combined_df = pd.concat([payroll_df, unemployment_df, labor_force_df], axis=1)

        result = {
            'payrolls': payroll_df,
            'unemployment': unemployment_df,
            'labor_force': labor_force_df,
            'combined': combined_df
        }

        _save_cache(result, cache_dir)
        logger.info("Cached employment data to %s", cache_dir)

        # Print summary
        logger.info("=" * 60)
        logger.info("=== Employment Data Summary ===")
        logger.info("=" * 60)
        logger.info("Date range: %s to %s", combined_df.index.min(), combined_df.index.max())
        logger.info("Observations: %d", len(combined_df))
        logger.info("Payroll series: %d", len(payroll_df.columns))
        logger.info("Unemployment series: %d", len(unemployment_df.columns))
        logger.info("Labor force series: %d", len(labor_force_df.columns))

        logger.info("Latest values:")
        for col in combined_df.columns:
            latest = combined_df[col].dropna().iloc[-1] if len(combined_df[col].dropna()) > 0 else 'N/A'
            if isinstance(latest, float):
                if 'Rate' in col or 'Ratio' in col or 'Participation' in col:
                    logger.info("  %s: %.1f%%", col, latest)
                elif 'Level' in col or 'Payrolls' in col or 'Employment' in col:
                    logger.info("  %s: %.0fK", col, latest)
                else:
                    logger.info("  %s: %.2f", col, latest)
            else:
                logger.info("  %s: %s", col, latest)

        return result

    except Exception as e:
        logger.error("downloading employment data: %s", e)
        return None


def load_jobless_claims_data(
    start_date='2000-01-01',
    end_date=None,
    cache_path='./data/fred'
):
    """
    Load unemployment insurance claims data from FRED.

    Provides initial and continuing claims data:
    - Initial Claims (weekly)
    - Continuing Claims (weekly)
    - Insured Unemployment Rate

    Parameters:
    -----------
    start_date : str
        Start date in 'YYYY-MM-DD' format
    end_date : str
        End date in 'YYYY-MM-DD' format (defaults to today)
    cache_path : str
        Directory to cache downloaded data

    Returns:
    --------
    dict : Dictionary containing:
        - 'claims': Initial and continuing claims
        - 'combined': All claims data
    """
    if end_date is None:
        end_date = datetime.today().strftime('%Y-%m-%d')

    os.makedirs(cache_path, exist_ok=True)
    cache_dir = os.path.join(cache_path, f'jobless_claims_{start_date}_{end_date}')

    if os.path.exists(cache_dir):
        logger.info("Loading cached jobless claims from %s", cache_dir)
        return _load_cache(cache_dir)

    logger.info("Downloading jobless claims data from FRED...")

    claims_series = {
        'Initial_Claims': 'ICSA',               # Initial Unemployment Claims
        'Continuing_Claims': 'CCSA',            # Continuing Claims
        'Initial_Claims_4WMA': 'IC4WSA',        # 4-Week Moving Average
        'Insured_Unemployment_Rate': 'IURSA',   # Insured Unemployment Rate
    }

    try:
        logger.info("--- Unemployment Insurance Claims ---")
        claims_df = _download_fred_series(claims_series, start_date, end_date)

        result = {
            'claims': claims_df,
            'combined': claims_df
        }

        _save_cache(result, cache_dir)
        logger.info("Cached jobless claims to %s", cache_dir)

        # Print summary
        logger.info("=" * 60)
        logger.info("=== Jobless Claims Summary ===")
        logger.info("=" * 60)
        logger.info("Date range: %s to %s", claims_df.index.min(), claims_df.index.max())
        logger.info("Observations: %d", len(claims_df))

        logger.info("Latest values:")
        for col in claims_df.columns:
            latest = claims_df[col].dropna().iloc[-1] if len(claims_df[col].dropna()) > 0 else 'N/A'
            if isinstance(latest, float):
                if 'Rate' in col:
                    logger.info("  %s: %.1f%%", col, latest)
                else:
                    logger.info("  %s: %.0f", col, latest)
            else:
                logger.info("  %s: %s", col, latest)

        return result

    except Exception as e:
        logger.error("downloading jobless claims: %s", e)
        return None


def load_wages_hours_data(
    start_date='2000-01-01',
    end_date=None,
    cache_path='./data/fred'
):
    """
    Load wages and hours worked data from FRED.

    Provides compensation and hours data:
    - Average Hourly Earnings
    - Average Weekly Hours
    - Unit Labor Costs
    - Employment Cost Index

    Parameters:
    -----------
    start_date : str
        Start date in 'YYYY-MM-DD' format
    end_date : str
        End date in 'YYYY-MM-DD' format (defaults to today)
    cache_path : str
        Directory to cache downloaded data

    Returns:
    --------
    dict : Dictionary containing:
        - 'wages': Wage and earnings data
        - 'hours': Hours worked data
        - 'combined': All wages/hours data
    """
    if end_date is None:
        end_date = datetime.today().strftime('%Y-%m-%d')

    os.makedirs(cache_path, exist_ok=True)
    cache_dir = os.path.join(cache_path, f'wages_hours_{start_date}_{end_date}')

    if os.path.exists(cache_dir):
        logger.info("Loading cached wages/hours data from %s", cache_dir)
        return _load_cache(cache_dir)

    logger.info("Downloading wages and hours data from FRED...")

    # Wages and earnings
    wages_series = {
        'Avg_Hourly_Earnings': 'CES0500000003',  # Average Hourly Earnings (Private)
        'Avg_Hourly_Earnings_Production': 'AHETPI',  # Production/Nonsupervisory
        'Avg_Weekly_Earnings': 'CES0500000011',  # Average Weekly Earnings
        'Employment_Cost_Index': 'ECIWAG',      # ECI: Wages and Salaries
        'Unit_Labor_Costs': 'ULCNFB',           # Unit Labor Costs (Nonfarm Business)
        'Compensation_Per_Hour': 'COMPNFB',     # Compensation Per Hour
    }

    # Hours worked
    hours_series = {
        'Avg_Weekly_Hours': 'AWHAETP',          # Average Weekly Hours (Private)
        'Avg_Weekly_Hours_Production': 'AWHI',  # Production/Nonsupervisory
        'Avg_Weekly_Hours_Manufacturing': 'AWHMANU',  # Manufacturing
        'Aggregate_Weekly_Hours': 'AWHI',       # Aggregate Weekly Hours Index
    }

    try:
        # Download wages data
        logger.info("--- Wages and Earnings ---")
        wages_df = _download_fred_series(wages_series, start_date, end_date)

        # Download hours data
        logger.info("--- Hours Worked ---")
        hours_df = _download_fred_series(hours_series, start_date, end_date)

        # Combine all data
        combined_df = pd.concat([wages_df, hours_df], axis=1)

        result = {
            'wages': wages_df,
            'hours': hours_df,
            'combined': combined_df
        }

        _save_cache(result, cache_dir)
        logger.info("Cached wages/hours data to %s", cache_dir)

        # Print summary
        logger.info("=" * 60)
        logger.info("=== Wages and Hours Summary ===")
        logger.info("=" * 60)
        logger.info("Date range: %s to %s", combined_df.index.min(), combined_df.index.max())
        logger.info("Observations: %d", len(combined_df))
        logger.info("Wages series: %d", len(wages_df.columns))
        logger.info("Hours series: %d", len(hours_df.columns))

        logger.info("Latest values:")
        for col in combined_df.columns:
            latest = combined_df[col].dropna().iloc[-1] if len(combined_df[col].dropna()) > 0 else 'N/A'
            if isinstance(latest, float):
                if 'Hourly' in col or 'Earnings' in col:
                    logger.info("  %s: $%.2f", col, latest)
                elif 'Hours' in col:
                    logger.info("  %s: %.1f hrs", col, latest)
                else:
                    logger.info("  %s: %.2f", col, latest)
            else:
                logger.info("  %s: %s", col, latest)

        return result

    except Exception as e:
        logger.error("downloading wages/hours data: %s", e)
        return None


def load_jolts_data(
    start_date='2000-01-01',
    end_date=None,
    cache_path='./data/fred'
):
    """
    Load JOLTS (Job Openings and Labor Turnover Survey) data from FRED.

    Provides job openings, hires, separations data:
    - Job Openings
    - Hires
    - Quits
    - Layoffs and Discharges

    Parameters:
    -----------
    start_date : str
        Start date in 'YYYY-MM-DD' format
    end_date : str
        End date in 'YYYY-MM-DD' format (defaults to today)
    cache_path : str
        Directory to cache downloaded data

    Returns:
    --------
    dict : Dictionary containing:
        - 'openings': Job openings data
        - 'turnover': Hires, quits, separations
        - 'combined': All JOLTS data
    """
    if end_date is None:
        end_date = datetime.today().strftime('%Y-%m-%d')

    os.makedirs(cache_path, exist_ok=True)
    cache_dir = os.path.join(cache_path, f'jolts_data_{start_date}_{end_date}')

    if os.path.exists(cache_dir):
        logger.info("Loading cached JOLTS data from %s", cache_dir)
        return _load_cache(cache_dir)

    logger.info("Downloading JOLTS data from FRED...")

    # Job openings
    openings_series = {
        'Job_Openings': 'JTSJOL',               # Job Openings Level (thousands)
        'Job_Openings_Rate': 'JTSJOR',          # Job Openings Rate
    }

    # Labor turnover
    turnover_series = {
        'Hires': 'JTSHIL',                      # Hires Level
        'Hires_Rate': 'JTSHIR',                 # Hires Rate
        'Quits': 'JTSQUL',                      # Quits Level
        'Quits_Rate': 'JTSQUR',                 # Quits Rate
        'Total_Separations': 'JTSTSL',          # Total Separations Level
        'Total_Separations_Rate': 'JTSTSR',     # Total Separations Rate
        'Layoffs_Discharges': 'JTSLDL',         # Layoffs and Discharges Level
    }

    try:
        # Download job openings
        logger.info("--- Job Openings ---")
        openings_df = _download_fred_series(openings_series, start_date, end_date)

        # Download turnover data
        logger.info("--- Labor Turnover ---")
        turnover_df = _download_fred_series(turnover_series, start_date, end_date)

        # Combine all data
        combined_df = pd.concat([openings_df, turnover_df], axis=1)

        result = {
            'openings': openings_df,
            'turnover': turnover_df,
            'combined': combined_df
        }

        _save_cache(result, cache_dir)
        logger.info("Cached JOLTS data to %s", cache_dir)

        # Print summary
        logger.info("=" * 60)
        logger.info("=== JOLTS Summary ===")
        logger.info("=" * 60)
        logger.info("Date range: %s to %s", combined_df.index.min(), combined_df.index.max())
        logger.info("Observations: %d", len(combined_df))
        logger.info("Openings series: %d", len(openings_df.columns))
        logger.info("Turnover series: %d", len(turnover_df.columns))

        logger.info("Latest values:")
        for col in combined_df.columns:
            latest = combined_df[col].dropna().iloc[-1] if len(combined_df[col].dropna()) > 0 else 'N/A'
            if isinstance(latest, float):
                if 'Rate' in col:
                    logger.info("  %s: %.1f%%", col, latest)
                else:
                    logger.info("  %s: %.0fK", col, latest)
            else:
                logger.info("  %s: %s", col, latest)

        return result

    except Exception as e:
        logger.error("downloading JOLTS data: %s", e)
        return None


def load_comprehensive_employment_data(
    start_date='2000-01-01',
    end_date=None,
    cache_path='./data/fred',
    force_refresh=False
):
    """
    Load comprehensive employment dataset from FRED (2000-2025).

    This function aggregates:
    - Employment levels and payrolls
    - Unemployment rates (U-3, U-6)
    - Labor force participation
    - Jobless claims (initial, continuing)
    - Wages and hours worked
    - JOLTS data (job openings, hires, quits)

    Parameters:
    -----------
    start_date : str
        Start date in 'YYYY-MM-DD' format
    end_date : str
        End date in 'YYYY-MM-DD' format (defaults to today)
    cache_path : str
        Directory to cache downloaded data

    Returns:
    --------
    dict : Dictionary containing:
        - 'employment': Employment and unemployment data
        - 'claims': Jobless claims data
        - 'wages_hours': Wages and hours data
        - 'jolts': JOLTS data
        - 'combined': All employment measures merged on date
        - 'summary_stats': Summary statistics for all series
    """
    if end_date is None:
        end_date = datetime.today().strftime('%Y-%m-%d')

    os.makedirs(cache_path, exist_ok=True)
    cache_dir = os.path.join(
        cache_path,
        f'comprehensive_employment_{start_date}_{end_date}'
    )

    if force_refresh and os.path.exists(cache_dir):
        shutil.rmtree(cache_dir)
        logger.info("Force refresh: removed %s", cache_dir)

    if os.path.exists(cache_dir):
        logger.info("Loading cached comprehensive employment data from %s", cache_dir)
        return _load_cache(cache_dir)
    logger.info("=" * 60)
    logger.info("Loading Comprehensive Employment Data (2000-2025)")
    logger.info("=" * 60)

    # Load employment data
    logger.info("[1/4] Loading employment and unemployment data...")
    employment_data = load_employment_data(start_date, end_date, cache_path)

    # Load jobless claims
    logger.info("[2/4] Loading jobless claims...")
    claims_data = load_jobless_claims_data(start_date, end_date, cache_path)

    # Load wages and hours
    logger.info("[3/4] Loading wages and hours...")
    wages_hours_data = load_wages_hours_data(start_date, end_date, cache_path)

    # Load JOLTS data
    logger.info("[4/4] Loading JOLTS data...")
    jolts_data = load_jolts_data(start_date, end_date, cache_path)

    # Combine all data
    combined_dfs = []

    if employment_data and 'combined' in employment_data:
        combined_dfs.append(employment_data['combined'])

    if claims_data and 'combined' in claims_data:
        combined_dfs.append(claims_data['combined'])

    if wages_hours_data and 'combined' in wages_hours_data:
        combined_dfs.append(wages_hours_data['combined'])

    if jolts_data and 'combined' in jolts_data:
        combined_dfs.append(jolts_data['combined'])

    if combined_dfs:
        combined_df = pd.concat(combined_dfs, axis=1)
        # Remove duplicate columns if any
        combined_df = combined_df.loc[:, ~combined_df.columns.duplicated()]
    else:
        combined_df = pd.DataFrame()

    # Calculate summary statistics
    summary_stats = {}
    if len(combined_df) > 0:
        for col in combined_df.columns:
            series = combined_df[col].dropna()
            if len(series) > 0:
                summary_stats[col] = {
                    'count': len(series),
                    'mean': series.mean(),
                    'std': series.std(),
                    'min': series.min(),
                    'max': series.max(),
                    'latest': series.iloc[-1],
                    'start_date': series.index.min(),
                    'end_date': series.index.max()
                }

    result = {
        'employment': employment_data,
        'claims': claims_data,
        'wages_hours': wages_hours_data,
        'jolts': jolts_data,
        'combined': combined_df,
        'summary_stats': summary_stats
    }

    _save_cache(result, cache_dir)
    logger.info("Cached comprehensive employment data to %s", cache_dir)

    # Print final summary
    logger.info("=" * 60)
    logger.info("=== Comprehensive Employment Summary ===")
    logger.info("=" * 60)
    logger.info("Total series loaded: %d", len(combined_df.columns))
    logger.info("Date range: %s to %s", combined_df.index.min(), combined_df.index.max())
    logger.info("Total observations: %d", len(combined_df))

    logger.info("--- Series Categories ---")
    if employment_data:
        logger.info("Employment/Unemployment: %d series", len(employment_data['combined'].columns))
    if claims_data:
        logger.info("Jobless claims: %d series", len(claims_data['combined'].columns))
    if wages_hours_data:
        logger.info("Wages/Hours: %d series", len(wages_hours_data['combined'].columns))
    if jolts_data:
        logger.info("JOLTS: %d series", len(jolts_data['combined'].columns))

    # Key metrics
    if employment_data and 'unemployment' in employment_data:
        unemp_df = employment_data['unemployment']
        if 'Unemployment_Rate' in unemp_df.columns:
            latest_unemp = unemp_df['Unemployment_Rate'].dropna().iloc[-1]
            logger.info("Latest Unemployment Rate: %.1f%%", latest_unemp)

    if employment_data and 'payrolls' in employment_data:
        payroll_df = employment_data['payrolls']
        if 'Nonfarm_Payrolls' in payroll_df.columns:
            latest_payrolls = payroll_df['Nonfarm_Payrolls'].dropna().iloc[-1]
            logger.info("Latest Nonfarm Payrolls: %.0fK", latest_payrolls)

    if jolts_data and 'openings' in jolts_data:
        open_df = jolts_data['openings']
        if 'Job_Openings' in open_df.columns:
            latest_openings = open_df['Job_Openings'].dropna().iloc[-1]
            logger.info("Latest Job Openings: %.0fK", latest_openings)

    logger.info("=" * 60)
    logger.info("Citation:")
    logger.info("U.S. Bureau of Labor Statistics (BLS)")
    logger.info("Federal Reserve Economic Data (FRED), Federal Reserve Bank of St. Louis")
    logger.info("https://fred.stlouisfed.org/")
    logger.info("=" * 60)

    return result


def load_additional_macro_data(
    start_date='2000-01-01',
    end_date=None,
    cache_path='./data/fred',
    force_refresh=False
):
    """
    Load additional macroeconomic indicators from FRED.

    Returns unemployment, GDP growth, interest rates, etc.
    """
    if end_date is None:
        end_date = datetime.today().strftime('%Y-%m-%d')

    os.makedirs(cache_path, exist_ok=True)
    cache_dir = os.path.join(cache_path, f'macro_data_{start_date}_{end_date}')

    if force_refresh and os.path.exists(cache_dir):
        shutil.rmtree(cache_dir)
        logger.info("Force refresh: removed %s", cache_dir)

    if os.path.exists(cache_dir):
        logger.info("Loading cached macro data from %s", cache_dir)
        return _load_cache(cache_dir)

    logger.info("Downloading macro data from FRED...")

    series = {
        'Unemployment_Rate': 'UNRATE',
        'Labor_Force_Participation': 'CIVPART',
        'Fed_Funds_Rate': 'FEDFUNDS',
        'T10Y_Rate': 'DGS10',
        'T2Y_Rate': 'DGS2',
        'T3M_Rate': 'DGS3MO',
        'Real_GDP': 'GDPC1',
        'GDP_Growth': 'A191RL1Q225SBEA',
        'M2': 'M2SL',
        'Housing_Starts': 'HOUST',
        'Home_Price_Index': 'CSUSHPISA',
        'Consumer_Sentiment': 'UMCSENT',
        'Dollar_Index': 'DTWEXBGS',
    }

    try:
        macro_df = _download_fred_series(series, start_date, end_date)

        _save_cache(macro_df, cache_dir)
        logger.info("Cached macro data to %s", cache_dir)

        logger.info("=== Macro Data Summary ===")
        logger.info("Date range: %s to %s", macro_df.index.min(), macro_df.index.max())
        logger.info("Series downloaded: %d", len(macro_df.columns))
        logger.info("Latest values:\n%s", macro_df.iloc[-1])

        return macro_df

    except Exception as e:
        logger.error("downloading macro data: %s", e)
        return None


def get_inflation_regime(inflation_data, threshold_low=2.0, threshold_high=4.0):
    """
    Classify inflation regimes (low, moderate, high) based on Core PCE.

    Parameters:
    -----------
    inflation_data : dict
        Output from load_inflation_data()
    threshold_low : float
        Threshold for low inflation (%)
    threshold_high : float
        Threshold for high inflation (%)

    Returns:
    --------
    pd.Series : Inflation regime classification
    """
    core_pce = inflation_data['yoy']['Core_PCE_YoY']

    regime = pd.Series(index=core_pce.index, dtype='object')
    regime[core_pce < threshold_low] = 'Low Inflation'
    regime[(core_pce >= threshold_low) & (core_pce < threshold_high)] = 'Moderate Inflation'
    regime[core_pce >= threshold_high] = 'High Inflation'

    return regime

