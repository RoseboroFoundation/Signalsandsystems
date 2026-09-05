"""
ETL (Extract, Transform, Load) pipeline for Signals & Systems research.

Loads all datasets defined in clean.py's data dictionary into a unified
data store, with options for selective loading, caching, and status reporting.

Usage:
    # Load everything
    from ETL import run_etl
    data = run_etl()

    # Load specific categories
    data = run_etl(categories=['inflation', 'rates', 'employment'])

    # Force refresh (bypass cache)
    data = run_etl(force_refresh=True)

    # Quick summary of what's loaded
    from ETL import summarize
    summarize(data)
"""

import logging
import time
from datetime import datetime

from clean import (
    import_culture_war_data,
    get_stock_data,
    download_vix_data,
    download_fama_french_factors,
    download_industry_portfolios,
    load_culture_war_companies,
    build_control_companies,
    Form4Downloader,
    SECFilingDownloader,
    PartyPlatformDownloader,
    load_political_events,
    load_political_exposure,
    load_inflation_data,
    load_inflation_expectations_data,
    load_comprehensive_inflation_data,
    load_treasury_yields,
    load_policy_rates,
    load_credit_spreads,
    load_comprehensive_rates_data,
    load_industrial_production_data,
    load_ip_growth_rates,
    load_comprehensive_ip_data,
    load_money_supply_data,
    load_money_velocity_data,
    load_fed_balance_sheet_data,
    load_comprehensive_m2_data,
    load_gdp_data,
    load_gdp_components_data,
    load_gdp_by_industry_data,
    load_comprehensive_gdp_data,
    load_employment_data,
    load_jobless_claims_data,
    load_wages_hours_data,
    load_jolts_data,
    load_comprehensive_employment_data,
    load_additional_macro_data,
    load_news_data,
    clean_all_data,
)

logger = logging.getLogger(__name__)

# =============================================================================
# DATA DICTIONARY
# =============================================================================
# Maps each dataset key to its loader function, arguments, dependencies,
# and the category it belongs to.

START_DATE = '2000-01-01'
END_DATE = '2025-12-31'
CACHE_PATH = './data/fred'


# ---------------------------------------------------------------------------
# Wrapper loaders for dependency-injected and class-based downloaders.
# Every DATA_DICTIONARY entry has a callable 'loader' — no string sentinels,
# no key-name dispatch.  Wrappers that need other datasets receive `data_dict`
# as first arg; _load_single dispatches via _DATA_DICT_LOADERS.
# ---------------------------------------------------------------------------

def _load_stockdata(data_dict, start_date=START_DATE, end_date=END_DATE, **_kw):
    """Load stock prices for culture war tickers from Yahoo Finance."""
    cw = data_dict.get('culturewardata')
    if cw is None:
        logger.warning("Skipping stockdata: culturewardata not loaded")
        return None
    # Use load_culture_war_companies to get both treatment and control tickers,
    # matching the ticker universe used by _load_form4 and _load_sec_fundamentals.
    tickers = load_culture_war_companies(cw)
    return get_stock_data(tickers, start_date=start_date, end_date=end_date)


def _load_controlcompanies(data_dict, **_kw):
    """Build control companies table from culture war dataset."""
    cw = data_dict.get('culturewardata')
    if cw is None:
        logger.warning("Skipping controlcompanies: culturewardata not loaded")
        return None
    return build_control_companies(cw)


def _load_form4(data_dict, start_date=START_DATE, end_date=END_DATE, **_kw):
    """Load SEC Form 4 insider trading data for culture war tickers."""
    cw = data_dict.get('culturewardata')
    if cw is None:
        logger.warning("Skipping form4data: culturewardata not loaded")
        return None
    tickers = load_culture_war_companies(cw)
    return Form4Downloader().build_form4_dataset(
        tickers, start_date=start_date, end_date=end_date, save_csv=True,
    )


def _load_sec_fundamentals(data_dict, start_date=START_DATE, end_date=END_DATE, **_kw):
    """Load SEC 10-K/10-Q fundamentals for culture war tickers."""
    cw = data_dict.get('culturewardata')
    if cw is None:
        logger.warning("Skipping sec_fundamentals: culturewardata not loaded")
        return None
    tickers = load_culture_war_companies(cw)
    return SECFilingDownloader().build_fundamentals_dataset(
        tickers, start_date=start_date, end_date=end_date, save_csv=True,
    )


def _load_party_platforms(**_kw):
    """Load Republican & Democratic party platforms (2000-2024)."""
    return PartyPlatformDownloader().download_all_platforms(save_csv=True)


def _load_political_events(start_date=START_DATE, end_date=END_DATE, **_kw):
    """Load political fundamental events (congressional votes, EOs, court decisions)."""
    return load_political_events(
        start_date=start_date, end_date=end_date,
        cache_dir='./data/political_events',
    )


def _load_political_exposure(data_dict, start_date=START_DATE, end_date=END_DATE, **_kw):
    """Load political exposure data (lobbying, PAC contributions)."""
    cw = data_dict.get('culturewardata')
    tickers = load_culture_war_companies(cw) if cw is not None else None
    return load_political_exposure(
        tickers=tickers, start_date=start_date, end_date=end_date,
        cache_dir='./data/political_exposure',
    )


# Loaders that receive data_dict as first arg (dependency-injected)
_DATA_DICT_LOADERS = {_load_stockdata, _load_controlcompanies,
                      _load_form4, _load_sec_fundamentals,
                      _load_political_exposure}

# Loaders that take only kwargs (no data_dict, no positional args)
_KWARGS_ONLY_LOADERS = {_load_party_platforms, _load_political_events}


DATA_DICTIONARY = {
    # --- Culture War & Market Data ---
    # args convention: () = no positional args; dependency-injected loaders
    # receive their args in _load_single via the depends_on mechanism.
    'culturewardata': {
        'loader': import_culture_war_data,
        'args': ('Culture_War_Companies_160_fullmeta.csv',),
        'kwargs': {},
        'category': 'market',
        'description': 'Culture war companies events dataset',
        'depends_on': [],
    },
    'stockdata': {
        'loader': _load_stockdata,
        'args': (),
        'kwargs': {'start_date': START_DATE, 'end_date': END_DATE},
        'category': 'market',
        'description': 'Historical stock prices from Yahoo Finance',
        'depends_on': ['culturewardata'],
    },
    'vixdata': {
        'loader': download_vix_data,
        'args': (),
        'kwargs': {},
        'category': 'market',
        'description': 'CBOE Volatility Index (VIX) from FRED',
        'depends_on': [],
    },
    'ff_factors': {
        'loader': download_fama_french_factors,
        'args': (),
        'kwargs': {
            'start_date': START_DATE,
            'frequency': 'daily',
            'output_dir': './fama_french_data',
        },
        'category': 'market',
        'description': 'Fama-French 3-factor, 5-factor, and Momentum',
        'depends_on': [],
    },
    'controlcompanies': {
        'loader': _load_controlcompanies,
        'args': (),
        'kwargs': {},
        'category': 'market',
        'description': 'Control companies matched to culture war treatment firms',
        'depends_on': ['culturewardata'],
    },
    'industry_portfolios': {
        'loader': download_industry_portfolios,
        'args': (),
        'kwargs': {
            'num_industries': 10,
            'start_date': START_DATE,
            'frequency': 'daily',
            'output_dir': './fama_french_data',
        },
        'category': 'market',
        'description': 'Fama-French industry portfolio returns',
        'depends_on': [],
    },
    'newsdata': {
        'loader': load_news_data,
        'args': (),
        'kwargs': {
            'cache_file': './news_data/culture_war_news_2000_2025_final.csv',
            'refresh': False,
        },
        'category': 'market',
        'description': 'News articles from Guardian, NYT, Reddit',
        'depends_on': [],
    },

    # --- SEC Filings ---
    'form4data': {
        'loader': _load_form4,
        'args': (),
        'kwargs': {'start_date': START_DATE, 'end_date': END_DATE},
        'category': 'sec',
        'description': 'SEC Form 4 insider trading filings',
        'depends_on': ['culturewardata'],
    },
    'sec_fundamentals': {
        'loader': _load_sec_fundamentals,
        'args': (),
        'kwargs': {'start_date': START_DATE, 'end_date': END_DATE},
        'category': 'sec',
        'description': 'SEC 10-K/10-Q financial fundamentals (XBRL)',
        'depends_on': ['culturewardata'],
    },

    # --- Political ---
    'party_platforms': {
        'loader': _load_party_platforms,
        'args': (),
        'kwargs': {},
        'category': 'political',
        'description': 'Republican & Democratic party platforms (2000-2024)',
        'depends_on': [],
    },
    'political_events': {
        'loader': _load_political_events,
        'args': (),
        'kwargs': {'start_date': START_DATE, 'end_date': END_DATE},
        'category': 'political',
        'description': 'Political fundamental events (congressional votes, executive orders, court decisions)',
        'depends_on': [],
    },
    'political_exposure': {
        'loader': _load_political_exposure,
        'args': (),
        'kwargs': {'start_date': START_DATE, 'end_date': END_DATE},
        'category': 'political',
        'description': 'Firm-level political exposure (lobbying, PAC contributions)',
        'depends_on': ['culturewardata'],
    },

    # --- Inflation ---
    'inflationdata': {
        'loader': load_inflation_data,
        'args': (),
        'kwargs': {'start_date': START_DATE, 'end_date': END_DATE, 'cache_path': CACHE_PATH},
        'category': 'inflation',
        'description': 'Core inflation measures (CPI, PCE, PPI, GDP Deflator)',
        'depends_on': [],
    },
    'inflation_expectations': {
        'loader': load_inflation_expectations_data,
        'args': (),
        'kwargs': {'start_date': START_DATE, 'end_date': END_DATE, 'cache_path': CACHE_PATH},
        'category': 'inflation',
        'description': 'Breakeven inflation, survey expectations, Fed measures',
        'depends_on': [],
    },
    'comprehensive_inflation': {
        'loader': load_comprehensive_inflation_data,
        'args': (),
        'kwargs': {'start_date': START_DATE, 'end_date': END_DATE, 'cache_path': CACHE_PATH},
        'category': 'inflation',
        'description': 'All inflation measures combined with component-level CPI',
        'depends_on': [],
    },

    # --- Interest Rates ---
    'treasury_yields': {
        'loader': load_treasury_yields,
        'args': (),
        'kwargs': {'start_date': START_DATE, 'end_date': END_DATE, 'cache_path': CACHE_PATH},
        'category': 'rates',
        'description': 'Treasury yield curve (1M-30Y) and TIPS real yields',
        'depends_on': [],
    },
    'policy_rates': {
        'loader': load_policy_rates,
        'args': (),
        'kwargs': {'start_date': START_DATE, 'end_date': END_DATE, 'cache_path': CACHE_PATH},
        'category': 'rates',
        'description': 'Fed Funds, SOFR, Prime, Discount rates',
        'depends_on': [],
    },
    'credit_spreads': {
        'loader': load_credit_spreads,
        'args': (),
        'kwargs': {'start_date': START_DATE, 'end_date': END_DATE, 'cache_path': CACHE_PATH},
        'category': 'rates',
        'description': 'Corporate yields, credit spreads, mortgage rates',
        'depends_on': [],
    },
    'comprehensive_rates': {
        'loader': load_comprehensive_rates_data,
        'args': (),
        'kwargs': {'start_date': START_DATE, 'end_date': END_DATE, 'cache_path': CACHE_PATH},
        'category': 'rates',
        'description': 'All rates with yield curve metrics (slope, curvature, inversion)',
        'depends_on': [],
    },

    # --- Industrial Production ---
    'industrial_production': {
        'loader': load_industrial_production_data,
        'args': (),
        'kwargs': {'start_date': START_DATE, 'end_date': END_DATE, 'cache_path': CACHE_PATH},
        'category': 'production',
        'description': 'IP indices, sector production, capacity utilization',
        'depends_on': [],
    },
    'ip_growth': {
        'loader': load_ip_growth_rates,
        'args': (),
        'kwargs': {'start_date': START_DATE, 'end_date': END_DATE, 'cache_path': CACHE_PATH},
        'category': 'production',
        'description': 'IP growth rates (YoY, MoM) and diffusion indices',
        'depends_on': [],
    },
    'comprehensive_ip': {
        'loader': load_comprehensive_ip_data,
        'args': (),
        'kwargs': {'start_date': START_DATE, 'end_date': END_DATE, 'cache_path': CACHE_PATH},
        'category': 'production',
        'description': 'All industrial production measures combined',
        'depends_on': [],
    },

    # --- Money Supply ---
    'money_supply': {
        'loader': load_money_supply_data,
        'args': (),
        'kwargs': {'start_date': START_DATE, 'end_date': END_DATE, 'cache_path': CACHE_PATH},
        'category': 'money',
        'description': 'M1, M2, monetary base, and components',
        'depends_on': [],
    },
    'money_velocity': {
        'loader': load_money_velocity_data,
        'args': (),
        'kwargs': {'start_date': START_DATE, 'end_date': END_DATE, 'cache_path': CACHE_PATH},
        'category': 'money',
        'description': 'M1 and M2 velocity of money',
        'depends_on': [],
    },
    'fed_balance_sheet': {
        'loader': load_fed_balance_sheet_data,
        'args': (),
        'kwargs': {'start_date': START_DATE, 'end_date': END_DATE, 'cache_path': CACHE_PATH},
        'category': 'money',
        'description': 'Fed total assets, Treasury/MBS holdings, reserves',
        'depends_on': [],
    },
    'comprehensive_m2': {
        'loader': load_comprehensive_m2_data,
        'args': (),
        'kwargs': {'start_date': START_DATE, 'end_date': END_DATE, 'cache_path': CACHE_PATH},
        'category': 'money',
        'description': 'All money supply measures with growth rates',
        'depends_on': [],
    },

    # --- GDP ---
    'gdp_data': {
        'loader': load_gdp_data,
        'args': (),
        'kwargs': {'start_date': START_DATE, 'end_date': END_DATE, 'cache_path': CACHE_PATH},
        'category': 'gdp',
        'description': 'Nominal/Real GDP, growth rates, per capita',
        'depends_on': [],
    },
    'gdp_components': {
        'loader': load_gdp_components_data,
        'args': (),
        'kwargs': {'start_date': START_DATE, 'end_date': END_DATE, 'cache_path': CACHE_PATH},
        'category': 'gdp',
        'description': 'GDP = C + I + G + (X-M) expenditure components',
        'depends_on': [],
    },
    'gdp_industry': {
        'loader': load_gdp_by_industry_data,
        'args': (),
        'kwargs': {'start_date': START_DATE, 'end_date': END_DATE, 'cache_path': CACHE_PATH},
        'category': 'gdp',
        'description': 'GDP by industry/sector (value added)',
        'depends_on': [],
    },
    'comprehensive_gdp': {
        'loader': load_comprehensive_gdp_data,
        'args': (),
        'kwargs': {'start_date': START_DATE, 'end_date': END_DATE, 'cache_path': CACHE_PATH},
        'category': 'gdp',
        'description': 'All GDP measures combined',
        'depends_on': [],
    },

    # --- Employment ---
    'employment_data': {
        'loader': load_employment_data,
        'args': (),
        'kwargs': {'start_date': START_DATE, 'end_date': END_DATE, 'cache_path': CACHE_PATH},
        'category': 'employment',
        'description': 'Payrolls, unemployment rates, labor force participation',
        'depends_on': [],
    },
    'jobless_claims': {
        'loader': load_jobless_claims_data,
        'args': (),
        'kwargs': {'start_date': START_DATE, 'end_date': END_DATE, 'cache_path': CACHE_PATH},
        'category': 'employment',
        'description': 'Initial/continuing claims, insured unemployment rate',
        'depends_on': [],
    },
    'wages_hours': {
        'loader': load_wages_hours_data,
        'args': (),
        'kwargs': {'start_date': START_DATE, 'end_date': END_DATE, 'cache_path': CACHE_PATH},
        'category': 'employment',
        'description': 'Average earnings, weekly hours, ECI, unit labor costs',
        'depends_on': [],
    },
    'jolts_data': {
        'loader': load_jolts_data,
        'args': (),
        'kwargs': {'start_date': START_DATE, 'end_date': END_DATE, 'cache_path': CACHE_PATH},
        'category': 'employment',
        'description': 'Job openings, hires, quits, layoffs (JOLTS)',
        'depends_on': [],
    },
    'comprehensive_employment': {
        'loader': load_comprehensive_employment_data,
        'args': (),
        'kwargs': {'start_date': START_DATE, 'end_date': END_DATE, 'cache_path': CACHE_PATH},
        'category': 'employment',
        'description': 'All employment measures combined',
        'depends_on': [],
    },

    # --- Additional Macro ---
    'additional_macro': {
        'loader': load_additional_macro_data,
        'args': (),
        'kwargs': {'start_date': START_DATE, 'end_date': END_DATE, 'cache_path': CACHE_PATH},
        'category': 'macro',
        'description': 'Consumer Sentiment, Housing Starts, Home Prices, Dollar Index',
        'depends_on': [],
    },
}

# Category descriptions for reporting
CATEGORIES = {
    'market': 'Market Data (stocks, VIX, Fama-French, industry portfolios, news)',
    'sec': 'SEC Filings (Form 4 insider trading, 10-K/10-Q fundamentals)',
    'political': 'Political (party platforms, events, lobbying exposure)',
    'inflation': 'Inflation (CPI, PCE, PPI, expectations, components)',
    'rates': 'Interest Rates (Treasury curve, policy rates, credit spreads)',
    'production': 'Industrial Production (IP indices, capacity utilization)',
    'money': 'Money Supply (M1, M2, velocity, Fed balance sheet)',
    'gdp': 'GDP (headline, components, by industry)',
    'employment': 'Employment (payrolls, unemployment, JOLTS, wages)',
    'macro': 'Additional Macro (sentiment, housing, dollar)',
}

ALL_CATEGORIES = list(CATEGORIES.keys())


# =============================================================================
# ETL PIPELINE
# =============================================================================
def run_etl(
    categories=None,
    keys=None,
    start_date=None,
    end_date=None,
    force_refresh=False,
    clean=True,
    verbose=True,
):
    """
    Run the ETL pipeline to load datasets from clean.py's data dictionary.

    Parameters
    ----------
    categories : list[str], optional
        Which categories to load. Options: 'market', 'sec', 'political',
        'inflation', 'rates', 'production', 'money', 'gdp', 'employment',
        'macro'. Defaults to all categories.
    keys : list[str], optional
        Specific dataset keys to load (overrides categories).
        E.g. ['inflationdata', 'treasury_yields', 'employment_data']
    start_date : str, optional
        Override default start date (2000-01-01).
    end_date : str, optional
        Override default end date (2025-12-31).
    force_refresh : bool
        If True, bypass cache and re-download from sources.
    clean : bool
        If True, apply cleaning (ffill, interpolation) after loading.
    verbose : bool
        If True, print progress and summary.

    Returns
    -------
    dict
        Dictionary of loaded (and optionally cleaned) datasets.
    """
    if categories is None and keys is None:
        categories = ALL_CATEGORIES

    # Determine which keys to load
    if keys is not None:
        load_keys = [k for k in keys if k in DATA_DICTIONARY]
        invalid = set(keys) - set(DATA_DICTIONARY.keys())
        if invalid:
            logger.warning("Unknown keys (skipped): %s", invalid)
    else:
        load_keys = [
            k for k, v in DATA_DICTIONARY.items()
            if v['category'] in categories
        ]

    # Resolve dependencies: add any required keys not already in list
    resolved = _resolve_dependencies(load_keys)

    extra = [k for k in resolved if k not in load_keys]
    if extra and verbose:
        logger.info("  Auto-added dependencies: %s", extra)

    if verbose:
        logger.info("=" * 60)
        logger.info("  Signals & Systems ETL Pipeline")
        logger.info("=" * 60)
        logger.info("  Datasets to load: %d", len(resolved))
        if categories and keys is None:
            logger.info("  Categories: %s", ', '.join(categories))
        logger.info("  Date range: %s to %s",
                     start_date or START_DATE,
                     end_date or END_DATE)
        logger.info("  Force refresh: %s", force_refresh)
        logger.info("  Clean data: %s", clean)
        logger.info("=" * 60)

    data_dict = {}
    timings = {}
    errors = {}

    for key in resolved:
        entry = DATA_DICTIONARY[key]

        if verbose:
            logger.info("[LOADING] %s - %s", key, entry['description'])

        t0 = time.time()
        try:
            result = _load_single(key, entry, data_dict, start_date, end_date, force_refresh)
            data_dict[key] = result
            elapsed = time.time() - t0
            timings[key] = elapsed

            if verbose:
                status = "OK" if result is not None else "EMPTY"
                logger.info("  [%s] %s (%.1fs)", status, key, elapsed)
        except Exception as e:
            data_dict[key] = None
            elapsed = time.time() - t0
            timings[key] = elapsed
            errors[key] = str(e)
            logger.error("  [FAIL] %s: %s (%.1fs)", key, e, elapsed)

    # Apply cleaning
    if clean:
        if verbose:
            logger.info("Applying data cleaning...")
        data_dict = clean_all_data(data_dict, verbose=verbose)

    # Print summary
    if verbose:
        _print_summary(data_dict, timings, errors)

    return data_dict


def _resolve_dependencies(load_keys):
    """Ensure dependencies are loaded before dependents, preserving order."""
    resolved = []
    seen = set()

    def _add(key):
        if key in seen:
            return
        entry = DATA_DICTIONARY.get(key)
        if entry is None:
            return
        for dep in entry.get('depends_on', []):
            _add(dep)
        seen.add(key)
        resolved.append(key)

    for key in load_keys:
        _add(key)

    return resolved


def _load_single(key, entry, data_dict, start_date, end_date, force_refresh):
    """Load a single dataset, handling dependencies and argument overrides."""
    loader = entry['loader']
    args = entry.get('args') or ()
    kwargs = dict(entry.get('kwargs') or {})

    # Override date range if specified
    if start_date and 'start_date' in kwargs:
        kwargs['start_date'] = start_date
    if end_date and 'end_date' in kwargs:
        kwargs['end_date'] = end_date

    # Pass force_refresh if the loader supports it
    if force_refresh:
        if 'force_refresh' in _get_param_names(loader):
            kwargs['force_refresh'] = force_refresh
        else:
            logger.debug("force_refresh has no effect on '%s' (loader does not support it)", key)

    # Dependency-injected loaders: receive data_dict as first arg
    if loader in _DATA_DICT_LOADERS:
        return loader(data_dict, **kwargs)

    # Kwargs-only loaders (no positional args, no data_dict)
    if loader in _KWARGS_ONLY_LOADERS:
        return loader(**kwargs)

    # Standard loaders: called with positional args + kwargs
    return loader(*args, **kwargs)


def _get_param_names(func):
    """Get parameter names from a function signature."""
    import inspect
    try:
        return list(inspect.signature(func).parameters.keys())
    except (ValueError, TypeError):
        return []


def _print_summary(data_dict, timings, errors):
    """Print a summary report of the ETL results."""
    total_time = sum(timings.values())

    logger.info("")
    logger.info("=" * 60)
    logger.info("  ETL Summary")
    logger.info("=" * 60)

    loaded = {k: v for k, v in data_dict.items() if v is not None}
    failed = {k: v for k, v in data_dict.items() if v is None}

    logger.info("  Loaded:  %d datasets", len(loaded))
    logger.info("  Failed:  %d datasets", len(failed))
    logger.info("  Total time: %.1fs", total_time)
    logger.info("")

    # Group by category
    for cat_key, cat_desc in CATEGORIES.items():
        cat_keys = [k for k, v in DATA_DICTIONARY.items()
                    if v['category'] == cat_key and k in data_dict]
        if not cat_keys:
            continue

        logger.info("  --- %s ---", cat_desc)
        for key in cat_keys:
            val = data_dict[key]
            t = timings.get(key, 0)
            if val is None:
                err = errors.get(key, 'unknown')
                logger.info("    %-30s FAILED (%s)", key, err)
            else:
                shape = _get_shape(val)
                logger.info("    %-30s %s  (%.1fs)", key, shape, t)
        logger.info("")

    if errors:
        logger.info("  --- Errors ---")
        for key, err in errors.items():
            logger.info("    %s: %s", key, err)

    logger.info("=" * 60)


def _get_shape(data):
    """Get a human-readable shape description for a dataset."""
    import pandas as pd

    if isinstance(data, pd.DataFrame):
        return f"{data.shape[0]:,} rows x {data.shape[1]} cols"
    elif isinstance(data, dict):
        # Count total series across sub-DataFrames
        n_keys = len(data)
        combined = data.get('combined')
        if combined is not None and isinstance(combined, pd.DataFrame):
            return f"dict({n_keys} keys, combined: {combined.shape[0]:,} x {combined.shape[1]})"
        return f"dict({n_keys} keys)"
    elif data is None:
        return "None"
    else:
        return str(type(data).__name__)


# =============================================================================
# CONVENIENCE FUNCTIONS
# =============================================================================
def summarize(data_dict):
    """Print a quick summary of a loaded data dictionary."""
    import pandas as pd

    print("\n" + "=" * 70)
    print("  Dataset Summary")
    print("=" * 70)

    for key, val in data_dict.items():
        entry = DATA_DICTIONARY.get(key, {})
        desc = entry.get('description', '')
        cat = entry.get('category', '?')

        if val is None:
            print(f"  [{cat:>10}] {key:<30} -- not loaded --")
        elif isinstance(val, pd.DataFrame):
            print(f"  [{cat:>10}] {key:<30} {val.shape[0]:>6,} rows x {val.shape[1]:>3} cols")
        elif isinstance(val, dict):
            combined = val.get('combined')
            if combined is not None and isinstance(combined, pd.DataFrame):
                print(f"  [{cat:>10}] {key:<30} {combined.shape[0]:>6,} rows x {combined.shape[1]:>3} cols (combined)")
            else:
                print(f"  [{cat:>10}] {key:<30} dict with {len(val)} keys")
        else:
            print(f"  [{cat:>10}] {key:<30} {type(val).__name__}")

    print("=" * 70)


def list_datasets():
    """Print all available datasets and their descriptions."""
    print("\n" + "=" * 70)
    print("  Available Datasets")
    print("=" * 70)

    for cat_key, cat_desc in CATEGORIES.items():
        print(f"\n  --- {cat_desc} ---")
        for key, entry in DATA_DICTIONARY.items():
            if entry['category'] == cat_key:
                deps = entry.get('depends_on', [])
                dep_str = f" (requires: {', '.join(deps)})" if deps else ""
                print(f"    {key:<30} {entry['description']}{dep_str}")

    print("\n" + "=" * 70)


def load_macro_only(clean=True):
    """Load only macroeconomic data (no market/stock data)."""
    return run_etl(
        categories=['inflation', 'rates', 'production', 'money', 'gdp', 'employment', 'macro'],
        clean=clean,
    )


def load_market_only(clean=True):
    """Load only market data (stocks, VIX, Fama-French, industry portfolios, news)."""
    return run_etl(categories=['market'], clean=clean)


def load_sec_only(clean=True):
    """Load SEC filings data (Form 4 insider trading, 10-K/10-Q fundamentals)."""
    return run_etl(categories=['sec'], clean=clean)


def load_political_only(clean=True):
    """Load political data (party platforms)."""
    return run_etl(categories=['political'], clean=clean)


# =============================================================================
# MAIN
# =============================================================================
if __name__ == '__main__':
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s'
    )

    print("Signals & Systems - ETL Pipeline")
    print(f"Started at {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print()

    data = run_etl()
    summarize(data)

    print(f"\nCompleted at {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
