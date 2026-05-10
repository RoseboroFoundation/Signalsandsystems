"""
Clean package — data cleaning and aggregation for Culture War Companies research.

Public API re-exports for backward compatibility with ``from clean import ...``.
"""

# Config & helpers
from .config import (
    API_KEY,
    START_DATE,
    END_DATE,
    VIX_OUTPUT_FILE,
    SCRIPT_DIR,
    logger,
    import_culture_war_data,
    _validate_fred_api_key,
    _download_fred_series,
)

# Validation
from .validation import validate_dataframe

# Cache utilities
from .cache import _save_cache, _load_cache

# Market data
from .market_data import (
    get_stock_data,
    download_vix_data,
    download_fama_french_factors,
    download_industry_portfolios,
)

# SEC Form 4
from .sec_form4 import Form4Downloader

# SEC 10-K / 10-Q filings
from .sec_filings import SECFilingDownloader

# News
from .news import (
    NewsArticle,
    CompanyNewsAggregator,
    load_news_data,
    scrape_culture_war_news,
    enrich_news_with_text,
)

# FRED macro loaders
from .fred_loaders import (
    load_inflation_data,
    load_inflation_expectations_data,
    load_comprehensive_inflation_data,
    load_treasury_yields,
    load_policy_rates,
    load_credit_spreads,
    calculate_yield_curve_metrics,
    load_comprehensive_rates_data,
    load_industrial_production_data,
    load_ip_growth_rates,
    load_comprehensive_ip_data,
    load_money_supply_data,
    load_money_velocity_data,
    load_fed_balance_sheet_data,
    load_m2_growth_rates,
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
    get_inflation_regime,
)

# Party Platforms
from .party_platforms import PartyPlatformDownloader

# Political events & exposure (Essay 3 reframe)
from .political_events import load_political_events
from .political_exposure import load_political_exposure

# Orchestration
from .orchestration import (
    load_culture_war_companies,
    build_control_companies,
    load_data,
    clean_dataframe,
    clean_all_data,
    get_clean_data,
    analyze_news_sentiment_around_events,
    get_news_for_ticker,
)
