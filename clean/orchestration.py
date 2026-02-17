"""Orchestration: load_data, cleaning, analysis, and main entry point."""

import os
from typing import List

import pandas as pd

from .config import logger, import_culture_war_data, SCRIPT_DIR
from .market_data import get_stock_data, download_vix_data, download_fama_french_factors
from .sec_form4 import Form4Downloader
from .news import load_news_data
from .fred_loaders import (
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
    get_inflation_regime,
)

def load_culture_war_companies(culture_war_data: pd.DataFrame) -> List[str]:
    """
    Extract unique company tickers from culture war dataset.

    Parameters:
    -----------
    culture_war_data : pd.DataFrame
        Your culture war events dataset

    Returns:
    --------
    List[str] : Unique ticker symbols
    """
    possible_ticker_cols = ['Ticker', 'ticker', 'TICKER', 'Symbol', 'symbol']
    possible_company_cols = ['Company', 'company', 'COMPANY', 'company_name']

    ticker_col = None
    for col in possible_ticker_cols:
        if col in culture_war_data.columns:
            ticker_col = col
            break

    if ticker_col:
        tickers = culture_war_data[ticker_col].unique().tolist()
        tickers = [
            t for t in tickers
            if pd.notna(t) and
            str(t).strip() not in ['', 'Private', 'N/A', 'NA', 'None']
        ]
        return tickers

    for col in possible_company_cols:
        if col in culture_war_data.columns:
            companies = culture_war_data[col].unique().tolist()
            logger.info("Warning: Found company names but not tickers. "
                       "You'll need to map company names to tickers.")
            return [c for c in companies if pd.notna(c)]

    logger.error("Available columns: %s", culture_war_data.columns.tolist())
    raise ValueError("Cannot find ticker or company column in culture war data")


def load_data():
    """
    Load all datasets into a single dictionary.

    Returns:
    --------
    dict : Dictionary containing all loaded datasets:
        - culturewardata: Culture war companies events
        - stockdata: Historical stock prices
        - vixdata: VIX volatility index
        - ff_factors: Fama-French factors (FF3, FF5, MOM)
        - form4data: SEC Form 4 insider trading
        - newsdata: News articles from Guardian, NYT, Reddit
        - inflationdata: Inflation measures from FRED
        - inflation_expectations: Breakeven inflation & survey expectations
        - comprehensive_inflation: All inflation measures combined
        - treasury_yields: Treasury yield curve (1M to 30Y, TIPS)
        - policy_rates: Fed Funds, SOFR, Prime, discount rates
        - credit_spreads: Corporate yields, credit spreads, mortgages
        - comprehensive_rates: All rates with yield curve metrics
        - industrial_production: IP indices, sectors, capacity utilization
        - ip_growth: IP growth rates (YoY, MoM) and diffusion indices
        - comprehensive_ip: All IP measures combined
        - money_supply: M1, M2, monetary base, components
        - money_velocity: M1 and M2 velocity
        - fed_balance_sheet: Fed assets, reserves, balance sheet
        - comprehensive_m2: All M2 measures with growth rates
        - gdp_data: Nominal/Real GDP, growth rates, per capita
        - gdp_components: Consumption, Investment, Government, Trade
        - gdp_industry: GDP by industry/sector (value added)
        - comprehensive_gdp: All GDP measures combined
        - employment_data: Payrolls, unemployment rates, labor force
        - jobless_claims: Initial/continuing claims, insured unemployment
        - wages_hours: Average earnings, hours worked, labor costs
        - jolts_data: Job openings, hires, quits, separations
        - comprehensive_employment: All employment measures combined
        - additional_macro: Consumer Sentiment, Housing, Dollar Index
    """
    data_dict = {}

    # Load culture war companies data
    try:
        data_dict['culturewardata'] = import_culture_war_data(
            'Culture_War_Companies_160_fullmeta.csv'
        )
        logger.info("Loaded culture war data")
    except Exception as e:
        logger.error("loading culture war data: %s", e)
        data_dict['culturewardata'] = None

    # Load stock data
    try:
        if data_dict['culturewardata'] is not None:
            tickers = data_dict['culturewardata']['Ticker'].unique().tolist()
            data_dict['stockdata'] = get_stock_data(
                tickers, start_date='2000-01-01', end_date='2025-12-31'
            )
            logger.info("Loaded stock data for %d tickers", len(data_dict['stockdata']))
        else:
            data_dict['stockdata'] = None
    except Exception as e:
        logger.error("loading stock data: %s", e)
        data_dict['stockdata'] = None

    # Load VIX data
    try:
        data_dict['vixdata'] = download_vix_data()
        logger.info("Loaded VIX data")
    except Exception as e:
        logger.error("loading VIX data: %s", e)
        data_dict['vixdata'] = None

    # Load Fama-French factors
    try:
        data_dict['ff_factors'] = download_fama_french_factors(
            start_date='2000-01-01',
            frequency='daily',
            output_dir='./fama_french_data'
        )
        logger.info("Loaded Fama-French factors")
    except Exception as e:
        logger.error("loading Fama-French factors: %s", e)
        data_dict['ff_factors'] = None

    # Load Form 4 insider trading data
    try:
        form4_downloader = Form4Downloader()
        if data_dict['culturewardata'] is not None:
            tickers = load_culture_war_companies(data_dict['culturewardata'])
            data_dict['form4data'] = form4_downloader.build_form4_dataset(
                tickers,
                start_date='2000-01-01',
                end_date='2025-12-31',
                save_csv=True
            )
            logger.info("Loaded Form 4 data")
        else:
            data_dict['form4data'] = None
    except Exception as e:
        logger.error("loading Form 4 data: %s", e)
        data_dict['form4data'] = None

    # Load news data
    try:
        data_dict['newsdata'] = load_news_data(
            cache_file='./news_data/culture_war_news_2000_2025_final.csv',
            refresh=False
        )
        logger.info("Loaded news data")
    except Exception as e:
        logger.error("loading news data: %s", e)
        data_dict['newsdata'] = None

    # Load inflation data (core measures)
    try:
        data_dict['inflationdata'] = load_inflation_data(
            start_date='2000-01-01',
            end_date='2025-12-31',
            cache_path='./data/fred'
        )
        logger.info("Loaded inflation data")
    except Exception as e:
        logger.error("loading inflation data: %s", e)
        data_dict['inflationdata'] = None

    # Load inflation expectations (breakeven, surveys, Fed measures)
    try:
        data_dict['inflation_expectations'] = load_inflation_expectations_data(
            start_date='2000-01-01',
            end_date='2025-12-31',
            cache_path='./data/fred'
        )
        logger.info("Loaded inflation expectations data")
    except Exception as e:
        logger.error("loading inflation expectations data: %s", e)
        data_dict['inflation_expectations'] = None

    # Load comprehensive inflation data (all measures combined)
    try:
        data_dict['comprehensive_inflation'] = load_comprehensive_inflation_data(
            start_date='2000-01-01',
            end_date='2025-12-31',
            cache_path='./data/fred'
        )
        logger.info("Loaded comprehensive inflation data")
    except Exception as e:
        logger.error("loading comprehensive inflation data: %s", e)
        data_dict['comprehensive_inflation'] = None

    # Load Treasury yields
    try:
        data_dict['treasury_yields'] = load_treasury_yields(
            start_date='2000-01-01',
            end_date='2025-12-31',
            cache_path='./data/fred'
        )
        logger.info("Loaded Treasury yields data")
    except Exception as e:
        logger.error("loading Treasury yields data: %s", e)
        data_dict['treasury_yields'] = None

    # Load policy rates (Fed Funds, SOFR, Prime)
    try:
        data_dict['policy_rates'] = load_policy_rates(
            start_date='2000-01-01',
            end_date='2025-12-31',
            cache_path='./data/fred'
        )
        logger.info("Loaded policy rates data")
    except Exception as e:
        logger.error("loading policy rates data: %s", e)
        data_dict['policy_rates'] = None

    # Load credit spreads and mortgage rates
    try:
        data_dict['credit_spreads'] = load_credit_spreads(
            start_date='2000-01-01',
            end_date='2025-12-31',
            cache_path='./data/fred'
        )
        logger.info("Loaded credit spreads data")
    except Exception as e:
        logger.error("loading credit spreads data: %s", e)
        data_dict['credit_spreads'] = None

    # Load comprehensive rates data (all rates combined with curve metrics)
    try:
        data_dict['comprehensive_rates'] = load_comprehensive_rates_data(
            start_date='2000-01-01',
            end_date='2025-12-31',
            cache_path='./data/fred'
        )
        logger.info("Loaded comprehensive rates data")
    except Exception as e:
        logger.error("loading comprehensive rates data: %s", e)
        data_dict['comprehensive_rates'] = None

    # Load industrial production data
    try:
        data_dict['industrial_production'] = load_industrial_production_data(
            start_date='2000-01-01',
            end_date='2025-12-31',
            cache_path='./data/fred'
        )
        logger.info("Loaded industrial production data")
    except Exception as e:
        logger.error("loading industrial production data: %s", e)
        data_dict['industrial_production'] = None

    # Load IP growth rates and diffusion indices
    try:
        data_dict['ip_growth'] = load_ip_growth_rates(
            start_date='2000-01-01',
            end_date='2025-12-31',
            cache_path='./data/fred'
        )
        logger.info("Loaded IP growth rates data")
    except Exception as e:
        logger.error("loading IP growth rates data: %s", e)
        data_dict['ip_growth'] = None

    # Load comprehensive IP data (all IP measures combined)
    try:
        data_dict['comprehensive_ip'] = load_comprehensive_ip_data(
            start_date='2000-01-01',
            end_date='2025-12-31',
            cache_path='./data/fred'
        )
        logger.info("Loaded comprehensive IP data")
    except Exception as e:
        logger.error("loading comprehensive IP data: %s", e)
        data_dict['comprehensive_ip'] = None

    # Load money supply data (M1, M2, components)
    try:
        data_dict['money_supply'] = load_money_supply_data(
            start_date='2000-01-01',
            end_date='2025-12-31',
            cache_path='./data/fred'
        )
        logger.info("Loaded money supply data")
    except Exception as e:
        logger.error("loading money supply data: %s", e)
        data_dict['money_supply'] = None

    # Load money velocity data
    try:
        data_dict['money_velocity'] = load_money_velocity_data(
            start_date='2000-01-01',
            end_date='2025-12-31',
            cache_path='./data/fred'
        )
        logger.info("Loaded money velocity data")
    except Exception as e:
        logger.error("loading money velocity data: %s", e)
        data_dict['money_velocity'] = None

    # Load Fed balance sheet data
    try:
        data_dict['fed_balance_sheet'] = load_fed_balance_sheet_data(
            start_date='2000-01-01',
            end_date='2025-12-31',
            cache_path='./data/fred'
        )
        logger.info("Loaded Fed balance sheet data")
    except Exception as e:
        logger.error("loading Fed balance sheet data: %s", e)
        data_dict['fed_balance_sheet'] = None

    # Load comprehensive M2 data (all money supply measures combined)
    try:
        data_dict['comprehensive_m2'] = load_comprehensive_m2_data(
            start_date='2000-01-01',
            end_date='2025-12-31',
            cache_path='./data/fred'
        )
        logger.info("Loaded comprehensive M2 data")
    except Exception as e:
        logger.error("loading comprehensive M2 data: %s", e)
        data_dict['comprehensive_m2'] = None

    # Load GDP headline data
    try:
        data_dict['gdp_data'] = load_gdp_data(
            start_date='2000-01-01',
            end_date='2025-12-31',
            cache_path='./data/fred'
        )
        logger.info("Loaded GDP data")
    except Exception as e:
        logger.error("loading GDP data: %s", e)
        data_dict['gdp_data'] = None

    # Load GDP components (C + I + G + NX)
    try:
        data_dict['gdp_components'] = load_gdp_components_data(
            start_date='2000-01-01',
            end_date='2025-12-31',
            cache_path='./data/fred'
        )
        logger.info("Loaded GDP components data")
    except Exception as e:
        logger.error("loading GDP components data: %s", e)
        data_dict['gdp_components'] = None

    # Load GDP by industry
    try:
        data_dict['gdp_industry'] = load_gdp_by_industry_data(
            start_date='2000-01-01',
            end_date='2025-12-31',
            cache_path='./data/fred'
        )
        logger.info("Loaded GDP by industry data")
    except Exception as e:
        logger.error("loading GDP by industry data: %s", e)
        data_dict['gdp_industry'] = None

    # Load comprehensive GDP data (all GDP measures combined)
    try:
        data_dict['comprehensive_gdp'] = load_comprehensive_gdp_data(
            start_date='2000-01-01',
            end_date='2025-12-31',
            cache_path='./data/fred'
        )
        logger.info("Loaded comprehensive GDP data")
    except Exception as e:
        logger.error("loading comprehensive GDP data: %s", e)
        data_dict['comprehensive_gdp'] = None

    # Load employment data (payrolls, unemployment, labor force)
    try:
        data_dict['employment_data'] = load_employment_data(
            start_date='2000-01-01',
            end_date='2025-12-31',
            cache_path='./data/fred'
        )
        logger.info("Loaded employment data")
    except Exception as e:
        logger.error("loading employment data: %s", e)
        data_dict['employment_data'] = None

    # Load jobless claims data
    try:
        data_dict['jobless_claims'] = load_jobless_claims_data(
            start_date='2000-01-01',
            end_date='2025-12-31',
            cache_path='./data/fred'
        )
        logger.info("Loaded jobless claims data")
    except Exception as e:
        logger.error("loading jobless claims data: %s", e)
        data_dict['jobless_claims'] = None

    # Load wages and hours data
    try:
        data_dict['wages_hours'] = load_wages_hours_data(
            start_date='2000-01-01',
            end_date='2025-12-31',
            cache_path='./data/fred'
        )
        logger.info("Loaded wages and hours data")
    except Exception as e:
        logger.error("loading wages and hours data: %s", e)
        data_dict['wages_hours'] = None

    # Load JOLTS data (job openings, hires, quits)
    try:
        data_dict['jolts_data'] = load_jolts_data(
            start_date='2000-01-01',
            end_date='2025-12-31',
            cache_path='./data/fred'
        )
        logger.info("Loaded JOLTS data")
    except Exception as e:
        logger.error("loading JOLTS data: %s", e)
        data_dict['jolts_data'] = None

    # Load comprehensive employment data (all employment measures combined)
    try:
        data_dict['comprehensive_employment'] = load_comprehensive_employment_data(
            start_date='2000-01-01',
            end_date='2025-12-31',
            cache_path='./data/fred'
        )
        logger.info("Loaded comprehensive employment data")
    except Exception as e:
        logger.error("loading comprehensive employment data: %s", e)
        data_dict['comprehensive_employment'] = None

    # Load additional macro data (Consumer Sentiment, Housing, Dollar Index)
    try:
        data_dict['additional_macro'] = load_additional_macro_data(
            start_date='2000-01-01',
            end_date='2025-12-31',
            cache_path='./data/fred'
        )
        logger.info("Loaded additional macro data")
    except Exception as e:
        logger.error("loading additional macro data: %s", e)
        data_dict['additional_macro'] = None

    return data_dict


# =============================================================================
# DATA CLEANING FUNCTIONS
# =============================================================================
def clean_dataframe(df, method='ffill', max_gap=5):
    """
    Clean a single DataFrame by handling missing values and standardizing format.

    Parameters:
    -----------
    df : pd.DataFrame
        DataFrame to clean
    method : str
        Method for filling missing values: 'ffill', 'bfill', 'interpolate', 'drop'
    max_gap : int
        Maximum consecutive NaN values to fill (prevents filling large gaps)

    Returns:
    --------
    pd.DataFrame : Cleaned DataFrame
    """
    if df is None or (hasattr(df, 'empty') and df.empty):
        return df

    # Make a copy to avoid modifying the original
    cleaned = df.copy()

    # Ensure datetime index if applicable
    if hasattr(cleaned, 'index') and not isinstance(cleaned.index, pd.DatetimeIndex):
        try:
            if cleaned.index.dtype == 'object':
                cleaned.index = pd.to_datetime(cleaned.index, errors='coerce')
        except Exception as e:
            logger.debug("Failed to convert index to datetime: %s", e)

    # Sort by index if datetime
    if isinstance(cleaned.index, pd.DatetimeIndex):
        cleaned = cleaned.sort_index()

    # Handle missing values based on method
    if method == 'ffill':
        cleaned = cleaned.ffill(limit=max_gap)
    elif method == 'bfill':
        cleaned = cleaned.bfill(limit=max_gap)
    elif method == 'interpolate':
        cleaned = cleaned.interpolate(method='time', limit=max_gap)
    elif method == 'drop':
        cleaned = cleaned.dropna()

    # Remove any remaining rows that are entirely NaN
    cleaned = cleaned.dropna(how='all')

    return cleaned


def clean_all_data(data_dict, verbose=True):
    """
    Clean all datasets in the data dictionary.

    Applies appropriate cleaning methods to each dataset type:
    - Time series data: forward fill with interpolation for small gaps
    - Stock data: forward fill (markets closed on weekends/holidays)
    - Cross-sectional data: drop missing values

    Parameters:
    -----------
    data_dict : dict
        Dictionary of datasets from load_data()
    verbose : bool
        If True, print cleaning summary

    Returns:
    --------
    dict : Dictionary of cleaned datasets
    """
    if verbose:
        logger.info("=" * 60)
        logger.info("=== Cleaning All Datasets ===")
        logger.info("=" * 60)

    cleaned_dict = {}

    # Define cleaning strategies for each dataset type
    time_series_keys = [
        'inflationdata', 'inflation_expectations', 'comprehensive_inflation',
        'treasury_yields', 'policy_rates', 'credit_spreads', 'comprehensive_rates',
        'industrial_production', 'ip_growth', 'comprehensive_ip',
        'money_supply', 'money_velocity', 'fed_balance_sheet', 'comprehensive_m2',
        'gdp_data', 'gdp_components', 'gdp_industry', 'comprehensive_gdp',
        'employment_data', 'jobless_claims', 'wages_hours', 'jolts_data',
        'comprehensive_employment', 'additional_macro', 'vixdata'
    ]

    for key, data in data_dict.items():
        if data is None:
            cleaned_dict[key] = None
            if verbose:
                logger.info("  %s: Skipped (None)", key)
            continue

        try:
            if key == 'stockdata':
                # Stock data is a dict of DataFrames
                if isinstance(data, dict):
                    cleaned_stocks = {}
                    for ticker, stock_df in data.items():
                        if stock_df is not None and not stock_df.empty:
                            cleaned_stocks[ticker] = clean_dataframe(stock_df, method='ffill')
                    cleaned_dict[key] = cleaned_stocks
                    if verbose:
                        logger.info("  %s: Cleaned %d ticker DataFrames", key, len(cleaned_stocks))
                else:
                    cleaned_dict[key] = data

            elif key == 'ff_factors':
                # Fama-French factors is a dict of DataFrames
                if isinstance(data, dict):
                    cleaned_ff = {}
                    for factor_name, factor_df in data.items():
                        if factor_df is not None and hasattr(factor_df, 'empty') and not factor_df.empty:
                            cleaned_ff[factor_name] = clean_dataframe(factor_df, method='ffill')
                        else:
                            cleaned_ff[factor_name] = factor_df
                    cleaned_dict[key] = cleaned_ff
                    if verbose:
                        logger.info("  %s: Cleaned %d factor DataFrames", key, len(cleaned_ff))
                else:
                    cleaned_dict[key] = data

            elif key in time_series_keys:
                # Handle dict of DataFrames or single DataFrame
                if isinstance(data, dict):
                    cleaned_ts = {}
                    for sub_key, sub_df in data.items():
                        if sub_df is not None and hasattr(sub_df, 'empty') and not sub_df.empty:
                            cleaned_ts[sub_key] = clean_dataframe(sub_df, method='ffill')
                        else:
                            cleaned_ts[sub_key] = sub_df
                    cleaned_dict[key] = cleaned_ts
                    if verbose:
                        logger.info("  %s: Cleaned %d sub-DataFrames", key, len(cleaned_ts))
                elif isinstance(data, pd.DataFrame):
                    cleaned_dict[key] = clean_dataframe(data, method='ffill')
                    if verbose:
                        orig_nulls = data.isnull().sum().sum()
                        new_nulls = cleaned_dict[key].isnull().sum().sum() if cleaned_dict[key] is not None else 0
                        logger.info("  %s: Cleaned (NaN: %d -> %d)", key, orig_nulls, new_nulls)
                else:
                    cleaned_dict[key] = data

            elif key in ['culturewardata', 'newsdata', 'form4data']:
                # Cross-sectional data - keep as is (already cleaned during load)
                cleaned_dict[key] = data
                if verbose:
                    if isinstance(data, pd.DataFrame):
                        logger.info("  %s: Kept as-is (%d rows)", key, data.shape[0])
                    else:
                        logger.info("  %s: Kept as-is", key)

            else:
                # Unknown data type - keep as is
                cleaned_dict[key] = data
                if verbose:
                    logger.info("  %s: Kept as-is (unknown type)", key)

        except Exception as e:
            logger.error("%s: error during cleaning - %s", key, e)
            cleaned_dict[key] = data

    if verbose:
        logger.info("=" * 60)
        logger.info("Data cleaning complete!")
        logger.info("=" * 60)

    return cleaned_dict


def get_clean_data():
    """
    Load all data and apply cleaning.

    This is a convenience function that calls load_data() followed by clean_all_data().

    Returns:
    --------
    dict : Dictionary of cleaned datasets
    """
    data_dict = load_data()
    cleaned_dict = clean_all_data(data_dict, verbose=True)
    return cleaned_dict


# =============================================================================
# ANALYSIS FUNCTIONS
# =============================================================================
def analyze_news_sentiment_around_events(data_dict):
    """Analyze news volume and sentiment around culture war events."""
    news = data_dict['newsdata']
    culture_wars = data_dict['culturewardata']

    if news is None or len(news) == 0:
        logger.info("No news data available")
        return None

    logger.info("Available columns in culture_wars data:")
    logger.info("%s", culture_wars.columns.tolist())

    # Detect actual column names
    date_col = None
    desc_col = None
    cat_col = None

    for col in culture_wars.columns:
        if 'date' in col.lower():
            date_col = col
        if 'description' in col.lower() or 'event' in col.lower():
            if desc_col is None:
                desc_col = col
        if 'category' in col.lower() or 'type' in col.lower():
            cat_col = col

    logger.info("Using columns:")
    logger.info("  Date: %s", date_col)
    logger.info("  Description: %s", desc_col)
    logger.info("  Category: %s", cat_col)

    merge_cols = ['Ticker']
    if date_col:
        merge_cols.append(date_col)
    if desc_col:
        merge_cols.append(desc_col)
    if cat_col:
        merge_cols.append(cat_col)

    analysis_df = news.merge(
        culture_wars[merge_cols],
        left_on='ticker',
        right_on='Ticker',
        how='inner'
    )

    if date_col:
        analysis_df[date_col] = pd.to_datetime(analysis_df[date_col])
        analysis_df['days_from_event'] = (
            analysis_df['published_date'] - analysis_df[date_col]
        ).dt.days

        event_window = analysis_df[analysis_df['days_from_event'].abs() <= 30]

        if cat_col:
            logger.info("=== News Coverage by Event Category ===")
            category_coverage = event_window.groupby(cat_col).agg({
                'title': 'count',
                'ticker': 'nunique'
            }).rename(columns={'title': 'article_count', 'ticker': 'company_count'})

            logger.info("%s", category_coverage)

        return event_window
    else:
        logger.info("No date column found - cannot calculate event windows")
        return analysis_df


def get_news_for_ticker(data_dict, ticker, days_window=30):
    """Get all news for a specific ticker around its culture war event(s)."""
    news = data_dict['newsdata']
    culture_wars = data_dict['culturewardata']

    if news is None or len(news) == 0:
        logger.info("No news data available")
        return None

    date_col = None
    desc_col = None

    for col in culture_wars.columns:
        if 'date' in col.lower():
            date_col = col
        if 'description' in col.lower() or 'event' in col.lower():
            if desc_col is None:
                desc_col = col

    events = culture_wars[culture_wars['Ticker'] == ticker]
    ticker_news = news[news['ticker'] == ticker].copy()

    if date_col:
        for _, event in events.iterrows():
            event_date = pd.to_datetime(event[date_col])
            event_desc = event[desc_col] if desc_col else "Culture war event"

            window_news = ticker_news[
                (ticker_news['published_date'] >= event_date - pd.Timedelta(days=days_window)) &
                (ticker_news['published_date'] <= event_date + pd.Timedelta(days=days_window))
            ]

            logger.info("=== %s: %s ===", ticker, event_desc)
            logger.info("Event Date: %s", event_date.date())
            logger.info("Articles in +/-%d day window: %d", days_window, len(window_news))

            if len(window_news) > 0:
                logger.info("Top 5 articles:")
                for _, row in window_news.head().iterrows():
                    logger.info(
                        "  [%s] %s: %s",
                        row['published_date'].date(),
                        row['source'],
                        row['title']
                    )
    else:
        logger.info("No date column found. Showing all %d articles for %s", len(ticker_news), ticker)

    return ticker_news


# =============================================================================
# MAIN ENTRY POINT
# =============================================================================
if __name__ == "__main__":
    print("=" * 60)
    print("Loading and cleaning all datasets...")
    print("=" * 60)
    data_dict = get_clean_data()

    # Print summary
    print("\n" + "=" * 60)
    print("=== Data Dictionary Summary ===")
    print("=" * 60)
    for key, value in data_dict.items():
        print(f"\n{key}:")
        if isinstance(value, dict):
            for subkey, df in value.items():
                if df is not None:
                    print(f"  {subkey}: {df.shape if hasattr(df, 'shape') else 'N/A'}")
                else:
                    print(f"  {subkey}: Not loaded")
        elif value is not None:
            if hasattr(value, 'shape'):
                print(f"  Shape: {value.shape}")
            else:
                print("  Status: Loaded")
        else:
            print("  Status: Not loaded")

    # Show culture war data structure
    if data_dict['culturewardata'] is not None:
        print("\n" + "=" * 60)
        print("=== Culture War Data Structure ===")
        print("=" * 60)
        print("Columns:", data_dict['culturewardata'].columns.tolist())
        print("\nFirst few rows:")
        print(data_dict['culturewardata'].head())

    # Show inflation data summary
    if data_dict['inflationdata'] is not None:
        print("\n" + "=" * 60)
        print("=== Inflation Data Summary ===")
        print("=" * 60)
        inflation = data_dict['inflationdata']

        print("\nRaw indices shape:", inflation['raw'].shape)
        print("Year-over-year changes shape:", inflation['yoy'].shape)
        print("Month-over-month changes shape:", inflation['mom'].shape)

        print("\nLatest inflation readings (YoY %):")
        print(inflation['yoy'].iloc[-1])

        core_pce_yoy = inflation['yoy']['Core_PCE_YoY']
        latest_inflation = core_pce_yoy.iloc[-1]

        if latest_inflation < 2.0:
            regime = "Low Inflation"
        elif latest_inflation < 4.0:
            regime = "Moderate Inflation"
        else:
            regime = "High Inflation"

        print(f"\nCurrent inflation regime (based on Core PCE): {regime}")
        print(f"  Core PCE YoY: {latest_inflation:.2f}%")

    # Run news analysis if available
    if data_dict['newsdata'] is not None and len(data_dict['newsdata']) > 0:
        print("\n" + "=" * 60)
        print("=== News Data Analysis ===")
        print("=" * 60)

        event_news = analyze_news_sentiment_around_events(data_dict)

        if 'DIS' in data_dict['newsdata']['ticker'].values:
            dis_news = get_news_for_ticker(data_dict, 'DIS', days_window=60)

        # Export merged dataset
        culture_wars = data_dict['culturewardata']
        news = data_dict['newsdata']

        date_col = None
        for col in culture_wars.columns:
            if 'date' in col.lower():
                date_col = col
                break

        event_news_df = news.merge(
            culture_wars,
            left_on='ticker',
            right_on='Ticker',
            how='inner'
        )

        if date_col:
            event_news_df[date_col] = pd.to_datetime(event_news_df[date_col])
            event_news_df['days_from_event'] = (
                event_news_df['published_date'] - event_news_df[date_col]
            ).dt.days

        os.makedirs('./analysis_data', exist_ok=True)
        event_news_df.to_csv('./analysis_data/event_news_merged.csv', index=False)
        print(f"\nSaved merged event-news dataset: {len(event_news_df):,} records")
    else:
        print("\n" + "=" * 60)
        print("=== News Data ===")
        print("=" * 60)
        print("No news data available yet. Run news aggregator to collect data.")

    # Save inflation plot
    if data_dict['inflationdata'] is not None:
        try:
            import matplotlib.pyplot as plt

            inflation = data_dict['inflationdata']
            regimes = get_inflation_regime(inflation)

            fig, axes = plt.subplots(2, 1, figsize=(12, 8))

            inflation['yoy'][['CPI_YoY', 'Core_CPI_YoY', 'Core_PCE_YoY']].plot(
                ax=axes[0],
                title='Inflation Measures (Year-over-Year %)',
                ylabel='YoY Change (%)'
            )
            axes[0].axhline(y=2.0, color='r', linestyle='--', label='Fed Target')
            axes[0].legend()
            axes[0].grid(True, alpha=0.3)

            regime_numeric = regimes.map({
                'Low Inflation': 0,
                'Moderate Inflation': 1,
                'High Inflation': 2
            })
            regime_numeric.plot(
                ax=axes[1],
                title='Inflation Regime (Based on Core PCE)',
                ylabel='Regime',
                style='o-'
            )
            axes[1].set_yticks([0, 1, 2])
            axes[1].set_yticklabels(['Low', 'Moderate', 'High'])
            axes[1].grid(True, alpha=0.3)

            plt.tight_layout()
            plt.savefig('inflation_analysis.png', dpi=300, bbox_inches='tight')
            print("\nSaved plot to inflation_analysis.png")
        except ImportError:
            print("\nMatplotlib not available - skipping plot generation")

    # Final summary
    print("\n" + "=" * 60)
    print("=== Complete Dataset Summary ===")
    print("=" * 60)
    print("\nDatasets loaded:")
    for key, value in data_dict.items():
        status = "Loaded" if value is not None else "Not loaded"
        print(f"  {key}: {status}")

    print("\n" + "=" * 60)
    print("Data loading complete!")
    print("=" * 60)
