"""Market data loaders: stock prices, VIX, Fama-French factors."""

import os
from datetime import datetime

import pandas as pd
import yfinance as yf
from fredapi import Fred

from .config import API_KEY, START_DATE, END_DATE, VIX_OUTPUT_FILE, logger

def get_stock_data(tickers, start_date='2000-01-01', end_date='2025-12-31'):
    """
    Downloads stock data for given tickers from Yahoo Finance.

    Parameters:
    -----------
    tickers : list
        List of stock tickers
    start_date : str
        Start date in YYYY-MM-DD format
    end_date : str
        End date in YYYY-MM-DD format

    Returns:
    --------
    dict
        Dictionary with tickers as keys and dataframes as values
    """
    stock_data = {}
    failed_tickers = []

    for ticker in tickers:
        try:
            logger.info("Downloading data for %s...", ticker)
            data = yf.download(
                ticker,
                start=start_date,
                end=end_date,
                progress=False,
                auto_adjust=False
            )

            if not data.empty:
                data = data.reset_index()
                data.insert(0, 'Ticker', ticker)

                column_order = [
                    'Ticker', 'Date', 'Open', 'High', 'Low',
                    'Close', 'Volume', 'Adj Close'
                ]
                data = data[column_order]

                stock_data[ticker] = data
                logger.info("  Successfully downloaded %d rows for %s", len(data), ticker)
            else:
                failed_tickers.append(ticker)
                logger.info("  No data found for %s", ticker)

        except Exception as e:
            failed_tickers.append(ticker)
            logger.error("Error downloading %s: %s", ticker, e)

    if failed_tickers:
        logger.error("Failed to download data for: %s", failed_tickers)

    return stock_data


# =============================================================================
# VIX DATA
# =============================================================================
def download_vix_data():
    """Download VIX data from FRED and save to CSV."""

    if not API_KEY:
        logger.error("ERROR: FRED API key not found or not set!")
        logger.error("Please follow these steps:")
        logger.error("1. Create a file named '.env' in the same directory as this script")
        logger.error("2. Add this line to the .env file:")
        logger.error("   FRED_API_KEY=your_actual_api_key_here")
        logger.error("3. Get your free API key at:")
        logger.error("   https://fred.stlouisfed.org/docs/api/api_key.html")
        return None

    try:
        logger.info("Connecting to FRED...")
        fred = Fred(api_key=API_KEY)

        logger.info("Downloading VIX data from %s to %s...", START_DATE, END_DATE)
        vix_series = fred.get_series(
            'VIXCLS',
            observation_start=START_DATE,
            observation_end=END_DATE
        )

        vix_df = pd.DataFrame({
            'date': vix_series.index,
            'vix': vix_series.values
        })

        vix_df = vix_df.dropna()
        vix_df.to_csv(VIX_OUTPUT_FILE, index=False)

        # Log summary
        logger.info("=" * 60)
        logger.info("DOWNLOAD COMPLETE!")
        logger.info("=" * 60)
        logger.info("File saved: %s", VIX_OUTPUT_FILE)
        logger.info("Date range: %s to %s", vix_df['date'].min().date(), vix_df['date'].max().date())
        logger.info("Total observations: %s", f"{len(vix_df):,}")

        logger.info("VIX Summary Statistics:")
        logger.info("-" * 60)
        stats = vix_df['vix'].describe()
        logger.info("Count:  %.0f", stats['count'])
        logger.info("Mean:   %.2f", stats['mean'])
        logger.info("Std:    %.2f", stats['std'])
        logger.info("Min:    %.2f", stats['min'])
        logger.info("25%%:    %.2f", stats['25%'])
        logger.info("50%%:    %.2f", stats['50%'])
        logger.info("75%%:    %.2f", stats['75%'])
        logger.info("Max:    %.2f", stats['max'])

        # Find extremes
        max_idx = vix_df['vix'].idxmax()
        min_idx = vix_df['vix'].idxmin()

        logger.info("Highest VIX:")
        logger.info("   %.2f on %s", vix_df.loc[max_idx, 'vix'], vix_df.loc[max_idx, 'date'].date())

        logger.info("Lowest VIX:")
        logger.info("   %.2f on %s", vix_df.loc[min_idx, 'vix'], vix_df.loc[min_idx, 'date'].date())

        vix_gt_30 = (vix_df['vix'] > 30).sum()
        vix_gt_40 = (vix_df['vix'] > 40).sum()
        vix_gt_50 = (vix_df['vix'] > 50).sum()
        total = len(vix_df)
        logger.info("Quick Analysis:")
        logger.info("Days with VIX > 30: %s (%.1f%%)", f"{vix_gt_30:,}", vix_gt_30 / total * 100)
        logger.info("Days with VIX > 40: %s (%.1f%%)", f"{vix_gt_40:,}", vix_gt_40 / total * 100)
        logger.info("Days with VIX > 50: %s (%.1f%%)", f"{vix_gt_50:,}", vix_gt_50 / total * 100)

        logger.info("Citation:")
        logger.info("Chicago Board Options Exchange, CBOE Volatility Index: VIX [VIXCLS],")
        logger.info("retrieved from FRED, Federal Reserve Bank of St. Louis;")
        logger.info("https://fred.stlouisfed.org/series/VIXCLS")

        return vix_df

    except Exception as e:
        logger.error("Error: %s", e)
        logger.error("Troubleshooting:")
        logger.error("1. Verify your API key is correct in the .env file")
        logger.error("2. Check your internet connection")
        return None


# =============================================================================
# FAMA-FRENCH DATA
# =============================================================================
def download_fama_french_factors(
    start_date='1926-07-01',
    end_date=None,
    frequency='daily',
    output_dir=None
):
    """
    Download Fama-French factor data from Kenneth French's data library.

    Parameters:
    -----------
    start_date : str, default '1926-07-01'
        Start date in 'YYYY-MM-DD' format
    end_date : str, optional
        End date in 'YYYY-MM-DD' format (defaults to today)
    frequency : str, default 'daily'
        'daily', 'monthly', or 'annual'
    output_dir : str, optional
        Directory to save CSV files

    Returns:
    --------
    dict : Dictionary containing DataFrames for different factor models
    """
    if end_date is None:
        end_date = datetime.today().strftime('%Y-%m-%d')

    freq_map = {
        'daily': 'F-F_Research_Data_Factors_daily',
        'monthly': 'F-F_Research_Data_Factors',
        'annual': 'F-F_Research_Data_Factors'
    }

    results = {}

    import pandas_datareader as pdr

    try:
        # Download 3-Factor Model
        logger.info("Downloading Fama-French 3-Factor Model...")
        ff3 = pdr.DataReader(
            freq_map[frequency],
            'famafrench',
            start=start_date,
            end=end_date
        )[0]
        results['FF3'] = ff3

        # Download 5-Factor Model
        logger.info("Downloading Fama-French 5-Factor Model...")
        ff5_name = (
            'F-F_Research_Data_5_Factors_2x3_daily'
            if frequency == 'daily'
            else 'F-F_Research_Data_5_Factors_2x3'
        )
        ff5 = pdr.DataReader(
            ff5_name,
            'famafrench',
            start=start_date,
            end=end_date
        )[0]
        results['FF5'] = ff5

        # Download Momentum Factor
        logger.info("Downloading Momentum Factor...")
        mom_name = (
            'F-F_Momentum_Factor_daily'
            if frequency == 'daily'
            else 'F-F_Momentum_Factor'
        )
        mom = pdr.DataReader(
            mom_name,
            'famafrench',
            start=start_date,
            end=end_date
        )[0]
        results['MOM'] = mom

        # Save to CSV if path provided
        if output_dir:
            os.makedirs(output_dir, exist_ok=True)

            for name, df in results.items():
                filename = f"{name}_{frequency}_{start_date}_to_{end_date}.csv"
                filepath = os.path.join(output_dir, filename)
                df.to_csv(filepath)
                logger.info("Saved %s to %s", name, filepath)

        logger.info("Download complete! Date range: %s to %s", ff3.index[0], ff3.index[-1])
        return results

    except Exception as e:
        logger.error("downloading data: %s", e)
        return None


def download_industry_portfolios(
    num_industries=10,
    start_date='1926-07-01',
    end_date=None,
    frequency='daily',
    output_dir=None
):
    """
    Download Fama-French industry portfolio returns.

    Parameters:
    -----------
    num_industries : int
        Number of industries (5, 10, 12, 17, 30, 38, 48, or 49)
    start_date : str
        Start date in 'YYYY-MM-DD' format
    end_date : str, optional
        End date (defaults to today)
    frequency : str
        'daily' or 'monthly'
    output_dir : str, optional
        Directory to save CSV files

    Returns:
    --------
    pd.DataFrame : Industry portfolio returns
    """
    if end_date is None:
        end_date = datetime.today().strftime('%Y-%m-%d')

    freq_suffix = '_daily' if frequency == 'daily' else ''
    dataset_name = f'{num_industries}_Industry_Portfolios{freq_suffix}'

    import pandas_datareader as pdr

    try:
        logger.info("Downloading %s Industry Portfolios...", num_industries)
        ind_portfolios = pdr.DataReader(
            dataset_name,
            'famafrench',
            start=start_date,
            end=end_date
        )[0]

        if output_dir:
            os.makedirs(output_dir, exist_ok=True)
            filename = f"Industry_{num_industries}_{frequency}_{start_date}_to_{end_date}.csv"
            filepath = os.path.join(output_dir, filename)
            ind_portfolios.to_csv(filepath)
            logger.info("Saved to %s", filepath)

        return ind_portfolios

    except Exception as e:
        logger.error("downloading industry portfolios: %s", e)
        return None

