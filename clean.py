"""
Data cleaning and aggregation module for Culture War Companies research.

This module provides functions to:
- Import and clean culture war companies data
- Download stock data from Yahoo Finance
- Download VIX data from FRED
- Download Fama-French factor data
- Download SEC Form 4 insider trading data
- Aggregate news from Guardian, NYT, and Reddit
- Load inflation and macroeconomic data from FRED (2000-2025):
    * Core inflation measures (CPI, Core CPI, PCE, Core PCE, PPI, GDP Deflator)
    * Breakeven inflation rates (5Y, 10Y, 5Y5Y Forward)
    * Survey-based expectations (U of Michigan 1Y & 5Y)
    * Federal Reserve measures (Dallas Trimmed Mean PCE, Atlanta Sticky/Flexible CPI,
      Cleveland Median CPI, Cleveland 16% Trimmed Mean CPI)
    * Component-level inflation (Food, Energy, Shelter, Medical, Transportation, etc.)
    * Import/Export price indices
- Load interest rates data from FRED (2000-2025):
    * Treasury yield curve (1M, 3M, 6M, 1Y, 2Y, 3Y, 5Y, 7Y, 10Y, 20Y, 30Y)
    * TIPS real yields (5Y, 10Y, 20Y, 30Y)
    * Federal Reserve policy rates (Fed Funds effective/target, Discount Rate)
    * Money market rates (SOFR, Prime Rate, EFFR, Overnight Bank Funding)
    * Corporate bond yields (Moody's AAA/BAA, ICE BofA IG/HY indices)
    * Credit spreads (BAA-10Y, AAA-10Y, IG spread, HY spread, TED spread)
    * Mortgage rates (30Y, 15Y, 5/1 ARM)
    * Yield curve metrics (slope, curvature, inversion indicators)
- Load Industrial Production (IP) data from FRED (2000-2025):
    * Total Industrial Production Index (INDPRO)
    * Sector-level production (Manufacturing, Mining, Utilities)
    * Industry-specific production (Motor Vehicles, Chemicals, Machinery, etc.)
    * Capacity Utilization rates (Total, Manufacturing, Mining, Utilities)
    * Growth rates (Year-over-Year, Month-over-Month)
    * Diffusion indices (1M, 3M, 6M)
- Load M2 Money Supply data from FRED (2000-2025):
    * Money supply aggregates (M1, M2, Monetary Base)
    * Money supply components (Currency, Deposits, Money Funds)
    * Money velocity (M1V, M2V)
    * Federal Reserve balance sheet (Total Assets, Treasury/MBS Holdings)
    * Bank reserves (Required, Excess, Total)
    * M2 growth rates (YoY, MoM, 3M/6M annualized)
"""

# =============================================================================
# IMPORTS
# =============================================================================
import os
import time
import logging
from datetime import datetime, timedelta
from pathlib import Path
from typing import List, Dict, Optional
from dataclasses import dataclass

import pandas as pd
import pandas_datareader as pdr
import yfinance as yf
import requests
import praw
from bs4 import BeautifulSoup
from fredapi import Fred
from dotenv import load_dotenv

# =============================================================================
# CONFIGURATION
# =============================================================================
# Load environment variables
load_dotenv()

# Get the directory where this script is located
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))

# API Keys
API_KEY = os.getenv('FRED_API_KEY')

# Date range defaults
START_DATE = '2000-01-01'
END_DATE = '2025-12-31'

# Output file names
VIX_OUTPUT_FILE = 'vix_data_2000_2025.csv'

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


# =============================================================================
# CULTURE WAR COMPANIES DATA
# =============================================================================
def import_culture_war_data(file_path):
    """
    Imports and cleans the Culture War Companies dataset.

    Parameters:
    -----------
    file_path : str
        Path to the CSV file

    Returns:
    --------
    pd.DataFrame : Cleaned culture war companies data
    """
    # Convert relative path to absolute path based on script location
    if not os.path.isabs(file_path):
        file_path = os.path.join(SCRIPT_DIR, file_path)

    df = pd.read_csv(file_path)

    # Make "Event Date" a datetime object
    df['Event Date'] = pd.to_datetime(df['Event Date'], errors='coerce')

    return df


# =============================================================================
# STOCK DATA
# =============================================================================
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
            print(f"Downloading data for {ticker}...")
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
                print(f"  Successfully downloaded {len(data)} rows for {ticker}")
            else:
                failed_tickers.append(ticker)
                print(f"  No data found for {ticker}")

        except Exception as e:
            failed_tickers.append(ticker)
            print(f"  Error downloading {ticker}: {e}")

    if failed_tickers:
        print(f"\nFailed to download data for: {failed_tickers}")

    return stock_data


# =============================================================================
# VIX DATA
# =============================================================================
def download_vix_data():
    """Download VIX data from FRED and save to CSV."""

    if not API_KEY:
        print("\nERROR: FRED API key not found or not set!")
        print("\nPlease follow these steps:")
        print("1. Create a file named '.env' in the same directory as this script")
        print("2. Add this line to the .env file:")
        print("   FRED_API_KEY=your_actual_api_key_here")
        print("\n3. Get your free API key at:")
        print("   https://fred.stlouisfed.org/docs/api/api_key.html")
        return None

    try:
        print("Connecting to FRED...")
        fred = Fred(api_key=API_KEY)

        print(f"Downloading VIX data from {START_DATE} to {END_DATE}...")
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

        # Print summary
        print("\n" + "=" * 60)
        print("DOWNLOAD COMPLETE!")
        print("=" * 60)
        print(f"File saved: {VIX_OUTPUT_FILE}")
        print(f"Date range: {vix_df['date'].min().date()} to {vix_df['date'].max().date()}")
        print(f"Total observations: {len(vix_df):,}")

        print("\nVIX Summary Statistics:")
        print("-" * 60)
        stats = vix_df['vix'].describe()
        print(f"Count:  {stats['count']:,.0f}")
        print(f"Mean:   {stats['mean']:.2f}")
        print(f"Std:    {stats['std']:.2f}")
        print(f"Min:    {stats['min']:.2f}")
        print(f"25%:    {stats['25%']:.2f}")
        print(f"50%:    {stats['50%']:.2f}")
        print(f"75%:    {stats['75%']:.2f}")
        print(f"Max:    {stats['max']:.2f}")

        # Find extremes
        max_idx = vix_df['vix'].idxmax()
        min_idx = vix_df['vix'].idxmin()

        print("\nHighest VIX:")
        print(f"   {vix_df.loc[max_idx, 'vix']:.2f} on {vix_df.loc[max_idx, 'date'].date()}")

        print("Lowest VIX:")
        print(f"   {vix_df.loc[min_idx, 'vix']:.2f} on {vix_df.loc[min_idx, 'date'].date()}")

        print("\nQuick Analysis:")
        print(f"Days with VIX > 30: {(vix_df['vix'] > 30).sum():,} ({(vix_df['vix'] > 30).sum()/len(vix_df)*100:.1f}%)")
        print(f"Days with VIX > 40: {(vix_df['vix'] > 40).sum():,} ({(vix_df['vix'] > 40).sum()/len(vix_df)*100:.1f}%)")
        print(f"Days with VIX > 50: {(vix_df['vix'] > 50).sum():,} ({(vix_df['vix'] > 50).sum()/len(vix_df)*100:.1f}%)")

        print("\nCitation:")
        print("Chicago Board Options Exchange, CBOE Volatility Index: VIX [VIXCLS],")
        print("retrieved from FRED, Federal Reserve Bank of St. Louis;")
        print("https://fred.stlouisfed.org/series/VIXCLS")

        return vix_df

    except Exception as e:
        print(f"\nError: {str(e)}")
        print("\nTroubleshooting:")
        print("1. Verify your API key is correct in the .env file")
        print("2. Check your internet connection")
        return None


# =============================================================================
# FAMA-FRENCH DATA
# =============================================================================
def download_fama_french_factors(
    start_date='1926-07-01',
    end_date=None,
    frequency='daily',
    save_path=None
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
    save_path : str, optional
        Path to save CSV file

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

    try:
        # Download 3-Factor Model
        print("Downloading Fama-French 3-Factor Model...")
        ff3 = pdr.DataReader(
            freq_map[frequency],
            'famafrench',
            start=start_date,
            end=end_date
        )[0]
        results['FF3'] = ff3

        # Download 5-Factor Model
        print("Downloading Fama-French 5-Factor Model...")
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
        print("Downloading Momentum Factor...")
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
        if save_path:
            os.makedirs(save_path, exist_ok=True)

            for name, df in results.items():
                filename = f"{name}_{frequency}_{start_date}_to_{end_date}.csv"
                filepath = os.path.join(save_path, filename)
                df.to_csv(filepath)
                print(f"Saved {name} to {filepath}")

        print(f"\nDownload complete! Date range: {ff3.index[0]} to {ff3.index[-1]}")
        return results

    except Exception as e:
        print(f"Error downloading data: {e}")
        return None


def download_industry_portfolios(
    num_industries=10,
    start_date='1926-07-01',
    end_date=None,
    frequency='daily',
    save_path=None
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
    save_path : str, optional
        Path to save CSV file

    Returns:
    --------
    pd.DataFrame : Industry portfolio returns
    """
    if end_date is None:
        end_date = datetime.today().strftime('%Y-%m-%d')

    freq_suffix = '_daily' if frequency == 'daily' else ''
    dataset_name = f'{num_industries}_Industry_Portfolios{freq_suffix}'

    try:
        print(f"Downloading {num_industries} Industry Portfolios...")
        ind_portfolios = pdr.DataReader(
            dataset_name,
            'famafrench',
            start=start_date,
            end=end_date
        )[0]

        if save_path:
            os.makedirs(save_path, exist_ok=True)
            filename = f"Industry_{num_industries}_{frequency}_{start_date}_to_{end_date}.csv"
            filepath = os.path.join(save_path, filename)
            ind_portfolios.to_csv(filepath)
            print(f"Saved to {filepath}")

        return ind_portfolios

    except Exception as e:
        print(f"Error downloading industry portfolios: {e}")
        return None


# =============================================================================
# SEC FORM 4 DATA
# =============================================================================
class Form4Downloader:
    """Download and parse SEC Form 4 filings."""

    def __init__(self, output_dir='./sec_form4_data'):
        self.output_dir = output_dir
        os.makedirs(output_dir, exist_ok=True)
        self.headers = {
            'User-Agent': 'Ashley Roseboro ashley@roseboroholdings.com',
            'Accept-Encoding': 'gzip, deflate',
            'Host': 'www.sec.gov'
        }

    def get_company_cik(self, ticker: str) -> str:
        """Get company CIK from ticker symbol."""
        # Implementation placeholder - would use SEC EDGAR API
        _ = ticker  # Acknowledge parameter until implementation
        return None

    def download_form4_filings(
        self,
        ticker: str,
        cik: str,
        start_date: str = '2000-01-01',
        end_date: str = '2025-12-31'
    ) -> List[Dict]:
        """
        Download all Form 4 filings for a company within date range.

        Returns:
        --------
        List[Dict] : List of filing metadata
        """
        filings = []

        try:
            cik_no_leading = str(int(cik))
            url = (
                f"https://www.sec.gov/cgi-bin/browse-edgar?"
                f"action=getcompany&CIK={cik_no_leading}&type=4&"
                f"dateb=&owner=include&count=100&search_text="
            )

            response = requests.get(url, headers=self.headers)
            time.sleep(0.2)

            if response.status_code == 200:
                soup = BeautifulSoup(response.text, 'html.parser')
                filing_table = soup.find('table', {'class': 'tableFile2'})

                if not filing_table:
                    print(f"  No Form 4 filings table found for {ticker}")
                    return []

                rows = filing_table.find_all('tr')[1:]

                for row in rows:
                    cols = row.find_all('td')
                    if len(cols) >= 4:
                        filing_type = cols[0].text.strip()

                        if filing_type == '4':
                            filing_date = cols[3].text.strip()

                            if start_date <= filing_date <= end_date:
                                doc_link = cols[1].find('a')
                                if doc_link:
                                    doc_url = 'https://www.sec.gov' + doc_link.get('href')
                                    accession = cols[4].text.strip()

                                    filing_info = {
                                        'ticker': ticker,
                                        'cik': cik,
                                        'filing_date': filing_date,
                                        'accession_number': accession,
                                        'filing_url': doc_url
                                    }
                                    filings.append(filing_info)

                if len(filings) > 0:
                    print(f"  Found {len(filings)} Form 4 filings for {ticker}")
                else:
                    print(f"  No Form 4 filings in date range for {ticker}")

                return filings
            else:
                print(f"  HTTP {response.status_code} for {ticker}")
                return []

        except Exception as e:
            print(f"  Error downloading Form 4 filings for {ticker}: {e}")
            return []

    def parse_form4_xml(self, filing_url: str) -> List[Dict]:
        """
        Parse Form 4 to extract insider trading details.

        Returns:
        --------
        List[Dict] : Parsed transaction data
        """
        try:
            response = requests.get(filing_url, headers=self.headers)
            time.sleep(0.2)
        except Exception as e:
            print(f"Error fetching filing page: {e}")
            return []

        if response.status_code != 200:
            return []

        soup = BeautifulSoup(response.text, 'html.parser')

        # Find the XML document link
        xml_link = None
        for link in soup.find_all('a'):
            href = link.get('href', '')
            if '.xml' in href.lower() and 'primary_doc' not in href:
                xml_link = (
                    'https://www.sec.gov' + href
                    if not href.startswith('http')
                    else href
                )
                break

        if not xml_link:
            print("    No XML document found")
            return []

        # Fetch and parse the XML
        xml_response = requests.get(xml_link, headers=self.headers)
        time.sleep(0.2)

        if xml_response.status_code != 200:
            return []

        xml_soup = BeautifulSoup(xml_response.content, 'xml')

        # Extract reporting owner
        reporting_owner = xml_soup.find('reportingOwner')
        if reporting_owner:
            owner_name_tag = reporting_owner.find('rptOwnerName')
            owner_name = owner_name_tag.text if owner_name_tag else 'Unknown'
        else:
            owner_name = 'Unknown'

        transactions = []

        # Parse non-derivative transactions
        for trans in xml_soup.find_all('nonDerivativeTransaction'):
            try:
                trans_date_tag = trans.find('transactionDate')
                trans_code_tag = trans.find('transactionCode')
                shares_tag = trans.find('transactionShares')
                price_tag = trans.find('transactionPricePerShare')
                acq_disp_tag = trans.find('transactionAcquiredDisposedCode')
                shares_owned_tag = trans.find('sharesOwnedFollowingTransaction')

                transaction = {
                    'owner_name': owner_name,
                    'transaction_date': (
                        trans_date_tag.find('value').text
                        if trans_date_tag and trans_date_tag.find('value')
                        else None
                    ),
                    'transaction_code': (
                        trans_code_tag.text if trans_code_tag else None
                    ),
                    'shares': (
                        float(shares_tag.find('value').text)
                        if shares_tag and shares_tag.find('value')
                        else 0
                    ),
                    'price_per_share': (
                        float(price_tag.find('value').text)
                        if price_tag and price_tag.find('value')
                        else None
                    ),
                    'acquired_disposed': (
                        acq_disp_tag.find('value').text
                        if acq_disp_tag and acq_disp_tag.find('value')
                        else None
                    ),
                    'shares_owned_after': (
                        float(shares_owned_tag.find('value').text)
                        if shares_owned_tag and shares_owned_tag.find('value')
                        else None
                    )
                }
                transactions.append(transaction)
            except Exception:
                continue

        return transactions

    def build_form4_dataset(
        self,
        culture_war_companies: List[str],
        start_date: str = '2000-01-01',
        end_date: str = '2025-12-31',
        save_csv: bool = True
    ) -> pd.DataFrame:
        """
        Build complete Form 4 dataset for all culture war companies.

        Parameters:
        -----------
        culture_war_companies : List[str]
            List of ticker symbols
        start_date : str
            Start date for filing search
        end_date : str
            End date for filing search
        save_csv : bool
            Whether to save to CSV

        Returns:
        --------
        pd.DataFrame : Complete Form 4 transaction dataset
        """
        all_transactions = []

        for ticker in culture_war_companies:
            print(f"\nProcessing {ticker}...")

            cik = self.get_company_cik(ticker)
            if not cik:
                continue

            filings = self.download_form4_filings(ticker, cik, start_date, end_date)

            for filing in filings:
                transactions = self.parse_form4_xml(filing['filing_url'])

                for trans in transactions:
                    trans['ticker'] = ticker
                    trans['cik'] = cik
                    trans['filing_date'] = filing['filing_date']
                    trans['accession_number'] = filing['accession_number']
                    trans['filing_url'] = filing['filing_url']

                all_transactions.extend(transactions)
                time.sleep(0.15)

        df = pd.DataFrame(all_transactions)

        if len(df) > 0:
            df['transaction_date'] = pd.to_datetime(df['transaction_date'])
            df['filing_date'] = pd.to_datetime(df['filing_date'])
            df['transaction_value'] = df['shares'] * df['price_per_share']
            df = df.sort_values('transaction_date')

            if save_csv:
                output_file = os.path.join(
                    self.output_dir,
                    f'form4_transactions_{start_date}_to_{end_date}.csv'
                )
                df.to_csv(output_file, index=False)
                print(f"\nSaved {len(df)} transactions to {output_file}")

        return df


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
            print("Warning: Found company names but not tickers. "
                  "You'll need to map company names to tickers.")
            return [c for c in companies if pd.notna(c)]

    print(f"Available columns: {culture_war_data.columns.tolist()}")
    raise ValueError("Cannot find ticker or company column in culture war data")


# =============================================================================
# NEWS DATA
# =============================================================================
@dataclass
class NewsArticle:
    """Structure for news articles."""
    ticker: str
    company_name: str
    source: str
    title: str
    url: str
    published_date: Optional[datetime]
    snippet: Optional[str]
    culture_war_event: Optional[str] = None
    event_date: Optional[datetime] = None
    search_query: Optional[str] = None
    author: Optional[str] = None
    section: Optional[str] = None
    word_count: Optional[int] = None


class CompanyNewsAggregator:
    """Aggregates news from Guardian, Reddit, and NYT (2000-2025)."""

    def __init__(
        self,
        guardian_api_key: Optional[str] = None,
        nyt_api_key: Optional[str] = None,
        reddit_client_id: Optional[str] = None,
        reddit_client_secret: Optional[str] = None,
        reddit_user_agent: Optional[str] = None
    ):
        """
        Initialize the news aggregator.

        Args:
            guardian_api_key: The Guardian API key
            nyt_api_key: New York Times API key
            reddit_client_id: Reddit API client ID
            reddit_client_secret: Reddit API client secret
            reddit_user_agent: Reddit user agent string
        """
        self.guardian_api_key = guardian_api_key or os.getenv('GUARDIAN_API_KEY')
        self.nyt_api_key = nyt_api_key or os.getenv('NYT_API_KEY')

        self.reddit = None
        if all([reddit_client_id, reddit_client_secret, reddit_user_agent]):
            try:
                self.reddit = praw.Reddit(
                    client_id=reddit_client_id,
                    client_secret=reddit_client_secret,
                    user_agent=reddit_user_agent
                )
                logger.info("Reddit client initialized successfully")
            except Exception as e:
                logger.warning(f"Failed to initialize Reddit client: {e}")

        self.last_nyt_request = None
        self.nyt_requests_this_minute = 0

    def _build_search_queries(
        self,
        company_name: str,
        culture_war_event: str = None,
        include_insider_trading: bool = True
    ) -> List[str]:
        """
        Build search queries for news articles about culture war events and insider trading.

        Args:
            company_name: Full company name
            culture_war_event: Description of the culture war event
            include_insider_trading: Whether to include insider trading queries

        Returns:
            List of search query strings
        """
        queries = [company_name]

        if culture_war_event:
            # Extract key terms from the culture war event
            event_lower = culture_war_event.lower()

            # Add the full company + event query
            queries.append(f"{company_name} {culture_war_event[:50]}")

            # Add specific keyword-based queries
            if 'boycott' in event_lower:
                queries.append(f"{company_name} boycott")
            if 'pride' in event_lower or 'lgbtq' in event_lower or 'trans' in event_lower:
                queries.append(f"{company_name} LGBTQ")
                queries.append(f"{company_name} Pride transgender")
            if 'backlash' in event_lower:
                queries.append(f"{company_name} backlash")
            if 'controversy' in event_lower or 'controversial' in event_lower:
                queries.append(f"{company_name} controversy")
            if 'campaign' in event_lower or 'ad' in event_lower:
                queries.append(f"{company_name} advertisement controversy")
            if 'racist' in event_lower or 'racial' in event_lower or 'race' in event_lower:
                queries.append(f"{company_name} racism")
            if 'political' in event_lower or 'conservative' in event_lower or 'liberal' in event_lower:
                queries.append(f"{company_name} political")
            if 'kaepernick' in event_lower:
                queries.append(f"{company_name} Kaepernick")
            if 'dylan mulvaney' in event_lower:
                queries.append(f"{company_name} Dylan Mulvaney")

        # Add insider trading queries
        if include_insider_trading:
            queries.append(f"{company_name} insider trading")
            queries.append(f"{company_name} executive stock sales")
            queries.append(f"{company_name} SEC filing insider")

        # Remove duplicates while preserving order
        seen = set()
        unique_queries = []
        for q in queries:
            if q.lower() not in seen:
                seen.add(q.lower())
                unique_queries.append(q)

        return unique_queries[:8]  # Limit to 8 queries

    def search_guardian(
        self,
        ticker: str,
        company_name: str,
        culture_war_event: str = None,
        event_date: datetime = None,
        start_date: datetime = None,
        end_date: datetime = None,
        max_results: int = 200
    ) -> List[NewsArticle]:
        """
        Search The Guardian API for articles about culture war events.

        Args:
            ticker: Company ticker symbol
            company_name: Full company name
            culture_war_event: Description of the culture war event
            event_date: Date of the culture war event (for reference)
            start_date: Start date for search
            end_date: End date for search
            max_results: Maximum number of results

        Returns:
            List of NewsArticle objects
        """
        articles = []

        if not self.guardian_api_key:
            logger.warning("Guardian API key not provided. Skipping Guardian search.")
            return articles

        # Set date range
        if start_date is None:
            start_date = datetime(2000, 1, 1)
        if end_date is None:
            end_date = datetime(2025, 12, 31)

        # Build search queries
        search_queries = self._build_search_queries(company_name, culture_war_event)

        logger.info(f"  Guardian: Searching for {ticker} with {len(search_queries)} queries")
        logger.info(f"    Date range: {start_date.date()} to {end_date.date()}")

        try:
            for query in search_queries:
                page = 1
                total_pages = 1
                query_articles = 0
                max_per_query = max_results // len(search_queries)

                while page <= total_pages and query_articles < max_per_query:
                    url = "https://content.guardianapis.com/search"
                    params = {
                        'q': query,
                        'from-date': start_date.strftime('%Y-%m-%d'),
                        'to-date': end_date.strftime('%Y-%m-%d'),
                        'page': page,
                        'page-size': 50,
                        'show-fields': 'headline,trailText,wordcount,byline',
                        'show-tags': 'all',
                        'api-key': self.guardian_api_key
                    }

                    try:
                        response = requests.get(url, params=params, timeout=15)
                        response.raise_for_status()
                        data = response.json()

                        if data['response']['status'] == 'ok':
                            total_pages = min(data['response']['pages'], 3)

                            for item in data['response']['results']:
                                try:
                                    pub_date = datetime.strptime(
                                        item['webPublicationDate'],
                                        '%Y-%m-%dT%H:%M:%SZ'
                                    )

                                    fields = item.get('fields', {})

                                    article = NewsArticle(
                                        ticker=ticker,
                                        company_name=company_name,
                                        source='The Guardian',
                                        title=fields.get('headline', item['webTitle']),
                                        url=item['webUrl'],
                                        published_date=pub_date,
                                        snippet=fields.get('trailText', ''),
                                        culture_war_event=culture_war_event,
                                        event_date=event_date,
                                        search_query=query,
                                        author=fields.get('byline'),
                                        section=item.get('sectionName'),
                                        word_count=fields.get('wordcount')
                                    )
                                    articles.append(article)
                                    query_articles += 1

                                except Exception as e:
                                    logger.debug(f"Error parsing Guardian article: {e}")
                                    continue

                        page += 1
                        time.sleep(0.2)

                    except requests.exceptions.RequestException as e:
                        logger.warning(f"Error fetching Guardian page {page}: {e}")
                        time.sleep(2)
                        break

                logger.info(f"    Query '{query[:40]}...': {query_articles} articles")
                time.sleep(0.3)

        except Exception as e:
            logger.error(f"Error in Guardian search for {ticker}: {e}")

        logger.info(f"  Guardian total: {len(articles)} articles")
        return articles

    def search_nyt(
        self,
        ticker: str,
        company_name: str,
        culture_war_event: str = None,
        event_date: datetime = None,
        start_date: datetime = None,
        end_date: datetime = None,
        max_results: int = 200
    ) -> List[NewsArticle]:
        """
        Search New York Times Article Search API for culture war event articles.

        NYT API rate limit: 500 requests per day, 5 requests per minute

        Args:
            ticker: Company ticker symbol
            company_name: Full company name
            culture_war_event: Description of the culture war event
            event_date: Date of the culture war event (for reference)
            start_date: Start date for search
            end_date: End date for search
            max_results: Maximum number of results

        Returns:
            List of NewsArticle objects
        """
        articles = []

        if not self.nyt_api_key:
            logger.warning("NYT API key not provided. Skipping NYT search.")
            return articles

        # Set date range
        if start_date is None:
            start_date = datetime(2000, 1, 1)
        if end_date is None:
            end_date = datetime(2025, 12, 31)

        # Build search queries
        search_queries = self._build_search_queries(company_name, culture_war_event)

        logger.info(f"  NYT: Searching for {ticker} with {len(search_queries)} queries")
        logger.info(f"    Date range: {start_date.date()} to {end_date.date()}")

        try:
            for query in search_queries:
                page = 0
                query_articles = 0
                max_per_query = max_results // len(search_queries)

                while page < 10 and query_articles < max_per_query:
                    self._nyt_rate_limit()

                    url = "https://api.nytimes.com/svc/search/v2/articlesearch.json"
                    params = {
                        'q': query,
                        'begin_date': start_date.strftime('%Y%m%d'),
                        'end_date': end_date.strftime('%Y%m%d'),
                        'page': page,
                        'api-key': self.nyt_api_key,
                        'sort': 'relevance'
                    }

                    try:
                        response = requests.get(url, params=params, timeout=15)

                        if response.status_code == 429:
                            logger.warning("NYT rate limit hit, waiting 60 seconds...")
                            time.sleep(60)
                            continue

                        response.raise_for_status()
                        data = response.json()

                        if data['status'] == 'OK':
                            docs = data['response']['docs']

                            if not docs:
                                break

                            for doc in docs:
                                try:
                                    pub_date = datetime.strptime(
                                        doc['pub_date'],
                                        '%Y-%m-%dT%H:%M:%S%z'
                                    ).replace(tzinfo=None)

                                    author = None
                                    if doc.get('byline', {}).get('original'):
                                        author = doc['byline']['original']

                                    article = NewsArticle(
                                        ticker=ticker,
                                        company_name=company_name,
                                        source='New York Times',
                                        title=doc.get('headline', {}).get('main', ''),
                                        url=doc.get('web_url', ''),
                                        published_date=pub_date,
                                        snippet=doc.get('snippet', ''),
                                        culture_war_event=culture_war_event,
                                        event_date=event_date,
                                        search_query=query,
                                        author=author,
                                        section=doc.get('section_name'),
                                        word_count=doc.get('word_count')
                                    )
                                    articles.append(article)
                                    query_articles += 1

                                except Exception as e:
                                    logger.debug(f"Error parsing NYT article: {e}")
                                    continue

                        page += 1

                    except requests.exceptions.RequestException as e:
                        logger.warning(f"Error fetching NYT page {page}: {e}")
                        time.sleep(5)
                        break

                logger.info(f"    Query '{query[:40]}...': {query_articles} articles")
                time.sleep(1)

        except Exception as e:
            logger.error(f"Error in NYT search for {ticker}: {e}")

        logger.info(f"  NYT total: {len(articles)} articles")
        return articles

    def _nyt_rate_limit(self):
        """Enforce NYT API rate limit: 5 requests per minute."""
        now = datetime.now()

        if self.last_nyt_request:
            time_diff = (now - self.last_nyt_request).total_seconds()

            if time_diff < 60:
                if self.nyt_requests_this_minute >= 5:
                    sleep_time = 60 - time_diff + 1
                    logger.info(f"NYT rate limit: sleeping {sleep_time:.1f}s")
                    time.sleep(sleep_time)
                    self.nyt_requests_this_minute = 0
            else:
                self.nyt_requests_this_minute = 0

        self.last_nyt_request = datetime.now()
        self.nyt_requests_this_minute += 1

    def search_reddit(
        self,
        ticker: str,
        company_name: str,
        culture_war_event: str = None,
        event_date: datetime = None,
        start_date: datetime = None,
        end_date: datetime = None,
        max_results: int = 200,
        subreddits: List[str] = None
    ) -> List[NewsArticle]:
        """
        Search Reddit for company culture war event mentions.

        Args:
            ticker: Company ticker symbol
            company_name: Full company name
            culture_war_event: Description of the culture war event
            event_date: Date of the culture war event (for reference)
            start_date: Start date for search
            end_date: End date for search
            max_results: Maximum number of results per subreddit
            subreddits: List of subreddits to search

        Returns:
            List of NewsArticle objects
        """
        articles = []

        if not self.reddit:
            logger.warning("Reddit client not initialized. Skipping Reddit search.")
            return articles

        # Set date range
        if start_date is None:
            start_date = datetime(2000, 1, 1)
        if end_date is None:
            end_date = datetime(2025, 12, 31)

        if subreddits is None:
            subreddits = [
                'news', 'business', 'investing', 'stocks', 'wallstreetbets',
                'finance', 'economy', 'worldnews', 'politics', 'technology',
                'entertainment', 'Conservative', 'progressive', 'capitalism',
                'OutOfTheLoop', 'nottheonion'
            ]

        # Build search queries using the helper method
        search_queries = self._build_search_queries(company_name, culture_war_event)

        logger.info(f"  Reddit: Searching for {ticker} with {len(search_queries)} queries")
        logger.info(f"    Date range: {start_date.date()} to {end_date.date()}")

        try:
            for subreddit_name in subreddits:
                try:
                    subreddit = self.reddit.subreddit(subreddit_name)

                    for query in search_queries:
                        try:
                            for submission in subreddit.search(
                                query,
                                time_filter='all',
                                limit=max_results // len(search_queries),
                                sort='relevance'
                            ):
                                created = datetime.fromtimestamp(submission.created_utc)

                                if created < start_date or created > end_date:
                                    continue

                                article = NewsArticle(
                                    ticker=ticker,
                                    company_name=company_name,
                                    source=f'Reddit r/{subreddit_name}',
                                    title=submission.title,
                                    url=f"https://reddit.com{submission.permalink}",
                                    published_date=created,
                                    snippet=(
                                        submission.selftext[:500]
                                        if submission.selftext
                                        else None
                                    ),
                                    culture_war_event=culture_war_event,
                                    event_date=event_date,
                                    search_query=query,
                                    author=(
                                        str(submission.author)
                                        if submission.author
                                        else None
                                    )
                                )
                                articles.append(article)

                            time.sleep(0.5)

                        except Exception as e:
                            logger.debug(
                                f"Error searching '{query}' in r/{subreddit_name}: {e}"
                            )
                            continue

                    time.sleep(1)

                except Exception as e:
                    logger.debug(f"Error accessing r/{subreddit_name}: {e}")
                    continue

        except Exception as e:
            logger.error(f"Error in Reddit search for {ticker}: {e}")

        logger.info(f"  Reddit total: {len(articles)} posts")
        return articles

    def aggregate_culture_war_news(
        self,
        culture_war_df: pd.DataFrame,
        start_date: str = '2000-01-01',
        end_date: str = '2025-12-31',
        max_results_per_source: int = 200,
        sources: List[str] = None,
        checkpoint_file: str = 'news_checkpoint.csv'
    ) -> pd.DataFrame:
        """
        Aggregate news from all sources for culture war events.

        Searches for news about each company's culture war event across
        the full date range. Also searches for insider trading news
        related to these companies.

        Args:
            culture_war_df: DataFrame with columns:
                - 'Company': Company name
                - 'Ticker': Stock ticker symbol
                - 'Culture War Event': Description of the event
                - 'Event Date': Date of the event
            start_date: Start date for search (default: '2000-01-01')
            end_date: End date for search (default: '2025-12-31')
            max_results_per_source: Max results per source per event
            sources: List of sources to use ['guardian', 'nyt', 'reddit']
            checkpoint_file: File to save progress

        Returns:
            DataFrame with all aggregated news articles
        """
        if sources is None:
            sources = ['guardian', 'nyt', 'reddit']

        all_articles = []

        checkpoint_path = Path(checkpoint_file)
        processed_events = set()

        if checkpoint_path.exists():
            try:
                checkpoint_df = pd.read_csv(checkpoint_path)
                checkpoint_df['published_date'] = pd.to_datetime(
                    checkpoint_df['published_date']
                )
                all_articles.extend(checkpoint_df.to_dict('records'))

                # Track processed events by ticker + event_date
                for _, row in checkpoint_df.iterrows():
                    ticker = row.get('ticker', '')
                    event_date = row.get('event_date', '')
                    if ticker and event_date:
                        processed_events.add(f"{ticker}_{event_date}")

                logger.info(f"Loaded checkpoint with {len(checkpoint_df)} articles")
                logger.info(f"Processed events: {len(processed_events)}")
            except Exception as e:
                logger.warning(f"Error loading checkpoint: {e}")

        total_events = len(culture_war_df)

        for idx, row in culture_war_df.iterrows():
            # Extract event details
            company_name = row.get('Company', '')
            ticker = row.get('Ticker', '')
            culture_war_event = row.get('Culture War Event', '')
            event_date_raw = row.get('Event Date', None)

            # Skip if no ticker
            if not ticker or pd.isna(ticker) or ticker in ['Private', 'N/A']:
                logger.info(f"Skipping {company_name} - no valid ticker")
                continue

            # Parse event date
            event_date = None
            if event_date_raw and not pd.isna(event_date_raw):
                try:
                    event_date = pd.to_datetime(event_date_raw)
                except Exception:
                    logger.warning(f"Could not parse event date: {event_date_raw}")

            # Check if already processed
            event_key = f"{ticker}_{event_date}"
            if event_key in processed_events:
                logger.info(f"Skipping {ticker} - already processed")
                continue

            logger.info(f"\n{'=' * 60}")
            logger.info(f"[{idx+1}/{total_events}] {company_name} ({ticker})")
            logger.info(f"Event: {culture_war_event[:80]}...")
            if event_date:
                logger.info(f"Event Date: {event_date.date()}")
            logger.info(f"{'=' * 60}")

            event_articles = []

            # Parse date range
            search_start = datetime.strptime(start_date, '%Y-%m-%d')
            search_end = datetime.strptime(end_date, '%Y-%m-%d')

            # Search Guardian
            if 'guardian' in sources:
                logger.info("Searching The Guardian...")
                articles = self.search_guardian(
                    ticker=ticker,
                    company_name=company_name,
                    culture_war_event=culture_war_event,
                    event_date=event_date,
                    start_date=search_start,
                    end_date=search_end,
                    max_results=max_results_per_source
                )
                event_articles.extend(articles)

            # Search NYT
            if 'nyt' in sources:
                logger.info("Searching New York Times...")
                articles = self.search_nyt(
                    ticker=ticker,
                    company_name=company_name,
                    culture_war_event=culture_war_event,
                    event_date=event_date,
                    start_date=search_start,
                    end_date=search_end,
                    max_results=max_results_per_source
                )
                event_articles.extend(articles)

            # Search Reddit
            if 'reddit' in sources:
                logger.info("Searching Reddit...")
                articles = self.search_reddit(
                    ticker=ticker,
                    company_name=company_name,
                    culture_war_event=culture_war_event,
                    event_date=event_date,
                    start_date=search_start,
                    end_date=search_end,
                    max_results=max_results_per_source
                )
                event_articles.extend(articles)

            # Save progress
            if event_articles:
                all_articles.extend([vars(a) for a in event_articles])
                checkpoint_df = pd.DataFrame(all_articles)
                checkpoint_df.to_csv(checkpoint_path, index=False)
                logger.info(f"Checkpoint saved: {len(all_articles)} total articles")

            processed_events.add(event_key)
            time.sleep(2)

        if len(all_articles) > 0:
            df = pd.DataFrame(all_articles)

            original_len = len(df)
            df = df.drop_duplicates(subset=['url'], keep='first')
            logger.info(f"\nRemoved {original_len - len(df)} duplicate articles")

            df['published_date'] = pd.to_datetime(df['published_date'])
            df = df.sort_values('published_date', ascending=False)
        else:
            df = pd.DataFrame()

        return df

    def save_news(self, news_df: pd.DataFrame, output_path: str):
        """Save news articles to CSV with summary statistics."""
        if len(news_df) > 0:
            news_df.to_csv(output_path, index=False)
            logger.info(f"\n{'=' * 60}")
            logger.info(f"SAVED: {len(news_df)} articles to {output_path}")
            logger.info(f"{'=' * 60}")

            logger.info("\n=== SUMMARY STATISTICS ===")
            logger.info(f"Total articles: {len(news_df):,}")
            logger.info(
                f"Date range: {news_df['published_date'].min().date()} "
                f"to {news_df['published_date'].max().date()}"
            )
            logger.info(f"Unique companies: {news_df['ticker'].nunique()}")

            logger.info("\n--- Articles by Source ---")
            for source, count in news_df['source'].value_counts().items():
                logger.info(f"  {source}: {count:,}")

            logger.info("\n--- Top 10 Companies by Article Count ---")
            for ticker, count in news_df['ticker'].value_counts().head(10).items():
                logger.info(f"  {ticker}: {count:,}")

            logger.info("\n--- Articles by Year ---")
            yearly = news_df['published_date'].dt.year.value_counts().sort_index()
            for year, count in yearly.items():
                logger.info(f"  {year}: {count:,}")
        else:
            logger.warning("No articles to save")


# =============================================================================
# INFLATION DATA
# =============================================================================
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
    cache_file = os.path.join(cache_path, f'inflation_data_{start_date}_{end_date}.pkl')

    if os.path.exists(cache_file):
        print(f"Loading cached inflation data from {cache_file}")
        return pd.read_pickle(cache_file)

    print("Downloading inflation data from FRED...")

    series = {
        'CPI': 'CPIAUCSL',
        'Core_CPI': 'CPILFESL',
        'PCE': 'PCEPI',
        'Core_PCE': 'PCEPILFE',
        'PPI': 'PPIACO',
        'GDP_Deflator': 'GDPDEF',
    }

    try:
        raw_data = {}
        for name, code in series.items():
            print(f"  Downloading {name} ({code})...")
            df = pdr.DataReader(code, 'fred', start=start_date, end=end_date)
            raw_data[name] = df

        inflation_raw = pd.DataFrame({
            name: df.iloc[:, 0] for name, df in raw_data.items()
        })

        print("Calculating year-over-year changes...")
        inflation_yoy = inflation_raw.pct_change(periods=12) * 100
        inflation_yoy.columns = [f'{col}_YoY' for col in inflation_yoy.columns]

        print("Calculating month-over-month changes...")
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

        pd.to_pickle(result, cache_file)
        print(f"Cached inflation data to {cache_file}")

        print("\n=== Inflation Data Summary ===")
        print(f"Date range: {inflation_raw.index.min()} to {inflation_raw.index.max()}")
        print(f"Observations: {len(inflation_raw)}")
        print("\nLatest values (Year-over-Year %):")
        print(inflation_yoy.iloc[-1])

        return result

    except Exception as e:
        print(f"Error downloading inflation data: {e}")
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
    cache_file = os.path.join(
        cache_path,
        f'inflation_expectations_{start_date}_{end_date}.pkl'
    )

    if os.path.exists(cache_file):
        print(f"Loading cached inflation expectations from {cache_file}")
        return pd.read_pickle(cache_file)

    print("Downloading inflation expectations data from FRED...")

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
        print("\n--- Breakeven Inflation (Market-Based) ---")
        breakeven_data = {}
        for name, code in breakeven_series.items():
            try:
                print(f"  Downloading {name} ({code})...")
                df = pdr.DataReader(code, 'fred', start=start_date, end=end_date)
                breakeven_data[name] = df.iloc[:, 0]
            except Exception as e:
                print(f"  Warning: Could not download {name}: {e}")

        breakeven_df = pd.DataFrame(breakeven_data)

        # Download survey expectations
        print("\n--- Survey-Based Expectations ---")
        survey_data = {}
        for name, code in survey_series.items():
            try:
                print(f"  Downloading {name} ({code})...")
                df = pdr.DataReader(code, 'fred', start=start_date, end=end_date)
                survey_data[name] = df.iloc[:, 0]
            except Exception as e:
                print(f"  Warning: Could not download {name}: {e}")

        survey_df = pd.DataFrame(survey_data)

        # Download Fed measures
        print("\n--- Federal Reserve Inflation Measures ---")
        fed_data = {}
        for name, code in fed_series.items():
            try:
                print(f"  Downloading {name} ({code})...")
                df = pdr.DataReader(code, 'fred', start=start_date, end=end_date)
                fed_data[name] = df.iloc[:, 0]
            except Exception as e:
                print(f"  Warning: Could not download {name}: {e}")

        fed_df = pd.DataFrame(fed_data)

        # Combine all data
        combined_df = pd.concat([breakeven_df, survey_df, fed_df], axis=1)

        result = {
            'breakeven': breakeven_df,
            'survey': survey_df,
            'fed_measures': fed_df,
            'combined': combined_df
        }

        pd.to_pickle(result, cache_file)
        print(f"\nCached inflation expectations to {cache_file}")

        # Print summary
        print("\n" + "=" * 60)
        print("=== Inflation Expectations Summary ===")
        print("=" * 60)

        if len(breakeven_df) > 0:
            print(f"\nBreakeven Inflation (Market-Based):")
            print(f"  Date range: {breakeven_df.index.min()} to {breakeven_df.index.max()}")
            print(f"  Series: {list(breakeven_df.columns)}")
            print(f"  Latest values:")
            for col in breakeven_df.columns:
                latest = breakeven_df[col].dropna().iloc[-1] if len(breakeven_df[col].dropna()) > 0 else 'N/A'
                print(f"    {col}: {latest:.2f}%" if isinstance(latest, float) else f"    {col}: {latest}")

        if len(survey_df) > 0:
            print(f"\nSurvey-Based Expectations:")
            print(f"  Date range: {survey_df.index.min()} to {survey_df.index.max()}")
            print(f"  Series: {list(survey_df.columns)}")
            print(f"  Latest values:")
            for col in survey_df.columns:
                latest = survey_df[col].dropna().iloc[-1] if len(survey_df[col].dropna()) > 0 else 'N/A'
                print(f"    {col}: {latest:.2f}%" if isinstance(latest, float) else f"    {col}: {latest}")

        if len(fed_df) > 0:
            print(f"\nFederal Reserve Measures:")
            print(f"  Date range: {fed_df.index.min()} to {fed_df.index.max()}")
            print(f"  Series: {list(fed_df.columns)}")
            print(f"  Latest values:")
            for col in fed_df.columns:
                latest = fed_df[col].dropna().iloc[-1] if len(fed_df[col].dropna()) > 0 else 'N/A'
                print(f"    {col}: {latest:.2f}%" if isinstance(latest, float) else f"    {col}: {latest}")

        print("\n" + "=" * 60)
        print("Citation:")
        print("Federal Reserve Economic Data (FRED), Federal Reserve Bank of St. Louis")
        print("https://fred.stlouisfed.org/")
        print("=" * 60)

        return result

    except Exception as e:
        print(f"Error downloading inflation expectations data: {e}")
        return None


def load_comprehensive_inflation_data(
    start_date='2000-01-01',
    end_date=None,
    cache_path='./data/fred'
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
    cache_file = os.path.join(
        cache_path,
        f'comprehensive_inflation_{start_date}_{end_date}.pkl'
    )

    if os.path.exists(cache_file):
        print(f"Loading cached comprehensive inflation data from {cache_file}")
        return pd.read_pickle(cache_file)

    print("=" * 60)
    print("Loading Comprehensive Inflation Data (2000-2025)")
    print("=" * 60)

    # Load core inflation data
    print("\n[1/3] Loading core inflation measures...")
    core_data = load_inflation_data(start_date, end_date, cache_path)

    # Load expectations data
    print("\n[2/3] Loading inflation expectations...")
    expectations_data = load_inflation_expectations_data(start_date, end_date, cache_path)

    # Load component-level inflation
    print("\n[3/3] Loading component-level inflation...")
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

    component_data = {}
    for name, code in component_series.items():
        try:
            print(f"  Downloading {name} ({code})...")
            df = pdr.DataReader(code, 'fred', start=start_date, end=end_date)
            component_data[name] = df.iloc[:, 0]
        except Exception as e:
            print(f"  Warning: Could not download {name}: {e}")

    components_df = pd.DataFrame(component_data)

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

    pd.to_pickle(result, cache_file)
    print(f"\nCached comprehensive inflation data to {cache_file}")

    # Print final summary
    print("\n" + "=" * 60)
    print("=== Comprehensive Inflation Data Summary ===")
    print("=" * 60)
    print(f"Total series loaded: {len(combined_df.columns)}")
    print(f"Date range: {combined_df.index.min()} to {combined_df.index.max()}")
    print(f"Total observations: {len(combined_df)}")

    print("\n--- Series Categories ---")
    print(f"Core inflation measures: {len(core_data['combined'].columns) if core_data else 0}")
    print(f"Expectations measures: {len(expectations_data['combined'].columns) if expectations_data else 0}")
    print(f"Component measures: {len(components_yoy.columns)}")

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
    cache_file = os.path.join(cache_path, f'treasury_yields_{start_date}_{end_date}.pkl')

    if os.path.exists(cache_file):
        print(f"Loading cached Treasury yields from {cache_file}")
        return pd.read_pickle(cache_file)

    print("Downloading Treasury yield data from FRED...")

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
        print("\n--- Nominal Treasury Yields ---")
        nominal_data = {}
        for name, code in nominal_series.items():
            try:
                print(f"  Downloading {name} ({code})...")
                df = pdr.DataReader(code, 'fred', start=start_date, end=end_date)
                nominal_data[name] = df.iloc[:, 0]
            except Exception as e:
                print(f"  Warning: Could not download {name}: {e}")

        nominal_df = pd.DataFrame(nominal_data)

        # Download TIPS yields
        print("\n--- TIPS (Real) Yields ---")
        tips_data = {}
        for name, code in tips_series.items():
            try:
                print(f"  Downloading {name} ({code})...")
                df = pdr.DataReader(code, 'fred', start=start_date, end=end_date)
                tips_data[name] = df.iloc[:, 0]
            except Exception as e:
                print(f"  Warning: Could not download {name}: {e}")

        tips_df = pd.DataFrame(tips_data)

        # Combine all yields
        combined_df = pd.concat([nominal_df, tips_df], axis=1)

        result = {
            'nominal': nominal_df,
            'real': tips_df,
            'combined': combined_df
        }

        pd.to_pickle(result, cache_file)
        print(f"\nCached Treasury yields to {cache_file}")

        # Print summary
        print("\n" + "=" * 60)
        print("=== Treasury Yields Summary ===")
        print("=" * 60)
        print(f"Date range: {combined_df.index.min()} to {combined_df.index.max()}")
        print(f"Observations: {len(combined_df)}")
        print(f"Nominal series: {len(nominal_df.columns)}")
        print(f"TIPS series: {len(tips_df.columns)}")

        print("\nLatest yields (%):")
        for col in combined_df.columns:
            latest = combined_df[col].dropna().iloc[-1] if len(combined_df[col].dropna()) > 0 else 'N/A'
            print(f"  {col}: {latest:.2f}%" if isinstance(latest, float) else f"  {col}: {latest}")

        return result

    except Exception as e:
        print(f"Error downloading Treasury yields: {e}")
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
    cache_file = os.path.join(cache_path, f'policy_rates_{start_date}_{end_date}.pkl')

    if os.path.exists(cache_file):
        print(f"Loading cached policy rates from {cache_file}")
        return pd.read_pickle(cache_file)

    print("Downloading policy rates from FRED...")

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
        print("\n--- Federal Funds & Discount Rates ---")
        fed_data = {}
        for name, code in fed_series.items():
            try:
                print(f"  Downloading {name} ({code})...")
                df = pdr.DataReader(code, 'fred', start=start_date, end=end_date)
                fed_data[name] = df.iloc[:, 0]
            except Exception as e:
                print(f"  Warning: Could not download {name}: {e}")

        fed_df = pd.DataFrame(fed_data)

        # Download money market rates
        print("\n--- Money Market Rates ---")
        mm_data = {}
        for name, code in money_market_series.items():
            try:
                print(f"  Downloading {name} ({code})...")
                df = pdr.DataReader(code, 'fred', start=start_date, end=end_date)
                mm_data[name] = df.iloc[:, 0]
            except Exception as e:
                print(f"  Warning: Could not download {name}: {e}")

        mm_df = pd.DataFrame(mm_data)

        # Combine all rates
        combined_df = pd.concat([fed_df, mm_df], axis=1)

        result = {
            'fed_funds': fed_df,
            'money_market': mm_df,
            'combined': combined_df
        }

        pd.to_pickle(result, cache_file)
        print(f"\nCached policy rates to {cache_file}")

        # Print summary
        print("\n" + "=" * 60)
        print("=== Policy Rates Summary ===")
        print("=" * 60)
        print(f"Date range: {combined_df.index.min()} to {combined_df.index.max()}")
        print(f"Observations: {len(combined_df)}")

        print("\nLatest rates (%):")
        for col in combined_df.columns:
            latest = combined_df[col].dropna().iloc[-1] if len(combined_df[col].dropna()) > 0 else 'N/A'
            print(f"  {col}: {latest:.2f}%" if isinstance(latest, float) else f"  {col}: {latest}")

        return result

    except Exception as e:
        print(f"Error downloading policy rates: {e}")
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
    cache_file = os.path.join(cache_path, f'credit_spreads_{start_date}_{end_date}.pkl')

    if os.path.exists(cache_file):
        print(f"Loading cached credit spreads from {cache_file}")
        return pd.read_pickle(cache_file)

    print("Downloading credit spreads and corporate yields from FRED...")

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
        print("\n--- Corporate Bond Yields ---")
        corp_data = {}
        for name, code in corporate_series.items():
            try:
                print(f"  Downloading {name} ({code})...")
                df = pdr.DataReader(code, 'fred', start=start_date, end=end_date)
                corp_data[name] = df.iloc[:, 0]
            except Exception as e:
                print(f"  Warning: Could not download {name}: {e}")

        corp_df = pd.DataFrame(corp_data)

        # Download credit spreads
        print("\n--- Credit Spreads ---")
        spread_data = {}
        for name, code in spread_series.items():
            try:
                print(f"  Downloading {name} ({code})...")
                df = pdr.DataReader(code, 'fred', start=start_date, end=end_date)
                spread_data[name] = df.iloc[:, 0]
            except Exception as e:
                print(f"  Warning: Could not download {name}: {e}")

        spread_df = pd.DataFrame(spread_data)

        # Download mortgage rates
        print("\n--- Mortgage Rates ---")
        mortgage_data = {}
        for name, code in mortgage_series.items():
            try:
                print(f"  Downloading {name} ({code})...")
                df = pdr.DataReader(code, 'fred', start=start_date, end=end_date)
                mortgage_data[name] = df.iloc[:, 0]
            except Exception as e:
                print(f"  Warning: Could not download {name}: {e}")

        mortgage_df = pd.DataFrame(mortgage_data)

        # Combine all data
        combined_df = pd.concat([corp_df, spread_df, mortgage_df], axis=1)

        result = {
            'corporate': corp_df,
            'spreads': spread_df,
            'mortgage': mortgage_df,
            'combined': combined_df
        }

        pd.to_pickle(result, cache_file)
        print(f"\nCached credit spreads to {cache_file}")

        # Print summary
        print("\n" + "=" * 60)
        print("=== Credit Spreads Summary ===")
        print("=" * 60)
        print(f"Date range: {combined_df.index.min()} to {combined_df.index.max()}")
        print(f"Observations: {len(combined_df)}")
        print(f"Corporate yields: {len(corp_df.columns)}")
        print(f"Credit spreads: {len(spread_df.columns)}")
        print(f"Mortgage rates: {len(mortgage_df.columns)}")

        print("\nLatest values:")
        for col in combined_df.columns:
            latest = combined_df[col].dropna().iloc[-1] if len(combined_df[col].dropna()) > 0 else 'N/A'
            if isinstance(latest, float):
                unit = "bps" if "Spread" in col and latest > 10 else "%"
                print(f"  {col}: {latest:.2f}{unit}")
            else:
                print(f"  {col}: {latest}")

        return result

    except Exception as e:
        print(f"Error downloading credit spreads: {e}")
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
        print("Treasury data not available")
        return None

    nominal = treasury_data['nominal']
    metrics = pd.DataFrame(index=nominal.index)

    print("Calculating yield curve metrics...")

    # Yield curve slopes
    if 'Treasury_10Y' in nominal.columns and 'Treasury_2Y' in nominal.columns:
        metrics['Slope_10Y_2Y'] = nominal['Treasury_10Y'] - nominal['Treasury_2Y']
        print("  Calculated 10Y-2Y slope")

    if 'Treasury_10Y' in nominal.columns and 'Treasury_3M' in nominal.columns:
        metrics['Slope_10Y_3M'] = nominal['Treasury_10Y'] - nominal['Treasury_3M']
        print("  Calculated 10Y-3M slope")

    if 'Treasury_30Y' in nominal.columns and 'Treasury_5Y' in nominal.columns:
        metrics['Slope_30Y_5Y'] = nominal['Treasury_30Y'] - nominal['Treasury_5Y']
        print("  Calculated 30Y-5Y slope")

    if 'Treasury_2Y' in nominal.columns and 'Treasury_3M' in nominal.columns:
        metrics['Slope_2Y_3M'] = nominal['Treasury_2Y'] - nominal['Treasury_3M']
        print("  Calculated 2Y-3M slope")

    # Yield curve curvature (butterfly spread)
    if all(col in nominal.columns for col in ['Treasury_2Y', 'Treasury_5Y', 'Treasury_10Y']):
        metrics['Curvature_2_5_10'] = (
            2 * nominal['Treasury_5Y'] -
            nominal['Treasury_2Y'] -
            nominal['Treasury_10Y']
        )
        print("  Calculated 2-5-10 curvature (butterfly)")

    if all(col in nominal.columns for col in ['Treasury_3M', 'Treasury_2Y', 'Treasury_10Y']):
        metrics['Curvature_3M_2Y_10Y'] = (
            2 * nominal['Treasury_2Y'] -
            nominal['Treasury_3M'] -
            nominal['Treasury_10Y']
        )
        print("  Calculated 3M-2Y-10Y curvature")

    # Inversion indicators
    if 'Slope_10Y_2Y' in metrics.columns:
        metrics['Inverted_10Y_2Y'] = (metrics['Slope_10Y_2Y'] < 0).astype(int)
        print("  Calculated 10Y-2Y inversion indicator")

    if 'Slope_10Y_3M' in metrics.columns:
        metrics['Inverted_10Y_3M'] = (metrics['Slope_10Y_3M'] < 0).astype(int)
        print("  Calculated 10Y-3M inversion indicator")

    # Near-term forward spread (recession predictor)
    if 'Treasury_3M' in nominal.columns and 'Treasury_1Y' in nominal.columns:
        # 18-month forward 3-month rate minus current 3-month rate (approximation)
        metrics['Near_Term_Forward_Spread'] = nominal['Treasury_1Y'] - nominal['Treasury_3M']
        print("  Calculated near-term forward spread")

    # Level (average of key tenors)
    if all(col in nominal.columns for col in ['Treasury_2Y', 'Treasury_5Y', 'Treasury_10Y']):
        metrics['Curve_Level'] = (
            nominal['Treasury_2Y'] +
            nominal['Treasury_5Y'] +
            nominal['Treasury_10Y']
        ) / 3
        print("  Calculated curve level")

    # Print summary
    print("\n" + "=" * 60)
    print("=== Yield Curve Metrics Summary ===")
    print("=" * 60)
    print(f"Metrics calculated: {len(metrics.columns)}")
    print(f"Date range: {metrics.index.min()} to {metrics.index.max()}")

    print("\nLatest values:")
    for col in metrics.columns:
        latest = metrics[col].dropna().iloc[-1] if len(metrics[col].dropna()) > 0 else 'N/A'
        if isinstance(latest, (int, float)):
            if 'Inverted' in col:
                print(f"  {col}: {'Yes' if latest == 1 else 'No'}")
            else:
                print(f"  {col}: {latest:.2f}%")
        else:
            print(f"  {col}: {latest}")

    # Inversion statistics
    if 'Inverted_10Y_2Y' in metrics.columns:
        inversion_pct = metrics['Inverted_10Y_2Y'].mean() * 100
        print(f"\n10Y-2Y Inversion frequency: {inversion_pct:.1f}% of observations")

    if 'Inverted_10Y_3M' in metrics.columns:
        inversion_pct = metrics['Inverted_10Y_3M'].mean() * 100
        print(f"10Y-3M Inversion frequency: {inversion_pct:.1f}% of observations")

    return metrics


def load_comprehensive_rates_data(
    start_date='2000-01-01',
    end_date=None,
    cache_path='./data/fred'
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
    cache_file = os.path.join(
        cache_path,
        f'comprehensive_rates_{start_date}_{end_date}.pkl'
    )

    if os.path.exists(cache_file):
        print(f"Loading cached comprehensive rates data from {cache_file}")
        return pd.read_pickle(cache_file)

    print("=" * 60)
    print("Loading Comprehensive Rates Data (2000-2025)")
    print("=" * 60)

    # Load Treasury yields
    print("\n[1/4] Loading Treasury yields...")
    treasury_data = load_treasury_yields(start_date, end_date, cache_path)

    # Load policy rates
    print("\n[2/4] Loading policy rates...")
    policy_data = load_policy_rates(start_date, end_date, cache_path)

    # Load credit spreads
    print("\n[3/4] Loading credit spreads...")
    credit_data = load_credit_spreads(start_date, end_date, cache_path)

    # Calculate yield curve metrics
    print("\n[4/4] Calculating yield curve metrics...")
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

    pd.to_pickle(result, cache_file)
    print(f"\nCached comprehensive rates data to {cache_file}")

    # Print final summary
    print("\n" + "=" * 60)
    print("=== Comprehensive Rates Data Summary ===")
    print("=" * 60)
    print(f"Total series loaded: {len(combined_df.columns)}")
    print(f"Date range: {combined_df.index.min()} to {combined_df.index.max()}")
    print(f"Total observations: {len(combined_df)}")

    print("\n--- Series Categories ---")
    print(f"Treasury yields: {len(treasury_data['combined'].columns) if treasury_data else 0}")
    print(f"Policy rates: {len(policy_data['combined'].columns) if policy_data else 0}")
    print(f"Credit/Mortgage: {len(credit_data['combined'].columns) if credit_data else 0}")
    print(f"Curve metrics: {len(curve_metrics.columns) if curve_metrics is not None else 0}")

    print("\n" + "=" * 60)
    print("Citation:")
    print("Federal Reserve Economic Data (FRED), Federal Reserve Bank of St. Louis")
    print("https://fred.stlouisfed.org/")
    print("=" * 60)

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
    cache_file = os.path.join(cache_path, f'industrial_production_{start_date}_{end_date}.pkl')

    if os.path.exists(cache_file):
        print(f"Loading cached industrial production data from {cache_file}")
        return pd.read_pickle(cache_file)

    print("Downloading Industrial Production data from FRED...")

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
        print("\n--- Total Industrial Production ---")
        total_data = {}
        for name, code in total_series.items():
            try:
                print(f"  Downloading {name} ({code})...")
                df = pdr.DataReader(code, 'fred', start=start_date, end=end_date)
                total_data[name] = df.iloc[:, 0]
            except Exception as e:
                print(f"  Warning: Could not download {name}: {e}")

        total_df = pd.DataFrame(total_data)

        # Download sector production
        print("\n--- Sector-Level Production ---")
        sector_data = {}
        for name, code in sector_series.items():
            try:
                print(f"  Downloading {name} ({code})...")
                df = pdr.DataReader(code, 'fred', start=start_date, end=end_date)
                sector_data[name] = df.iloc[:, 0]
            except Exception as e:
                print(f"  Warning: Could not download {name}: {e}")

        sector_df = pd.DataFrame(sector_data)

        # Download capacity utilization
        print("\n--- Capacity Utilization ---")
        capacity_data = {}
        for name, code in capacity_series.items():
            try:
                print(f"  Downloading {name} ({code})...")
                df = pdr.DataReader(code, 'fred', start=start_date, end=end_date)
                capacity_data[name] = df.iloc[:, 0]
            except Exception as e:
                print(f"  Warning: Could not download {name}: {e}")

        capacity_df = pd.DataFrame(capacity_data)

        # Combine all data
        combined_df = pd.concat([total_df, sector_df, capacity_df], axis=1)

        result = {
            'total': total_df,
            'sectors': sector_df,
            'capacity': capacity_df,
            'combined': combined_df
        }

        pd.to_pickle(result, cache_file)
        print(f"\nCached industrial production data to {cache_file}")

        # Print summary
        print("\n" + "=" * 60)
        print("=== Industrial Production Summary ===")
        print("=" * 60)
        print(f"Date range: {combined_df.index.min()} to {combined_df.index.max()}")
        print(f"Observations: {len(combined_df)}")
        print(f"Total IP series: {len(total_df.columns)}")
        print(f"Sector series: {len(sector_df.columns)}")
        print(f"Capacity utilization series: {len(capacity_df.columns)}")

        print("\nLatest values:")
        for col in combined_df.columns:
            latest = combined_df[col].dropna().iloc[-1] if len(combined_df[col].dropna()) > 0 else 'N/A'
            if isinstance(latest, float):
                if 'CapUtil' in col:
                    print(f"  {col}: {latest:.1f}%")
                else:
                    print(f"  {col}: {latest:.2f}")
            else:
                print(f"  {col}: {latest}")

        return result

    except Exception as e:
        print(f"Error downloading industrial production data: {e}")
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
    cache_file = os.path.join(cache_path, f'ip_growth_{start_date}_{end_date}.pkl')

    if os.path.exists(cache_file):
        print(f"Loading cached IP growth data from {cache_file}")
        return pd.read_pickle(cache_file)

    print("Downloading IP growth rates from FRED...")

    # Pre-calculated growth rates from FRED
    growth_series = {
        'IP_YoY_Change': 'INDPRO',              # Will calculate YoY
        'IP_Manufacturing_YoY': 'IPMAN',        # Will calculate YoY
    }

    # Diffusion indices (percent of industries expanding)
    diffusion_series = {
        'IP_Diffusion_1M': 'IPDIFF1M',          # 1-Month Diffusion Index
        'IP_Diffusion_3M': 'IPDIFF3M',          # 3-Month Diffusion Index
        'IP_Diffusion_6M': 'IPDIFF6M',          # 6-Month Diffusion Index
    }

    try:
        # Download base IP series for growth calculation
        print("\n--- Industrial Production Indices ---")
        ip_data = {}
        for name, code in growth_series.items():
            try:
                base_name = name.replace('_YoY_Change', '').replace('_YoY', '')
                print(f"  Downloading {base_name} ({code})...")
                df = pdr.DataReader(code, 'fred', start=start_date, end=end_date)
                ip_data[base_name] = df.iloc[:, 0]
            except Exception as e:
                print(f"  Warning: Could not download {name}: {e}")

        ip_df = pd.DataFrame(ip_data)

        # Calculate YoY growth rates
        print("\nCalculating year-over-year growth rates...")
        yoy_df = ip_df.pct_change(periods=12) * 100
        yoy_df.columns = [f'{col}_YoY' for col in yoy_df.columns]

        # Calculate MoM growth rates
        print("Calculating month-over-month growth rates...")
        mom_df = ip_df.pct_change() * 100
        mom_df.columns = [f'{col}_MoM' for col in mom_df.columns]

        # Download diffusion indices
        print("\n--- Diffusion Indices ---")
        diffusion_data = {}
        for name, code in diffusion_series.items():
            try:
                print(f"  Downloading {name} ({code})...")
                df = pdr.DataReader(code, 'fred', start=start_date, end=end_date)
                diffusion_data[name] = df.iloc[:, 0]
            except Exception as e:
                print(f"  Warning: Could not download {name}: {e}")

        diffusion_df = pd.DataFrame(diffusion_data)

        # Combine all data
        combined_df = pd.concat([yoy_df, mom_df, diffusion_df], axis=1)

        result = {
            'raw': ip_df,
            'yoy': yoy_df,
            'mom': mom_df,
            'diffusion': diffusion_df,
            'combined': combined_df
        }

        pd.to_pickle(result, cache_file)
        print(f"\nCached IP growth data to {cache_file}")

        # Print summary
        print("\n" + "=" * 60)
        print("=== IP Growth Rates Summary ===")
        print("=" * 60)
        print(f"Date range: {combined_df.index.min()} to {combined_df.index.max()}")
        print(f"Observations: {len(combined_df)}")

        print("\nLatest growth rates:")
        for col in yoy_df.columns:
            latest = yoy_df[col].dropna().iloc[-1] if len(yoy_df[col].dropna()) > 0 else 'N/A'
            if isinstance(latest, float):
                print(f"  {col}: {latest:.2f}%")
            else:
                print(f"  {col}: {latest}")

        if len(diffusion_df) > 0:
            print("\nLatest diffusion indices:")
            for col in diffusion_df.columns:
                latest = diffusion_df[col].dropna().iloc[-1] if len(diffusion_df[col].dropna()) > 0 else 'N/A'
                if isinstance(latest, float):
                    print(f"  {col}: {latest:.1f}")
                else:
                    print(f"  {col}: {latest}")

        return result

    except Exception as e:
        print(f"Error downloading IP growth data: {e}")
        return None


def load_comprehensive_ip_data(
    start_date='2000-01-01',
    end_date=None,
    cache_path='./data/fred'
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
    cache_file = os.path.join(
        cache_path,
        f'comprehensive_ip_{start_date}_{end_date}.pkl'
    )

    if os.path.exists(cache_file):
        print(f"Loading cached comprehensive IP data from {cache_file}")
        return pd.read_pickle(cache_file)

    print("=" * 60)
    print("Loading Comprehensive Industrial Production Data (2000-2025)")
    print("=" * 60)

    # Load production data
    print("\n[1/2] Loading industrial production indices...")
    production_data = load_industrial_production_data(start_date, end_date, cache_path)

    # Load growth rates
    print("\n[2/2] Loading growth rates and diffusion indices...")
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

    pd.to_pickle(result, cache_file)
    print(f"\nCached comprehensive IP data to {cache_file}")

    # Print final summary
    print("\n" + "=" * 60)
    print("=== Comprehensive Industrial Production Summary ===")
    print("=" * 60)
    print(f"Total series loaded: {len(combined_df.columns)}")
    print(f"Date range: {combined_df.index.min()} to {combined_df.index.max()}")
    print(f"Total observations: {len(combined_df)}")

    print("\n--- Series Categories ---")
    if production_data:
        print(f"Production indices: {len(production_data['combined'].columns)}")
    if growth_data:
        print(f"Growth rates & diffusion: {len(growth_data['combined'].columns)}")

    # Key metrics
    if production_data and 'total' in production_data:
        total_df = production_data['total']
        if 'IP_Total' in total_df.columns:
            latest_ip = total_df['IP_Total'].dropna().iloc[-1]
            print(f"\nLatest Total IP Index: {latest_ip:.2f}")

    if production_data and 'capacity' in production_data:
        cap_df = production_data['capacity']
        if 'CapUtil_Total' in cap_df.columns:
            latest_cap = cap_df['CapUtil_Total'].dropna().iloc[-1]
            print(f"Latest Capacity Utilization: {latest_cap:.1f}%")

    if growth_data and 'yoy' in growth_data:
        yoy_df = growth_data['yoy']
        if 'IP_Total_YoY' in yoy_df.columns or 'IP_YoY' in yoy_df.columns:
            col = 'IP_Total_YoY' if 'IP_Total_YoY' in yoy_df.columns else 'IP_YoY'
            latest_growth = yoy_df[col].dropna().iloc[-1] if len(yoy_df[col].dropna()) > 0 else None
            if latest_growth is not None:
                print(f"Latest IP YoY Growth: {latest_growth:.2f}%")

    print("\n" + "=" * 60)
    print("Citation:")
    print("Board of Governors of the Federal Reserve System (US)")
    print("Federal Reserve Economic Data (FRED), Federal Reserve Bank of St. Louis")
    print("https://fred.stlouisfed.org/")
    print("=" * 60)

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
    cache_file = os.path.join(cache_path, f'money_supply_{start_date}_{end_date}.pkl')

    if os.path.exists(cache_file):
        print(f"Loading cached money supply data from {cache_file}")
        return pd.read_pickle(cache_file)

    print("Downloading Money Supply data from FRED...")

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
        print("\n--- Money Supply Aggregates ---")
        aggregate_data = {}
        for name, code in aggregate_series.items():
            try:
                print(f"  Downloading {name} ({code})...")
                df = pdr.DataReader(code, 'fred', start=start_date, end=end_date)
                aggregate_data[name] = df.iloc[:, 0]
            except Exception as e:
                print(f"  Warning: Could not download {name}: {e}")

        aggregate_df = pd.DataFrame(aggregate_data)

        # Download money supply components
        print("\n--- Money Supply Components ---")
        component_data = {}
        for name, code in component_series.items():
            try:
                print(f"  Downloading {name} ({code})...")
                df = pdr.DataReader(code, 'fred', start=start_date, end=end_date)
                component_data[name] = df.iloc[:, 0]
            except Exception as e:
                print(f"  Warning: Could not download {name}: {e}")

        component_df = pd.DataFrame(component_data)

        # Combine all data
        combined_df = pd.concat([aggregate_df, component_df], axis=1)

        result = {
            'aggregates': aggregate_df,
            'components': component_df,
            'combined': combined_df
        }

        pd.to_pickle(result, cache_file)
        print(f"\nCached money supply data to {cache_file}")

        # Print summary
        print("\n" + "=" * 60)
        print("=== Money Supply Summary ===")
        print("=" * 60)
        print(f"Date range: {combined_df.index.min()} to {combined_df.index.max()}")
        print(f"Observations: {len(combined_df)}")
        print(f"Aggregate series: {len(aggregate_df.columns)}")
        print(f"Component series: {len(component_df.columns)}")

        print("\nLatest values (Billions USD):")
        for col in combined_df.columns:
            latest = combined_df[col].dropna().iloc[-1] if len(combined_df[col].dropna()) > 0 else 'N/A'
            if isinstance(latest, float):
                print(f"  {col}: ${latest:,.1f}B")
            else:
                print(f"  {col}: {latest}")

        return result

    except Exception as e:
        print(f"Error downloading money supply data: {e}")
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
    cache_file = os.path.join(cache_path, f'money_velocity_{start_date}_{end_date}.pkl')

    if os.path.exists(cache_file):
        print(f"Loading cached money velocity data from {cache_file}")
        return pd.read_pickle(cache_file)

    print("Downloading Money Velocity data from FRED...")

    velocity_series = {
        'M1_Velocity': 'M1V',                   # Velocity of M1 Money Stock
        'M2_Velocity': 'M2V',                   # Velocity of M2 Money Stock
    }

    try:
        print("\n--- Money Velocity ---")
        velocity_data = {}
        for name, code in velocity_series.items():
            try:
                print(f"  Downloading {name} ({code})...")
                df = pdr.DataReader(code, 'fred', start=start_date, end=end_date)
                velocity_data[name] = df.iloc[:, 0]
            except Exception as e:
                print(f"  Warning: Could not download {name}: {e}")

        velocity_df = pd.DataFrame(velocity_data)

        result = {
            'velocity': velocity_df,
            'combined': velocity_df
        }

        pd.to_pickle(result, cache_file)
        print(f"\nCached money velocity data to {cache_file}")

        # Print summary
        print("\n" + "=" * 60)
        print("=== Money Velocity Summary ===")
        print("=" * 60)
        print(f"Date range: {velocity_df.index.min()} to {velocity_df.index.max()}")
        print(f"Observations: {len(velocity_df)}")

        print("\nLatest values:")
        for col in velocity_df.columns:
            latest = velocity_df[col].dropna().iloc[-1] if len(velocity_df[col].dropna()) > 0 else 'N/A'
            if isinstance(latest, float):
                print(f"  {col}: {latest:.2f}")
            else:
                print(f"  {col}: {latest}")

        return result

    except Exception as e:
        print(f"Error downloading money velocity data: {e}")
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
    cache_file = os.path.join(cache_path, f'fed_balance_sheet_{start_date}_{end_date}.pkl')

    if os.path.exists(cache_file):
        print(f"Loading cached Fed balance sheet data from {cache_file}")
        return pd.read_pickle(cache_file)

    print("Downloading Federal Reserve Balance Sheet data from FRED...")

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
        print("\n--- Federal Reserve Assets ---")
        asset_data = {}
        for name, code in asset_series.items():
            try:
                print(f"  Downloading {name} ({code})...")
                df = pdr.DataReader(code, 'fred', start=start_date, end=end_date)
                asset_data[name] = df.iloc[:, 0]
            except Exception as e:
                print(f"  Warning: Could not download {name}: {e}")

        asset_df = pd.DataFrame(asset_data)

        # Download reserve measures
        print("\n--- Bank Reserves ---")
        reserve_data = {}
        for name, code in reserve_series.items():
            try:
                print(f"  Downloading {name} ({code})...")
                df = pdr.DataReader(code, 'fred', start=start_date, end=end_date)
                reserve_data[name] = df.iloc[:, 0]
            except Exception as e:
                print(f"  Warning: Could not download {name}: {e}")

        reserve_df = pd.DataFrame(reserve_data)

        # Combine all data
        combined_df = pd.concat([asset_df, reserve_df], axis=1)

        result = {
            'assets': asset_df,
            'reserves': reserve_df,
            'combined': combined_df
        }

        pd.to_pickle(result, cache_file)
        print(f"\nCached Fed balance sheet data to {cache_file}")

        # Print summary
        print("\n" + "=" * 60)
        print("=== Fed Balance Sheet Summary ===")
        print("=" * 60)
        print(f"Date range: {combined_df.index.min()} to {combined_df.index.max()}")
        print(f"Observations: {len(combined_df)}")
        print(f"Asset series: {len(asset_df.columns)}")
        print(f"Reserve series: {len(reserve_df.columns)}")

        print("\nLatest values (Millions/Billions USD):")
        for col in combined_df.columns:
            latest = combined_df[col].dropna().iloc[-1] if len(combined_df[col].dropna()) > 0 else 'N/A'
            if isinstance(latest, float):
                if latest > 1000000:
                    print(f"  {col}: ${latest/1000000:,.2f}T")
                elif latest > 1000:
                    print(f"  {col}: ${latest/1000:,.1f}B")
                else:
                    print(f"  {col}: ${latest:,.1f}M")
            else:
                print(f"  {col}: {latest}")

        return result

    except Exception as e:
        print(f"Error downloading Fed balance sheet data: {e}")
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
        print("Money supply data not available")
        return None

    aggregates = money_supply_data['aggregates']
    growth = pd.DataFrame(index=aggregates.index)

    print("Calculating M2 growth rates...")

    # Year-over-Year growth
    if 'M2' in aggregates.columns:
        growth['M2_YoY'] = aggregates['M2'].pct_change(periods=12) * 100
        print("  Calculated M2 YoY growth")

    if 'M1' in aggregates.columns:
        growth['M1_YoY'] = aggregates['M1'].pct_change(periods=12) * 100
        print("  Calculated M1 YoY growth")

    # Month-over-Month growth (annualized)
    if 'M2' in aggregates.columns:
        growth['M2_MoM'] = aggregates['M2'].pct_change() * 100
        growth['M2_MoM_Annualized'] = aggregates['M2'].pct_change() * 100 * 12
        print("  Calculated M2 MoM growth")

    if 'M1' in aggregates.columns:
        growth['M1_MoM'] = aggregates['M1'].pct_change() * 100
        growth['M1_MoM_Annualized'] = aggregates['M1'].pct_change() * 100 * 12
        print("  Calculated M1 MoM growth")

    # 3-month and 6-month annualized growth
    if 'M2' in aggregates.columns:
        growth['M2_3M_Annualized'] = aggregates['M2'].pct_change(periods=3) * 100 * 4
        growth['M2_6M_Annualized'] = aggregates['M2'].pct_change(periods=6) * 100 * 2
        print("  Calculated M2 3M and 6M annualized growth")

    # Print summary
    print("\n" + "=" * 60)
    print("=== M2 Growth Rates Summary ===")
    print("=" * 60)
    print(f"Date range: {growth.index.min()} to {growth.index.max()}")

    print("\nLatest growth rates:")
    for col in growth.columns:
        latest = growth[col].dropna().iloc[-1] if len(growth[col].dropna()) > 0 else 'N/A'
        if isinstance(latest, float):
            print(f"  {col}: {latest:.2f}%")
        else:
            print(f"  {col}: {latest}")

    return growth


def load_comprehensive_m2_data(
    start_date='2000-01-01',
    end_date=None,
    cache_path='./data/fred'
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
    cache_file = os.path.join(
        cache_path,
        f'comprehensive_m2_{start_date}_{end_date}.pkl'
    )

    if os.path.exists(cache_file):
        print(f"Loading cached comprehensive M2 data from {cache_file}")
        return pd.read_pickle(cache_file)

    print("=" * 60)
    print("Loading Comprehensive M2 Money Supply Data (2000-2025)")
    print("=" * 60)

    # Load money supply data
    print("\n[1/4] Loading money supply aggregates and components...")
    money_supply_data = load_money_supply_data(start_date, end_date, cache_path)

    # Load velocity data
    print("\n[2/4] Loading money velocity...")
    velocity_data = load_money_velocity_data(start_date, end_date, cache_path)

    # Load Fed balance sheet
    print("\n[3/4] Loading Fed balance sheet...")
    fed_data = load_fed_balance_sheet_data(start_date, end_date, cache_path)

    # Calculate growth rates
    print("\n[4/4] Calculating M2 growth rates...")
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

    pd.to_pickle(result, cache_file)
    print(f"\nCached comprehensive M2 data to {cache_file}")

    # Print final summary
    print("\n" + "=" * 60)
    print("=== Comprehensive M2 Money Supply Summary ===")
    print("=" * 60)
    print(f"Total series loaded: {len(combined_df.columns)}")
    print(f"Date range: {combined_df.index.min()} to {combined_df.index.max()}")
    print(f"Total observations: {len(combined_df)}")

    print("\n--- Series Categories ---")
    if money_supply_data:
        print(f"Money supply measures: {len(money_supply_data['combined'].columns)}")
    if velocity_data:
        print(f"Velocity measures: {len(velocity_data['combined'].columns)}")
    if fed_data:
        print(f"Fed balance sheet: {len(fed_data['combined'].columns)}")
    if growth_rates is not None:
        print(f"Growth rates: {len(growth_rates.columns)}")

    # Key metrics
    if money_supply_data and 'aggregates' in money_supply_data:
        agg_df = money_supply_data['aggregates']
        if 'M2' in agg_df.columns:
            latest_m2 = agg_df['M2'].dropna().iloc[-1]
            print(f"\nLatest M2 Money Stock: ${latest_m2:,.1f}B")

    if growth_rates is not None and 'M2_YoY' in growth_rates.columns:
        latest_growth = growth_rates['M2_YoY'].dropna().iloc[-1]
        print(f"Latest M2 YoY Growth: {latest_growth:.2f}%")

    if velocity_data and 'velocity' in velocity_data:
        vel_df = velocity_data['velocity']
        if 'M2_Velocity' in vel_df.columns:
            latest_vel = vel_df['M2_Velocity'].dropna().iloc[-1]
            print(f"Latest M2 Velocity: {latest_vel:.2f}")

    print("\n" + "=" * 60)
    print("Citation:")
    print("Board of Governors of the Federal Reserve System (US)")
    print("Federal Reserve Economic Data (FRED), Federal Reserve Bank of St. Louis")
    print("https://fred.stlouisfed.org/")
    print("=" * 60)

    return result


def load_additional_macro_data(
    start_date='2000-01-01',
    end_date=None,
    cache_path='./data/fred'
):
    """
    Load additional macroeconomic indicators from FRED.

    Returns unemployment, GDP growth, interest rates, etc.
    """
    if end_date is None:
        end_date = datetime.today().strftime('%Y-%m-%d')

    os.makedirs(cache_path, exist_ok=True)
    cache_file = os.path.join(cache_path, f'macro_data_{start_date}_{end_date}.pkl')

    if os.path.exists(cache_file):
        print(f"Loading cached macro data from {cache_file}")
        return pd.read_pickle(cache_file)

    print("Downloading macro data from FRED...")

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
        macro_data = {}
        for name, code in series.items():
            try:
                print(f"  Downloading {name} ({code})...")
                df = pdr.DataReader(code, 'fred', start=start_date, end=end_date)
                macro_data[name] = df.iloc[:, 0]
            except Exception as e:
                print(f"  Warning: Could not download {name}: {e}")

        macro_df = pd.DataFrame(macro_data)

        pd.to_pickle(macro_df, cache_file)
        print(f"Cached macro data to {cache_file}")

        print("\n=== Macro Data Summary ===")
        print(f"Date range: {macro_df.index.min()} to {macro_df.index.max()}")
        print(f"Series downloaded: {len(macro_df.columns)}")
        print("\nLatest values:")
        print(macro_df.iloc[-1])

        return macro_df

    except Exception as e:
        print(f"Error downloading macro data: {e}")
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


def load_news_data(
    cache_file='./news_data/culture_war_news_2000_2025_final.csv',
    refresh=False,
    sources=None
):
    """
    Load news articles data from Guardian, NYT, and Reddit.

    Parameters:
    -----------
    cache_file : str
        Path to cached news CSV file
    refresh : bool
        If True, re-download data even if cache exists
    sources : list
        List of news sources to include ['guardian', 'nyt', 'reddit']

    Returns:
    --------
    pd.DataFrame : News articles
    """
    if sources is None:
        sources = ['guardian', 'nyt', 'reddit']

    if os.path.exists(cache_file) and not refresh:
        print(f"Loading cached news data from {cache_file}")
        news_df = pd.read_csv(cache_file)
        news_df['published_date'] = pd.to_datetime(news_df['published_date'])

        if sources:
            source_patterns = {
                'guardian': 'The Guardian',
                'nyt': 'New York Times',
                'reddit': 'Reddit r/'
            }
            keep_sources = []
            for src in sources:
                if src in source_patterns:
                    keep_sources.append(source_patterns[src])

            if keep_sources:
                news_df = news_df[
                    news_df['source'].str.contains(
                        '|'.join(keep_sources), case=False, na=False
                    )
                ]

        return news_df
    else:
        print(f"News data not found at {cache_file}")
        print("Run scrape_culture_war_news() to generate this data")
        return None


def scrape_culture_war_news(
    culture_war_csv: str = 'Culture_War_Companies_160_fullmeta.csv',
    output_file: str = './news_data/culture_war_news.csv',
    start_date: str = '2000-01-01',
    end_date: str = '2025-12-31',
    max_results_per_source: int = 200,
    sources: List[str] = None
) -> pd.DataFrame:
    """
    Scrape news about culture war companies and their events.

    This function searches for news articles about each company's culture war
    event across the full date range. It also searches for insider trading
    news related to these companies.

    Parameters:
    -----------
    culture_war_csv : str
        Path to the culture war companies CSV file
    output_file : str
        Path to save the output CSV file
    start_date : str
        Start date for search (default: '2000-01-01')
    end_date : str
        End date for search (default: '2025-12-31')
    max_results_per_source : int
        Maximum articles per source per event (default: 200)
    sources : list
        List of sources to use ['guardian', 'nyt', 'reddit']

    Returns:
    --------
    pd.DataFrame : DataFrame with all scraped news articles

    Note:
    -----
    Requires API keys to be set in environment variables or .env file:
    - GUARDIAN_API_KEY: For The Guardian API
    - NYT_API_KEY: For New York Times API
    - REDDIT_CLIENT_ID, REDDIT_CLIENT_SECRET, REDDIT_USER_AGENT: For Reddit API
    """
    if sources is None:
        sources = ['guardian', 'nyt', 'reddit']

    # Load culture war data
    print("Loading culture war companies data...")
    culture_war_df = import_culture_war_data(culture_war_csv)
    print(f"Loaded {len(culture_war_df)} culture war events")

    # Create output directory if needed
    output_dir = os.path.dirname(output_file)
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)

    # Initialize news aggregator
    print("\nInitializing news aggregator...")
    aggregator = CompanyNewsAggregator(
        guardian_api_key=os.getenv('GUARDIAN_API_KEY'),
        nyt_api_key=os.getenv('NYT_API_KEY'),
        reddit_client_id=os.getenv('REDDIT_CLIENT_ID'),
        reddit_client_secret=os.getenv('REDDIT_CLIENT_SECRET'),
        reddit_user_agent=os.getenv('REDDIT_USER_AGENT', 'CultureWarResearch/1.0')
    )

    # Check which APIs are available
    available_sources = []
    if aggregator.guardian_api_key:
        available_sources.append('guardian')
        print("  Guardian API: Available")
    else:
        print("  Guardian API: Not configured (set GUARDIAN_API_KEY)")

    if aggregator.nyt_api_key:
        available_sources.append('nyt')
        print("  NYT API: Available")
    else:
        print("  NYT API: Not configured (set NYT_API_KEY)")

    if aggregator.reddit:
        available_sources.append('reddit')
        print("  Reddit API: Available")
    else:
        print("  Reddit API: Not configured (set REDDIT_CLIENT_ID, REDDIT_CLIENT_SECRET)")

    # Filter sources to only available ones
    sources = [s for s in sources if s in available_sources]

    if not sources:
        print("\nERROR: No API keys configured. Please set at least one of:")
        print("  - GUARDIAN_API_KEY")
        print("  - NYT_API_KEY")
        print("  - REDDIT_CLIENT_ID + REDDIT_CLIENT_SECRET")
        return pd.DataFrame()

    print(f"\nUsing sources: {sources}")
    print(f"Date range: {start_date} to {end_date}")
    print(f"Max results per source: {max_results_per_source}")

    # Run the aggregation
    print("\n" + "=" * 60)
    print("Starting news scraping...")
    print("=" * 60)

    news_df = aggregator.aggregate_culture_war_news(
        culture_war_df=culture_war_df,
        start_date=start_date,
        end_date=end_date,
        max_results_per_source=max_results_per_source,
        sources=sources,
        checkpoint_file=output_file.replace('.csv', '_checkpoint.csv')
    )

    # Save final results
    if len(news_df) > 0:
        aggregator.save_news(news_df, output_file)
    else:
        print("\nNo articles found. Check API keys and try again.")

    return news_df


# =============================================================================
# DATA LOADING
# =============================================================================
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
    """
    data_dict = {}

    # Load culture war companies data
    try:
        data_dict['culturewardata'] = import_culture_war_data(
            'Culture_War_Companies_160_fullmeta.csv'
        )
        print("Loaded culture war data")
    except Exception as e:
        print(f"Error loading culture war data: {e}")
        data_dict['culturewardata'] = None

    # Load stock data
    try:
        if data_dict['culturewardata'] is not None:
            tickers = data_dict['culturewardata']['Ticker'].unique().tolist()
            data_dict['stockdata'] = get_stock_data(
                tickers, start_date='2000-01-01', end_date='2025-12-31'
            )
            print(f"Loaded stock data for {len(data_dict['stockdata'])} tickers")
        else:
            data_dict['stockdata'] = None
    except Exception as e:
        print(f"Error loading stock data: {e}")
        data_dict['stockdata'] = None

    # Load VIX data
    try:
        data_dict['vixdata'] = download_vix_data()
        print("Loaded VIX data")
    except Exception as e:
        print(f"Error loading VIX data: {e}")
        data_dict['vixdata'] = None

    # Load Fama-French factors
    try:
        data_dict['ff_factors'] = download_fama_french_factors(
            start_date='2000-01-01',
            frequency='daily',
            save_path='./fama_french_data'
        )
        print("Loaded Fama-French factors")
    except Exception as e:
        print(f"Error loading Fama-French factors: {e}")
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
            print("Loaded Form 4 data")
        else:
            data_dict['form4data'] = None
    except Exception as e:
        print(f"Error loading Form 4 data: {e}")
        data_dict['form4data'] = None

    # Load news data
    try:
        data_dict['newsdata'] = load_news_data(
            cache_file='./news_data/culture_war_news_2000_2025_final.csv',
            refresh=False
        )
        print("Loaded news data")
    except Exception as e:
        print(f"Error loading news data: {e}")
        data_dict['newsdata'] = None

    # Load inflation data (core measures)
    try:
        data_dict['inflationdata'] = load_inflation_data(
            start_date='2000-01-01',
            end_date='2025-12-31',
            cache_path='./data/fred'
        )
        print("Loaded inflation data")
    except Exception as e:
        print(f"Error loading inflation data: {e}")
        data_dict['inflationdata'] = None

    # Load inflation expectations (breakeven, surveys, Fed measures)
    try:
        data_dict['inflation_expectations'] = load_inflation_expectations_data(
            start_date='2000-01-01',
            end_date='2025-12-31',
            cache_path='./data/fred'
        )
        print("Loaded inflation expectations data")
    except Exception as e:
        print(f"Error loading inflation expectations data: {e}")
        data_dict['inflation_expectations'] = None

    # Load comprehensive inflation data (all measures combined)
    try:
        data_dict['comprehensive_inflation'] = load_comprehensive_inflation_data(
            start_date='2000-01-01',
            end_date='2025-12-31',
            cache_path='./data/fred'
        )
        print("Loaded comprehensive inflation data")
    except Exception as e:
        print(f"Error loading comprehensive inflation data: {e}")
        data_dict['comprehensive_inflation'] = None

    # Load Treasury yields
    try:
        data_dict['treasury_yields'] = load_treasury_yields(
            start_date='2000-01-01',
            end_date='2025-12-31',
            cache_path='./data/fred'
        )
        print("Loaded Treasury yields data")
    except Exception as e:
        print(f"Error loading Treasury yields data: {e}")
        data_dict['treasury_yields'] = None

    # Load policy rates (Fed Funds, SOFR, Prime)
    try:
        data_dict['policy_rates'] = load_policy_rates(
            start_date='2000-01-01',
            end_date='2025-12-31',
            cache_path='./data/fred'
        )
        print("Loaded policy rates data")
    except Exception as e:
        print(f"Error loading policy rates data: {e}")
        data_dict['policy_rates'] = None

    # Load credit spreads and mortgage rates
    try:
        data_dict['credit_spreads'] = load_credit_spreads(
            start_date='2000-01-01',
            end_date='2025-12-31',
            cache_path='./data/fred'
        )
        print("Loaded credit spreads data")
    except Exception as e:
        print(f"Error loading credit spreads data: {e}")
        data_dict['credit_spreads'] = None

    # Load comprehensive rates data (all rates combined with curve metrics)
    try:
        data_dict['comprehensive_rates'] = load_comprehensive_rates_data(
            start_date='2000-01-01',
            end_date='2025-12-31',
            cache_path='./data/fred'
        )
        print("Loaded comprehensive rates data")
    except Exception as e:
        print(f"Error loading comprehensive rates data: {e}")
        data_dict['comprehensive_rates'] = None

    # Load industrial production data
    try:
        data_dict['industrial_production'] = load_industrial_production_data(
            start_date='2000-01-01',
            end_date='2025-12-31',
            cache_path='./data/fred'
        )
        print("Loaded industrial production data")
    except Exception as e:
        print(f"Error loading industrial production data: {e}")
        data_dict['industrial_production'] = None

    # Load IP growth rates and diffusion indices
    try:
        data_dict['ip_growth'] = load_ip_growth_rates(
            start_date='2000-01-01',
            end_date='2025-12-31',
            cache_path='./data/fred'
        )
        print("Loaded IP growth rates data")
    except Exception as e:
        print(f"Error loading IP growth rates data: {e}")
        data_dict['ip_growth'] = None

    # Load comprehensive IP data (all IP measures combined)
    try:
        data_dict['comprehensive_ip'] = load_comprehensive_ip_data(
            start_date='2000-01-01',
            end_date='2025-12-31',
            cache_path='./data/fred'
        )
        print("Loaded comprehensive IP data")
    except Exception as e:
        print(f"Error loading comprehensive IP data: {e}")
        data_dict['comprehensive_ip'] = None

    # Load money supply data (M1, M2, components)
    try:
        data_dict['money_supply'] = load_money_supply_data(
            start_date='2000-01-01',
            end_date='2025-12-31',
            cache_path='./data/fred'
        )
        print("Loaded money supply data")
    except Exception as e:
        print(f"Error loading money supply data: {e}")
        data_dict['money_supply'] = None

    # Load money velocity data
    try:
        data_dict['money_velocity'] = load_money_velocity_data(
            start_date='2000-01-01',
            end_date='2025-12-31',
            cache_path='./data/fred'
        )
        print("Loaded money velocity data")
    except Exception as e:
        print(f"Error loading money velocity data: {e}")
        data_dict['money_velocity'] = None

    # Load Fed balance sheet data
    try:
        data_dict['fed_balance_sheet'] = load_fed_balance_sheet_data(
            start_date='2000-01-01',
            end_date='2025-12-31',
            cache_path='./data/fred'
        )
        print("Loaded Fed balance sheet data")
    except Exception as e:
        print(f"Error loading Fed balance sheet data: {e}")
        data_dict['fed_balance_sheet'] = None

    # Load comprehensive M2 data (all money supply measures combined)
    try:
        data_dict['comprehensive_m2'] = load_comprehensive_m2_data(
            start_date='2000-01-01',
            end_date='2025-12-31',
            cache_path='./data/fred'
        )
        print("Loaded comprehensive M2 data")
    except Exception as e:
        print(f"Error loading comprehensive M2 data: {e}")
        data_dict['comprehensive_m2'] = None

    return data_dict


# =============================================================================
# ANALYSIS FUNCTIONS
# =============================================================================
def analyze_news_sentiment_around_events(data_dict):
    """Analyze news volume and sentiment around culture war events."""
    news = data_dict['newsdata']
    culture_wars = data_dict['culturewardata']

    if news is None or len(news) == 0:
        print("No news data available")
        return None

    print("Available columns in culture_wars data:")
    print(culture_wars.columns.tolist())

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

    print(f"\nUsing columns:")
    print(f"  Date: {date_col}")
    print(f"  Description: {desc_col}")
    print(f"  Category: {cat_col}")

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
            print("\n=== News Coverage by Event Category ===")
            category_coverage = event_window.groupby(cat_col).agg({
                'title': 'count',
                'ticker': 'nunique'
            }).rename(columns={'title': 'article_count', 'ticker': 'company_count'})

            print(category_coverage)

        return event_window
    else:
        print("No date column found - cannot calculate event windows")
        return analysis_df


def get_news_for_ticker(data_dict, ticker, days_window=30):
    """Get all news for a specific ticker around its culture war event(s)."""
    news = data_dict['newsdata']
    culture_wars = data_dict['culturewardata']

    if news is None or len(news) == 0:
        print("No news data available")
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

            print(f"\n=== {ticker}: {event_desc} ===")
            print(f"Event Date: {event_date.date()}")
            print(f"Articles in +/-{days_window} day window: {len(window_news)}")

            if len(window_news) > 0:
                print("\nTop 5 articles:")
                for _, row in window_news.head().iterrows():
                    print(
                        f"  [{row['published_date'].date()}] "
                        f"{row['source']}: {row['title']}"
                    )
    else:
        print(f"No date column found. Showing all {len(ticker_news)} articles for {ticker}")

    return ticker_news


# =============================================================================
# MAIN ENTRY POINT
# =============================================================================
if __name__ == "__main__":
    print("=" * 60)
    print("Loading all datasets...")
    print("=" * 60)
    data_dict = load_data()

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
