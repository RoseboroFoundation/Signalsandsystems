"""
Data cleaning and aggregation module for Culture War Companies research.

This module provides functions to:
- Import and clean culture war companies data
- Download stock data from Yahoo Finance
- Download VIX data from FRED
- Download Fama-French factor data
- Download SEC Form 4 insider trading data
- Aggregate news from Guardian, NYT, and Reddit
- Load inflation and macroeconomic data from FRED
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

    def search_guardian(
        self,
        ticker: str,
        company_name: str,
        start_date: datetime = None,
        end_date: datetime = None,
        max_results_per_year: int = 200
    ) -> List[NewsArticle]:
        """
        Search The Guardian API for historical articles (1999-present).

        Args:
            ticker: Company ticker symbol
            company_name: Full company name
            start_date: Start date for search
            end_date: End date for search
            max_results_per_year: Maximum number of results per year

        Returns:
            List of NewsArticle objects
        """
        articles = []

        if not self.guardian_api_key:
            logger.warning("Guardian API key not provided. Skipping Guardian search.")
            return articles

        if start_date is None:
            start_date = datetime(2000, 1, 1)
        if end_date is None:
            end_date = datetime.now()

        try:
            current_start = start_date

            while current_start < end_date:
                current_end = min(
                    datetime(current_start.year + 1, 1, 1) - timedelta(days=1),
                    end_date
                )

                logger.info(f"  Guardian: Searching {current_start.year} for {ticker}")

                page = 1
                total_pages = 1
                articles_this_year = 0

                while page <= total_pages and articles_this_year < max_results_per_year:
                    url = "https://content.guardianapis.com/search"
                    params = {
                        'q': company_name,
                        'from-date': current_start.strftime('%Y-%m-%d'),
                        'to-date': current_end.strftime('%Y-%m-%d'),
                        'page': page,
                        'page-size': 50,
                        'show-fields': 'headline,trailText,webPublicationDate,bodyText,wordcount,byline',
                        'show-tags': 'all',
                        'api-key': self.guardian_api_key
                    }

                    try:
                        response = requests.get(url, params=params, timeout=15)
                        response.raise_for_status()
                        data = response.json()

                        if data['response']['status'] == 'ok':
                            total_pages = data['response']['pages']

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
                                        author=fields.get('byline'),
                                        section=item.get('sectionName'),
                                        word_count=fields.get('wordcount')
                                    )
                                    articles.append(article)
                                    articles_this_year += 1

                                except Exception as e:
                                    logger.debug(f"Error parsing Guardian article: {e}")
                                    continue

                        page += 1
                        time.sleep(0.2)

                    except requests.exceptions.RequestException as e:
                        logger.warning(f"Error fetching Guardian page {page}: {e}")
                        time.sleep(2)
                        break

                logger.info(f"    Found {articles_this_year} Guardian articles in {current_start.year}")
                current_start = datetime(current_start.year + 1, 1, 1)
                time.sleep(0.5)

        except Exception as e:
            logger.error(f"Error in Guardian search for {ticker}: {e}")

        return articles

    def search_nyt(
        self,
        ticker: str,
        company_name: str,
        start_date: datetime = None,
        end_date: datetime = None,
        max_results_per_year: int = 200
    ) -> List[NewsArticle]:
        """
        Search New York Times Article Search API (1851-present).

        NYT API rate limit: 500 requests per day, 5 requests per minute

        Args:
            ticker: Company ticker symbol
            company_name: Full company name
            start_date: Start date for search
            end_date: End date for search
            max_results_per_year: Maximum number of results per year

        Returns:
            List of NewsArticle objects
        """
        articles = []

        if not self.nyt_api_key:
            logger.warning("NYT API key not provided. Skipping NYT search.")
            return articles

        if start_date is None:
            start_date = datetime(2000, 1, 1)
        if end_date is None:
            end_date = datetime.now()

        try:
            current_start = start_date

            while current_start < end_date:
                current_end = min(
                    datetime(current_start.year + 1, 1, 1) - timedelta(days=1),
                    end_date
                )

                logger.info(f"  NYT: Searching {current_start.year} for {ticker}")

                page = 0
                articles_this_year = 0

                while page < 100 and articles_this_year < max_results_per_year:
                    self._nyt_rate_limit()

                    url = "https://api.nytimes.com/svc/search/v2/articlesearch.json"
                    params = {
                        'q': company_name,
                        'begin_date': current_start.strftime('%Y%m%d'),
                        'end_date': current_end.strftime('%Y%m%d'),
                        'page': page,
                        'api-key': self.nyt_api_key,
                        'sort': 'newest'
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
                                        author=author,
                                        section=doc.get('section_name'),
                                        word_count=doc.get('word_count')
                                    )
                                    articles.append(article)
                                    articles_this_year += 1

                                except Exception as e:
                                    logger.debug(f"Error parsing NYT article: {e}")
                                    continue

                        page += 1

                    except requests.exceptions.RequestException as e:
                        logger.warning(f"Error fetching NYT page {page}: {e}")
                        time.sleep(5)
                        break

                logger.info(f"    Found {articles_this_year} NYT articles in {current_start.year}")
                current_start = datetime(current_start.year + 1, 1, 1)
                time.sleep(1)

        except Exception as e:
            logger.error(f"Error in NYT search for {ticker}: {e}")

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
        start_date: datetime = None,
        end_date: datetime = None,
        max_results: int = 100,
        subreddits: List[str] = None
    ) -> List[NewsArticle]:
        """
        Search Reddit for company mentions.

        Args:
            ticker: Company ticker symbol
            company_name: Full company name
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

        if start_date is None:
            start_date = datetime(2000, 1, 1)
        if end_date is None:
            end_date = datetime.now()

        if subreddits is None:
            subreddits = [
                'news', 'business', 'investing', 'stocks', 'wallstreetbets',
                'finance', 'economy', 'worldnews', 'politics', 'technology',
                'entertainment', 'Conservative', 'progressive', 'capitalism',
                'Socialism_101'
            ]

        try:
            search_queries = [
                company_name,
                ticker,
                f"{company_name} boycott",
                f"{company_name} controversy",
                f"{ticker} stock"
            ]

            logger.info(f"  Reddit: Searching for {ticker}")

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

            logger.info(f"    Found {len(articles)} Reddit posts")

        except Exception as e:
            logger.error(f"Error in Reddit search for {ticker}: {e}")

        return articles

    def aggregate_news(
        self,
        companies_df: pd.DataFrame,
        start_date: datetime = None,
        end_date: datetime = None,
        max_results_per_source: int = 200,
        sources: List[str] = None,
        checkpoint_file: str = 'news_checkpoint.csv'
    ) -> pd.DataFrame:
        """
        Aggregate news from all sources for all companies (2000-2025).

        Args:
            companies_df: DataFrame with 'ticker' and 'company_name' columns
            start_date: Start date (default: 2000-01-01)
            end_date: End date (default: today)
            max_results_per_source: Max results per source per company per year
            sources: List of sources to use ['guardian', 'nyt', 'reddit']
            checkpoint_file: File to save progress

        Returns:
            DataFrame with all aggregated news articles
        """
        if start_date is None:
            start_date = datetime(2000, 1, 1)
        if end_date is None:
            end_date = datetime.now()

        if sources is None:
            sources = ['guardian', 'nyt', 'reddit']

        all_articles = []

        checkpoint_path = Path(checkpoint_file)
        processed_companies = {}

        if checkpoint_path.exists():
            try:
                checkpoint_df = pd.read_csv(checkpoint_path)
                checkpoint_df['published_date'] = pd.to_datetime(
                    checkpoint_df['published_date']
                )
                all_articles.extend(checkpoint_df.to_dict('records'))

                for _, row in checkpoint_df.iterrows():
                    ticker = row['ticker']
                    source = row['source']
                    if ticker not in processed_companies:
                        processed_companies[ticker] = set()
                    processed_companies[ticker].add(source)

                logger.info(f"Loaded checkpoint with {len(checkpoint_df)} articles")
                logger.info(f"Processed companies: {list(processed_companies.keys())}")
            except Exception as e:
                logger.warning(f"Error loading checkpoint: {e}")

        total_companies = len(companies_df)

        for idx, row in companies_df.iterrows():
            ticker = row['ticker']
            company_name = row['company_name']

            logger.info(f"\n{'=' * 60}")
            logger.info(f"[{idx+1}/{total_companies}] Processing {ticker} ({company_name})")
            logger.info(f"{'=' * 60}")

            company_articles = []
            processed_sources = processed_companies.get(ticker, set())

            if 'guardian' in sources:
                guardian_source = 'The Guardian'
                if guardian_source not in processed_sources:
                    logger.info("Searching The Guardian (2000-2025)...")
                    articles = self.search_guardian(
                        ticker, company_name, start_date, end_date,
                        max_results_per_source
                    )
                    company_articles.extend(articles)
                    logger.info(f"  Total Guardian articles: {len(articles)}")
                else:
                    logger.info("Skipping Guardian (already processed)")

            if 'nyt' in sources:
                nyt_source = 'New York Times'
                if nyt_source not in processed_sources:
                    logger.info("Searching New York Times (2000-2025)...")
                    articles = self.search_nyt(
                        ticker, company_name, start_date, end_date,
                        max_results_per_source
                    )
                    company_articles.extend(articles)
                    logger.info(f"  Total NYT articles: {len(articles)}")
                else:
                    logger.info("Skipping NYT (already processed)")

            if 'reddit' in sources:
                reddit_source_pattern = 'Reddit r/'
                reddit_processed = any(
                    reddit_source_pattern in s for s in processed_sources
                )
                if not reddit_processed:
                    logger.info("Searching Reddit (2000-2025)...")
                    articles = self.search_reddit(
                        ticker, company_name, start_date, end_date,
                        max_results_per_source
                    )
                    company_articles.extend(articles)
                    logger.info(f"  Total Reddit posts: {len(articles)}")
                else:
                    logger.info("Skipping Reddit (already processed)")

            if company_articles:
                all_articles.extend([vars(a) for a in company_articles])

                checkpoint_df = pd.DataFrame(all_articles)
                checkpoint_df.to_csv(checkpoint_path, index=False)
                logger.info(f"\nCheckpoint saved: {len(all_articles)} total articles")

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
        print("Run news aggregator separately to generate this data")
        return None


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

    # Load inflation data
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
