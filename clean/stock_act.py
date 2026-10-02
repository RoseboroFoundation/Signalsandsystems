"""STOCK Act congressional trading data loader.

Downloads and normalizes congressional financial disclosures (Periodic
Transaction Reports) from multiple free sources:

  1. Senate eFD system (efdsearch.senate.gov) — direct scraping
  2. House Clerk bulk XML + PDF parsing (disclosures-clerk.house.gov)
  3. Kadoa congress-trading-monitor (GitHub, pre-scraped fallback)

Returns a unified DataFrame of legislator trades with columns:
    MEMBER, CHAMBER, STATE, PARTY, TRANSACTION_DATE, DISCLOSURE_DATE,
    TICKER, ASSET_NAME, ASSET_TYPE, TRADE_TYPE (Purchase/Sale),
    AMOUNT_LOW, AMOUNT_HIGH, OWNER (Self/Spouse/Child/Joint),
    IS_LATE_FILING, FILING_URL

Data coverage: 2012-present (STOCK Act effective April 2012).

References
----------
STOCK Act of 2012, Pub. L. 112-105.
Ziobrowski, A.J. et al. (2004). Abnormal returns from the common stock
    investments of the U.S. Senate. JFQA, 39(4).
Eggers, A.C. & Hainmueller, J. (2013). Capitol losses: The mediocre
    performance of Congressional stock portfolios. JoP, 75(2).
"""

import io
import json
import os
import re
import time
import zipfile
from datetime import datetime
from typing import Optional

import pandas as pd
import requests
import yaml
from bs4 import BeautifulSoup

from .config import logger

# ═══════════════════════════════════════════════════════════════════════
# CONSTANTS
# ═══════════════════════════════════════════════════════════════════════

OUTPUT_DIR = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
    'data', 'stock_act'
)

# Amount range text → (low, high) in dollars
AMOUNT_RANGES = {
    '$1,001 - $15,000':         (1_001, 15_000),
    '$15,001 - $50,000':        (15_001, 50_000),
    '$50,001 - $100,000':       (50_001, 100_000),
    '$100,001 - $250,000':      (100_001, 250_000),
    '$250,001 - $500,000':      (250_001, 500_000),
    '$500,001 - $1,000,000':    (500_001, 1_000_000),
    '$1,000,001 - $5,000,000':  (1_000_001, 5_000_000),
    '$5,000,001 - $25,000,000': (5_000_001, 25_000_000),
    '$25,000,001 - $50,000,000': (25_000_001, 50_000_000),
    'Over $50,000,000':         (50_000_001, 100_000_000),
}

# Senate eFD session management
_SENATE_BASE = 'https://efdsearch.senate.gov'
_SENATE_SEARCH = f'{_SENATE_BASE}/search/'
_SENATE_AGREE = f'{_SENATE_BASE}/search/home/'
_SENATE_DATA = f'{_SENATE_BASE}/search/report/data/'

# House Clerk bulk XML
_HOUSE_ZIP_URL = 'https://disclosures-clerk.house.gov/public_disc/financial-pdfs/{}FD.zip'

# Kadoa GitHub (pre-scraped fallback)
_KADOA_RAW = 'https://raw.githubusercontent.com/kadoa-org/congress-trading-monitor/main/public/data'

# Rate limiting
_REQUEST_DELAY = 1.5  # seconds between requests


# ═══════════════════════════════════════════════════════════════════════
# SENATE eFD SCRAPER
# ═══════════════════════════════════════════════════════════════════════

class SenateDisclosureScraper:
    """Scrape Periodic Transaction Reports from the Senate eFD system."""

    def __init__(self):
        self.session = requests.Session()
        self.session.headers.update({
            'User-Agent': (
                'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) '
                'AppleWebKit/537.36 (KHTML, like Gecko) '
                'Chrome/120.0.0.0 Safari/537.36'
            ),
        })
        self._authenticated = False

    def _authenticate(self):
        """Accept the Senate eFD terms of use and get CSRF token."""
        # Step 1: GET the search page to get CSRF token
        resp = self.session.get(_SENATE_SEARCH)
        resp.raise_for_status()

        soup = BeautifulSoup(resp.text, 'html.parser')
        csrf_input = soup.find('input', {'name': 'csrfmiddlewaretoken'})
        if not csrf_input:
            raise RuntimeError("Could not find CSRF token on Senate eFD page")

        csrf_token = csrf_input['value']

        # Step 2: POST to accept the prohibition agreement
        resp = self.session.post(
            _SENATE_AGREE,
            data={
                'csrfmiddlewaretoken': csrf_token,
                'prohibition_agreement': '1',
            },
            headers={
                'Referer': _SENATE_SEARCH,
                'Origin': _SENATE_BASE,
            },
        )
        resp.raise_for_status()

        self._authenticated = True
        logger.info("Senate eFD: authenticated and accepted terms")

    def _get_csrf_token(self):
        """Get CSRF token from session cookies."""
        return self.session.cookies.get('csrftoken', '')

    def search_ptrs(self, start_date='01/01/2012', end_date=None,
                    max_results=50_000):
        """Search for all Periodic Transaction Reports.

        Parameters
        ----------
        start_date : str
            MM/DD/YYYY format.
        end_date : str, optional
            MM/DD/YYYY format. Defaults to today.
        max_results : int
            Maximum number of PTR filings to retrieve.

        Returns
        -------
        list of dict
            Each dict has: first_name, last_name, date_received, report_url
        """
        if not self._authenticated:
            self._authenticate()

        if end_date is None:
            end_date = datetime.now().strftime('%m/%d/%Y')

        results = []
        offset = 0
        page_size = 100

        while offset < max_results:
            csrf = self._get_csrf_token()
            resp = self.session.post(
                _SENATE_DATA,
                data={
                    'start': str(offset),
                    'length': str(page_size),
                    'report_types': '[11]',  # PTR
                    'filer_types': '[1]',    # Senators only
                    'submitted_start_date': f'{start_date} 00:00:00',
                    'submitted_end_date': f'{end_date} 23:59:59',
                    'candidate_state': '',
                    'senator_state': '',
                    'office_id': '',
                    'first_name': '',
                    'last_name': '',
                    'csrfmiddlewaretoken': csrf,
                },
                headers={
                    'X-CSRFToken': csrf,
                    'Referer': _SENATE_SEARCH,
                    'Origin': _SENATE_BASE,
                    'X-Requested-With': 'XMLHttpRequest',
                },
            )
            resp.raise_for_status()
            data = resp.json()

            total = data.get('recordsFiltered', 0)
            rows = data.get('data', [])
            if not rows:
                break

            for row in rows:
                # row = [first, last, ?, link_html, date_received]
                first_name = row[0].strip()
                last_name = row[1].strip()
                date_received = row[4].strip() if len(row) > 4 else ''

                # Parse the link HTML to get the report URL
                link_html = row[3] if len(row) > 3 else ''
                url_match = re.search(r'href="([^"]+)"', link_html)
                report_url = url_match.group(1) if url_match else ''
                if report_url and not report_url.startswith('http'):
                    report_url = _SENATE_BASE + report_url

                results.append({
                    'first_name': first_name,
                    'last_name': last_name,
                    'date_received': date_received,
                    'report_url': report_url,
                })

            offset += page_size
            logger.info("  Senate PTRs: fetched %d / %d", len(results), total)

            if offset >= total:
                break

            time.sleep(_REQUEST_DELAY)

        logger.info("Senate PTR search complete: %d filings found", len(results))
        return results

    def parse_ptr(self, report_url):
        """Parse a single PTR report page into transaction rows.

        Parameters
        ----------
        report_url : str
            Full URL to the PTR report page.

        Returns
        -------
        list of dict
            Each dict has: transaction_date, ticker, asset_name,
            asset_type, trade_type, amount, owner, comment
        """
        if '/paper/' in report_url:
            # Paper filings can't be parsed
            return []

        resp = self.session.get(report_url, headers={
            'Referer': _SENATE_SEARCH,
        })
        resp.raise_for_status()

        soup = BeautifulSoup(resp.text, 'html.parser')
        tbody = soup.find('tbody')
        if not tbody:
            return []

        transactions = []
        for tr in tbody.find_all('tr'):
            tds = tr.find_all('td')
            if len(tds) < 8:
                continue

            # Extract text from each column
            tx_date = tds[1].get_text(strip=True)
            owner = tds[2].get_text(strip=True) if len(tds) > 2 else 'Self'
            ticker = tds[3].get_text(strip=True)
            asset_name = tds[4].get_text(strip=True)
            asset_type = tds[5].get_text(strip=True)
            trade_type = tds[6].get_text(strip=True)
            amount = tds[7].get_text(strip=True)
            comment = tds[8].get_text(strip=True) if len(tds) > 8 else ''

            # Clean ticker — sometimes has '--' or extra whitespace
            ticker = ticker.replace('--', '').strip()
            if not ticker or ticker.lower() in ('n/a', 'na', ''):
                ticker = None

            transactions.append({
                'transaction_date': tx_date,
                'ticker': ticker,
                'asset_name': asset_name,
                'asset_type': asset_type,
                'trade_type': trade_type,
                'amount': amount,
                'owner': owner,
                'comment': comment,
            })

        return transactions

    def download_all(self, start_date='01/01/2012', end_date=None,
                     max_ptrs=None):
        """Download all Senate PTR transactions.

        Parameters
        ----------
        start_date : str
            MM/DD/YYYY format.
        end_date : str, optional
        max_ptrs : int, optional
            Max PTR filings to parse (for testing). None = all.

        Returns
        -------
        pd.DataFrame
        """
        ptrs = self.search_ptrs(start_date, end_date)
        if max_ptrs:
            ptrs = ptrs[:max_ptrs]

        all_txns = []
        for i, ptr in enumerate(ptrs):
            if not ptr['report_url']:
                continue

            try:
                txns = self.parse_ptr(ptr['report_url'])
                for txn in txns:
                    txn['member'] = f"{ptr['first_name']} {ptr['last_name']}"
                    txn['disclosure_date'] = ptr['date_received']
                    txn['filing_url'] = ptr['report_url']
                    txn['chamber'] = 'Senate'
                    all_txns.append(txn)
            except Exception as e:
                logger.warning("  Failed to parse PTR %s: %s",
                               ptr['report_url'], e)

            if (i + 1) % 50 == 0:
                logger.info("  Parsed %d / %d Senate PTRs (%d transactions)",
                            i + 1, len(ptrs), len(all_txns))

            time.sleep(_REQUEST_DELAY)

        logger.info("Senate download complete: %d transactions from %d PTRs",
                     len(all_txns), len(ptrs))
        return _normalize_transactions(pd.DataFrame(all_txns))


# ═══════════════════════════════════════════════════════════════════════
# HOUSE CLERK BULK XML
# ═══════════════════════════════════════════════════════════════════════

def download_house_index(years=None):
    """Download House financial disclosure XML indexes.

    Parameters
    ----------
    years : list of int, optional
        Years to download. Defaults to 2012-current year.

    Returns
    -------
    pd.DataFrame
        PTR filing metadata (member, state, filing date, doc ID).
        Does NOT contain transaction details (those are in PDFs).
    """
    if years is None:
        years = list(range(2012, datetime.now().year + 1))

    all_records = []
    for year in years:
        url = _HOUSE_ZIP_URL.format(year)
        try:
            resp = requests.get(url, timeout=30)
            resp.raise_for_status()
        except Exception as e:
            logger.warning("  House XML %d: %s", year, e)
            continue

        with zipfile.ZipFile(io.BytesIO(resp.content)) as zf:
            xml_files = [f for f in zf.namelist() if f.endswith('.xml')]
            for xml_file in xml_files:
                with zf.open(xml_file) as f:
                    soup = BeautifulSoup(f.read(), 'lxml-xml')
                    for member in soup.find_all('Member'):
                        filing_type = member.find('FilingType')
                        if filing_type and filing_type.text == 'P':  # PTR only
                            record = {
                                'first_name': _tag_text(member, 'First'),
                                'last_name': _tag_text(member, 'Last'),
                                'state_district': _tag_text(member, 'StateDst'),
                                'filing_date': _tag_text(member, 'FilingDate'),
                                'doc_id': _tag_text(member, 'DocID'),
                                'year': year,
                            }
                            all_records.append(record)

        logger.info("  House %d: %d PTR filings",
                     year, sum(1 for r in all_records if r['year'] == year))
        time.sleep(_REQUEST_DELAY)

    df = pd.DataFrame(all_records)
    if not df.empty:
        df['chamber'] = 'House'
        df['member'] = (df['first_name'].str.strip() + ' '
                        + df['last_name'].str.strip())
        df['state'] = df['state_district'].str[:2]
        df['filing_date'] = pd.to_datetime(df['filing_date'], errors='coerce')
        df['pdf_url'] = df.apply(
            lambda r: (f'https://disclosures-clerk.house.gov/public_disc/'
                       f'ptr-pdfs/{r["year"]}/{r["doc_id"]}.pdf'),
            axis=1
        )

    logger.info("House index complete: %d PTR filings across %d years",
                len(df), len(years))
    return df


def _tag_text(parent, tag_name, default=''):
    """Safely extract text from a BeautifulSoup tag."""
    tag = parent.find(tag_name)
    return tag.text.strip() if tag and tag.text else default


# ═══════════════════════════════════════════════════════════════════════
# KADOA PRE-SCRAPED DATA (FALLBACK / SUPPLEMENT)
# ═══════════════════════════════════════════════════════════════════════

def download_kadoa_data():
    """Download pre-scraped congressional trading data from Kadoa GitHub.

    This is the easiest and most comprehensive source — covers Senate,
    House, and Executive Branch from 2012-present with normalized fields.

    Returns
    -------
    pd.DataFrame
    """
    # Try the aggregated trades file
    urls_to_try = [
        f'{_KADOA_RAW}/trades.json',
        f'{_KADOA_RAW}/all_trades.json',
        f'{_KADOA_RAW}/congress_trades.json',
    ]

    for url in urls_to_try:
        try:
            resp = requests.get(url, timeout=30)
            if resp.status_code == 200:
                data = resp.json()
                if isinstance(data, list) and len(data) > 0:
                    df = pd.DataFrame(data)
                    logger.info("Kadoa: loaded %d trades from %s",
                                len(df), url)
                    return _normalize_kadoa(df)
        except Exception as e:
            logger.debug("Kadoa URL %s failed: %s", url, e)
            continue

    # Try senate-stock-watcher-data as alternative
    alt_url = ('https://raw.githubusercontent.com/timothycarambat/'
               'senate-stock-watcher-data/main/all_transactions.json')
    try:
        resp = requests.get(alt_url, timeout=30)
        if resp.status_code == 200:
            data = resp.json()
            if isinstance(data, list):
                df = pd.DataFrame(data)
                logger.info("Senate Stock Watcher: loaded %d trades", len(df))
                return _normalize_senate_watcher(df)
    except Exception as e:
        logger.warning("Senate Stock Watcher fallback failed: %s", e)

    logger.warning("No Kadoa/pre-scraped data available")
    return pd.DataFrame()


def _normalize_kadoa(df):
    """Normalize Kadoa GitHub data to standard schema."""
    col_map = {}
    # Map whatever columns exist to our schema (order matters — first match wins)
    for src, dst in [
        ('filer_name', 'MEMBER'), ('name', 'MEMBER'), ('member', 'MEMBER'),
        ('chamber', 'CHAMBER'),
        ('state', 'STATE'), ('filer_state', 'STATE'),
        ('party', 'PARTY'), ('filer_party', 'PARTY'),
        ('transaction_date', 'TRANSACTION_DATE'), ('trade_date', 'TRANSACTION_DATE'),
        ('filing_date', 'DISCLOSURE_DATE'), ('disclosure_date', 'DISCLOSURE_DATE'),
        ('ticker', 'TICKER'), ('symbol', 'TICKER'),
        ('asset_description', 'ASSET_NAME'), ('asset_name', 'ASSET_NAME'),
        ('asset_type', 'ASSET_TYPE'),
        ('transaction_type', 'TRADE_TYPE'), ('type', 'TRADE_TYPE'),
        ('amount_range_label', 'AMOUNT_TEXT'),
        ('owner', 'OWNER'),
        ('doc_url', 'FILING_URL'),
    ]:
        if src in df.columns and dst not in col_map.values():
            col_map[src] = dst

    df = df.rename(columns=col_map)

    # Kadoa provides numeric amount_range_low/high directly
    if 'amount_range_low' in df.columns:
        df['AMOUNT_LOW'] = pd.to_numeric(df['amount_range_low'], errors='coerce')
    if 'amount_range_high' in df.columns:
        df['AMOUNT_HIGH'] = pd.to_numeric(df['amount_range_high'], errors='coerce')

    # Filter to congressional only (exclude executive branch)
    if 'branch' in df.columns:
        df = df[df['branch'] == 'congress'].copy()
        logger.info("Filtered to congress only: %d trades", len(df))

    return _finalize_schema(df)


def _normalize_senate_watcher(df):
    """Normalize Senate Stock Watcher JSON data."""
    # This data nests transactions inside senator records
    if 'transactions' in df.columns:
        # Explode nested transactions
        rows = []
        for _, senator in df.iterrows():
            member = f"{senator.get('first_name', '')} {senator.get('last_name', '')}"
            txns = senator.get('transactions', [])
            if isinstance(txns, list):
                for txn in txns:
                    txn['MEMBER'] = member.strip()
                    txn['CHAMBER'] = 'Senate'
                    rows.append(txn)
        df = pd.DataFrame(rows)

    col_map = {}
    for src, dst in [
        ('transaction_date', 'TRANSACTION_DATE'),
        ('ticker', 'TICKER'),
        ('asset_description', 'ASSET_NAME'),
        ('asset_type', 'ASSET_TYPE'),
        ('type', 'TRADE_TYPE'),
        ('amount', 'AMOUNT_TEXT'),
        ('owner', 'OWNER'),
        ('disclosure_date', 'DISCLOSURE_DATE'),
        ('date_recieved', 'DISCLOSURE_DATE'),  # their typo
    ]:
        if src in df.columns and dst not in col_map.values():
            col_map[src] = dst

    df = df.rename(columns=col_map)
    return _finalize_schema(df)


# ═══════════════════════════════════════════════════════════════════════
# NORMALIZATION
# ═══════════════════════════════════════════════════════════════════════

def _normalize_transactions(df):
    """Normalize Senate eFD scraped transactions to standard schema."""
    if df.empty:
        return df

    col_map = {
        'member': 'MEMBER',
        'chamber': 'CHAMBER',
        'transaction_date': 'TRANSACTION_DATE',
        'disclosure_date': 'DISCLOSURE_DATE',
        'ticker': 'TICKER',
        'asset_name': 'ASSET_NAME',
        'asset_type': 'ASSET_TYPE',
        'trade_type': 'TRADE_TYPE',
        'amount': 'AMOUNT_TEXT',
        'owner': 'OWNER',
        'filing_url': 'FILING_URL',
        'comment': 'COMMENT',
    }
    df = df.rename(columns=col_map)
    return _finalize_schema(df)


def _finalize_schema(df):
    """Apply final schema normalization to any source."""
    if df.empty:
        return df

    # Ensure all required columns exist
    for col in ['MEMBER', 'CHAMBER', 'STATE', 'PARTY',
                'TRANSACTION_DATE', 'DISCLOSURE_DATE',
                'TICKER', 'ASSET_NAME', 'ASSET_TYPE',
                'TRADE_TYPE', 'AMOUNT_TEXT', 'OWNER',
                'FILING_URL']:
        if col not in df.columns:
            df[col] = None

    # Parse dates
    for col in ['TRANSACTION_DATE', 'DISCLOSURE_DATE']:
        df[col] = pd.to_datetime(df[col], errors='coerce')

    # Normalize trade type
    df['TRADE_TYPE'] = df['TRADE_TYPE'].str.strip().str.lower()
    trade_type_map = {
        'purchase': 'buy', 'buy': 'buy', 'buy (partial)': 'buy',
        'sale': 'sell', 'sale (full)': 'sell', 'sale (partial)': 'sell',
        'sell': 'sell',
        'exchange': 'exchange',
    }
    df['TRADE_TYPE_CLEAN'] = df['TRADE_TYPE'].map(trade_type_map).fillna('other')

    # Parse amount ranges (only if not already populated)
    if 'AMOUNT_LOW' not in df.columns:
        df['AMOUNT_LOW'] = None
    if 'AMOUNT_HIGH' not in df.columns:
        df['AMOUNT_HIGH'] = None

    # Fill missing from AMOUNT_TEXT
    missing_amt = df['AMOUNT_LOW'].isna() & df['AMOUNT_TEXT'].notna()
    if missing_amt.any():
        df.loc[missing_amt, 'AMOUNT_LOW'] = df.loc[missing_amt, 'AMOUNT_TEXT'].map(
            lambda x: AMOUNT_RANGES.get(str(x).strip(), (None, None))[0]
        )
        df.loc[missing_amt, 'AMOUNT_HIGH'] = df.loc[missing_amt, 'AMOUNT_TEXT'].map(
            lambda x: AMOUNT_RANGES.get(str(x).strip(), (None, None))[1]
        )

    df['AMOUNT_MIDPOINT'] = df.apply(
        lambda r: (r['AMOUNT_LOW'] + r['AMOUNT_HIGH']) / 2
        if pd.notna(r['AMOUNT_LOW']) and pd.notna(r['AMOUNT_HIGH'])
        else None,
        axis=1
    )

    # Normalize owner
    df['OWNER'] = df['OWNER'].fillna('Self').str.strip()

    # Normalize chamber
    df['CHAMBER'] = df['CHAMBER'].str.strip().str.title()

    # Filing latency (days between transaction and disclosure)
    mask = df['TRANSACTION_DATE'].notna() & df['DISCLOSURE_DATE'].notna()
    df.loc[mask, 'FILING_LATENCY_DAYS'] = (
        (df.loc[mask, 'DISCLOSURE_DATE'] - df.loc[mask, 'TRANSACTION_DATE'])
        .dt.days
    )
    # STOCK Act requires filing within 45 days
    df['IS_LATE_FILING'] = df['FILING_LATENCY_DAYS'] > 45

    # Clean ticker
    df['TICKER'] = (df['TICKER']
                    .str.strip()
                    .str.upper()
                    .replace({'--': None, 'N/A': None, 'NA': None, '': None}))

    # Filter to stock/option trades only (exclude bonds, mutual funds, etc.)
    stock_types = {'stock', 'stock option', 'common stock', 'equity',
                   'options', 'call option', 'put option',
                   'exchange-traded fund', 'etf',
                   'st',  # Senate eFD code for Stock Transaction
                   'other securities'}
    if 'ASSET_TYPE' in df.columns:
        df['IS_EQUITY'] = df['ASSET_TYPE'].str.lower().str.strip().isin(stock_types)
    else:
        df['IS_EQUITY'] = True  # assume equity if no type info

    logger.info("Normalized %d transactions (%d with tickers, %d equity, "
                "%d buys, %d sells)",
                len(df),
                df['TICKER'].notna().sum(),
                df['IS_EQUITY'].sum(),
                (df['TRADE_TYPE_CLEAN'] == 'buy').sum(),
                (df['TRADE_TYPE_CLEAN'] == 'sell').sum())

    return df


# ═══════════════════════════════════════════════════════════════════════
# MEMBER-TO-VOTE MATCHING
# ═══════════════════════════════════════════════════════════════════════

# Crosswalk: legislator name → BioGuide ID / GovTrack ID for vote matching
# This enables joining trades to roll-call votes

def load_legislators_crosswalk():
    """Load the @unitedstates/congress-legislators YAML/CSV crosswalk.

    Downloads from the canonical GitHub source maintained by the
    @unitedstates project. Maps legislator names to IDs used in
    GovTrack roll-call vote data.

    Returns
    -------
    pd.DataFrame
        Columns: full_name, last_name, first_name, bioguide_id,
        govtrack_id, party, state, chamber, start_date, end_date
    """
    url = ('https://raw.githubusercontent.com/unitedstates/'
           'congress-legislators/main/legislators-current.yaml')
    hist_url = ('https://raw.githubusercontent.com/unitedstates/'
                'congress-legislators/main/legislators-historical.yaml')

    all_legislators = []
    for u in [url, hist_url]:
        try:
            resp = requests.get(u, timeout=30)
            resp.raise_for_status()
            data = yaml.safe_load(resp.text)
        except Exception as e:
            logger.warning("Failed to load legislators from %s: %s", u, e)
            continue

        for leg in data:
            name = leg.get('name', {})
            bio = leg.get('id', {})
            for term in leg.get('terms', []):
                all_legislators.append({
                    'full_name': f"{name.get('first', '')} {name.get('last', '')}",
                    'last_name': name.get('last', ''),
                    'first_name': name.get('first', ''),
                    'bioguide_id': bio.get('bioguide', ''),
                    'govtrack_id': bio.get('govtrack', ''),
                    'party': term.get('party', ''),
                    'state': term.get('state', ''),
                    'chamber': 'Senate' if term.get('type') == 'sen' else 'House',
                    'start_date': term.get('start', ''),
                    'end_date': term.get('end', ''),
                })

    df = pd.DataFrame(all_legislators)
    if not df.empty:
        df['start_date'] = pd.to_datetime(df['start_date'], errors='coerce')
        df['end_date'] = pd.to_datetime(df['end_date'], errors='coerce')

    logger.info("Legislators crosswalk: %d term records, %d unique members",
                len(df), df['bioguide_id'].nunique())
    return df


# ═══════════════════════════════════════════════════════════════════════
# SENATE EFD CSV NORMALIZER
# ═══════════════════════════════════════════════════════════════════════

def _normalize_senate_efd_csv(path):
    """Normalize the raw Senate eFD scraped CSV into the standard schema."""
    df = pd.read_csv(path)
    logger.info("Senate eFD CSV: %d rows from %s", len(df), path)

    # Rename columns to match standard schema
    col_map = {
        'member': 'MEMBER',
        'chamber': 'CHAMBER',
        'transaction_date': 'TRANSACTION_DATE',
        'disclosure_date': 'DISCLOSURE_DATE',
        'ticker': 'TICKER',
        'asset_name': 'ASSET_NAME',
        'asset_type': 'ASSET_TYPE',
        'trade_type': 'TRADE_TYPE',
        'amount': 'AMOUNT_TEXT',
        'owner': 'OWNER',
        'filing_url': 'FILING_URL',
        'comment': 'COMMENT',
    }
    df = df.rename(columns=col_map)

    return _finalize_schema(df)


# ═══════════════════════════════════════════════════════════════════════
# UNIFIED LOADER
# ═══════════════════════════════════════════════════════════════════════

def load_stock_act_trades(source='auto', cache=True,
                          start_date='01/01/2012', max_ptrs=None):
    """Load STOCK Act congressional trading data.

    Parameters
    ----------
    source : str
        'auto' (try pre-scraped first, then Senate eFD),
        'kadoa' (GitHub pre-scraped only),
        'senate' (direct Senate eFD scraping),
        'all' (combine Senate scraping + Kadoa)
    cache : bool
        If True, save/load from local CSV cache.
    start_date : str
        Start date for Senate eFD search (MM/DD/YYYY).
    max_ptrs : int, optional
        Max PTR filings to parse (for testing).

    Returns
    -------
    pd.DataFrame
        Unified, normalized congressional trading data.
    """
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    cache_path = os.path.join(OUTPUT_DIR, 'stock_act_trades.csv')

    # Check cache
    if cache and os.path.exists(cache_path):
        df = pd.read_csv(cache_path, parse_dates=['TRANSACTION_DATE', 'DISCLOSURE_DATE'])
        logger.info("Loaded %d cached STOCK Act trades from %s", len(df), cache_path)
        return df

    df = pd.DataFrame()

    if source in ('auto', 'kadoa'):
        logger.info("Attempting Kadoa/pre-scraped download...")
        df = download_kadoa_data()
        if not df.empty and source == 'auto':
            logger.info("Using pre-scraped data (%d trades)", len(df))

    if source == 'senate_efd':
        # Load from previously scraped Senate eFD CSV
        efd_path = os.path.join(OUTPUT_DIR, 'senate_efd_trades.csv')
        if os.path.exists(efd_path):
            df = _normalize_senate_efd_csv(efd_path)
        else:
            logger.warning("No Senate eFD CSV at %s — run scrape first", efd_path)

    if source == 'combined':
        # Combine Kadoa (2024-2026, Senate+House) with Senate eFD (2013-2026)
        kadoa = download_kadoa_data()
        efd_path = os.path.join(OUTPUT_DIR, 'senate_efd_trades.csv')
        efd = _normalize_senate_efd_csv(efd_path) if os.path.exists(efd_path) else pd.DataFrame()
        frames = [f for f in [kadoa, efd] if not f.empty]
        if frames:
            df = pd.concat(frames, ignore_index=True)
            # Deduplicate on (MEMBER, TRANSACTION_DATE, TICKER, TRADE_TYPE_CLEAN)
            dedup_cols = ['MEMBER', 'TRANSACTION_DATE', 'TICKER', 'TRADE_TYPE_CLEAN']
            existing_cols = [c for c in dedup_cols if c in df.columns]
            if existing_cols:
                before = len(df)
                df = df.drop_duplicates(subset=existing_cols, keep='first')
                logger.info("Combined: %d trades (deduped from %d)",
                            len(df), before)

    if df.empty and source in ('auto', 'senate', 'all'):
        logger.info("Scraping Senate eFD directly...")
        scraper = SenateDisclosureScraper()
        df = scraper.download_all(start_date=start_date, max_ptrs=max_ptrs)

    if source == 'all' and not df.empty:
        # Supplement with pre-scraped if we scraped Senate directly
        kadoa = download_kadoa_data()
        if not kadoa.empty:
            # Deduplicate on (MEMBER, TRANSACTION_DATE, TICKER, TRADE_TYPE)
            combined = pd.concat([df, kadoa], ignore_index=True)
            dedup_cols = ['MEMBER', 'TRANSACTION_DATE', 'TICKER', 'TRADE_TYPE_CLEAN']
            existing_cols = [c for c in dedup_cols if c in combined.columns]
            combined = combined.drop_duplicates(subset=existing_cols, keep='first')
            logger.info("Combined: %d trades (deduped from %d + %d)",
                        len(combined), len(df), len(kadoa))
            df = combined

    # Also load House index (metadata only — no transaction details from PDFs)
    try:
        house_idx = download_house_index()
        if not house_idx.empty:
            house_idx.to_csv(
                os.path.join(OUTPUT_DIR, 'house_ptr_index.csv'),
                index=False
            )
            logger.info("Saved House PTR index: %d filings", len(house_idx))
    except Exception as e:
        logger.warning("House index download failed: %s", e)

    # Also load legislators crosswalk
    try:
        legislators = load_legislators_crosswalk()
        if not legislators.empty:
            legislators.to_csv(
                os.path.join(OUTPUT_DIR, 'legislators_crosswalk.csv'),
                index=False
            )

            # Enrich trades with party/state from crosswalk
            if not df.empty:
                df = _enrich_with_legislators(df, legislators)
    except Exception as e:
        logger.warning("Legislators crosswalk failed: %s", e)

    # Cache
    if cache and not df.empty:
        df.to_csv(cache_path, index=False)
        logger.info("Cached %d STOCK Act trades to %s", len(df), cache_path)

    return df


def _enrich_with_legislators(trades, legislators):
    """Add party and state from legislators crosswalk via fuzzy name match."""
    if trades.empty or legislators.empty:
        return trades

    # Build a lookup: last_name → list of (full_name, party, state, chamber)
    from collections import defaultdict
    name_lookup = defaultdict(list)
    for _, leg in legislators.iterrows():
        name_lookup[leg['last_name'].lower()].append({
            'full_name': leg['full_name'],
            'party': leg['party'],
            'state': leg['state'],
            'chamber': leg['chamber'],
        })

    enriched_party = []
    enriched_state = []

    for _, trade in trades.iterrows():
        member = str(trade.get('MEMBER', ''))
        parts = member.split()
        last = parts[-1].lower() if parts else ''

        candidates = name_lookup.get(last, [])
        match = None
        if len(candidates) == 1:
            match = candidates[0]
        elif len(candidates) > 1:
            # Try to match first name
            first = parts[0].lower() if parts else ''
            for c in candidates:
                if c['full_name'].lower().startswith(first):
                    match = c
                    break
            if not match:
                match = candidates[0]

        if match:
            enriched_party.append(match['party'])
            enriched_state.append(match['state'])
        else:
            enriched_party.append(trade.get('PARTY'))
            enriched_state.append(trade.get('STATE'))

    # Only fill where missing
    if trades['PARTY'].isna().any():
        trades.loc[trades['PARTY'].isna(), 'PARTY'] = [
            p for p, orig in zip(enriched_party, trades['PARTY'])
            if pd.isna(orig)
        ]
    if trades['STATE'].isna().any():
        trades.loc[trades['STATE'].isna(), 'STATE'] = [
            s for s, orig in zip(enriched_state, trades['STATE'])
            if pd.isna(orig)
        ]

    n_party = trades['PARTY'].notna().sum()
    n_state = trades['STATE'].notna().sum()
    logger.info("Enriched: %d/%d with party, %d/%d with state",
                n_party, len(trades), n_state, len(trades))

    return trades
