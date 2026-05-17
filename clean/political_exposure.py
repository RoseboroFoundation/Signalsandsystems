"""Political exposure data — lobbying expenditures and PAC contributions.

Provides firm-level political connection scores used as the cross-sectional
moderator in Essay 3 (Jagolinzer et al. 2020 channel).

Data sources:
  - OpenSecrets bulk data (lobbying_data.csv, pac_data.csv)
  - FEC API (fallback for PAC data)
  - Manual ticker-to-company mapping for matching
"""

import os
import time

import numpy as np
import pandas as pd
import requests

from .config import logger

# ═══════════════════════════════════════════════════════════════════════
# TICKER-TO-COMPANY MAPPING
# ═══════════════════════════════════════════════════════════════════════

TICKER_COMPANY_MAP = {
    # Technology
    'AAPL': 'Apple Inc', 'MSFT': 'Microsoft Corp', 'GOOGL': 'Alphabet Inc',
    'GOOG': 'Alphabet Inc', 'META': 'Meta Platforms', 'AMZN': 'Amazon',
    'NFLX': 'Netflix', 'CRM': 'Salesforce', 'ORCL': 'Oracle Corp',
    'IBM': 'International Business Machines', 'INTC': 'Intel Corp',
    'AMD': 'Advanced Micro Devices', 'NVDA': 'Nvidia', 'ADBE': 'Adobe',
    'CSCO': 'Cisco Systems', 'QCOM': 'Qualcomm', 'TXN': 'Texas Instruments',
    'AVGO': 'Broadcom', 'NOW': 'ServiceNow', 'UBER': 'Uber Technologies',
    'LYFT': 'Lyft', 'SNAP': 'Snap Inc', 'TWTR': 'Twitter',
    'PYPL': 'PayPal', 'SQ': 'Block Inc',

    # Finance
    'JPM': 'JPMorgan Chase', 'BAC': 'Bank of America', 'WFC': 'Wells Fargo',
    'GS': 'Goldman Sachs', 'MS': 'Morgan Stanley', 'C': 'Citigroup',
    'BRK.B': 'Berkshire Hathaway', 'AXP': 'American Express',
    'V': 'Visa', 'MA': 'Mastercard', 'BLK': 'BlackRock',
    'SCHW': 'Charles Schwab', 'USB': 'U.S. Bancorp', 'PNC': 'PNC Financial',

    # Healthcare
    'JNJ': 'Johnson & Johnson', 'PFE': 'Pfizer', 'UNH': 'UnitedHealth',
    'MRK': 'Merck', 'ABBV': 'AbbVie', 'TMO': 'Thermo Fisher',
    'ABT': 'Abbott Laboratories', 'LLY': 'Eli Lilly', 'BMY': 'Bristol-Myers Squibb',
    'AMGN': 'Amgen', 'GILD': 'Gilead Sciences', 'CVS': 'CVS Health',
    'CI': 'Cigna', 'HUM': 'Humana', 'MDT': 'Medtronic',
    'ISRG': 'Intuitive Surgical', 'MRNA': 'Moderna',

    # Energy
    'XOM': 'Exxon Mobil', 'CVX': 'Chevron', 'COP': 'ConocoPhillips',
    'SLB': 'Schlumberger', 'EOG': 'EOG Resources', 'PXD': 'Pioneer Natural Resources',
    'MPC': 'Marathon Petroleum', 'VLO': 'Valero Energy', 'PSX': 'Phillips 66',
    'OXY': 'Occidental Petroleum', 'HAL': 'Halliburton',

    # Consumer / Retail
    'WMT': 'Walmart', 'TGT': 'Target', 'COST': 'Costco',
    'HD': 'Home Depot', 'LOW': "Lowe's", 'NKE': 'Nike',
    'SBUX': 'Starbucks', 'MCD': "McDonald's", 'YUM': 'Yum! Brands',
    'KO': 'Coca-Cola', 'PEP': 'PepsiCo', 'PG': 'Procter & Gamble',
    'CL': 'Colgate-Palmolive', 'KMB': 'Kimberly-Clark',
    'MDLZ': 'Mondelez', 'GIS': 'General Mills', 'K': 'Kellogg',
    'HSY': 'Hershey', 'KHC': 'Kraft Heinz', 'STZ': 'Constellation Brands',
    'BUD': 'Anheuser-Busch InBev', 'TAP': 'Molson Coors',
    'DIS': 'Walt Disney', 'CMCSA': 'Comcast', 'NWSA': 'News Corp',

    # Defense / Aerospace
    'BA': 'Boeing', 'LMT': 'Lockheed Martin', 'RTX': 'Raytheon',
    'NOC': 'Northrop Grumman', 'GD': 'General Dynamics',
    'LHX': 'L3Harris Technologies',

    # Transportation
    'DAL': 'Delta Air Lines', 'UAL': 'United Airlines', 'AAL': 'American Airlines',
    'LUV': 'Southwest Airlines', 'FDX': 'FedEx', 'UPS': 'United Parcel Service',
    'UNP': 'Union Pacific', 'CSX': 'CSX Corp',

    # Auto
    'TSLA': 'Tesla', 'F': 'Ford Motor', 'GM': 'General Motors',
    'TM': 'Toyota Motor', 'HMC': 'Honda Motor',

    # Telecom / Media
    'T': 'AT&T', 'VZ': 'Verizon', 'TMUS': 'T-Mobile',
    'CHTR': 'Charter Communications',

    # Industrials
    'GE': 'General Electric', 'HON': 'Honeywell', 'MMM': '3M',
    'CAT': 'Caterpillar', 'DE': 'Deere & Company', 'EMR': 'Emerson Electric',
    'ITW': 'Illinois Tool Works',

    # Utilities / Real Estate
    'NEE': 'NextEra Energy', 'DUK': 'Duke Energy', 'SO': 'Southern Company',
    'AEP': 'American Electric Power', 'D': 'Dominion Energy',

    # Other culture-war-relevant
    'ANF': 'Abercrombie & Fitch', 'AEO': 'American Eagle',
    'GPS': 'Gap Inc', 'LULU': 'Lululemon', 'ATUS': 'Altice USA',
}


# ═══════════════════════════════════════════════════════════════════════
# DATA LOADERS
# ═══════════════════════════════════════════════════════════════════════

def _fetch_lobbyview_data(tickers, cache_dir):
    """Fetch lobbying data from MIT LobbyView REST API.

    API docs: https://rest-api.lobbyview.org/
    Free for academic use, requires API token from lobbyview.org.
    """
    api_token = os.environ.get('LOBBYVIEW_API_TOKEN', '')
    if not api_token:
        logger.info("  LOBBYVIEW_API_TOKEN not set. Skipping LobbyView API.")
        return pd.DataFrame()

    base_url = 'https://rest-api.lobbyview.org'
    headers = {'Authorization': f'Token {api_token}'}
    all_rows = []

    # Look up each company by name
    for ticker in (tickers or []):
        company_name = TICKER_COMPANY_MAP.get(ticker)
        if not company_name:
            continue

        try:
            resp = requests.get(
                f'{base_url}/api/lobbying',
                params={'client_name': company_name, 'page_size': 100},
                headers=headers,
                timeout=15,
            )
            if resp.status_code != 200:
                continue
            data = resp.json()
            results = data.get('results', [])
            for r in results:
                all_rows.append({
                    'registrant_name': r.get('registrant_name', ''),
                    'client_name': r.get('client_name', company_name),
                    'amount': r.get('amount', 0),
                    'year': r.get('year', 0),
                    'filing_type': r.get('filing_type', ''),
                    'issue_codes': r.get('general_issue_code', ''),
                    '_ticker': ticker,
                })
            time.sleep(0.5)
        except Exception as e:
            logger.debug("LobbyView API error for %s: %s", ticker, e)
            continue

    if all_rows:
        df = pd.DataFrame(all_rows)
        # Cache locally for future runs
        csv_path = os.path.join(cache_dir, 'lobbying_data.csv')
        df.to_csv(csv_path, index=False)
        logger.info("  Fetched %d lobbying records from LobbyView API", len(df))
        return df

    return pd.DataFrame()


def _fetch_fec_pac_data(tickers, cache_dir):
    """Fetch PAC data from FEC API.

    Uses DEMO_KEY (limited rate) unless FEC_API_KEY is set.
    Register for free at https://api.open.fec.gov/developers/
    """
    api_key = os.environ.get('FEC_API_KEY', 'DEMO_KEY')
    base_url = 'https://api.open.fec.gov/v1'
    all_rows = []

    for ticker in (tickers or []):
        company_name = TICKER_COMPANY_MAP.get(ticker)
        if not company_name:
            continue

        try:
            # Search for committee by company name
            resp = requests.get(
                f'{base_url}/committees/',
                params={
                    'api_key': api_key,
                    'q': company_name,
                    'committee_type': ['N', 'Q', 'W'],  # PAC types
                    'per_page': 5,
                },
                timeout=15,
            )
            if resp.status_code != 200:
                continue

            data = resp.json()
            for comm in data.get('results', []):
                committee_id = comm.get('committee_id', '')
                committee_name = comm.get('name', '')

                # Get total receipts by cycle
                resp2 = requests.get(
                    f'{base_url}/committee/{committee_id}/totals/',
                    params={'api_key': api_key, 'per_page': 20},
                    timeout=15,
                )
                if resp2.status_code != 200:
                    continue

                for total in resp2.json().get('results', []):
                    cycle = total.get('cycle', 0)
                    amount = total.get('receipts', 0) or 0
                    if cycle and amount:
                        all_rows.append({
                            'committee_name': committee_name,
                            'contributor_name': company_name,
                            'amount': amount,
                            'year': cycle,
                            'recipient_party': '',
                            '_ticker': ticker,
                        })

            time.sleep(1.0)  # Rate limit
        except Exception as e:
            logger.debug("FEC API error for %s: %s", ticker, e)
            continue

    if all_rows:
        df = pd.DataFrame(all_rows)
        csv_path = os.path.join(cache_dir, 'pac_data.csv')
        df.to_csv(csv_path, index=False)
        logger.info("  Fetched %d PAC records from FEC API", len(df))
        return df

    return pd.DataFrame()


def load_lobbying_data(cache_dir, tickers=None):
    """Load lobbying data from CSV or LobbyView API.

    Expected CSV columns: registrant_name, client_name, amount, year,
    filing_type, issue_codes
    """
    csv_path = os.path.join(cache_dir, 'lobbying_data.csv')
    if not os.path.exists(csv_path):
        # Try LobbyView API
        logger.info("  Lobbying CSV not found. Trying LobbyView API...")
        api_df = _fetch_lobbyview_data(tickers, cache_dir)
        if not api_df.empty:
            return api_df

        logger.warning(
            "Lobbying data not available. Options:\n"
            "  1. Set LOBBYVIEW_API_TOKEN env var (free academic: https://lobbyview.org)\n"
            "  2. Download from https://www.opensecrets.org/bulk-data\n"
            "  3. Download from https://lda.senate.gov/filings/xml/\n"
            "Save as %s", csv_path
        )
        return pd.DataFrame(columns=[
            'registrant_name', 'client_name', 'amount', 'year',
            'filing_type', 'issue_codes',
        ])

    logger.info("  Loading lobbying data from %s", csv_path)
    df = pd.read_csv(csv_path)
    df['amount'] = pd.to_numeric(df.get('amount', pd.Series()), errors='coerce').fillna(0)
    df['year'] = pd.to_numeric(df.get('year', pd.Series()), errors='coerce')
    logger.info("  Loaded %d lobbying records", len(df))
    return df


def load_pac_data(cache_dir, tickers=None):
    """Load PAC contribution data from CSV or FEC API.

    Expected CSV columns: committee_name, contributor_name, amount, year,
    recipient_party
    """
    csv_path = os.path.join(cache_dir, 'pac_data.csv')
    if not os.path.exists(csv_path):
        # Try FEC API
        logger.info("  PAC CSV not found. Trying FEC API...")
        api_df = _fetch_fec_pac_data(tickers, cache_dir)
        if not api_df.empty:
            return api_df

        logger.warning(
            "PAC data not available. Options:\n"
            "  1. Set FEC_API_KEY env var (free: https://api.open.fec.gov/developers/)\n"
            "  2. Download from https://www.fec.gov/data/browse-data/?tab=bulk-data\n"
            "Save as %s", csv_path
        )
        return pd.DataFrame(columns=[
            'committee_name', 'contributor_name', 'amount', 'year',
            'recipient_party',
        ])

    logger.info("  Loading PAC data from %s", csv_path)
    df = pd.read_csv(csv_path)
    df['amount'] = pd.to_numeric(df.get('amount', pd.Series()), errors='coerce').fillna(0)
    df['year'] = pd.to_numeric(df.get('year', pd.Series()), errors='coerce')
    logger.info("  Loaded %d PAC records", len(df))
    return df


def match_tickers_to_lobby_data(tickers, lobbying_df, pac_df):
    """Match stock tickers to lobbying/PAC data using company name mapping.

    Returns aggregated exposure by ticker × year.
    """
    if lobbying_df.empty and pac_df.empty:
        return pd.DataFrame(columns=[
            'TICKER', 'YEAR', 'COMPANY_NAME', 'LOBBYING_TOTAL', 'PAC_TOTAL',
            'N_LOBBYISTS', 'N_BILLS_LOBBIED',
        ])

    # Build reverse mapping: lowercase company name fragments → ticker
    name_to_ticker = {}
    for ticker, name in TICKER_COMPANY_MAP.items():
        if ticker in (tickers or TICKER_COMPANY_MAP.keys()):
            # Store multiple fragments for fuzzy matching
            name_lower = name.lower()
            name_to_ticker[name_lower] = ticker
            # Also store first word for partial matching
            first_word = name_lower.split()[0]
            if len(first_word) > 3:
                name_to_ticker[first_word] = ticker

    def _match_name(company_name):
        """Find ticker for a company name via substring matching."""
        if not isinstance(company_name, str):
            return None
        name_lower = company_name.lower().strip()
        # Exact match first
        for pattern, ticker in name_to_ticker.items():
            if pattern in name_lower or name_lower in pattern:
                return ticker
        return None

    rows = []
    target_tickers = set(tickers) if tickers else set(TICKER_COMPANY_MAP.keys())

    # Process lobbying data
    lobby_by_ticker = {}
    if not lobbying_df.empty:
        for col in ['client_name', 'registrant_name']:
            if col not in lobbying_df.columns:
                continue
            lobbying_df[f'_ticker_{col}'] = lobbying_df[col].apply(_match_name)

        # Use client_name match first, then registrant_name
        name_col = 'client_name' if 'client_name' in lobbying_df.columns else 'registrant_name'
        ticker_col = f'_ticker_{name_col}'
        if ticker_col in lobbying_df.columns:
            matched = lobbying_df[lobbying_df[ticker_col].notna()].copy()
            if not matched.empty:
                grouped = matched.groupby([ticker_col, 'year']).agg(
                    LOBBYING_TOTAL=('amount', 'sum'),
                    N_LOBBYISTS=('registrant_name', 'nunique') if 'registrant_name' in matched.columns else ('amount', 'count'),
                    N_BILLS_LOBBIED=('issue_codes', 'nunique') if 'issue_codes' in matched.columns else ('amount', 'count'),
                ).reset_index()
                grouped.columns = ['TICKER', 'YEAR', 'LOBBYING_TOTAL', 'N_LOBBYISTS', 'N_BILLS_LOBBIED']
                for _, r in grouped.iterrows():
                    lobby_by_ticker[(r['TICKER'], int(r['YEAR']))] = {
                        'LOBBYING_TOTAL': r['LOBBYING_TOTAL'],
                        'N_LOBBYISTS': r['N_LOBBYISTS'],
                        'N_BILLS_LOBBIED': r['N_BILLS_LOBBIED'],
                    }

    # Process PAC data
    pac_by_ticker = {}
    if not pac_df.empty:
        for col in ['contributor_name', 'committee_name']:
            if col not in pac_df.columns:
                continue
            pac_df[f'_ticker_{col}'] = pac_df[col].apply(_match_name)

        name_col = 'contributor_name' if 'contributor_name' in pac_df.columns else 'committee_name'
        ticker_col = f'_ticker_{name_col}'
        if ticker_col in pac_df.columns:
            matched = pac_df[pac_df[ticker_col].notna()].copy()
            if not matched.empty:
                grouped = matched.groupby([ticker_col, 'year']).agg(
                    PAC_TOTAL=('amount', 'sum'),
                ).reset_index()
                grouped.columns = ['TICKER', 'YEAR', 'PAC_TOTAL']
                for _, r in grouped.iterrows():
                    pac_by_ticker[(r['TICKER'], int(r['YEAR']))] = r['PAC_TOTAL']

    # Build unified exposure table
    years = range(2000, 2026)
    for ticker in target_tickers:
        company_name = TICKER_COMPANY_MAP.get(ticker, ticker)
        for year in years:
            lobby = lobby_by_ticker.get((ticker, year), {})
            pac_total = pac_by_ticker.get((ticker, year), 0)
            lobbying_total = lobby.get('LOBBYING_TOTAL', 0)

            # Skip years with no data at all
            if lobbying_total == 0 and pac_total == 0:
                continue

            rows.append({
                'TICKER': ticker,
                'YEAR': year,
                'COMPANY_NAME': company_name,
                'LOBBYING_TOTAL': lobbying_total,
                'PAC_TOTAL': pac_total,
                'N_LOBBYISTS': lobby.get('N_LOBBYISTS', 0),
                'N_BILLS_LOBBIED': lobby.get('N_BILLS_LOBBIED', 0),
            })

    return pd.DataFrame(rows)


def compute_political_connection_score(exposure_df):
    """Compute composite political connection score per ticker-year.

    Score = 0.6 * lobbying_percentile + 0.4 * pac_percentile (within year).
    """
    if exposure_df.empty:
        exposure_df['POLITICAL_CONNECTION_SCORE'] = pd.Series(dtype=float)
        return exposure_df

    df = exposure_df.copy()

    # Compute percentile ranks within each year
    df['LOBBY_PCT'] = df.groupby('YEAR')['LOBBYING_TOTAL'].rank(pct=True)
    df['PAC_PCT'] = df.groupby('YEAR')['PAC_TOTAL'].rank(pct=True)

    df['POLITICAL_CONNECTION_SCORE'] = 0.6 * df['LOBBY_PCT'] + 0.4 * df['PAC_PCT']

    df = df.drop(columns=['LOBBY_PCT', 'PAC_PCT'])
    return df


# ═══════════════════════════════════════════════════════════════════════
# MAIN LOADER
# ═══════════════════════════════════════════════════════════════════════

def load_political_exposure(tickers=None, start_date='2000-01-01',
                            end_date='2025-12-31',
                            cache_dir='./data/political_exposure',
                            force_refresh=False):
    """Load firm-level political exposure data.

    Parameters
    ----------
    tickers : list of str, optional
        Stock tickers to include. If None, uses all mapped tickers.
    start_date : str
        Start of date range (filters by year).
    end_date : str
        End of date range.
    cache_dir : str
        Directory for data files.
    force_refresh : bool
        If True, recompute from source files.

    Returns
    -------
    pd.DataFrame
        Columns: TICKER, YEAR, COMPANY_NAME, LOBBYING_TOTAL, PAC_TOTAL,
        N_LOBBYISTS, N_BILLS_LOBBIED, POLITICAL_CONNECTION_SCORE,
        TOP_LOBBYING_ISSUES, SECTOR.
    """
    os.makedirs(cache_dir, exist_ok=True)

    # Check processed cache
    processed_path = os.path.join(cache_dir, 'political_exposure.csv')
    if os.path.exists(processed_path) and not force_refresh:
        age_days = (time.time() - os.path.getmtime(processed_path)) / 86400
        if age_days < 30:
            logger.info("Loading cached political exposure (%.1f days old)", age_days)
            df = pd.read_csv(processed_path)
            if tickers:
                df = df[df['TICKER'].isin(tickers)]
            logger.info("  %d exposure records loaded from cache", len(df))
            return df

    logger.info("Building political exposure dataset...")

    # Load source data (pass tickers for API fallback)
    target_tickers = list(tickers) if tickers else list(TICKER_COMPANY_MAP.keys())
    lobbying_df = load_lobbying_data(cache_dir, tickers=target_tickers)
    pac_df = load_pac_data(cache_dir, tickers=target_tickers)

    if lobbying_df.empty and pac_df.empty:
        logger.warning("No lobbying or PAC data available. "
                       "Political exposure will be empty.")
        return pd.DataFrame(columns=[
            'TICKER', 'YEAR', 'COMPANY_NAME', 'LOBBYING_TOTAL', 'PAC_TOTAL',
            'N_LOBBYISTS', 'N_BILLS_LOBBIED', 'POLITICAL_CONNECTION_SCORE',
            'TOP_LOBBYING_ISSUES', 'SECTOR',
        ])

    # Match tickers to lobbying/PAC data
    exposure = match_tickers_to_lobby_data(tickers, lobbying_df, pac_df)

    if exposure.empty:
        logger.warning("No ticker matches found in lobbying/PAC data")
        return pd.DataFrame(columns=[
            'TICKER', 'YEAR', 'COMPANY_NAME', 'LOBBYING_TOTAL', 'PAC_TOTAL',
            'N_LOBBYISTS', 'N_BILLS_LOBBIED', 'POLITICAL_CONNECTION_SCORE',
            'TOP_LOBBYING_ISSUES', 'SECTOR',
        ])

    # Compute political connection score
    exposure = compute_political_connection_score(exposure)

    # Add placeholder columns
    if 'TOP_LOBBYING_ISSUES' not in exposure.columns:
        exposure['TOP_LOBBYING_ISSUES'] = ''
    if 'SECTOR' not in exposure.columns:
        exposure['SECTOR'] = ''

    # Filter to date range
    start_year = int(start_date[:4])
    end_year = int(end_date[:4])
    exposure = exposure[
        (exposure['YEAR'] >= start_year) & (exposure['YEAR'] <= end_year)
    ].copy()

    # Save processed cache
    exposure.to_csv(processed_path, index=False)

    logger.info("Political exposure: %d records, %d tickers, %d years",
                len(exposure), exposure['TICKER'].nunique(),
                exposure['YEAR'].nunique())

    return exposure
