"""Shared configuration, constants, logger, and FRED helpers."""

import os
import logging
from datetime import datetime, timedelta

import pandas as pd
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Get the directory where the *original* script lived (project root)
SCRIPT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# API Keys
API_KEY = os.getenv('FRED_API_KEY')

# Date range defaults — dissertation analysis window (intentionally fixed)
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


def _validate_fred_api_key():
    """Raise ValueError if FRED API key is not configured."""
    if not API_KEY:
        raise ValueError(
            "FRED API key not found. Set FRED_API_KEY in .env. "
            "Get a free key at: https://fred.stlouisfed.org/docs/api/api_key.html"
        )


def _download_fred_series(series_dict, start_date, end_date):
    """Download multiple FRED series into a DataFrame."""
    import pandas_datareader as pdr

    _validate_fred_api_key()
    data = {}
    for name, code in series_dict.items():
        try:
            logger.debug("Downloading %s (%s)...", name, code)
            df = pdr.DataReader(code, 'fred', start=start_date, end=end_date)
            data[name] = df.iloc[:, 0]
        except Exception as e:
            logger.warning("Could not download %s (%s): %s", name, code, e)
    return pd.DataFrame(data)


def _load_sec_cik_mapping(cache_file, user_agent=None):
    """Load or refresh ticker→CIK mapping from SEC (shared by sec_filings + sec_form4).

    Parameters
    ----------
    cache_file : str
        Path to the JSON cache file.
    user_agent : str, optional
        SEC-required User-Agent header.

    Returns
    -------
    dict : {ticker: cik_str}
    """
    import json
    import time
    import requests

    if os.path.exists(cache_file):
        age = time.time() - os.path.getmtime(cache_file)
        if age < 7 * 24 * 3600:
            try:
                with open(cache_file, 'r') as f:
                    return json.load(f)
            except Exception:
                pass

    logger.info("Downloading ticker-to-CIK mapping from SEC...")
    url = "https://www.sec.gov/files/company_tickers.json"
    headers = {'User-Agent': user_agent or 'research@example.com',
               'Accept-Encoding': 'gzip, deflate',
               'Host': 'www.sec.gov'}
    try:
        resp = requests.get(url, headers=headers, timeout=30)
        time.sleep(0.15)
        if resp.status_code != 200:
            logger.warning("CIK download failed: HTTP %s", resp.status_code)
            return {}
    except Exception as e:
        logger.warning("CIK download failed: %s", e)
        return {}

    data = resp.json()
    mapping = {}
    for entry in data.values():
        t = entry.get('ticker', '').upper()
        c = str(entry.get('cik_str', '')).zfill(10)
        if t and c:
            mapping[t] = c

    os.makedirs(os.path.dirname(cache_file) or '.', exist_ok=True)
    with open(cache_file, 'w') as f:
        json.dump(mapping, f)
    logger.info("Loaded %d ticker-to-CIK mappings", len(mapping))
    return mapping


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
