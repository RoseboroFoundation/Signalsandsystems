"""SEC Form 4 insider trading data downloader."""

import os
import time
import json
import logging
from datetime import datetime
from typing import List, Dict

import pandas as pd
import requests
from bs4 import BeautifulSoup

from .config import logger

class Form4Downloader:
    """Download and parse SEC Form 4 filings."""

    def __init__(self, output_dir='./sec_form4_data'):
        self.output_dir = output_dir
        os.makedirs(output_dir, exist_ok=True)
        # SEC requires a User-Agent with contact info
        self.user_agent = os.getenv('SEC_USER_AGENT')
        if not self.user_agent:
            logger.warning(
                "SEC_USER_AGENT not set. SEC requires contact info. "
                "Set SEC_USER_AGENT='Name email' in .env"
            )
            self.user_agent = 'SignalsAndSystems/1.0'

    def _get_headers(self, url):
        """Get headers for SEC API requests with correct Host."""
        if 'data.sec.gov' in url:
            host = 'data.sec.gov'
        else:
            host = 'www.sec.gov'
        return {
            'User-Agent': self.user_agent,
            'Accept-Encoding': 'gzip, deflate',
            'Host': host
        }

    def get_company_cik(self, ticker: str) -> str:
        """
        Get company CIK from ticker symbol using SEC EDGAR API.

        Parameters:
        -----------
        ticker : str
            Stock ticker symbol

        Returns:
        --------
        str : CIK number (zero-padded to 10 digits) or None if not found
        """
        # Cache the ticker-to-CIK mapping
        if not hasattr(self, '_cik_mapping'):
            self._cik_mapping = self._load_cik_mapping()

        ticker_upper = ticker.upper()
        if ticker_upper in self._cik_mapping:
            return self._cik_mapping[ticker_upper]
        return None

    def _load_cik_mapping(self) -> Dict[str, str]:
        """
        Load ticker to CIK mapping from SEC.

        Returns:
        --------
        Dict[str, str] : Mapping of ticker symbols to CIK numbers
        """
        cache_file = os.path.join(self.output_dir, 'ticker_cik_mapping.json')

        # Check for cached mapping (refresh if older than 7 days)
        if os.path.exists(cache_file):
            file_age = time.time() - os.path.getmtime(cache_file)
            if file_age < 7 * 24 * 60 * 60:  # 7 days
                try:
                    with open(cache_file, 'r') as f:
                        logger.info("Loading cached CIK mapping...")
                        return json.load(f)
                except Exception as e:
                    logger.debug("Failed to read CIK cache: %s", e)

        logger.info("Downloading ticker-to-CIK mapping from SEC...")
        try:
            # SEC provides a JSON file with all company tickers and CIKs
            url = "https://www.sec.gov/files/company_tickers.json"
            response = requests.get(url, headers=self._get_headers(url))
            time.sleep(0.2)

            if response.status_code == 200:
                data = response.json()
                mapping = {}
                for entry in data.values():
                    ticker = entry.get('ticker', '').upper()
                    cik = str(entry.get('cik_str', '')).zfill(10)
                    if ticker and cik:
                        mapping[ticker] = cik

                # Cache the mapping
                with open(cache_file, 'w') as f:
                    json.dump(mapping, f)

                logger.info("Loaded %d ticker-to-CIK mappings", len(mapping))
                return mapping
            else:
                logger.error("Failed to download CIK mapping: HTTP %s", response.status_code)
                return {}
        except Exception as e:
            logger.error("loading CIK mapping: %s", e)
            return {}

    def download_form4_filings(
        self,
        ticker: str,
        cik: str,
        start_date: str = '2000-01-01',
        end_date: str = '2025-12-31',
        max_filings: int = 500
    ) -> List[Dict]:
        """
        Download all Form 4 filings for a company within date range.

        Uses the SEC EDGAR JSON API for reliable data retrieval.

        Parameters:
        -----------
        ticker : str
            Stock ticker symbol
        cik : str
            SEC CIK number
        start_date : str
            Start date (YYYY-MM-DD)
        end_date : str
            End date (YYYY-MM-DD)
        max_filings : int
            Maximum number of filings to retrieve

        Returns:
        --------
        List[Dict] : List of filing metadata
        """
        filings = []
        start_dt = datetime.strptime(start_date, '%Y-%m-%d')
        end_dt = datetime.strptime(end_date, '%Y-%m-%d')

        try:
            # Use the SEC JSON API for submissions
            cik_padded = cik.zfill(10)
            url = f"https://data.sec.gov/submissions/CIK{cik_padded}.json"

            response = requests.get(url, headers=self._get_headers(url))
            time.sleep(0.15)

            if response.status_code != 200:
                logger.error("  HTTP %s for %s", response.status_code, ticker)
                return []

            data = response.json()

            # Get recent filings
            recent_filings = data.get('filings', {}).get('recent', {})
            forms = recent_filings.get('form', [])
            dates = recent_filings.get('filingDate', [])
            accessions = recent_filings.get('accessionNumber', [])
            primary_docs = recent_filings.get('primaryDocument', [])

            # Process each filing
            for i, form in enumerate(forms):
                if form in ['4', '4/A']:  # Form 4 and amendments
                    if i >= len(dates):
                        continue

                    filing_date = dates[i]

                    # Check date range
                    filing_dt = datetime.strptime(filing_date, '%Y-%m-%d')
                    if start_dt <= filing_dt <= end_dt:
                        accession = accessions[i].replace('-', '')
                        primary_doc = primary_docs[i] if i < len(primary_docs) else ''

                        # Construct the filing URL
                        doc_url = (
                            f"https://www.sec.gov/Archives/edgar/data/"
                            f"{int(cik)}/{accession}/{primary_doc}"
                        )

                        filing_info = {
                            'ticker': ticker,
                            'cik': cik,
                            'filing_date': filing_date,
                            'accession_number': accessions[i],
                            'filing_url': doc_url,
                            'form_type': form
                        }
                        filings.append(filing_info)

                        if len(filings) >= max_filings:
                            break

            # Also check older filings if available
            older_files = data.get('filings', {}).get('files', [])
            for file_info in older_files[:3]:  # Check up to 3 older filing batches
                if len(filings) >= max_filings:
                    break

                file_url = f"https://data.sec.gov/submissions/{file_info.get('name', '')}"
                try:
                    file_response = requests.get(file_url, headers=self._get_headers(file_url))
                    time.sleep(0.15)

                    if file_response.status_code == 200:
                        file_data = file_response.json()
                        forms = file_data.get('form', [])
                        dates = file_data.get('filingDate', [])
                        accessions = file_data.get('accessionNumber', [])
                        primary_docs = file_data.get('primaryDocument', [])

                        for i, form in enumerate(forms):
                            if form in ['4', '4/A']:
                                if i >= len(dates):
                                    continue

                                filing_date = dates[i]
                                filing_dt = datetime.strptime(filing_date, '%Y-%m-%d')
                                if start_dt <= filing_dt <= end_dt:
                                    accession = accessions[i].replace('-', '')
                                    primary_doc = primary_docs[i] if i < len(primary_docs) else ''

                                    doc_url = (
                                        f"https://www.sec.gov/Archives/edgar/data/"
                                        f"{int(cik)}/{accession}/{primary_doc}"
                                    )

                                    filing_info = {
                                        'ticker': ticker,
                                        'cik': cik,
                                        'filing_date': filing_date,
                                        'accession_number': accessions[i],
                                        'filing_url': doc_url,
                                        'form_type': form
                                    }
                                    filings.append(filing_info)

                                    if len(filings) >= max_filings:
                                        break
                except Exception as e:
                    logger.warning("Failed to fetch older filing batch: %s", e)
                    continue

            if len(filings) > 0:
                logger.info("  Found %d Form 4 filings for %s", len(filings), ticker)
            else:
                logger.info("  No Form 4 filings in date range for %s", ticker)

            return filings

        except Exception as e:
            logger.error("Error downloading Form 4 filings for %s: %s", ticker, e)
            return []

    def parse_form4_xml(self, filing_url: str) -> List[Dict]:
        """
        Parse Form 4 to extract insider trading details.

        Returns:
        --------
        List[Dict] : Parsed transaction data
        """
        try:
            response = requests.get(filing_url, headers=self._get_headers(filing_url))
            time.sleep(0.2)
        except Exception as e:
            logger.error("fetching filing page: %s", e)
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
            logger.info("    No XML document found")
            return []

        # Fetch and parse the XML
        xml_response = requests.get(xml_link, headers=self._get_headers(xml_link))
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
            except Exception as e:
                logger.debug("Failed to parse Form 4 transaction: %s", e)
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
            logger.info("Processing %s...", ticker)

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
                logger.info("Saved %d transactions to %s", len(df), output_file)

        return df

