"""SEC 10-K and 10-Q filing downloader and parser."""

import os
import re
import time
import json
import logging
from datetime import datetime
from typing import List, Dict, Optional

import pandas as pd
import requests
from bs4 import BeautifulSoup

from .config import logger


class SECFilingDownloader:
    """Download and parse SEC 10-K and 10-Q filings via EDGAR."""

    FORM_TYPES_10K = ('10-K', '10-K/A', '10-KSB', '10-KSB/A')
    FORM_TYPES_10Q = ('10-Q', '10-Q/A', '10-QSB', '10-QSB/A')

    def __init__(self, output_dir='./sec_filings_data'):
        self.output_dir = output_dir
        os.makedirs(output_dir, exist_ok=True)
        self.user_agent = os.getenv('SEC_USER_AGENT')
        if not self.user_agent:
            logger.warning(
                "SEC_USER_AGENT not set. SEC requires contact info. "
                "Set SEC_USER_AGENT='Name email' in .env"
            )
            self.user_agent = 'SignalsAndSystems/1.0'

    def _get_headers(self, url: str) -> dict:
        """Get headers with correct Host for SEC endpoints."""
        if 'data.sec.gov' in url:
            host = 'data.sec.gov'
        elif 'efts.sec.gov' in url:
            host = 'efts.sec.gov'
        else:
            host = 'www.sec.gov'
        return {
            'User-Agent': self.user_agent,
            'Accept-Encoding': 'gzip, deflate',
            'Host': host,
        }

    def _request(self, url: str, delay: float = 0.15) -> Optional[requests.Response]:
        """Rate-limited GET with error handling."""
        try:
            resp = requests.get(url, headers=self._get_headers(url), timeout=30)
            time.sleep(delay)
            if resp.status_code != 200:
                logger.debug("HTTP %s for %s", resp.status_code, url)
                return None
            return resp
        except Exception as e:
            logger.warning("Request failed for %s: %s", url, e)
            return None

    # ── CIK lookup ────────────────────────────────────────────────────

    def get_company_cik(self, ticker: str) -> Optional[str]:
        """Get CIK (zero-padded to 10 digits) from ticker."""
        if not hasattr(self, '_cik_mapping'):
            self._cik_mapping = self._load_cik_mapping()
        return self._cik_mapping.get(ticker.upper())

    def _load_cik_mapping(self) -> Dict[str, str]:
        """Load or refresh the ticker→CIK mapping from SEC."""
        cache_file = os.path.join(self.output_dir, 'ticker_cik_mapping.json')

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
        resp = self._request(url)
        if resp is None:
            return {}

        data = resp.json()
        mapping = {}
        for entry in data.values():
            t = entry.get('ticker', '').upper()
            c = str(entry.get('cik_str', '')).zfill(10)
            if t and c:
                mapping[t] = c

        with open(cache_file, 'w') as f:
            json.dump(mapping, f)
        logger.info("Loaded %d ticker-to-CIK mappings", len(mapping))
        return mapping

    # ── Filing index retrieval ────────────────────────────────────────

    def _collect_filings_from_batch(
        self,
        batch: dict,
        target_forms: tuple,
        start_dt: datetime,
        end_dt: datetime,
        ticker: str,
        cik: str,
    ) -> List[Dict]:
        """Extract matching filings from one EDGAR submissions batch."""
        forms = batch.get('form', [])
        dates = batch.get('filingDate', [])
        accessions = batch.get('accessionNumber', [])
        primary_docs = batch.get('primaryDocument', [])
        report_dates = batch.get('reportDate', [])

        filings = []
        for i, form in enumerate(forms):
            if form not in target_forms:
                continue
            if i >= len(dates):
                continue

            filing_dt = datetime.strptime(dates[i], '%Y-%m-%d')
            if not (start_dt <= filing_dt <= end_dt):
                continue

            accession = accessions[i]
            accession_clean = accession.replace('-', '')
            primary_doc = primary_docs[i] if i < len(primary_docs) else ''
            report_date = report_dates[i] if i < len(report_dates) else ''

            doc_url = (
                f"https://www.sec.gov/Archives/edgar/data/"
                f"{int(cik)}/{accession_clean}/{primary_doc}"
            )

            filings.append({
                'ticker': ticker,
                'cik': cik,
                'form_type': form,
                'filing_date': dates[i],
                'report_date': report_date,
                'accession_number': accession,
                'primary_document': primary_doc,
                'filing_url': doc_url,
            })

        return filings

    def download_filing_index(
        self,
        ticker: str,
        cik: str,
        form_types: tuple = None,
        start_date: str = '2000-01-01',
        end_date: str = '2025-12-31',
    ) -> List[Dict]:
        """
        Download filing metadata for a company from EDGAR.

        Parameters
        ----------
        ticker : str
            Stock ticker symbol.
        cik : str
            SEC CIK number.
        form_types : tuple, optional
            Filing types to include. Default: 10-K + 10-Q (including amendments).
        start_date, end_date : str
            Date range (YYYY-MM-DD).

        Returns
        -------
        List[Dict]
            Filing metadata dicts.
        """
        if form_types is None:
            form_types = self.FORM_TYPES_10K + self.FORM_TYPES_10Q

        start_dt = datetime.strptime(start_date, '%Y-%m-%d')
        end_dt = datetime.strptime(end_date, '%Y-%m-%d')

        cik_padded = cik.zfill(10)
        url = f"https://data.sec.gov/submissions/CIK{cik_padded}.json"
        resp = self._request(url)
        if resp is None:
            logger.error("Could not fetch submissions for %s (CIK %s)", ticker, cik)
            return []

        data = resp.json()

        # Recent filings
        recent = data.get('filings', {}).get('recent', {})
        filings = self._collect_filings_from_batch(
            recent, form_types, start_dt, end_dt, ticker, cik
        )

        # Older filing batches
        older_files = data.get('filings', {}).get('files', [])
        for file_info in older_files:
            file_url = f"https://data.sec.gov/submissions/{file_info.get('name', '')}"
            file_resp = self._request(file_url)
            if file_resp is None:
                continue
            filings.extend(
                self._collect_filings_from_batch(
                    file_resp.json(), form_types, start_dt, end_dt, ticker, cik
                )
            )

        logger.info("  %s: %d filings found", ticker, len(filings))
        return filings

    # ── XBRL financial data extraction ────────────────────────────────

    def _get_company_facts(self, cik: str) -> Optional[dict]:
        """Fetch the full XBRL companyfacts JSON from EDGAR."""
        cik_padded = cik.zfill(10)
        url = f"https://data.sec.gov/api/xbrl/companyfacts/CIK{cik_padded}.json"
        resp = self._request(url, delay=0.2)
        if resp is None:
            return None
        return resp.json()

    def extract_financial_metrics(
        self,
        cik: str,
        ticker: str = '',
    ) -> pd.DataFrame:
        """
        Extract key financial metrics from XBRL companyfacts.

        Returns a DataFrame with one row per (ticker, form, period_end)
        and columns for each available metric.

        Metrics pulled (us-gaap taxonomy):
        - Revenue, NetIncome, Assets, Liabilities, StockholdersEquity
        - OperatingIncome, EPS (basic & diluted), CashFromOperations
        - LongTermDebt, TotalDebt, R&D expense
        """
        facts = self._get_company_facts(cik)
        if facts is None:
            logger.warning("No XBRL facts for CIK %s", cik)
            return pd.DataFrame()

        us_gaap = facts.get('facts', {}).get('us-gaap', {})
        if not us_gaap:
            logger.warning("No us-gaap facts for CIK %s", cik)
            return pd.DataFrame()

        # Concepts to extract — (concept_name, output_column)
        concepts = [
            ('Revenues', 'REVENUE'),
            ('RevenueFromContractWithCustomerExcludingAssessedTax', 'REVENUE'),
            ('SalesRevenueNet', 'REVENUE'),
            ('NetIncomeLoss', 'NET_INCOME'),
            ('Assets', 'TOTAL_ASSETS'),
            ('Liabilities', 'TOTAL_LIABILITIES'),
            ('StockholdersEquity', 'STOCKHOLDERS_EQUITY'),
            ('OperatingIncomeLoss', 'OPERATING_INCOME'),
            ('EarningsPerShareBasic', 'EPS_BASIC'),
            ('EarningsPerShareDiluted', 'EPS_DILUTED'),
            ('NetCashProvidedByOperatingActivities', 'CASH_FROM_OPERATIONS'),
            ('LongTermDebt', 'LONG_TERM_DEBT'),
            ('LongTermDebtAndCapitalLeaseObligations', 'LONG_TERM_DEBT'),
            ('ResearchAndDevelopmentExpense', 'RD_EXPENSE'),
            ('CommonStockSharesOutstanding', 'SHARES_OUTSTANDING'),
        ]

        rows = {}  # keyed by (form, end_date)

        for concept_name, col_name in concepts:
            concept = us_gaap.get(concept_name)
            if concept is None:
                continue

            # Prefer USD units; EPS is USD/shares
            units = concept.get('units', {})
            unit_data = units.get('USD') or units.get('USD/shares') or units.get('shares')
            if not unit_data:
                continue

            for entry in unit_data:
                form = entry.get('form', '')
                if form not in ('10-K', '10-Q', '10-K/A', '10-Q/A'):
                    continue

                end_date = entry.get('end', '')
                if not end_date:
                    continue

                key = (form.replace('/A', ''), end_date)
                if key not in rows:
                    rows[key] = {
                        'TICKER': ticker,
                        'CIK': cik,
                        'FORM_TYPE': key[0],
                        'PERIOD_END': end_date,
                        'FILED': entry.get('filed', ''),
                    }

                # First value wins per concept (avoids duplicates)
                if col_name not in rows[key]:
                    rows[key][col_name] = entry.get('val')

        if not rows:
            return pd.DataFrame()

        df = pd.DataFrame(list(rows.values()))
        df['PERIOD_END'] = pd.to_datetime(df['PERIOD_END'])
        df['FILED'] = pd.to_datetime(df['FILED'], errors='coerce')
        df = df.sort_values(['TICKER', 'PERIOD_END']).reset_index(drop=True)
        return df

    # ── Full-text extraction (fallback for non-XBRL filings) ──────────

    def download_filing_text(self, filing_url: str) -> Optional[str]:
        """
        Download the full text of a filing.

        Returns the raw HTML/text content, or None on failure.
        """
        resp = self._request(filing_url, delay=0.2)
        if resp is None:
            return None
        return resp.text

    # ── Filing text section extraction ────────────────────────────────

    # Regex patterns for common 10-K/10-Q section headers.
    # These handle variations like "Item 1A.", "ITEM 1A -", "Item 1A:",
    # as well as HTML anchors and bold/italic wrappers.
    # Regex patterns for locating filing sections.
    # Uses "last match" strategy to skip table-of-contents echoes.
    # Two-tier matching: first try specific pattern (e.g. "Item 7. Management"),
    # then fall back to bare item number (e.g. "ITEM 7.") for layouts that
    # split the label across HTML cells.
    _SECTION_PATTERNS = {
        '10-K': {
            'Item 1A - Risk Factors': (
                [
                    re.compile(
                        r'Item\s+1A[\.\:\s,\-\—\–]+Risk\s+Factors',
                        re.IGNORECASE,
                    ),
                    re.compile(r'Item\s+1A\b', re.IGNORECASE),
                ],
                re.compile(
                    r'Item\s+(?:1B|1C|2)\b',
                    re.IGNORECASE,
                ),
            ),
            'Item 7 - MD&A': (
                [
                    re.compile(
                        r'Item\s+7(?!A)[\.\:\s,\-\—\–]+(?:Management|MD)',
                        re.IGNORECASE,
                    ),
                    re.compile(r'Item\s+7(?!A)\b', re.IGNORECASE),
                ],
                re.compile(r'Item\s+7A\b', re.IGNORECASE),
            ),
        },
        '10-Q': {
            'Item 2 - MD&A': (
                [
                    re.compile(
                        r'Item\s+2(?!A)[\.\:\s,\-\—\–]+(?:Management|MD)',
                        re.IGNORECASE,
                    ),
                    re.compile(r'Item\s+2(?!A)\b', re.IGNORECASE),
                ],
                re.compile(r'Item\s+3\b', re.IGNORECASE),
            ),
        },
    }

    # Map form types (including amendments) to pattern keys.
    _FORM_TO_PATTERN_KEY = {}
    for ft in FORM_TYPES_10K:
        _FORM_TO_PATTERN_KEY[ft] = '10-K'
    for ft in FORM_TYPES_10Q:
        _FORM_TO_PATTERN_KEY[ft] = '10-Q'

    _MAX_SECTION_CHARS = 50_000

    def _extract_filing_sections(
        self, html: str, form_type: str
    ) -> List[Dict]:
        """
        Extract text sections from a filing HTML document.

        Parameters
        ----------
        html : str
            Raw HTML content of the filing.
        form_type : str
            e.g. '10-K', '10-Q', '10-K/A'.

        Returns
        -------
        List[Dict]
            Each dict has keys: ``section`` (str) and ``text`` (str).
        """
        pattern_key = self._FORM_TO_PATTERN_KEY.get(form_type)
        if pattern_key is None:
            return []

        section_defs = self._SECTION_PATTERNS.get(pattern_key, {})
        results = []

        for section_name, (start_patterns, end_re) in section_defs.items():
            # Two-tier matching: try specific patterns first, fall back to
            # bare item number.  Use the *last* match for each pattern
            # because the table of contents often repeats item names.
            if not isinstance(start_patterns, list):
                start_patterns = [start_patterns]

            match_start = None
            for pat in start_patterns:
                starts = list(pat.finditer(html))
                if starts:
                    match_start = starts[-1]
                    break

            if match_start is None:
                continue

            start_pos = match_start.start()

            # Find the end marker after the start position.
            end_match = end_re.search(html, pos=start_pos + len(match_start.group()))
            if end_match:
                end_pos = end_match.start()
            else:
                # Fallback: take up to 500 KB from the start marker
                end_pos = min(start_pos + 500_000, len(html))

            raw_section = html[start_pos:end_pos]

            # Strip HTML tags to get plain text.
            soup = BeautifulSoup(raw_section, 'html.parser')

            # Remove script/style elements.
            for tag in soup.find_all(['script', 'style']):
                tag.decompose()

            text = soup.get_text(separator=' ')

            # Clean up whitespace: collapse runs of whitespace but preserve
            # paragraph breaks (double newlines).
            text = re.sub(r'[ \t]+', ' ', text)
            text = re.sub(r'\n{3,}', '\n\n', text)
            text = text.strip()

            if len(text) < 100:
                # Likely a false positive (just the heading, no content).
                continue

            if len(text) > self._MAX_SECTION_CHARS:
                text = text[: self._MAX_SECTION_CHARS]

            results.append({
                'section': section_name,
                'text': text,
            })

        return results

    def build_filing_text_dataset(
        self,
        tickers: List[str],
        form_types: tuple = None,
        start_date: str = '2000-01-01',
        end_date: str = '2025-12-31',
        max_filings_per_ticker: int = 10,
        save_csv: bool = True,
        checkpoint: bool = True,
    ) -> pd.DataFrame:
        """
        Build a dataset of extracted text sections (Risk Factors, MD&A)
        from 10-K and 10-Q filings for a list of tickers.

        Parameters
        ----------
        tickers : List[str]
            Ticker symbols to process.
        form_types : tuple, optional
            Filing types. Default: 10-K + 10-Q (including amendments).
        start_date, end_date : str
            Date range (YYYY-MM-DD).
        max_filings_per_ticker : int
            Download the most recent N filings per ticker (default 10).
        save_csv : bool
            Write final results to CSV.
        checkpoint : bool
            Save progress after each ticker for resume capability.

        Returns
        -------
        pd.DataFrame
            Columns: TICKER, CIK, FORM_TYPE, FILING_DATE, SECTION, TEXT.
        """
        if form_types is None:
            form_types = self.FORM_TYPES_10K + self.FORM_TYPES_10Q

        checkpoint_file = os.path.join(
            self.output_dir, 'filing_text_checkpoint.csv'
        )
        output_file = os.path.join(
            self.output_dir,
            f'sec_filing_text_{start_date}_to_{end_date}.csv',
        )

        # Load checkpoint if available.
        rows_so_far: List[Dict] = []
        processed_tickers: set = set()

        if checkpoint and os.path.exists(checkpoint_file):
            try:
                ckpt_df = pd.read_csv(checkpoint_file)
                rows_so_far = ckpt_df.to_dict('records')
                processed_tickers = set(ckpt_df['TICKER'].unique())
                logger.info(
                    "Loaded checkpoint: %d rows, %d tickers already done",
                    len(rows_so_far),
                    len(processed_tickers),
                )
            except Exception as e:
                logger.warning("Could not load checkpoint: %s", e)

        failed = []

        for i, ticker in enumerate(tickers, 1):
            if ticker in processed_tickers:
                logger.info(
                    "[%d/%d] %s already in checkpoint — skipping",
                    i, len(tickers), ticker,
                )
                continue

            logger.info("[%d/%d] Processing %s...", i, len(tickers), ticker)

            cik = self.get_company_cik(ticker)
            if not cik:
                logger.warning("  No CIK found for %s — skipping", ticker)
                failed.append(ticker)
                continue

            # Get filing index (metadata).
            filings = self.download_filing_index(
                ticker, cik, form_types, start_date, end_date
            )
            if not filings:
                logger.info("  %s: no filings found", ticker)
                continue

            # Sort by filing date descending, take most recent N.
            filings.sort(key=lambda f: f['filing_date'], reverse=True)
            filings = filings[:max_filings_per_ticker]

            ticker_sections = 0
            for filing in filings:
                html = self.download_filing_text(filing['filing_url'])
                if html is None:
                    logger.debug(
                        "  Could not download %s", filing['filing_url']
                    )
                    continue

                sections = self._extract_filing_sections(
                    html, filing['form_type']
                )
                for sec in sections:
                    rows_so_far.append({
                        'TICKER': ticker,
                        'CIK': cik,
                        'FORM_TYPE': filing['form_type'],
                        'FILING_DATE': filing['filing_date'],
                        'SECTION': sec['section'],
                        'TEXT': sec['text'],
                    })
                    ticker_sections += 1

            logger.info(
                "  %s: %d filings, %d sections extracted",
                ticker, len(filings), ticker_sections,
            )

            # Write checkpoint after each ticker.
            if checkpoint and rows_so_far:
                pd.DataFrame(rows_so_far).to_csv(
                    checkpoint_file, index=False
                )

        if failed:
            logger.warning(
                "%d tickers failed CIK lookup: %s",
                len(failed),
                ', '.join(failed[:20]),
            )

        if not rows_so_far:
            logger.warning("No text sections extracted for any ticker")
            return pd.DataFrame()

        result = pd.DataFrame(rows_so_far)
        result['FILING_DATE'] = pd.to_datetime(
            result['FILING_DATE'], errors='coerce'
        )
        result = result.sort_values(
            ['TICKER', 'FILING_DATE']
        ).reset_index(drop=True)

        if save_csv:
            result.to_csv(output_file, index=False)
            logger.info(
                "Saved %d rows (%d tickers) to %s",
                len(result),
                result['TICKER'].nunique(),
                output_file,
            )

        # Clean up checkpoint after successful completion.
        if checkpoint and os.path.exists(checkpoint_file):
            os.remove(checkpoint_file)
            logger.info("Removed checkpoint file")

        return result

    # ── Batch pipeline ────────────────────────────────────────────────

    def build_fundamentals_dataset(
        self,
        tickers: List[str],
        start_date: str = '2000-01-01',
        end_date: str = '2025-12-31',
        save_csv: bool = True,
    ) -> pd.DataFrame:
        """
        Build a panel dataset of financial fundamentals from 10-K/10-Q
        filings for a list of tickers.

        Uses the XBRL companyfacts API for structured data extraction —
        no HTML parsing needed for most companies.

        Parameters
        ----------
        tickers : List[str]
            Ticker symbols to process.
        start_date, end_date : str
            Date range for filtering filings.
        save_csv : bool
            If True, write results to CSV.

        Returns
        -------
        pd.DataFrame
            Panel with columns: TICKER, CIK, FORM_TYPE, PERIOD_END,
            FILED, REVENUE, NET_INCOME, TOTAL_ASSETS, etc.
        """
        all_dfs = []
        start_dt = pd.Timestamp(start_date)
        end_dt = pd.Timestamp(end_date)
        failed = []

        for i, ticker in enumerate(tickers, 1):
            logger.info("[%d/%d] Processing %s...", i, len(tickers), ticker)

            cik = self.get_company_cik(ticker)
            if not cik:
                logger.warning("  No CIK found for %s — skipping", ticker)
                failed.append(ticker)
                continue

            df = self.extract_financial_metrics(cik, ticker)
            if df.empty:
                logger.warning("  No XBRL data for %s — skipping", ticker)
                failed.append(ticker)
                continue

            # Filter to date range
            df = df[(df['PERIOD_END'] >= start_dt) & (df['PERIOD_END'] <= end_dt)]
            if df.empty:
                logger.info("  %s: no filings in date range", ticker)
                continue

            all_dfs.append(df)
            logger.info("  %s: %d periods extracted", ticker, len(df))

        if failed:
            logger.warning(
                "%d tickers failed: %s", len(failed), ', '.join(failed[:20])
            )

        if not all_dfs:
            logger.warning("No data extracted for any ticker")
            return pd.DataFrame()

        result = pd.concat(all_dfs, ignore_index=True)
        result = result.sort_values(['TICKER', 'PERIOD_END']).reset_index(drop=True)

        if save_csv:
            output_file = os.path.join(
                self.output_dir,
                f'sec_fundamentals_{start_date}_to_{end_date}.csv',
            )
            result.to_csv(output_file, index=False)
            logger.info(
                "Saved %d rows (%d tickers) to %s",
                len(result), result['TICKER'].nunique(), output_file,
            )

        return result

    def build_filing_index(
        self,
        tickers: List[str],
        form_types: tuple = None,
        start_date: str = '2000-01-01',
        end_date: str = '2025-12-31',
        save_csv: bool = True,
    ) -> pd.DataFrame:
        """
        Build a filing index (metadata only, no financials) for a list
        of tickers. Useful for tracking filing dates, accession numbers,
        and URLs.

        Parameters
        ----------
        tickers : List[str]
            Ticker symbols.
        form_types : tuple, optional
            Filing types. Default: 10-K + 10-Q.
        start_date, end_date : str
            Date range.
        save_csv : bool
            Write to CSV.

        Returns
        -------
        pd.DataFrame
            Filing metadata.
        """
        if form_types is None:
            form_types = self.FORM_TYPES_10K + self.FORM_TYPES_10Q

        all_filings = []
        for i, ticker in enumerate(tickers, 1):
            logger.info("[%d/%d] Indexing %s...", i, len(tickers), ticker)

            cik = self.get_company_cik(ticker)
            if not cik:
                logger.warning("  No CIK for %s", ticker)
                continue

            filings = self.download_filing_index(
                ticker, cik, form_types, start_date, end_date
            )
            all_filings.extend(filings)

        if not all_filings:
            return pd.DataFrame()

        df = pd.DataFrame(all_filings)
        df['filing_date'] = pd.to_datetime(df['filing_date'])
        df['report_date'] = pd.to_datetime(df['report_date'], errors='coerce')
        df = df.sort_values(['ticker', 'filing_date']).reset_index(drop=True)

        if save_csv:
            output_file = os.path.join(
                self.output_dir,
                f'sec_filing_index_{start_date}_to_{end_date}.csv',
            )
            df.to_csv(output_file, index=False)
            logger.info("Saved %d filings to %s", len(df), output_file)

        return df
