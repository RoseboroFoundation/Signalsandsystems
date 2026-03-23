"""Download Republican and Democratic party platforms (2000–2024).

Primary sources: RNC (gop.com) and DNC (democrats.org) official PDFs.
Fallback: American Presidency Project (UCSB) for years no longer hosted
by the national committees.
"""

import io
import os
import time
from typing import Dict, List, Optional

import pandas as pd
import requests
from bs4 import BeautifulSoup

from .config import logger

# ── Source registry ───────────────────────────────────────────────────
# Each entry lists the official party PDF first, then the APP fallback.
# 'source' is tagged so the output tracks provenance.

_SOURCES: Dict[int, Dict[str, list]] = {
    2000: {
        'Republican': [
            # RNC no longer hosts 2000
            {'url': 'https://www.presidency.ucsb.edu/documents/republican-party-platform-of-2000',
             'type': 'html', 'source': 'APP'},
        ],
        'Democratic': [
            {'url': 'https://www.presidency.ucsb.edu/documents/democratic-party-platform-of-2000',
             'type': 'html', 'source': 'APP'},
        ],
    },
    2004: {
        'Republican': [
            {'url': 'https://www.presidency.ucsb.edu/documents/2004-republican-party-platform',
             'type': 'html', 'source': 'APP'},
        ],
        'Democratic': [
            {'url': 'https://www.presidency.ucsb.edu/documents/2004-democratic-party-platform',
             'type': 'html', 'source': 'APP'},
        ],
    },
    2008: {
        'Republican': [
            {'url': 'https://www.presidency.ucsb.edu/documents/2008-republican-party-platform',
             'type': 'html', 'source': 'APP'},
        ],
        'Democratic': [
            {'url': 'https://www.presidency.ucsb.edu/documents/2008-democratic-party-platform',
             'type': 'html', 'source': 'APP'},
        ],
    },
    2012: {
        'Republican': [
            {'url': 'https://prod-cdn-static.gop.com/media/documents/DRAFT_12_FINAL%5B1%5D-ben_1468872234.pdf',
             'type': 'pdf', 'source': 'RNC'},
            {'url': 'https://www.presidency.ucsb.edu/documents/2012-republican-party-platform',
             'type': 'html', 'source': 'APP'},
        ],
        'Democratic': [
            {'url': 'https://www.presidency.ucsb.edu/documents/2012-democratic-party-platform',
             'type': 'html', 'source': 'APP'},
        ],
    },
    2016: {
        'Republican': [
            {'url': 'https://prod-cdn-static.gop.com/static/home/data/platform.pdf',
             'type': 'pdf', 'source': 'RNC'},
            {'url': 'https://www.presidency.ucsb.edu/documents/2016-republican-party-platform',
             'type': 'html', 'source': 'APP'},
        ],
        'Democratic': [
            {'url': 'https://democrats.org/wp-content/uploads/sites/2/2019/07/2016_DNC_Platform.pdf',
             'type': 'pdf', 'source': 'DNC'},
            {'url': 'https://www.presidency.ucsb.edu/documents/2016-democratic-party-platform',
             'type': 'html', 'source': 'APP'},
        ],
    },
    2020: {
        'Republican': [
            # RNC re-adopted 2016 platform; resolution PDF is the official 2020 document
            {'url': 'https://prod-static.gop.com/media/Resolution_Platform.pdf',
             'type': 'pdf', 'source': 'RNC'},
            {'url': 'https://www.presidency.ucsb.edu/documents/2016-republican-party-platform',
             'type': 'html', 'source': 'APP'},
        ],
        'Democratic': [
            {'url': 'https://democrats.org/wp-content/uploads/sites/2/2020/08/2020-Democratic-Party-Platform.pdf',
             'type': 'pdf', 'source': 'DNC'},
            {'url': 'https://www.presidency.ucsb.edu/documents/2020-democratic-party-platform',
             'type': 'html', 'source': 'APP'},
        ],
    },
    2024: {
        'Republican': [
            {'url': 'https://prod-static.gop.com/media/RNC2024-Platform.pdf',
             'type': 'pdf', 'source': 'RNC'},
            {'url': 'https://www.presidency.ucsb.edu/documents/2024-republican-party-platform',
             'type': 'html', 'source': 'APP'},
        ],
        'Democratic': [
            {'url': 'https://democrats.org/wp-content/uploads/2024/09/2024_Democratic_Party_Platform_8a2cf8.pdf',
             'type': 'pdf', 'source': 'DNC'},
            {'url': 'https://www.presidency.ucsb.edu/documents/2024-democratic-party-platform',
             'type': 'html', 'source': 'APP'},
        ],
    },
}


def _try_import_pypdf():
    """Lazy-import pypdf (only needed when a PDF source is hit)."""
    try:
        from pypdf import PdfReader
        return PdfReader
    except ImportError:
        try:
            from PyPDF2 import PdfReader
            return PdfReader
        except ImportError:
            return None


class PartyPlatformDownloader:
    """Download national party platforms — official PDFs first, APP fallback."""

    def __init__(self, output_dir: str = './party_platforms_data'):
        self.output_dir = output_dir
        os.makedirs(output_dir, exist_ok=True)
        self.session = requests.Session()
        self.session.headers.update({
            'User-Agent': 'SignalsAndSystems/1.0 (academic research)',
            'Accept': 'text/html,application/xhtml+xml,application/pdf',
        })

    # ── HTTP helpers ──────────────────────────────────────────────────

    def _request(self, url: str, delay: float = 1.0) -> Optional[requests.Response]:
        """Rate-limited GET with retries for flaky CDNs."""
        for attempt in range(2):
            try:
                resp = self.session.get(url, timeout=30)
                time.sleep(delay)
                if resp.status_code == 200:
                    return resp
                logger.debug(
                    "HTTP %s for %s (attempt %d)", resp.status_code, url, attempt + 1
                )
            except Exception as e:
                logger.debug("Request error for %s: %s (attempt %d)", url, e, attempt + 1)
        return None

    # ── Extractors ────────────────────────────────────────────────────

    def _extract_text_from_pdf(self, content: bytes) -> Optional[str]:
        """Extract text from PDF bytes using pypdf/PyPDF2."""
        PdfReader = _try_import_pypdf()
        if PdfReader is None:
            logger.warning(
                "pypdf not installed — cannot extract PDF text. "
                "Install with: pip install pypdf"
            )
            return None

        try:
            reader = PdfReader(io.BytesIO(content))
            pages = []
            for page in reader.pages:
                text = page.extract_text()
                if text:
                    pages.append(text)
            if not pages:
                return None
            return '\n\n'.join(pages)
        except Exception as e:
            logger.warning("PDF extraction failed: %s", e)
            return None

    def _extract_text_from_app(self, html: str) -> Optional[str]:
        """Extract platform body text from an American Presidency Project page."""
        soup = BeautifulSoup(html, 'html.parser')

        content_div = soup.find('div', class_='field-docs-content')
        if content_div is None:
            content_div = soup.find('div', class_='field--name-field-docs-content')
        if content_div is None:
            content_div = soup.find('article')
        if content_div is None:
            logger.warning("Could not locate platform text in APP page")
            return None

        paragraphs = content_div.find_all(['p', 'h1', 'h2', 'h3', 'h4', 'h5', 'li'])
        if paragraphs:
            text = '\n\n'.join(
                p.get_text(strip=True) for p in paragraphs if p.get_text(strip=True)
            )
        else:
            text = content_div.get_text(separator='\n', strip=True)

        return text if text else None

    # ── Core download logic ───────────────────────────────────────────

    def download_platform(self, year: int, party: str) -> Optional[Dict]:
        """
        Download a single party platform, trying official sources first.

        Parameters
        ----------
        year : int
            Election year (2000, 2004, …, 2024).
        party : str
            'Republican' or 'Democratic'.

        Returns
        -------
        dict with keys: year, party, source, url, text, word_count,
              note, downloaded_at.  None if all sources fail.
        """
        if year not in _SOURCES:
            logger.error("No platform sources defined for %d", year)
            return None

        candidates = _SOURCES[year].get(party)
        if not candidates:
            logger.error("No %s sources for %d", party, year)
            return None

        text = None
        used_source = None
        used_url = None

        for entry in candidates:
            url = entry['url']
            src_type = entry['type']
            src_label = entry['source']

            logger.info(
                "  Trying %d %s platform from %s: %s", year, party, src_label, url
            )

            resp = self._request(url)
            if resp is None:
                logger.info("    %s unavailable, trying next source...", src_label)
                continue

            if src_type == 'pdf':
                text = self._extract_text_from_pdf(resp.content)
                if text is None:
                    logger.info("    PDF extraction failed, trying next source...")
                    continue
                # Save the raw PDF alongside the text
                pdf_path = os.path.join(
                    self.output_dir,
                    f'{party.lower()}_platform_{year}.pdf',
                )
                with open(pdf_path, 'wb') as f:
                    f.write(resp.content)
                logger.info("    Saved PDF → %s", pdf_path)
            else:
                text = self._extract_text_from_app(resp.text)
                if text is None:
                    logger.info("    HTML extraction failed, trying next source...")
                    continue

            used_source = src_label
            used_url = url
            break

        if text is None:
            logger.error("All sources failed for %d %s platform", year, party)
            return None

        note = ''
        if year == 2020 and party == 'Republican':
            note = 'Re-adopted 2016 platform (no new platform issued)'

        # Save text file
        txt_path = os.path.join(
            self.output_dir,
            f'{party.lower()}_platform_{year}.txt',
        )
        with open(txt_path, 'w', encoding='utf-8') as f:
            f.write(text)

        word_count = len(text.split())
        logger.info(
            "    ✓ %d %s: %d words from %s → %s",
            year, party, word_count, used_source, txt_path,
        )

        return {
            'year': year,
            'party': party,
            'source': used_source,
            'url': used_url,
            'text': text,
            'word_count': word_count,
            'note': note,
            'file_path': txt_path,
            'downloaded_at': pd.Timestamp.now().isoformat(),
        }

    # ── Batch download ────────────────────────────────────────────────

    def download_all_platforms(
        self,
        years: List[int] = None,
        parties: List[str] = None,
        save_csv: bool = True,
    ) -> pd.DataFrame:
        """
        Download all party platforms for the specified years.

        Tries official RNC/DNC PDFs first, falls back to the American
        Presidency Project for years no longer hosted by the parties.

        Parameters
        ----------
        years : list of int, optional
            Election years. Default: all available (2000–2024).
        parties : list of str, optional
            Default: ['Republican', 'Democratic'].
        save_csv : bool
            Save metadata CSV and individual text files.

        Returns
        -------
        pd.DataFrame
            Metadata per platform (year, party, source, url, word_count,
            note, file_path).
        """
        if years is None:
            years = sorted(_SOURCES.keys())
        if parties is None:
            parties = ['Republican', 'Democratic']

        results = []
        for year in years:
            for party in parties:
                logger.info("Processing %d %s platform...", year, party)
                record = self.download_platform(year, party)
                if record is not None:
                    results.append(record)

        if not results:
            logger.warning("No platforms downloaded")
            return pd.DataFrame()

        df = pd.DataFrame(results)

        if save_csv:
            meta_cols = [
                'year', 'party', 'source', 'url', 'word_count',
                'note', 'file_path', 'downloaded_at',
            ]
            meta_df = df[[c for c in meta_cols if c in df.columns]]
            csv_path = os.path.join(self.output_dir, 'platform_index.csv')
            meta_df.to_csv(csv_path, index=False)
            logger.info("Saved platform index (%d rows) to %s", len(meta_df), csv_path)

        return df

    # ── Load / corpus helpers ─────────────────────────────────────────

    def load_platform_text(self, year: int, party: str) -> Optional[str]:
        """Load a previously downloaded platform from disk."""
        txt_path = os.path.join(
            self.output_dir,
            f'{party.lower()}_platform_{year}.txt',
        )
        if not os.path.exists(txt_path):
            logger.warning("Platform file not found: %s", txt_path)
            return None
        with open(txt_path, 'r', encoding='utf-8') as f:
            return f.read()

    def build_platform_corpus(self) -> pd.DataFrame:
        """
        Load all downloaded platforms into a single DataFrame
        suitable for NLP / content analysis.

        Returns
        -------
        pd.DataFrame
            Columns: year, party, text, word_count.
        """
        rows = []
        for year in sorted(_SOURCES.keys()):
            for party in ['Republican', 'Democratic']:
                text = self.load_platform_text(year, party)
                if text:
                    rows.append({
                        'year': year,
                        'party': party,
                        'text': text,
                        'word_count': len(text.split()),
                    })
        return pd.DataFrame(rows)
