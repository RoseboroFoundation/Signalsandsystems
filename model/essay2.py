"""
Essay 2 — NLP Pipeline for Culture War Event Analysis.

Scores news articles, 10-K, and 10-Q filings with FinBERT to produce
event-window sentiment measures for the cross-sectional DiD in
``essay2_did.py``.

Three text sources are processed per company:
  1. News articles — Guardian, NYT, Reddit (pre-scraped in news_data/)
  2. 10-K annual reports — MD&A (Item 7) and Risk Factors (Item 1A)
  3. 10-Q quarterly reports — MD&A (Item 2) and Risk Factors (Item 1A)

Filing text is downloaded from SEC EDGAR, parsed into sections, chunked
to fit FinBERT's 512-token context, and scored.  The output is a unified
NLP panel keyed by (TICKER, EVENT_DATE, SOURCE, WINDOW) that feeds
directly into the DiD panel construction.

Also provides backward-compatible ``FactorModelResult`` and
``factor_model()`` used by the Streamlit dashboard.

References
----------
Loughran, T. & McDonald, B. (2011). When is a liability not a liability?
    Textual analysis, dictionaries, and 10-Ks. Journal of Finance, 66(1).
Huang, A.H., Wang, H. & Yang, Y. (2023). FinBERT: A large language model
    for extracting information from financial text. Contemporary Accounting
    Research, 40(2).
"""

import logging
import re
import threading
import time
import warnings
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import statsmodels.api as sm

from .datastore import DataStore
from .essay1 import (
    RegimeResult,
    estimate_vix_regimes,
    SentimentRegimeAnalysis,
    _score_finbert,
    _sentiment_to_numeric,
    _FINBERT_BATCH_SIZE,
    _FF5_ALL,
    benjamini_hochberg,
)

logger = logging.getLogger(__name__)

# ── FinBERT Lazy Singleton ───────────────────────────────────────────
# Ensures the ~440MB model is loaded at most once per process, even when
# callers pass finbert_pipeline=None to multiple scoring functions.
# Thread-safe: uses a lock to prevent double-loading under Streamlit's
# multi-threaded executor.
#
# ImportError propagation: if transformers/torch are not installed, the
# ImportError propagates to the caller.  Callers that wrap scoring in
# try/except ImportError (e.g. run_nlp_analysis) will still catch it
# because _get_finbert_pipeline is called *inside* the try block.

_finbert_singleton = None
_finbert_lock = threading.Lock()


def _get_finbert_pipeline(pipe=None):
    """Return the provided pipeline or lazily load a shared singleton.

    Raises ImportError if transformers/torch are not installed.
    """
    global _finbert_singleton
    if pipe is not None:
        return pipe
    with _finbert_lock:
        if _finbert_singleton is None:
            from transformers import pipeline as hf_pipeline
            logger.info("Loading FinBERT pipeline (singleton, ~440MB)...")
            _finbert_singleton = hf_pipeline(
                'sentiment-analysis',
                model='ProsusAI/finbert',
                tokenizer='ProsusAI/finbert',
            )
    return _finbert_singleton


# ── NLP Configuration ─────────────────────────────────────────────────

# Filing section extraction patterns (case-insensitive).
# Each tuple: (section_name, start_pattern, stop_patterns)
# stop_patterns = list of Item headers that signal section end.
_10K_SECTIONS = [
    (
        'RISK_FACTORS',
        r'item\s+1a[\.\s\u2014\u2013\-]+risk\s+factors',
        [r'item\s+1b', r'item\s+2'],
    ),
    (
        'MDA',
        r'item\s+7[\.\s\u2014\u2013\-]+management.{0,5}s?\s+discussion',
        [r'item\s+7a', r'item\s+8'],
    ),
]

_10Q_SECTIONS = [
    (
        'MDA',
        r'item\s+2[\.\s\u2014\u2013\-]+management.{0,5}s?\s+discussion',
        [r'item\s+3', r'item\s+4'],
    ),
    (
        'RISK_FACTORS',
        r'item\s+1a[\.\s\u2014\u2013\-]+risk\s+factors',
        [r'item\s+2', r'item\s+1b'],
    ),
]

# Maximum characters per filing section before chunking.
# FinBERT context = 512 tokens ≈ 2000 chars.  We chunk at this boundary.
_CHUNK_MAX_CHARS = 1800

# Minimum chars for a valid filing section (below this → likely TOC reference).
# A genuine MD&A is typically 5,000-50,000 chars; 2,000 chars ≈ 300 words.
_MIN_SECTION_CHARS = 2000

# Event window (trading days) for matching news/filings to events
_NEWS_WINDOW_DAYS = 30       # calendar days around event for news
_FILING_WINDOW_DAYS = 180    # calendar days before event for filings


# ── Data Classes ──────────────────────────────────────────────────────

# NOTE: FilingSentiment and EventNLPResult are part of the public API
# (exported in __init__.py) for typed consumers.  Internal pipeline code
# uses dicts/DataFrames for flexibility, but external callers may
# construct these for type-safe interop.
@dataclass
class FilingSentiment:
    """FinBERT sentiment for a single SEC filing section."""
    ticker: str
    form_type: str              # '10-K' or '10-Q'
    filing_date: pd.Timestamp
    section: str                # 'MDA' or 'RISK_FACTORS'
    section_header: str         # original Item header text (excluded from scoring)
    n_chunks: int               # number of text chunks scored
    sent_mean: float            # confidence-weighted mean sentiment across chunks
    sent_std: float             # std of chunk sentiments
    pct_positive: float
    pct_negative: float
    pct_neutral: float
    text_length: int            # chars in extracted section


@dataclass
class EventNLPResult:
    """Combined NLP results for a single ticker across all sources."""
    ticker: str
    news_pre: float = np.nan    # mean news sentiment in pre-event window
    news_post: float = np.nan   # mean news sentiment in post-event window
    news_n_pre: int = 0
    news_n_post: int = 0
    filing_mda_tone: float = np.nan     # most recent MDA sentiment
    filing_risk_tone: float = np.nan    # most recent Risk Factors sentiment
    filing_form_type: str = ''
    filing_date: Optional[pd.Timestamp] = None


@dataclass
class NLPAnalysis:
    """Complete NLP analysis across all companies and sources."""
    news_sentiment: pd.DataFrame        # per-article FinBERT scores
    filing_sentiment: pd.DataFrame      # per-filing-section scores
    event_nlp_panel: pd.DataFrame       # (TICKER, EVENT_DATE) NLP features for DiD
    n_articles_scored: int
    n_filings_scored: int
    n_tickers: int


# ── Filing Text Extraction ────────────────────────────────────────────

def _clean_html_to_text(html: str) -> str:
    """Strip HTML tags and normalize whitespace."""
    try:
        from bs4 import BeautifulSoup
        soup = BeautifulSoup(html, 'lxml')
        # Remove script/style elements
        for tag in soup(['script', 'style', 'head']):
            tag.decompose()
        text = soup.get_text(separator=' ')
    except Exception as e:
        # Fallback: regex tag removal (lower quality — misses JS/CSS/metadata)
        logger.warning("BeautifulSoup failed (%s), using regex fallback — "
                       "filing text quality may be degraded", e)
        text = re.sub(r'<[^>]+>', ' ', html)

    # Normalize whitespace
    text = re.sub(r'\s+', ' ', text).strip()
    return text


def _extract_section(text: str, start_pattern: str, stop_patterns: List[str]) -> Tuple[str, str]:
    """
    Extract a section from filing plain text using regex boundaries.

    Finds the start pattern, then reads until the first stop pattern
    or end of text.  Returns (section_body, section_header) tuple.
    The header (e.g. "ITEM 7. MANAGEMENT'S DISCUSSION...") is excluded
    from the body to avoid skewing FinBERT scores with boilerplate
    regulatory language.  Returns ('', '') if section not found.
    """
    # Find section start
    start_match = re.search(start_pattern, text, re.IGNORECASE)
    if start_match is None:
        return ('', '')

    # Start body after the header to avoid FinBERT scoring boilerplate
    section_header = text[start_match.start():start_match.end()].strip()
    section_body_start = start_match.end()

    # Find section end (next Item header)
    section_end = len(text)
    for stop_pat in stop_patterns:
        stop_match = re.search(stop_pat, text[start_match.end():], re.IGNORECASE)
        if stop_match is not None:
            candidate = start_match.end() + stop_match.start()
            section_end = min(section_end, candidate)

    section = text[section_body_start:section_end].strip()

    # Skip if too short (likely a table of contents reference, not the actual section).
    # Dual-TOC filings can cause stop_pattern matches close to section_body_start,
    # producing near-empty sections — log the header for post-hoc diagnosis.
    if len(section) < _MIN_SECTION_CHARS:
        logger.debug("Section '%s' too short (%d chars, header='%s'), "
                     "likely TOC reference or dual-TOC collision",
                     start_pattern[:30], len(section), section_header)
        return ('', '')

    return (section, section_header)


def _chunk_text(text: str, max_chars: int = _CHUNK_MAX_CHARS) -> List[str]:
    """
    Split text into chunks that fit FinBERT's context window.

    Splits on sentence boundaries when possible to preserve meaning.
    """
    if len(text) <= max_chars:
        return [text]

    chunks = []
    # Split on sentence endings
    sentences = re.split(r'(?<=[.!?])\s+', text)
    current = ''

    for sent in sentences:
        if len(current) + len(sent) + 1 > max_chars:
            if current:
                chunks.append(current.strip())
            # Handle sentences longer than max_chars
            if len(sent) > max_chars:
                for i in range(0, len(sent), max_chars):
                    chunks.append(sent[i:i + max_chars])
                current = ''
            else:
                current = sent
        else:
            current = current + ' ' + sent if current else sent

    if current.strip():
        chunks.append(current.strip())

    # Drop very short trailing chunks (<100 chars) that would have outsized
    # influence on SENT_STD.  Log the drop so we can audit chunk distributions.
    _MIN_CHUNK_CHARS = 100
    filtered = [c for c in chunks if len(c) >= _MIN_CHUNK_CHARS]
    n_dropped = len(chunks) - len(filtered)
    if n_dropped > 0:
        logger.debug("Dropped %d short chunk(s) (<%d chars) from %d total",
                     n_dropped, _MIN_CHUNK_CHARS, len(chunks))
    return filtered if filtered else chunks  # keep at least one chunk


def download_and_parse_filings(
    tickers: List[str],
    start_date: str = '2000-01-01',
    end_date: str = '2025-12-31',
    output_dir: str = './sec_filings_data',
) -> pd.DataFrame:
    """
    Download 10-K/10-Q filings from EDGAR and extract text sections.

    For each filing, extracts MD&A and Risk Factors sections.
    Returns a DataFrame with columns:
        TICKER, FORM_TYPE, FILING_DATE, SECTION, TEXT, TEXT_LENGTH

    Parameters
    ----------
    tickers : list of str
    start_date, end_date : str
    output_dir : str
        Directory for the SEC filing downloader cache.

    Returns
    -------
    pd.DataFrame
    """
    from clean.sec_filings import SECFilingDownloader

    downloader = SECFilingDownloader(output_dir=output_dir)
    rows = []

    for i, ticker in enumerate(tickers, 1):
        logger.info("[%d/%d] Downloading filings for %s", i, len(tickers), ticker)

        cik = downloader.get_company_cik(ticker)
        if not cik:
            logger.warning("  No CIK for %s, skipping", ticker)
            continue

        # Get filing index
        filings = downloader.download_filing_index(
            ticker, cik, start_date=start_date, end_date=end_date,
        )

        for filing in filings:
            form_type = filing['form_type']
            filing_date = filing['filing_date']
            filing_url = filing['filing_url']

            # Select section patterns based on form type
            if form_type in ('10-K', '10-K/A', '10-KSB', '10-KSB/A'):
                section_defs = _10K_SECTIONS
            elif form_type in ('10-Q', '10-Q/A', '10-QSB', '10-QSB/A'):
                section_defs = _10Q_SECTIONS
            else:
                continue

            # Download filing text
            html = downloader.download_filing_text(filing_url)
            if html is None:
                logger.debug("  %s %s: download failed", ticker, filing_date)
                continue

            plain_text = _clean_html_to_text(html)
            if len(plain_text) < 1000:
                logger.debug("  %s %s: text too short (%d chars)",
                             ticker, filing_date, len(plain_text))
                continue

            # Extract each section
            for section_name, start_pat, stop_pats in section_defs:
                section_text, section_header = _extract_section(plain_text, start_pat, stop_pats)
                if not section_text:
                    continue

                rows.append({
                    'TICKER': ticker,
                    'FORM_TYPE': form_type.replace('/A', ''),
                    'FILING_DATE': filing_date,
                    'SECTION': section_name,
                    'TEXT': section_text,
                    'SECTION_HEADER': section_header,
                    'TEXT_LENGTH': len(section_text),
                })

            logger.debug("  %s %s: extracted %d sections",
                         ticker, filing_date,
                         sum(1 for r in rows if r['TICKER'] == ticker
                             and r['FILING_DATE'] == filing_date))

        # Progress checkpoint
        if i % 10 == 0:
            logger.info("  Progress: %d/%d tickers, %d sections extracted",
                         i, len(tickers), len(rows))

    if not rows:
        logger.warning("No filing sections extracted from %d tickers", len(tickers))
        return pd.DataFrame()

    df = pd.DataFrame(rows)
    df['FILING_DATE'] = pd.to_datetime(df['FILING_DATE'], errors='coerce')
    df = df.sort_values(['TICKER', 'FILING_DATE', 'SECTION']).reset_index(drop=True)

    logger.info("Filing extraction complete: %d sections from %d tickers, "
                "%d 10-K, %d 10-Q",
                len(df), df['TICKER'].nunique(),
                (df['FORM_TYPE'] == '10-K').sum(),
                (df['FORM_TYPE'] == '10-Q').sum())

    return df


# ── FinBERT Scoring ───────────────────────────────────────────────────

def score_news_sentiment(
    news_df: pd.DataFrame,
    finbert_pipeline=None,
) -> pd.DataFrame:
    """
    Score news articles with FinBERT.

    Expects columns: TITLE, SNIPPET (or TEXT), TICKER, DATE.
    Adds columns: FINBERT_LABEL, FINBERT_CONF, SENTIMENT, SENT_WEIGHTED.

    Parameters
    ----------
    news_df : pd.DataFrame
    finbert_pipeline : optional
        Pre-loaded FinBERT pipeline to avoid reloading.

    Returns
    -------
    pd.DataFrame with sentiment columns added.
    """
    df = news_df.copy()

    # Build text field
    if 'TEXT' not in df.columns:
        title = df.get('TITLE', pd.Series(dtype=str)).fillna('')
        snippet = df.get('SNIPPET', pd.Series(dtype=str)).fillna('')
        df['TEXT'] = (title + ' ' + snippet).str.strip()

    df = df[df['TEXT'].str.len() > 10].reset_index(drop=True)

    if df.empty:
        logger.warning("No news articles with sufficient text to score")
        return df

    logger.info("Scoring %d news articles with FinBERT...", len(df))
    pipe = _get_finbert_pipeline(finbert_pipeline)
    scores = _score_finbert(df['TEXT'].tolist(), pipe=pipe)

    df['FINBERT_LABEL'] = [s['label'] for s in scores]
    df['FINBERT_CONF'] = [s['score'] for s in scores]
    df['SENTIMENT'] = df['FINBERT_LABEL'].apply(_sentiment_to_numeric)
    df['SENT_WEIGHTED'] = df['SENTIMENT'] * df['FINBERT_CONF']

    # Drop failed scores (None from _score_finbert error handling)
    df = df.dropna(subset=['FINBERT_LABEL', 'SENTIMENT'])

    logger.info("News sentiment: %d scored — %d positive, %d negative, %d neutral",
                len(df),
                (df['FINBERT_LABEL'] == 'positive').sum(),
                (df['FINBERT_LABEL'] == 'negative').sum(),
                (df['FINBERT_LABEL'] == 'neutral').sum())

    return df


def score_filing_sentiment(
    filings_df: pd.DataFrame,
    finbert_pipeline=None,
) -> pd.DataFrame:
    """
    Score SEC filing sections with FinBERT.

    Long sections are chunked to fit FinBERT's 512-token context.
    Chunk-level scores are aggregated to section-level means.

    Parameters
    ----------
    filings_df : pd.DataFrame
        Must have columns: TICKER, FORM_TYPE, FILING_DATE, SECTION, TEXT.
    finbert_pipeline : optional
        Pre-loaded FinBERT pipeline.

    Returns
    -------
    pd.DataFrame with columns: TICKER, FORM_TYPE, FILING_DATE, SECTION,
        N_CHUNKS, SENT_CONF_WEIGHTED_MEAN, SENT_STD, PCT_POSITIVE,
        PCT_NEGATIVE, PCT_NEUTRAL, TEXT_LENGTH.

    Notes
    -----
    SENT_CONF_WEIGHTED_MEAN is the mean of (sentiment_direction * confidence)
    across chunks, where sentiment_direction is +1/0/-1 from the FinBERT
    label.  This downweights low-confidence predictions but differs from
    the unweighted sentiment means in Loughran & McDonald (2011).  The DiD
    coefficient on filing tone should be interpreted as the effect of
    confidence-weighted filing sentiment, not raw directional tone.
    """
    if filings_df.empty:
        return pd.DataFrame()

    # Reset index to ensure chunk_index mapping is positionally consistent,
    # even if filings_df was passed with a non-default index (e.g. after merge/groupby).
    filings_df = filings_df.reset_index(drop=True)

    results = []
    all_chunks = []
    chunk_index = []  # maps chunk -> (row_idx)

    # Build all chunks first for efficient batched scoring
    for idx, row in filings_df.iterrows():
        chunks = _chunk_text(row['TEXT'])
        for chunk in chunks:
            all_chunks.append(chunk)
            chunk_index.append(idx)

    if not all_chunks:
        return pd.DataFrame()

    logger.info("Scoring %d filing chunks (%d sections) with FinBERT...",
                len(all_chunks), len(filings_df))

    pipe = _get_finbert_pipeline(finbert_pipeline)
    scores = _score_finbert(all_chunks, pipe=pipe)

    # Map scores back to filing sections
    chunk_sentiments = {}  # idx -> list of weighted sentiments
    chunk_labels = {}      # idx -> list of labels

    for i, (score, idx) in enumerate(zip(scores, chunk_index)):
        if score['label'] is None:
            continue
        sent = _sentiment_to_numeric(score['label'])
        weighted = sent * (score['score'] or 0.0)

        if idx not in chunk_sentiments:
            chunk_sentiments[idx] = []
            chunk_labels[idx] = []
        chunk_sentiments[idx].append(weighted)
        chunk_labels[idx].append(score['label'])

    # Aggregate to section level
    for idx, row in filings_df.iterrows():
        sentiments = chunk_sentiments.get(idx, [])
        labels = chunk_labels.get(idx, [])

        if not sentiments:
            continue

        n = len(sentiments)
        # Flag single-chunk sections — sentiment is used but has no variance measure
        if n == 1:
            logger.debug("Single-chunk section %s/%s/%s — SENT_STD will be 0.0, "
                         "mean may be unreliable",
                         row['TICKER'], row['SECTION'], row.get('FILING_DATE', ''))
        results.append({
            'TICKER': row['TICKER'],
            'FORM_TYPE': row['FORM_TYPE'],
            'FILING_DATE': row['FILING_DATE'],
            'SECTION': row['SECTION'],
            'N_CHUNKS': n,
            'SENT_CONF_WEIGHTED_MEAN': np.mean(sentiments),
            'SENT_STD': np.std(sentiments) if n > 1 else 0.0,
            'PCT_POSITIVE': sum(1 for l in labels if l == 'positive') / n,
            'PCT_NEGATIVE': sum(1 for l in labels if l == 'negative') / n,
            'PCT_NEUTRAL': sum(1 for l in labels if l == 'neutral') / n,
            'TEXT_LENGTH': row.get('TEXT_LENGTH', len(row.get('TEXT', ''))),
        })

    df = pd.DataFrame(results)
    if not df.empty:
        df['FILING_DATE'] = pd.to_datetime(df['FILING_DATE'], errors='coerce')

    logger.info("Filing sentiment: %d sections scored across %d tickers",
                len(df), df['TICKER'].nunique() if not df.empty else 0)

    return df


# ── Event-Window NLP Panel ────────────────────────────────────────────

def build_event_nlp_panel(
    store: DataStore,
    news_scored: pd.DataFrame = None,
    filing_scored: pd.DataFrame = None,
    news_window_days: int = _NEWS_WINDOW_DAYS,
    filing_window_days: int = _FILING_WINDOW_DAYS,
) -> pd.DataFrame:
    """
    Build the (TICKER, EVENT_DATE) NLP feature panel for DiD.

    For each culture war event, computes:
      - Pre-event and post-event news sentiment (mean, count)
      - Most recent 10-K/10-Q MD&A and Risk Factor tone
      - Filing tone change (current vs prior filing)

    Parameters
    ----------
    store : DataStore
    news_scored : pd.DataFrame, optional
        Pre-scored news (from score_news_sentiment). If None, skips news.
    filing_scored : pd.DataFrame, optional
        Pre-scored filings (from score_filing_sentiment). If None, skips filings.
    news_window_days : int
        Calendar days around event for news matching.
    filing_window_days : int
        Calendar days before event for filing matching.

    Returns
    -------
    pd.DataFrame with columns: TICKER, EVENT_DATE, EVENT_ID,
        NEWS_SENT_PRE, NEWS_SENT_POST, NEWS_N_PRE, NEWS_N_POST,
        NEWS_SENT_CHANGE, FILING_MDA_TONE, FILING_RISK_TONE,
        FILING_MDA_CHANGE, FILING_RISK_CHANGE, FILING_FORM_TYPE,
        FILING_DATE.
    """
    events_df = store.read_table('CULTURE_WAR_COMPANIES')
    if events_df.empty:
        # Fall back to events attribute — warn because it may have different schema
        if hasattr(store, 'events') and not store.events.empty:
            store_cols = {c.upper() for c in store.events.columns}
            has_ticker = 'TICKER' in store_cols
            has_date = bool({'EVENT_DATE', 'DATE'} & store_cols)
            if not has_ticker or not has_date:
                logger.error(
                    "store.events fallback lacks required columns "
                    "(need TICKER + EVENT_DATE/DATE, has: %s) — "
                    "cannot build NLP panel", store_cols)
                return pd.DataFrame()
            logger.warning(
                "CULTURE_WAR_COMPANIES table empty — falling back to "
                "store.events (%d rows). Verify schema compatibility.",
                len(store.events))
            events_df = store.events.copy()
        else:
            events_df = pd.DataFrame()
    if events_df.empty:
        logger.error("No events found for NLP panel construction")
        return pd.DataFrame()

    events_df.columns = [c.upper() for c in events_df.columns]
    date_col = 'EVENT_DATE' if 'EVENT_DATE' in events_df.columns else 'DATE'
    events_df[date_col] = pd.to_datetime(events_df[date_col], errors='coerce')
    events_df = events_df.dropna(subset=[date_col])

    rows = []

    for _, event in events_df.iterrows():
        ticker = event.get('TICKER', None)
        event_date = event[date_col]
        event_id = event.get('EVENT_ID', f"{ticker}_{event_date.date()}")

        if ticker is None:
            continue

        row = {
            'TICKER': ticker,
            'EVENT_DATE': event_date,
            'EVENT_ID': event_id,
        }

        # ── News sentiment around event ──
        if news_scored is not None and not news_scored.empty:
            news_dates = pd.to_datetime(news_scored.get('DATE', pd.Series(dtype='datetime64[ns]')))
            ticker_news = news_scored[
                (news_scored['TICKER'] == ticker) &
                news_dates.notna()
            ].copy()

            if not ticker_news.empty:
                ticker_news['_DATE'] = pd.to_datetime(ticker_news['DATE'])
                pre_mask = (
                    (ticker_news['_DATE'] >= event_date - pd.Timedelta(days=news_window_days)) &
                    (ticker_news['_DATE'] < event_date)
                )
                post_mask = (
                    (ticker_news['_DATE'] >= event_date) &
                    (ticker_news['_DATE'] <= event_date + pd.Timedelta(days=news_window_days))
                )

                pre_news = ticker_news[pre_mask]
                post_news = ticker_news[post_mask]

                sent_col = 'SENT_WEIGHTED' if 'SENT_WEIGHTED' in ticker_news.columns else None
                row['NEWS_SENT_PRE'] = pre_news[sent_col].mean() if sent_col and len(pre_news) > 0 else np.nan
                row['NEWS_SENT_POST'] = post_news[sent_col].mean() if sent_col and len(post_news) > 0 else np.nan
                row['NEWS_N_PRE'] = len(pre_news)
                row['NEWS_N_POST'] = len(post_news)
                _post = row.get('NEWS_SENT_POST', np.nan)
                _pre = row.get('NEWS_SENT_PRE', np.nan)
                row['NEWS_SENT_CHANGE'] = (_post - _pre) if (
                    not np.isnan(_post) and not np.isnan(_pre)) else np.nan

        # ── Filing sentiment (most recent before event) ──
        if filing_scored is not None and not filing_scored.empty:
            # Primary: filings within the window
            ticker_filings = filing_scored[
                (filing_scored['TICKER'] == ticker) &
                (filing_scored['FILING_DATE'] < event_date) &
                (filing_scored['FILING_DATE'] >= event_date - pd.Timedelta(days=filing_window_days))
            ].sort_values('FILING_DATE', ascending=False)

            # Fallback: most recent filing regardless of window
            if ticker_filings.empty:
                ticker_filings = filing_scored[
                    (filing_scored['TICKER'] == ticker) &
                    (filing_scored['FILING_DATE'] < event_date)
                ].sort_values('FILING_DATE', ascending=False).head(4)
                if not ticker_filings.empty:
                    logger.debug("%s: filing fallback used (most recent = %s)",
                                 ticker, ticker_filings.iloc[0]['FILING_DATE'])

            has_sent = 'SENT_CONF_WEIGHTED_MEAN' in ticker_filings.columns

            # Most recent MDA
            mda_sections = ['MDA', 'Item 7 - MD&A', 'Item 2 - MD&A']
            mda = ticker_filings[ticker_filings['SECTION'].isin(mda_sections)]
            if not mda.empty:
                latest_mda = mda.iloc[0]
                row['FILING_MDA_TONE'] = latest_mda['SENT_CONF_WEIGHTED_MEAN'] if has_sent else np.nan
                row['FILING_FORM_TYPE'] = latest_mda['FORM_TYPE']
                row['FILING_DATE'] = latest_mda['FILING_DATE']

                if has_sent and len(mda) >= 2:
                    row['FILING_MDA_CHANGE'] = mda.iloc[0]['SENT_CONF_WEIGHTED_MEAN'] - mda.iloc[1]['SENT_CONF_WEIGHTED_MEAN']

            # Most recent Risk Factors
            risk_sections = ['RISK_FACTORS', 'Item 1A - Risk Factors']
            risk = ticker_filings[ticker_filings['SECTION'].isin(risk_sections)]
            if not risk.empty:
                row['FILING_RISK_TONE'] = risk.iloc[0]['SENT_CONF_WEIGHTED_MEAN'] if has_sent else np.nan

                if has_sent and len(risk) >= 2:
                    row['FILING_RISK_CHANGE'] = risk.iloc[0]['SENT_CONF_WEIGHTED_MEAN'] - risk.iloc[1]['SENT_CONF_WEIGHTED_MEAN']

        rows.append(row)

    if not rows:
        logger.error("No event NLP features computed")
        return pd.DataFrame()

    panel = pd.DataFrame(rows)

    # Fill default columns
    for col in ['NEWS_SENT_PRE', 'NEWS_SENT_POST', 'NEWS_N_PRE', 'NEWS_N_POST',
                'NEWS_SENT_CHANGE', 'FILING_MDA_TONE', 'FILING_RISK_TONE',
                'FILING_MDA_CHANGE', 'FILING_RISK_CHANGE']:
        if col not in panel.columns:
            panel[col] = np.nan
    for col in ['NEWS_N_PRE', 'NEWS_N_POST']:
        panel[col] = panel[col].fillna(0).astype(int)

    n_change = panel['NEWS_SENT_CHANGE'].notna().sum() if 'NEWS_SENT_CHANGE' in panel.columns else 0
    logger.info("Event NLP panel: %d events, %d with news, %d with filings, "
                "%d with NEWS_SENT_CHANGE",
                len(panel),
                panel['NEWS_SENT_PRE'].notna().sum(),
                panel['FILING_MDA_TONE'].notna().sum(),
                n_change)

    return panel


# ── Full NLP Pipeline ─────────────────────────────────────────────────

def run_nlp_analysis(
    store: DataStore,
    news_path: str = None,
    download_filings: bool = True,
    filing_sections_df: pd.DataFrame = None,
    finbert_pipeline=None,
) -> Optional[NLPAnalysis]:
    """
    Run the complete Essay 2 NLP pipeline.

    Steps
    -----
    1. Load and score news articles with FinBERT
    2. Download 10-K/10-Q filing text, extract sections
    3. Score filing sections with FinBERT
    4. Build event-window NLP panel

    Parameters
    ----------
    store : DataStore
    news_path : str, optional
        Path to news CSV.  If None, uses default location.
    download_filings : bool
        If True, download filing text from EDGAR (slow, rate-limited).
        If False, skip filing analysis.
    filing_sections_df : pd.DataFrame, optional
        Pre-extracted filing sections (skips EDGAR download).
    finbert_pipeline : optional
        Pre-loaded FinBERT pipeline (avoids reloading ~440MB model).

    Returns
    -------
    NLPAnalysis or None
    """
    # ── Step 1: Load and score news ──
    logger.info("Step 1: Loading news articles...")

    news_df = _load_news(store, news_path)
    news_scored = pd.DataFrame()

    if not news_df.empty:
        try:
            news_scored = score_news_sentiment(news_df, finbert_pipeline=finbert_pipeline)
        except ImportError:
            logger.warning("FinBERT unavailable (torch/transformers not installed). "
                           "Skipping news sentiment scoring — text still available for alignment.")
            # Keep raw news with TEXT column for alignment pipeline
            news_scored = news_df.copy()
            if 'TEXT' not in news_scored.columns:
                title = news_scored.get('TITLE', pd.Series(dtype=str)).fillna('')
                snippet = news_scored.get('SNIPPET', pd.Series(dtype=str)).fillna('')
                news_scored['TEXT'] = (title + ' ' + snippet).str.strip()
            news_scored = news_scored[news_scored['TEXT'].str.len() > 10].reset_index(drop=True)
    else:
        logger.warning("No news articles found")

    # ── Step 2: Filing text extraction ──
    filing_sections = filing_sections_df if filing_sections_df is not None else pd.DataFrame()

    if download_filings and filing_sections.empty:
        logger.info("Step 2: Downloading filing text from EDGAR...")

        # Get all tickers (treatment + control)
        tickers = list(set(store.get_event_tickers() + store.get_control_tickers()))
        tickers = [t for t in tickers if t]  # remove empties

        if tickers:
            filing_sections = download_and_parse_filings(tickers)
        else:
            logger.warning("No tickers found for filing download")
    elif not download_filings and filing_sections.empty:
        logger.info("Step 2: Skipping filing download (download_filings=False)")

    # ── Step 3: Score filing sections ──
    filing_scored = pd.DataFrame()

    if not filing_sections.empty:
        try:
            logger.info("Step 3: Scoring filing sections with FinBERT...")
            filing_scored = score_filing_sentiment(
                filing_sections, finbert_pipeline=finbert_pipeline)
        except ImportError:
            logger.warning("FinBERT unavailable — skipping filing sentiment scoring. "
                           "Filing text still available for alignment.")
            filing_scored = filing_sections.copy()
    else:
        logger.info("Step 3: No filing sections to score")

    # ── Step 4: Build event NLP panel ──
    logger.info("Step 4: Building event NLP panel...")

    event_panel = build_event_nlp_panel(
        store,
        news_scored=news_scored if not news_scored.empty else None,
        filing_scored=filing_scored if not filing_scored.empty else None,
    )

    n_articles = len(news_scored) if not news_scored.empty else 0
    n_filings = len(filing_scored) if not filing_scored.empty else 0
    n_tickers = event_panel['TICKER'].nunique() if not event_panel.empty else 0

    logger.info("NLP analysis complete: %d articles, %d filing sections, %d tickers",
                n_articles, n_filings, n_tickers)

    return NLPAnalysis(
        news_sentiment=news_scored,
        filing_sentiment=filing_scored,
        event_nlp_panel=event_panel,
        n_articles_scored=n_articles,
        n_filings_scored=n_filings,
        n_tickers=n_tickers,
    )


def save_nlp_results(
    store: DataStore,
    result: NLPAnalysis,
) -> dict:
    """Persist NLP results to the database."""
    saved = {}
    timestamp = pd.Timestamp.now().isoformat()

    if not result.news_sentiment.empty:
        # Save summary (not full text) to keep table size manageable.
        # WARNING: TEXT column is intentionally stripped here.  If you later
        # load ESSAY2_NEWS_SENTIMENT and pass it to compute_political_alignment,
        # the alignment pipeline requires TEXT for corpus construction.  In that
        # case, reload from the original news source, not this persisted table.
        news_summary = result.news_sentiment[[
            c for c in result.news_sentiment.columns if c != 'TEXT'
        ]].copy()
        saved['ESSAY2_NEWS_SENTIMENT'] = store.write_table(
            news_summary.assign(RUN_TIMESTAMP=timestamp),
            'ESSAY2_NEWS_SENTIMENT', replace=True,
        )

    if not result.filing_sentiment.empty:
        saved['ESSAY2_FILING_SENTIMENT'] = store.write_table(
            result.filing_sentiment.assign(RUN_TIMESTAMP=timestamp),
            'ESSAY2_FILING_SENTIMENT', replace=True,
        )

    if not result.event_nlp_panel.empty:
        saved['ESSAY2_EVENT_NLP'] = store.write_table(
            result.event_nlp_panel.assign(RUN_TIMESTAMP=timestamp),
            'ESSAY2_EVENT_NLP', replace=True,
        )

    logger.info("NLP results: saved %d/%d tables", len(saved), 3)
    return saved


# ── News Loading (shared with essay1) ────────────────────────────────

# Culture-war relevance terms for filtering out off-topic articles.
# Articles must contain at least one of these in TITLE or SNIPPET.
# Deliberately excludes overly broad terms ('brand', 'campaign', 'ban',
# 'political') that match non-culture-war articles.
#
# Some entries are intentional stems for substring matching:
#   'polariz' → matches polarize, polarizing, polarization
#   'ideolog' → matches ideology, ideological, ideologue
# Full phrases (e.g. 'cancel culture') require exact substring match.
#
# Immutable: _RELEVANCE_PATTERN is compiled once at import time from these.
# If you change these, _RELEVANCE_PATTERN must be rebuilt.
_CULTURE_WAR_TERMS = (
    'boycott', 'buycott', 'backlash', 'controversy', 'protest',
    'cancel culture', 'woke', 'political stance', 'diversity equity',
    'pride month', 'conservative backlash', 'liberal backlash',
    'transgender', 'dei initiative', 'dei policy', 'esg policy',
    'esg initiative', 'activist pressure', 'activist investor',
    'culture war', 'brand activism', 'corporate activism',
    'outrage', 'polariz', 'partisan', 'ideolog',
    'inclusion initiative', 'equity initiative', 'lgbtq', 'blm',
    'immigration', 'gun control', 'abortion', 'censorship',
    'free speech', 'critical race', 'anti-woke',
)

# Terms requiring word-boundary markers to avoid substring false positives
_CULTURE_WAR_BOUNDARY_TERMS = frozenset({
    'protest', 'outrage', 'abortion', 'transgender', 'censorship',
})


def _build_relevance_pattern() -> str:
    """Build compiled regex pattern for culture-war relevance filtering."""
    parts = []
    for t in _CULTURE_WAR_TERMS:
        escaped = re.escape(t)
        if t in _CULTURE_WAR_BOUNDARY_TERMS:
            parts.append(r'\b' + escaped + r'\b')
        else:
            parts.append(escaped)
    return '|'.join(parts)


_RELEVANCE_PATTERN = _build_relevance_pattern()


def _load_news(store: DataStore, news_path: str = None) -> pd.DataFrame:
    """
    Load news articles from CSV or database.

    Tries news_path first, then default location, then database table.
    Applies a culture-war relevance filter to remove off-topic articles
    (sports, weather, etc.) that would contaminate sentiment scoring.
    """
    # Try explicit path
    if news_path and Path(news_path).exists():
        logger.info("Loading news from %s", news_path)
        df = pd.read_csv(news_path)
    else:
        # Try default location
        default_path = Path(__file__).resolve().parent.parent / 'news_data' / 'culture_war_news.csv'
        if default_path.exists():
            logger.info("Loading news from %s", default_path)
            df = pd.read_csv(default_path)
        else:
            # Try database
            df = store.read_table('CULTURE_WAR_NEWS')
            if df.empty:
                return pd.DataFrame()

    # Normalize columns
    df.columns = [c.upper() for c in df.columns]

    # Build DATE from PUBLISHED_DATE
    if 'PUBLISHED_DATE' in df.columns and 'DATE' not in df.columns:
        df['DATE'] = pd.to_datetime(df['PUBLISHED_DATE'], errors='coerce')

    if 'DATE' in df.columns:
        df['DATE'] = pd.to_datetime(df['DATE'], errors='coerce')

    # Build TEXT from TITLE + SNIPPET if not present
    if 'TEXT' not in df.columns:
        title = df.get('TITLE', pd.Series(dtype=str)).fillna('')
        snippet = df.get('SNIPPET', pd.Series(dtype=str)).fillna('')
        df['TEXT'] = (title + ' ' + snippet).str.strip()

    # Normalize TICKER column
    if 'TICKER' in df.columns:
        df['TICKER'] = df['TICKER'].str.upper().str.strip()

    # ── Relevance filter: remove off-topic articles (vectorized) ──
    n_before = len(df)
    title_col = df['TITLE'].fillna('') if 'TITLE' in df.columns else pd.Series('', index=df.index)
    snippet_col = df['SNIPPET'].fillna('') if 'SNIPPET' in df.columns else pd.Series('', index=df.index)
    _combined_text = (title_col + ' ' + snippet_col).str.lower()
    df = df[_combined_text.str.contains(_RELEVANCE_PATTERN, regex=True, na=False)].reset_index(drop=True)
    n_removed = n_before - len(df)
    logger.info("Relevance filter: %d → %d articles (%d removed, %.1f%%)",
                n_before, len(df), n_removed,
                100 * n_removed / n_before if n_before > 0 else 0)

    logger.info("Loaded %d relevant news articles for %d tickers",
                len(df), df['TICKER'].nunique() if 'TICKER' in df.columns else 0)

    return df


# ── Political Alignment via Platform Comparison ──────────────────────
#
# Three-signal approach to measuring political alignment:
#
#   Signal 1 — Distinctive phrases (TF-IDF difference):
#     Extract terms that discriminate between R and D platforms.
#     Project company text onto ONLY these discriminating terms.
#     This filters out shared political vocabulary ("economy", "security")
#     that adds noise without alignment signal.
#
#   Signal 2 — Source weighting (filing text > news):
#     Filing text (10-K/10-Q MD&A) is the company's own voice.
#     News text is journalists'/Reddit's voice ABOUT the company.
#     Filing similarity is weighted 3x higher than news similarity.
#
#   Signal 3 — Stance detection (FinBERT on political topics):
#     "Opposing gun control" and "supporting gun control" both match
#     "gun control" in TF-IDF, but carry opposite political signals.
#     FinBERT sentiment on text windows around distinctive phrases
#     captures whether the company is FOR or AGAINST each topic.
#     Positive sentiment on R-topics → R-leaning, and vice versa.
#
# Final score = weighted combination of all three signals.
#
# References:
#   Gentzkow, M. & Shapiro, J. (2010). What drives media slant?
#       Econometrica, 78(1).
#   Loughran, T. & McDonald, B. (2011). When is a liability not a
#       liability? Journal of Finance, 66(1).

_PLATFORM_DATA_DIR = Path(__file__).resolve().parent.parent / 'party_platforms_data'

# Election years with available platforms
_PLATFORM_YEARS = [2004, 2008, 2012, 2016, 2020, 2024]

# Source weighting: filing text is the company's own voice
_FILING_WEIGHT = 3.0
_NEWS_WEIGHT = 1.0

# Number of distinctive phrases per party to extract
_N_DISTINCTIVE = 200

# Context window (chars) around a distinctive phrase for stance detection
_STANCE_CONTEXT_CHARS = 300

# Signal weights in composite score
_W_DISTINCTIVE = 0.4    # distinctive-phrase cosine similarity
_W_STANCE = 0.4          # FinBERT stance on political topics
_W_COSINE = 0.2          # raw cosine similarity (original v1 method)

# Minimum classifier agreement vs hand-coded labels (3-class baseline = 33%)
_MIN_AGREEMENT_RATE = 0.55


@dataclass
class PoliticalAlignmentResult:
    """Text-derived political alignment scores for all companies."""
    company_scores: pd.DataFrame    # TICKER, ALIGNMENT_SCORE, + component scores
    event_scores: pd.DataFrame      # TICKER, EVENT_DATE, ALIGNMENT_SCORE
    distinctive_phrases: pd.DataFrame  # PHRASE, PARTY, TFIDF_DIFF
    platform_vocab_size: int
    n_companies: int
    validation: pd.DataFrame        # comparison with hand-coded leaning
    agreement_rate: float = np.nan   # computed vs hand-coded agreement
    conservative_threshold: float = 0.05
    liberal_threshold: float = -0.05
    # Effective weights (after renormalization if stance unavailable)
    w_distinctive: float = _W_DISTINCTIVE
    w_stance: float = _W_STANCE
    w_cosine: float = _W_COSINE


def load_platform_corpus(
    platform_dir: str = None,
) -> pd.DataFrame:
    """
    Load all party platform texts into a DataFrame.

    Returns
    -------
    pd.DataFrame with columns: YEAR, PARTY, TEXT, WORD_COUNT
    """
    pdir = Path(platform_dir) if platform_dir else _PLATFORM_DATA_DIR

    if not pdir.exists():
        logger.warning(
            "Platform data directory not found at %s. "
            "Political alignment unavailable. "
            "Run: python scripts/download_platforms.py", pdir
        )
        return pd.DataFrame()

    rows = []
    for year in _PLATFORM_YEARS:
        for party in ['republican', 'democratic']:
            txt_path = pdir / f'{party}_platform_{year}.txt'
            if not txt_path.exists():
                logger.warning("Platform file not found: %s", txt_path)
                continue
            text = txt_path.read_text(encoding='utf-8')
            rows.append({
                'YEAR': year,
                'PARTY': party.capitalize(),
                'TEXT': text,
                'WORD_COUNT': len(text.split()),
            })

    df = pd.DataFrame(rows)
    logger.info("Loaded %d platform texts (%d R, %d D)",
                len(df),
                (df['PARTY'] == 'Republican').sum() if not df.empty else 0,
                (df['PARTY'] == 'Democratic').sum() if not df.empty else 0)
    return df


def _nearest_platform_years(event_year: int, n: int = 2) -> List[int]:
    """Return the n nearest prior platform years to smooth temporal bias.

    For events before the first available platform year, uses the
    earliest n available platforms with a look-ahead warning.  Always
    returns exactly min(n, len(_PLATFORM_YEARS)) years so that the
    averaging loop processes a consistent number of years per ticker.
    """
    prior = sorted([y for y in _PLATFORM_YEARS if y <= event_year], reverse=True)
    if prior:
        result = prior[:n]
        if len(result) < n:
            logger.debug("Event year %d: only %d prior platform year(s) available %s "
                         "(requested %d)", event_year, len(result), result, n)
        return result
    # Pre-platform era: use earliest available, flag look-ahead
    earliest_n = sorted(_PLATFORM_YEARS)[:n]
    logger.warning(
        "Event year %d predates earliest platform (%d) — "
        "look-ahead bias possible in alignment score (using %s)",
        event_year, earliest_n[0], earliest_n,
    )
    return earliest_n


# ── Signal 1: Distinctive Phrases ────────────────────────────────────

def extract_distinctive_phrases(
    platforms: pd.DataFrame,
    n_phrases: int = _N_DISTINCTIVE,
) -> Tuple[pd.DataFrame, object]:
    """
    Find terms that discriminate between Republican and Democratic platforms.

    Fits TF-IDF on two mega-documents (all R text concatenated, all D text
    concatenated), then ranks terms by |tfidf_R - tfidf_D|.  The top N
    per party become the distinctive phrase lexicon.

    Parameters
    ----------
    platforms : pd.DataFrame
        Must have PARTY and TEXT columns.
    n_phrases : int
        Number of distinctive phrases per party.

    Returns
    -------
    (phrases_df, fitted_vectorizer)
        phrases_df: PHRASE, PARTY, TFIDF_DIFF, RANK
        vectorizer: fitted TfidfVectorizer (for projecting company text)
    """
    from sklearn.feature_extraction.text import TfidfVectorizer

    # Concatenate all text per party across years
    r_text = ' '.join(platforms[platforms['PARTY'] == 'Republican']['TEXT'].tolist())
    d_text = ' '.join(platforms[platforms['PARTY'] == 'Democratic']['TEXT'].tolist())

    vectorizer = TfidfVectorizer(
        max_features=15000,
        ngram_range=(1, 3),
        stop_words='english',
        min_df=1,
        sublinear_tf=True,
    )

    tfidf_matrix = vectorizer.fit_transform([r_text, d_text])
    feature_names = vectorizer.get_feature_names_out()

    # TF-IDF difference: positive = R-distinctive, negative = D-distinctive
    r_vec = tfidf_matrix[0].toarray().flatten()
    d_vec = tfidf_matrix[1].toarray().flatten()
    diff = r_vec - d_vec

    # Build ranked phrase list
    sorted_indices = np.argsort(diff)

    rows = []
    # D-distinctive (most negative diff)
    for rank, idx in enumerate(sorted_indices[:n_phrases], 1):
        rows.append({
            'PHRASE': feature_names[idx],
            'PARTY': 'Democratic',
            'TFIDF_R': float(r_vec[idx]),
            'TFIDF_D': float(d_vec[idx]),
            'TFIDF_DIFF': float(diff[idx]),
            'RANK': rank,
        })

    # R-distinctive (most positive diff)
    for rank, idx in enumerate(sorted_indices[-n_phrases:][::-1], 1):
        rows.append({
            'PHRASE': feature_names[idx],
            'PARTY': 'Republican',
            'TFIDF_R': float(r_vec[idx]),
            'TFIDF_D': float(d_vec[idx]),
            'TFIDF_DIFF': float(diff[idx]),
            'RANK': rank,
        })

    phrases_df = pd.DataFrame(rows)

    logger.info("Distinctive phrases: %d R-distinctive, %d D-distinctive "
                "(from %d total features)",
                (phrases_df['PARTY'] == 'Republican').sum(),
                (phrases_df['PARTY'] == 'Democratic').sum(),
                len(feature_names))

    return phrases_df, vectorizer


def _score_distinctive_similarity(
    company_text: str,
    vectorizer,
    platforms: pd.DataFrame,
    platform_year: int,
    phrases_df: pd.DataFrame,
) -> Dict[str, float]:
    """
    Compute cosine similarity using only distinctive-phrase features.

    Projects company text into the platform TF-IDF space, zeros out
    all features that are NOT in the distinctive phrase set, then
    computes cosine similarity to R and D platform vectors.
    """
    from sklearn.metrics.pairwise import cosine_similarity

    feature_names = list(vectorizer.get_feature_names_out())
    distinctive_terms = set(phrases_df['PHRASE'].tolist())

    # Build mask: 1 for distinctive features, 0 for shared vocabulary
    mask = np.array([1.0 if f in distinctive_terms else 0.0 for f in feature_names])

    # Concatenate platform text for the target year
    r_plat = platforms[(platforms['PARTY'] == 'Republican') &
                       (platforms['YEAR'] == platform_year)]
    d_plat = platforms[(platforms['PARTY'] == 'Democratic') &
                       (platforms['YEAR'] == platform_year)]

    if r_plat.empty or d_plat.empty:
        return {'sim_r': 0.0, 'sim_d': 0.0, 'distinctive_align': 0.0}

    # Transform all three texts
    company_vec = vectorizer.transform([company_text]).toarray().flatten() * mask
    r_vec = vectorizer.transform([r_plat.iloc[0]['TEXT']]).toarray().flatten() * mask
    d_vec = vectorizer.transform([d_plat.iloc[0]['TEXT']]).toarray().flatten() * mask

    # Cosine similarity (manual, since vectors are already flattened)
    def _cos(a, b):
        dot = np.dot(a, b)
        na = np.linalg.norm(a)
        nb = np.linalg.norm(b)
        if na == 0 or nb == 0:
            return 0.0
        return float(dot / (na * nb))

    sim_r = _cos(company_vec, r_vec)
    sim_d = _cos(company_vec, d_vec)

    return {
        'sim_r': sim_r,
        'sim_d': sim_d,
        'distinctive_align': sim_r - sim_d,
    }


# ── Signal 2: Source-Weighted Corpus ─────────────────────────────────

def _build_weighted_company_corpus(
    store: DataStore,
    news_df: pd.DataFrame = None,
    filing_sections: pd.DataFrame = None,
    filing_weight: float = _FILING_WEIGHT,
    news_weight: float = _NEWS_WEIGHT,
) -> Tuple[Dict[str, str], Dict[str, str], Dict[str, str]]:
    """
    Build separate text corpora for news and filings, plus a weighted
    combined corpus.

    Filing text is repeated filing_weight/news_weight times in the
    combined corpus so TF-IDF gives it proportionally more influence.

    Returns
    -------
    (combined_corpus, news_corpus, filing_corpus)
        Each is dict {TICKER: text}.
    """
    news_corpus = {}
    filing_corpus = {}

    if news_df is not None and not news_df.empty and 'TEXT' in news_df.columns:
        news_corpus = (
            news_df.groupby('TICKER')['TEXT']
            .apply(lambda t: ' '.join(t.dropna().astype(str)))
            .to_dict()
        )

    if filing_sections is not None and not filing_sections.empty and 'TEXT' in filing_sections.columns:
        filing_corpus = (
            filing_sections.groupby('TICKER')['TEXT']
            .apply(lambda t: ' '.join(t.dropna().astype(str)))
            .to_dict()
        )

    # Build weighted combination
    all_tickers = set(news_corpus.keys()) | set(filing_corpus.keys())
    combined = {}

    # Repeat ratio: how many times to replicate filing text relative to news.
    # Note: integer repetition is a coarse proxy for continuous weighting in
    # TF-IDF space.  E.g. filing_weight=1.5 rounds to 2x, not 1.5x.
    # For more precise weighting, consider corpus-level IDF adjustment.
    repeat_count = max(1, int(round(filing_weight / news_weight)))
    if abs(filing_weight / news_weight - repeat_count) > 0.1:
        logger.debug("Filing repeat_count=%d approximates requested weight "
                     "ratio %.2f (error=%.2f)",
                     repeat_count, filing_weight / news_weight,
                     abs(filing_weight / news_weight - repeat_count))

    for ticker in all_tickers:
        parts = []
        if ticker in news_corpus:
            parts.append(news_corpus[ticker])
        if ticker in filing_corpus:
            # Repeat filing text to upweight it in TF-IDF
            parts.extend([filing_corpus[ticker]] * repeat_count)
        combined[ticker] = ' '.join(parts).strip()

    combined = {k: v for k, v in combined.items() if v}

    n_both = len(set(news_corpus.keys()) & set(filing_corpus.keys()))
    logger.info("Weighted corpus: %d tickers (%d news-only, %d filing-only, "
                "%d both, filing weight=%dx)",
                len(combined),
                len(set(news_corpus.keys()) - set(filing_corpus.keys())),
                len(set(filing_corpus.keys()) - set(news_corpus.keys())),
                n_both, repeat_count)

    return combined, news_corpus, filing_corpus


# ── Signal 3: Stance Detection ───────────────────────────────────────

def _extract_topic_windows(
    text: str,
    phrases: List[str],
    context_chars: int = _STANCE_CONTEXT_CHARS,
) -> List[Tuple[str, str]]:
    """
    Find text windows around distinctive political phrases.

    Returns list of (phrase, context_window) tuples.
    Each window is the surrounding text used for FinBERT stance scoring.
    """
    text_lower = text.lower()
    windows = []

    for phrase in phrases:
        # Find all occurrences
        start = 0
        phrase_lower = phrase.lower()
        while True:
            idx = text_lower.find(phrase_lower, start)
            if idx == -1:
                break
            # Extract surrounding context
            win_start = max(0, idx - context_chars)
            win_end = min(len(text), idx + len(phrase) + context_chars)
            window = text[win_start:win_end].strip()
            if len(window) > 20:
                windows.append((phrase, window))
            start = idx + len(phrase)

    return windows


def compute_stance_scores(
    company_corpus: Dict[str, str],
    phrases_df: pd.DataFrame,
    finbert_pipeline=None,
) -> Dict[str, Dict[str, float]]:
    """
    Score company stance on political topics using FinBERT.

    For each company, finds text windows around distinctive phrases,
    scores them with FinBERT, then computes:
      - mean sentiment on R-distinctive topics
      - mean sentiment on D-distinctive topics
      - stance_alignment = sent_on_R_topics - sent_on_D_topics

    Positive stance_alignment means the company speaks positively about
    R topics (or negatively about D topics), suggesting R-leaning.

    Parameters
    ----------
    company_corpus : dict
        {TICKER: text}
    phrases_df : pd.DataFrame
        Distinctive phrases with PHRASE and PARTY columns.
    finbert_pipeline : optional
        Pre-loaded FinBERT pipeline.

    Returns
    -------
    dict {TICKER: {'stance_r': float, 'stance_d': float, 'stance_align': float,
                    'n_r_windows': int, 'n_d_windows': int}}
    """
    r_phrases = phrases_df[phrases_df['PARTY'] == 'Republican']['PHRASE'].tolist()[:100]
    d_phrases = phrases_df[phrases_df['PARTY'] == 'Democratic']['PHRASE'].tolist()[:100]

    results = {}

    # Collect all windows across all companies for batched scoring
    all_windows = []     # (ticker, party, window_text)
    for ticker, text in company_corpus.items():
        for phrase, window in _extract_topic_windows(text, r_phrases):
            all_windows.append((ticker, 'Republican', window))
        for phrase, window in _extract_topic_windows(text, d_phrases):
            all_windows.append((ticker, 'Democratic', window))

    if not all_windows:
        logger.warning("No topic windows found in company text")
        return {}

    # Log phrase coverage (derived from already-collected windows to avoid O(n×m) scan)
    n_r_windows = sum(1 for _, party, _ in all_windows if party == 'Republican')
    n_d_windows = sum(1 for _, party, _ in all_windows if party == 'Democratic')
    r_tickers = len(set(t for t, party, _ in all_windows if party == 'Republican'))
    d_tickers = len(set(t for t, party, _ in all_windows if party == 'Democratic'))
    logger.info("Phrase coverage: %d R windows across %d/%d companies, "
                "%d D windows across %d/%d companies",
                n_r_windows, r_tickers, len(company_corpus),
                n_d_windows, d_tickers, len(company_corpus))

    logger.info("Stance detection: %d topic windows across %d companies",
                len(all_windows), len(company_corpus))

    # Score all windows in one batch
    window_texts = [w[2] for w in all_windows]

    # Truncate long windows for FinBERT
    window_texts = [t[:1800] for t in window_texts]

    pipe = _get_finbert_pipeline(finbert_pipeline)
    scores = _score_finbert(window_texts, pipe=pipe)

    # Aggregate per company per party
    ticker_party_sents = {}  # (ticker, party) -> list of sentiments

    for (ticker, party, _), score in zip(all_windows, scores):
        if score['label'] is None:
            continue
        sent = _sentiment_to_numeric(score['label'])
        weighted = sent * (score['score'] or 0.0)
        key = (ticker, party)
        if key not in ticker_party_sents:
            ticker_party_sents[key] = []
        ticker_party_sents[key].append(weighted)

    # Compute stance per company
    all_tickers = set(t for t, _ in ticker_party_sents.keys())
    for ticker in all_tickers:
        r_sents = ticker_party_sents.get((ticker, 'Republican'), [])
        d_sents = ticker_party_sents.get((ticker, 'Democratic'), [])

        stance_r = np.mean(r_sents) if r_sents else 0.0
        stance_d = np.mean(d_sents) if d_sents else 0.0

        # Positive stance on R topics + negative stance on D topics → R-leaning
        # We want: "speaks favorably about R issues, unfavorably about D issues"
        stance_align = stance_r - stance_d

        results[ticker] = {
            'stance_r': float(stance_r),
            'stance_d': float(stance_d),
            'stance_align': float(stance_align),
            'n_r_windows': len(r_sents),
            'n_d_windows': len(d_sents),
        }

    logger.info("Stance scores: %d companies — mean R=%.3f, mean D=%.3f",
                len(results),
                np.mean([r['stance_r'] for r in results.values()]),
                np.mean([r['stance_d'] for r in results.values()]))

    return results


# ── Alignment Classification ─────────────────────────────────────────

def classify_alignment(
    score: float,
    conservative_threshold: float = 0.05,
    liberal_threshold: float = -0.05,
) -> str:
    """
    Classify firm political lean from continuous alignment score.

    Positive score = Republican-leaning = Conservative
    Negative score = Democratic-leaning = Liberal
    Near-zero = Mixed

    Parameters
    ----------
    score : float
        Composite alignment score.
    conservative_threshold : float
        Score above this → Conservative.
    liberal_threshold : float
        Score below this → Liberal.

    Returns
    -------
    str : 'Conservative', 'Liberal', or 'Mixed'
    """
    if score > conservative_threshold:
        return 'Conservative'
    elif score < liberal_threshold:
        return 'Liberal'
    else:
        return 'Mixed'


def threshold_sensitivity_table(
    company_scores: pd.DataFrame,
    thresholds: list = None,
) -> pd.DataFrame:
    """
    Build a threshold-sensitivity table for political-alignment classification.

    For each symmetric threshold tau in *thresholds*, classifies firms as
    Conservative (ALIGNMENT_SCORE > tau), Liberal (< -tau), or Mixed, and
    reports group counts, mean scores, and Cohen's d between the
    Conservative and Liberal groups as a separation metric.

    Parameters
    ----------
    company_scores : pd.DataFrame
        Must contain ALIGNMENT_SCORE (continuous).
    thresholds : list of float, optional
        Symmetric threshold grid.  Default [0.05, 0.10, 0.15, 0.20, 0.25].

    Returns
    -------
    pd.DataFrame
        One row per threshold with columns:
        TAU, N_CONSERVATIVE, N_LIBERAL, N_MIXED, PCT_CONSERVATIVE,
        PCT_LIBERAL, PCT_MIXED, MEAN_SCORE_C, MEAN_SCORE_L, MEAN_SCORE_M,
        COHENS_D.
    """
    if thresholds is None:
        thresholds = [0.05, 0.10, 0.15, 0.20, 0.25]

    scores = company_scores['ALIGNMENT_SCORE']
    n_total = len(scores)

    rows = []
    for tau in thresholds:
        classified = company_scores['ALIGNMENT_SCORE'].apply(
            lambda s, _t=tau: classify_alignment(s, _t, -_t))

        n_c = (classified == 'Conservative').sum()
        n_l = (classified == 'Liberal').sum()
        n_m = (classified == 'Mixed').sum()

        mask_c = classified == 'Conservative'
        mask_l = classified == 'Liberal'
        mask_m = classified == 'Mixed'

        # Cohen's d: separation between Conservative and Liberal groups
        s_c = scores[mask_c]
        s_l = scores[mask_l]
        if len(s_c) >= 2 and len(s_l) >= 2:
            pooled_std = np.sqrt(
                ((len(s_c) - 1) * s_c.std()**2 + (len(s_l) - 1) * s_l.std()**2)
                / (len(s_c) + len(s_l) - 2)
            )
            cohens_d = (s_c.mean() - s_l.mean()) / pooled_std if pooled_std > 0 else np.nan
        else:
            cohens_d = np.nan

        rows.append({
            'TAU': tau,
            'N_CONSERVATIVE': n_c,
            'N_LIBERAL': n_l,
            'N_MIXED': n_m,
            'PCT_CONSERVATIVE': 100 * n_c / n_total if n_total else np.nan,
            'PCT_LIBERAL': 100 * n_l / n_total if n_total else np.nan,
            'PCT_MIXED': 100 * n_m / n_total if n_total else np.nan,
            'MEAN_SCORE_C': scores[mask_c].mean() if mask_c.any() else np.nan,
            'MEAN_SCORE_L': scores[mask_l].mean() if mask_l.any() else np.nan,
            'MEAN_SCORE_M': scores[mask_m].mean() if mask_m.any() else np.nan,
            'COHENS_D': cohens_d,
        })

    return pd.DataFrame(rows)


# ── Composite Alignment ──────────────────────────────────────────────

def compute_political_alignment(
    store: DataStore,
    news_scored: pd.DataFrame = None,
    filing_sections: pd.DataFrame = None,
    platform_dir: str = None,
    run_stance: bool = True,
    finbert_pipeline=None,
    w_distinctive: float = _W_DISTINCTIVE,
    w_stance: float = _W_STANCE,
    w_cosine: float = _W_COSINE,
    conservative_threshold: float = 0.05,
    liberal_threshold: float = -0.05,
) -> Optional[PoliticalAlignmentResult]:
    """
    Compute text-derived political alignment for each company.

    Combines three signals:
      1. Distinctive-phrase cosine similarity (TF-IDF diff filtering)
      2. FinBERT stance on political topics (optional, requires torch)
      3. Source-weighted raw cosine similarity (filings weighted 3x)

    Final score:
        ALIGNMENT = w1 * distinctive + w2 * stance + w3 * cosine
    where each component is normalized to roughly [-1, +1].

    Parameters
    ----------
    store : DataStore
    news_scored : pd.DataFrame, optional
        News with TEXT column (FinBERT scoring not required here).
    filing_sections : pd.DataFrame, optional
        Filing sections with TEXT column.
    platform_dir : str, optional
    run_stance : bool
        If True, run FinBERT stance detection (requires torch).
        If False, uses only TF-IDF signals (faster, no GPU needed).
    finbert_pipeline : optional
        Pre-loaded FinBERT pipeline for stance detection.
    w_distinctive, w_stance, w_cosine : float
        Signal weights.  Renormalized internally if stance is skipped.
    conservative_threshold : float
        Alignment score above this → Conservative. Default 0.05.
        Adjust based on threshold sensitivity output after first run.
    liberal_threshold : float
        Alignment score below this → Liberal. Default -0.05.
        Should be negative; consider asymmetric values if distribution skews.

    Returns
    -------
    PoliticalAlignmentResult or None
    """
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.metrics.pairwise import cosine_similarity

    # Step 1: Load platforms
    platforms = load_platform_corpus(platform_dir)
    if platforms.empty:
        logger.error("No platform texts available")
        return None

    # Step 2: Extract distinctive phrases
    logger.info("Step 1: Extracting distinctive platform phrases...")
    phrases_df, disc_vectorizer = extract_distinctive_phrases(platforms)

    # Step 3: Build weighted company corpus
    # Guard: news_scored must have TEXT column for corpus construction.
    # If loaded from ESSAY2_NEWS_SENTIMENT (which strips TEXT), warn early.
    if news_scored is not None and not news_scored.empty and 'TEXT' not in news_scored.columns:
        logger.warning(
            "news_scored is missing TEXT column — was it loaded from "
            "ESSAY2_NEWS_SENTIMENT (which strips TEXT to save space)? "
            "Alignment pipeline needs TEXT. News corpus will be empty.")
    logger.info("Step 2: Building source-weighted company corpus...")
    combined_corpus, news_corpus, filing_corpus = _build_weighted_company_corpus(
        store, news_df=news_scored, filing_sections=filing_sections)

    if not combined_corpus:
        logger.error("No company text available for alignment")
        return None

    # Step 4: Fit cosine vectorizer on per-year platforms (for raw cosine signal)
    cosine_vectorizer = TfidfVectorizer(
        max_features=10000,
        ngram_range=(1, 2),
        stop_words='english',
        min_df=1,
        max_df=0.95,
        sublinear_tf=True,
    )
    platforms = platforms.reset_index(drop=True)
    platform_texts = platforms['TEXT'].tolist()
    cosine_vectorizer.fit(platform_texts)
    platform_vectors = cosine_vectorizer.transform(platform_texts)

    # Step 5: Get event years per ticker
    events_df = store.read_table('CULTURE_WAR_COMPANIES')
    if events_df.empty:
        logger.error("CULTURE_WAR_COMPANIES table empty — cannot determine event years. "
                     "Alignment scores require event dates to select the correct "
                     "platform years. Populate the table before running alignment.")
        return None
    events_df.columns = [c.upper() for c in events_df.columns]
    if 'EVENT_DATE' not in events_df.columns and 'DATE' not in events_df.columns:
        logger.error("CULTURE_WAR_COMPANIES table missing date column "
                     "(need EVENT_DATE or DATE, have: %s)", list(events_df.columns))
        return None
    date_col = 'EVENT_DATE' if 'EVENT_DATE' in events_df.columns else 'DATE'

    events_df[date_col] = pd.to_datetime(events_df[date_col], errors='coerce')
    ticker_years = (
        events_df.dropna(subset=[date_col])
        .groupby('TICKER')[date_col]
        .first()
        .dt.year
        .to_dict()
    )

    # Step 6: Stance detection (optional)
    stance_scores = {}
    has_stance = False

    if run_stance:
        try:
            logger.info("Step 3: Running FinBERT stance detection...")
            stance_scores = compute_stance_scores(
                combined_corpus, phrases_df, finbert_pipeline=finbert_pipeline)
            has_stance = bool(stance_scores)
            if has_stance:
                logger.info("Stance detection: %d companies scored", len(stance_scores))
        except Exception as e:
            logger.warning("Stance detection failed (torch not available?): %s", e)
            has_stance = False

    # Renormalize weights if stance is unavailable
    if not has_stance:
        total = w_distinctive + w_cosine
        w_distinctive = w_distinctive / total if total > 0 else 0.5
        w_cosine = w_cosine / total if total > 0 else 0.5
        w_stance = 0.0
        logger.info("Stance unavailable — reweighted: distinctive=%.3f, cosine=%.3f",
                     w_distinctive, w_cosine)

    # Log effective weights (always, for audit trail in DiD interpretation)
    logger.info("Composite alignment weights: distinctive=%.3f, stance=%.3f, cosine=%.3f "
                "(sum=%.3f)", w_distinctive, w_stance, w_cosine,
                w_distinctive + w_stance + w_cosine)

    # Step 7: Score each company
    logger.info("Step 4: Computing composite alignment scores...")
    company_rows = []
    _n_default_year = 0

    for ticker, text in combined_corpus.items():
        if len(text.split()) < 50:
            continue

        event_year = ticker_years.get(ticker, None)
        if event_year is None:
            event_year = 2020
            _n_default_year += 1
            logger.debug("%s: no event year found, defaulting to 2020", ticker)
        # Average across two nearest platform years to reduce temporal bias
        years = _nearest_platform_years(event_year, n=2)

        # Signal 1: Distinctive-phrase similarity (averaged across platform years)
        disc_results = [_score_distinctive_similarity(
            text, disc_vectorizer, platforms, y, phrases_df) for y in years]
        disc_align = np.mean([d['distinctive_align'] for d in disc_results])
        disc_sim_r = np.mean([d['sim_r'] for d in disc_results])
        disc_sim_d = np.mean([d['sim_d'] for d in disc_results])

        # Signal 2: Raw cosine similarity (averaged across platform years)
        company_vec = cosine_vectorizer.transform([text])
        cos_r_vals, cos_d_vals = [], []
        for y in years:
            r_mask = (platforms['YEAR'] == y) & (platforms['PARTY'] == 'Republican')
            d_mask = (platforms['YEAR'] == y) & (platforms['PARTY'] == 'Democratic')
            r_idx = platforms[r_mask].index.tolist()
            d_idx = platforms[d_mask].index.tolist()
            if not r_idx or not d_idx:
                continue
            cos_r_vals.append(float(cosine_similarity(company_vec, platform_vectors[r_idx[0]])[0, 0]))
            cos_d_vals.append(float(cosine_similarity(company_vec, platform_vectors[d_idx[0]])[0, 0]))

        if not cos_r_vals:
            continue

        cos_r = np.mean(cos_r_vals)
        cos_d = np.mean(cos_d_vals)
        cosine_align = cos_r - cos_d

        # Signal 3: Stance
        stance = stance_scores.get(ticker, {})
        stance_align = stance.get('stance_align', 0.0)

        company_rows.append({
            'TICKER': ticker,
            'DISTINCTIVE_ALIGN': disc_align,
            'STANCE_ALIGN': stance_align,
            'COSINE_ALIGN': cosine_align,
            'SIM_REPUBLICAN': cos_r,
            'SIM_DEMOCRATIC': cos_d,
            'DISC_SIM_R': disc_sim_r,
            'DISC_SIM_D': disc_sim_d,
            'STANCE_R': stance.get('stance_r', np.nan),
            'STANCE_D': stance.get('stance_d', np.nan),
            'N_R_WINDOWS': stance.get('n_r_windows', 0),
            'N_D_WINDOWS': stance.get('n_d_windows', 0),
            'PLATFORM_YEAR': years[0],  # primary year
            'HAS_FILING': ticker in filing_corpus,
            'HAS_NEWS': ticker in news_corpus,
        })

    if _n_default_year > 0:
        logger.warning("%d/%d tickers had no event year in CULTURE_WAR_COMPANIES "
                       "and defaulted to 2020 platform years (likely control-only "
                       "tickers)", _n_default_year, len(combined_corpus))

    company_df = pd.DataFrame(company_rows)

    if company_df.empty:
        logger.error("No alignment scores computed")
        return None

    # ── Normalize each signal to [-1, +1] using 5th/95th percentile ──
    def _normalize_signal(series: pd.Series, signal_name: str = '') -> pd.Series:
        p05 = series.quantile(0.05)
        p95 = series.quantile(0.95)
        logger.info("Normalization '%s' (n=%d): p05=%.4f, p95=%.4f, "
                     "min=%.4f, max=%.4f, mean=%.4f, std=%.4f",
                     signal_name, len(series), p05, p95,
                     series.min(), series.max(), series.mean(), series.std())
        if p95 == p05:
            logger.warning("Normalization '%s': p05==p95 (%.4f), returning zeros",
                           signal_name, p05)
            return pd.Series(0.0, index=series.index)
        return (2 * (series - p05) / (p95 - p05) - 1).clip(-1, 1)

    company_df['DISTINCTIVE_ALIGN_NORM'] = _normalize_signal(
        company_df['DISTINCTIVE_ALIGN'], 'DISTINCTIVE_ALIGN')
    if has_stance:
        company_df['STANCE_ALIGN_NORM'] = _normalize_signal(
            company_df['STANCE_ALIGN'], 'STANCE_ALIGN')
    else:
        company_df['STANCE_ALIGN_NORM'] = 0.0
    company_df['COSINE_ALIGN_NORM'] = _normalize_signal(
        company_df['COSINE_ALIGN'], 'COSINE_ALIGN')

    # Composite score from normalized signals
    company_df['ALIGNMENT_SCORE'] = (
        w_distinctive * company_df['DISTINCTIVE_ALIGN_NORM']
        + w_stance * company_df['STANCE_ALIGN_NORM']
        + w_cosine * company_df['COSINE_ALIGN_NORM']
    )

    logger.info("Composite alignment: %d companies — "
                "mean=%.4f, std=%.4f (distinctive=%.4f, stance=%.4f, cosine=%.4f)",
                len(company_df),
                company_df['ALIGNMENT_SCORE'].mean(),
                company_df['ALIGNMENT_SCORE'].std(),
                company_df['DISTINCTIVE_ALIGN'].mean(),
                company_df['STANCE_ALIGN'].mean(),
                company_df['COSINE_ALIGN'].mean())

    # Step 8: Build event-level scores
    event_rows = []
    if not events_df.empty:
        for _, event in events_df.iterrows():
            ticker = event.get('TICKER')
            event_date = event.get(date_col)
            if ticker is None or pd.isna(event_date):
                continue
            match = company_df[company_df['TICKER'] == ticker]
            if match.empty:
                continue
            m = match.iloc[0]
            event_rows.append({
                'TICKER': ticker,
                'EVENT_DATE': event_date,
                'ALIGNMENT_SCORE': m['ALIGNMENT_SCORE'],
                'DISTINCTIVE_ALIGN': m['DISTINCTIVE_ALIGN'],
                'STANCE_ALIGN': m['STANCE_ALIGN'],
                'COSINE_ALIGN': m['COSINE_ALIGN'],
            })
    event_df = pd.DataFrame(event_rows)

    # Step 9: Classify firms from continuous score (BEFORE validation merge
    # so COMPUTED_LEANING is available in the validation table)
    company_df['COMPUTED_LEANING'] = company_df['ALIGNMENT_SCORE'].apply(
        lambda s: classify_alignment(s, conservative_threshold, liberal_threshold))
    for leaning in ['Conservative', 'Liberal', 'Mixed']:
        n = (company_df['COMPUTED_LEANING'] == leaning).sum()
        logger.info("  Computed leaning: %s = %d", leaning, n)

    # Step 10: Validate against hand-coded leaning
    _agreement_rate = np.nan
    validation = pd.DataFrame()
    if not events_df.empty and 'ESTIMATED_POLITICAL_LEANING' in events_df.columns:
        lean_map = events_df.drop_duplicates('TICKER')[
            ['TICKER', 'ESTIMATED_POLITICAL_LEANING']
        ].copy()
        validation = company_df.merge(lean_map, on='TICKER', how='inner')

        if not validation.empty:
            for leaning in ['Liberal', 'Conservative', 'Mixed']:
                sub = validation[validation['ESTIMATED_POLITICAL_LEANING'] == leaning]
                if not sub.empty:
                    logger.info("  %s (n=%d): composite=%.4f  distinct=%.4f  "
                                "stance=%.4f  cosine=%.4f",
                                leaning, len(sub),
                                sub['ALIGNMENT_SCORE'].mean(),
                                sub['DISTINCTIVE_ALIGN'].mean(),
                                sub['STANCE_ALIGN'].mean(),
                                sub['COSINE_ALIGN'].mean())

            # Agreement rate between computed and hand-coded classification
            if 'COMPUTED_LEANING' in validation.columns:
                _agreement_rate = (validation['COMPUTED_LEANING'] ==
                                   validation['ESTIMATED_POLITICAL_LEANING']).mean()
                logger.info("Classification agreement rate: %.1f%% (%d/%d)",
                            100 * _agreement_rate,
                            int(_agreement_rate * len(validation)),
                            len(validation))

                # Full 3×3 confusion matrix (rows=hand-coded, cols=computed)
                logger.info("  Confusion matrix (rows=hand-coded, cols=computed):")
                for actual in ['Conservative', 'Liberal', 'Mixed']:
                    sub = validation[validation['ESTIMATED_POLITICAL_LEANING'] == actual]
                    row_counts = []
                    for predicted in ['Conservative', 'Liberal', 'Mixed']:
                        n = (sub['COMPUTED_LEANING'] == predicted).sum() if not sub.empty else 0
                        row_counts.append(f"{predicted[:4]}={n:>3}")
                    logger.info("  %-14s → %s", actual, "  ".join(row_counts))

                # Minimum agreement rate check
                if _agreement_rate < _MIN_AGREEMENT_RATE:
                    logger.warning(
                        "Classification agreement rate %.1f%% is below minimum "
                        "threshold %.1f%%. Consider: (1) adjusting thresholds, "
                        "(2) reweighting signals, (3) reviewing hand-coded labels.",
                        100 * _agreement_rate, 100 * _MIN_AGREEMENT_RATE
                    )

    return PoliticalAlignmentResult(
        company_scores=company_df,
        event_scores=event_df,
        distinctive_phrases=phrases_df,
        platform_vocab_size=len(cosine_vectorizer.vocabulary_),
        n_companies=len(company_df),
        validation=validation,
        agreement_rate=_agreement_rate,
        conservative_threshold=conservative_threshold,
        liberal_threshold=liberal_threshold,
        w_distinctive=w_distinctive,
        w_stance=w_stance,
        w_cosine=w_cosine,
    )


def save_alignment_results(
    store: DataStore,
    result: PoliticalAlignmentResult,
) -> dict:
    """Persist political alignment results to the database."""
    saved = {}
    timestamp = pd.Timestamp.now().isoformat()

    # Save run configuration for reproducibility
    config_df = pd.DataFrame([{
        'CONSERVATIVE_THRESHOLD': result.conservative_threshold,
        'LIBERAL_THRESHOLD': result.liberal_threshold,
        'AGREEMENT_RATE': result.agreement_rate,
        'W_DISTINCTIVE': result.w_distinctive,
        'W_STANCE': result.w_stance,
        'W_COSINE': result.w_cosine,
        'N_CONSERVATIVE': (result.company_scores['COMPUTED_LEANING'] == 'Conservative').sum()
            if 'COMPUTED_LEANING' in result.company_scores.columns else 0,
        'N_LIBERAL': (result.company_scores['COMPUTED_LEANING'] == 'Liberal').sum()
            if 'COMPUTED_LEANING' in result.company_scores.columns else 0,
        'N_MIXED': (result.company_scores['COMPUTED_LEANING'] == 'Mixed').sum()
            if 'COMPUTED_LEANING' in result.company_scores.columns else 0,
        'N_COMPANIES': result.n_companies,
        'RUN_TIMESTAMP': timestamp,
    }])
    saved['ESSAY2_ALIGNMENT_CONFIG'] = store.write_table(
        config_df, 'ESSAY2_ALIGNMENT_CONFIG', replace=True)

    if not result.company_scores.empty:
        saved['ESSAY2_POLITICAL_ALIGNMENT'] = store.write_table(
            result.company_scores.assign(RUN_TIMESTAMP=timestamp),
            'ESSAY2_POLITICAL_ALIGNMENT', replace=True,
        )

    if not result.event_scores.empty:
        saved['ESSAY2_EVENT_ALIGNMENT'] = store.write_table(
            result.event_scores.assign(RUN_TIMESTAMP=timestamp),
            'ESSAY2_EVENT_ALIGNMENT', replace=True,
        )

    if not result.distinctive_phrases.empty:
        saved['ESSAY2_DISTINCTIVE_PHRASES'] = store.write_table(
            result.distinctive_phrases.assign(RUN_TIMESTAMP=timestamp),
            'ESSAY2_DISTINCTIVE_PHRASES', replace=True,
        )

    if not result.validation.empty:
        saved['ESSAY2_ALIGNMENT_VALIDATION'] = store.write_table(
            result.validation.assign(RUN_TIMESTAMP=timestamp),
            'ESSAY2_ALIGNMENT_VALIDATION', replace=True,
        )

    logger.info("Alignment results: saved %d tables", len(saved))
    return saved


# ─── NLP Covariate Balance ───────────────────────────────────────────

_NLP_BALANCE_COLS = [
    'NEWS_SENT_PRE',        # pre-event news sentiment — valid for balance
    'FILING_MDA_TONE',      # most recent pre-event MD&A tone — valid
    'FILING_RISK_TONE',     # most recent pre-event risk factors tone — valid
    # NEWS_SENT_POST excluded: post-event outcome, not a pre-treatment covariate
    # NEWS_SENT_CHANGE excluded: post-event outcome, not a pre-treatment covariate
]


def compute_nlp_covariate_balance(
    company_scores: pd.DataFrame,
    event_panel: pd.DataFrame,
    control_tickers: List[str] = None,
) -> pd.DataFrame:
    """
    Standardised mean differences (SMD) of pre-treatment NLP features between
    treatment (Conservative / Liberal) firms and control firms.

    Parameters
    ----------
    control_tickers : list of str, optional
        Matched control tickers from the DiD design. If None, falls back to
        Mixed-leaning firms as a descriptive proxy (not causal balance).

    Returns a DataFrame with one row per covariate per comparison, including
    the SMD, means, pooled SD, and a flag for |SMD| > 0.25 (conventional
    imbalance threshold).
    """
    if 'TICKER' not in event_panel.columns or 'TICKER' not in company_scores.columns:
        logger.warning("Cannot compute covariate balance: TICKER column missing")
        return pd.DataFrame()

    # Aggregate event-level NLP to firm level (mean across events)
    available_cols = [c for c in _NLP_BALANCE_COLS if c in event_panel.columns]
    missing_cols = [c for c in _NLP_BALANCE_COLS if c not in event_panel.columns]
    if missing_cols:
        logger.info("Balance check: columns not in event panel, skipped: %s", missing_cols)
    if not available_cols:
        logger.warning("No NLP balance columns found in event panel")
        return pd.DataFrame()

    firm_nlp = (
        event_panel
        .groupby('TICKER')[available_cols]
        .mean()
        .reset_index()
    )

    # Determine control group and comparison type
    if control_tickers:
        control = firm_nlp[firm_nlp['TICKER'].isin(control_tickers)]
        comparison_type = 'matched_controls'
        control_label = 'Control'
    else:
        # Fallback: Mixed firms as descriptive proxy (not DiD balance)
        merged_tmp = company_scores[['TICKER', 'COMPUTED_LEANING']].merge(
            firm_nlp, on='TICKER', how='inner',
        )
        control = merged_tmp[merged_tmp['COMPUTED_LEANING'] == 'Mixed'][
            ['TICKER'] + available_cols
        ]
        comparison_type = 'mixed_as_proxy'
        control_label = 'Mixed (proxy)'
        logger.warning(
            "No control_tickers supplied — using Mixed as proxy. "
            "This is descriptive only, not DiD pre-treatment balance."
        )

    # Build treatment groups from company scores
    merged = company_scores[['TICKER', 'COMPUTED_LEANING']].merge(
        firm_nlp, on='TICKER', how='inner',
    )

    if merged.empty or 'COMPUTED_LEANING' not in merged.columns:
        logger.warning("No data for covariate balance after merge")
        return pd.DataFrame()

    if control.empty:
        logger.warning("No control firms for covariate balance")
        return pd.DataFrame()

    rows = []

    for treat_label in ('Conservative', 'Liberal'):
        treated = merged[merged['COMPUTED_LEANING'] == treat_label]
        if treated.empty:
            continue

        for col in available_cols:
            t_vals = treated[col].dropna()
            c_vals = control[col].dropna()
            if len(t_vals) < 2 or len(c_vals) < 2:
                continue

            t_mean = t_vals.mean()
            c_mean = c_vals.mean()
            pooled_sd = np.sqrt(
                (t_vals.var(ddof=1) + c_vals.var(ddof=1)) / 2
            )
            smd = (t_mean - c_mean) / pooled_sd if pooled_sd > 0 else np.nan

            rows.append({
                'COMPARISON': f'{treat_label} vs {control_label}',
                'COMPARISON_TYPE': comparison_type,
                'COVARIATE': col,
                'TREAT_MEAN': round(t_mean, 6),
                'CONTROL_MEAN': round(c_mean, 6),
                'POOLED_SD': round(pooled_sd, 6),
                'SMD': round(smd, 4) if not np.isnan(smd) else np.nan,
                'IMBALANCED': abs(smd) > 0.25 if not np.isnan(smd) else None,
                'N_TREAT': len(t_vals),
                'N_CONTROL': len(c_vals),
            })

    result = pd.DataFrame(rows)
    if not result.empty:
        n_imbal = result['IMBALANCED'].sum()
        logger.info(
            "NLP covariate balance: %d covariates checked, %d imbalanced (|SMD|>0.25)",
            len(result), n_imbal,
        )
    return result


# ─── Backward Compatibility ──────────────────────────────────────────

@dataclass
class FactorModelResult:
    """Result from a factor model regression."""
    ticker: str
    model_name: str
    alpha: float
    alpha_t: float
    alpha_p: float
    betas: dict
    r_squared: float
    adj_r_squared: float
    n_obs: int
    residuals: pd.Series = field(repr=False)
    summary: object = field(repr=False, default=None)


def factor_model(
    store: DataStore,
    ticker: str,
    model: str = 'FF5',
    start_date: str = None,
    end_date: str = None,
) -> Optional[FactorModelResult]:
    """
    Estimate a Fama-French factor model for a given ticker.
    """
    returns = store.get_ticker_returns(ticker)
    if returns.empty or 'RETURN' not in returns.columns:
        logger.warning("No return data for %s", ticker)
        return None

    if model == 'FF3':
        factor_cols = ['MKT_RF', 'SMB', 'HML']
        fdata = store.ff3.copy()
    elif model == 'FF5':
        factor_cols = ['MKT_RF', 'SMB', 'HML', 'RMW', 'CMA']
        fdata = store.ff5.copy()
    elif model == 'FF5+MOM':
        factor_cols = ['MKT_RF', 'SMB', 'HML', 'RMW', 'CMA', 'MOM']
        fdata = store.factors.copy()
        missing = [c for c in factor_cols if c not in fdata.columns]
        if missing:
            logger.error("FF5+MOM model requires columns %s but store.factors "
                         "is missing %s", factor_cols, missing)
            return None
    else:
        raise ValueError(f"Unknown model: {model}. Use 'FF3', 'FF5', or 'FF5+MOM'.")

    if fdata.empty:
        logger.warning("No factor data available for model %s", model)
        return None

    for col in factor_cols + ['RF']:
        if col in fdata.columns and fdata[col].abs().max() > 1:
            fdata[col] = fdata[col] / 100

    merged = returns[['DATE', 'RETURN']].merge(fdata, on='DATE', how='inner')

    if start_date:
        merged = merged[merged['DATE'] >= pd.Timestamp(start_date)]
    if end_date:
        merged = merged[merged['DATE'] <= pd.Timestamp(end_date)]

    merged = merged.dropna(subset=['RETURN'] + factor_cols + ['RF'])
    if len(merged) < 30:
        logger.warning("Insufficient observations for %s (%d)", ticker, len(merged))
        return None

    merged['EXCESS_RETURN'] = merged['RETURN'] - merged['RF']

    y = merged['EXCESS_RETURN']
    X = sm.add_constant(merged[factor_cols], has_constant='add')

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        result = sm.OLS(y, X).fit(cov_type='HC1')

    betas = {col: result.params[col] for col in factor_cols}

    return FactorModelResult(
        ticker=ticker,
        model_name=model,
        alpha=result.params['const'],
        alpha_t=result.tvalues['const'],
        alpha_p=result.pvalues['const'],
        betas=betas,
        r_squared=result.rsquared,
        adj_r_squared=result.rsquared_adj,
        n_obs=result.nobs,
        residuals=pd.Series(result.resid.values, index=merged['DATE'].values),
        summary=result,
    )


# =========================================================================
# MAIN
# =========================================================================

if __name__ == '__main__':
    import argparse
    from datetime import datetime
    from pathlib import Path as _Path

    parser = argparse.ArgumentParser(description='Essay 2 — NLP + Political Alignment')
    parser.add_argument('--conservative-threshold', type=float, default=0.05,
                        help='Alignment score above this → Conservative (default: 0.05)')
    parser.add_argument('--liberal-threshold', type=float, default=-0.05,
                        help='Alignment score below this → Liberal (default: -0.05)')
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
    )

    print("Dissertation Essay 2 — NLP + Political Alignment")
    print(f"Started at {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print()

    store = DataStore()
    _backend = getattr(store, 'backend', 'sqlite')
    print(f"  Backend: {_backend}")
    if _backend != 'aws':
        print("  WARNING: Not connected to AWS — results will only be saved locally.")
        print("  To upload to AWS, ensure AWS_PROFILE is set and credentials are valid.")

    # ── Step 1: Load pre-built filing text from clean.py ──
    print("=" * 60)
    print("  Step 1: Load filing text (from clean.py pipeline)")
    print("=" * 60)

    filing_path = _Path(__file__).resolve().parent.parent / 'sec_filings_data' / 'sec_filing_text_2010-01-01_to_2025-12-31.csv'
    filing_sections_df = pd.DataFrame()
    if filing_path.exists():
        filing_sections_df = pd.read_csv(filing_path)
        filing_sections_df.columns = [c.upper() for c in filing_sections_df.columns]
        print(f"  Loaded {len(filing_sections_df)} filing sections from {filing_path.name}")
        print(f"  Tickers: {filing_sections_df['TICKER'].nunique()}")
        print(f"  Sections: {filing_sections_df['SECTION'].value_counts().to_dict()}")
    else:
        print(f"  WARNING: Filing text not found at {filing_path}")
        print("  Run clean.py first: clean.sec_filings.build_filing_text_dataset()")

    # ── Step 2: NLP analysis (news + filing sentiment) ──
    print()
    print("=" * 60)
    print("  Step 2: NLP analysis (news sentiment + filing sentiment)")
    print("=" * 60)

    nlp_result = run_nlp_analysis(
        store,
        download_filings=False,
        filing_sections_df=filing_sections_df if not filing_sections_df.empty else None,
    )

    if nlp_result is not None:
        print(f"  News articles scored: {nlp_result.n_articles_scored}")
        print(f"  Filing sections scored: {nlp_result.n_filings_scored}")
        if not nlp_result.event_nlp_panel.empty:
            print(f"  Event NLP panel: {len(nlp_result.event_nlp_panel)} rows, "
                  f"{nlp_result.event_nlp_panel['TICKER'].nunique()} tickers")

        saved = save_nlp_results(store, nlp_result)
        for table, res in saved.items():
            print(f"    {table}: {res}")
        print(f"  Saved {len(saved)} tables")
    else:
        print("  FAILED — no NLP results")

    # ── Step 3: Political alignment (3-signal: distinctive + stance + cosine) ──
    print()
    print("=" * 60)
    print("  Step 3: Political alignment scoring")
    print("  (distinctive phrases + stance + cosine similarity)")
    print("=" * 60)

    news_scored = nlp_result.news_sentiment if nlp_result else None
    alignment_result = compute_political_alignment(
        store,
        news_scored=news_scored,
        filing_sections=filing_sections_df if not filing_sections_df.empty else None,
        run_stance=True,
        conservative_threshold=args.conservative_threshold,
        liberal_threshold=args.liberal_threshold,
    )
    print(f"  Thresholds: conservative > {args.conservative_threshold}, "
          f"liberal < {args.liberal_threshold}")

    if alignment_result is not None:
        # Company-level scores
        cs = alignment_result.company_scores
        if not cs.empty:
            print(f"  Companies scored: {len(cs)}")
            print(f"  Score range: [{cs['ALIGNMENT_SCORE'].min():.3f}, "
                  f"{cs['ALIGNMENT_SCORE'].max():.3f}]")
            print(f"  Mean: {cs['ALIGNMENT_SCORE'].mean():.3f}, "
                  f"Std: {cs['ALIGNMENT_SCORE'].std():.3f}")

            # Top 5 most R-aligned and D-aligned
            sorted_cs = cs.sort_values('ALIGNMENT_SCORE')
            print()
            print("  Most D-aligned:")
            for _, row in sorted_cs.head(5).iterrows():
                print(f"    {row['TICKER']:<8} {row['ALIGNMENT_SCORE']:+.3f}")
            print("  Most R-aligned:")
            for _, row in sorted_cs.tail(5).iterrows():
                print(f"    {row['TICKER']:<8} {row['ALIGNMENT_SCORE']:+.3f}")

        # Distinctive phrases
        if not alignment_result.distinctive_phrases.empty:
            dp = alignment_result.distinctive_phrases
            n_d = len(dp[dp['PARTY'] == 'Democratic'])
            n_r = len(dp[dp['PARTY'] == 'Republican'])
            print(f"\n  Distinctive phrases: {n_d} Democratic, {n_r} Republican")

        # Validation against hand-coded leaning
        if not alignment_result.validation.empty:
            val = alignment_result.validation
            print(f"\n  --- Validation vs Hand-Coded Leaning ---")
            for leaning in ['Liberal', 'Conservative', 'Mixed']:
                sub = val[val['ESTIMATED_POLITICAL_LEANING'] == leaning]
                if not sub.empty:
                    print(f"    {leaning:<14} n={len(sub):>3}  "
                          f"mean={sub['ALIGNMENT_SCORE'].mean():+.4f}  "
                          f"median={sub['ALIGNMENT_SCORE'].median():+.4f}")

        # Agreement rate (from dataclass — single source of truth)
        if not np.isnan(alignment_result.agreement_rate):
            val = alignment_result.validation
            print(f"\n  Classification agreement rate: {alignment_result.agreement_rate:.1%} "
                  f"({int(alignment_result.agreement_rate * len(val))}/{len(val)})")

            # Print full confusion matrix
            print(f"  Confusion matrix (rows=hand-coded, cols=computed):")
            for actual in ['Conservative', 'Liberal', 'Mixed']:
                sub = val[val['ESTIMATED_POLITICAL_LEANING'] == actual]
                row_counts = []
                for predicted in ['Conservative', 'Liberal', 'Mixed']:
                    n = (sub['COMPUTED_LEANING'] == predicted).sum() if not sub.empty else 0
                    row_counts.append(f"{predicted[:4]}={n:>3}")
                print(f"    {actual:<14} → {'  '.join(row_counts)}")

        # Computed leaning distribution
        if 'COMPUTED_LEANING' in cs.columns:
            print(f"\n  --- Computed Leaning Distribution ---")
            for leaning in ['Conservative', 'Liberal', 'Mixed']:
                n = (cs['COMPUTED_LEANING'] == leaning).sum()
                print(f"    {leaning:<14} n={n}")

        # Threshold sensitivity analysis
        if 'ALIGNMENT_SCORE' in cs.columns:
            sens_df = threshold_sensitivity_table(cs)

            print(f"\n  --- Threshold Sensitivity (Table) ---")
            header = (f"  {'tau':>5}  {'N_C':>4}  {'N_L':>4}  {'N_M':>4}  "
                      f"{'%C':>5}  {'%L':>5}  {'%M':>5}  "
                      f"{'MeanC':>7}  {'MeanL':>7}  {'MeanM':>7}  "
                      f"{'d':>5}")
            print(header)
            print("  " + "-" * (len(header) - 2))
            for _, r in sens_df.iterrows():
                print(f"  {r['TAU']:>5.2f}  {int(r['N_CONSERVATIVE']):>4}  "
                      f"{int(r['N_LIBERAL']):>4}  {int(r['N_MIXED']):>4}  "
                      f"{r['PCT_CONSERVATIVE']:>5.1f}  {r['PCT_LIBERAL']:>5.1f}  "
                      f"{r['PCT_MIXED']:>5.1f}  "
                      f"{r['MEAN_SCORE_C']:>+7.3f}  {r['MEAN_SCORE_L']:>+7.3f}  "
                      f"{r['MEAN_SCORE_M']:>+7.3f}  "
                      f"{r['COHENS_D']:>5.2f}")

            # Save sensitivity table
            saved_sens = store.write_table(
                sens_df, 'ESSAY2_THRESHOLD_SENSITIVITY', replace=True)
            print(f"\n  Saved ESSAY2_THRESHOLD_SENSITIVITY: {saved_sens}")

            # Skew detection: suggest asymmetric thresholds if needed.
            median_score = cs['ALIGNMENT_SCORE'].median()
            logger.info("Alignment distribution median: %+.4f (n=%d)",
                        median_score, len(cs))
            if abs(median_score) > 0.05:
                suggested_c = args.conservative_threshold + median_score
                suggested_l = args.liberal_threshold + median_score
                logger.warning(
                    "Alignment distribution skewed (median=%+.4f, p05=%.4f, p95=%.4f) "
                    "— consider asymmetric thresholds: Conservative > %+.3f, "
                    "Liberal < %+.3f",
                    median_score,
                    cs['ALIGNMENT_SCORE'].quantile(0.05),
                    cs['ALIGNMENT_SCORE'].quantile(0.95),
                    suggested_c, suggested_l)
                print(f"\n  Distribution skewed (median={median_score:+.4f}). "
                      f"Confusion matrix above used symmetric thresholds "
                      f"(+/-{args.conservative_threshold}).")
                print(f"  Re-run with: --conservative-threshold {suggested_c:+.3f} "
                      f"--liberal-threshold {suggested_l:+.3f}")

        # Event-level scores
        if not alignment_result.event_scores.empty:
            es = alignment_result.event_scores
            print(f"\n  Event-level scores: {len(es)} events")

        # Save
        saved = save_alignment_results(store, alignment_result)
        for table, res in saved.items():
            print(f"    {table}: {res}")
        print(f"  Saved {len(saved)} tables")

        # ── Step 3b: NLP covariate balance ──
        event_panel = nlp_result.event_nlp_panel if nlp_result else pd.DataFrame()
        if not event_panel.empty and 'COMPUTED_LEANING' in cs.columns:
            print()
            print("  --- NLP Covariate Balance (Pre-Treatment Features) ---")

            # Use matched control tickers if available (from essay1_matched DiD design)
            _ctrl_tickers = None
            try:
                _ctrl_tickers = list(store.get_control_tickers() or [])
                if _ctrl_tickers:
                    print(f"  Using {len(_ctrl_tickers)} matched control tickers")
                else:
                    _ctrl_tickers = None
            except Exception:
                pass
            if not _ctrl_tickers:
                print("  NOTE: No matched controls — using Mixed firms as descriptive proxy")

            balance_df = compute_nlp_covariate_balance(cs, event_panel, _ctrl_tickers)
            if not balance_df.empty:
                for _, row in balance_df.iterrows():
                    flag = " ***" if row['IMBALANCED'] else ""
                    print(f"    {row['COMPARISON']:<25} {row['COVARIATE']:<20} "
                          f"SMD={row['SMD']:+.3f}  (T={row['N_TREAT']}, C={row['N_CONTROL']}){flag}")
                store.write_table(balance_df, 'ESSAY2_NLP_COVARIATE_BALANCE', replace=True)
                print(f"  Saved ESSAY2_NLP_COVARIATE_BALANCE ({len(balance_df)} rows)")
            else:
                print("  No covariate balance results (insufficient data)")
    else:
        print("  FAILED — no alignment results")

    # ── Step 4: Upload results to AWS (S3 + Glue) ──
    # Only needed if store is SQLite — if store is already AWS, writes went there.
    print()
    print("=" * 60)
    print("  Step 4: Upload results to AWS")
    print("=" * 60)

    if _backend == 'aws':
        print("  Skipped — store already writing to AWS via DataStore backend")
    else:
        # Collect all ESSAY2 table names written during this run
        _essay2_tables = [
            'ESSAY2_NEWS_SENTIMENT', 'ESSAY2_FILING_SENTIMENT', 'ESSAY2_EVENT_NLP',
            'ESSAY2_POLITICAL_ALIGNMENT', 'ESSAY2_EVENT_ALIGNMENT',
            'ESSAY2_DISTINCTIVE_PHRASES', 'ESSAY2_ALIGNMENT_VALIDATION',
            'ESSAY2_ALIGNMENT_CONFIG', 'ESSAY2_NLP_COVARIATE_BALANCE',
        ]

        try:
            from Database import AthenaLoader
        except ImportError:
            print("  AWS upload skipped — Database module not available.")
        else:
            try:
                aws_loader = AthenaLoader()
                aws_loader.connect()
                print(f"  Connected to AWS ({aws_loader.database} / {aws_loader.s3_bucket})")

                n_uploaded = 0
                for table_name in _essay2_tables:
                    try:
                        df = store.read_table(table_name)
                        if df.empty:
                            continue
                        res = aws_loader.write_table(df, table_name, replace=True)
                        status = res.get('status', 'UNKNOWN')
                        rows = res.get('rows', len(df))
                        print(f"    {table_name}: {status} ({rows} rows)")
                        n_uploaded += 1
                    except Exception as e:
                        print(f"    {table_name}: FAILED — {e}")

                aws_loader.close()
                print(f"  Uploaded {n_uploaded} tables to AWS")
            except Exception as e:
                print(f"  AWS upload failed: {type(e).__name__}: {e}")
                print("  Results are saved locally in SQLite.")

    store.close()
    print()
    print(f"Completed at {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
