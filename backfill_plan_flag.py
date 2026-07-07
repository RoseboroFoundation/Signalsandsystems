"""Backfill 10b5-1 plan flag for post-2023 Form 4 filings.

The SEC amendment (effective 2023-02-27) added <transactionPlanAdoption>
to Form 4 XML, indicating whether a trade was made pursuant to a Rule 10b5-1(c)
plan.  Our existing CSV predates this extraction.

SPEED STRATEGY (1 request per filing instead of 3):
  Most filing URLs in the CSV are XSLT-rendered:
    .../Archives/edgar/data/{cik}/{acc}/xslF345X*/filename.xml
  The raw Form 4 XML is at the same path without the xslF345X*/ component:
    .../Archives/edgar/data/{cik}/{acc}/filename.xml
  We construct this URL directly and fetch in one request.
  Fallback to the full parse_form4_xml index-page approach if that fails.

Usage:
    cd /Users/administrator/Projects/signalsandsystems
    .venv/bin/python backfill_plan_flag.py
"""

import logging
import os
import re
import shutil
import sys
import time

import pandas as pd
import requests
from bs4 import BeautifulSoup

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s %(levelname)s: %(message)s',
    datefmt='%H:%M:%S',
)
logger = logging.getLogger(__name__)

FORM4_DIR = './sec_form4_data'
COMBINED_CSV = os.path.join(
    FORM4_DIR, 'form4_transactions_2000-01-01_to_2025-12-31.csv'
)
CUTOFF_DATE = pd.Timestamp('2023-02-27')
MATCH_COLS = ['accession_number', 'transaction_date', 'transaction_code', 'shares']

SEC_USER_AGENT = os.getenv('SEC_USER_AGENT', 'SignalsAndSystems/1.0')

# ── Helpers ───────────────────────────────────────────────────────────────────

_XSLT_RE = re.compile(r'/xslF345X\w+/', re.IGNORECASE)


def _direct_xml_url(filing_url: str) -> str | None:
    """Strip the xslF345X*/ directory from the URL to get the raw XML URL.

    e.g. .../000123/xslF345X05/doc4.xml  →  .../000123/doc4.xml
    Returns None if the URL doesn't match the XSLT pattern.
    """
    if _XSLT_RE.search(filing_url):
        return _XSLT_RE.sub('/', filing_url)
    return None


def _headers(url: str) -> dict:
    host = 'data.sec.gov' if 'data.sec.gov' in url else 'www.sec.gov'
    return {
        'User-Agent': SEC_USER_AGENT,
        'Accept-Encoding': 'gzip, deflate',
        'Host': host,
    }


_PLAN_RE = re.compile(r'10b5[\-\.]1|rule\s+10b5', re.IGNORECASE)


def _extract_plan_flag_from_xml(xml_bytes: bytes) -> list[dict]:
    """Parse raw Form 4 XML bytes and return a list of transaction dicts
    containing only the fields needed for the plan-flag join.

    The SEC's 2023 amendment added a Rule 10b5-1(c) plan checkbox to the
    paper Form 4, but the EDGAR XML schema has no structured field for it.
    Disclosures appear in free-text footnotes referenced by <footnoteId>
    elements inside each transaction block.  We detect 10b5-1 plan trades
    by checking whether any footnote linked to a transaction mentions
    '10b5-1' or 'rule 10b5-1'.  Absent any such footnote = False (not a
    plan trade), since post-2023 filers are required to disclose plans.
    """
    soup = BeautifulSoup(xml_bytes, 'lxml-xml')
    if not (soup.find('ownershipDocument') or soup.find('reportingOwner')):
        return []

    # Build footnote id → text map
    footnote_map: dict[str, str] = {}
    for fn in soup.find_all('footnote'):
        fid = fn.get('id')
        if fid:
            footnote_map[fid] = fn.get_text(' ', strip=True)

    records = []
    for trans in soup.find_all('nonDerivativeTransaction'):
        try:
            date_tag   = trans.find('transactionDate')
            code_tag   = trans.find('transactionCode')
            shares_tag = trans.find('transactionShares')

            td = None
            if date_tag and date_tag.find('value'):
                try:
                    td = pd.to_datetime(date_tag.find('value').text, errors='coerce')
                except Exception:
                    pass

            code   = code_tag.text.strip() if code_tag else None
            shares = None
            if shares_tag and shares_tag.find('value'):
                try:
                    shares = float(shares_tag.find('value').text)
                except Exception:
                    pass

            # Check referenced footnotes for 10b5-1 plan language
            fn_ids = [f.get('id') for f in trans.find_all('footnoteId') if f.get('id')]
            is_plan = False
            for fid in fn_ids:
                txt = footnote_map.get(fid, '')
                if _PLAN_RE.search(txt):
                    is_plan = True
                    break

            records.append({
                'transaction_date':  td,
                'transaction_code':  code,
                'shares':            shares,
                'is_10b5_1_plan':    is_plan,
                'plan_adoption_date': None,   # not in XML; footnote is sufficient
            })
        except Exception:
            continue
    return records


def _fetch_and_parse(filing_url: str, acc: str) -> list[dict] | None:
    """Fetch a filing and return per-transaction plan-flag records, or None on failure.

    Tries the direct raw-XML URL first (1 request).  Falls back to the
    XSLT-rendered URL + index-page lookup (2 more requests) if needed.
    """
    session = requests.Session()

    # ── Strategy 1: direct raw XML ──────────────────────────────────────────
    direct = _direct_xml_url(filing_url)
    if direct:
        try:
            r = session.get(direct, headers=_headers(direct), timeout=20)
            time.sleep(0.1)
            if r.status_code == 200 and b'<ownershipDocument' in r.content:
                return _extract_plan_flag_from_xml(r.content)
        except Exception:
            pass

    # ── Strategy 2: try the original URL as-is (sometimes IS raw XML) ───────
    try:
        r = session.get(filing_url, headers=_headers(filing_url), timeout=20)
        time.sleep(0.1)
        if r.status_code == 200 and b'<ownershipDocument' in r.content:
            return _extract_plan_flag_from_xml(r.content)
    except Exception:
        pass

    # ── Strategy 3: fetch the filing index page, find the .xml link ─────────
    # Derive index URL from the accession number in the filing URL
    acc_clean = acc.replace('-', '')
    m = re.search(r'Archives/edgar/data/(\d+)/', filing_url)
    if not m:
        return None
    cik_num = m.group(1)
    index_url = (
        f"https://www.sec.gov/Archives/edgar/data/{cik_num}/{acc_clean}/"
    )
    try:
        ir = session.get(index_url, headers=_headers(index_url), timeout=20)
        time.sleep(0.1)
        if ir.status_code != 200:
            return None
        idx_soup = BeautifulSoup(ir.text, 'html.parser')
        xml_link = None
        for link in idx_soup.find_all('a'):
            href = link.get('href', '')
            if (href.endswith('.xml')
                    and 'xsl' not in href.lower()
                    and 'primary_doc' not in href):
                xml_link = (
                    'https://www.sec.gov' + href
                    if not href.startswith('http') else href
                )
                break
        if not xml_link:
            return None
        xr = session.get(xml_link, headers=_headers(xml_link), timeout=20)
        time.sleep(0.1)
        if xr.status_code == 200 and b'<ownershipDocument' in xr.content:
            return _extract_plan_flag_from_xml(xr.content)
    except Exception:
        pass

    return None


# ── Main ──────────────────────────────────────────────────────────────────────

def backfill():
    # 1. Load combined CSV
    logger.info("Loading %s …", COMBINED_CSV)
    df = pd.read_csv(COMBINED_CSV, low_memory=False)
    df['filing_date']      = pd.to_datetime(df['filing_date'],      errors='coerce')
    df['transaction_date'] = pd.to_datetime(df['transaction_date'], errors='coerce')
    df['shares']           = pd.to_numeric(df['shares'],            errors='coerce')

    total_rows = len(df)
    logger.info("  %d total rows", total_rows)

    if 'is_10b5_1_plan'    not in df.columns: df['is_10b5_1_plan']    = None
    if 'plan_adoption_date' not in df.columns: df['plan_adoption_date'] = None

    # 2. Identify post-cutoff filings
    post = df[df['filing_date'] >= CUTOFF_DATE]
    unique_filings = (
        post[['accession_number', 'filing_url', 'ticker']]
        .drop_duplicates(subset=['accession_number'])
        .reset_index(drop=True)
    )
    logger.info(
        "  %d rows >= %s across %d unique filings (%d tickers)",
        len(post), CUTOFF_DATE.date(), len(unique_filings),
        unique_filings['ticker'].nunique(),
    )

    # 3. Re-parse each filing (fast path: direct XML URL)
    plan_records = []
    n_ok = n_fail = n_empty = 0
    t0 = time.time()

    for i, row in unique_filings.iterrows():
        acc    = row['accession_number']
        url    = row['filing_url']
        ticker = row['ticker']

        result = _fetch_and_parse(url, acc)

        if result is None:
            n_fail += 1
        elif len(result) == 0:
            n_empty += 1
        else:
            n_ok += 1
            for rec in result:
                rec['accession_number'] = acc
                plan_records.append(rec)

        if (i + 1) % 500 == 0:
            elapsed = time.time() - t0
            rate    = (i + 1) / elapsed
            eta_min = (len(unique_filings) - i - 1) / rate / 60
            logger.info(
                "  %d/%d filings (ok=%d fail=%d empty=%d) | %.1f/min | ETA %.0f min",
                i + 1, len(unique_filings), n_ok, n_fail, n_empty, rate * 60, eta_min,
            )

    elapsed_total = time.time() - t0
    logger.info(
        "Parsing done in %.1f min: %d ok, %d failed, %d empty, %d plan records",
        elapsed_total / 60, n_ok, n_fail, n_empty, len(plan_records),
    )

    if not plan_records:
        logger.error("No plan records extracted — aborting.")
        return

    # 4. Merge flags back
    plan_df = pd.DataFrame(plan_records)
    plan_df['transaction_date'] = pd.to_datetime(plan_df['transaction_date'], errors='coerce')
    plan_df['shares']           = pd.to_numeric(plan_df['shares'], errors='coerce')
    plan_df = plan_df.drop_duplicates(subset=MATCH_COLS, keep='first')

    df_post_idx = df.index[df['filing_date'] >= CUTOFF_DATE]
    df_post     = df.loc[df_post_idx, MATCH_COLS].copy()
    df_post['transaction_date'] = pd.to_datetime(df_post['transaction_date'], errors='coerce')

    merged = df_post.merge(
        plan_df[MATCH_COLS + ['is_10b5_1_plan', 'plan_adoption_date']],
        on=MATCH_COLS, how='left',
    )
    merged.index = df_post_idx

    df.loc[df_post_idx, 'is_10b5_1_plan']    = merged['is_10b5_1_plan']
    df.loc[df_post_idx, 'plan_adoption_date'] = merged['plan_adoption_date']

    n_true  = (df['is_10b5_1_plan'] == True).sum()   # noqa: E712
    n_false = (df['is_10b5_1_plan'] == False).sum()  # noqa: E712
    n_none  = df['is_10b5_1_plan'].isna().sum()
    logger.info(
        "Flag totals: %d True (in plan) / %d False (no plan) / %d None (unknown/pre-2023)",
        n_true, n_false, n_none,
    )

    # 5. Write
    bak = COMBINED_CSV + '.bak'
    shutil.copy2(COMBINED_CSV, bak)
    logger.info("Backed up original → %s", bak)
    df.to_csv(COMBINED_CSV, index=False)
    logger.info("Saved updated combined CSV (%d rows)", len(df))

    # 6. Patch per-ticker CSVs
    tickers_affected = unique_filings['ticker'].unique()
    patched = 0
    for ticker in tickers_affected:
        tf = os.path.join(FORM4_DIR, f'form4_{ticker}.csv')
        if not os.path.exists(tf):
            continue
        try:
            tdf = pd.read_csv(tf, low_memory=False)
            tdf['filing_date']      = pd.to_datetime(tdf['filing_date'],      errors='coerce')
            tdf['transaction_date'] = pd.to_datetime(tdf['transaction_date'], errors='coerce')
            tdf['shares']           = pd.to_numeric(tdf['shares'],            errors='coerce')
            if 'is_10b5_1_plan'    not in tdf.columns: tdf['is_10b5_1_plan']    = None
            if 'plan_adoption_date' not in tdf.columns: tdf['plan_adoption_date'] = None

            t_post_idx = tdf.index[tdf['filing_date'] >= CUTOFF_DATE]
            if len(t_post_idx) == 0:
                continue
            t_post = tdf.loc[t_post_idx, MATCH_COLS].copy()
            t_post['transaction_date'] = pd.to_datetime(
                t_post['transaction_date'], errors='coerce')
            t_merged = t_post.merge(
                plan_df[MATCH_COLS + ['is_10b5_1_plan', 'plan_adoption_date']],
                on=MATCH_COLS, how='left',
            )
            t_merged.index = t_post_idx
            tdf.loc[t_post_idx, 'is_10b5_1_plan']    = t_merged['is_10b5_1_plan']
            tdf.loc[t_post_idx, 'plan_adoption_date'] = t_merged['plan_adoption_date']
            tdf.to_csv(tf, index=False)
            patched += 1
        except Exception as e:
            logger.warning("  Could not patch %s: %s", tf, e)

    logger.info("Patched %d per-ticker CSVs. Done.", patched)


if __name__ == '__main__':
    backfill()
