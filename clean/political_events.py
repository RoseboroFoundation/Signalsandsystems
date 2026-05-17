"""Political fundamental events — congressional votes, executive orders, court decisions.

Data sources:
  - GovTrack API v2 (congressional roll-call votes)
  - Federal Register API (executive orders)
  - Supreme Court Database CSV (court decisions)

Returns a unified DataFrame of political events with policy area classification
and NAICS sector mapping for treatment assignment in Essay 3.
"""

import json
import os
import time
from datetime import datetime

import pandas as pd
import requests

from .config import logger

# ═══════════════════════════════════════════════════════════════════════
# POLICY AREA MAPPINGS
# ═══════════════════════════════════════════════════════════════════════

POLICY_AREA_TO_NAICS = {
    'healthcare':      ['621', '622', '623', '624', '325'],
    'finance':         ['521', '522', '523', '524'],
    'energy':          ['211', '213', '221', '324', '333'],
    'technology':      ['334', '511', '517', '518', '519'],
    'defense':         ['332', '336', '541'],
    'agriculture':     ['111', '112', '311', '312'],
    'transportation':  ['481', '482', '483', '484', '485', '486'],
    'environment':     ['221', '562'],
    'labor':           [],  # Broad market impact
    'trade':           [],
    'tax':             [],
    'education':       ['611'],
    'housing':         ['236', '531', '522'],
    'immigration':     [],
    'antitrust':       [],
    'unknown':         [],
}

# GovTrack CRS bill subjects → standardized policy area
CRS_TO_POLICY_AREA = {
    'Health': 'healthcare',
    'Finance and Financial Sector': 'finance',
    'Energy': 'energy',
    'Science, Technology, Communications': 'technology',
    'Armed Forces and National Security': 'defense',
    'Agriculture and Food': 'agriculture',
    'Transportation and Public Works': 'transportation',
    'Environmental Protection': 'environment',
    'Labor and Employment': 'labor',
    'Foreign Trade and International Finance': 'trade',
    'Taxation': 'tax',
    'Education': 'education',
    'Housing and Community Development': 'housing',
    'Immigration': 'immigration',
    'Commerce': 'antitrust',
    'Economics and Public Finance': 'finance',
    'Social Welfare': 'labor',
    'Crime and Law Enforcement': 'unknown',
    'Government Operations and Politics': 'unknown',
    'International Affairs': 'trade',
    'Public Lands and Natural Resources': 'environment',
    'Water Resources Development': 'environment',
    'Native Americans': 'unknown',
    'Sports and Recreation': 'unknown',
    'Animals': 'agriculture',
    'Civil Rights and Liberties, Minority Issues': 'labor',
    'Families': 'labor',
    'Emergency Management': 'defense',
    'Law': 'unknown',
    'Congress': 'unknown',
}

# Supreme Court Database issueArea codes → policy area
SCDB_ISSUE_TO_POLICY = {
    1: 'unknown',      # Criminal Procedure
    2: 'labor',        # Civil Rights
    3: 'unknown',      # First Amendment
    4: 'unknown',      # Due Process
    5: 'unknown',      # Privacy
    6: 'unknown',      # Attorneys
    7: 'labor',        # Unions
    8: 'finance',      # Economic Activity
    9: 'unknown',      # Judicial Power
    10: 'tax',         # Federalism
    11: 'trade',       # Interstate Relations
    12: 'tax',         # Federal Taxation
    13: 'environment', # Miscellaneous
    14: 'unknown',     # Private Action
}

# Keyword patterns for classifying event descriptions
_POLICY_KEYWORDS = {
    'healthcare': ['health', 'medical', 'medicare', 'medicaid', 'hospital',
                   'pharmaceutical', 'drug', 'fda', 'affordable care',
                   'insurance coverage', 'patient', 'vaccine', 'pandemic'],
    'finance': ['bank', 'financial', 'securities', 'dodd-frank', 'wall street',
                'credit', 'mortgage', 'lending', 'federal reserve', 'fdic',
                'sec ', 'stock market', 'derivatives', 'bailout'],
    'energy': ['energy', 'oil', 'gas', 'pipeline', 'drilling', 'coal',
               'nuclear', 'renewable', 'solar', 'wind', 'emission',
               'petroleum', 'fracking', 'keystone', 'opec'],
    'technology': ['technology', 'internet', 'cyber', 'broadband', 'telecom',
                   'spectrum', 'privacy', 'data protection', 'net neutrality',
                   'artificial intelligence', 'semiconductor', 'chip'],
    'defense': ['defense', 'military', 'armed forces', 'veteran', 'pentagon',
                'weapon', 'missile', 'nato', 'national security', 'homeland'],
    'agriculture': ['agriculture', 'farm', 'crop', 'food safety', 'usda',
                    'livestock', 'subsid', 'ethanol', 'organic'],
    'transportation': ['transportation', 'highway', 'aviation', 'railroad',
                       'transit', 'infrastructure', 'bridge', 'port',
                       'shipping', 'airline', 'faa', 'dot '],
    'environment': ['environment', 'climate', 'clean air', 'clean water',
                    'epa', 'pollution', 'carbon', 'endangered', 'conservation',
                    'toxic', 'superfund', 'paris agreement'],
    'labor': ['labor', 'minimum wage', 'worker', 'union', 'employment',
              'workplace', 'osha', 'overtime', 'pension', 'retirement'],
    'trade': ['trade', 'tariff', 'import', 'export', 'nafta', 'wto',
              'sanction', 'embargo', 'customs', 'tpp', 'usmca'],
    'tax': ['tax', 'irs', 'deduction', 'revenue', 'fiscal', 'budget',
            'deficit', 'appropriation', 'spending bill'],
    'education': ['education', 'school', 'student', 'college', 'university',
                  'teacher', 'title ix', 'student loan', 'pell grant'],
    'housing': ['housing', 'mortgage', 'hud', 'fannie mae', 'freddie mac',
                'home ownership', 'rent', 'affordable housing', 'foreclosure'],
    'antitrust': ['antitrust', 'monopoly', 'merger', 'competition',
                  'ftc', 'market concentration', 'price fixing'],
}


# ═══════════════════════════════════════════════════════════════════════
# CLASSIFIERS
# ═══════════════════════════════════════════════════════════════════════

def classify_policy_area(text, subjects=None):
    """Classify a political event into a policy area using keywords + CRS subjects."""
    if subjects:
        for subj in subjects:
            area = CRS_TO_POLICY_AREA.get(subj)
            if area and area != 'unknown':
                return area

    if not text:
        return 'unknown'

    text_lower = text.lower()
    scores = {}
    for area, keywords in _POLICY_KEYWORDS.items():
        score = sum(1 for kw in keywords if kw in text_lower)
        if score > 0:
            scores[area] = score

    if scores:
        return max(scores, key=scores.get)
    return 'unknown'


# ═══════════════════════════════════════════════════════════════════════
# DATA FETCHERS
# ═══════════════════════════════════════════════════════════════════════

def _api_get(url, params=None, retries=3, delay=1.0):
    """GET with retry and rate limiting."""
    for attempt in range(retries):
        try:
            resp = requests.get(url, params=params, timeout=30)
            if resp.status_code == 429:
                wait = delay * (2 ** attempt)
                logger.warning("Rate limited, waiting %.1fs", wait)
                time.sleep(wait)
                continue
            resp.raise_for_status()
            return resp.json()
        except requests.RequestException as e:
            if attempt < retries - 1:
                time.sleep(delay)
            else:
                logger.warning("API request failed after %d retries: %s", retries, e)
                return None
    return None


def fetch_congressional_votes(start_date, end_date, cache_dir):
    """Fetch roll-call votes from GovTrack API.

    Filters to final passage votes only (category=passage).
    """
    raw_dir = os.path.join(cache_dir, 'raw', 'votes')
    os.makedirs(raw_dir, exist_ok=True)

    cache_file = os.path.join(raw_dir, 'congressional_votes.json')
    if os.path.exists(cache_file):
        age_days = (time.time() - os.path.getmtime(cache_file)) / 86400
        if age_days < 7:
            logger.info("  Using cached congressional votes (%.1f days old)", age_days)
            with open(cache_file) as f:
                return json.load(f)

    logger.info("  Fetching congressional votes from GovTrack API...")
    base_url = 'https://www.govtrack.us/api/v2/vote'

    # Paginate in yearly chunks to avoid API result limits
    all_votes = []
    start_year = int(start_date[:4])
    end_year = int(end_date[:4])

    for year in range(start_year, end_year + 1):
        year_start = f'{year}-01-01'
        year_end = f'{year}-12-31'
        offset = 0
        limit = 100
        year_count = 0

        while True:
            params = {
                'created__gte': year_start,
                'created__lte': year_end,
                'order_by': 'created',
                'limit': limit,
                'offset': offset,
            }
            data = _api_get(base_url, params=params)
            if data is None:
                break

            objects = data.get('objects', [])
            if not objects:
                break

            all_votes.extend(objects)
            year_count += len(objects)
            offset += limit

            if offset >= data.get('meta', {}).get('total_count', 0):
                break

            time.sleep(0.5)

        if year_count > 0:
            logger.info("    %d: %d votes (running total: %d)",
                        year, year_count, len(all_votes))

    logger.info("  Total congressional votes: %d", len(all_votes))

    with open(cache_file, 'w') as f:
        json.dump(all_votes, f)

    return all_votes


def _parse_congressional_votes(raw_votes):
    """Parse raw GovTrack vote JSON into DataFrame rows.

    Filters to passage votes only (final passage, not procedural/cloture).
    """
    rows = []
    for v in raw_votes:
        # Filter to passage votes only (client-side; API doesn't support category filter)
        category_raw = v.get('category', '')
        if category_raw not in ('passage', 'passage-suspension', 'veto-override'):
            continue

        vote_id = v.get('id', '')
        created = v.get('created', '')
        chamber = v.get('chamber_label', v.get('chamber', ''))
        category = v.get('category_label', category_raw)
        result = v.get('result', '')
        question = v.get('question', '')
        total_plus = v.get('total_plus', 0) or 0
        total_minus = v.get('total_minus', 0) or 0
        margin = abs(total_plus - total_minus)
        total_votes = total_plus + total_minus
        is_close = (margin / total_votes < 0.10) if total_votes > 0 else False

        # Get bill info if available
        related_bill = v.get('related_bill')
        bill_number = ''
        bill_title = ''
        subjects = []
        if related_bill:
            bill_number = related_bill.get('display_number', '')
            bill_title = related_bill.get('title_without_number', '')
            bill_title_short = related_bill.get('title', '')

        description = question or bill_title or f"Vote {vote_id}"
        policy_area = classify_policy_area(description, subjects)
        naics = ','.join(POLICY_AREA_TO_NAICS.get(policy_area, []))

        try:
            event_date = pd.Timestamp(created).normalize()
        except Exception:
            continue

        rows.append({
            'EVENT_ID': f"vote_{chamber[0]}_{event_date.year}_{vote_id}",
            'EVENT_DATE': event_date,
            'EVENT_TYPE': 'CONGRESSIONAL_VOTE',
            'EVENT_SUBTYPE': category,
            'DESCRIPTION': description[:500],
            'CHAMBER': chamber,
            'BILL_NUMBER': bill_number,
            'RESULT': result,
            'POLICY_AREA': policy_area,
            'AFFECTED_NAICS': naics,
            'VOTE_MARGIN': margin,
            'IS_CLOSE_VOTE': is_close,
        })

    return rows


def fetch_executive_orders(start_date, end_date, cache_dir):
    """Fetch executive orders from Federal Register API."""
    raw_dir = os.path.join(cache_dir, 'raw', 'executive_orders')
    os.makedirs(raw_dir, exist_ok=True)

    cache_file = os.path.join(raw_dir, 'executive_orders.json')
    if os.path.exists(cache_file):
        age_days = (time.time() - os.path.getmtime(cache_file)) / 86400
        if age_days < 7:
            logger.info("  Using cached executive orders (%.1f days old)", age_days)
            with open(cache_file) as f:
                return json.load(f)

    logger.info("  Fetching executive orders from Federal Register API...")
    base_url = 'https://www.federalregister.gov/api/v1/documents.json'

    all_orders = []
    page = 1

    while True:
        params = {
            'conditions[presidential_document_type]': 'executive_order',
            'conditions[publication_date][gte]': start_date,
            'conditions[publication_date][lte]': end_date,
            'per_page': 100,
            'page': page,
            'order': 'oldest',
            'fields[]': [
                'document_number', 'title', 'signing_date',
                'publication_date', 'executive_order_number',
                'abstract', 'topics', 'subtype',
            ],
        }
        data = _api_get(base_url, params=params)
        if data is None:
            break

        results = data.get('results', [])
        if not results:
            break

        all_orders.extend(results)
        page += 1
        logger.info("    Fetched page %d (%d orders total)", page - 1, len(all_orders))

        total_pages = data.get('total_pages', 1)
        if page > total_pages:
            break

        time.sleep(0.5)

    logger.info("  Total executive orders: %d", len(all_orders))

    with open(cache_file, 'w') as f:
        json.dump(all_orders, f)

    return all_orders


def _parse_executive_orders(raw_orders):
    """Parse raw Federal Register EO JSON into DataFrame rows."""
    rows = []
    for eo in raw_orders:
        doc_num = eo.get('document_number', '')
        eo_num = eo.get('executive_order_number', '')
        title = eo.get('title', '')
        abstract = eo.get('abstract', '') or ''
        signing_date = eo.get('signing_date') or eo.get('publication_date', '')
        topics = eo.get('topics', []) or []

        description = f"EO {eo_num}: {title}" if eo_num else title
        combined_text = f"{title} {abstract} {' '.join(topics)}"
        policy_area = classify_policy_area(combined_text)
        naics = ','.join(POLICY_AREA_TO_NAICS.get(policy_area, []))

        try:
            event_date = pd.Timestamp(signing_date).normalize()
        except Exception:
            continue

        rows.append({
            'EVENT_ID': f"eo_{eo_num or doc_num}",
            'EVENT_DATE': event_date,
            'EVENT_TYPE': 'EXECUTIVE_ORDER',
            'EVENT_SUBTYPE': policy_area,
            'DESCRIPTION': description[:500],
            'CHAMBER': 'Executive',
            'BILL_NUMBER': f"EO {eo_num}" if eo_num else doc_num,
            'RESULT': 'Signed',
            'POLICY_AREA': policy_area,
            'AFFECTED_NAICS': naics,
            'VOTE_MARGIN': None,
            'IS_CLOSE_VOTE': False,
        })

    return rows


def _download_scdb(cache_dir):
    """Download Supreme Court Database CSV from wustl.edu."""
    import zipfile
    from io import BytesIO

    # SCDB case-centered with citation data
    url = 'http://scdb.wustl.edu/_brickFiles/2024_01/SCDB_2024_01_caseCentered_Citation.csv.zip'
    csv_path = os.path.join(cache_dir, 'scdb_cases.csv')

    logger.info("  Downloading Supreme Court Database from wustl.edu...")
    try:
        resp = requests.get(url, timeout=60)
        resp.raise_for_status()
        with zipfile.ZipFile(BytesIO(resp.content)) as zf:
            # Extract the CSV (first file in the zip)
            csv_names = [n for n in zf.namelist() if n.endswith('.csv')]
            if not csv_names:
                logger.warning("No CSV found in SCDB zip")
                return False
            with zf.open(csv_names[0]) as src, open(csv_path, 'wb') as dst:
                dst.write(src.read())
        logger.info("  SCDB downloaded: %s", csv_path)
        return True
    except Exception as e:
        logger.warning("Failed to download SCDB: %s", e)
        return False


def load_court_decisions(cache_dir):
    """Load Supreme Court decisions from SCDB CSV.

    Auto-downloads from http://scdb.wustl.edu if not present locally.
    """
    csv_path = os.path.join(cache_dir, 'scdb_cases.csv')
    if not os.path.exists(csv_path):
        os.makedirs(cache_dir, exist_ok=True)
        if not _download_scdb(cache_dir):
            logger.warning("SCDB not available. Court decisions will be empty.")
            return []

    logger.info("  Loading Supreme Court decisions from %s", csv_path)
    try:
        df = pd.read_csv(csv_path, encoding='latin-1')
    except Exception as e:
        logger.warning("Failed to read SCDB CSV: %s", e)
        return []

    rows = []
    for _, case in df.iterrows():
        date_str = case.get('dateDecision', '')
        case_name = case.get('caseName', '')
        issue_area = case.get('issueArea', None)
        maj_votes = case.get('majVotes', 0)
        min_votes = case.get('minVotes', 0)
        direction = case.get('decisionDirection', '')

        try:
            event_date = pd.Timestamp(str(date_str)).normalize()
        except Exception:
            continue

        policy_area = SCDB_ISSUE_TO_POLICY.get(issue_area, 'unknown')
        naics = ','.join(POLICY_AREA_TO_NAICS.get(policy_area, []))
        margin = abs(int(maj_votes or 0) - int(min_votes or 0))
        total = int(maj_votes or 0) + int(min_votes or 0)
        is_close = margin <= 2 if total > 0 else False

        # Direction: 1=conservative, 2=liberal, 3=unspecifiable
        result_map = {1: 'Conservative', 2: 'Liberal', 3: 'Unspecified'}
        result = result_map.get(direction, 'Unknown')

        case_id = case.get('caseId', f"scotus_{event_date.strftime('%Y%m%d')}")

        rows.append({
            'EVENT_ID': f"scotus_{case_id}",
            'EVENT_DATE': event_date,
            'EVENT_TYPE': 'COURT_DECISION',
            'EVENT_SUBTYPE': policy_area,
            'DESCRIPTION': str(case_name)[:500],
            'CHAMBER': 'Judicial',
            'BILL_NUMBER': str(case.get('docket', '')),
            'RESULT': result,
            'POLICY_AREA': policy_area,
            'AFFECTED_NAICS': naics,
            'VOTE_MARGIN': margin,
            'IS_CLOSE_VOTE': is_close,
        })

    logger.info("  Loaded %d Supreme Court decisions", len(rows))
    return rows


# ═══════════════════════════════════════════════════════════════════════
# MAIN LOADER
# ═══════════════════════════════════════════════════════════════════════

def load_political_events(start_date='2000-01-01', end_date='2025-12-31',
                          cache_dir='./data/political_events',
                          force_refresh=False):
    """Load unified political events dataset.

    Combines congressional votes, executive orders, and court decisions
    into a single DataFrame with standardized schema.

    Parameters
    ----------
    start_date : str
        Start of date range.
    end_date : str
        End of date range.
    cache_dir : str
        Directory for caching raw and processed data.
    force_refresh : bool
        If True, bypass cache and re-fetch from APIs.

    Returns
    -------
    pd.DataFrame
        Political events with columns: EVENT_ID, EVENT_DATE, EVENT_TYPE,
        EVENT_SUBTYPE, DESCRIPTION, CHAMBER, BILL_NUMBER, RESULT,
        POLICY_AREA, AFFECTED_NAICS, VOTE_MARGIN, IS_CLOSE_VOTE.
    """
    os.makedirs(cache_dir, exist_ok=True)

    # Check processed cache
    processed_path = os.path.join(cache_dir, 'political_events.csv')
    if os.path.exists(processed_path) and not force_refresh:
        age_days = (time.time() - os.path.getmtime(processed_path)) / 86400
        if age_days < 7:
            logger.info("Loading cached political events (%.1f days old)", age_days)
            df = pd.read_csv(processed_path)
            df['EVENT_DATE'] = pd.to_datetime(df['EVENT_DATE'], errors='coerce')
            logger.info("  %d political events loaded from cache", len(df))
            return df

    logger.info("Building political events dataset...")
    all_rows = []

    # 1. Congressional votes
    try:
        raw_votes = fetch_congressional_votes(start_date, end_date, cache_dir)
        vote_rows = _parse_congressional_votes(raw_votes)
        all_rows.extend(vote_rows)
        logger.info("  Parsed %d congressional votes", len(vote_rows))
    except Exception as e:
        logger.warning("Failed to fetch congressional votes: %s", e)

    # 2. Executive orders
    try:
        raw_eos = fetch_executive_orders(start_date, end_date, cache_dir)
        eo_rows = _parse_executive_orders(raw_eos)
        all_rows.extend(eo_rows)
        logger.info("  Parsed %d executive orders", len(eo_rows))
    except Exception as e:
        logger.warning("Failed to fetch executive orders: %s", e)

    # 3. Supreme Court decisions
    try:
        court_rows = load_court_decisions(cache_dir)
        all_rows.extend(court_rows)
    except Exception as e:
        logger.warning("Failed to load court decisions: %s", e)

    if not all_rows:
        logger.warning("No political events loaded from any source")
        return pd.DataFrame(columns=[
            'EVENT_ID', 'EVENT_DATE', 'EVENT_TYPE', 'EVENT_SUBTYPE',
            'DESCRIPTION', 'CHAMBER', 'BILL_NUMBER', 'RESULT',
            'POLICY_AREA', 'AFFECTED_NAICS', 'VOTE_MARGIN', 'IS_CLOSE_VOTE',
        ])

    df = pd.DataFrame(all_rows)
    df['EVENT_DATE'] = pd.to_datetime(df['EVENT_DATE'], errors='coerce')

    # Filter to date range
    mask = (df['EVENT_DATE'] >= pd.Timestamp(start_date)) & \
           (df['EVENT_DATE'] <= pd.Timestamp(end_date))
    df = df[mask].copy()

    # Drop events with unknown policy area (not useful for treatment assignment)
    # Keep them but log the count
    n_unknown = (df['POLICY_AREA'] == 'unknown').sum()
    if n_unknown > 0:
        logger.info("  %d events with unclassified policy area (kept as 'unknown')",
                     n_unknown)

    # Sort by date
    df = df.sort_values('EVENT_DATE').reset_index(drop=True)

    # Save processed cache
    df.to_csv(processed_path, index=False)
    logger.info("Political events: %d total (%d votes, %d EOs, %d court)",
                len(df),
                (df['EVENT_TYPE'] == 'CONGRESSIONAL_VOTE').sum(),
                (df['EVENT_TYPE'] == 'EXECUTIVE_ORDER').sum(),
                (df['EVENT_TYPE'] == 'COURT_DECISION').sum())

    return df
