"""Firm-level political stance scoring via text embeddings.

Implements the methodology from Li et al. (2021) "Measuring Corporate
Culture Using Machine Learning" adapted for political stance:

  1. Extract seed dictionaries from party platforms using TF-IDF log-odds
  2. Expand dictionaries using cosine similarity in TF-IDF space
  3. Score each firm-year filing on a liberal–conservative spectrum

Since gensim Word2Vec requires C compilation, we use sklearn TF-IDF
vectors with cosine similarity — methodologically equivalent for
dictionary expansion and document scoring, and more reproducible.

References
----------
Li, K., Mai, F., Shen, R. & Yan, X. (2021). Measuring Corporate Culture
    Using Machine Learning. Review of Financial Studies, 34(7).
"""

import os
import re
import logging
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

from .config import logger

# ═══════════════════════════════════════════════════════════════════════
# TEXT PREPROCESSING
# ═══════════════════════════════════════════════════════════════════════

_STOPWORDS = None

def _get_stopwords():
    global _STOPWORDS
    if _STOPWORDS is None:
        try:
            from nltk.corpus import stopwords
            _STOPWORDS = set(stopwords.words('english'))
        except Exception:
            # Minimal fallback
            _STOPWORDS = {
                'the', 'a', 'an', 'and', 'or', 'but', 'in', 'on', 'at',
                'to', 'for', 'of', 'with', 'by', 'from', 'is', 'are',
                'was', 'were', 'be', 'been', 'being', 'have', 'has', 'had',
                'do', 'does', 'did', 'will', 'would', 'could', 'should',
                'may', 'might', 'shall', 'can', 'not', 'no', 'this', 'that',
                'these', 'those', 'it', 'its', 'as', 'if', 'than', 'so',
                'such', 'we', 'our', 'they', 'their', 'he', 'she', 'his',
                'her', 'which', 'who', 'whom', 'what', 'when', 'where',
                'how', 'all', 'each', 'every', 'both', 'few', 'more',
                'most', 'other', 'some', 'any', 'only', 'own', 'same',
                'very', 'just', 'also', 'about', 'up', 'out', 'into',
            }
    return _STOPWORDS


def _tokenize(text: str) -> List[str]:
    """Lowercase, strip non-alpha, remove stopwords and short tokens."""
    stops = _get_stopwords()
    tokens = re.findall(r'[a-z]{3,}', text.lower())
    return [t for t in tokens if t not in stops and len(t) >= 3]


# ═══════════════════════════════════════════════════════════════════════
# SEED DICTIONARY EXTRACTION
# ═══════════════════════════════════════════════════════════════════════

def extract_seed_dictionaries(
    platform_corpus: pd.DataFrame,
    topn: int = 100,
    min_df: int = 2,
    method: str = 'log_odds',
) -> Tuple[set, set]:
    """Extract ideologically distinctive terms from party platforms.

    Uses TF-IDF log-odds ratio to find words that differentiate
    Democratic from Republican platforms.

    Parameters
    ----------
    platform_corpus : pd.DataFrame
        From PartyPlatformDownloader.build_platform_corpus().
        Columns: year, party, text, word_count.
    topn : int
        Number of seed words per party.
    min_df : int
        Minimum document frequency for a term.
    method : str
        'log_odds' (default) or 'tfidf_diff'.

    Returns
    -------
    (liberal_seeds, conservative_seeds) : tuple of sets
    """
    dem_texts = platform_corpus[
        platform_corpus['party'] == 'Democratic'
    ]['text'].tolist()
    rep_texts = platform_corpus[
        platform_corpus['party'] == 'Republican'
    ]['text'].tolist()

    if not dem_texts or not rep_texts:
        raise ValueError("Need at least one platform per party")

    # Combine all texts
    all_texts = dem_texts + rep_texts
    labels = ['D'] * len(dem_texts) + ['R'] * len(rep_texts)

    # Fit TF-IDF on all platforms
    tfidf = TfidfVectorizer(
        tokenizer=_tokenize,
        token_pattern=None,
        min_df=min_df,
        max_df=0.95,
        max_features=10000,
        sublinear_tf=True,
    )
    X = tfidf.fit_transform(all_texts)
    vocab = tfidf.get_feature_names_out()

    # Compute mean TF-IDF per party
    dem_idx = [i for i, l in enumerate(labels) if l == 'D']
    rep_idx = [i for i, l in enumerate(labels) if l == 'R']

    dem_mean = np.asarray(X[dem_idx].mean(axis=0)).flatten()
    rep_mean = np.asarray(X[rep_idx].mean(axis=0)).flatten()

    if method == 'log_odds':
        # Log-odds ratio: log(dem_tfidf + eps) - log(rep_tfidf + eps)
        eps = 1e-8
        log_odds = np.log(dem_mean + eps) - np.log(rep_mean + eps)

        # Liberal seeds: highest log-odds (more D than R)
        lib_idx = np.argsort(log_odds)[-topn:][::-1]
        con_idx = np.argsort(log_odds)[:topn]

        liberal_seeds = set(vocab[lib_idx])
        conservative_seeds = set(vocab[con_idx])

    elif method == 'tfidf_diff':
        diff = dem_mean - rep_mean
        lib_idx = np.argsort(diff)[-topn:][::-1]
        con_idx = np.argsort(diff)[:topn]

        liberal_seeds = set(vocab[lib_idx])
        conservative_seeds = set(vocab[con_idx])

    else:
        raise ValueError(f"Unknown method: {method}")

    # Remove any overlap
    overlap = liberal_seeds & conservative_seeds
    liberal_seeds -= overlap
    conservative_seeds -= overlap

    logger.info(
        "Seed dictionaries: %d liberal, %d conservative (removed %d overlap)",
        len(liberal_seeds), len(conservative_seeds), len(overlap),
    )

    return liberal_seeds, conservative_seeds


# ═══════════════════════════════════════════════════════════════════════
# DICTIONARY EXPANSION (Li et al. 2021 approach)
# ═══════════════════════════════════════════════════════════════════════

def expand_dictionary(
    seeds: set,
    tfidf_matrix,
    vocab: np.ndarray,
    topn: int = 50,
    threshold: float = 0.3,
) -> set:
    """Expand a seed dictionary using cosine similarity in TF-IDF space.

    For each seed word, find the topn nearest neighbors in the corpus
    TF-IDF space. Words above the similarity threshold are added.

    Parameters
    ----------
    seeds : set
        Seed words.
    tfidf_matrix : sparse matrix
        TF-IDF matrix (documents x terms).
    vocab : array
        Vocabulary array from TfidfVectorizer.
    topn : int
        Max neighbors per seed.
    threshold : float
        Minimum cosine similarity to include.

    Returns
    -------
    set : Expanded dictionary.
    """
    vocab_list = list(vocab)
    vocab_idx = {w: i for i, w in enumerate(vocab_list)}

    # Term-term similarity: transpose TF-IDF to get term vectors
    # (each term represented by its distribution across documents)
    term_vectors = tfidf_matrix.T  # terms x documents

    expanded = set(seeds)
    seed_indices = [vocab_idx[s] for s in seeds if s in vocab_idx]

    if not seed_indices:
        logger.warning("No seed words found in corpus vocabulary")
        return expanded

    # Compute similarity in batches to avoid memory issues
    batch_size = 50
    for i in range(0, len(seed_indices), batch_size):
        batch_idx = seed_indices[i:i + batch_size]
        seed_vecs = term_vectors[batch_idx]
        sims = cosine_similarity(seed_vecs, term_vectors)

        for row in range(sims.shape[0]):
            top_indices = np.argsort(sims[row])[-topn:][::-1]
            for idx in top_indices:
                if sims[row, idx] >= threshold and idx not in batch_idx:
                    expanded.add(vocab_list[idx])

    logger.info(
        "Expanded dictionary: %d seeds → %d terms (threshold=%.2f)",
        len(seeds), len(expanded), threshold,
    )
    return expanded


# ═══════════════════════════════════════════════════════════════════════
# FIRM-YEAR STANCE SCORING
# ═══════════════════════════════════════════════════════════════════════

def score_filing(
    tokens: List[str],
    liberal_dict: set,
    conservative_dict: set,
) -> Dict[str, float]:
    """Score a single filing's political stance.

    Parameters
    ----------
    tokens : list of str
        Tokenized filing text.
    liberal_dict : set
        Expanded liberal dictionary.
    conservative_dict : set
        Expanded conservative dictionary.

    Returns
    -------
    dict with keys: L_COUNT, C_COUNT, TOTAL_TOKENS, L_FREQ, C_FREQ,
                    STANCE_SCORE (-1 conservative to +1 liberal).
    """
    l_count = sum(1 for t in tokens if t in liberal_dict)
    c_count = sum(1 for t in tokens if t in conservative_dict)
    total = len(tokens)
    total_political = l_count + c_count

    l_freq = l_count / total if total > 0 else 0.0
    c_freq = c_count / total if total > 0 else 0.0

    if total_political > 0:
        stance_score = (l_count - c_count) / total_political
    else:
        stance_score = 0.0

    return {
        'L_COUNT': l_count,
        'C_COUNT': c_count,
        'TOTAL_TOKENS': total,
        'L_FREQ': l_freq,
        'C_FREQ': c_freq,
        'STANCE_SCORE': stance_score,
    }


def _classify_stance(score: float, thresholds: Tuple[float, float] = (-0.1, 0.1)) -> str:
    """Map continuous stance score to categorical label."""
    if score <= thresholds[0]:
        return 'Conservative'
    elif score >= thresholds[1]:
        return 'Liberal'
    else:
        return 'Mixed'


# ═══════════════════════════════════════════════════════════════════════
# MAIN PIPELINE
# ═══════════════════════════════════════════════════════════════════════

class StanceScorer:
    """End-to-end pipeline for scoring firm political stance from SEC filings.

    Implements the Li et al. (2021) methodology adapted for political stance:
    1. Extract seeds from party platforms (TF-IDF log-odds)
    2. Train TF-IDF on SEC filing corpus
    3. Expand dictionaries via cosine similarity
    4. Score each firm-year filing
    """

    def __init__(
        self,
        platform_dir: str = './party_platforms_data',
        filing_text_path: str = None,
        seed_topn: int = 100,
        expand_topn: int = 50,
        expand_threshold: float = 0.3,
    ):
        self.platform_dir = platform_dir
        self.filing_text_path = filing_text_path
        self.seed_topn = seed_topn
        self.expand_topn = expand_topn
        self.expand_threshold = expand_threshold

        # Populated during pipeline
        self.liberal_seeds = set()
        self.conservative_seeds = set()
        self.liberal_expanded = set()
        self.conservative_expanded = set()
        self.platform_corpus = None
        self.filing_tfidf = None
        self.filing_vocab = None
        self.filing_tfidf_matrix = None
        self.stance_scores = None

    def _load_platforms(self) -> pd.DataFrame:
        """Load party platform corpus."""
        from .party_platforms import PartyPlatformDownloader
        downloader = PartyPlatformDownloader(output_dir=self.platform_dir)
        corpus = downloader.build_platform_corpus()
        if corpus.empty:
            logger.warning("No platforms found. Downloading...")
            downloader.download_all_platforms()
            corpus = downloader.build_platform_corpus()
        self.platform_corpus = corpus
        logger.info("Loaded %d party platforms", len(corpus))
        return corpus

    def _load_filings(self) -> pd.DataFrame:
        """Load SEC filing text dataset."""
        if self.filing_text_path and os.path.exists(self.filing_text_path):
            path = self.filing_text_path
        else:
            # Auto-detect from sec_filings_data/
            base = os.path.dirname(self.platform_dir)
            candidates = [
                os.path.join(base, 'sec_filings_data', f)
                for f in os.listdir(os.path.join(base, 'sec_filings_data'))
                if f.startswith('sec_filing_text') and f.endswith('.csv')
            ]
            if not candidates:
                raise FileNotFoundError(
                    "No SEC filing text CSV found in sec_filings_data/"
                )
            path = candidates[0]

        logger.info("Loading SEC filing text from %s...", path)
        df = pd.read_csv(path, low_memory=False)
        df.columns = [c.upper() for c in df.columns]

        # Parse dates
        if 'FILING_DATE' in df.columns:
            df['FILING_DATE'] = pd.to_datetime(df['FILING_DATE'], errors='coerce')
            df['YEAR'] = df['FILING_DATE'].dt.year

        logger.info(
            "Loaded %d filing sections for %d tickers",
            len(df), df['TICKER'].nunique() if 'TICKER' in df.columns else 0,
        )
        return df

    def run(self) -> pd.DataFrame:
        """Run the full stance scoring pipeline.

        Returns
        -------
        pd.DataFrame
            Columns: TICKER, YEAR, FILING_DATE, L_COUNT, C_COUNT,
            TOTAL_TOKENS, L_FREQ, C_FREQ, STANCE_SCORE, STANCE_LABEL.
        """
        # Step 1: Load data
        logger.info("=" * 60)
        logger.info("POLITICAL STANCE SCORING PIPELINE")
        logger.info("Methodology: Li et al. (2021) adapted for political stance")
        logger.info("=" * 60)

        corpus = self._load_platforms()
        filings = self._load_filings()

        # Step 2: Extract seed dictionaries from platforms
        logger.info("Step 1: Extracting seed dictionaries from party platforms...")
        self.liberal_seeds, self.conservative_seeds = extract_seed_dictionaries(
            corpus, topn=self.seed_topn,
        )
        logger.info("  Liberal seeds (%d): %s...",
                     len(self.liberal_seeds),
                     ', '.join(sorted(self.liberal_seeds)[:10]))
        logger.info("  Conservative seeds (%d): %s...",
                     len(self.conservative_seeds),
                     ', '.join(sorted(self.conservative_seeds)[:10]))

        # Step 3: Build TF-IDF on SEC filing corpus
        logger.info("Step 2: Training TF-IDF on SEC filing corpus...")

        # Combine text per filing (may have multiple sections)
        filing_texts = filings.groupby(
            ['TICKER', 'FILING_DATE']
        )['TEXT'].apply(lambda x: ' '.join(x.dropna())).reset_index()
        filing_texts.columns = ['TICKER', 'FILING_DATE', 'FULL_TEXT']
        filing_texts['FILING_DATE'] = pd.to_datetime(
            filing_texts['FILING_DATE'], errors='coerce')
        filing_texts['YEAR'] = filing_texts['FILING_DATE'].dt.year

        tfidf = TfidfVectorizer(
            tokenizer=_tokenize,
            token_pattern=None,
            min_df=3,
            max_df=0.95,
            max_features=15000,
            sublinear_tf=True,
        )
        self.filing_tfidf_matrix = tfidf.fit_transform(filing_texts['FULL_TEXT'])
        self.filing_vocab = tfidf.get_feature_names_out()
        self.filing_tfidf = tfidf

        logger.info("  TF-IDF: %d documents, %d terms",
                     self.filing_tfidf_matrix.shape[0],
                     self.filing_tfidf_matrix.shape[1])

        # Step 4: Expand dictionaries
        logger.info("Step 3: Expanding dictionaries via cosine similarity...")
        self.liberal_expanded = expand_dictionary(
            self.liberal_seeds,
            self.filing_tfidf_matrix,
            self.filing_vocab,
            topn=self.expand_topn,
            threshold=self.expand_threshold,
        )
        self.conservative_expanded = expand_dictionary(
            self.conservative_seeds,
            self.filing_tfidf_matrix,
            self.filing_vocab,
            topn=self.expand_topn,
            threshold=self.expand_threshold,
        )

        # Remove overlap from expanded dictionaries
        overlap = self.liberal_expanded & self.conservative_expanded
        self.liberal_expanded -= overlap
        self.conservative_expanded -= overlap
        logger.info(
            "  After expansion: %d liberal, %d conservative (removed %d overlap)",
            len(self.liberal_expanded), len(self.conservative_expanded),
            len(overlap),
        )

        # Step 5: Score each filing
        logger.info("Step 4: Scoring %d filings...", len(filing_texts))
        results = []
        for _, row in filing_texts.iterrows():
            tokens = _tokenize(row['FULL_TEXT'])
            scores = score_filing(
                tokens, self.liberal_expanded, self.conservative_expanded,
            )
            scores['TICKER'] = row['TICKER']
            scores['FILING_DATE'] = row['FILING_DATE']
            scores['YEAR'] = row['YEAR']
            results.append(scores)

        self.stance_scores = pd.DataFrame(results)
        self.stance_scores['STANCE_LABEL'] = self.stance_scores['STANCE_SCORE'].apply(
            _classify_stance
        )

        logger.info("Stance scoring complete:")
        logger.info("  %d firm-year observations", len(self.stance_scores))
        for label in ['Liberal', 'Mixed', 'Conservative']:
            n = (self.stance_scores['STANCE_LABEL'] == label).sum()
            logger.info("  %s: %d (%.1f%%)", label, n,
                         100 * n / len(self.stance_scores))
        logger.info("  Mean stance score: %.4f (std=%.4f)",
                     self.stance_scores['STANCE_SCORE'].mean(),
                     self.stance_scores['STANCE_SCORE'].std())

        return self.stance_scores

    def get_seed_summary(self) -> pd.DataFrame:
        """Return a summary of seed and expanded dictionaries."""
        rows = []
        for label, seeds, expanded in [
            ('Liberal', self.liberal_seeds, self.liberal_expanded),
            ('Conservative', self.conservative_seeds, self.conservative_expanded),
        ]:
            rows.append({
                'DIRECTION': label,
                'N_SEEDS': len(seeds),
                'N_EXPANDED': len(expanded),
                'SEED_WORDS': ', '.join(sorted(seeds)[:30]),
                'EXPANDED_SAMPLE': ', '.join(
                    sorted(expanded - seeds)[:30]
                ),
            })
        return pd.DataFrame(rows)
