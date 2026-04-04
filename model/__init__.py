"""
Model package — culture war events as systematic risk factors.

Provides modeling for three interconnected dissertation essays:

  Essay 1: Volatility regimes and the Fama-French five-factor model
  Essay 2: Culture war event study with regime conditioning
  Essay 3: Systematic risk — macro regime effects on event impacts (in progress)

Public API re-exports for backward compatibility with ``from Model import ...``.
"""

__version__ = "0.1.0"
__author__ = "Ashley Roseboro"

# ── DataStore (shared) ──────────────────────────────────────────────────

from .datastore import DataStore

# ── Essay 1 utilities (shared across essays) ────────────────────────────

from .essay1 import (
    benjamini_hochberg,
    assemble_macro_controls,
    compute_fomo_z,
)

# ── Essay 1 — Volatility Regimes & FF5 ─────────────────────────────────

from .essay1 import (
    RegimeResult,
    estimate_vix_regimes,
    select_n_regimes,
    RegimeFactorResult,
    FF5RegimeAnalysis,
    ff5_by_regime,
    CultureWarRegimeAnalysis,
    culture_war_by_regime,
    SentimentRegimeAnalysis,
    sentiment_by_regime,
    save_results as save_essay1_results,
)

# ── Essay 1 — Matched Control Analysis ─────────────────────────────────

from .essay1_matched import (
    MatchedControlResult,
    StockRegimeResult,
    ff5_matched_control_analysis,
    save_matched_results,
)

# ── Essay 2 — Factor Model (backward compat) + NLP Pipeline ──────────

from .essay2 import (
    FactorModelResult,
    factor_model,
    NLPAnalysis,
    FilingSentiment,
    EventNLPResult,
    score_news_sentiment,
    score_filing_sentiment,
    download_and_parse_filings,
    build_event_nlp_panel,
    run_nlp_analysis,
    save_nlp_results,
    PoliticalAlignmentResult,
    load_platform_corpus,
    extract_distinctive_phrases,
    compute_stance_scores,
    compute_political_alignment,
    save_alignment_results,
)

# ── Essay 2 — Culture War Event DiD ──────────────────────────────────

from .essay2_did import (
    EventCAR,
    DiDResult,
    ParallelTrendsResult,
    build_car_panel,
    compute_car,
    run_did,
    parallel_trends_test,
    save_did_results,
)

# ── Essay 3 — Insider Trading ─────────────────────────────────────────

from .essay3 import (
    classify_inflation_regime,
    run_essay3,
    save_essay3_results,
)

# ── Reporting ───────────────────────────────────────────────────────────

from .reporting import (
    summary_statistics,
    run_and_save,
)

# ── Public API ──────────────────────────────────────────────────────────

__all__ = [
    # Shared
    "DataStore",
    "benjamini_hochberg",
    "assemble_macro_controls",
    # Essay 1 — Regimes & FF5
    "RegimeResult",
    "estimate_vix_regimes",
    "select_n_regimes",
    "RegimeFactorResult",
    "FF5RegimeAnalysis",
    "ff5_by_regime",
    "CultureWarRegimeAnalysis",
    "culture_war_by_regime",
    "SentimentRegimeAnalysis",
    "sentiment_by_regime",
    "save_essay1_results",
    # Essay 1 — Matched Controls
    "MatchedControlResult",
    "StockRegimeResult",
    "ff5_matched_control_analysis",
    "save_matched_results",
    # Essay 1 — FOMO z-score
    "compute_fomo_z",
    # Essay 2 — Factor Model + NLP
    "FactorModelResult",
    "factor_model",
    "NLPAnalysis",
    "FilingSentiment",
    "EventNLPResult",
    "score_news_sentiment",
    "score_filing_sentiment",
    "download_and_parse_filings",
    "build_event_nlp_panel",
    "run_nlp_analysis",
    "save_nlp_results",
    "PoliticalAlignmentResult",
    "load_platform_corpus",
    "extract_distinctive_phrases",
    "compute_stance_scores",
    "compute_political_alignment",
    "save_alignment_results",
    # Essay 2 — Event DiD
    "EventCAR",
    "DiDResult",
    "ParallelTrendsResult",
    "build_car_panel",
    "compute_car",
    "run_did",
    "parallel_trends_test",
    "save_did_results",
    # Essay 3 — Systematic Risk (in progress)
    "classify_inflation_regime",
    # Reporting
    "summary_statistics",
    "run_and_save",
]
