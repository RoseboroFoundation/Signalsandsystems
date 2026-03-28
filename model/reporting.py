"""Summary statistics and orchestration runner for all essays."""

import logging
from dataclasses import dataclass
from typing import Optional

import pandas as pd

from .datastore import DataStore
from .essay1 import (
    RegimeResult,
    FF5RegimeAnalysis,
    CultureWarRegimeAnalysis,
    SentimentRegimeAnalysis,
    estimate_vix_regimes,
    ff5_by_regime,
    culture_war_by_regime,
    sentiment_by_regime,
    assemble_macro_controls,
    save_results as save_essay1_results,
)
from .essay1_matched import (
    MatchedControlResult,
    ff5_matched_control_analysis,
    save_matched_results,
)

logger = logging.getLogger(__name__)


@dataclass
class DissertationResults:
    """Complete output from all essays."""
    regime_result: Optional[RegimeResult] = None
    ff5_analysis: Optional[FF5RegimeAnalysis] = None
    cw_analysis: Optional[CultureWarRegimeAnalysis] = None
    sentiment_analysis: Optional[SentimentRegimeAnalysis] = None
    matched_result: Optional[MatchedControlResult] = None


def summary_statistics(store: DataStore) -> dict:
    """
    Generate summary statistics for the loaded data.

    Returns a dict of DataFrames with descriptive stats.
    """
    stats_dict = {}

    # Event summary
    if not store.events.empty:
        evt = store.events.copy()
        stats_dict['events'] = pd.DataFrame({
            'Total Events': [len(evt)],
            'Unique Tickers': [evt['TICKER'].nunique()],
            'Liberal': [(evt['ESTIMATED_POLITICAL_LEANING'] == 'Liberal').sum()],
            'Conservative': [(evt['ESTIMATED_POLITICAL_LEANING'] == 'Conservative').sum()],
            'Mixed': [(evt['ESTIMATED_POLITICAL_LEANING'] == 'Mixed').sum()],
            'Date Range': [f"{evt['EVENT_DATE'].min()} to {evt['EVENT_DATE'].max()}"],
            'Industries': [evt['INDUSTRY'].nunique()],
        })

    # Stock returns summary
    if not store.stock_returns.empty and 'RETURN' in store.stock_returns.columns:
        rets = store.stock_returns['RETURN'].dropna()
        stats_dict['returns'] = pd.DataFrame({
            'Obs': [len(rets)],
            'Mean Daily': [f"{rets.mean():.4f}"],
            'Std Daily': [f"{rets.std():.4f}"],
            'Min': [f"{rets.min():.4f}"],
            'Max': [f"{rets.max():.4f}"],
            'Tickers': [store.stock_returns['TICKER'].nunique()],
        })

    # Factor summary
    if not store.ff5.empty:
        factor_cols = ['MKT_RF', 'SMB', 'HML', 'RMW', 'CMA']
        available = [c for c in factor_cols if c in store.ff5.columns]
        stats_dict['factors'] = store.ff5[available].describe().round(4)

    return stats_dict


def run_and_save(backend=None, save=True):
    """
    Run Essay 1 analysis and optionally save results.

    Execution order is intentional — matched control analysis
    consumes regime assignments. Nothing is re-estimated.

    Parameters
    ----------
    backend : str, optional
        Force 'aws' or 'sqlite'. Default: auto-detect.
    save : bool
        If True, write results back to the database.

    Returns
    -------
    DissertationResults
    """
    store = DataStore(backend=backend)
    output = DissertationResults()

    print("=" * 60)
    print(f"  Essay 1: Volatility Regimes & FF5 ({store.backend} backend)")
    print("=" * 60)

    # Step 1: Estimate VIX regimes
    print("\nEstimating VIX regimes (Markov switching)...")
    regime_result = estimate_vix_regimes(store, n_regimes=3)
    output.regime_result = regime_result

    if regime_result is not None:
        for label, mean in regime_result.regime_means.items():
            n = regime_result.regime_summary.loc[
                regime_result.regime_summary['REGIME'] == label, 'N_DAYS'
            ].values[0]
            print(f"    {label}: mean VIX={mean:.1f} ({n} days)")

    # Step 2: Macro controls
    macro_controls = assemble_macro_controls(store)

    # Step 3: FF5 by regime (baseline + controlled)
    print("\nRunning FF5 factor analysis by regime...")
    ff5_analysis = ff5_by_regime(
        store, regime_result=regime_result, macro_controls=macro_controls)
    output.ff5_analysis = ff5_analysis

    if ff5_analysis is not None:
        print(f"  Chow test: F={ff5_analysis.chow_test['f_stat']:.2f}, "
              f"p={ff5_analysis.chow_test['p_value']:.4f}")

    # Step 4: Culture war stocks by regime
    print("\nRunning culture war stock analysis...")
    cw_analysis = culture_war_by_regime(
        store, regime_result=regime_result, ff5_analysis=ff5_analysis,
        macro_controls=macro_controls)
    output.cw_analysis = cw_analysis

    if cw_analysis is not None:
        print(f"  {cw_analysis.n_stocks} stocks, {cw_analysis.n_failed} failed")

    # Step 5: Matched control analysis
    print("\nRunning matched control analysis...")
    matched_result = ff5_matched_control_analysis(
        store, regime_result=regime_result)
    output.matched_result = matched_result

    if matched_result is not None:
        print(f"  {matched_result.n_pairs} pairs, "
              f"{matched_result.n_pairs_complete} complete")

    # Step 6: Sentiment analysis (FinBERT + FOMO z-scores)
    print("\nRunning sentiment analysis...")
    sentiment_analysis = sentiment_by_regime(
        store, regime_result=regime_result)
    output.sentiment_analysis = sentiment_analysis

    if sentiment_analysis is not None:
        print(f"  {len(sentiment_analysis.sentiment_daily)} daily observations")

    # Save
    if save and ff5_analysis is not None:
        save_essay1_results(
            store, ff5_analysis, cw_analysis,
            sentiment_analysis=sentiment_analysis,
        )
    if save and matched_result is not None:
        save_matched_results(store, matched_result)

    store.close()
    print("\n" + "=" * 60)
    print("  Complete")
    print("=" * 60)

    return output
