"""
Essay 1 — Complete Analysis with Matched Controls.

Full Essay 1 pipeline: volatility regime estimation, FF5 factor analysis,
culture war stock regressions, matched control comparisons, FinBERT
sentiment scoring, and FOMO z-scores.

Matched control identification: treatment and control firms are matched on
industry (NAICS), so any residual difference in regime-conditional
factor loadings is attributable to culture war exposure rather than
sector or firm-characteristic effects.

Unlike the spanning regression (MKT_RF ~ SMB + HML + RMW + CMA),
matched control regressions use the full FF5 pricing regression
(R_i - RF ~ MKT_RF + SMB + HML + RMW + CMA) for individual stock returns.

WMT-as-multiple-control: regressions run at the unique-firm level so that
a control firm appearing in multiple pairs has its betas estimated once.
Deltas are computed at the pair level, and the paired t-test operates on
pairs, giving each treatment firm exactly one control.

References
----------
Barillas, F. & Shanken, J. (2017). Which alpha? Review of Financial
    Studies, 30(4).
Hamilton, J.D. (1989). A new approach to the economic analysis of
    nonstationary time series and the business cycle. Econometrica, 57(2).
Fama, E.F. & French, K.R. (2015). A five-factor model. Journal of
    Financial Economics, 116(1).
"""

import logging
import warnings
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import pandas as pd
import statsmodels.api as sm
from scipy import stats

from .datastore import DataStore
from .essay1 import (
    RegimeResult,
    estimate_vix_regimes,
    select_n_regimes,
    FF5RegimeAnalysis,
    ff5_by_regime,
    CultureWarRegimeAnalysis,
    culture_war_by_regime,
    assemble_macro_controls,
    SentimentRegimeAnalysis,
    sentiment_by_regime,
    save_results as save_essay1_results,
    _FF5_ALL,
    _HAC_MAXLAGS,
    _MIN_REGIME_OBS,
    benjamini_hochberg,
)

logger = logging.getLogger(__name__)

# FDR threshold for Benjamini-Hochberg correction.  q=0.10 (not 0.05)
# because the matched sample has limited pair count, making 0.05 overly
# conservative for an exploratory dissertation analysis.  This choice
# must be stated in the paper's methodology section.
_BH_FDR_Q = 0.10


@dataclass
class StockRegimeResult:
    """FF5 regression result for a single stock in a single regime."""
    ticker: str
    regime: str
    sufficient_obs: bool
    n_obs: int
    alpha: float = np.nan
    alpha_t: float = np.nan
    alpha_p: float = np.nan
    r_squared: float = np.nan
    betas: Dict[str, float] = field(default_factory=dict)
    t_stats: Dict[str, float] = field(default_factory=dict)
    p_values: Dict[str, float] = field(default_factory=dict)


@dataclass
class MatchedControlResult:
    """
    Treatment vs. control FF5 comparison across volatility regimes.

    Compares regime-conditional factor loadings between culture war
    (treatment) firms and their industry-matched control firms.
    The delta (treatment - control) isolates culture war exposure
    from firm characteristics captured by the matching.
    """
    treatment_results: pd.DataFrame
    control_results: pd.DataFrame
    delta_betas: pd.DataFrame
    paired_ttest: pd.DataFrame
    regime_amplification: pd.DataFrame
    sign_consistency: pd.DataFrame
    n_pairs: int
    n_pairs_complete: int
    coverage: pd.DataFrame = field(default_factory=pd.DataFrame)  # per-ticker regime coverage


def _validate_inputs(store: DataStore, controls_df: pd.DataFrame) -> List[str]:
    """Check all required store attributes and columns before computation.

    Returns a list of error strings (empty if valid).
    """
    errors = []
    if not hasattr(store, 'ff5') or store.ff5.empty:
        errors.append("store.ff5 is empty or missing")
    else:
        missing_cols = [c for c in _FF5_ALL + ['RF'] if c not in store.ff5.columns]
        if missing_cols:
            errors.append(f"store.ff5 missing columns: {missing_cols}")

    required_ctrl_cols = ['TREATMENT_TICKER', 'CONTROL_TICKER']
    missing_ctrl = [c for c in required_ctrl_cols if c not in controls_df.columns]
    if missing_ctrl:
        errors.append(f"CONTROL_COMPANIES missing columns: {missing_ctrl}")

    return errors


def _run_stock_ff5(
    ticker: str,
    store: DataStore,
    factor_regime: pd.DataFrame,
    factor_cols: List[str],
    regime_labels: List[str],
) -> Dict[str, StockRegimeResult]:
    """Run FF5 regression for a single stock within each regime.

    Returns dict {regime_label: StockRegimeResult}.
    Regimes with insufficient observations get a result with
    sufficient_obs=False and NaN coefficients.
    """
    returns = store.get_ticker_returns(ticker)
    if returns.empty or 'RETURN' not in returns.columns:
        logger.warning("%s: no return data found, skipping", ticker)
        return {}

    ret = returns[['DATE', 'RETURN']].copy()
    ret['DATE'] = pd.to_datetime(ret['DATE'], errors='coerce')
    merged = ret.merge(factor_regime, on='DATE', how='inner')
    merged = merged.dropna(subset=['RETURN'] + factor_cols + ['RF'])
    # Convert percent to decimal if needed (median-based check)
    if merged['RETURN'].abs().max() > 1.5:
        logger.warning("%s: returns appear to be in percent, dividing by 100", ticker)
        merged['RETURN'] = merged['RETURN'] / 100
    merged['EXCESS_RETURN'] = merged['RETURN'] - merged['RF']

    results: Dict[str, StockRegimeResult] = {}
    for label in regime_labels:
        sub = merged[merged['REGIME_LABEL'] == label]
        n_obs = len(sub)

        if n_obs < _MIN_REGIME_OBS:
            results[label] = StockRegimeResult(
                ticker=ticker, regime=label, sufficient_obs=False, n_obs=n_obs)
            logger.debug("%s in %s: only %d obs (need %d), skipping",
                         ticker, label, n_obs, _MIN_REGIME_OBS)
            continue

        y = sub['EXCESS_RETURN']
        X = sm.add_constant(sub[factor_cols], has_constant='add')

        try:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                fit = sm.OLS(y, X).fit(
                    cov_type='HAC', cov_kwds={'maxlags': _HAC_MAXLAGS},
                )

            betas = {f: fit.params[f] for f in factor_cols}
            t_stats = {f: fit.tvalues[f] for f in factor_cols}
            p_values = {f: fit.pvalues[f] for f in factor_cols}

            results[label] = StockRegimeResult(
                ticker=ticker, regime=label, sufficient_obs=True,
                n_obs=int(fit.nobs),
                alpha=fit.params['const'],
                alpha_t=fit.tvalues['const'],
                alpha_p=fit.pvalues['const'],
                r_squared=fit.rsquared,
                betas=betas, t_stats=t_stats, p_values=p_values,
            )
        except Exception as e:
            logger.warning("FF5 failed for %s in %s: %s", ticker, label, e)
            results[label] = StockRegimeResult(
                ticker=ticker, regime=label, sufficient_obs=False, n_obs=n_obs)

    return results


def _build_result_rows(
    tickers,
    group_label: str,
    ticker_results: Dict[str, Dict[str, StockRegimeResult]],
    pricing_factors: List[str],
    labels: List[str],
) -> pd.DataFrame:
    """Build a DataFrame of FF5 results for a group of tickers."""
    rows = []
    for ticker in tickers:
        res = ticker_results.get(ticker, {})
        for label in labels:
            sr = res.get(label)
            if sr is not None and sr.sufficient_obs:
                row = {'TICKER': ticker, 'GROUP': group_label,
                       'REGIME': label, 'N_OBS': sr.n_obs,
                       'ALPHA': sr.alpha, 'ALPHA_T': sr.alpha_t,
                       'ALPHA_P': sr.alpha_p, 'R_SQUARED': sr.r_squared}
                for f in pricing_factors:
                    row[f'{f}_BETA'] = sr.betas.get(f, np.nan)
                    row[f'{f}_T'] = sr.t_stats.get(f, np.nan)
                    row[f'{f}_P'] = sr.p_values.get(f, np.nan)
                rows.append(row)
    return pd.DataFrame(rows)


def ff5_matched_control_analysis(
    store: DataStore,
    regime_result: RegimeResult = None,
    n_regimes: int = 3,
) -> Optional[MatchedControlResult]:
    """
    Compare FF5 factor loadings between culture war firms and
    industry-matched control firms across volatility regimes.

    Tests
    -----
    1. Paired t-test: mean(delta_beta) = 0 per factor per regime
    2. Regime amplification test: delta(HighVol) - delta(LowVol) per factor
    3. Sign consistency: % of pairs with same-sign delta

    Parameters
    ----------
    store : DataStore
    regime_result : RegimeResult, optional
        Pre-computed regime assignments. If None, estimates them.
    n_regimes : int
        Number of regimes (used only if regime_result is None).

    Returns
    -------
    MatchedControlResult or None
    """
    # Load control companies table
    controls_df = store.read_table('CONTROL_COMPANIES')
    if controls_df.empty:
        logger.error("No CONTROL_COMPANIES table found")
        return None

    errors = _validate_inputs(store, controls_df)
    if errors:
        for e in errors:
            logger.error("Input validation: %s", e)
        return None

    if regime_result is None:
        regime_result = estimate_vix_regimes(store, n_regimes=n_regimes)
        if regime_result is None:
            return None

    # Prepare factors (convert percent to decimal)
    pricing_factors = _FF5_ALL
    factors = store.ff5[['DATE'] + pricing_factors + ['RF']].dropna().copy()
    factors['DATE'] = pd.to_datetime(factors['DATE'], errors='coerce')
    for col in pricing_factors + ['RF']:
        if factors[col].abs().max() > 1.5:
            logger.warning("Factor %s appears to be in percent, dividing by 100", col)
            factors[col] = factors[col] / 100

    # Merge factors with regime assignments
    factor_regime = factors.merge(
        regime_result.regime_assignments[['DATE', 'REGIME_LABEL']],
        on='DATE', how='inner',
    )

    labels = sorted(regime_result.regime_means.keys(),
                    key=lambda x: regime_result.regime_means[x])

    # Collect unique tickers to avoid redundant regressions
    all_treatments = controls_df['TREATMENT_TICKER'].unique()
    all_controls = controls_df['CONTROL_TICKER'].unique()
    all_tickers = set(all_treatments) | set(all_controls)

    logger.info("Running matched control FF5: %d treatment, %d control, "
                "%d unique tickers, %d pairs",
                len(all_treatments), len(all_controls),
                len(all_tickers), len(controls_df))

    # Run regressions for all unique tickers
    ticker_results = {}
    for ticker in all_tickers:
        ticker_results[ticker] = _run_stock_ff5(
            ticker, store, factor_regime, pricing_factors, regime_labels=labels)

    treatment_df = _build_result_rows(
        all_treatments, 'TREATMENT', ticker_results, pricing_factors, labels)
    control_df = _build_result_rows(
        all_controls, 'CONTROL', ticker_results, pricing_factors, labels)

    # Compute deltas at the pair level
    delta_rows = []

    for _, pair in controls_df.iterrows():
        treat_ticker = pair['TREATMENT_TICKER']
        ctrl_ticker = pair['CONTROL_TICKER']
        treat_res = ticker_results.get(treat_ticker, {})
        ctrl_res = ticker_results.get(ctrl_ticker, {})

        for label in labels:
            t = treat_res.get(label)
            c = ctrl_res.get(label)
            if t is None or c is None or not t.sufficient_obs or not c.sufficient_obs:
                if t is not None and not t.sufficient_obs:
                    logger.debug("Pair %s/%s: treatment insufficient in %s",
                                 treat_ticker, ctrl_ticker, label)
                if c is not None and not c.sufficient_obs:
                    logger.debug("Pair %s/%s: control insufficient in %s",
                                 treat_ticker, ctrl_ticker, label)
                continue

            row = {
                'TREATMENT_TICKER': treat_ticker,
                'CONTROL_TICKER': ctrl_ticker,
                'TREATMENT_COMPANY': pair.get('TREATMENT_COMPANY', ''),
                'CONTROL_FIRM': pair.get('CONTROL_FIRM', ''),
                'INDUSTRY': pair.get('INDUSTRY', ''),
                'REGIME': label,
            }

            row['ALPHA_DELTA'] = t.alpha - c.alpha
            row['R_SQUARED_TREAT'] = t.r_squared
            row['R_SQUARED_CTRL'] = c.r_squared

            for f in pricing_factors:
                delta_val = t.betas.get(f, np.nan) - c.betas.get(f, np.nan)
                if np.isnan(delta_val):
                    logger.debug("NaN delta for %s/%s factor %s in %s",
                                 treat_ticker, ctrl_ticker, f, label)
                row[f'{f}_DELTA'] = delta_val
                row[f'{f}_TREAT'] = t.betas.get(f, np.nan)
                row[f'{f}_CTRL'] = c.betas.get(f, np.nan)

            delta_rows.append(row)

    delta_df = pd.DataFrame(delta_rows)
    n_pairs = controls_df.shape[0]

    if delta_df.empty:
        logger.error("No matched pairs produced delta results")
        return None

    # Count pairs with results in all regimes
    pair_key = delta_df.groupby(
        ['TREATMENT_TICKER', 'CONTROL_TICKER'])['REGIME'].nunique()
    n_complete = (pair_key == len(labels)).sum()

    logger.info("Delta results: %d pair-regime obs, %d/%d pairs complete",
                len(delta_df), n_complete, n_pairs)

    # --- Test 1: Paired t-test per factor per regime ---
    ttest_rows = []
    delta_cols = [f'{f}_DELTA' for f in pricing_factors] + ['ALPHA_DELTA']

    for label in labels:
        sub = delta_df[delta_df['REGIME'] == label]
        if len(sub) < 5:
            continue
        for dc in delta_cols:
            # Aggregate to one delta per treatment firm (average across its
            # matched controls) so the t-test assumes independent observations.
            # Without this, control firms reused across pairs induce correlation.
            agg_deltas = (sub.groupby('TREATMENT_TICKER')[dc]
                          .mean()
                          .dropna())
            if len(agg_deltas) < 5:
                continue
            t_stat, p_val = stats.ttest_1samp(agg_deltas, 0)
            ttest_rows.append({
                'REGIME': label,
                'VARIABLE': dc.replace('_DELTA', ''),
                'MEAN_DELTA': agg_deltas.mean(),
                'STD_DELTA': agg_deltas.std(),
                'N_TREATMENT_FIRMS': len(agg_deltas),
                'N_RAW_PAIRS': len(sub[dc].dropna()),
                'T_STAT': t_stat,
                'P_VALUE': p_val,
            })

    paired_ttest = pd.DataFrame(ttest_rows)

    # --- Test 2: Regime amplification — delta(HighVol) vs delta(LowVol) ---
    # Compares extreme regimes only (lowest vs highest mean VIX).
    # Middle regimes (e.g., "Normal" in K=3) are intentionally excluded:
    # the hypothesis is that culture war effects amplify at volatility
    # extremes, not that they change monotonically across all regimes.
    did_rows = []
    if len(labels) < 2:
        logger.warning("Regime amplification requires at least 2 regimes, skipping")
    else:
        low_label = labels[0]
        high_label = labels[-1]
        low_deltas = delta_df[delta_df['REGIME'] == low_label]
        high_deltas = delta_df[delta_df['REGIME'] == high_label]

        # Merge on pair key
        pair_cols = ['TREATMENT_TICKER', 'CONTROL_TICKER']
        for dc in delta_cols:
            low_vals = low_deltas[pair_cols + [dc]].rename(
                columns={dc: 'LOW'})
            high_vals = high_deltas[pair_cols + [dc]].rename(
                columns={dc: 'HIGH'})
            merged_did = low_vals.merge(high_vals, on=pair_cols, how='inner')

            logger.info("Regime amplification %s: %d/%d pairs in both extremes",
                        dc.replace('_DELTA', ''), len(merged_did), n_pairs)

            if len(merged_did) < 5:
                continue

            # Aggregate to treatment-firm level before the t-test to avoid
            # pseudoreplication when one control appears in multiple pairs
            # (same pattern as Tests 1 and 3).
            agg = merged_did.groupby('TREATMENT_TICKER')[['LOW', 'HIGH']].mean()
            diff = agg['HIGH'] - agg['LOW']

            if len(diff) < 5:
                continue

            t_stat, p_val = stats.ttest_1samp(diff, 0)
            did_rows.append({
                'VARIABLE': dc.replace('_DELTA', ''),
                'MEAN_DELTA_LOW': agg['LOW'].mean(),
                'MEAN_DELTA_HIGH': agg['HIGH'].mean(),
                'MEAN_DIFF': diff.mean(),
                'N_TREATMENT_FIRMS': len(diff),
                'N_RAW_PAIRS': len(merged_did),
                'T_STAT': t_stat,
                'P_VALUE': p_val,
            })

    if did_rows:
        regime_amplification = pd.DataFrame(did_rows)
    else:
        logger.info("Regime amplification: no results produced")
        regime_amplification = pd.DataFrame()

    # --- Test 3: Sign consistency ---
    # Aggregate to treatment-firm level (matching t-test logic) to avoid
    # pseudoreplication when one control appears in multiple pairs.
    sign_rows = []
    for label in labels:
        sub = delta_df[delta_df['REGIME'] == label]
        if sub.empty:
            continue
        for dc in delta_cols:
            agg_vals = (sub.groupby('TREATMENT_TICKER')[dc]
                        .mean()
                        .dropna())
            if len(agg_vals) < 5:
                continue
            n_pos = (agg_vals > 0).sum()
            n_neg = (agg_vals < 0).sum()
            n_total = len(agg_vals)
            majority_sign = 'positive' if n_pos >= n_neg else 'negative'
            consistency = max(n_pos, n_neg) / n_total
            # Binomial test: H0 = 50% positive
            binom_p = stats.binomtest(n_pos, n_total, 0.5).pvalue
            sign_rows.append({
                'REGIME': label,
                'VARIABLE': dc.replace('_DELTA', ''),
                'N_POSITIVE': int(n_pos),
                'N_NEGATIVE': int(n_neg),
                'N_TREATMENT_FIRMS': n_total,
                'PCT_MAJORITY': consistency,
                'MAJORITY_SIGN': majority_sign,
                'BINOMIAL_P': binom_p,
            })

    sign_consistency = pd.DataFrame(sign_rows)

    # Coverage table: which tickers have results in which regimes
    coverage_rows = []
    for ticker, res in ticker_results.items():
        for label in labels:
            sr = res.get(label)
            if sr is None:
                # Ticker had no return data at all (empty returns or
                # missing RETURN column), so _run_stock_ff5 returned {}.
                status = 'NO_DATA'
            elif sr.sufficient_obs:
                status = 'OK'
            elif sr.n_obs < _MIN_REGIME_OBS:
                status = 'INSUFFICIENT_OBS'
            else:
                # n_obs >= _MIN_REGIME_OBS but sufficient_obs is False:
                # the regression was attempted (passed the obs guard)
                # but raised an exception during OLS fitting.
                status = 'REGRESSION_FAILED'
            coverage_rows.append({
                'TICKER': ticker,
                'REGIME': label,
                'HAS_RESULT': sr is not None and sr.sufficient_obs,
                'STATUS': status,
                'N_OBS': sr.n_obs if sr is not None else 0,
            })
    coverage_df = pd.DataFrame(coverage_rows)

    # BH correction — applied per family to avoid cross-DataFrame index
    # collision risk.  All tests address a single hypothesis family
    # (culture war regime effects) so per-family BH at the same q is
    # equivalent in FDR control to cross-family pooling.
    for test_df, pcol in [
        (paired_ttest, 'P_VALUE'),
        (regime_amplification, 'P_VALUE'),
        (sign_consistency, 'BINOMIAL_P'),
    ]:
        if not test_df.empty and pcol in test_df.columns:
            bh_sig = benjamini_hochberg(test_df[pcol].tolist(), q=_BH_FDR_Q)
            test_df['BH_SIGNIFICANT'] = bh_sig

    logger.info("Matched control analysis complete: %d pairs, %d delta obs",
                n_pairs, len(delta_df))

    return MatchedControlResult(
        treatment_results=treatment_df,
        control_results=control_df,
        delta_betas=delta_df,
        paired_ttest=paired_ttest,
        regime_amplification=regime_amplification,
        sign_consistency=sign_consistency,
        n_pairs=n_pairs,
        n_pairs_complete=int(n_complete),
        coverage=coverage_df,
    )


def save_matched_results(
    store: DataStore,
    result: MatchedControlResult,
) -> dict:
    """Persist matched control analysis results to the database."""
    results = {}
    timestamp = pd.Timestamp.now().isoformat()

    results['ESSAY1_MATCHED_DELTAS'] = store.write_table(
        result.delta_betas.assign(RUN_TIMESTAMP=timestamp),
        'ESSAY1_MATCHED_DELTAS', replace=True,
    )

    results['ESSAY1_MATCHED_TTEST'] = store.write_table(
        result.paired_ttest.assign(RUN_TIMESTAMP=timestamp),
        'ESSAY1_MATCHED_TTEST', replace=True,
    )

    if not result.regime_amplification.empty:
        results['ESSAY1_MATCHED_AMPLIFICATION'] = store.write_table(
            result.regime_amplification.assign(RUN_TIMESTAMP=timestamp),
            'ESSAY1_MATCHED_AMPLIFICATION', replace=True,
        )

    if not result.sign_consistency.empty:
        results['ESSAY1_MATCHED_SIGN'] = store.write_table(
            result.sign_consistency.assign(RUN_TIMESTAMP=timestamp),
            'ESSAY1_MATCHED_SIGN', replace=True,
        )

    if not result.coverage.empty:
        results['ESSAY1_MATCHED_COVERAGE'] = store.write_table(
            result.coverage.assign(RUN_TIMESTAMP=timestamp),
            'ESSAY1_MATCHED_COVERAGE', replace=True,
        )

    saved = sum(1 for v in results.values() if v is not None)
    logger.info("Matched control: saved %d/%d tables", saved, len(results))
    return results


# =========================================================================
# FULL ESSAY 1 PIPELINE
# =========================================================================

@dataclass
class FullEssay1Result:
    """Complete Essay 1 results: regime estimation, FF5 analysis,
    culture war stocks, matched controls, sentiment, and macro."""
    regime_result: Optional[RegimeResult] = None
    model_selection: Optional[pd.DataFrame] = None
    ff5_analysis: Optional[FF5RegimeAnalysis] = None
    macro_controls: Optional[pd.DataFrame] = None
    cw_analysis: Optional[CultureWarRegimeAnalysis] = None
    sentiment_analysis: Optional[SentimentRegimeAnalysis] = None
    matched_control: Optional[MatchedControlResult] = None


def run_full_analysis(
    store: DataStore,
    n_regimes: int = 3,
    run_model_selection: bool = True,
    run_sentiment: bool = True,
    news_path: str = None,
    finbert_pipeline=None,
) -> FullEssay1Result:
    """
    Run the complete Essay 1 pipeline.

    Steps
    -----
    1. Markov regime-switching on VIX
    2. Model selection (K=2,3,4)
    3. FF5 spanning regression by regime (+ Chow test + interaction model)
    4. Macro controls assembly
    5. Culture war stocks by regime (unmatched)
    6. Matched control FF5 analysis (+ paired t-test + DiD + sign consistency)
    7. FinBERT sentiment & FOMO z-scores

    Parameters
    ----------
    store : DataStore
    n_regimes : int
        Number of regimes for the primary analysis.
    run_model_selection : bool
        If True, compare K=2,3,4 regime models.
    run_sentiment : bool
        If True, run FinBERT sentiment scoring (requires ~440MB model).
    news_path : str, optional
        Path to news CSV for sentiment analysis.
    finbert_pipeline : optional
        Pre-loaded FinBERT pipeline to avoid reloading.

    Returns
    -------
    FullEssay1Result
    """
    result = FullEssay1Result()

    # Step 1: Estimate VIX regimes
    logger.info("Step 1: Markov regime-switching on VIX")
    result.regime_result = estimate_vix_regimes(store, n_regimes=n_regimes)
    if result.regime_result is None:
        logger.error("Regime estimation failed — aborting pipeline")
        return result

    # Step 2: Model selection
    if run_model_selection:
        logger.info("Step 2: Model selection (K=2,3,4)")
        result.model_selection = select_n_regimes(store)

    # Step 3: FF5 spanning regression by regime
    logger.info("Step 3: FF5 factor analysis by regime")
    result.ff5_analysis = ff5_by_regime(
        store, regime_result=result.regime_result)

    # Step 4: Macro controls
    logger.info("Step 4: Macro controls assembly")
    result.macro_controls = assemble_macro_controls(store)

    # Step 5: Culture war stocks by regime (unmatched)
    logger.info("Step 5: Culture war stocks by regime")
    result.cw_analysis = culture_war_by_regime(
        store, regime_result=result.regime_result)

    # Step 6: Matched control FF5 analysis
    logger.info("Step 6: Matched control FF5 analysis")
    result.matched_control = ff5_matched_control_analysis(
        store, regime_result=result.regime_result, n_regimes=n_regimes)

    # Step 7: FinBERT sentiment & FOMO z-scores
    if run_sentiment:
        logger.info("Step 7: FinBERT sentiment & FOMO z-scores")
        result.sentiment_analysis = sentiment_by_regime(
            store, regime_result=result.regime_result,
            n_regimes=n_regimes, news_path=news_path,
            finbert_pipeline=finbert_pipeline,
        )

    return result


def save_all_results(
    store: DataStore,
    result: FullEssay1Result,
) -> dict:
    """Persist all Essay 1 results to the database."""
    all_saved = {}

    # Save model selection (K=2,3,4 comparison)
    if result.model_selection is not None and not result.model_selection.empty:
        all_saved['ESSAY1_MODEL_SELECTION'] = store.write_table(
            result.model_selection,
            'ESSAY1_MODEL_SELECTION', replace=True,
        )

    # Save essay1.py results (FF5, culture war, sentiment)
    essay1_saved = save_essay1_results(
        store,
        ff5_analysis=result.ff5_analysis,
        cw_analysis=result.cw_analysis,
        sentiment_analysis=result.sentiment_analysis,
    )
    all_saved.update(essay1_saved)

    # Save matched control results
    if result.matched_control is not None:
        matched_saved = save_matched_results(store, result.matched_control)
        all_saved.update(matched_saved)

    logger.info("Saved all Essay 1 results: %d tables", len(all_saved))
    return all_saved


# =========================================================================
# MAIN — run and test
# =========================================================================

if __name__ == '__main__':
    from datetime import datetime

    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
    )

    print("Dissertation Essay 1 — Full Analysis with Matched Controls")
    print(f"Started at {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print()

    store = DataStore()

    # Step 1: Estimate VIX regimes
    print("=" * 60)
    print("  Step 1: Markov regime-switching on VIX")
    print("=" * 60)
    regime_result = estimate_vix_regimes(store, n_regimes=3)
    if regime_result is not None:
        print(f"  Regimes: {regime_result.n_regimes}")
        for label, mean in regime_result.regime_means.items():
            n = regime_result.regime_summary.loc[
                regime_result.regime_summary['REGIME'] == label, 'N_DAYS'
            ].values[0]
            dur = regime_result.expected_durations[label]
            print(f"    {label}: mean VIX={mean:.1f} ({n} days, E[dur]={dur:.0f}d)")
        print(f"  AIC={regime_result.aic:.1f}, BIC={regime_result.bic:.1f}")
    else:
        print("  FAILED — no VIX data or estimation error")

    # Step 2: Model selection
    print()
    print("=" * 60)
    print("  Step 2: Model selection (K=2,3,4)")
    print("=" * 60)
    selection = select_n_regimes(store)
    if not selection.empty:
        for _, row in selection.iterrows():
            print(f"    K={int(row['K'])}: AIC={row['AIC']:.1f}, "
                  f"BIC={row['BIC']:.1f}, LL={row['LOG_LIKELIHOOD']:.1f}")

    # Step 3: FF5 by regime
    print()
    print("=" * 60)
    print("  Step 3: FF5 factor analysis by regime")
    print("=" * 60)
    ff5_analysis = ff5_by_regime(store, regime_result=regime_result)

    if ff5_analysis is not None:
        chow = ff5_analysis.chow_test
        print(f"  Chow test: F={chow['f_stat']:.2f}, p={chow['p_value']:.4f}")
        print(f"  Structural break: {'YES' if chow['significant_005'] else 'NO'} (p<0.05)")
        if not ff5_analysis.coefficient_comparison.empty:
            print("  Coefficient comparison:")
            print(ff5_analysis.coefficient_comparison.to_string(index=False))
    else:
        print("  FAILED — insufficient data")

    # Step 4: Macro controls
    print()
    print("=" * 60)
    print("  Step 4: Macro controls")
    print("=" * 60)
    macro = assemble_macro_controls(store)
    print(f"  Macro panel: {len(macro)} rows, {len(macro.columns)-1} variables")

    # Step 5: Culture war stocks by regime
    print()
    print("=" * 60)
    print("  Step 5: Culture war stocks by regime")
    print("=" * 60)
    cw = culture_war_by_regime(store, regime_result=regime_result)
    if cw is not None:
        print(f"  Stocks analysed: {cw.n_stocks}")
        print(f"  Stocks failed: {cw.n_failed}")
        if not cw.summary.empty:
            print(f"  Summary rows: {len(cw.summary)}")
    else:
        print("  FAILED — no event tickers or data")

    # Step 6: Matched control analysis
    print()
    print("=" * 60)
    print("  Step 6: Matched control FF5 analysis")
    print("=" * 60)
    matched = ff5_matched_control_analysis(store, regime_result=regime_result)

    if matched is not None:
        print(f"  Pairs: {matched.n_pairs} total, {matched.n_pairs_complete} complete")
        print(f"  Treatment results: {len(matched.treatment_results)} rows")
        print(f"  Control results:   {len(matched.control_results)} rows")
        print(f"  Delta betas:       {len(matched.delta_betas)} rows")

        # Paired t-test summary
        if not matched.paired_ttest.empty:
            print()
            print("  --- Paired t-test (delta = 0) ---")
            for _, row in matched.paired_ttest.iterrows():
                sig = "*" if row['P_VALUE'] < 0.05 else ""
                bh_flag = " [BH]" if row.get('BH_SIGNIFICANT', False) else ""
                print(f"    {row['REGIME']:20s} {row['VARIABLE']:10s} "
                      f"delta={row['MEAN_DELTA']:+.4f} t={row['T_STAT']:+.2f} "
                      f"p={row['P_VALUE']:.4f}{sig}{bh_flag}")

        # Regime amplification
        if not matched.regime_amplification.empty:
            print()
            print("  --- Regime amplification (High - Low) ---")
            for _, row in matched.regime_amplification.iterrows():
                sig = "*" if row['P_VALUE'] < 0.05 else ""
                print(f"    {row['VARIABLE']:10s} diff={row['MEAN_DIFF']:+.4f} "
                      f"t={row['T_STAT']:+.2f} p={row['P_VALUE']:.4f}{sig}")

        # Sign consistency
        if not matched.sign_consistency.empty:
            print()
            print("  --- Sign consistency ---")
            for _, row in matched.sign_consistency.iterrows():
                print(f"    {row['REGIME']:20s} {row['VARIABLE']:10s} "
                      f"{row['PCT_MAJORITY']:.0%} {row['MAJORITY_SIGN']} "
                      f"n={row['N_TREATMENT_FIRMS']} "
                      f"(binom p={row['BINOMIAL_P']:.4f})")

        # Coverage
        if not matched.coverage.empty:
            total = len(matched.coverage)
            covered = matched.coverage['HAS_RESULT'].sum()
            print(f"\n  Coverage: {covered}/{total} ticker-regime slots "
                  f"({covered/total:.0%})")
    else:
        print("  FAILED — no control companies or data")

    # Step 7: FinBERT sentiment & FOMO z-scores
    print()
    print("=" * 60)
    print("  Step 7: FinBERT sentiment & FOMO z-scores")
    print("=" * 60)
    sentiment = None
    if regime_result is not None:
        news_path = Path(__file__).resolve().parent.parent / 'news_data' / 'culture_war_news.csv'
        sentiment = sentiment_by_regime(
            store, regime_result=regime_result,
            n_regimes=regime_result.n_regimes,
            news_path=news_path,
        )
        if sentiment is not None:
            print(f"  Articles loaded: {sentiment.n_articles}")
            print(f"  Articles scored: {sentiment.n_scored}")
            print(f"  Daily rows: {len(sentiment.sentiment_daily)}")
            print(f"  FOMO by regime:")
            for _, row in sentiment.fomo_by_regime.iterrows():
                print(f"    {row['REGIME']}: mean_z={row['MEAN_FOMO_Z']:.3f}, "
                      f"euphoria={row['PCT_EUPHORIA']*100:.1f}%, "
                      f"panic={row['PCT_PANIC']*100:.1f}%")
        else:
            print("  FAILED — no news data or FinBERT error")
    else:
        print("  SKIPPED — no regime result")

    # Step 8: Save all results to database
    print()
    print("=" * 60)
    print("  Step 8: Save results to database")
    print("=" * 60)
    full_result = FullEssay1Result(
        regime_result=regime_result,
        model_selection=selection,
        ff5_analysis=ff5_analysis,
        macro_controls=macro,
        cw_analysis=cw,
        sentiment_analysis=sentiment,
        matched_control=matched,
    )
    all_saved = save_all_results(store, full_result)

    if all_saved:
        for table, res in all_saved.items():
            print(f"    {table}: {res}")
        print(f"  Saved {len(all_saved)} tables")
    else:
        print("  Nothing to save (no results produced)")

    # BH sanity check (diagnostic, not a pipeline step)
    print()
    print("  --- BH sanity check (q=%.2f) ---" % _BH_FDR_Q)
    test_pvals = [0.001, 0.008, 0.039, 0.041, 0.23, 0.45, 0.89]
    bh = benjamini_hochberg(test_pvals, q=_BH_FDR_Q)
    print(f"  Input p-values: {test_pvals}")
    print(f"  Rejected (FDR={_BH_FDR_Q:.0%}): {bh}")

    store.close()
    print()
    print(f"Completed at {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
