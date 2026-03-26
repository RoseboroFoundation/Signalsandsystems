"""
Essay 1 — Matched Control Analysis.

Compares FF5 factor loadings between culture war (treatment) firms and
their industry-matched control firms across volatility regimes.

The identification logic: treatment and control firms are matched on
industry (NAICS), so any residual difference in regime-conditional
factor loadings is attributable to culture war exposure rather than
sector or firm-characteristic effects.

Unlike the spanning regression in essay1.py (MKT_RF ~ SMB + HML + RMW + CMA),
this module uses the full FF5 pricing regression (R_i - RF ~ MKT_RF + SMB +
HML + RMW + CMA) for individual stock returns.  The dependent variable is
each firm's excess return, and all five factors serve as regressors.

WMT-as-multiple-control: regressions run at the unique-firm level so that
a control firm appearing in multiple pairs has its betas estimated once.
Deltas are computed at the pair level, and the paired t-test operates on
pairs, giving each treatment firm exactly one control.

References
----------
Barillas, F. & Shanken, J. (2017). Which alpha? Review of Financial
    Studies, 30(4).
"""

import logging
import warnings
from dataclasses import dataclass, field
from typing import Dict, List, Optional

import numpy as np
import pandas as pd
import statsmodels.api as sm
from scipy import stats

from .datastore import DataStore
from .essay1 import (
    RegimeResult,
    estimate_vix_regimes,
    _FF5_ALL,
    _HAC_MAXLAGS,
    _MIN_REGIME_OBS,
    benjamini_hochberg,
)

logger = logging.getLogger(__name__)


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
        return {}

    ret = returns[['DATE', 'RETURN']].copy()
    ret['DATE'] = pd.to_datetime(ret['DATE'], errors='coerce')
    merged = ret.merge(factor_regime, on='DATE', how='inner')
    merged = merged.dropna(subset=['RETURN'] + factor_cols + ['RF'])
    # Convert percent to decimal if needed (median-based check)
    if merged['RETURN'].abs().median() > 0.1:
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
        if factors[col].abs().median() > 0.1:
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
                row[f'{f}_DELTA'] = t.betas.get(f, np.nan) - c.betas.get(f, np.nan)
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
                'N_PAIRS': len(agg_deltas),
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
    if len(labels) >= 2:
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

            diff = merged_did['HIGH'] - merged_did['LOW']
            t_stat, p_val = stats.ttest_1samp(diff, 0)
            did_rows.append({
                'VARIABLE': dc.replace('_DELTA', ''),
                'MEAN_DELTA_LOW': merged_did['LOW'].mean(),
                'MEAN_DELTA_HIGH': merged_did['HIGH'].mean(),
                'MEAN_DIFF': diff.mean(),
                'N_PAIRS': len(diff),
                'T_STAT': t_stat,
                'P_VALUE': p_val,
            })

    regime_amplification = pd.DataFrame(did_rows)

    # --- Test 3: Sign consistency ---
    sign_rows = []
    for label in labels:
        sub = delta_df[delta_df['REGIME'] == label]
        if sub.empty:
            continue
        for dc in delta_cols:
            vals = sub[dc].dropna()
            if len(vals) < 5:
                continue
            n_pos = (vals > 0).sum()
            n_neg = (vals < 0).sum()
            n_total = len(vals)
            majority_sign = 'positive' if n_pos >= n_neg else 'negative'
            consistency = max(n_pos, n_neg) / n_total
            # Binomial test: H0 = 50% positive
            binom_p = stats.binomtest(n_pos, n_total, 0.5).pvalue
            sign_rows.append({
                'REGIME': label,
                'VARIABLE': dc.replace('_DELTA', ''),
                'N_POSITIVE': int(n_pos),
                'N_NEGATIVE': int(n_neg),
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
    # collision risk. q=0.10: more lenient FDR threshold given limited
    # pair count. All tests address a single hypothesis family (culture
    # war regime effects) so per-family BH at the same q is equivalent
    # in FDR control to cross-family pooling.
    for test_df, pcol in [
        (paired_ttest, 'P_VALUE'),
        (regime_amplification, 'P_VALUE'),
        (sign_consistency, 'BINOMIAL_P'),
    ]:
        if not test_df.empty and pcol in test_df.columns:
            bh_sig = benjamini_hochberg(test_df[pcol].tolist(), q=0.10)
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
# MAIN — run and test
# =========================================================================

if __name__ == '__main__':
    from datetime import datetime

    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
    )

    print("Dissertation Essay 1 — Matched Control Analysis")
    print(f"Started at {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print()

    store = DataStore()

    # Step 1: Estimate regimes (shared with essay1.py)
    print("=" * 60)
    print("  Step 1: Estimate VIX regimes")
    print("=" * 60)
    regime_result = estimate_vix_regimes(store, n_regimes=3)
    if regime_result is not None:
        for label, mean in regime_result.regime_means.items():
            print(f"    {label}: mean VIX={mean:.1f}")
    else:
        print("  FAILED — no VIX data")

    # Step 2: Matched control analysis
    print()
    print("=" * 60)
    print("  Step 2: Matched control FF5 analysis")
    print("=" * 60)
    result = ff5_matched_control_analysis(store, regime_result=regime_result)

    if result is not None:
        print(f"  Pairs: {result.n_pairs} total, {result.n_pairs_complete} complete")
        print(f"  Treatment results: {len(result.treatment_results)} rows")
        print(f"  Control results:   {len(result.control_results)} rows")
        print(f"  Delta betas:       {len(result.delta_betas)} rows")

        # Paired t-test summary
        if not result.paired_ttest.empty:
            print()
            print("  --- Paired t-test (delta = 0) ---")
            for _, row in result.paired_ttest.iterrows():
                sig = "*" if row['P_VALUE'] < 0.05 else ""
                bh = " [BH]" if row.get('BH_SIGNIFICANT', False) else ""
                print(f"    {row['REGIME']:20s} {row['VARIABLE']:10s} "
                      f"delta={row['MEAN_DELTA']:+.4f} t={row['T_STAT']:+.2f} "
                      f"p={row['P_VALUE']:.4f}{sig}{bh}")

        # Regime amplification
        if not result.regime_amplification.empty:
            print()
            print("  --- Regime amplification (High - Low) ---")
            for _, row in result.regime_amplification.iterrows():
                sig = "*" if row['P_VALUE'] < 0.05 else ""
                print(f"    {row['VARIABLE']:10s} diff={row['MEAN_DIFF']:+.4f} "
                      f"t={row['T_STAT']:+.2f} p={row['P_VALUE']:.4f}{sig}")

        # Sign consistency
        if not result.sign_consistency.empty:
            print()
            print("  --- Sign consistency ---")
            for _, row in result.sign_consistency.iterrows():
                print(f"    {row['REGIME']:20s} {row['VARIABLE']:10s} "
                      f"{row['PCT_MAJORITY']:.0%} {row['MAJORITY_SIGN']} "
                      f"(binom p={row['BINOMIAL_P']:.4f})")

        # Coverage
        if not result.coverage.empty:
            total = len(result.coverage)
            covered = result.coverage['HAS_RESULT'].sum()
            print(f"\n  Coverage: {covered}/{total} ticker-regime slots "
                  f"({covered/total:.0%})")
    else:
        print("  FAILED — no control companies or data")

    # Step 3: Save results to database
    print()
    print("=" * 60)
    print("  Step 3: Save results to database")
    print("=" * 60)
    if result is not None:
        saved = save_matched_results(store, result)
        if saved:
            for table, res in saved.items():
                print(f"    {table}: {res}")
            print(f"  Saved {len(saved)} tables")
        else:
            print("  Nothing saved")
    else:
        print("  Skipped — no results to save")

    store.close()
    print()
    print(f"Completed at {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
