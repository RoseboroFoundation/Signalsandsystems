"""Orthogonalized Culture Factor — Integration with Essays 1, 2, and 3.

Runs the Baker-Wurgler (2006) orthogonalization from clean.orthogonalize,
then tests whether each essay's key findings survive after controlling for
the political component of the culture factor.

Design:
  - Builds monthly CULTURE_RAW and CULTURE_ORTHO time series
  - For each essay, re-runs the core specification with both factors
  - Reports horse-race results: does the orthogonalized factor dominate?

This answers Dr. Hu's critique: if culture is confounded with politics,
controlling for politics should absorb the culture effect.  If culture
survives orthogonalization, the factor captures something distinct.
"""

import logging
import warnings
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, Optional

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
)

logger = logging.getLogger(__name__)


# ═════════════════════════════════════════════════════════════════════
# DATA CLASSES
# ═════════════════════════════════════════════════════════════════════

@dataclass
class HorseRaceResult:
    """Comparison of raw vs orthogonalized culture factor in a regression."""
    spec_name: str
    n_obs: int
    # Raw culture factor
    raw_beta: float = np.nan
    raw_t: float = np.nan
    raw_p: float = np.nan
    # Orthogonalized culture factor
    ortho_beta: float = np.nan
    ortho_t: float = np.nan
    ortho_p: float = np.nan
    # Horse race (both included)
    horse_raw_beta: float = np.nan
    horse_raw_t: float = np.nan
    horse_raw_p: float = np.nan
    horse_ortho_beta: float = np.nan
    horse_ortho_t: float = np.nan
    horse_ortho_p: float = np.nan
    # Model fit
    r2_raw: float = np.nan
    r2_ortho: float = np.nan
    r2_horse: float = np.nan


@dataclass
class IntegrationResult:
    """Full integration results across all three essays."""
    # Orthogonalization summary
    first_stage_r2: float = np.nan
    n_political_proxies: int = 0
    estimation_method: str = ''
    # Essay results
    essay1_results: pd.DataFrame = field(default_factory=pd.DataFrame)
    essay2_results: pd.DataFrame = field(default_factory=pd.DataFrame)
    essay3_results: pd.DataFrame = field(default_factory=pd.DataFrame)
    # Combined summary
    summary: pd.DataFrame = field(default_factory=pd.DataFrame)


# ═════════════════════════════════════════════════════════════════════
# CULTURE FACTOR CONSTRUCTION
# ═════════════════════════════════════════════════════════════════════

def _build_culture_factors(store: DataStore):
    """Build raw and orthogonalized monthly culture factors.

    Returns (DataFrame with CULTURE_RAW/CULTURE_ORTHO, OrthogonalizationResult)
    or (None, None) on failure.
    """
    from clean.orthogonalize import (
        build_political_proxies,
        build_culture_index,
        orthogonalize_culture,
    )

    # Load events
    events_df = store.read_table('CULTURE_WAR_COMPANIES')
    if events_df.empty:
        logger.error("No CULTURE_WAR_COMPANIES table")
        return None, None

    events_df.columns = [c.upper() for c in events_df.columns]

    # Rename columns for orthogonalize module expectations
    col_map = {}
    for c in events_df.columns:
        if 'EVENT' in c and 'DATE' in c:
            col_map[c] = 'Event Date'
        if 'ESTIMATED' in c and 'POLITICAL' in c:
            col_map[c] = 'Estimated Political Leaning'
    events_df = events_df.rename(columns=col_map)
    events_df['Event Date'] = pd.to_datetime(events_df['Event Date'], errors='coerce')

    # Build culture index
    culture_raw = build_culture_index(events_df, freq='M', method='net_score')
    if culture_raw.std() < 1e-10:
        logger.error("Culture index has zero/near-zero variance")
        return None, None

    # Build political proxies
    political = build_political_proxies(freq='M')

    # Orthogonalize (full sample for descriptive; rolling for robustness)
    ortho_result = orthogonalize_culture(
        culture_raw, political, method='full_sample')

    culture_ortho = ortho_result.culture_orthogonal

    # Align and return
    df = pd.DataFrame({
        'CULTURE_RAW': culture_raw,
        'CULTURE_ORTHO': culture_ortho,
    })
    df.index.name = 'DATE'
    df = df.dropna()

    logger.info("Culture factors: %d months, first-stage R²=%.4f",
                len(df), ortho_result.first_stage_r2)
    return df, ortho_result


def _merge_culture_to_daily(culture_monthly: pd.DataFrame,
                             daily_dates: pd.Series) -> pd.DataFrame:
    """Map monthly culture factors to daily dates using one-month lag.

    Each trading day in month t is assigned the culture factor from month
    t-1 to avoid within-month look-ahead bias (Pagan 1984).
    """
    daily = pd.DataFrame({'DATE': pd.to_datetime(daily_dates)})
    # Lag by one month: assign month t-1 factor to month t dates
    daily['MONTH_START'] = (daily['DATE'].dt.to_period('M') - 1).dt.to_timestamp()

    culture = culture_monthly.copy()
    culture = culture.reset_index()
    culture.rename(columns={'DATE': 'MONTH_START'}, inplace=True)

    merged = daily.merge(culture, on='MONTH_START', how='left')
    return merged.set_index('DATE')[['CULTURE_RAW', 'CULTURE_ORTHO']]


# ═════════════════════════════════════════════════════════════════════
# ESSAY 1: FF5 + Culture Factor in Stock-Level Regressions
# ═════════════════════════════════════════════════════════════════════

def _run_essay1_integration(
    store: DataStore,
    culture_factors: pd.DataFrame,
    regime_result: RegimeResult,
) -> pd.DataFrame:
    """Test whether culture factor adds to FF5 in explaining CW stock returns.

    For each regime, runs three specs on pooled CW stock excess returns:
      1. FF5 + CULTURE_RAW
      2. FF5 + CULTURE_ORTHO
      3. FF5 + CULTURE_RAW + CULTURE_ORTHO (horse race)
    """
    events_df = store.read_table('CULTURE_WAR_COMPANIES')
    if events_df.empty:
        return pd.DataFrame()

    events_df.columns = [c.upper() for c in events_df.columns]
    tickers = events_df['TICKER'].dropna().unique().tolist()

    # Prepare factors
    factors = store.ff5[['DATE'] + _FF5_ALL + ['RF']].dropna().copy()
    factors['DATE'] = pd.to_datetime(factors['DATE'], errors='coerce')
    for col in _FF5_ALL + ['RF']:
        if (factors[col].abs() > 1.5).mean() > 0.10:
            factors[col] = factors[col] / 100

    # Merge with regime assignments
    regime_df = regime_result.regime_assignments[['DATE', 'REGIME_LABEL']].copy()
    regime_df['DATE'] = pd.to_datetime(regime_df['DATE'])
    factors = factors.merge(regime_df, on='DATE', how='inner')

    # Add culture factors (daily via month mapping)
    culture_daily = _merge_culture_to_daily(culture_factors, factors['DATE'])
    factors = factors.join(culture_daily, on='DATE')
    factors = factors.dropna(subset=['CULTURE_RAW', 'CULTURE_ORTHO'])

    # Collect pooled returns for all CW tickers
    all_rows = []
    for ticker in tickers:
        ret = store.get_ticker_returns(ticker)
        if ret.empty or 'RETURN' not in ret.columns:
            continue
        ret = ret[['DATE', 'RETURN']].copy()
        ret['DATE'] = pd.to_datetime(ret['DATE'], errors='coerce')
        ret['TICKER'] = ticker
        all_rows.append(ret)

    if not all_rows:
        logger.warning("Essay 1 integration: no return data")
        return pd.DataFrame()

    returns = pd.concat(all_rows, ignore_index=True)
    merged = returns.merge(factors, on='DATE', how='inner')
    merged = merged.dropna(subset=['RETURN'] + _FF5_ALL + ['RF'])
    if (merged['RETURN'].abs() > 1.5).mean() > 0.10:
        merged['RETURN'] = merged['RETURN'] / 100
    merged['EXCESS_RETURN'] = merged['RETURN'] - merged['RF']

    labels = sorted(regime_result.regime_means.keys(),
                    key=lambda x: regime_result.regime_means[x])

    results = []
    for regime in labels + ['ALL']:
        sub = merged if regime == 'ALL' else merged[merged['REGIME_LABEL'] == regime]
        if len(sub) < 30:
            continue

        y = sub['EXCESS_RETURN']
        X_base = sub[_FF5_ALL]

        # Note: raw and ortho have ~0.98 correlation so including both
        # would produce near-perfect multicollinearity (VIF ~28+).
        # We compare each separately against the FF5 baseline instead.
        for spec_name, extra_cols in [
            ('FF5 + RAW', ['CULTURE_RAW']),
            ('FF5 + ORTHO', ['CULTURE_ORTHO']),
        ]:
            X = sm.add_constant(pd.concat([X_base, sub[extra_cols]], axis=1))
            try:
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore")
                    fit = sm.OLS(y, X).fit(
                        cov_type='HAC', cov_kwds={'maxlags': _HAC_MAXLAGS})

                row = {
                    'ESSAY': 1,
                    'REGIME': regime,
                    'SPEC': spec_name,
                    'N_OBS': int(fit.nobs),
                    'R2': fit.rsquared,
                    'ADJ_R2': fit.rsquared_adj,
                    'F_STAT': fit.fvalue,
                    'F_PVAL': fit.f_pvalue,
                }
                for col in extra_cols:
                    if col in fit.params.index:
                        row[f'{col}_BETA'] = fit.params[col]
                        row[f'{col}_T'] = fit.tvalues[col]
                        row[f'{col}_P'] = fit.pvalues[col]
                results.append(row)
            except Exception as e:
                logger.warning("Essay 1 %s/%s failed: %s", regime, spec_name, e)

    return pd.DataFrame(results)


# ═════════════════════════════════════════════════════════════════════
# ESSAY 2: DiD with Culture Factor Control
# ═════════════════════════════════════════════════════════════════════

def _run_essay2_integration(
    store: DataStore,
    culture_factors: pd.DataFrame,
) -> pd.DataFrame:
    """Test whether treatment-control CAR gap survives culture factor control.

    Cross-sectional regression on post-event CARs (not a full DiD panel —
    the multi-window panel contains only post-event CARs by construction):
      CAR = a + b1*Treat + b2*CULTURE + e

    The coefficient on TREAT measures the treatment-control CAR difference;
    adding the culture factor tests whether the aggregate culture climate
    absorbs the cross-sectional treatment effect.
    """
    from .essay2_did import build_multi_window_panel, _MULTI_WINDOWS

    panel = build_multi_window_panel(store)
    if panel is None or panel.empty:
        logger.warning("Essay 2 integration: no panel data")
        return pd.DataFrame()

    # Add culture factors by event date month (lagged to avoid look-ahead)
    panel['EVENT_DATE'] = pd.to_datetime(panel['EVENT_DATE'], errors='coerce')
    panel['MONTH_START'] = (
        panel['EVENT_DATE'].dt.to_period('M') - 1
    ).dt.to_timestamp()

    culture = culture_factors.reset_index()
    culture.rename(columns={'DATE': 'MONTH_START'}, inplace=True)
    panel = panel.merge(culture, on='MONTH_START', how='left')
    panel = panel.dropna(subset=['CULTURE_RAW', 'CULTURE_ORTHO', 'CAR'])

    if len(panel) < 20:
        logger.warning("Essay 2 integration: insufficient obs after merge (%d)", len(panel))
        return pd.DataFrame()

    panel['TREAT'] = panel['IS_TREATMENT'].astype(float)

    results = []
    # Test key windows
    test_windows = [w for w in [10, 30, 60] if w in panel['WINDOW'].unique()]
    if not test_windows:
        test_windows = panel['WINDOW'].unique()[:3]

    for window in test_windows:
        sub = panel[panel['WINDOW'] == window].copy()
        if len(sub) < 20:
            continue

        y = sub['CAR']
        base_vars = ['TREAT']

        # No horse-race (raw+ortho) due to ~0.98 correlation
        for spec_name, culture_cols in [
            ('Baseline', []),
            ('+ RAW culture', ['CULTURE_RAW']),
            ('+ ORTHO culture', ['CULTURE_ORTHO']),
        ]:
            X_cols = base_vars + culture_cols
            X = sm.add_constant(sub[X_cols])
            try:
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore")
                    fit = sm.OLS(y, X).fit(cov_type='HC1')

                row = {
                    'ESSAY': 2,
                    'WINDOW': f'[-1, +{window}]',
                    'SPEC': spec_name,
                    'N_OBS': int(fit.nobs),
                    'R2': fit.rsquared,
                    'TREAT_BETA': fit.params['TREAT'] if 'TREAT' in fit.params.index else np.nan,
                    'TREAT_T': fit.tvalues['TREAT'] if 'TREAT' in fit.tvalues.index else np.nan,
                    'TREAT_P': fit.pvalues['TREAT'] if 'TREAT' in fit.pvalues.index else np.nan,
                }
                for col in culture_cols:
                    if col in fit.params.index:
                        row[f'{col}_BETA'] = fit.params[col]
                        row[f'{col}_T'] = fit.tvalues[col]
                        row[f'{col}_P'] = fit.pvalues[col]
                results.append(row)
            except Exception as e:
                logger.warning("Essay 2 w=%d %s failed: %s", window, spec_name, e)

    return pd.DataFrame(results)


# ═════════════════════════════════════════════════════════════════════
# ESSAY 3: Insider Trading Profitability with Culture Factor
# ═════════════════════════════════════════════════════════════════════

def _run_essay3_integration(
    store: DataStore,
    culture_factors: pd.DataFrame,
) -> pd.DataFrame:
    """Test whether insider trading accuracy relates to culture factor.

    Uses the Essay 3 panel (if available) and runs:
      Profitable ~ Political + CULTURE + controls

    to see if the culture factor captures additional variation in
    insider trading profitability beyond political event classification.
    """
    panel = store.read_table('ESSAY3_PANEL')
    if panel.empty:
        # Try building from scratch if panel not cached
        logger.warning("Essay 3 integration: ESSAY3_PANEL not found, "
                       "attempting to build from Form 4 data")
        try:
            from .essay3 import _load_form4, _build_ticker_naics_map
            form4 = _load_form4()
            if form4.empty:
                logger.warning("Essay 3 integration: no Form 4 data")
                return pd.DataFrame()
        except Exception as e:
            logger.warning("Essay 3 integration: cannot load data: %s", e)
            return pd.DataFrame()
        return pd.DataFrame()

    panel.columns = [c.upper() for c in panel.columns]

    # Need a date column to merge culture factors
    date_col = None
    for c in ['TRANSACTION_DATE', 'EVENT_DATE', 'DATE', 'FILING_DATE']:
        if c in panel.columns:
            date_col = c
            break
    if date_col is None:
        logger.warning("Essay 3 integration: no date column in panel")
        return pd.DataFrame()

    panel[date_col] = pd.to_datetime(panel[date_col], errors='coerce')
    # Lag by one month to avoid look-ahead bias
    panel['MONTH_START'] = (
        panel[date_col].dt.to_period('M') - 1
    ).dt.to_timestamp()

    culture = culture_factors.reset_index()
    culture.rename(columns={'DATE': 'MONTH_START'}, inplace=True)
    panel = panel.merge(culture, on='MONTH_START', how='left')

    # Find the profitability column
    profit_col = None
    for c in ['EVENT_PROFITABLE', 'PROFITABLE', 'PROFITABLE_30',
              'DIRECTIONALLY_ACCURATE']:
        if c in panel.columns:
            profit_col = c
            break
    if profit_col is None:
        logger.warning("Essay 3 integration: no profitability column")
        return pd.DataFrame()

    panel = panel.dropna(subset=[profit_col, 'CULTURE_RAW', 'CULTURE_ORTHO'])
    if len(panel) < 30:
        logger.warning("Essay 3 integration: insufficient obs (%d)", len(panel))
        return pd.DataFrame()

    y = panel[profit_col].astype(float)

    # Build control variables from whatever's available
    possible_controls = ['IS_SELL', 'LOG_TRADE_VALUE', 'PROXIMITY_DAYS',
                         'IS_POLITICAL']
    controls = [c for c in possible_controls if c in panel.columns]

    # Cluster groups for cluster-robust SEs (firm-level)
    ticker_col = 'TICKER' if 'TICKER' in panel.columns else None

    results = []
    # No horse-race (raw+ortho) due to ~0.98 correlation
    for spec_name, culture_cols in [
        ('Baseline', []),
        ('+ RAW', ['CULTURE_RAW']),
        ('+ ORTHO', ['CULTURE_ORTHO']),
    ]:
        X_cols = controls + culture_cols
        if not X_cols:
            X = sm.add_constant(pd.DataFrame(index=panel.index))
        else:
            X = sm.add_constant(panel[X_cols])

        try:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                # LPM with cluster-robust SEs at firm level if available
                if ticker_col and len(panel[ticker_col].unique()) > 1:
                    fit = sm.OLS(y, X).fit(
                        cov_type='cluster',
                        cov_kwds={'groups': panel[ticker_col]})
                else:
                    fit = sm.OLS(y, X).fit(cov_type='HC3')

            row = {
                'ESSAY': 3,
                'SPEC': spec_name,
                'DEP_VAR': profit_col,
                'N_OBS': int(fit.nobs),
                'R2': fit.rsquared,
            }
            for col in culture_cols:
                if col in fit.params.index:
                    row[f'{col}_BETA'] = fit.params[col]
                    row[f'{col}_T'] = fit.tvalues[col]
                    row[f'{col}_P'] = fit.pvalues[col]
            for col in controls:
                if col in fit.params.index:
                    row[f'{col}_BETA'] = fit.params[col]
                    row[f'{col}_P'] = fit.pvalues[col]
            results.append(row)
        except Exception as e:
            logger.warning("Essay 3 %s failed: %s", spec_name, e)

    return pd.DataFrame(results)


# ═════════════════════════════════════════════════════════════════════
# MAIN PIPELINE
# ═════════════════════════════════════════════════════════════════════

def run_orthogonal_integration(store: DataStore = None) -> IntegrationResult:
    """Run the full orthogonalized culture factor integration.

    Returns IntegrationResult with comparative results for all three essays.
    """
    if store is None:
        store = DataStore()

    result = IntegrationResult()

    # Step 1: Build culture factors
    print("=" * 70)
    print("  ORTHOGONALIZED CULTURE FACTOR — FULL INTEGRATION")
    print("  Baker-Wurgler (2006) purging of political confounds")
    print("=" * 70)
    print()

    print("Step 1: Building culture factors...")
    culture_factors, ortho_result = _build_culture_factors(store)
    if culture_factors is None:
        print("  FAILED — cannot build culture factors")
        return result

    result.first_stage_r2 = ortho_result.first_stage_r2
    result.n_political_proxies = len(ortho_result.political_proxies_used)
    result.estimation_method = ortho_result.estimation_method

    print(f"  Culture index: {len(culture_factors)} months")
    print(f"  First-stage R² = {ortho_result.first_stage_r2:.4f} "
          f"({ortho_result.first_stage_r2*100:.1f}% political)")
    print(f"  Political proxies: {', '.join(ortho_result.political_proxies_used)}")
    print(f"  Correlation(raw, ortho) = "
          f"{culture_factors['CULTURE_RAW'].corr(culture_factors['CULTURE_ORTHO']):.3f}")
    print()

    # First-stage coefficients
    print("  First-stage coefficients (Culture = a + b'Political + e):")
    coef_df = ortho_result.first_stage_coefficients
    for _, row in coef_df.iterrows():
        if row['VARIABLE'] == 'const':
            continue
        beta = row['COEFFICIENT']
        pval = row['P_VALUE']
        sig = '***' if pval < 0.01 else '**' if pval < 0.05 else '*' if pval < 0.10 else ''
        print(f"    {row['VARIABLE']:25s}  beta={beta:+.4f}  p={pval:.4f} {sig}")
    print()

    # Step 2: Essay 1
    print("Step 2: Essay 1 — FF5 + Culture Factor by Regime...")
    regime_result = estimate_vix_regimes(store, n_regimes=3)
    if regime_result is not None:
        result.essay1_results = _run_essay1_integration(
            store, culture_factors, regime_result)
        if not result.essay1_results.empty:
            _print_essay1_results(result.essay1_results)
        else:
            print("  No results produced")
    else:
        print("  SKIPPED — regime estimation failed")
    print()

    # Step 3: Essay 2
    print("Step 3: Essay 2 — Cross-Sectional CAR with Culture Control...")
    result.essay2_results = _run_essay2_integration(store, culture_factors)
    if not result.essay2_results.empty:
        _print_essay2_results(result.essay2_results)
    else:
        print("  No results produced (may need ESSAY2_POLITICAL_ALIGNMENT table)")
    print()

    # Step 4: Essay 3
    print("Step 4: Essay 3 — Insider Trading + Culture Factor...")
    result.essay3_results = _run_essay3_integration(store, culture_factors)
    if not result.essay3_results.empty:
        _print_essay3_results(result.essay3_results)
    else:
        print("  No results produced (may need ESSAY3_PANEL table)")
    print()

    # Step 5: Summary
    print("=" * 70)
    print("  SUMMARY: Does Culture Survive Orthogonalization?")
    print("=" * 70)
    _print_summary(result)

    return result


# ═════════════════════════════════════════════════════════════════════
# DISPLAY HELPERS
# ═════════════════════════════════════════════════════════════════════

def _sig_stars(p):
    if pd.isna(p):
        return ''
    if p < 0.01:
        return '***'
    if p < 0.05:
        return '**'
    if p < 0.10:
        return '*'
    return ''


def _print_essay1_results(df: pd.DataFrame):
    print()
    print("  Essay 1: FF5 + Culture Factor (pooled CW stocks, lagged 1 month)")
    print("  " + "-" * 68)
    print(f"  {'Regime':<16} {'Spec':<14} {'N':>7} {'R²':>7} "
          f"{'Culture β':>11} {'t':>7} {'p':>8}")
    print("  " + "-" * 68)

    for _, r in df.iterrows():
        # Show whichever culture column is in this spec
        cult_b = r.get('CULTURE_ORTHO_BETA', r.get('CULTURE_RAW_BETA', np.nan))
        cult_t = r.get('CULTURE_ORTHO_T', r.get('CULTURE_RAW_T', np.nan))
        cult_p = r.get('CULTURE_ORTHO_P', r.get('CULTURE_RAW_P', np.nan))

        b_str = f"{cult_b:+.6f}" if pd.notna(cult_b) else "          "
        t_str = f"{cult_t:+.2f}" if pd.notna(cult_t) else "      "
        p_str = f"{cult_p:.4f}{_sig_stars(cult_p)}" if pd.notna(cult_p) else "       "

        print(f"  {r['REGIME']:<16} {r['SPEC']:<14} {r['N_OBS']:>7} "
              f"{r['R2']:>7.4f} {b_str:>11} {t_str:>7} {p_str:>8}")


def _print_essay2_results(df: pd.DataFrame):
    print()
    print("  Essay 2: Cross-Sectional CAR (Treat-Control) with Culture Controls")
    print("  " + "-" * 76)
    print(f"  {'Window':<14} {'Spec':<18} {'N':>6} {'R²':>7} "
          f"{'Treat β':>10} {'p':>8} {'Culture β':>10} {'p':>8}")
    print("  " + "-" * 76)

    for _, r in df.iterrows():
        treat_b = r.get('TREAT_BETA', np.nan)
        treat_p = r.get('TREAT_P', np.nan)
        # Show whichever culture col is present
        cult_b = r.get('CULTURE_ORTHO_BETA', r.get('CULTURE_RAW_BETA', np.nan))
        cult_p = r.get('CULTURE_ORTHO_P', r.get('CULTURE_RAW_P', np.nan))

        treat_str = f"{treat_b:+.6f}" if pd.notna(treat_b) else "         "
        treat_p_str = f"{treat_p:.4f}{_sig_stars(treat_p)}" if pd.notna(treat_p) else "        "
        cult_str = f"{cult_b:+.6f}" if pd.notna(cult_b) else "         "
        cult_p_str = f"{cult_p:.4f}{_sig_stars(cult_p)}" if pd.notna(cult_p) else "        "

        print(f"  {r['WINDOW']:<14} {r['SPEC']:<18} {r['N_OBS']:>6} "
              f"{r['R2']:>7.4f} {treat_str:>10} {treat_p_str:>8} "
              f"{cult_str:>10} {cult_p_str:>8}")


def _print_essay3_results(df: pd.DataFrame):
    print()
    print("  Essay 3: Insider Trading Profitability + Culture Factor")
    print("  " + "-" * 60)
    print(f"  {'Spec':<16} {'N':>6} {'R²':>7} "
          f"{'C_RAW β':>9} {'p':>7} {'C_ORT β':>9} {'p':>7}")
    print("  " + "-" * 60)

    for _, r in df.iterrows():
        raw_b = r.get('CULTURE_RAW_BETA', np.nan)
        raw_p = r.get('CULTURE_RAW_P', np.nan)
        ort_b = r.get('CULTURE_ORTHO_BETA', np.nan)
        ort_p = r.get('CULTURE_ORTHO_P', np.nan)

        raw_str = f"{raw_b:+.5f}" if pd.notna(raw_b) else "       "
        raw_p_str = f"{raw_p:.4f}{_sig_stars(raw_p)}" if pd.notna(raw_p) else "       "
        ort_str = f"{ort_b:+.5f}" if pd.notna(ort_b) else "       "
        ort_p_str = f"{ort_p:.4f}{_sig_stars(ort_p)}" if pd.notna(ort_p) else "       "

        print(f"  {r['SPEC']:<16} {r['N_OBS']:>6} "
              f"{r['R2']:>7.4f} {raw_str:>9} {raw_p_str:>7} "
              f"{ort_str:>9} {ort_p_str:>7}")


def _print_summary(result: IntegrationResult):
    print()
    print(f"  First-stage R² = {result.first_stage_r2:.4f}")
    print(f"  → {result.first_stage_r2*100:.1f}% of culture factor variance "
          f"is attributable to politics")
    print(f"  → {(1-result.first_stage_r2)*100:.1f}% is genuinely cultural "
          f"(orthogonal to political affiliation)")
    print()

    # Check each essay
    for essay_num, df_name in [(1, 'essay1_results'),
                                (2, 'essay2_results'),
                                (3, 'essay3_results')]:
        df = getattr(result, df_name)
        if df.empty:
            print(f"  Essay {essay_num}: NO DATA (tables not available)")
            continue

        # Check if orthogonalized factor is significant in any spec
        ortho_p_col = 'CULTURE_ORTHO_P'
        if ortho_p_col in df.columns:
            min_p = df[ortho_p_col].min()
            n_sig = (df[ortho_p_col] < 0.05).sum()
            total = df[ortho_p_col].notna().sum()
            if n_sig > 0:
                print(f"  Essay {essay_num}: CULTURE_ORTHO significant in "
                      f"{n_sig}/{total} specs (min p={min_p:.4f})")
                print(f"    → Culture effect SURVIVES orthogonalization")
            else:
                print(f"  Essay {essay_num}: CULTURE_ORTHO not significant "
                      f"(min p={min_p:.4f})")
                print(f"    → Culture effect may be absorbed by politics")
        else:
            print(f"  Essay {essay_num}: {len(df)} results (culture cols not in output)")

    print()
    print("  INTERPRETATION:")
    print("  If CULTURE_ORTHO is significant → culture war effects are real,")
    print("  not merely a proxy for political party affiliation.")
    print("  If only CULTURE_RAW is significant → political confounding concern.")
    print("  Low first-stage R² already suggests culture ≠ politics.")
    print()


# ═════════════════════════════════════════════════════════════════════
# MAIN
# ═════════════════════════════════════════════════════════════════════

if __name__ == '__main__':
    import sys
    from datetime import datetime

    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
    )

    print(f"Started at {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print()

    store = DataStore()
    result = run_orthogonal_integration(store)
    store.close()

    print(f"Completed at {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
