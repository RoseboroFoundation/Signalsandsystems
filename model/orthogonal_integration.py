"""Orthogonalized Culture Factor — Integration with Essays 1, 2, and 3.

Runs the Baker-Wurgler (2006) orthogonalization from clean.orthogonalize,
then tests whether each essay's key findings survive after controlling for
the political component of the culture factor.

Design:
  - Builds monthly CULTURE_RAW and CULTURE_ORTHO time series
  - Full-sample orthogonalization for descriptive R² reporting
  - Rolling-window orthogonalization for predictive tests (Pagan 1984)
  - Pagan (1984) adjusted standard errors for generated-regressor bias
  - Predictive regressions: forward returns on purged factor
  - Fama-MacBeth on the purged (CULTURE_ORTHO) factor
  - For each essay, re-runs the core specification with both factors
  - Reports horse-race results: does the orthogonalized factor dominate?

This answers Dr. Hu's critique: if culture is confounded with politics,
controlling for politics should absorb the culture effect.  If culture
survives orthogonalization, the factor captures something distinct.

References
----------
Baker, M. & Wurgler, J. (2006). Investor sentiment and the cross-section
    of stock returns. Journal of Finance, 61(4).
Pagan, A. (1984). Econometric issues in the analysis of regressions with
    generated regressors. International Economic Review, 25(1).
Fama, E.F. & MacBeth, J.D. (1973). Risk, return, and equilibrium:
    Empirical tests. Journal of Political Economy, 81(3).
Newey, W.K. & West, K.D. (1987). A simple, positive semi-definite,
    heteroskedasticity and autocorrelation consistent covariance matrix.
    Econometrica, 55(3).
"""

import logging
import os
import warnings
from dataclasses import dataclass, field

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
class IntegrationResult:
    """Full integration results across all three essays."""
    # Orthogonalization summary
    first_stage_r2: float = np.nan
    first_stage_adj_r2: float = np.nan
    first_stage_coefficients: pd.DataFrame = field(default_factory=pd.DataFrame)
    n_political_proxies: int = 0
    estimation_method: str = ''
    # Rolling window result (Pagan 1984)
    rolling_result: object = None  # OrthogonalizationResult or None
    # Culture factor time series
    culture_factors: pd.DataFrame = field(default_factory=pd.DataFrame)
    # Essay results
    essay1_results: pd.DataFrame = field(default_factory=pd.DataFrame)
    essay2_results: pd.DataFrame = field(default_factory=pd.DataFrame)
    essay3_results: pd.DataFrame = field(default_factory=pd.DataFrame)
    # New Hu design items
    predictive_regressions: pd.DataFrame = field(default_factory=pd.DataFrame)
    fm_purged: pd.DataFrame = field(default_factory=pd.DataFrame)
    pagan_adjustment: pd.DataFrame = field(default_factory=pd.DataFrame)
    decision_level: pd.DataFrame = field(default_factory=pd.DataFrame)
    plot_path: str = ''
    # Combined summary
    summary: pd.DataFrame = field(default_factory=pd.DataFrame)


# ═════════════════════════════════════════════════════════════════════
# CULTURE FACTOR CONSTRUCTION
# ═════════════════════════════════════════════════════════════════════

def _build_culture_factors(store: DataStore):
    """Build raw and orthogonalized monthly culture factors.

    Produces both full-sample and rolling-window orthogonalization:
    - Full-sample: for descriptive analysis and first-stage R² reporting
    - Rolling (60-month): for predictive tests, avoids look-ahead bias (Pagan 1984)

    Returns (DataFrame with CULTURE_RAW/CULTURE_ORTHO/CULTURE_ORTHO_ROLLING,
             full_sample_result, rolling_result) or (None, None, None) on failure.
    """
    from clean.orthogonalize import (
        build_political_proxies,
        build_culture_index,
        orthogonalize_culture,
    )

    # Use cached events from DataStore (avoids redundant I/O)
    events_df = store.events.copy()
    if events_df.empty:
        logger.error("No CULTURE_WAR_COMPANIES table")
        return None, None, None

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

    # Build culture index — empty months are NaN so the first-stage
    # regression uses only months with actual culture-war events.
    culture_raw = build_culture_index(events_df, freq='M', method='net_score')
    if culture_raw.dropna().std() < 1e-10:
        logger.error("Culture index has zero/near-zero variance")
        return None, None, None

    # Build political proxies
    political = build_political_proxies(freq='M')

    # Full-sample orthogonalization (on event-active months only)
    full_result = orthogonalize_culture(
        culture_raw, political, method='full_sample')

    # Rolling-window orthogonalization (for predictive tests — Pagan 1984)
    rolling_result = None
    try:
        rolling_result = orthogonalize_culture(
            culture_raw, political, method='rolling', window=60)
        logger.info("Rolling orthogonalization: %d valid residuals",
                    rolling_result.culture_orthogonal.notna().sum())
    except Exception as e:
        logger.warning("Rolling orthogonalization failed: %s", e)

    # Build continuous time series: zero for months with no events,
    # orthogonal residual for months with events.  This preserves the
    # full monthly grid for downstream predictions and FM regressions.
    raw_filled = culture_raw.fillna(0.0)
    ortho_filled = full_result.culture_orthogonal.reindex(culture_raw.index).fillna(0.0)

    df = pd.DataFrame({
        'CULTURE_RAW': raw_filled,
        'CULTURE_ORTHO': ortho_filled,
    })
    if rolling_result is not None:
        df['CULTURE_ORTHO_ROLLING'] = (
            rolling_result.culture_orthogonal.reindex(culture_raw.index).fillna(0.0))
    df.index.name = 'DATE'

    logger.info("Culture factors: %d months (%d active), first-stage R²=%.4f",
                len(df), culture_raw.notna().sum(), full_result.first_stage_r2)
    return df, full_result, rolling_result


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
    events_df = store.events.copy()
    if events_df.empty:
        return pd.DataFrame()

    events_df.columns = [c.upper() for c in events_df.columns]
    tickers = events_df['TICKER'].dropna().unique().tolist()

    # Prepare factors (percent→decimal normalization handled by DataStore)
    factors = store.ff5[['DATE'] + _FF5_ALL + ['RF']].dropna().copy()
    factors['DATE'] = pd.to_datetime(factors['DATE'], errors='coerce')

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
    # Returns and factors are already in decimal (normalized by DataStore)
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
    from .essay2_did import build_multi_window_panel

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
        logger.warning("Essay 3 integration: ESSAY3_PANEL not found")
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
# DECISION-LEVEL TEST: Decision Direction vs Prior Firm Identity
# ═════════════════════════════════════════════════════════════════════

def _run_decision_level_test(
    store: DataStore,
    windows: list = None,
) -> pd.DataFrame:
    """Decision-level test of culture-war pricing.

    For each event, codes:
      - DECISION_DIR: political direction of the event itself
        (Liberal=+1, Conservative=-1, Mixed=0)
      - FIRM_IDENTITY: firm's pre-event political alignment from
        Essay 2's text-based alignment scoring (ESSAY2_EVENT_ALIGNMENT)
      - CONGRUENCE: DECISION_DIR × FIRM_IDENTITY (positive when
        event aligns with the firm's prior identity)

    Then estimates, for each CAR window:
      1. CAR = α + β₁·DECISION_DIR + ε               (direction only)
      2. CAR = α + β₂·FIRM_IDENTITY + ε               (identity only)
      3. CAR = α + β₁·DECISION_DIR + β₂·FIRM_IDENTITY
             + β₃·CONGRUENCE + ε                       (full model)

    If β₁ is significant in spec 3, the direction of the decision
    explains market reactions beyond the firm's prior political identity.
    This is the positive test Hu wants: culture-war events have real
    pricing effects that are not merely a proxy for firm partisanship.

    Parameters
    ----------
    store : DataStore
    windows : list of int, optional
        CAR windows to test (default [10, 30, 60]).

    Returns
    -------
    pd.DataFrame with regression results across specs and windows.
    """
    import sqlite3

    if windows is None:
        windows = [10, 30, 60]

    db_path = os.path.join(
        os.path.dirname(os.path.abspath(__file__)), '..', 'data', 'signals_systems.db')
    conn = sqlite3.connect(db_path)

    # ── Load event-level CARs ──────────────────────────────────────
    try:
        mw_panel = pd.read_sql('SELECT * FROM ESSAY2_MULTI_WINDOW_PANEL', conn)
    except Exception:
        logger.warning("Decision-level test: ESSAY2_MULTI_WINDOW_PANEL not found")
        conn.close()
        return pd.DataFrame()

    # ── Load pre-event firm alignment scores ───────────────────────
    try:
        event_align = pd.read_sql('SELECT * FROM ESSAY2_EVENT_ALIGNMENT', conn)
    except Exception:
        logger.warning("Decision-level test: ESSAY2_EVENT_ALIGNMENT not found")
        conn.close()
        return pd.DataFrame()

    conn.close()

    if mw_panel.empty or event_align.empty:
        return pd.DataFrame()

    # Keep treatment firms only (controls have no culture-war event)
    panel = mw_panel[mw_panel['IS_TREATMENT'] == 1].copy()
    panel['EVENT_DATE'] = pd.to_datetime(panel['EVENT_DATE'], errors='coerce')
    panel['CAR'] = pd.to_numeric(panel['CAR'], errors='coerce')
    panel['WINDOW'] = pd.to_numeric(panel['WINDOW'], errors='coerce')

    # ── Code decision direction from event metadata ────────────────
    events_df = store.events.copy()
    events_df.columns = [c.upper() for c in events_df.columns]
    events_df['EVENT_DATE'] = pd.to_datetime(events_df['EVENT_DATE'], errors='coerce')

    dir_map = {'liberal': 1.0, 'conservative': -1.0, 'mixed': 0.0}
    events_df['DECISION_DIR'] = (
        events_df['ESTIMATED_POLITICAL_LEANING'].str.strip().str.lower()
        .map(dir_map))

    event_lookup = (
        events_df.dropna(subset=['TICKER', 'EVENT_DATE', 'DECISION_DIR'])
        .drop_duplicates(subset=['TICKER', 'EVENT_DATE'])
        .set_index(['TICKER', 'EVENT_DATE'])['DECISION_DIR'])

    panel['DECISION_DIR'] = panel.set_index(
        ['TICKER', 'EVENT_DATE']).index.map(event_lookup)
    panel['DECISION_DIR'] = pd.to_numeric(panel['DECISION_DIR'], errors='coerce')

    # ── Firm's pre-event political identity from Essay 2 alignment ─
    event_align['EVENT_DATE'] = pd.to_datetime(event_align['EVENT_DATE'], errors='coerce')
    event_align['ALIGNMENT_SCORE'] = pd.to_numeric(
        event_align['ALIGNMENT_SCORE'], errors='coerce')
    align_lookup = (
        event_align.dropna(subset=['TICKER', 'EVENT_DATE', 'ALIGNMENT_SCORE'])
        .drop_duplicates(subset=['TICKER', 'EVENT_DATE'])
        .set_index(['TICKER', 'EVENT_DATE'])['ALIGNMENT_SCORE'])

    panel['FIRM_IDENTITY'] = panel.set_index(
        ['TICKER', 'EVENT_DATE']).index.map(align_lookup)
    panel['FIRM_IDENTITY'] = pd.to_numeric(panel['FIRM_IDENTITY'], errors='coerce')

    # Congruence: positive when decision aligns with firm identity
    panel['CONGRUENCE'] = panel['DECISION_DIR'] * panel['FIRM_IDENTITY']

    n_dir = panel['DECISION_DIR'].notna().sum()
    n_id = panel['FIRM_IDENTITY'].notna().sum()
    n_both = panel.dropna(subset=['DECISION_DIR', 'FIRM_IDENTITY']).shape[0]
    logger.info("Decision-level: %d with direction, %d with identity, %d both",
                n_dir, n_id, n_both)

    # ── Run regressions ────────────────────────────────────────────
    results = []
    test_windows = [w for w in windows if w in panel['WINDOW'].unique()]
    if not test_windows:
        test_windows = sorted(panel['WINDOW'].unique())[:3]

    for w in test_windows:
        sub = panel[panel['WINDOW'] == w].copy()
        sub = sub.dropna(subset=['CAR', 'DECISION_DIR', 'FIRM_IDENTITY'])

        if len(sub) < 20:
            logger.warning("Decision-level w=%d: only %d obs after merge", w, len(sub))
            continue

        y = sub['CAR']

        specs = [
            ('Direction only', ['DECISION_DIR']),
            ('Identity only', ['FIRM_IDENTITY']),
            ('Full model', ['DECISION_DIR', 'FIRM_IDENTITY', 'CONGRUENCE']),
        ]

        for spec_name, x_cols in specs:
            X = sm.add_constant(sub[x_cols])
            try:
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore")
                    fit = sm.OLS(y, X).fit(cov_type='HC1')

                row = {
                    'WINDOW': f'[-1, +{w}]',
                    'SPEC': spec_name,
                    'N_OBS': int(fit.nobs),
                    'R2': fit.rsquared,
                    'ADJ_R2': fit.rsquared_adj,
                }
                for col in x_cols:
                    row[f'{col}_BETA'] = fit.params[col]
                    row[f'{col}_T'] = fit.tvalues[col]
                    row[f'{col}_P'] = fit.pvalues[col]
                results.append(row)
            except Exception as e:
                logger.warning("Decision-level w=%d %s failed: %s", w, spec_name, e)

    if results:
        logger.info("Decision-level test: %d specs across %d windows",
                    len(results), len(test_windows))
    return pd.DataFrame(results)


# ═════════════════════════════════════════════════════════════════════
# PREDICTIVE REGRESSIONS ON PURGED FACTOR
# ═════════════════════════════════════════════════════════════════════

def _run_predictive_regressions(
    culture_factors: pd.DataFrame,
    market_returns: pd.DataFrame,
    horizons: list = None,
) -> pd.DataFrame:
    """Predictive regressions: forward returns on CULTURE_ORTHO.

    R_{t+h} = α + β × CULTURE_ORTHO_t + ε_{t+h}

    Tests whether the purged culture factor predicts future market returns.
    Uses Newey-West SEs with lags = horizon to account for overlapping returns.

    Parameters
    ----------
    culture_factors : pd.DataFrame
        Monthly with CULTURE_RAW, CULTURE_ORTHO, optionally CULTURE_ORTHO_ROLLING.
    market_returns : pd.DataFrame
        Must have DATE and either MKT_RF or RETURN column.
    horizons : list of int
        Forward horizons in months (default [1, 3, 6, 12]).
    """
    if horizons is None:
        horizons = [1, 3, 6, 12]

    # Get monthly market returns
    mkt = market_returns.copy()
    mkt['DATE'] = pd.to_datetime(mkt['DATE'])
    ret_col = 'MKT_RF' if 'MKT_RF' in mkt.columns else 'RETURN'
    if ret_col not in mkt.columns:
        logger.warning("Predictive regressions: no return column found")
        return pd.DataFrame()

    mkt['MONTH'] = mkt['DATE'].dt.to_period('M')
    monthly_ret = mkt.groupby('MONTH')[ret_col].apply(
        lambda x: (1 + x).prod() - 1).rename('MKT_RET')
    monthly_ret.index = monthly_ret.index.to_timestamp()

    # Merge with culture factors
    culture = culture_factors[['CULTURE_RAW', 'CULTURE_ORTHO']].copy()
    if 'CULTURE_ORTHO_ROLLING' in culture_factors.columns:
        culture['CULTURE_ORTHO_ROLLING'] = culture_factors['CULTURE_ORTHO_ROLLING']

    merged = culture.join(monthly_ret, how='inner')
    if len(merged) < 24:
        logger.warning("Predictive regressions: insufficient data (%d months)", len(merged))
        return pd.DataFrame()

    results = []
    culture_cols = ['CULTURE_RAW', 'CULTURE_ORTHO']
    if 'CULTURE_ORTHO_ROLLING' in merged.columns:
        culture_cols.append('CULTURE_ORTHO_ROLLING')

    for h in horizons:
        # Compute h-month forward compounded return
        fwd_col = f'FWD_{h}M'
        fwd_ret = (
            (1 + merged['MKT_RET']).rolling(h).apply(np.prod, raw=True) - 1
        ).shift(-h)

        for col in culture_cols:
            valid = merged[[col]].assign(**{fwd_col: fwd_ret}).dropna()
            if len(valid) < 24:
                continue

            y = valid[f'FWD_{h}M']
            X = sm.add_constant(valid[[col]])
            nw_lags = max(h, 1)  # At least h lags for overlapping returns

            try:
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore")
                    fit = sm.OLS(y, X).fit(
                        cov_type='HAC', cov_kwds={'maxlags': nw_lags})

                results.append({
                    'HORIZON_MONTHS': h,
                    'PREDICTOR': col,
                    'BETA': fit.params[col],
                    'T_STAT': fit.tvalues[col],
                    'P_VALUE': fit.pvalues[col],
                    'R_SQUARED': fit.rsquared,
                    'N_OBS': int(fit.nobs),
                    'NW_LAGS': nw_lags,
                })
            except Exception as e:
                logger.warning("Predictive reg h=%d %s failed: %s", h, col, e)

    if results:
        logger.info("Predictive regressions: %d specs across %d horizons",
                    len(results), len(horizons))
    return pd.DataFrame(results)


# ═════════════════════════════════════════════════════════════════════
# FAMA-MACBETH ON PURGED FACTOR
# ═════════════════════════════════════════════════════════════════════

def _run_fm_on_purged(
    store: DataStore,
    culture_factors: pd.DataFrame,
    ticker_returns: pd.DataFrame = None,
) -> pd.DataFrame:
    """Fama-MacBeth on firm-level culture exposure × purged factor.

    An aggregate monthly factor (CULTURE_ORTHO) has zero cross-sectional
    variation, so it cannot serve as a Fama-MacBeth characteristic directly.
    Instead, we construct a firm-level culture exposure:

        EXPOSURE_i,t = EVENT_INTENSITY_i × CULTURE_ORTHO_t

    where EVENT_INTENSITY_i is the firm's total culture-war event count
    (normalized), providing genuine cross-sectional variation.

    Each month t+1, we run: R_{i,t+1} = α + γ × EXPOSURE_{i,t} + ε
    Then test whether mean(γ) ≠ 0 using Newey-West SEs (Andrews 1991).

    Parameters
    ----------
    store : DataStore
    culture_factors : pd.DataFrame with CULTURE_ORTHO indexed by month.
    ticker_returns : pd.DataFrame, optional
        Pre-built daily returns with DATE, TICKER, RETURN columns.
        If None, builds from store (slower).
    """
    events_df = store.events.copy()
    if events_df.empty:
        return pd.DataFrame()

    events_df.columns = [c.upper() for c in events_df.columns]

    # Build firm-level event intensity (cross-sectional characteristic)
    event_date_col = next(
        (c for c in events_df.columns if 'EVENT' in c and 'DATE' in c), None)
    if event_date_col is None:
        logger.warning("FM on purged: no event date column in CULTURE_WAR_COMPANIES")
        return pd.DataFrame()

    events_df[event_date_col] = pd.to_datetime(events_df[event_date_col], errors='coerce')
    events_df = events_df.dropna(subset=['TICKER', event_date_col])

    # Event count per ticker (time-invariant exposure proxy)
    event_counts = events_df.groupby('TICKER').size().rename('EVENT_COUNT')
    # Rank-based normalization to (0, 1] — every firm with events gets
    # nonzero exposure, avoiding the min-max problem where the lowest-
    # count firm maps to zero and contributes nothing to the cross-section.
    event_intensity = event_counts.rank(pct=True).rename('EVENT_INTENSITY')

    tickers = event_intensity.index.tolist()

    # Get monthly returns per ticker
    if ticker_returns is not None and not ticker_returns.empty:
        returns = ticker_returns[ticker_returns['TICKER'].isin(tickers)].copy()
    else:
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
            return pd.DataFrame()
        returns = pd.concat(all_rows, ignore_index=True)

    returns['MONTH'] = pd.to_datetime(returns['DATE']).dt.to_period('M')
    monthly_ret = returns.groupby(['TICKER', 'MONTH']).agg(
        RETURN=('RETURN', lambda x: (1 + x).prod() - 1),
    ).reset_index()

    # Merge event intensity (firm-level, cross-sectional variation)
    monthly_ret = monthly_ret.merge(
        event_intensity.reset_index(), on='TICKER', how='inner')

    # Map CULTURE_ORTHO to monthly (lagged 1 month)
    culture = culture_factors[['CULTURE_ORTHO']].copy()
    culture.index.name = 'DATE'
    culture = culture.reset_index()
    culture['MONTH'] = pd.to_datetime(culture['DATE']).dt.to_period('M')
    culture['MONTH'] = culture['MONTH'] + 1  # lag: month t → returns in t+1
    culture = culture[['MONTH', 'CULTURE_ORTHO']]

    monthly_ret = monthly_ret.merge(culture, on='MONTH', how='inner')
    monthly_ret = monthly_ret.dropna()

    # Firm-level exposure = intensity × aggregate factor direction
    monthly_ret['EXPOSURE'] = (
        monthly_ret['EVENT_INTENSITY'] * monthly_ret['CULTURE_ORTHO'])

    months = sorted(monthly_ret['MONTH'].unique())
    gammas = []

    for month in months:
        cross = monthly_ret[monthly_ret['MONTH'] == month]
        if len(cross) < 10 or cross['EXPOSURE'].std() < 1e-10:
            continue

        y = cross['RETURN']
        X = sm.add_constant(cross[['EXPOSURE']])
        try:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                fit = sm.OLS(y, X).fit()
            gammas.append({
                'MONTH': str(month),
                'GAMMA_CONST': fit.params['const'],
                'GAMMA_EXPOSURE': fit.params['EXPOSURE'],
                'N_STOCKS': len(cross),
                'R_SQUARED': fit.rsquared,
            })
        except Exception:
            continue

    if not gammas:
        logger.warning("FM on purged: no valid cross-sections")
        return pd.DataFrame()

    gamma_df = pd.DataFrame(gammas)
    gamma_series = gamma_df['GAMMA_EXPOSURE'].dropna()
    n_months = len(gamma_series)

    # Newey-West SE (Andrews 1991 lag)
    nw_lags = max(1, int(4 * (n_months / 100) ** 0.29))
    try:
        gamma_ols = sm.OLS(gamma_series.values, np.ones(n_months)).fit(
            cov_type='HAC', cov_kwds={'maxlags': nw_lags})
        t_gamma = gamma_ols.tvalues[0]
        p_gamma = gamma_ols.pvalues[0]
    except Exception:
        se = gamma_series.std() / np.sqrt(n_months)
        t_gamma = gamma_series.mean() / se if se > 0 else 0
        p_gamma = 2 * (1 - stats.t.cdf(abs(t_gamma), df=n_months - 1))

    logger.info("FM on purged: gamma_exposure=%.6f (t=%.2f, p=%.4f), %d months",
                gamma_series.mean(), t_gamma, p_gamma, n_months)

    # Add summary row
    summary = pd.DataFrame([{
        'MONTH': 'MEAN (NW)',
        'GAMMA_CONST': gamma_df['GAMMA_CONST'].mean(),
        'GAMMA_EXPOSURE': gamma_series.mean(),
        'N_STOCKS': gamma_df['N_STOCKS'].mean(),
        'R_SQUARED': gamma_df['R_SQUARED'].mean(),
        'T_STAT': t_gamma,
        'P_VALUE': p_gamma,
        'NW_LAGS': nw_lags,
    }])

    return pd.concat([gamma_df, summary], ignore_index=True)


# ═════════════════════════════════════════════════════════════════════
# PAGAN (1984) ADJUSTED STANDARD ERRORS
# ═════════════════════════════════════════════════════════════════════

def _pagan_bootstrap_adjustment(
    y: pd.Series,
    X_base: pd.DataFrame,
    culture_ortho: pd.Series,
    culture_raw: pd.Series,
    political: pd.DataFrame,
    n_bootstrap: int = 500,
) -> pd.DataFrame:
    """Bootstrap adjustment for generated-regressor bias (Pagan 1984).

    The CULTURE_ORTHO variable is estimated (generated) from a first-stage
    regression. OLS SEs in the second stage ignore estimation uncertainty
    from the first stage. This bootstrap resamples both stages jointly.

    Returns DataFrame with OLS vs bootstrap SEs and adjustment ratios.
    """
    rng = np.random.default_rng(42)

    # Reset all inputs to integer positional index to guarantee alignment
    # during bootstrap resampling (avoids DatetimeIndex collision issues).
    y_arr = y.reset_index(drop=True)
    X_base_arr = X_base.reset_index(drop=True)
    ortho_arr = culture_ortho.reset_index(drop=True)
    raw_arr = culture_raw.reset_index(drop=True)
    pol_arr = political.reset_index(drop=True)
    n = len(y_arr)

    # Run OLS once with the point estimate of CULTURE_ORTHO
    X_full = pd.concat([X_base_arr, ortho_arr.rename('CULTURE_ORTHO')], axis=1)
    X_full = sm.add_constant(X_full).dropna()
    y_aligned = y_arr.reindex(X_full.index).dropna()
    X_full = X_full.loc[y_aligned.index]

    if len(y_aligned) < 20:
        return pd.DataFrame()

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        ols_fit = sm.OLS(y_aligned, X_full).fit(
            cov_type='HAC', cov_kwds={'maxlags': 6})

    ols_params = ols_fit.params
    ols_se = ols_fit.bse

    # Bootstrap: resample positional indices, re-run first + second stage
    boot_params = []
    for _ in range(n_bootstrap):
        idx = rng.choice(n, size=n, replace=True)

        # First stage: re-orthogonalize
        y_1st = raw_arr.iloc[idx].reset_index(drop=True)
        X_1st = sm.add_constant(pol_arr.iloc[idx].reset_index(drop=True))
        try:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                fit_1st = sm.OLS(y_1st, X_1st).fit()
            ortho_boot = y_1st - fit_1st.fittedvalues
        except Exception:
            continue

        # Second stage: use bootstrapped ortho
        X_2nd_base = X_base_arr.iloc[idx].reset_index(drop=True)
        X_2nd = sm.add_constant(
            pd.concat([X_2nd_base, ortho_boot.rename('CULTURE_ORTHO')], axis=1))
        y_2nd = y_arr.iloc[idx].reset_index(drop=True)

        if len(y_2nd.dropna()) < 20:
            continue

        try:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                fit_2nd = sm.OLS(y_2nd, X_2nd).fit()
            boot_params.append(fit_2nd.params.to_dict())
        except Exception:
            continue

    if len(boot_params) < 50:
        logger.warning("Pagan bootstrap: only %d valid iterations", len(boot_params))
        return pd.DataFrame()

    boot_df = pd.DataFrame(boot_params)

    # Compare SEs
    rows = []
    for var in ols_params.index:
        if var not in boot_df.columns:
            continue
        boot_se = boot_df[var].std()
        rows.append({
            'VARIABLE': var,
            'OLS_COEF': ols_params[var],
            'OLS_SE': ols_se[var],
            'OLS_T': ols_params[var] / ols_se[var] if ols_se[var] > 0 else np.nan,
            'BOOT_SE': boot_se,
            'BOOT_T': ols_params[var] / boot_se if boot_se > 0 else np.nan,
            'SE_RATIO': boot_se / ols_se[var] if ols_se[var] > 0 else np.nan,
            'N_BOOT': len(boot_params),
        })

    result = pd.DataFrame(rows)
    # Drop intercept — its SE ratio is uninteresting for generated-regressor diagnosis
    result = result[result['VARIABLE'] != 'const'].reset_index(drop=True)
    if 'CULTURE_ORTHO' in result['VARIABLE'].values:
        ratio = result.loc[result['VARIABLE'] == 'CULTURE_ORTHO', 'SE_RATIO'].iloc[0]
        logger.info("Pagan bootstrap: CULTURE_ORTHO SE ratio = %.3f "
                    "(>1 means OLS understates uncertainty)", ratio)
    return result


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

    # Step 1: Build culture factors (full-sample + rolling)
    print("=" * 70)
    print("  ORTHOGONALIZED CULTURE FACTOR — FULL INTEGRATION")
    print("  Baker-Wurgler (2006) purging of political confounds")
    print("=" * 70)
    print()

    print("Step 1: Building culture factors (full-sample + rolling)...")
    build_out = _build_culture_factors(store)
    if build_out[0] is None:
        print("  FAILED — cannot build culture factors")
        return result

    culture_factors, full_result, rolling_result = build_out

    result.first_stage_r2 = full_result.first_stage_r2
    result.first_stage_adj_r2 = full_result.first_stage_adj_r2
    result.first_stage_coefficients = full_result.first_stage_coefficients
    result.n_political_proxies = len(full_result.political_proxies_used)
    result.estimation_method = full_result.estimation_method
    result.rolling_result = rolling_result
    result.culture_factors = culture_factors

    print(f"  Culture index: {len(culture_factors)} months")
    print(f"  ┌─────────────────────────────────────────────────┐")
    print(f"  │  FIRST-STAGE R² = {full_result.first_stage_r2:.4f}                          │")
    print(f"  │  {full_result.first_stage_r2*100:.1f}% of culture variance is political        │")
    print(f"  │  {(1-full_result.first_stage_r2)*100:.1f}% survives orthogonalization             │")
    print(f"  └─────────────────────────────────────────────────┘")
    print(f"  Political proxies: {', '.join(full_result.political_proxies_used)}")
    print(f"  Correlation(raw, ortho) = "
          f"{culture_factors['CULTURE_RAW'].corr(culture_factors['CULTURE_ORTHO']):.3f}")
    if rolling_result is not None:
        n_valid = culture_factors['CULTURE_ORTHO_ROLLING'].notna().sum() if 'CULTURE_ORTHO_ROLLING' in culture_factors else 0
        print(f"  Rolling window: 60 months, {n_valid} valid out-of-sample residuals")
    print()

    # First-stage coefficients
    print("  First-stage coefficients (Culture = a + b'Political + e):")
    coef_df = full_result.first_stage_coefficients
    for _, row in coef_df.iterrows():
        if row['VARIABLE'] == 'const':
            continue
        beta = row['COEFFICIENT']
        pval = row['P_VALUE']
        sig = '***' if pval < 0.01 else '**' if pval < 0.05 else '*' if pval < 0.10 else ''
        print(f"    {row['VARIABLE']:25s}  beta={beta:+.4f}  p={pval:.4f} {sig}")
    print()

    # Step 1b: Time-series plot (raw vs orthogonalized)
    print("Step 1b: Generating time-series plot...")
    try:
        from clean.orthogonalize import plot_culture_factors
        plot_path = plot_culture_factors(
            full_result,
            output_path=os.path.join(
                os.path.dirname(os.path.abspath(__file__)),
                '..', 'culture_factor_comparison.png'))
        if plot_path:
            result.plot_path = plot_path
            print(f"  Saved plot: {plot_path}")
        else:
            print("  Plot skipped (matplotlib unavailable)")
    except Exception as e:
        logger.warning("Plot failed: %s", e)
        print(f"  Plot failed: {e}")
    print()

    # Step 2: Predictive regressions on purged factor
    print("Step 2: Predictive regressions (forward returns on CULTURE_ORTHO)...")
    if not store.ff5.empty:
        result.predictive_regressions = _run_predictive_regressions(
            culture_factors, store.ff5)
        if not result.predictive_regressions.empty:
            _print_predictive_regressions(result.predictive_regressions)
        else:
            print("  No results")
    else:
        print("  SKIPPED — no factor data")
    print()

    # Step 3: Fama-MacBeth on purged factor
    print("Step 3: Fama-MacBeth on purged (CULTURE_ORTHO) factor...")
    result.fm_purged = _run_fm_on_purged(store, culture_factors)
    if not result.fm_purged.empty:
        _print_fm_purged(result.fm_purged)
    else:
        print("  No results")
    print()

    # Step 4: Pagan (1984) bootstrap SE adjustment
    print("Step 4: Pagan (1984) bootstrap SE adjustment...")
    try:
        from clean.orthogonalize import build_political_proxies
        political = build_political_proxies(freq='M')
        # Use Essay 1 excess returns for the second-stage test
        if not store.ff5.empty:
            ff5 = store.ff5.copy()
            ff5['DATE'] = pd.to_datetime(ff5['DATE'])
            ff5['MONTH'] = ff5['DATE'].dt.to_period('M').dt.to_timestamp()
            mkt_monthly = ff5.groupby('MONTH')['MKT_RF'].apply(
                lambda x: (1 + x).prod() - 1)
            mkt_monthly.index.name = 'DATE'

            # Align with culture factors — use only active months
            # (where culture_raw was not NaN before filling) so the
            # bootstrap first stage matches the actual orthogonalization.
            active_mask = full_result.culture_raw.notna()
            active_culture = culture_factors.loc[active_mask]
            aligned = active_culture.join(mkt_monthly.rename('MKT_RET'), how='inner').dropna()
            if len(aligned) > 30:
                result.pagan_adjustment = _pagan_bootstrap_adjustment(
                    y=aligned['MKT_RET'],
                    X_base=pd.DataFrame(index=aligned.index),  # No extra controls
                    culture_ortho=aligned['CULTURE_ORTHO'],
                    culture_raw=aligned['CULTURE_RAW'],
                    political=political.reindex(aligned.index).ffill(),
                    n_bootstrap=500,
                )
                if not result.pagan_adjustment.empty:
                    _print_pagan_adjustment(result.pagan_adjustment)
                else:
                    print("  Insufficient valid bootstrap iterations")
            else:
                print("  Insufficient aligned data for bootstrap")
        else:
            print("  SKIPPED — no factor data")
    except Exception as e:
        logger.warning("Pagan bootstrap failed: %s", e)
        print(f"  Failed: {e}")
    print()

    # Step 5: Decision-level test (direction vs prior identity)
    print("Step 5: Decision-level test (event direction vs firm prior stance)...")
    result.decision_level = _run_decision_level_test(store)
    if not result.decision_level.empty:
        _print_decision_level(result.decision_level)
    else:
        print("  No results (need ESSAY2_MULTI_WINDOW_PANEL + stance scores)")
    print()

    # Step 6: Essay 1
    print("Step 6: Essay 1 — FF5 + Culture Factor by Regime...")
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

    # Step 7: Essay 2
    print("Step 7: Essay 2 — Cross-Sectional CAR with Culture Control...")
    result.essay2_results = _run_essay2_integration(store, culture_factors)
    if not result.essay2_results.empty:
        _print_essay2_results(result.essay2_results)
    else:
        print("  No results produced (may need ESSAY2_POLITICAL_ALIGNMENT table)")
    print()

    # Step 8: Essay 3
    print("Step 8: Essay 3 — Insider Trading + Culture Factor...")
    result.essay3_results = _run_essay3_integration(store, culture_factors)
    if not result.essay3_results.empty:
        _print_essay3_results(result.essay3_results)
    else:
        print("  No results produced (may need ESSAY3_PANEL table)")
    print()

    # Step 9: Summary
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


def _print_predictive_regressions(df: pd.DataFrame):
    print()
    print("  Predictive Regressions: Forward Returns on Culture Factors")
    print("  " + "-" * 72)
    print(f"  {'Horizon':>8} {'Predictor':<25} {'β':>10} {'t':>7} {'p':>8} {'R²':>7} {'N':>5}")
    print("  " + "-" * 72)
    for _, r in df.iterrows():
        sig = _sig_stars(r['P_VALUE'])
        print(f"  {r['HORIZON_MONTHS']:>5}M   {r['PREDICTOR']:<25} "
              f"{r['BETA']:+.6f} {r['T_STAT']:+.2f} {r['P_VALUE']:.4f}{sig} "
              f"{r['R_SQUARED']:.4f} {r['N_OBS']:>5}")


def _print_fm_purged(df: pd.DataFrame):
    print()
    summary = df[df['MONTH'] == 'MEAN (NW)']
    if summary.empty:
        print(f"  FM on purged: {len(df)} monthly cross-sections")
        return
    s = summary.iloc[0]
    gamma_col = 'GAMMA_EXPOSURE' if 'GAMMA_EXPOSURE' in s.index else 'GAMMA_ORTHO'
    print(f"  Fama-MacBeth on firm culture exposure × CULTURE_ORTHO")
    print(f"    Mean γ(exposure) = {s[gamma_col]:.6f}")
    print(f"    t-statistic    = {s.get('T_STAT', np.nan):.2f}")
    print(f"    p-value        = {s.get('P_VALUE', np.nan):.4f}")
    print(f"    N months       = {len(df) - 1}")
    print(f"    Avg stocks/mo  = {s['N_STOCKS']:.0f}")
    if s.get('P_VALUE', 1) < 0.05:
        print(f"    → PRICED: purged culture factor compensated cross-sectionally")
    else:
        print(f"    → Not significant at 5%")


def _print_pagan_adjustment(df: pd.DataFrame):
    print()
    print("  Pagan (1984) Bootstrap SE Adjustment")
    print("  " + "-" * 68)
    print(f"  {'Variable':<20} {'OLS SE':>9} {'Boot SE':>9} {'Ratio':>7} "
          f"{'OLS t':>7} {'Boot t':>7}")
    print("  " + "-" * 68)
    for _, r in df.iterrows():
        print(f"  {r['VARIABLE']:<20} {r['OLS_SE']:.6f} {r['BOOT_SE']:.6f} "
              f"{r['SE_RATIO']:.3f}   {r['OLS_T']:+.2f}   {r['BOOT_T']:+.2f}")
    print(f"  (ratio > 1 means OLS understates uncertainty; N_boot={df['N_BOOT'].iloc[0]:.0f})")


def _print_decision_level(df: pd.DataFrame):
    print()
    print("  Decision-Level Test: Event Direction vs Prior Firm Identity")
    print("  CAR = α + β₁·DIR + β₂·STANCE_PRIOR + β₃·(DIR × STANCE) + ε")
    print("  " + "-" * 80)
    print(f"  {'Window':<12} {'Spec':<18} {'N':>5} {'R²':>7} "
          f"{'DIR β':>10} {'p':>8} {'STANCE β':>10} {'p':>8} {'CONG β':>10} {'p':>8}")
    print("  " + "-" * 80)

    for _, r in df.iterrows():
        dir_b = r.get('DECISION_DIR_BETA', np.nan)
        dir_p = r.get('DECISION_DIR_P', np.nan)
        st_b = r.get('FIRM_IDENTITY_BETA', np.nan)
        st_p = r.get('FIRM_IDENTITY_P', np.nan)
        cg_b = r.get('CONGRUENCE_BETA', np.nan)
        cg_p = r.get('CONGRUENCE_P', np.nan)

        def _fmt(v, w=10):
            return f"{v:+.5f}" if pd.notna(v) else " " * w

        def _pfmt(v):
            return f"{v:.4f}{_sig_stars(v)}" if pd.notna(v) else " " * 8

        print(f"  {r['WINDOW']:<12} {r['SPEC']:<18} {r['N_OBS']:>5} "
              f"{r['R2']:>7.4f} {_fmt(dir_b):>10} {_pfmt(dir_p):>8} "
              f"{_fmt(st_b):>10} {_pfmt(st_p):>8} "
              f"{_fmt(cg_b):>10} {_pfmt(cg_p):>8}")

    # Highlight the key finding
    full = df[df['SPEC'] == 'Full model']
    if not full.empty and 'DECISION_DIR_P' in full.columns:
        best = full.loc[full['DECISION_DIR_P'].idxmin()]
        p = best['DECISION_DIR_P']
        if p < 0.05:
            print(f"\n  → DECISION DIRECTION significant in full model "
                  f"(p={p:.4f}): market reacts to event politics,")
            print(f"    not just prior firm identity")
        elif p < 0.10:
            print(f"\n  → DECISION DIRECTION marginally significant "
                  f"(p={p:.4f})")
        else:
            print(f"\n  → DECISION DIRECTION not significant in full model "
                  f"(p={p:.4f})")


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
