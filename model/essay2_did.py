"""
Essay 2 — Difference-in-Differences on Culture War Event CARs.

Implements the cross-sectional DiD for culture war events:

    CAR_{i,e} = alpha
              + beta_1 * Treat_i
              + beta_2 * Post_e
              + beta_3 * (Treat_i x Post_e)
              + beta_4 * Lean_i
              + beta_5 * (Treat_i x Post_e x Lean_i)
              + beta_6 * FOMO_Z_{i,e}
              + gamma * Controls
              + epsilon_{i,e}

where:
    - CAR_{i,e} is the cumulative abnormal return for firm i around event e
    - Treat_i = 1 for culture war firms, 0 for matched controls
    - Post_e  = 1 for the post-event window
    - Lean_i  = political lean score (continuous or categorical)
    - FOMO_Z  = FinBERT FOMO z-score from Essay 1

The unit of observation is (firm, event, window).  Each treatment firm is
paired with its industry-matched control from the CONTROL_COMPANIES table.

Consumes Essay 1 outputs (finalized in PR #9 v4):
    - Regime assignments (RegimeResult) — Markov-switching on VIX
    - FF5 factor loadings for abnormal return estimation
    - Matched control pairs (CONTROL_COMPANIES)
    - FOMO z-scores (SentimentRegimeAnalysis) — FinBERT-scored, with
      failed articles excluded (not neutral-imputed) per v4 fix

References
----------
MacKinlay, A.C. (1997). Event studies in economics and finance. Journal
    of Economic Literature, 35(1).
Kolari, J.W. & Pynnonen, S. (2010). Event study testing with
    cross-sectional correlation of abnormal returns. Review of Financial
    Studies, 23(11).
"""

import logging
import warnings
from collections import namedtuple
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import statsmodels.api as sm
from scipy import stats
from scipy.linalg import qr as _qr

from .datastore import DataStore
from .essay1 import (
    RegimeResult,
    estimate_vix_regimes,
    SentimentRegimeAnalysis,
    _FF5_ALL,
    benjamini_hochberg,
)

logger = logging.getLogger(__name__)

# =========================================================================
# CONFIGURATION
# =========================================================================

# Event window parameters (trading days)
_ESTIMATION_WINDOW = (-250, -11)   # estimation period for normal returns
_PRE_EVENT_WINDOW = (-10, -1)      # pre-event window
_POST_EVENT_WINDOW = (0, 10)       # post-event window (day 0 = event)
_MIN_ESTIMATION_OBS = 120          # minimum obs in estimation window

# Multi-window event study windows: [-1, +N] for each post-event horizon
_MULTI_WINDOWS = [5, 10, 15, 20, 30, 60, 90]
_PRE_DAY = -1   # 1 trading day before event


# =========================================================================
# DATA CLASSES
# =========================================================================

@dataclass
class EventCAR:
    """Cumulative abnormal return for a single firm around a single event."""
    ticker: str
    event_id: str
    event_date: pd.Timestamp
    is_treatment: bool
    regime: str
    car_pre: float              # CAR over pre-event window
    car_post: float             # CAR over post-event window
    car_full: float             # CAR over full event window [-10, +10]
    n_estimation_obs: int       # obs in estimation window
    n_event_obs: int            # obs in event window
    alpha: float                # estimation-window alpha
    r_squared: float            # estimation-window R^2


@dataclass
class MultiWindowCAR:
    """CARs for a single firm/event across multiple post-event horizons."""
    ticker: str
    event_id: str
    event_date: pd.Timestamp
    is_treatment: bool
    regime: str
    lean: str                           # political leaning
    cars: Dict[int, float]              # {window_end: CAR[-1, +window_end]}
    n_estimation_obs: int
    alpha: float
    r_squared: float


@dataclass
class ParallelTrendsResult:
    """Pre-event parallel trends test for DiD validity."""
    daily_coefficients: pd.DataFrame    # DAY, TREAT_x_DAY_COEFF, SE, T_STAT, P_VALUE
    joint_f_stat: float                 # F-test: all Treat*Day coefficients = 0
    joint_p_value: float
    passes: bool                        # True if joint p > 0.05
    n_days: int
    n_observations: int


@dataclass
class DiDResult:
    """Cross-sectional DiD regression results."""
    car_panel: pd.DataFrame             # firm x event panel with CARs
    did_basic: object                   # beta_3 regression (Treat x Post)
    did_with_lean: object               # beta_5 regression (+ Lean interaction)
    did_with_fomo: object               # full model (+ FOMO z-score)
    parallel_trends: Optional[ParallelTrendsResult] = None
    n_events: int = 0
    n_treatment_firms: int = 0
    n_control_firms: int = 0
    n_observations: int = 0
    coefficient_table: pd.DataFrame = field(default_factory=pd.DataFrame)
    diagnostics: Optional["DiagnosticResults"] = None


@dataclass
class DiagnosticResults:
    """Statistical diagnostic tests for DiD regressions."""
    placebo_tests: pd.DataFrame
    bootstrap_ci: pd.DataFrame
    cluster_robust: pd.DataFrame
    normality: pd.DataFrame
    heteroskedasticity: pd.DataFrame
    vif: pd.DataFrame
    covariate_balance: pd.DataFrame
    autocorrelation: pd.DataFrame
    n_placebo_iterations: int = 500
    n_bootstrap_iterations: int = 1000


# Named tuple for normal-return estimation results.
# Avoids monkey-patching internal attributes on statsmodels fit objects.
# NOTE: Each instance holds a full DataFrame (event_data) and index (event_idx).
# For very large runs, callers could extract only the needed columns/slice
# to reduce memory, but in practice the per-estimate footprint is small
# relative to the factor/return DataFrames already in memory.
NormalReturnEstimate = namedtuple('NormalReturnEstimate', ['fit', 'event_data', 'event_idx'])


# =========================================================================
# CAR COMPUTATION
# =========================================================================

def _lookup_regime(event_date: pd.Timestamp, regime_dates: pd.DataFrame) -> str:
    """Return the regime label for the nearest trading day to event_date."""
    if regime_dates.empty:
        logger.warning("_lookup_regime: empty regime_dates — defaulting to 'Unknown'")
        return 'Unknown'
    match = regime_dates[regime_dates['DATE'] == event_date]
    if not match.empty:
        return match.iloc[0]['REGIME_LABEL']
    diffs = (regime_dates['DATE'] - event_date).abs()
    return regime_dates.loc[diffs.idxmin(), 'REGIME_LABEL']


def _estimate_normal_returns(
    ticker: str,
    store: DataStore,
    event_date: pd.Timestamp,
    estimation_window: Tuple[int, int] = _ESTIMATION_WINDOW,
) -> Optional[NormalReturnEstimate]:
    """
    Estimate FF5 normal-return model over the estimation window.

    Returns the fitted OLS model, or None if insufficient data.
    """
    returns = store.get_ticker_returns(ticker)
    if returns.empty or 'RETURN' not in returns.columns:
        return None

    factors = store.ff5[['DATE'] + _FF5_ALL + ['RF']].dropna().copy()
    factors['DATE'] = pd.to_datetime(factors['DATE'], errors='coerce')

    # Convert percent to decimal if needed
    for col in _FF5_ALL + ['RF']:
        if factors[col].abs().max() > 1.5:
            factors[col] = factors[col] / 100

    ret = returns[['DATE', 'RETURN']].copy()
    ret['DATE'] = pd.to_datetime(ret['DATE'], errors='coerce')
    if ret['RETURN'].abs().max() > 1.5:
        ret['RETURN'] = ret['RETURN'] / 100

    merged = ret.merge(factors, on='DATE', how='inner').sort_values('DATE').reset_index(drop=True)
    merged['EXCESS_RETURN'] = merged['RETURN'] - merged['RF']

    # Build trading-day index relative to event date
    all_dates = merged['DATE'].values
    event_date_np = np.datetime64(pd.Timestamp(event_date))
    event_idx = np.searchsorted(all_dates, event_date_np)
    if event_idx == 0 or event_idx >= len(merged):
        return None

    # Map calendar dates to trading-day offsets
    merged['TD_OFFSET'] = np.arange(len(merged)) - event_idx

    # Estimation window
    est = merged[
        (merged['TD_OFFSET'] >= estimation_window[0]) &
        (merged['TD_OFFSET'] <= estimation_window[1])
    ]

    if len(est) < _MIN_ESTIMATION_OBS:
        logger.debug("%s: only %d estimation obs (need %d) for event %s",
                     ticker, len(est), _MIN_ESTIMATION_OBS, event_date.date())
        return None

    y = est['EXCESS_RETURN']
    X = sm.add_constant(est[_FF5_ALL])

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        fit = sm.OLS(y, X).fit()

    return NormalReturnEstimate(fit=fit, event_data=merged, event_idx=event_idx)


def compute_car(
    ticker: str,
    store: DataStore,
    event_id: str,
    event_date: pd.Timestamp,
    is_treatment: bool,
    regime: str,
    pre_window: Tuple[int, int] = _PRE_EVENT_WINDOW,
    post_window: Tuple[int, int] = _POST_EVENT_WINDOW,
) -> Optional[EventCAR]:
    """
    Compute cumulative abnormal returns for a firm around an event.

    Uses a market-model (FF5) estimated over [-250, -11] to compute
    expected returns, then cumulates abnormal returns over the pre and
    post event windows.
    """
    estimate = _estimate_normal_returns(ticker, store, event_date)
    if estimate is None:
        return None

    merged = estimate.event_data
    ols_fit = estimate.fit
    full_window = (pre_window[0], post_window[1])

    event_obs = merged[
        (merged['TD_OFFSET'] >= full_window[0]) &
        (merged['TD_OFFSET'] <= full_window[1])
    ].copy()

    if len(event_obs) < 5:
        logger.debug("%s: only %d event-window obs for %s", ticker, len(event_obs), event_date.date())
        return None

    # Compute abnormal returns
    X_event = sm.add_constant(event_obs[_FF5_ALL])
    expected = ols_fit.predict(X_event)
    event_obs['AR'] = event_obs['EXCESS_RETURN'] - expected.values

    # CARs by sub-window
    pre_obs = event_obs[
        (event_obs['TD_OFFSET'] >= pre_window[0]) &
        (event_obs['TD_OFFSET'] <= pre_window[1])
    ]
    post_obs = event_obs[
        (event_obs['TD_OFFSET'] >= post_window[0]) &
        (event_obs['TD_OFFSET'] <= post_window[1])
    ]

    car_pre = pre_obs['AR'].sum() if len(pre_obs) > 0 else np.nan
    car_post = post_obs['AR'].sum() if len(post_obs) > 0 else np.nan
    car_full = event_obs['AR'].sum()

    return EventCAR(
        ticker=ticker,
        event_id=event_id,
        event_date=event_date,
        is_treatment=is_treatment,
        regime=regime,
        car_pre=car_pre,
        car_post=car_post,
        car_full=car_full,
        n_estimation_obs=int(ols_fit.nobs),
        n_event_obs=len(event_obs),
        alpha=ols_fit.params['const'],
        r_squared=ols_fit.rsquared,
    )


# =========================================================================
# MULTI-WINDOW EVENT STUDY
# =========================================================================

def compute_multi_window_car(
    ticker: str,
    store: DataStore,
    event_id: str,
    event_date: pd.Timestamp,
    is_treatment: bool,
    regime: str,
    lean: str,
    windows: List[int] = None,
) -> Optional[MultiWindowCAR]:
    """
    Compute CARs for multiple post-event horizons from a single FF5 estimation.

    For each window in `windows`, computes CAR[-1, +window].  The FF5 model
    is estimated once over [-250, -11], then abnormal returns are computed
    over the longest needed window and sliced for each horizon.

    Parameters
    ----------
    ticker : str
    store : DataStore
    event_id : str
    event_date : pd.Timestamp
    is_treatment : bool
    regime : str
    lean : str
        Political leaning label (Liberal, Conservative, Mixed).
    windows : list of int, optional
        Post-event horizons in trading days. Default: [5, 10, 15, 20, 30, 60, 90].

    Returns
    -------
    MultiWindowCAR or None
    """
    if windows is None:
        windows = _MULTI_WINDOWS

    estimate = _estimate_normal_returns(ticker, store, event_date)
    if estimate is None:
        return None

    merged = estimate.event_data
    ols_fit = estimate.fit
    max_window = max(windows)

    # Full range needed: day -1 to day +max_window
    event_obs = merged[
        (merged['TD_OFFSET'] >= _PRE_DAY) &
        (merged['TD_OFFSET'] <= max_window)
    ].copy()

    # NOTE: min-obs threshold is 3 here (vs 5 in compute_car) because the
    # per-window 50% fill check below (line ~380) provides window-level QC.
    # compute_car checks the full [-10,+10] range so needs a higher floor.
    if len(event_obs) < 3:
        logger.debug("%s: only %d obs for multi-window (need >=3) at %s",
                     ticker, len(event_obs), event_date.date())
        return None

    # Compute abnormal returns once for the full range
    X_event = sm.add_constant(event_obs[_FF5_ALL])
    expected = ols_fit.predict(X_event)
    event_obs['AR'] = event_obs['EXCESS_RETURN'] - expected.values

    # Slice CARs for each window
    cars = {}
    for w in windows:
        w_obs = event_obs[
            (event_obs['TD_OFFSET'] >= _PRE_DAY) &
            (event_obs['TD_OFFSET'] <= w)
        ]
        # Require at least 50% of expected trading days
        expected_days = w - _PRE_DAY + 1
        if len(w_obs) >= max(3, expected_days * 0.5):
            cars[w] = w_obs['AR'].sum()
        else:
            cars[w] = np.nan

    return MultiWindowCAR(
        ticker=ticker,
        event_id=event_id,
        event_date=event_date,
        is_treatment=is_treatment,
        regime=regime,
        lean=lean,
        cars=cars,
        n_estimation_obs=int(ols_fit.nobs),
        alpha=ols_fit.params['const'],
        r_squared=ols_fit.rsquared,
    )


def build_multi_window_panel(
    store: DataStore,
    regime_result: RegimeResult = None,
    windows: List[int] = None,
) -> Optional[pd.DataFrame]:
    """
    Build the (firm, event, window) panel for multi-window event study.

    Reads events from CULTURE_WAR_COMPANIES, computes CARs for each
    treatment firm and matched control across all post-event horizons.

    Returns a long-format DataFrame:
        TICKER, EVENT_ID, EVENT_DATE, IS_TREATMENT, REGIME, LEAN,
        WINDOW, CAR, N_EST_OBS, EST_R2
    """
    if windows is None:
        windows = _MULTI_WINDOWS

    # Load events from CULTURE_WAR_COMPANIES (the canonical source)
    events_df = store.read_table('CULTURE_WAR_COMPANIES')
    if events_df.empty:
        logger.error("No CULTURE_WAR_COMPANIES table found")
        return None

    events_df.columns = [c.upper() for c in events_df.columns]
    events_df['EVENT_DATE'] = pd.to_datetime(events_df['EVENT_DATE'], errors='coerce')
    events_df = events_df.dropna(subset=['EVENT_DATE'])

    # Load matched controls
    controls_df = store.read_table('CONTROL_COMPANIES')
    control_map = {}
    if not controls_df.empty:
        control_map = dict(zip(
            controls_df['TREATMENT_TICKER'],
            controls_df['CONTROL_TICKER'],
        ))
    if not control_map:
        logger.warning("build_multi_window_panel: no matched controls — "
                        "panel will contain treatment firms only")

    # Regime assignments
    if regime_result is None:
        regime_result = estimate_vix_regimes(store)
        if regime_result is None:
            logger.error("Cannot estimate VIX regimes")
            return None

    regime_dates = regime_result.regime_assignments[['DATE', 'REGIME_LABEL']].copy()
    regime_dates['DATE'] = pd.to_datetime(regime_dates['DATE'])

    # Fall back to computed alignment if events table lacks lean
    alignment_df = store.read_table('ESSAY2_POLITICAL_ALIGNMENT')
    alignment_map = {}
    if not alignment_df.empty and 'COMPUTED_LEANING' in alignment_df.columns:
        alignment_map = dict(zip(alignment_df['TICKER'], alignment_df['COMPUTED_LEANING']))

    rows = []
    n_computed = 0
    n_skipped = 0

    for idx, event in events_df.iterrows():
        event_date = event['EVENT_DATE']
        ticker = event.get('TICKER', None)
        if ticker is None:
            continue

        event_id = f"cw_{ticker}_{event_date.strftime('%Y%m%d')}"
        lean = event.get('ESTIMATED_POLITICAL_LEANING', '') or alignment_map.get(ticker, '')

        # Determine regime at event date
        regime_label = _lookup_regime(event_date, regime_dates)

        # Treatment firm
        treat_mw = compute_multi_window_car(
            ticker, store, event_id, event_date,
            is_treatment=True, regime=regime_label, lean=lean,
            windows=windows,
        )

        if treat_mw is None:
            n_skipped += 1
            continue

        n_computed += 1

        # Flatten to one row per window
        for w, car_val in treat_mw.cars.items():
            rows.append({
                'TICKER': ticker,
                'EVENT_ID': event_id,
                'EVENT_DATE': event_date,
                'IS_TREATMENT': True,
                'REGIME': regime_label,
                'LEAN': lean,
                'WINDOW': w,
                'CAR': car_val,
                'N_EST_OBS': treat_mw.n_estimation_obs,
                'EST_R2': treat_mw.r_squared,
            })

        # Matched control firm
        ctrl_ticker = control_map.get(ticker)
        if ctrl_ticker:
            ctrl_mw = compute_multi_window_car(
                ctrl_ticker, store, event_id, event_date,
                is_treatment=False, regime=regime_label, lean=lean,
                windows=windows,
            )
            if ctrl_mw is not None:
                for w, car_val in ctrl_mw.cars.items():
                    rows.append({
                        'TICKER': ctrl_ticker,
                        'EVENT_ID': event_id,
                        'EVENT_DATE': event_date,
                        'IS_TREATMENT': False,
                        'REGIME': regime_label,
                        'LEAN': lean,
                        'WINDOW': w,
                        'CAR': car_val,
                        'N_EST_OBS': ctrl_mw.n_estimation_obs,
                        'EST_R2': ctrl_mw.r_squared,
                    })

    if not rows:
        logger.error("Multi-window panel: no CARs computed (%d skipped)", n_skipped)
        return None

    panel = pd.DataFrame(rows)
    logger.info("Multi-window panel: %d rows, %d events, %d skipped, windows=%s",
                len(panel), n_computed, n_skipped, windows)
    return panel


@dataclass
class MultiWindowResult:
    """Results from the multi-window event study."""
    panel: pd.DataFrame                         # long-format (ticker, event, window)
    summary: pd.DataFrame                       # per-window summary stats
    treatment_vs_control: pd.DataFrame           # t-tests per window
    by_lean: pd.DataFrame                        # CARs by political lean x window
    n_events: int
    n_treatment: int
    n_control: int


def run_multi_window_event_study(
    store: DataStore,
    regime_result: RegimeResult = None,
    panel: pd.DataFrame = None,
    windows: List[int] = None,
) -> Optional[MultiWindowResult]:
    """
    Run multi-window event study: CAR[-1, +5/10/15/20/30/60/90].

    For each window, tests:
    1. Whether treatment CARs differ from zero (one-sample t-test)
    2. Whether treatment CARs differ from control (two-sample t-test)
    3. Whether CARs differ by political lean (ANOVA / Kruskal-Wallis)

    Parameters
    ----------
    store : DataStore
    regime_result : RegimeResult, optional
    panel : pd.DataFrame, optional
        Pre-computed multi-window panel. If None, builds from scratch.
    windows : list of int, optional

    Returns
    -------
    MultiWindowResult or None
    """
    if windows is None:
        windows = _MULTI_WINDOWS

    if panel is None:
        panel = build_multi_window_panel(store, regime_result, windows)
        if panel is None:
            return None

    # ---- Summary stats per window ----
    summary_rows = []
    for w in windows:
        w_data = panel[panel['WINDOW'] == w].dropna(subset=['CAR'])
        treat = w_data[w_data['IS_TREATMENT']]
        ctrl = w_data[~w_data['IS_TREATMENT']]

        # One-sample t-test: treatment CARs != 0
        if len(treat) >= 3:
            t_stat, p_val = stats.ttest_1samp(treat['CAR'], 0)
        else:
            t_stat, p_val = np.nan, np.nan

        summary_rows.append({
            'WINDOW': f'[-1, +{w}]',
            'WINDOW_DAYS': w,
            'N_TREAT': len(treat),
            'N_CTRL': len(ctrl),
            'MEAN_CAR_TREAT': treat['CAR'].mean() if len(treat) > 0 else np.nan,
            'STD_CAR_TREAT': treat['CAR'].std() if len(treat) > 0 else np.nan,
            'MEDIAN_CAR_TREAT': treat['CAR'].median() if len(treat) > 0 else np.nan,
            'MEAN_CAR_CTRL': ctrl['CAR'].mean() if len(ctrl) > 0 else np.nan,
            'T_STAT_VS_ZERO': t_stat,
            'P_VALUE_VS_ZERO': p_val,
        })

    summary_df = pd.DataFrame(summary_rows)

    # BH correction on the one-sample p-values
    if not summary_df.empty:
        p_vals = summary_df['P_VALUE_VS_ZERO'].tolist()
        summary_df['BH_SIGNIFICANT'] = benjamini_hochberg(p_vals, q=0.10)

    # ---- Treatment vs Control t-tests per window ----
    tvc_rows = []
    for w in windows:
        w_data = panel[panel['WINDOW'] == w].dropna(subset=['CAR'])
        treat = w_data[w_data['IS_TREATMENT']]['CAR']
        ctrl = w_data[~w_data['IS_TREATMENT']]['CAR']

        if len(treat) >= 3 and len(ctrl) >= 3:
            t_stat, p_val = stats.ttest_ind(treat, ctrl, equal_var=False)
            diff = treat.mean() - ctrl.mean()
            # Cohen's d
            pooled_std = np.sqrt((treat.std()**2 + ctrl.std()**2) / 2)
            cohens_d = diff / pooled_std if pooled_std > 0 else np.nan
        else:
            t_stat, p_val, diff, cohens_d = np.nan, np.nan, np.nan, np.nan

        tvc_rows.append({
            'WINDOW': f'[-1, +{w}]',
            'WINDOW_DAYS': w,
            'DIFF_TREAT_CTRL': diff,
            'T_STAT': t_stat,
            'P_VALUE': p_val,
            'COHENS_D': cohens_d,
        })

    tvc_df = pd.DataFrame(tvc_rows)
    if not tvc_df.empty:
        tvc_df['BH_SIGNIFICANT'] = benjamini_hochberg(tvc_df['P_VALUE'].tolist(), q=0.10)

    # ---- CARs by political lean x window ----
    lean_rows = []
    for w in windows:
        w_treat = panel[(panel['WINDOW'] == w) & (panel['IS_TREATMENT'])].dropna(subset=['CAR'])
        for lean_val in ['Liberal', 'Conservative', 'Mixed']:
            lean_data = w_treat[w_treat['LEAN'].str.strip() == lean_val]
            if len(lean_data) > 0:
                lean_rows.append({
                    'WINDOW': f'[-1, +{w}]',
                    'WINDOW_DAYS': w,
                    'LEAN': lean_val,
                    'N': len(lean_data),
                    'MEAN_CAR': lean_data['CAR'].mean(),
                    'STD_CAR': lean_data['CAR'].std(),
                    'MEDIAN_CAR': lean_data['CAR'].median(),
                })

    lean_df = pd.DataFrame(lean_rows)

    n_treat = panel[panel['IS_TREATMENT']]['TICKER'].nunique()
    n_ctrl = panel[~panel['IS_TREATMENT']]['TICKER'].nunique()
    n_events = panel['EVENT_ID'].nunique()

    logger.info("Multi-window event study: %d events, %d treatment, %d control, "
                "%d windows", n_events, n_treat, n_ctrl, len(windows))

    return MultiWindowResult(
        panel=panel,
        summary=summary_df,
        treatment_vs_control=tvc_df,
        by_lean=lean_df,
        n_events=n_events,
        n_treatment=n_treat,
        n_control=n_ctrl,
    )


def save_multi_window_results(
    store: DataStore,
    result: MultiWindowResult,
) -> dict:
    """Persist multi-window event study results to the database."""
    results = {}
    timestamp = pd.Timestamp.now().isoformat()

    results['ESSAY2_MULTI_WINDOW_PANEL'] = store.write_table(
        result.panel.assign(RUN_TIMESTAMP=timestamp),
        'ESSAY2_MULTI_WINDOW_PANEL', replace=True,
    )

    results['ESSAY2_MULTI_WINDOW_SUMMARY'] = store.write_table(
        result.summary.assign(RUN_TIMESTAMP=timestamp),
        'ESSAY2_MULTI_WINDOW_SUMMARY', replace=True,
    )

    results['ESSAY2_MULTI_WINDOW_TREAT_VS_CTRL'] = store.write_table(
        result.treatment_vs_control.assign(RUN_TIMESTAMP=timestamp),
        'ESSAY2_MULTI_WINDOW_TREAT_VS_CTRL', replace=True,
    )

    if not result.by_lean.empty:
        results['ESSAY2_MULTI_WINDOW_BY_LEAN'] = store.write_table(
            result.by_lean.assign(RUN_TIMESTAMP=timestamp),
            'ESSAY2_MULTI_WINDOW_BY_LEAN', replace=True,
        )

    saved = sum(1 for v in results.values() if v is not None)
    logger.info("Multi-window event study: saved %d/%d tables", saved, len(results))
    return results


# =========================================================================
# CONTAGION TEST
# =========================================================================

@dataclass
class ContagionResult:
    """Results from the industry contagion / spillover test."""
    peer_panel: pd.DataFrame            # (peer_ticker, event, window) CARs
    summary: pd.DataFrame               # per-window: mean peer CAR, t-test vs 0
    peer_vs_nonpeer: pd.DataFrame       # per-window: peer vs non-peer diff
    by_event_lean: pd.DataFrame         # contagion by triggering firm's lean
    n_events_with_peers: int
    n_unique_peers: int
    n_unique_nonpeers: int


def _compute_peer_cars(
    ticker: str,
    store: DataStore,
    event_date: pd.Timestamp,
    event_id: str,
    regime: str,
    role: str,                          # 'PEER' or 'NON_PEER'
    event_lean: str,
    windows: List[int] = None,
) -> Optional[dict]:
    """Compute multi-window CARs for a single peer/non-peer firm."""
    if windows is None:
        windows = _MULTI_WINDOWS

    estimate = _estimate_normal_returns(ticker, store, event_date)
    if estimate is None:
        return None

    merged = estimate.event_data
    ols_fit = estimate.fit
    max_window = max(windows)

    event_obs = merged[
        (merged['TD_OFFSET'] >= _PRE_DAY) &
        (merged['TD_OFFSET'] <= max_window)
    ].copy()

    if len(event_obs) < 3:
        return None

    X_event = sm.add_constant(event_obs[_FF5_ALL])
    expected = ols_fit.predict(X_event)
    event_obs['AR'] = event_obs['EXCESS_RETURN'] - expected.values

    rows = []
    for w in windows:
        w_obs = event_obs[
            (event_obs['TD_OFFSET'] >= _PRE_DAY) &
            (event_obs['TD_OFFSET'] <= w)
        ]
        expected_days = w - _PRE_DAY + 1
        if len(w_obs) >= max(3, expected_days * 0.5):
            car_val = w_obs['AR'].sum()
        else:
            car_val = np.nan

        rows.append({
            'TICKER': ticker,
            'EVENT_ID': event_id,
            'EVENT_DATE': event_date,
            'REGIME': regime,
            'ROLE': role,
            'EVENT_LEAN': event_lean,
            'WINDOW': w,
            'CAR': car_val,
            'N_EST_OBS': int(ols_fit.nobs),
            'EST_R2': ols_fit.rsquared,
        })

    return rows


def run_contagion_test(
    store: DataStore,
    regime_result: RegimeResult = None,
    windows: List[int] = None,
    max_nonpeers: int = 5,
) -> Optional[ContagionResult]:
    """
    Test for industry contagion / spillover from culture war events.

    For each event (firm i in industry j on date t):
      - PEER firms: other culture-war firms in the same industry (same NAICS
        or INDUSTRY label), excluding firm i
      - NON-PEER firms: culture-war firms in *different* industries, sampled
        to keep panel balanced

    Computes FF5 CARs for peers and non-peers across all windows, then tests:
      1. Peer CARs != 0 (direct contagion)
      2. Peer CARs != non-peer CARs (differential contagion)

    Parameters
    ----------
    store : DataStore
    regime_result : RegimeResult, optional
    windows : list of int, optional
    max_nonpeers : int
        Max non-peer firms to sample per event (for balance).

    Returns
    -------
    ContagionResult or None
    """
    if windows is None:
        windows = _MULTI_WINDOWS

    # Load events
    events_df = store.read_table('CULTURE_WAR_COMPANIES')
    if events_df.empty:
        logger.error("Contagion: no CULTURE_WAR_COMPANIES table")
        return None

    events_df.columns = [c.upper() for c in events_df.columns]
    events_df['EVENT_DATE'] = pd.to_datetime(events_df['EVENT_DATE'], errors='coerce')
    events_df = events_df.dropna(subset=['EVENT_DATE'])

    # Build industry lookup: ticker -> (INDUSTRY, NAICS_CODE)
    ticker_industry = {}
    for _, row in events_df.iterrows():
        t = row.get('TICKER')
        if t:
            ticker_industry[t] = (
                row.get('INDUSTRY', 'Unknown'),
                str(row.get('NAICS_CODE', '999999')),
            )

    all_tickers = list(ticker_industry.keys())

    # Fall back to computed alignment if events table lacks lean
    alignment_df = store.read_table('ESSAY2_POLITICAL_ALIGNMENT')
    alignment_map = {}
    if not alignment_df.empty and 'COMPUTED_LEANING' in alignment_df.columns:
        alignment_map = dict(zip(alignment_df['TICKER'], alignment_df['COMPUTED_LEANING']))

    # Regime assignments — callers should pass regime_result to avoid
    # redundant Markov-switching estimation (which is deterministic but slow).
    if regime_result is None:
        regime_result = estimate_vix_regimes(store)
        if regime_result is None:
            return None

    regime_dates = regime_result.regime_assignments[['DATE', 'REGIME_LABEL']].copy()
    regime_dates['DATE'] = pd.to_datetime(regime_dates['DATE'])

    all_rows = []
    n_events_with_peers = 0
    rng = np.random.RandomState(42)

    for _, event in events_df.iterrows():
        event_date = event['EVENT_DATE']
        event_ticker = event.get('TICKER')
        if event_ticker is None:
            continue

        event_id = f"cw_{event_ticker}_{event_date.strftime('%Y%m%d')}"
        event_lean = event.get('ESTIMATED_POLITICAL_LEANING', '') or alignment_map.get(event_ticker, '')
        event_ind, event_naics = ticker_industry.get(event_ticker, ('Unknown', '999999'))

        # Determine regime
        regime_label = _lookup_regime(event_date, regime_dates)

        # Identify peers: same industry OR same NAICS (excluding event firm and 999999)
        peers = []
        nonpeers = []
        for t in all_tickers:
            if t == event_ticker:
                continue
            t_ind, t_naics = ticker_industry.get(t, ('Unknown', '999999'))
            is_peer = False
            if event_naics != '999999' and t_naics == event_naics:
                is_peer = True
            elif event_ind != 'Unknown' and t_ind == event_ind:
                is_peer = True
            if is_peer:
                peers.append(t)
            else:
                nonpeers.append(t)

        if not peers:
            continue

        n_events_with_peers += 1

        # Compute peer CARs
        for peer_ticker in peers:
            rows = _compute_peer_cars(
                peer_ticker, store, event_date, event_id,
                regime_label, 'PEER', event_lean, windows,
            )
            if rows:
                all_rows.extend(rows)

        # Sample non-peers for balance
        sampled_nonpeers = nonpeers
        if len(nonpeers) > max_nonpeers:
            sampled_nonpeers = list(rng.choice(nonpeers, max_nonpeers, replace=False))

        for np_ticker in sampled_nonpeers:
            rows = _compute_peer_cars(
                np_ticker, store, event_date, event_id,
                regime_label, 'NON_PEER', event_lean, windows,
            )
            if rows:
                all_rows.extend(rows)

    if not all_rows:
        logger.error("Contagion: no peer CARs computed")
        return None

    peer_panel = pd.DataFrame(all_rows)
    logger.info("Contagion panel: %d rows, %d events with peers",
                len(peer_panel), n_events_with_peers)

    # ---- Summary: peer CARs vs zero per window ----
    summary_rows = []
    for w in windows:
        w_peers = peer_panel[
            (peer_panel['WINDOW'] == w) & (peer_panel['ROLE'] == 'PEER')
        ].dropna(subset=['CAR'])

        if len(w_peers) >= 3:
            t_stat, p_val = stats.ttest_1samp(w_peers['CAR'], 0)
        else:
            t_stat, p_val = np.nan, np.nan

        summary_rows.append({
            'WINDOW': f'[-1, +{w}]',
            'WINDOW_DAYS': w,
            'N_PEER_OBS': len(w_peers),
            'MEAN_PEER_CAR': w_peers['CAR'].mean() if len(w_peers) > 0 else np.nan,
            'STD_PEER_CAR': w_peers['CAR'].std() if len(w_peers) > 0 else np.nan,
            'MEDIAN_PEER_CAR': w_peers['CAR'].median() if len(w_peers) > 0 else np.nan,
            'T_STAT_VS_ZERO': t_stat,
            'P_VALUE_VS_ZERO': p_val,
        })

    summary_df = pd.DataFrame(summary_rows)
    if not summary_df.empty:
        summary_df['BH_SIGNIFICANT'] = benjamini_hochberg(
            summary_df['P_VALUE_VS_ZERO'].tolist(), q=0.10)

    # ---- Peer vs Non-Peer t-tests per window ----
    pvnp_rows = []
    for w in windows:
        w_data = peer_panel[peer_panel['WINDOW'] == w].dropna(subset=['CAR'])
        peers_car = w_data[w_data['ROLE'] == 'PEER']['CAR']
        nonpeers_car = w_data[w_data['ROLE'] == 'NON_PEER']['CAR']

        if len(peers_car) >= 3 and len(nonpeers_car) >= 3:
            t_stat, p_val = stats.ttest_ind(peers_car, nonpeers_car, equal_var=False)
            diff = peers_car.mean() - nonpeers_car.mean()
            pooled_std = np.sqrt((peers_car.std()**2 + nonpeers_car.std()**2) / 2)
            cohens_d = diff / pooled_std if pooled_std > 0 else np.nan
        else:
            t_stat, p_val, diff, cohens_d = np.nan, np.nan, np.nan, np.nan

        pvnp_rows.append({
            'WINDOW': f'[-1, +{w}]',
            'WINDOW_DAYS': w,
            'N_PEERS': len(peers_car),
            'N_NONPEERS': len(nonpeers_car),
            'MEAN_PEER_CAR': peers_car.mean() if len(peers_car) > 0 else np.nan,
            'MEAN_NONPEER_CAR': nonpeers_car.mean() if len(nonpeers_car) > 0 else np.nan,
            'DIFF_PEER_NONPEER': diff,
            'T_STAT': t_stat,
            'P_VALUE': p_val,
            'COHENS_D': cohens_d,
        })

    pvnp_df = pd.DataFrame(pvnp_rows)
    if not pvnp_df.empty:
        pvnp_df['BH_SIGNIFICANT'] = benjamini_hochberg(pvnp_df['P_VALUE'].tolist(), q=0.10)

    # ---- Contagion by triggering firm's political lean ----
    lean_rows = []
    for w in windows:
        w_peers = peer_panel[
            (peer_panel['WINDOW'] == w) & (peer_panel['ROLE'] == 'PEER')
        ].dropna(subset=['CAR'])

        for lean_val in ['Liberal', 'Conservative', 'Mixed']:
            lean_data = w_peers[w_peers['EVENT_LEAN'].str.strip() == lean_val]
            if len(lean_data) > 0:
                lean_rows.append({
                    'WINDOW': f'[-1, +{w}]',
                    'WINDOW_DAYS': w,
                    'EVENT_LEAN': lean_val,
                    'N': len(lean_data),
                    'MEAN_PEER_CAR': lean_data['CAR'].mean(),
                    'STD_PEER_CAR': lean_data['CAR'].std(),
                    'MEDIAN_PEER_CAR': lean_data['CAR'].median(),
                })

    lean_df = pd.DataFrame(lean_rows)

    n_unique_peers = peer_panel[peer_panel['ROLE'] == 'PEER']['TICKER'].nunique()
    n_unique_nonpeers = peer_panel[peer_panel['ROLE'] == 'NON_PEER']['TICKER'].nunique()

    logger.info("Contagion test: %d events, %d unique peers, %d unique non-peers",
                n_events_with_peers, n_unique_peers, n_unique_nonpeers)

    return ContagionResult(
        peer_panel=peer_panel,
        summary=summary_df,
        peer_vs_nonpeer=pvnp_df,
        by_event_lean=lean_df,
        n_events_with_peers=n_events_with_peers,
        n_unique_peers=n_unique_peers,
        n_unique_nonpeers=n_unique_nonpeers,
    )


def save_contagion_results(
    store: DataStore,
    result: ContagionResult,
) -> dict:
    """Persist contagion test results to the database."""
    results = {}
    timestamp = pd.Timestamp.now().isoformat()

    results['ESSAY2_CONTAGION_PANEL'] = store.write_table(
        result.peer_panel.assign(RUN_TIMESTAMP=timestamp),
        'ESSAY2_CONTAGION_PANEL', replace=True,
    )

    results['ESSAY2_CONTAGION_SUMMARY'] = store.write_table(
        result.summary.assign(RUN_TIMESTAMP=timestamp),
        'ESSAY2_CONTAGION_SUMMARY', replace=True,
    )

    results['ESSAY2_CONTAGION_PEER_VS_NONPEER'] = store.write_table(
        result.peer_vs_nonpeer.assign(RUN_TIMESTAMP=timestamp),
        'ESSAY2_CONTAGION_PEER_VS_NONPEER', replace=True,
    )

    if not result.by_event_lean.empty:
        results['ESSAY2_CONTAGION_BY_LEAN'] = store.write_table(
            result.by_event_lean.assign(RUN_TIMESTAMP=timestamp),
            'ESSAY2_CONTAGION_BY_LEAN', replace=True,
        )

    saved = sum(1 for v in results.values() if v is not None)
    logger.info("Contagion test: saved %d/%d tables", saved, len(results))
    return results


# =========================================================================
# CONSUMER-FACING vs B2B INDUSTRY CLASSIFICATION
# =========================================================================

# Keywords in industry names that indicate consumer-facing
_CONSUMER_KEYWORDS = {
    'retail', 'restaurant', 'food', 'beverage', 'apparel', 'coffeehouse',
    'department store', 'grocery', 'beauty', 'personal care', 'sporting',
    'candy', 'ice cream', 'tobacco', 'pharmacy', 'consumer', 'media',
    'streaming', 'entertainment', 'social media', 'e-commerce',
    'merchandise', 'farm', 'ranch',
}

_B2B_KEYWORDS = {
    'banking', 'finance', 'insurance', 'oil', 'gas', 'mining',
    'aerospace', 'defense', 'enterprise', 'software', 'cloud',
    'information technology', 'life sciences', 'manufacturing',
    'courier', 'logistics', 'wholesale', 'telecommunications',
    'fintech', 'payments',
}


def _classify_industry_facing(industry: str) -> str:
    """Classify an industry as CONSUMER, B2B, or MIXED."""
    ind_lower = industry.lower()
    is_consumer = any(kw in ind_lower for kw in _CONSUMER_KEYWORDS)
    is_b2b = any(kw in ind_lower for kw in _B2B_KEYWORDS)
    if is_consumer and not is_b2b:
        return 'CONSUMER'
    elif is_b2b and not is_consumer:
        return 'B2B'
    elif is_consumer and is_b2b:
        return 'MIXED'
    # Fallback: use NAICS 2-digit sector
    return 'UNCLASSIFIED'


def _get_naics_2_sector(naics_code: str) -> str:
    """Map 2-digit NAICS to a broad sector for non-peer filtering."""
    n2 = str(naics_code)[:2]
    sector_map = {
        '11': 'Agriculture', '21': 'Mining',
        '22': 'Utilities', '23': 'Construction',
        '31': 'Manufacturing', '32': 'Manufacturing', '33': 'Manufacturing',
        '42': 'Wholesale', '44': 'Retail', '45': 'Retail',
        '48': 'Transportation', '49': 'Transportation',
        '51': 'Information', '52': 'Finance',
        '53': 'Real Estate', '54': 'Professional',
        '55': 'Management', '56': 'Admin',
        '61': 'Education', '62': 'Healthcare',
        '71': 'Arts/Recreation', '72': 'Accommodation/Food',
        '81': 'Other Services', '92': 'Public Admin',
        '99': 'Unknown',
    }
    return sector_map.get(n2, 'Unknown')


# =========================================================================
# ENHANCED CONTAGION TESTS
# =========================================================================

@dataclass
class EnhancedContagionResult:
    """Results from tighter non-peer definition + industry heterogeneity."""
    # 1. Tight non-peer differential
    tight_peer_vs_nonpeer: pd.DataFrame
    # 2. Consumer vs B2B contagion
    by_industry_type: pd.DataFrame
    consumer_vs_b2b_tests: pd.DataFrame
    # 3. Mixed-lean mechanism analysis
    lean_mechanism: pd.DataFrame
    lean_pairwise_tests: pd.DataFrame
    # Panel for persistence
    panel: pd.DataFrame
    n_events: int


def run_enhanced_contagion(
    store: DataStore,
    regime_result: RegimeResult = None,
    windows: List[int] = None,
    max_nonpeers: int = 5,
) -> Optional[EnhancedContagionResult]:
    """
    Enhanced contagion tests with three improvements:

    1. **Tight non-peer definition**: Non-peers must be in a *different
       2-digit NAICS sector* (not just different industry label), excluding
       NAICS 999999.  This eliminates fuzzy same-sector contamination.

    2. **Consumer-facing vs B2B heterogeneity**: Tests whether contagion
       hits harder in consumer-facing industries (retail, food, media)
       vs B2B (banking, tech, logistics).

    3. **Mixed-lean mechanism**: Decomposes contagion by triggering firm's
       political lean with pairwise tests (Mixed vs Liberal, Mixed vs
       Conservative) to confirm uncertainty as the transmission channel.
    """
    if windows is None:
        windows = _MULTI_WINDOWS

    events_df = store.read_table('CULTURE_WAR_COMPANIES')
    if events_df.empty:
        return None

    events_df.columns = [c.upper() for c in events_df.columns]
    events_df['EVENT_DATE'] = pd.to_datetime(events_df['EVENT_DATE'], errors='coerce')
    events_df = events_df.dropna(subset=['EVENT_DATE'])

    # Build ticker -> (industry, naics, sector, facing) lookup
    ticker_meta = {}
    for _, row in events_df.iterrows():
        t = row.get('TICKER')
        if t and t not in ticker_meta:
            ind = row.get('INDUSTRY', 'Unknown')
            naics = str(row.get('NAICS_CODE', '999999'))
            sector = _get_naics_2_sector(naics)
            facing = _classify_industry_facing(ind)
            ticker_meta[t] = {
                'INDUSTRY': ind, 'NAICS': naics,
                'SECTOR': sector, 'FACING': facing,
            }

    all_tickers = list(ticker_meta.keys())

    # Log industry classification distribution (UNCLASSIFIED firms excluded
    # from consumer-vs-B2B heterogeneity analysis)
    n_unclassified = sum(1 for m in ticker_meta.values() if m['FACING'] == 'UNCLASSIFIED')
    if n_unclassified > 0:
        logger.info("Enhanced contagion: %d/%d firms UNCLASSIFIED (excluded from "
                     "consumer-vs-B2B analysis)", n_unclassified, len(ticker_meta))

    # Fall back to computed alignment if events table lacks lean
    alignment_df = store.read_table('ESSAY2_POLITICAL_ALIGNMENT')
    alignment_map = {}
    if not alignment_df.empty and 'COMPUTED_LEANING' in alignment_df.columns:
        alignment_map = dict(zip(alignment_df['TICKER'], alignment_df['COMPUTED_LEANING']))

    if regime_result is None:
        regime_result = estimate_vix_regimes(store)
        if regime_result is None:
            return None

    regime_dates = regime_result.regime_assignments[['DATE', 'REGIME_LABEL']].copy()
    regime_dates['DATE'] = pd.to_datetime(regime_dates['DATE'])

    all_rows = []
    n_events = 0
    rng = np.random.RandomState(42)

    for _, event in events_df.iterrows():
        event_date = event['EVENT_DATE']
        event_ticker = event.get('TICKER')
        if event_ticker is None:
            continue

        event_id = f"cw_{event_ticker}_{event_date.strftime('%Y%m%d')}"
        event_lean = event.get('ESTIMATED_POLITICAL_LEANING', '') or alignment_map.get(event_ticker, '')
        ev_meta = ticker_meta.get(event_ticker, {})
        event_ind = ev_meta.get('INDUSTRY', 'Unknown')
        event_naics = ev_meta.get('NAICS', '999999')
        event_sector = ev_meta.get('SECTOR', 'Unknown')
        event_facing = ev_meta.get('FACING', 'UNCLASSIFIED')

        # Regime
        regime_label = _lookup_regime(event_date, regime_dates)

        # Classify all other tickers
        peers = []
        tight_nonpeers = []  # different NAICS-2 sector, known sector

        for t in all_tickers:
            if t == event_ticker:
                continue
            t_meta = ticker_meta.get(t, {})
            t_ind = t_meta.get('INDUSTRY', 'Unknown')
            t_naics = t_meta.get('NAICS', '999999')
            t_sector = t_meta.get('SECTOR', 'Unknown')

            # Peer: same NAICS or same industry label
            is_peer = False
            if event_naics != '999999' and t_naics == event_naics:
                is_peer = True
            elif event_ind != 'Unknown' and t_ind == event_ind:
                is_peer = True

            if is_peer:
                peers.append(t)
            else:
                # Tight non-peer: different 2-digit NAICS sector
                # Both must have known sectors
                if (event_sector != 'Unknown' and t_sector != 'Unknown'
                        and event_sector != t_sector):
                    tight_nonpeers.append(t)

        if not peers:
            continue

        n_events += 1

        # Compute peer CARs
        for peer_ticker in peers:
            p_meta = ticker_meta.get(peer_ticker, {})
            rows = _compute_peer_cars(
                peer_ticker, store, event_date, event_id,
                regime_label, 'PEER', event_lean, windows,
            )
            if rows:
                for r in rows:
                    r['EVENT_INDUSTRY'] = event_ind
                    r['EVENT_SECTOR'] = event_sector
                    r['EVENT_FACING'] = event_facing
                    r['PEER_INDUSTRY'] = p_meta.get('INDUSTRY', '')
                    r['PEER_FACING'] = p_meta.get('FACING', '')
                all_rows.extend(rows)

        # Sample tight non-peers
        sampled = tight_nonpeers
        if len(tight_nonpeers) > max_nonpeers:
            sampled = list(rng.choice(tight_nonpeers, max_nonpeers, replace=False))

        for np_ticker in sampled:
            np_meta = ticker_meta.get(np_ticker, {})
            rows = _compute_peer_cars(
                np_ticker, store, event_date, event_id,
                regime_label, 'TIGHT_NON_PEER', event_lean, windows,
            )
            if rows:
                for r in rows:
                    r['EVENT_INDUSTRY'] = event_ind
                    r['EVENT_SECTOR'] = event_sector
                    r['EVENT_FACING'] = event_facing
                    r['PEER_INDUSTRY'] = np_meta.get('INDUSTRY', '')
                    r['PEER_FACING'] = np_meta.get('FACING', '')
                all_rows.extend(rows)

    if not all_rows:
        logger.error("Enhanced contagion: no data")
        return None

    panel = pd.DataFrame(all_rows)
    logger.info("Enhanced contagion panel: %d rows, %d events", len(panel), n_events)

    # ================================================================
    # 1. TIGHT NON-PEER DIFFERENTIAL
    # ================================================================
    tight_rows = []
    for w in windows:
        w_data = panel[panel['WINDOW'] == w].dropna(subset=['CAR'])
        peers_car = w_data[w_data['ROLE'] == 'PEER']['CAR']
        tnp_car = w_data[w_data['ROLE'] == 'TIGHT_NON_PEER']['CAR']

        if len(peers_car) >= 3 and len(tnp_car) >= 3:
            t_stat, p_val = stats.ttest_ind(peers_car, tnp_car, equal_var=False)
            diff = peers_car.mean() - tnp_car.mean()
            pooled_std = np.sqrt((peers_car.std()**2 + tnp_car.std()**2) / 2)
            cohens_d = diff / pooled_std if pooled_std > 0 else np.nan
        else:
            t_stat, p_val, diff, cohens_d = np.nan, np.nan, np.nan, np.nan

        tight_rows.append({
            'WINDOW': f'[-1, +{w}]',
            'WINDOW_DAYS': w,
            'N_PEERS': len(peers_car),
            'N_TIGHT_NONPEERS': len(tnp_car),
            'MEAN_PEER_CAR': peers_car.mean() if len(peers_car) > 0 else np.nan,
            'MEAN_NONPEER_CAR': tnp_car.mean() if len(tnp_car) > 0 else np.nan,
            'DIFF': diff,
            'T_STAT': t_stat,
            'P_VALUE': p_val,
            'COHENS_D': cohens_d,
        })

    tight_df = pd.DataFrame(tight_rows)
    if not tight_df.empty:
        tight_df['BH_SIGNIFICANT'] = benjamini_hochberg(tight_df['P_VALUE'].tolist(), q=0.10)

    # ================================================================
    # 2. CONSUMER vs B2B CONTAGION HETEROGENEITY
    # ================================================================
    # Peer CARs broken by the *event firm's* industry type
    peers_only = panel[panel['ROLE'] == 'PEER'].copy()

    industry_type_rows = []
    for w in windows:
        w_peers = peers_only[peers_only['WINDOW'] == w].dropna(subset=['CAR'])
        for facing in ['CONSUMER', 'B2B', 'UNCLASSIFIED']:
            f_data = w_peers[w_peers['EVENT_FACING'] == facing]
            if len(f_data) >= 3:
                t_stat, p_val = stats.ttest_1samp(f_data['CAR'], 0)
            else:
                t_stat, p_val = np.nan, np.nan

            industry_type_rows.append({
                'WINDOW': f'[-1, +{w}]',
                'WINDOW_DAYS': w,
                'EVENT_FACING': facing,
                'N': len(f_data),
                'MEAN_PEER_CAR': f_data['CAR'].mean() if len(f_data) > 0 else np.nan,
                'STD_PEER_CAR': f_data['CAR'].std() if len(f_data) > 0 else np.nan,
                'MEDIAN_PEER_CAR': f_data['CAR'].median() if len(f_data) > 0 else np.nan,
                'T_STAT_VS_ZERO': t_stat,
                'P_VALUE_VS_ZERO': p_val,
            })

    ind_type_df = pd.DataFrame(industry_type_rows)
    if not ind_type_df.empty:
        valid_p = ind_type_df['P_VALUE_VS_ZERO'].notna()
        if valid_p.any():
            bh_sig = benjamini_hochberg(
                ind_type_df.loc[valid_p, 'P_VALUE_VS_ZERO'].tolist(), q=0.10
            )
            ind_type_df['BH_SIGNIFICANT'] = False
            ind_type_df.loc[valid_p, 'BH_SIGNIFICANT'] = bh_sig
        else:
            ind_type_df['BH_SIGNIFICANT'] = False

    # Consumer vs B2B pairwise tests per window
    cvb_rows = []
    for w in windows:
        w_peers = peers_only[peers_only['WINDOW'] == w].dropna(subset=['CAR'])
        cons = w_peers[w_peers['EVENT_FACING'] == 'CONSUMER']['CAR']
        b2b = w_peers[w_peers['EVENT_FACING'] == 'B2B']['CAR']

        if len(cons) >= 3 and len(b2b) >= 3:
            t_stat, p_val = stats.ttest_ind(cons, b2b, equal_var=False)
            diff = cons.mean() - b2b.mean()
            pooled_std = np.sqrt((cons.std()**2 + b2b.std()**2) / 2)
            cohens_d = diff / pooled_std if pooled_std > 0 else np.nan
        else:
            t_stat, p_val, diff, cohens_d = np.nan, np.nan, np.nan, np.nan

        cvb_rows.append({
            'WINDOW': f'[-1, +{w}]',
            'WINDOW_DAYS': w,
            'N_CONSUMER': len(cons),
            'N_B2B': len(b2b),
            'MEAN_CONSUMER': cons.mean() if len(cons) > 0 else np.nan,
            'MEAN_B2B': b2b.mean() if len(b2b) > 0 else np.nan,
            'DIFF_CONS_B2B': diff,
            'T_STAT': t_stat,
            'P_VALUE': p_val,
            'COHENS_D': cohens_d,
        })

    cvb_df = pd.DataFrame(cvb_rows)
    if not cvb_df.empty:
        cvb_df['BH_SIGNIFICANT'] = benjamini_hochberg(cvb_df['P_VALUE'].tolist(), q=0.10)

    # ================================================================
    # 3. MIXED-LEAN MECHANISM: UNCERTAINTY CHANNEL
    # ================================================================
    # Per-lean peer CARs with t-tests vs zero + pairwise tests
    lean_mech_rows = []
    for w in windows:
        w_peers = peers_only[peers_only['WINDOW'] == w].dropna(subset=['CAR'])
        for lean_val in ['Liberal', 'Conservative', 'Mixed']:
            lean_data = w_peers[w_peers['EVENT_LEAN'].str.strip() == lean_val]
            if len(lean_data) >= 3:
                t_stat, p_val = stats.ttest_1samp(lean_data['CAR'], 0)
            else:
                t_stat, p_val = np.nan, np.nan
            lean_mech_rows.append({
                'WINDOW': f'[-1, +{w}]',
                'WINDOW_DAYS': w,
                'EVENT_LEAN': lean_val,
                'N': len(lean_data),
                'MEAN_PEER_CAR': lean_data['CAR'].mean() if len(lean_data) > 0 else np.nan,
                'STD_PEER_CAR': lean_data['CAR'].std() if len(lean_data) > 0 else np.nan,
                'T_STAT_VS_ZERO': t_stat,
                'P_VALUE_VS_ZERO': p_val,
            })

    lean_mech_df = pd.DataFrame(lean_mech_rows)

    # Pairwise: Mixed vs Liberal, Mixed vs Conservative
    pw_rows = []
    for w in windows:
        w_peers = peers_only[peers_only['WINDOW'] == w].dropna(subset=['CAR'])
        mixed = w_peers[w_peers['EVENT_LEAN'].str.strip() == 'Mixed']['CAR']
        liberal = w_peers[w_peers['EVENT_LEAN'].str.strip() == 'Liberal']['CAR']
        conserv = w_peers[w_peers['EVENT_LEAN'].str.strip() == 'Conservative']['CAR']

        for comparison, grp_a, grp_b in [
            ('Mixed vs Liberal', mixed, liberal),
            ('Mixed vs Conservative', mixed, conserv),
            ('Liberal vs Conservative', liberal, conserv),
        ]:
            if len(grp_a) >= 3 and len(grp_b) >= 3:
                t_stat, p_val = stats.ttest_ind(grp_a, grp_b, equal_var=False)
                diff = grp_a.mean() - grp_b.mean()
                pooled_std = np.sqrt((grp_a.std()**2 + grp_b.std()**2) / 2)
                cohens_d = diff / pooled_std if pooled_std > 0 else np.nan
            else:
                t_stat, p_val, diff, cohens_d = np.nan, np.nan, np.nan, np.nan

            pw_rows.append({
                'WINDOW': f'[-1, +{w}]',
                'WINDOW_DAYS': w,
                'COMPARISON': comparison,
                'DIFF': diff,
                'T_STAT': t_stat,
                'P_VALUE': p_val,
                'COHENS_D': cohens_d,
            })

    pw_df = pd.DataFrame(pw_rows)
    if not pw_df.empty:
        pw_df['BH_SIGNIFICANT'] = benjamini_hochberg(pw_df['P_VALUE'].tolist(), q=0.10)

    return EnhancedContagionResult(
        tight_peer_vs_nonpeer=tight_df,
        by_industry_type=ind_type_df,
        consumer_vs_b2b_tests=cvb_df,
        lean_mechanism=lean_mech_df,
        lean_pairwise_tests=pw_df,
        panel=panel,
        n_events=n_events,
    )


def save_enhanced_contagion(
    store: DataStore,
    result: EnhancedContagionResult,
) -> dict:
    """Persist enhanced contagion results."""
    results = {}
    ts = pd.Timestamp.now().isoformat()

    results['ESSAY2_CONTAGION_TIGHT_DIFF'] = store.write_table(
        result.tight_peer_vs_nonpeer.assign(RUN_TIMESTAMP=ts),
        'ESSAY2_CONTAGION_TIGHT_DIFF', replace=True,
    )
    results['ESSAY2_CONTAGION_BY_FACING'] = store.write_table(
        result.by_industry_type.assign(RUN_TIMESTAMP=ts),
        'ESSAY2_CONTAGION_BY_FACING', replace=True,
    )
    results['ESSAY2_CONTAGION_CONS_VS_B2B'] = store.write_table(
        result.consumer_vs_b2b_tests.assign(RUN_TIMESTAMP=ts),
        'ESSAY2_CONTAGION_CONS_VS_B2B', replace=True,
    )
    results['ESSAY2_CONTAGION_LEAN_MECH'] = store.write_table(
        result.lean_mechanism.assign(RUN_TIMESTAMP=ts),
        'ESSAY2_CONTAGION_LEAN_MECH', replace=True,
    )
    results['ESSAY2_CONTAGION_LEAN_PAIRWISE'] = store.write_table(
        result.lean_pairwise_tests.assign(RUN_TIMESTAMP=ts),
        'ESSAY2_CONTAGION_LEAN_PAIRWISE', replace=True,
    )
    results['ESSAY2_ENHANCED_CONTAGION_PANEL'] = store.write_table(
        result.panel.assign(RUN_TIMESTAMP=ts),
        'ESSAY2_ENHANCED_CONTAGION_PANEL', replace=True,
    )

    saved = sum(1 for v in results.values() if v is not None)
    logger.info("Enhanced contagion: saved %d/%d tables", saved, len(results))
    return results


# =========================================================================
# PANEL CONSTRUCTION (original DiD)
# =========================================================================

def build_car_panel(
    store: DataStore,
    regime_result: RegimeResult = None,
    sentiment_analysis: SentimentRegimeAnalysis = None,
) -> Optional[pd.DataFrame]:
    """
    Build the (firm, event) panel of CARs for DiD estimation.

    Loads culture war events, computes CARs for each treatment firm and
    its matched control, merges political lean scores and FOMO z-scores.

    Returns a DataFrame with columns:
        TICKER, EVENT_ID, EVENT_DATE, IS_TREATMENT, REGIME,
        CAR_PRE, CAR_POST, CAR_FULL, LEAN, FOMO_Z,
        TREATMENT_TICKER, CONTROL_TICKER
    """
    # Load events
    events_df = store.read_table('CULTURE_WAR_COMPANIES')
    if events_df.empty:
        logger.error("No CULTURE_WAR_COMPANIES table found")
        return None

    # Load matched controls
    controls_df = store.read_table('CONTROL_COMPANIES')
    if controls_df.empty:
        logger.error("No CONTROL_COMPANIES table found")
        return None

    # Regime assignments
    if regime_result is None:
        regime_result = estimate_vix_regimes(store)
        if regime_result is None:
            return None

    regime_dates = regime_result.regime_assignments[['DATE', 'REGIME_LABEL']].copy()
    regime_dates['DATE'] = pd.to_datetime(regime_dates['DATE'])

    # Political lean scores (optional)
    lean_map = {}
    lean_df = store.read_table('POLITICAL_LEAN')
    if not lean_df.empty and 'LEAN_SCORE' in lean_df.columns:
        lean_map = dict(zip(lean_df['TICKER'], lean_df['LEAN_SCORE']))
    else:
        # Fall back to computed alignment score from essay2.py pipeline
        alignment_df = store.read_table('ESSAY2_POLITICAL_ALIGNMENT')
        if not alignment_df.empty:
            if 'ALIGNMENT_SCORE' in alignment_df.columns:
                lean_map = dict(zip(alignment_df['TICKER'], alignment_df['ALIGNMENT_SCORE']))
            elif 'COMPUTED_LEANING' in alignment_df.columns:
                lean_map = dict(zip(alignment_df['TICKER'], alignment_df['COMPUTED_LEANING']))
                logger.warning("ALIGNMENT_SCORE not found — using COMPUTED_LEANING (categorical). "
                               "Lean interaction will be skipped unless converted to numeric.")

    # FOMO z-scores (optional, from Essay 1 sentiment analysis)
    fomo_map = {}
    if sentiment_analysis is not None and not sentiment_analysis.fomo_zscores.empty:
        fz = sentiment_analysis.fomo_zscores.copy()
        fz['DATE'] = pd.to_datetime(fz['DATE'])
        # Key: (ticker, date) -> fomo_z
        for _, row in fz.iterrows():
            fomo_map[(row['TICKER'], row['DATE'])] = row['FOMO_Z']

    # Normalise event dates
    events_df.columns = [c.upper() for c in events_df.columns]
    date_col = 'EVENT_DATE' if 'EVENT_DATE' in events_df.columns else 'DATE'
    events_df[date_col] = pd.to_datetime(events_df[date_col], errors='coerce')
    events_df = events_df.dropna(subset=[date_col])

    id_col = 'EVENT_ID' if 'EVENT_ID' in events_df.columns else None

    # Build control lookup: treatment_ticker -> control_ticker
    control_map = dict(zip(
        controls_df['TREATMENT_TICKER'],
        controls_df['CONTROL_TICKER'],
    ))

    rows = []
    n_events = 0
    n_skipped = 0
    _event_seq = 0

    for _, event in events_df.iterrows():
        event_date = event[date_col]
        event_id = event[id_col] if id_col else f"event_{_event_seq}"
        _event_seq += 1
        event_ticker = event.get('TICKER', None)

        if event_ticker is None:
            continue

        # Determine regime at event date
        regime_label = _lookup_regime(event_date, regime_dates)

        # Treatment firm CAR
        treat_car = compute_car(
            event_ticker, store, event_id, event_date,
            is_treatment=True, regime=regime_label,
        )

        # Control firm CAR
        ctrl_ticker = control_map.get(event_ticker)
        ctrl_car = None
        if ctrl_ticker:
            ctrl_car = compute_car(
                ctrl_ticker, store, event_id, event_date,
                is_treatment=False, regime=regime_label,
            )

        if treat_car is None:
            n_skipped += 1
            continue

        n_events += 1

        # Treatment row
        fomo_z = fomo_map.get((event_ticker, event_date), np.nan)
        rows.append({
            'TICKER': event_ticker,
            'EVENT_ID': event_id,
            'EVENT_DATE': event_date,
            'IS_TREATMENT': True,
            'REGIME': regime_label,
            'CAR_PRE': treat_car.car_pre,
            'CAR_POST': treat_car.car_post,
            'CAR_FULL': treat_car.car_full,
            'LEAN': lean_map.get(event_ticker, np.nan),
            'FOMO_Z': fomo_z,
            'TREATMENT_TICKER': event_ticker,
            'CONTROL_TICKER': ctrl_ticker or '',
            'N_EST_OBS': treat_car.n_estimation_obs,
            'EST_R2': treat_car.r_squared,
        })

        # Control row
        if ctrl_car is not None:
            ctrl_fomo_z = fomo_map.get((ctrl_ticker, event_date), np.nan)
            rows.append({
                'TICKER': ctrl_ticker,
                'EVENT_ID': event_id,
                'EVENT_DATE': event_date,
                'IS_TREATMENT': False,
                'REGIME': regime_label,
                'CAR_PRE': ctrl_car.car_pre,
                'CAR_POST': ctrl_car.car_post,
                'CAR_FULL': ctrl_car.car_full,
                'LEAN': lean_map.get(ctrl_ticker, np.nan),
                'FOMO_Z': ctrl_fomo_z,
                'TREATMENT_TICKER': event_ticker,
                'CONTROL_TICKER': ctrl_ticker,
                'N_EST_OBS': ctrl_car.n_estimation_obs,
                'EST_R2': ctrl_car.r_squared,
            })

    if not rows:
        logger.error("No CARs computed (events=%d, skipped=%d)", n_events, n_skipped)
        return None

    panel = pd.DataFrame(rows)
    logger.info("CAR panel: %d rows, %d events, %d skipped",
                len(panel), n_events, n_skipped)
    return panel


# =========================================================================
# DiD REGRESSIONS
# =========================================================================

def _run_did_regression(
    panel: pd.DataFrame,
    dep_var: str,
    regressors: List[str],
    cluster_var: str = 'EVENT_ID',
) -> Optional[sm.regression.linear_model.RegressionResultsWrapper]:
    """Run a single cross-sectional DiD regression with cluster-robust standard errors."""
    sub = panel.dropna(subset=[dep_var] + regressors)
    if len(sub) < len(regressors) + 5:
        logger.warning("Insufficient obs (%d) for DiD with %d regressors",
                       len(sub), len(regressors))
        return None

    y = sub[dep_var]
    X = sm.add_constant(sub[regressors])

    # Check for near-singular design matrix (common with many event dummies).
    # Drop collinear columns so OLS and the downstream F-test stay reliable.
    rank = np.linalg.matrix_rank(X.values)
    if rank < X.shape[1]:
        n_drop = X.shape[1] - rank
        logger.warning("Design matrix is rank-deficient: rank=%d, cols=%d — "
                       "dropping %d collinear columns.", rank, X.shape[1], n_drop)
        _, _, pivot = _qr(X.values, pivoting=True)
        keep_idx = sorted(pivot[:rank])
        dropped = [X.columns[i] for i in range(X.shape[1]) if i not in keep_idx]
        logger.warning("Dropped collinear columns: %s", dropped)
        if 'const' in dropped:
            logger.warning("Intercept ('const') was dropped — regression proceeds "
                           "without an intercept; coefficient interpretation changes.")
        X = X.iloc[:, keep_idx]

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        if cluster_var and cluster_var in sub.columns:
            fit = sm.OLS(y, X).fit(
                cov_type='cluster',
                cov_kwds={'groups': sub[cluster_var]},
            )
        else:
            fit = sm.OLS(y, X).fit(cov_type='HC1')

    return fit


def _build_coeff_table(
    fits: Dict[str, object],
) -> pd.DataFrame:
    """Build a formatted coefficient table across multiple specifications."""
    rows = []
    for spec_name, fit in fits.items():
        if fit is None:
            continue
        for var in fit.params.index:
            rows.append({
                'SPECIFICATION': spec_name,
                'VARIABLE': var,
                'COEFFICIENT': fit.params[var],
                'STD_ERROR': fit.bse[var],
                'T_STAT': fit.tvalues[var],
                'P_VALUE': fit.pvalues[var],
                'SIGNIFICANT_005': fit.pvalues[var] < 0.05,
                'SIGNIFICANT_001': fit.pvalues[var] < 0.01,
            })

    df = pd.DataFrame(rows)
    if not df.empty:
        bh_sig = benjamini_hochberg(df['P_VALUE'].tolist(), q=0.10)
        df['BH_SIGNIFICANT'] = bh_sig
    return df


def parallel_trends_test(
    store: DataStore,
    controls_df: pd.DataFrame,
    events_df: pd.DataFrame,
    regime_result: RegimeResult = None,
    pre_window: Tuple[int, int] = _PRE_EVENT_WINDOW,
) -> Optional[ParallelTrendsResult]:
    """
    Test H0: treatment and control firms trend identically in the
    pre-event window.

    Regresses daily abnormal returns on Treatment x Day interactions
    for each day in the pre-event window.  All interaction coefficients
    should be statistically indistinguishable from zero for the DiD
    to be valid.

    Model per event:
        AR_{i,d} = alpha + sum_{d in pre_window}(beta_d * Treat_i * D_d) + epsilon

    where D_d are day-of-window dummies and Treat_i = 1 for treatment firms.

    A joint F-test (all beta_d = 0) provides the overall parallel trends
    verdict.  Individual coefficients produce the "event-study plot"
    (Figure 1) showing flat pre-event trends.

    Note: day_range[0] is the baseline day — its Treat*Day coefficient
    is absorbed into TREAT. Reported interaction coefficients are relative
    to that baseline. Day dummies drop day_range[0] for collinearity,
    but all Treat*Day interactions are included since they are not
    collinear with each other.

    Parameters
    ----------
    store : DataStore
    controls_df : pd.DataFrame
        TREATMENT_TICKER, CONTROL_TICKER columns.
    events_df : pd.DataFrame
        Must contain EVENT_DATE (or DATE) and TICKER columns.
    regime_result : RegimeResult, optional
    pre_window : tuple
        (first_day, last_day) relative to event date, e.g. (-10, -1).

    Returns
    -------
    ParallelTrendsResult or None
    """
    events_df = events_df.copy()
    events_df.columns = [c.upper() for c in events_df.columns]
    date_col = 'EVENT_DATE' if 'EVENT_DATE' in events_df.columns else 'DATE'
    events_df[date_col] = pd.to_datetime(events_df[date_col], errors='coerce')
    events_df = events_df.dropna(subset=[date_col])

    control_map = dict(zip(
        controls_df['TREATMENT_TICKER'],
        controls_df['CONTROL_TICKER'],
    ))

    factors = store.ff5[['DATE'] + _FF5_ALL + ['RF']].dropna().copy()
    factors['DATE'] = pd.to_datetime(factors['DATE'], errors='coerce')
    for col in _FF5_ALL + ['RF']:
        if factors[col].abs().max() > 1.5:
            factors[col] = factors[col] / 100

    day_range = list(range(pre_window[0], pre_window[1] + 1))
    all_rows = []

    for _, event in events_df.iterrows():
        event_date = event[date_col]
        treat_ticker = event.get('TICKER', None)
        if treat_ticker is None:
            continue
        ctrl_ticker = control_map.get(treat_ticker)
        if ctrl_ticker is None:
            continue

        for ticker, is_treat in [(treat_ticker, True), (ctrl_ticker, False)]:
            estimate = _estimate_normal_returns(ticker, store, event_date)
            if estimate is None:
                continue

            merged = estimate.event_data
            pre_obs = merged[
                (merged['TD_OFFSET'] >= pre_window[0]) &
                (merged['TD_OFFSET'] <= pre_window[1])
            ].copy()

            if len(pre_obs) < 3:
                continue

            # Compute abnormal returns
            X_pre = sm.add_constant(pre_obs[_FF5_ALL])
            expected = estimate.fit.predict(X_pre)
            pre_obs['AR'] = pre_obs['EXCESS_RETURN'] - expected.values
            pre_obs['TREAT'] = int(is_treat)
            pre_obs['EVENT_DATE'] = event_date
            pre_obs['TICKER'] = ticker

            all_rows.append(pre_obs[['TD_OFFSET', 'AR', 'TREAT', 'EVENT_DATE', 'TICKER']])

    if not all_rows:
        logger.error("Parallel trends: no pre-event data collected")
        return None

    panel = pd.concat(all_rows, ignore_index=True)

    # Create day dummies and Treat x Day interactions
    interact_cols = []
    for d in day_range:
        col_name = f'TREAT_x_D{d}'
        panel[col_name] = ((panel['TREAT'] == 1) & (panel['TD_OFFSET'] == d)).astype(int)
        interact_cols.append(col_name)

    # Also include raw day dummies to absorb date fixed effects
    day_dummy_cols = []
    for d in day_range[1:]:  # drop first for identification
        col_name = f'D{d}'
        panel[col_name] = (panel['TD_OFFSET'] == d).astype(int)
        day_dummy_cols.append(col_name)

    base_regressors = ['TREAT'] + day_dummy_cols + interact_cols
    sub = panel.dropna(subset=['AR'] + base_regressors)

    if len(sub) < len(base_regressors) + 10:
        logger.warning("Parallel trends: insufficient obs (%d)", len(sub))
        return None

    # Event-date fixed effects (absorb cross-event heterogeneity)
    event_dummy_cols = []
    if sub['EVENT_DATE'].nunique() > 1:
        event_dummies = pd.get_dummies(
            sub['EVENT_DATE'].astype(str), drop_first=True, prefix='EV'
        )
        event_dummy_cols = list(event_dummies.columns)
        sub = pd.concat([sub, event_dummies], axis=1)

    regressors = base_regressors + event_dummy_cols

    y = sub['AR'].astype(float)
    X = sm.add_constant(sub[regressors].astype(float))

    # Cluster by TICKER to match DiD regressions (stacked pre/post observations
    # from the same firm are correlated; HC1 would understate uncertainty).
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        if 'TICKER' in sub.columns and sub['TICKER'].nunique() > 1:
            fit = sm.OLS(y, X).fit(
                cov_type='cluster',
                cov_kwds={'groups': sub['TICKER']},
            )
        else:
            fit = sm.OLS(y, X).fit(cov_type='HC1')

    # Extract Treat x Day coefficients
    coeff_rows = []
    for d in day_range:
        col_name = f'TREAT_x_D{d}'
        if col_name in fit.params.index:
            coeff_rows.append({
                'DAY': d,
                'COEFFICIENT': fit.params[col_name],
                'STD_ERROR': fit.bse[col_name],
                'T_STAT': fit.tvalues[col_name],
                'P_VALUE': fit.pvalues[col_name],
            })

    daily_coeff = pd.DataFrame(coeff_rows)

    # Joint F-test: all Treat x Day coefficients = 0
    # Wald test via hypothesis matrix
    r_matrix = np.zeros((len(interact_cols), len(fit.params)))
    for i, col_name in enumerate(interact_cols):
        if col_name in fit.params.index:
            j = list(fit.params.index).index(col_name)
            r_matrix[i, j] = 1.0

    # Drop all-zero rows (interaction terms absent from fit) to avoid degenerate F-test
    nonzero_mask = r_matrix.any(axis=1)
    if not nonzero_mask.all():
        n_missing = int((~nonzero_mask).sum())
        logger.warning("Parallel trends: %d/%d interact terms missing from fit — "
                       "F-test covers fewer constraints", n_missing, len(interact_cols))
        r_matrix = r_matrix[nonzero_mask]

    try:
        f_test = fit.f_test(r_matrix)
        joint_f = float(f_test.fvalue)
        joint_p = float(f_test.pvalue)
    except Exception as e:
        logger.warning("Parallel trends F-test failed: %s", e)
        joint_f = np.nan
        joint_p = np.nan

    passes = joint_p > 0.05 if not np.isnan(joint_p) else False

    logger.info("Parallel trends: F=%.2f, p=%.4f, passes=%s (%d obs, %d days)",
                joint_f, joint_p, passes, len(sub), len(day_range))

    return ParallelTrendsResult(
        daily_coefficients=daily_coeff,
        joint_f_stat=joint_f,
        joint_p_value=joint_p,
        passes=passes,
        n_days=len(day_range),
        n_observations=len(sub),
    )


def peer_parallel_trends_test(
    store: DataStore,
    pre_window: Tuple[int, int] = (-10, -1),
) -> Optional[ParallelTrendsResult]:
    """
    Pre-event parallel trends test for industry contagion validity.

    Tests H0: event firms and their industry peers trend identically
    in the pre-event window [-10, -1].

    Pools daily abnormal returns for (event_firm, peer) pairs across
    all culture war events that have same-industry peers.  Regresses:

        AR_{i,d} = alpha + sum_d(beta_d * EventFirm_i * D_d) + eps

    Joint F-test on all beta_d = 0 validates the parallel trends
    assumption for the contagion test.

    Returns
    -------
    ParallelTrendsResult or None
    """
    events_df = store.read_table('CULTURE_WAR_COMPANIES')
    if events_df.empty:
        return None

    events_df.columns = [c.upper() for c in events_df.columns]
    events_df['EVENT_DATE'] = pd.to_datetime(events_df['EVENT_DATE'], errors='coerce')
    events_df = events_df.dropna(subset=['EVENT_DATE'])

    # Build ticker -> (industry, naics) lookup
    ticker_industry = {}
    for _, row in events_df.iterrows():
        t = row.get('TICKER')
        if t:
            ticker_industry[t] = (
                row.get('INDUSTRY', 'Unknown'),
                str(row.get('NAICS_CODE', '999999')),
            )

    all_tickers = list(ticker_industry.keys())
    day_range = list(range(pre_window[0], pre_window[1] + 1))
    all_rows = []

    for _, event in events_df.iterrows():
        event_date = event['EVENT_DATE']
        event_ticker = event.get('TICKER')
        if event_ticker is None:
            continue

        event_ind, event_naics = ticker_industry.get(event_ticker, ('Unknown', '999999'))

        # Find peers
        peers = []
        for t in all_tickers:
            if t == event_ticker:
                continue
            t_ind, t_naics = ticker_industry.get(t, ('Unknown', '999999'))
            if (event_naics != '999999' and t_naics == event_naics) or \
               (event_ind != 'Unknown' and t_ind == event_ind):
                peers.append(t)

        if not peers:
            continue

        # Collect pre-event ARs for event firm (is_event=1) and peers (is_event=0)
        for ticker, is_event in [(event_ticker, True)] + [(p, False) for p in peers]:
            estimate = _estimate_normal_returns(ticker, store, event_date)
            if estimate is None:
                continue

            merged = estimate.event_data
            pre_obs = merged[
                (merged['TD_OFFSET'] >= pre_window[0]) &
                (merged['TD_OFFSET'] <= pre_window[1])
            ].copy()

            if len(pre_obs) < 3:
                continue

            X_pre = sm.add_constant(pre_obs[_FF5_ALL])
            expected = estimate.fit.predict(X_pre)
            pre_obs['AR'] = pre_obs['EXCESS_RETURN'] - expected.values
            pre_obs['EVENT_FIRM'] = int(is_event)
            pre_obs['EVENT_DATE'] = event_date

            all_rows.append(pre_obs[['TD_OFFSET', 'AR', 'EVENT_FIRM', 'EVENT_DATE']])

    if not all_rows:
        logger.error("Peer parallel trends: no pre-event data")
        return None

    panel = pd.concat(all_rows, ignore_index=True)

    # Create EventFirm x Day interactions
    interact_cols = []
    for d in day_range:
        col_name = f'EF_x_D{d}'
        panel[col_name] = ((panel['EVENT_FIRM'] == 1) & (panel['TD_OFFSET'] == d)).astype(int)
        interact_cols.append(col_name)

    # Day dummies (absorb time fixed effects)
    day_dummy_cols = []
    for d in day_range[1:]:
        col_name = f'D{d}'
        panel[col_name] = (panel['TD_OFFSET'] == d).astype(int)
        day_dummy_cols.append(col_name)

    base_regressors = ['EVENT_FIRM'] + day_dummy_cols + interact_cols
    sub = panel.dropna(subset=['AR'] + base_regressors)

    if len(sub) < len(base_regressors) + 10:
        logger.warning("Peer parallel trends: insufficient obs (%d)", len(sub))
        return None

    # Event-date fixed effects (absorb cross-event heterogeneity)
    event_dummy_cols = []
    if sub['EVENT_DATE'].nunique() > 1:
        event_dummies = pd.get_dummies(
            sub['EVENT_DATE'].astype(str), drop_first=True, prefix='EV'
        )
        event_dummy_cols = list(event_dummies.columns)
        sub = pd.concat([sub, event_dummies], axis=1)

    regressors = base_regressors + event_dummy_cols

    y = sub['AR'].astype(float)
    X = sm.add_constant(sub[regressors].astype(float))

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        if sub['EVENT_DATE'].nunique() > 1:
            fit = sm.OLS(y, X).fit(
                cov_type='cluster',
                cov_kwds={'groups': sub['EVENT_DATE']},
            )
        else:
            fit = sm.OLS(y, X).fit(cov_type='HC1')

    # Extract EventFirm x Day coefficients
    coeff_rows = []
    for d in day_range:
        col_name = f'EF_x_D{d}'
        if col_name in fit.params.index:
            coeff_rows.append({
                'DAY': d,
                'COEFFICIENT': fit.params[col_name],
                'STD_ERROR': fit.bse[col_name],
                'T_STAT': fit.tvalues[col_name],
                'P_VALUE': fit.pvalues[col_name],
            })

    daily_coeff = pd.DataFrame(coeff_rows)

    # Joint F-test
    r_matrix = np.zeros((len(interact_cols), len(fit.params)))
    for i, col_name in enumerate(interact_cols):
        if col_name in fit.params.index:
            j = list(fit.params.index).index(col_name)
            r_matrix[i, j] = 1.0

    # Drop all-zero rows (interaction terms absent from fit) to avoid degenerate F-test
    nonzero_mask = r_matrix.any(axis=1)
    if not nonzero_mask.all():
        n_missing = int((~nonzero_mask).sum())
        logger.warning("Peer parallel trends: %d/%d interact terms missing from fit — "
                       "F-test covers fewer constraints", n_missing, len(interact_cols))
        r_matrix = r_matrix[nonzero_mask]

    try:
        f_test = fit.f_test(r_matrix)
        joint_f = float(f_test.fvalue)
        joint_p = float(f_test.pvalue)
    except Exception as e:
        logger.warning("Peer parallel trends F-test failed: %s", e)
        joint_f = np.nan
        joint_p = np.nan

    passes = joint_p > 0.05 if not np.isnan(joint_p) else False

    logger.info("Peer parallel trends: F=%.2f, p=%.4f, passes=%s (%d obs, %d days)",
                joint_f, joint_p, passes, len(sub), len(day_range))

    return ParallelTrendsResult(
        daily_coefficients=daily_coeff,
        joint_f_stat=joint_f,
        joint_p_value=joint_p,
        passes=passes,
        n_days=len(day_range),
        n_observations=len(sub),
    )


def run_did(
    store: DataStore,
    regime_result: RegimeResult = None,
    sentiment_analysis: SentimentRegimeAnalysis = None,
    car_panel: pd.DataFrame = None,
) -> Optional[DiDResult]:
    """
    Run the Essay 2 cross-sectional DiD.

    Three specifications, each nesting the previous:
    1. Basic:     CAR ~ Treat + Post + Treat*Post
    2. With lean: + Lean + Treat*Post*Lean
    3. With FOMO: + FOMO_Z

    Parameters
    ----------
    store : DataStore
    regime_result : RegimeResult, optional
    sentiment_analysis : SentimentRegimeAnalysis, optional
        For FOMO z-scores.
    car_panel : pd.DataFrame, optional
        Pre-computed panel. If None, builds it from scratch.

    Returns
    -------
    DiDResult or None
    """
    if car_panel is None:
        car_panel = build_car_panel(
            store,
            regime_result=regime_result,
            sentiment_analysis=sentiment_analysis,
        )
        if car_panel is None:
            return None

    # Parallel trends pre-test (required for DiD validity)
    controls_df = store.read_table('CONTROL_COMPANIES')
    events_df = store.read_table('CULTURE_WAR_COMPANIES')
    pt_result = None
    if not controls_df.empty and not events_df.empty:
        pt_result = parallel_trends_test(
            store, controls_df, events_df, regime_result=regime_result)
        if pt_result is not None and not pt_result.passes:
            logger.warning(
                "PARALLEL TRENDS VIOLATED: F=%.2f, p=%.4f — "
                "DiD results should be interpreted with caution",
                pt_result.joint_f_stat, pt_result.joint_p_value)

    # Construct DiD variables
    panel = car_panel.copy()
    panel['TREAT'] = panel['IS_TREATMENT'].astype(int)

    # For cross-sectional DiD on post-event CARs, the "Post" dimension is
    # embedded in CAR_POST vs CAR_PRE. We stack them into long format.
    pre = panel[['TICKER', 'EVENT_ID', 'TREAT', 'REGIME', 'LEAN', 'FOMO_Z',
                 'CAR_PRE']].copy()
    pre['POST'] = 0
    pre = pre.rename(columns={'CAR_PRE': 'CAR'})

    post = panel[['TICKER', 'EVENT_ID', 'TREAT', 'REGIME', 'LEAN', 'FOMO_Z',
                  'CAR_POST']].copy()
    post['POST'] = 1
    post = post.rename(columns={'CAR_POST': 'CAR'})

    stacked = pd.concat([pre, post], ignore_index=True)
    stacked['TREAT_x_POST'] = stacked['TREAT'] * stacked['POST']

    # All specs use firm-clustered SEs because stacked pre/post observations
    # from the same firm are mechanically correlated.  HC1 would understate
    # uncertainty.  See Lambert review note on cross-sectional design.
    _cluster = 'TICKER'

    # Spec 1: Basic DiD
    did_basic = _run_did_regression(
        stacked, 'CAR', ['TREAT', 'POST', 'TREAT_x_POST'],
        cluster_var=_cluster,
    )

    # Spec 2: With political lean interaction
    did_with_lean = None
    if 'LEAN' in stacked.columns and stacked['LEAN'].notna().sum() > 10:
        # LEAN must be numeric for interaction terms. If categorical strings
        # (e.g. 'Liberal', 'Conservative') were loaded, skip the lean spec.
        if not pd.api.types.is_numeric_dtype(stacked['LEAN']):
            logger.warning("LEAN column is categorical (%s) — skipping lean interaction. "
                           "Use ALIGNMENT_SCORE (numeric) for DiD lean interactions.",
                           stacked['LEAN'].dtype)
        else:
            stacked['TREAT_x_POST_x_LEAN'] = (
                stacked['TREAT_x_POST'] * stacked['LEAN']
            )
            did_with_lean = _run_did_regression(
                stacked, 'CAR',
                ['TREAT', 'POST', 'TREAT_x_POST', 'LEAN', 'TREAT_x_POST_x_LEAN'],
                cluster_var=_cluster,
            )

    # Spec 3: Full model with FOMO z-score
    did_with_fomo = None
    if 'FOMO_Z' in stacked.columns and stacked['FOMO_Z'].notna().sum() > 10:
        fomo_regressors = ['TREAT', 'POST', 'TREAT_x_POST', 'FOMO_Z']
        if did_with_lean is not None:
            fomo_regressors = [
                'TREAT', 'POST', 'TREAT_x_POST', 'LEAN',
                'TREAT_x_POST_x_LEAN', 'FOMO_Z',
            ]
        did_with_fomo = _run_did_regression(
            stacked, 'CAR', fomo_regressors, cluster_var=_cluster,
        )

    # Coefficient table
    fits = {
        'basic': did_basic,
        'with_lean': did_with_lean,
        'with_fomo': did_with_fomo,
    }
    coeff_table = _build_coeff_table(fits)

    n_treatment = panel[panel['IS_TREATMENT']]['TICKER'].nunique()
    n_control = panel[~panel['IS_TREATMENT']]['TICKER'].nunique()
    n_events = panel['EVENT_ID'].nunique()

    logger.info("DiD complete: %d events, %d treatment firms, %d control firms, "
                "%d stacked obs", n_events, n_treatment, n_control, len(stacked))

    return DiDResult(
        car_panel=panel,
        did_basic=did_basic,
        did_with_lean=did_with_lean,
        did_with_fomo=did_with_fomo,
        parallel_trends=pt_result,
        n_events=n_events,
        n_treatment_firms=n_treatment,
        n_control_firms=n_control,
        n_observations=len(stacked),
        coefficient_table=coeff_table,
    )


# =========================================================================
# DIAGNOSTIC TESTS
# =========================================================================

def run_placebo_test(
    car_panel: pd.DataFrame,
    n_iterations: int = 500,
    seed: int = 42,
) -> pd.DataFrame:
    """
    Placebo test: randomly permute treatment assignment and re-run DiD.

    If the true treatment effect is real, the actual TREAT_x_POST coefficient
    should fall in the tails of the placebo distribution.

    Parameters
    ----------
    car_panel : pd.DataFrame
        Wide CAR panel with TICKER, EVENT_DATE, IS_TREATMENT, CAR_PRE, CAR_POST, etc.
    n_iterations : int
        Number of placebo permutations.
    seed : int

    Returns
    -------
    pd.DataFrame
        Columns: ITERATION, PLACEBO_COEFF, PLACEBO_T, PLACEBO_P,
        plus a summary row with ACTUAL_COEFF, ACTUAL_T, PERCENTILE_RANK.
    """
    try:
        rng = np.random.RandomState(seed)
        panel = car_panel.copy()
        panel['TREAT'] = panel['IS_TREATMENT'].astype(int)

        # Stack into pre/post for actual regression
        def _stack_panel(p):
            pre = p[['TICKER', 'EVENT_ID', 'TREAT', 'CAR_PRE']].copy()
            pre['POST'] = 0
            pre = pre.rename(columns={'CAR_PRE': 'CAR'})
            post = p[['TICKER', 'EVENT_ID', 'TREAT', 'CAR_POST']].copy()
            post['POST'] = 1
            post = post.rename(columns={'CAR_POST': 'CAR'})
            stacked = pd.concat([pre, post], ignore_index=True)
            stacked['TREAT_x_POST'] = stacked['TREAT'] * stacked['POST']
            return stacked

        # Actual regression — use firm-clustered SEs (consistent with run_did)
        stacked_actual = _stack_panel(panel)
        sub = stacked_actual.dropna(subset=['CAR'])
        y = sub['CAR']
        X = sm.add_constant(sub[['TREAT', 'POST', 'TREAT_x_POST']])
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            actual_fit = sm.OLS(y, X).fit(
                cov_type='cluster',
                cov_kwds={'groups': sub['TICKER']},
            )

        actual_coeff = actual_fit.params.get('TREAT_x_POST', np.nan)
        actual_t = actual_fit.tvalues.get('TREAT_x_POST', np.nan)

        # Placebo iterations — permute at firm level (treatment is a firm characteristic)
        # Verify panel is wide-format (one row per firm per event) before permutation
        _expected_firm_rows = panel.groupby(['TICKER', 'EVENT_ID']).size()
        if not (_expected_firm_rows == 1).all():
            n_dups = int((_expected_firm_rows > 1).sum())
            logger.error(
                "Placebo permutation expects wide-format panel (one row per "
                "firm×event), but found %d duplicates — aborting", n_dups)
            return pd.DataFrame()
        rows = []
        for i in range(n_iterations):
            p_shuffled = panel.copy()
            firm_treat = p_shuffled.drop_duplicates('TICKER')[['TICKER', 'TREAT']].copy()
            firm_treat['TREAT'] = rng.permutation(firm_treat['TREAT'].values)
            p_shuffled = p_shuffled.drop(columns=['TREAT']).merge(
                firm_treat, on='TICKER', how='left'
            )
            stacked = _stack_panel(p_shuffled)
            sub = stacked.dropna(subset=['CAR'])
            y = sub['CAR']
            X = sm.add_constant(sub[['TREAT', 'POST', 'TREAT_x_POST']])
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                fit = sm.OLS(y, X).fit(
                    cov_type='cluster',
                    cov_kwds={'groups': sub['TICKER']},
                )
            rows.append({
                'ITERATION': i + 1,
                'PLACEBO_COEFF': fit.params.get('TREAT_x_POST', np.nan),
                'PLACEBO_T': fit.tvalues.get('TREAT_x_POST', np.nan),
                'PLACEBO_P': fit.pvalues.get('TREAT_x_POST', np.nan),
            })

        result = pd.DataFrame(rows)

        # Percentile rank: fraction of placebo coefficients more extreme than actual
        placebo_coeffs = result['PLACEBO_COEFF'].dropna().values
        if len(placebo_coeffs) > 0:
            percentile_rank = np.mean(np.abs(placebo_coeffs) >= np.abs(actual_coeff))
        else:
            percentile_rank = np.nan

        # Add summary row with explicit column names (no blind rename)
        result['ROW_TYPE'] = 'PLACEBO'
        summary = pd.DataFrame([{
            'ITERATION': -1,
            'PLACEBO_COEFF': actual_coeff,
            'PLACEBO_T': actual_t,
            'PLACEBO_P': percentile_rank,
            'ROW_TYPE': 'ACTUAL_SUMMARY',
        }])
        result = pd.concat([result, summary], ignore_index=True)

        logger.info("Placebo test: %d iterations, actual coeff=%.6f, percentile_rank=%.4f",
                     n_iterations, actual_coeff, percentile_rank)
        return result

    except Exception as e:
        logger.warning("Placebo test failed: %s", e)
        return pd.DataFrame()


def run_bootstrap_car_ci(
    car_panel: pd.DataFrame,
    n_bootstrap: int = 1000,
    ci_level: float = 0.95,
    seed: int = 42,
) -> pd.DataFrame:
    """
    Block bootstrap confidence intervals for treatment vs control CAR difference.

    Resamples (TICKER, EVENT_DATE) pairs with replacement and computes
    mean CAR_POST for treatment and control in each resample.

    Returns
    -------
    pd.DataFrame
        Columns: STATISTIC, OBSERVED, BOOTSTRAP_MEAN, SE_BOOTSTRAP, CI_LOWER, CI_UPPER
    """
    try:
        rng = np.random.RandomState(seed)
        panel = car_panel.dropna(subset=['CAR_POST']).copy()

        # Observed values
        treat = panel[panel['IS_TREATMENT']]
        ctrl = panel[~panel['IS_TREATMENT']]
        obs_treat_mean = treat['CAR_POST'].mean()
        obs_ctrl_mean = ctrl['CAR_POST'].mean()
        obs_diff = obs_treat_mean - obs_ctrl_mean

        # Block bootstrap at (TICKER, EVENT_DATE) level
        firm_events = panel[['TICKER', 'EVENT_DATE']].drop_duplicates()
        n_fe = len(firm_events)

        boot_treat_means = []
        boot_ctrl_means = []
        boot_diffs = []

        for _ in range(n_bootstrap):
            idx = rng.choice(n_fe, size=n_fe, replace=True)
            sampled_fe = firm_events.iloc[idx]
            boot_panel = sampled_fe.merge(panel, on=['TICKER', 'EVENT_DATE'], how='left')
            b_treat = boot_panel[boot_panel['IS_TREATMENT']]['CAR_POST']
            b_ctrl = boot_panel[~boot_panel['IS_TREATMENT']]['CAR_POST']
            t_mean = b_treat.mean() if len(b_treat) > 0 else np.nan
            c_mean = b_ctrl.mean() if len(b_ctrl) > 0 else np.nan
            boot_treat_means.append(t_mean)
            boot_ctrl_means.append(c_mean)
            boot_diffs.append(t_mean - c_mean if not (np.isnan(t_mean) or np.isnan(c_mean)) else np.nan)

        alpha = 1 - ci_level
        results = []
        for stat_name, observed, boot_vals in [
            ('TREAT_MEAN', obs_treat_mean, boot_treat_means),
            ('CTRL_MEAN', obs_ctrl_mean, boot_ctrl_means),
            ('DIFF', obs_diff, boot_diffs),
        ]:
            arr = np.array([v for v in boot_vals if not np.isnan(v)])
            if len(arr) > 0:
                results.append({
                    'STATISTIC': stat_name,
                    'OBSERVED': observed,
                    'BOOTSTRAP_MEAN': np.mean(arr),
                    'SE_BOOTSTRAP': np.std(arr, ddof=1),
                    'CI_LOWER': np.percentile(arr, 100 * alpha / 2),
                    'CI_UPPER': np.percentile(arr, 100 * (1 - alpha / 2)),
                })

        logger.info("Bootstrap CI: %d resamples, observed diff=%.6f", n_bootstrap, obs_diff)
        return pd.DataFrame(results)

    except Exception as e:
        logger.warning("Bootstrap CI failed: %s", e)
        return pd.DataFrame()


def run_cluster_robust_did(
    car_panel: pd.DataFrame,
    fits: Dict[str, object],
) -> pd.DataFrame:
    """
    Compare cluster-by-TICKER SEs (primary) with cluster-by-EVENT_ID SEs.

    The main DiD regressions cluster by TICKER (firm-level correlation).
    This diagnostic re-runs with cluster-by-EVENT_ID to assess how much
    the clustering dimension matters for inference.

    Parameters
    ----------
    car_panel : pd.DataFrame
        Wide CAR panel.
    fits : dict
        {'basic': fit, 'with_lean': fit, 'with_fomo': fit}
        These fits already use cluster-by-TICKER SEs.

    Returns
    -------
    pd.DataFrame
        Columns: SPECIFICATION, VARIABLE, COEFF, SE_CLUSTER_FIRM,
                 SE_CLUSTER_EVENT, T_CLUSTER_FIRM, T_CLUSTER_EVENT
    """
    try:
        panel = car_panel.copy()
        panel['TREAT'] = panel['IS_TREATMENT'].astype(int)

        # Stack into pre/post
        pre = panel[['TICKER', 'EVENT_ID', 'TREAT', 'REGIME', 'LEAN', 'FOMO_Z',
                      'CAR_PRE']].copy()
        pre['POST'] = 0
        pre = pre.rename(columns={'CAR_PRE': 'CAR'})
        post = panel[['TICKER', 'EVENT_ID', 'TREAT', 'REGIME', 'LEAN', 'FOMO_Z',
                       'CAR_POST']].copy()
        post['POST'] = 1
        post = post.rename(columns={'CAR_POST': 'CAR'})
        stacked = pd.concat([pre, post], ignore_index=True)
        stacked['TREAT_x_POST'] = stacked['TREAT'] * stacked['POST']
        if ('LEAN' in stacked.columns
                and pd.api.types.is_numeric_dtype(stacked['LEAN'])
                and stacked['LEAN'].notna().sum() > 10):
            stacked['TREAT_x_POST_x_LEAN'] = stacked['TREAT_x_POST'] * stacked['LEAN']

        # Spec -> regressors (only include LEAN terms if numeric)
        _lean_avail = ('LEAN' in stacked.columns
                       and pd.api.types.is_numeric_dtype(stacked['LEAN'])
                       and 'TREAT_x_POST_x_LEAN' in stacked.columns)
        if _lean_avail:
            spec_regressors = {
                'basic': ['TREAT', 'POST', 'TREAT_x_POST'],
                'with_lean': ['TREAT', 'POST', 'TREAT_x_POST', 'LEAN', 'TREAT_x_POST_x_LEAN'],
                'with_fomo': ['TREAT', 'POST', 'TREAT_x_POST', 'LEAN', 'TREAT_x_POST_x_LEAN', 'FOMO_Z'],
            }
        else:
            spec_regressors = {
                'basic': ['TREAT', 'POST', 'TREAT_x_POST'],
                'with_lean': ['TREAT', 'POST', 'TREAT_x_POST'],
                'with_fomo': ['TREAT', 'POST', 'TREAT_x_POST', 'FOMO_Z'],
            }

        rows = []
        for spec_name, firm_fit in fits.items():
            if firm_fit is None:
                continue
            regressors = spec_regressors.get(spec_name, [])
            available = [r for r in regressors if r in stacked.columns]
            sub = stacked.dropna(subset=['CAR'] + available)
            if len(sub) < len(available) + 5:
                continue

            y = sub['CAR']
            X = sm.add_constant(sub[available])

            # Re-run with cluster-by-EVENT_ID for comparison
            # (the passed-in fits already cluster by TICKER)
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                if 'EVENT_ID' in sub.columns and sub['EVENT_ID'].nunique() > 1:
                    event_cluster_fit = sm.OLS(y, X).fit(
                        cov_type='cluster',
                        cov_kwds={'groups': sub['EVENT_ID']},
                    )
                else:
                    event_cluster_fit = sm.OLS(y, X).fit(cov_type='HC1')

            for var in firm_fit.params.index:
                if var in event_cluster_fit.params.index:
                    rows.append({
                        'SPECIFICATION': spec_name,
                        'VARIABLE': var,
                        'COEFF': firm_fit.params[var],
                        'SE_CLUSTER_FIRM': firm_fit.bse[var],
                        'SE_CLUSTER_EVENT': event_cluster_fit.bse[var],
                        'T_CLUSTER_FIRM': firm_fit.tvalues[var],
                        'T_CLUSTER_EVENT': event_cluster_fit.tvalues[var],
                    })

        logger.info("Cluster-robust SEs: %d coefficient rows", len(rows))
        return pd.DataFrame(rows)

    except Exception as e:
        logger.warning("Cluster-robust SEs failed: %s", e)
        return pd.DataFrame()


def run_normality_tests(
    fits: Dict[str, object],
) -> pd.DataFrame:
    """
    Jarque-Bera and Shapiro-Wilk tests on DiD residuals.

    Returns
    -------
    pd.DataFrame
        Columns: SPECIFICATION, JB_STAT, JB_P, SHAPIRO_STAT, SHAPIRO_P,
                 SKEWNESS, KURTOSIS, N_RESID
    """
    try:
        rows = []
        for spec_name, fit in fits.items():
            if fit is None:
                continue
            resid = fit.resid.values
            n = len(resid)

            jb_stat, jb_p = stats.jarque_bera(resid)

            # Shapiro-Wilk limited to 5000 observations
            shapiro_resid = resid[:5000] if n > 5000 else resid
            shapiro_stat, shapiro_p = stats.shapiro(shapiro_resid)

            rows.append({
                'SPECIFICATION': spec_name,
                'JB_STAT': jb_stat,
                'JB_P': jb_p,
                'SHAPIRO_STAT': shapiro_stat,
                'SHAPIRO_P': shapiro_p,
                'SKEWNESS': stats.skew(resid),
                'KURTOSIS': stats.kurtosis(resid),
                'N_RESID': n,
            })

        logger.info("Normality tests: %d specifications", len(rows))
        return pd.DataFrame(rows)

    except Exception as e:
        logger.warning("Normality tests failed: %s", e)
        return pd.DataFrame()


def run_heteroskedasticity_tests(
    fits: Dict[str, object],
) -> pd.DataFrame:
    """
    Breusch-Pagan and White tests for heteroskedasticity.

    Returns
    -------
    pd.DataFrame
        Columns: SPECIFICATION, BP_LM_STAT, BP_LM_P, BP_F_STAT, BP_F_P,
                 WHITE_LM_STAT, WHITE_LM_P
    """
    try:
        from statsmodels.stats.diagnostic import het_breuschpagan, het_white

        rows = []
        for spec_name, fit in fits.items():
            if fit is None:
                continue

            resid = fit.resid.values
            exog = fit.model.exog

            # Breusch-Pagan
            bp_lm, bp_lm_p, bp_f, bp_f_p = het_breuschpagan(resid, exog)

            # White test (can fail with too many cross-terms)
            white_lm, white_lm_p = np.nan, np.nan
            try:
                white_result = het_white(resid, exog)
                white_lm = white_result[0]
                white_lm_p = white_result[1]
            except Exception as e_white:
                logger.debug("White test failed for %s: %s", spec_name, e_white)

            rows.append({
                'SPECIFICATION': spec_name,
                'BP_LM_STAT': bp_lm,
                'BP_LM_P': bp_lm_p,
                'BP_F_STAT': bp_f,
                'BP_F_P': bp_f_p,
                'WHITE_LM_STAT': white_lm,
                'WHITE_LM_P': white_lm_p,
            })

        logger.info("Heteroskedasticity tests: %d specifications", len(rows))
        return pd.DataFrame(rows)

    except Exception as e:
        logger.warning("Heteroskedasticity tests failed: %s", e)
        return pd.DataFrame()


def compute_vif(
    car_panel: pd.DataFrame,
    regressor_sets: Dict[str, List[str]],
) -> pd.DataFrame:
    """
    Variance Inflation Factors for each DiD specification.

    Parameters
    ----------
    car_panel : pd.DataFrame
        Wide CAR panel.
    regressor_sets : dict
        Mapping spec name to list of regressor column names.

    Returns
    -------
    pd.DataFrame
        Columns: SPECIFICATION, VARIABLE, VIF
    """
    try:
        from statsmodels.stats.outliers_influence import variance_inflation_factor

        panel = car_panel.copy()
        panel['TREAT'] = panel['IS_TREATMENT'].astype(int)

        # Stack into pre/post
        pre = panel[['TICKER', 'EVENT_ID', 'TREAT', 'REGIME', 'LEAN', 'FOMO_Z',
                      'CAR_PRE']].copy()
        pre['POST'] = 0
        pre = pre.rename(columns={'CAR_PRE': 'CAR'})
        post = panel[['TICKER', 'EVENT_ID', 'TREAT', 'REGIME', 'LEAN', 'FOMO_Z',
                       'CAR_POST']].copy()
        post['POST'] = 1
        post = post.rename(columns={'CAR_POST': 'CAR'})
        stacked = pd.concat([pre, post], ignore_index=True)
        stacked['TREAT_x_POST'] = stacked['TREAT'] * stacked['POST']
        if ('LEAN' in stacked.columns
                and pd.api.types.is_numeric_dtype(stacked['LEAN'])
                and stacked['LEAN'].notna().sum() > 10):
            stacked['TREAT_x_POST_x_LEAN'] = stacked['TREAT_x_POST'] * stacked['LEAN']

        rows = []
        for spec_name, regressors in regressor_sets.items():
            available = [r for r in regressors if r in stacked.columns]
            sub = stacked.dropna(subset=['CAR'] + available)
            if len(sub) < len(available) + 5 or len(available) == 0:
                continue

            X = sm.add_constant(sub[available])
            X_arr = X.values

            for i, col_name in enumerate(X.columns):
                try:
                    vif_val = variance_inflation_factor(X_arr, i)
                    rows.append({
                        'SPECIFICATION': spec_name,
                        'VARIABLE': col_name,
                        'VIF': vif_val,
                    })
                except Exception:
                    rows.append({
                        'SPECIFICATION': spec_name,
                        'VARIABLE': col_name,
                        'VIF': np.nan,
                    })

        logger.info("VIF: %d variable entries", len(rows))
        return pd.DataFrame(rows)

    except Exception as e:
        logger.warning("VIF computation failed: %s", e)
        return pd.DataFrame()


def run_covariate_balance(
    car_panel: pd.DataFrame,
) -> pd.DataFrame:
    """
    Covariate balance check between treatment and control groups.

    Computes standardized mean differences (SMD) and Welch t-tests
    for numeric covariates. |SMD| < 0.1 indicates good balance.

    Returns
    -------
    pd.DataFrame
        Columns: VARIABLE, MEAN_TREAT, MEAN_CTRL, STD_TREAT, STD_CTRL,
                 SMD, P_VALUE, BALANCED
    """
    try:
        panel = car_panel.copy()
        treat = panel[panel['IS_TREATMENT']]
        ctrl = panel[~panel['IS_TREATMENT']]

        # Candidate covariates
        candidates = ['CAR_PRE', 'N_EST_OBS', 'EST_R2', 'FOMO_Z', 'LEAN']
        # Also check for LEAN_SCORE if present
        if 'LEAN_SCORE' in panel.columns:
            candidates.append('LEAN_SCORE')

        rows = []
        for var in candidates:
            if var not in panel.columns:
                continue
            t_vals = pd.to_numeric(treat[var], errors='coerce').dropna()
            c_vals = pd.to_numeric(ctrl[var], errors='coerce').dropna()
            if len(t_vals) < 3 or len(c_vals) < 3:
                continue

            mean_t = t_vals.mean()
            mean_c = c_vals.mean()
            std_t = t_vals.std()
            std_c = c_vals.std()

            # Standardized mean difference
            pooled_std = np.sqrt((std_t**2 + std_c**2) / 2)
            smd = (mean_t - mean_c) / pooled_std if pooled_std > 0 else np.nan

            # Welch t-test
            _, p_val = stats.ttest_ind(t_vals, c_vals, equal_var=False)

            rows.append({
                'VARIABLE': var,
                'MEAN_TREAT': mean_t,
                'MEAN_CTRL': mean_c,
                'STD_TREAT': std_t,
                'STD_CTRL': std_c,
                'SMD': smd,
                'P_VALUE': p_val,
                'BALANCED': abs(smd) < 0.1 if not np.isnan(smd) else False,
            })

        logger.info("Covariate balance: %d variables checked", len(rows))
        return pd.DataFrame(rows)

    except Exception as e:
        logger.warning("Covariate balance failed: %s", e)
        return pd.DataFrame()


def run_autocorrelation_test(
    fits: Dict[str, object],
) -> pd.DataFrame:
    """
    Durbin-Watson test for autocorrelation in DiD residuals.

    DW ~ 2 indicates no autocorrelation.
    DW < 1.5 suggests positive autocorrelation.
    DW > 2.5 suggests negative autocorrelation.

    Returns
    -------
    pd.DataFrame
        Columns: SPECIFICATION, DW_STAT, INTERPRETATION
    """
    try:
        from statsmodels.stats.stattools import durbin_watson

        rows = []
        for spec_name, fit in fits.items():
            if fit is None:
                continue
            dw = durbin_watson(fit.resid)
            # NOTE: Fixed thresholds (1.5/2.5) are conventional approximations.
            # Exact critical values depend on n and k (Durbin-Watson tables),
            # but the cross-sectional DiD residuals are large-sample (n >> k),
            # so the approximation is adequate for a diagnostic flag.
            if dw < 1.5:
                interp = 'Positive autocorrelation'
            elif dw > 2.5:
                interp = 'Negative autocorrelation'
            else:
                interp = 'No autocorrelation'

            rows.append({
                'SPECIFICATION': spec_name,
                'DW_STAT': dw,
                'INTERPRETATION': interp,
            })

        logger.info("Autocorrelation test: %d specifications", len(rows))
        return pd.DataFrame(rows)

    except Exception as e:
        logger.warning("Autocorrelation test failed: %s", e)
        return pd.DataFrame()


def run_diagnostics(
    store: DataStore,
    did_result: DiDResult,
    n_placebo: int = 500,
    n_bootstrap: int = 1000,
    seed: int = 42,
) -> DiagnosticResults:
    """
    Orchestrate all 8 diagnostic tests for DiD regressions.

    Parameters
    ----------
    store : DataStore
    did_result : DiDResult
        Contains car_panel and fit objects.
    n_placebo : int
    n_bootstrap : int
    seed : int

    Returns
    -------
    DiagnosticResults
    """
    car_panel = did_result.car_panel

    # Collect fit objects (stored directly on DiDResult)
    fits = {
        'basic': did_result.did_basic,
        'with_lean': did_result.did_with_lean,
        'with_fomo': did_result.did_with_fomo,
    }

    # If fit objects are None or not actual fit objects, re-run regressions
    needs_rerun = all(f is None for f in fits.values())
    if needs_rerun:
        logger.info("Diagnostics: re-running DiD regressions to obtain fit objects")
        panel = car_panel.copy()
        panel['TREAT'] = panel['IS_TREATMENT'].astype(int)
        pre = panel[['TICKER', 'EVENT_ID', 'TREAT', 'REGIME', 'LEAN', 'FOMO_Z',
                      'CAR_PRE']].copy()
        pre['POST'] = 0
        pre = pre.rename(columns={'CAR_PRE': 'CAR'})
        post = panel[['TICKER', 'EVENT_ID', 'TREAT', 'REGIME', 'LEAN', 'FOMO_Z',
                       'CAR_POST']].copy()
        post['POST'] = 1
        post = post.rename(columns={'CAR_POST': 'CAR'})
        stacked = pd.concat([pre, post], ignore_index=True)
        stacked['TREAT_x_POST'] = stacked['TREAT'] * stacked['POST']
        _lean_numeric = ('LEAN' in stacked.columns
                         and stacked['LEAN'].notna().sum() > 10
                         and pd.api.types.is_numeric_dtype(stacked['LEAN']))
        if _lean_numeric:
            stacked['TREAT_x_POST_x_LEAN'] = stacked['TREAT_x_POST'] * stacked['LEAN']

        # Cluster by TICKER — consistent with run_did()
        _cluster = 'TICKER'
        fits['basic'] = _run_did_regression(
            stacked, 'CAR', ['TREAT', 'POST', 'TREAT_x_POST'],
            cluster_var=_cluster)
        if _lean_numeric:
            fits['with_lean'] = _run_did_regression(
                stacked, 'CAR',
                ['TREAT', 'POST', 'TREAT_x_POST', 'LEAN', 'TREAT_x_POST_x_LEAN'],
                cluster_var=_cluster)
        if 'FOMO_Z' in stacked.columns and stacked['FOMO_Z'].notna().sum() > 10:
            fomo_regs = ['TREAT', 'POST', 'TREAT_x_POST', 'FOMO_Z']
            if _lean_numeric:
                fomo_regs = ['TREAT', 'POST', 'TREAT_x_POST', 'LEAN',
                             'TREAT_x_POST_x_LEAN', 'FOMO_Z']
            fits['with_fomo'] = _run_did_regression(
                stacked, 'CAR', fomo_regs, cluster_var=_cluster)

    # Regressor sets for VIF
    regressor_sets = {
        'basic': ['TREAT', 'POST', 'TREAT_x_POST'],
        'with_lean': ['TREAT', 'POST', 'TREAT_x_POST', 'LEAN', 'TREAT_x_POST_x_LEAN'],
        'with_fomo': ['TREAT', 'POST', 'TREAT_x_POST', 'LEAN', 'TREAT_x_POST_x_LEAN', 'FOMO_Z'],
    }

    logger.info("Running 8 diagnostic tests...")

    placebo = run_placebo_test(car_panel, n_iterations=n_placebo, seed=seed)
    bootstrap = run_bootstrap_car_ci(car_panel, n_bootstrap=n_bootstrap, seed=seed)
    cluster = run_cluster_robust_did(car_panel, fits)
    normality = run_normality_tests(fits)
    hetero = run_heteroskedasticity_tests(fits)
    vif = compute_vif(car_panel, regressor_sets)
    balance = run_covariate_balance(car_panel)
    autocorr = run_autocorrelation_test(fits)

    logger.info("Diagnostics complete: placebo=%d, bootstrap=%d, cluster=%d, "
                "normality=%d, hetero=%d, vif=%d, balance=%d, autocorr=%d",
                len(placebo), len(bootstrap), len(cluster),
                len(normality), len(hetero), len(vif), len(balance), len(autocorr))

    return DiagnosticResults(
        placebo_tests=placebo,
        bootstrap_ci=bootstrap,
        cluster_robust=cluster,
        normality=normality,
        heteroskedasticity=hetero,
        vif=vif,
        covariate_balance=balance,
        autocorrelation=autocorr,
        n_placebo_iterations=n_placebo,
        n_bootstrap_iterations=n_bootstrap,
    )


# =========================================================================
# PERSISTENCE
# =========================================================================

def save_did_results(
    store: DataStore,
    result: DiDResult,
) -> dict:
    """Persist Essay 2 DiD results to the database."""
    results = {}
    timestamp = pd.Timestamp.now().isoformat()

    results['ESSAY2_CAR_PANEL'] = store.write_table(
        result.car_panel.assign(RUN_TIMESTAMP=timestamp),
        'ESSAY2_CAR_PANEL', replace=True,
    )

    if not result.coefficient_table.empty:
        results['ESSAY2_DID_COEFFICIENTS'] = store.write_table(
            result.coefficient_table.assign(RUN_TIMESTAMP=timestamp),
            'ESSAY2_DID_COEFFICIENTS', replace=True,
        )

    if result.parallel_trends is not None and not result.parallel_trends.daily_coefficients.empty:
        pt_df = result.parallel_trends.daily_coefficients.copy()
        pt_df['JOINT_F_STAT'] = result.parallel_trends.joint_f_stat
        pt_df['JOINT_P_VALUE'] = result.parallel_trends.joint_p_value
        pt_df['PASSES'] = result.parallel_trends.passes
        results['ESSAY2_PARALLEL_TRENDS'] = store.write_table(
            pt_df.assign(RUN_TIMESTAMP=timestamp),
            'ESSAY2_PARALLEL_TRENDS', replace=True,
        )

    if result.diagnostics is not None:
        diag = result.diagnostics
        for attr_name, table_suffix in [
            ('placebo_tests', 'PLACEBO'), ('bootstrap_ci', 'BOOTSTRAP_CI'),
            ('cluster_robust', 'CLUSTER_ROBUST'), ('normality', 'NORMALITY'),
            ('heteroskedasticity', 'HETEROSKEDASTICITY'), ('vif', 'VIF'),
            ('covariate_balance', 'COVARIATE_BALANCE'), ('autocorrelation', 'AUTOCORRELATION'),
        ]:
            df = getattr(diag, attr_name)
            if df is not None and not df.empty:
                results[f'ESSAY2_DIAG_{table_suffix}'] = store.write_table(
                    df.assign(RUN_TIMESTAMP=timestamp), f'ESSAY2_DIAG_{table_suffix}', replace=True)

    saved = sum(1 for v in results.values() if v is not None)
    logger.info("Essay 2 DiD: saved %d/%d tables", saved, len(results))
    return results


# =========================================================================
# MAIN
# =========================================================================

if __name__ == '__main__':
    from datetime import datetime

    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
    )

    print("Dissertation Essay 2 — Culture War Event DiD")
    print(f"Started at {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print()

    store = DataStore()

    # Step 1: Regime estimation (shared with Essay 1)
    print("=" * 60)
    print("  Step 1: Estimate VIX regimes")
    print("=" * 60)
    regime_result = estimate_vix_regimes(store, n_regimes=3)
    if regime_result is not None:
        for label, mean in regime_result.regime_means.items():
            print(f"    {label}: mean VIX={mean:.1f}")
    else:
        print("  FAILED — no VIX data")

    # Step 2: Build CAR panel
    print()
    print("=" * 60)
    print("  Step 2: Build CAR panel")
    print("=" * 60)
    car_panel = build_car_panel(store, regime_result=regime_result)
    if car_panel is not None:
        print(f"  Panel: {len(car_panel)} rows")
        print(f"  Events: {car_panel['EVENT_ID'].nunique()}")
        print(f"  Treatment firms: {car_panel[car_panel['IS_TREATMENT']]['TICKER'].nunique()}")
        print(f"  Control firms: {car_panel[~car_panel['IS_TREATMENT']]['TICKER'].nunique()}")
        print(f"  Mean CAR_POST (treatment): "
              f"{car_panel[car_panel['IS_TREATMENT']]['CAR_POST'].mean():.4f}")
        print(f"  Mean CAR_POST (control):   "
              f"{car_panel[~car_panel['IS_TREATMENT']]['CAR_POST'].mean():.4f}")
    else:
        print("  FAILED — no events or data")

    # Step 3: DiD regressions
    print()
    print("=" * 60)
    print("  Step 3: DiD regressions")
    print("=" * 60)
    did_result = run_did(store, regime_result=regime_result, car_panel=car_panel)

    if did_result is not None:
        print(f"  Observations: {did_result.n_observations}")

        # Parallel trends pre-test
        pt = did_result.parallel_trends
        if pt is not None:
            print()
            print("  --- Parallel Trends Pre-Test ---")
            verdict = "PASS" if pt.passes else "FAIL"
            print(f"  Joint F={pt.joint_f_stat:.2f}, p={pt.joint_p_value:.4f} "
                  f"[{verdict}]  ({pt.n_observations} obs, {pt.n_days} days)")
            if not pt.daily_coefficients.empty:
                print("  Day-by-day Treat x Day coefficients:")
                for _, row in pt.daily_coefficients.iterrows():
                    sig = "*" if row['P_VALUE'] < 0.05 else ""
                    print(f"    Day {int(row['DAY']):+3d}: "
                          f"coeff={row['COEFFICIENT']:+.6f} "
                          f"(t={row['T_STAT']:+.2f}, p={row['P_VALUE']:.4f}){sig}")
            if not pt.passes:
                print("  WARNING: parallel trends assumption violated — "
                      "DiD estimates may be biased")
        else:
            print("\n  Parallel trends: SKIPPED (insufficient data)")

        for spec_name, fit in [
            ('Basic DiD', did_result.did_basic),
            ('With Political Lean', did_result.did_with_lean),
            ('With FOMO Z-score', did_result.did_with_fomo),
        ]:
            if fit is None:
                print(f"\n  --- {spec_name}: SKIPPED (insufficient data) ---")
                continue
            print(f"\n  --- {spec_name} ---")
            print(f"  R-squared: {fit.rsquared:.4f}")
            print(f"  N: {int(fit.nobs)}")
            for var in fit.params.index:
                sig = ""
                if fit.pvalues[var] < 0.01:
                    sig = "***"
                elif fit.pvalues[var] < 0.05:
                    sig = "**"
                elif fit.pvalues[var] < 0.10:
                    sig = "*"
                print(f"    {var:25s} {fit.params[var]:+.6f} "
                      f"(t={fit.tvalues[var]:+.2f}, p={fit.pvalues[var]:.4f}){sig}")
    else:
        print("  FAILED — no DiD results")

    # Step 4: Save DiD results
    print()
    print("=" * 60)
    print("  Step 4: Save DiD results")
    print("=" * 60)
    if did_result is not None:
        saved = save_did_results(store, did_result)
        for table, res in saved.items():
            print(f"    {table}: {res}")
        print(f"  Saved {len(saved)} tables")
    else:
        print("  Skipped — no results")

    # Step 4b: Diagnostic tests
    print()
    print("=" * 60)
    print("  Step 4b: Diagnostic tests for DiD regressions")
    print("=" * 60)
    if did_result is not None:
        diag = run_diagnostics(store, did_result)
        did_result.diagnostics = diag

        # Placebo test summary
        if not diag.placebo_tests.empty:
            actual_row = diag.placebo_tests[diag.placebo_tests['ROW_TYPE'] == 'ACTUAL_SUMMARY']
            if not actual_row.empty:
                ar = actual_row.iloc[0]
                print(f"  Placebo test ({diag.n_placebo_iterations} iterations):")
                print(f"    Actual TREAT_x_POST coeff: {ar['PLACEBO_COEFF']:.6f}")
                print(f"    Percentile rank: {ar['PLACEBO_P']:.4f}")
                print(f"    {'PASS' if ar['PLACEBO_P'] < 0.10 else 'FAIL'}: "
                      f"actual {'is' if ar['PLACEBO_P'] < 0.10 else 'is NOT'} "
                      f"in the extreme tail of placebo distribution")

        # Bootstrap CI
        if not diag.bootstrap_ci.empty:
            print(f"\n  Bootstrap CI ({diag.n_bootstrap_iterations} resamples):")
            for _, row in diag.bootstrap_ci.iterrows():
                print(f"    {row['STATISTIC']}: {row['OBSERVED']:.6f} "
                      f"[{row['CI_LOWER']:.6f}, {row['CI_UPPER']:.6f}]")

        # Cluster-robust SEs
        if not diag.cluster_robust.empty:
            print("\n  Cluster-robust SEs (TREAT_x_POST only):")
            txp = diag.cluster_robust[diag.cluster_robust['VARIABLE'] == 'TREAT_x_POST']
            for _, row in txp.iterrows():
                print(f"    {row['SPECIFICATION']}: SE_CLUSTER_EVENT={row['SE_CLUSTER_EVENT']:.6f}, "
                      f"SE_CLUSTER_FIRM={row['SE_CLUSTER_FIRM']:.6f}")

        # Normality
        if not diag.normality.empty:
            print("\n  Normality tests:")
            for _, row in diag.normality.iterrows():
                print(f"    {row['SPECIFICATION']}: JB={row['JB_STAT']:.2f} (p={row['JB_P']:.4f}), "
                      f"Shapiro={row['SHAPIRO_STAT']:.4f} (p={row['SHAPIRO_P']:.4f}), "
                      f"skew={row['SKEWNESS']:.3f}, kurt={row['KURTOSIS']:.3f}")

        # Heteroskedasticity
        if not diag.heteroskedasticity.empty:
            print("\n  Heteroskedasticity tests:")
            for _, row in diag.heteroskedasticity.iterrows():
                print(f"    {row['SPECIFICATION']}: BP LM={row['BP_LM_STAT']:.2f} "
                      f"(p={row['BP_LM_P']:.4f}), "
                      f"White LM={row['WHITE_LM_STAT']:.2f} (p={row['WHITE_LM_P']:.4f})")

        # VIF
        if not diag.vif.empty:
            print("\n  VIF (values > 10 indicate multicollinearity):")
            high_vif = diag.vif[diag.vif['VIF'] > 10]
            if not high_vif.empty:
                for _, row in high_vif.iterrows():
                    print(f"    WARNING: {row['SPECIFICATION']}/{row['VARIABLE']}: VIF={row['VIF']:.1f}")
            else:
                print("    All VIFs < 10 — no multicollinearity detected")

        # Covariate balance
        if not diag.covariate_balance.empty:
            print("\n  Covariate balance (treatment vs control):")
            for _, row in diag.covariate_balance.iterrows():
                bal = "BALANCED" if row['BALANCED'] else "IMBALANCED"
                print(f"    {row['VARIABLE']}: SMD={row['SMD']:.3f} "
                      f"(p={row['P_VALUE']:.4f}) [{bal}]")

        # Autocorrelation
        if not diag.autocorrelation.empty:
            print("\n  Durbin-Watson autocorrelation test:")
            for _, row in diag.autocorrelation.iterrows():
                print(f"    {row['SPECIFICATION']}: DW={row['DW_STAT']:.3f} "
                      f"({row['INTERPRETATION']})")

        # Re-save with diagnostics
        print()
        saved = save_did_results(store, did_result)
        diag_tables = [k for k in saved if 'DIAG' in k]
        print(f"  Saved {len(diag_tables)} diagnostic tables")
    else:
        print("  Skipped — no DiD results")

    # Step 5: Multi-window event study [-1, +5/10/15/20/30/60/90]
    print()
    print("=" * 60)
    print("  Step 5: Multi-window event study (FF5)")
    print("  Windows: [-1, +5], [-1, +10], [-1, +15], [-1, +20],")
    print("           [-1, +30], [-1, +60], [-1, +90]")
    print("=" * 60)
    mw_result = run_multi_window_event_study(
        store, regime_result=regime_result,
    )

    if mw_result is not None:
        print(f"  Events: {mw_result.n_events}")
        print(f"  Treatment firms: {mw_result.n_treatment}")
        print(f"  Control firms: {mw_result.n_control}")

        # Summary table
        print()
        print("  --- Treatment CARs by Window ---")
        print(f"  {'Window':<14} {'N':>5} {'Mean CAR':>10} {'Median':>10} "
              f"{'Std':>10} {'t-stat':>8} {'p-value':>8} {'Sig':>4}")
        print("  " + "-" * 78)
        for _, row in mw_result.summary.iterrows():
            sig = ""
            p = row['P_VALUE_VS_ZERO']
            if not np.isnan(p):
                if p < 0.01:
                    sig = "***"
                elif p < 0.05:
                    sig = "**"
                elif p < 0.10:
                    sig = "*"
            print(f"  {row['WINDOW']:<14} {row['N_TREAT']:>5.0f} "
                  f"{row['MEAN_CAR_TREAT']:>+10.4f} {row['MEDIAN_CAR_TREAT']:>+10.4f} "
                  f"{row['STD_CAR_TREAT']:>10.4f} "
                  f"{row['T_STAT_VS_ZERO']:>+8.2f} {row['P_VALUE_VS_ZERO']:>8.4f} "
                  f"{sig:>4}")

        # Treatment vs Control
        print()
        print("  --- Treatment vs Control (Welch t-test) ---")
        print(f"  {'Window':<14} {'Diff':>10} {'t-stat':>8} {'p-value':>8} "
              f"{'Cohen d':>8} {'Sig':>4}")
        print("  " + "-" * 56)
        for _, row in mw_result.treatment_vs_control.iterrows():
            sig = ""
            p = row['P_VALUE']
            if not np.isnan(p):
                if p < 0.01:
                    sig = "***"
                elif p < 0.05:
                    sig = "**"
                elif p < 0.10:
                    sig = "*"
            print(f"  {row['WINDOW']:<14} {row['DIFF_TREAT_CTRL']:>+10.4f} "
                  f"{row['T_STAT']:>+8.2f} {row['P_VALUE']:>8.4f} "
                  f"{row['COHENS_D']:>+8.3f} {sig:>4}")

        # By political lean
        if not mw_result.by_lean.empty:
            print()
            print("  --- Mean Treatment CARs by Political Lean ---")
            pivot = mw_result.by_lean.pivot_table(
                index='WINDOW', columns='LEAN', values='MEAN_CAR',
            )
            for lean_col in ['Conservative', 'Liberal', 'Mixed']:
                if lean_col not in pivot.columns:
                    pivot[lean_col] = np.nan
            print(f"  {'Window':<14} {'Conservative':>14} {'Liberal':>14} {'Mixed':>14}")
            print("  " + "-" * 58)
            for window_label in [f'[-1, +{w}]' for w in _MULTI_WINDOWS]:
                if window_label in pivot.index:
                    r = pivot.loc[window_label]
                    print(f"  {window_label:<14} {r.get('Conservative', np.nan):>+14.4f} "
                          f"{r.get('Liberal', np.nan):>+14.4f} "
                          f"{r.get('Mixed', np.nan):>+14.4f}")

        # Save
        print()
        saved = save_multi_window_results(store, mw_result)
        for table, res in saved.items():
            print(f"    {table}: {res}")
        print(f"  Saved {len(saved)} tables")
    else:
        print("  FAILED — no multi-window results")

    # Step 6: Contagion / spillover test
    print()
    print("=" * 60)
    print("  Step 6: Industry contagion test")
    print("  Do culture war events spill over to industry peers?")
    print("=" * 60)
    contagion_result = run_contagion_test(
        store, regime_result=regime_result,
    )

    if contagion_result is not None:
        print(f"  Events with peers: {contagion_result.n_events_with_peers}")
        print(f"  Unique peer firms: {contagion_result.n_unique_peers}")
        print(f"  Unique non-peer firms: {contagion_result.n_unique_nonpeers}")

        # Peer CARs vs zero
        print()
        print("  --- Peer CARs by Window (contagion = CARs != 0) ---")
        print(f"  {'Window':<14} {'N':>6} {'Mean CAR':>10} {'Median':>10} "
              f"{'t-stat':>8} {'p-value':>8} {'Sig':>4}")
        print("  " + "-" * 64)
        for _, row in contagion_result.summary.iterrows():
            sig = ""
            p = row['P_VALUE_VS_ZERO']
            if not np.isnan(p):
                if p < 0.01:
                    sig = "***"
                elif p < 0.05:
                    sig = "**"
                elif p < 0.10:
                    sig = "*"
            print(f"  {row['WINDOW']:<14} {row['N_PEER_OBS']:>6.0f} "
                  f"{row['MEAN_PEER_CAR']:>+10.4f} {row['MEDIAN_PEER_CAR']:>+10.4f} "
                  f"{row['T_STAT_VS_ZERO']:>+8.2f} {row['P_VALUE_VS_ZERO']:>8.4f} "
                  f"{sig:>4}")

        # Peer vs Non-Peer
        print()
        print("  --- Peer vs Non-Peer (differential contagion) ---")
        print(f"  {'Window':<14} {'Peer CAR':>10} {'Non-Peer':>10} {'Diff':>10} "
              f"{'t-stat':>8} {'p-value':>8} {'d':>7} {'Sig':>4}")
        print("  " + "-" * 79)
        for _, row in contagion_result.peer_vs_nonpeer.iterrows():
            sig = ""
            p = row['P_VALUE']
            if not np.isnan(p):
                if p < 0.01:
                    sig = "***"
                elif p < 0.05:
                    sig = "**"
                elif p < 0.10:
                    sig = "*"
            print(f"  {row['WINDOW']:<14} {row['MEAN_PEER_CAR']:>+10.4f} "
                  f"{row['MEAN_NONPEER_CAR']:>+10.4f} "
                  f"{row['DIFF_PEER_NONPEER']:>+10.4f} "
                  f"{row['T_STAT']:>+8.2f} {row['P_VALUE']:>8.4f} "
                  f"{row['COHENS_D']:>+7.3f} {sig:>4}")

        # By triggering firm's lean
        if not contagion_result.by_event_lean.empty:
            print()
            print("  --- Peer Contagion by Triggering Firm's Political Lean ---")
            pivot = contagion_result.by_event_lean.pivot_table(
                index='WINDOW', columns='EVENT_LEAN', values='MEAN_PEER_CAR',
            )
            for lean_col in ['Conservative', 'Liberal', 'Mixed']:
                if lean_col not in pivot.columns:
                    pivot[lean_col] = np.nan
            print(f"  {'Window':<14} {'Conservative':>14} {'Liberal':>14} {'Mixed':>14}")
            print("  " + "-" * 58)
            for window_label in [f'[-1, +{w}]' for w in _MULTI_WINDOWS]:
                if window_label in pivot.index:
                    r = pivot.loc[window_label]
                    print(f"  {window_label:<14} {r.get('Conservative', np.nan):>+14.4f} "
                          f"{r.get('Liberal', np.nan):>+14.4f} "
                          f"{r.get('Mixed', np.nan):>+14.4f}")

        # Save
        print()
        saved = save_contagion_results(store, contagion_result)
        for table, res in saved.items():
            print(f"    {table}: {res}")
        print(f"  Saved {len(saved)} tables")
    else:
        print("  FAILED — no contagion results")

    # Step 7: Enhanced contagion — tight non-peers, consumer/B2B, lean mechanism
    print()
    print("=" * 60)
    print("  Step 7: Enhanced contagion tests")
    print("  (a) Tight non-peer definition (different NAICS sector)")
    print("  (b) Consumer-facing vs B2B heterogeneity")
    print("  (c) Mixed-lean uncertainty mechanism")
    print("=" * 60)
    enhanced = run_enhanced_contagion(store, regime_result=regime_result)

    if enhanced is not None:
        # --- 7a: Tight non-peer differential ---
        print()
        print("  (a) Peer vs Tight Non-Peer (different NAICS-2 sector)")
        print(f"  {'Window':<14} {'Peer CAR':>10} {'Non-Peer':>10} {'Diff':>10} "
              f"{'t-stat':>8} {'p-value':>8} {'d':>7} {'Sig':>4}")
        print("  " + "-" * 79)
        for _, row in enhanced.tight_peer_vs_nonpeer.iterrows():
            sig = ""
            p = row['P_VALUE']
            if not np.isnan(p):
                if p < 0.01: sig = "***"
                elif p < 0.05: sig = "**"
                elif p < 0.10: sig = "*"
            print(f"  {row['WINDOW']:<14} {row['MEAN_PEER_CAR']:>+10.4f} "
                  f"{row['MEAN_NONPEER_CAR']:>+10.4f} "
                  f"{row['DIFF']:>+10.4f} "
                  f"{row['T_STAT']:>+8.2f} {row['P_VALUE']:>8.4f} "
                  f"{row['COHENS_D']:>+7.3f} {sig:>4}")

        # --- 7b: Consumer vs B2B ---
        print()
        print("  (b) Peer Contagion by Event Firm's Industry Type")
        print(f"  {'Window':<14} {'Type':<14} {'N':>5} {'Mean CAR':>10} "
              f"{'Median':>10} {'t vs 0':>8} {'p':>8}")
        print("  " + "-" * 78)
        for _, row in enhanced.by_industry_type.iterrows():
            if row['EVENT_FACING'] in ('CONSUMER', 'B2B'):
                sig = ""
                p = row['P_VALUE_VS_ZERO']
                if not np.isnan(p):
                    if p < 0.01: sig = "***"
                    elif p < 0.05: sig = "**"
                    elif p < 0.10: sig = "*"
                print(f"  {row['WINDOW']:<14} {row['EVENT_FACING']:<14} "
                      f"{row['N']:>5.0f} {row['MEAN_PEER_CAR']:>+10.4f} "
                      f"{row['MEDIAN_PEER_CAR']:>+10.4f} "
                      f"{row['T_STAT_VS_ZERO']:>+8.2f} {row['P_VALUE_VS_ZERO']:>8.4f}{sig}")

        print()
        print("  Consumer vs B2B t-test:")
        print(f"  {'Window':<14} {'Cons CAR':>10} {'B2B CAR':>10} {'Diff':>10} "
              f"{'t-stat':>8} {'p-value':>8} {'d':>7} {'Sig':>4}")
        print("  " + "-" * 79)
        for _, row in enhanced.consumer_vs_b2b_tests.iterrows():
            sig = ""
            p = row['P_VALUE']
            if not np.isnan(p):
                if p < 0.01: sig = "***"
                elif p < 0.05: sig = "**"
                elif p < 0.10: sig = "*"
            print(f"  {row['WINDOW']:<14} {row['MEAN_CONSUMER']:>+10.4f} "
                  f"{row['MEAN_B2B']:>+10.4f} "
                  f"{row['DIFF_CONS_B2B']:>+10.4f} "
                  f"{row['T_STAT']:>+8.2f} {row['P_VALUE']:>8.4f} "
                  f"{row['COHENS_D']:>+7.3f} {sig:>4}")

        # --- 7c: Mixed-lean mechanism ---
        print()
        print("  (c) Peer Contagion by Triggering Firm's Lean (with significance)")
        print(f"  {'Window':<14} {'Lean':<14} {'N':>5} {'Mean CAR':>10} "
              f"{'t vs 0':>8} {'p':>8}")
        print("  " + "-" * 66)
        for _, row in enhanced.lean_mechanism.iterrows():
            sig = ""
            p = row['P_VALUE_VS_ZERO']
            if not np.isnan(p):
                if p < 0.01: sig = "***"
                elif p < 0.05: sig = "**"
                elif p < 0.10: sig = "*"
            print(f"  {row['WINDOW']:<14} {row['EVENT_LEAN']:<14} "
                  f"{row['N']:>5.0f} {row['MEAN_PEER_CAR']:>+10.4f} "
                  f"{row['T_STAT_VS_ZERO']:>+8.2f} {row['P_VALUE_VS_ZERO']:>8.4f}{sig}")

        print()
        print("  Pairwise lean comparisons:")
        print(f"  {'Window':<14} {'Comparison':<24} {'Diff':>10} "
              f"{'t-stat':>8} {'p-value':>8} {'d':>7} {'Sig':>4}")
        print("  " + "-" * 83)
        for _, row in enhanced.lean_pairwise_tests.iterrows():
            sig = ""
            p = row['P_VALUE']
            if not np.isnan(p):
                if p < 0.01: sig = "***"
                elif p < 0.05: sig = "**"
                elif p < 0.10: sig = "*"
            print(f"  {row['WINDOW']:<14} {row['COMPARISON']:<24} "
                  f"{row['DIFF']:>+10.4f} "
                  f"{row['T_STAT']:>+8.2f} {row['P_VALUE']:>8.4f} "
                  f"{row['COHENS_D']:>+7.3f} {sig:>4}")

        # Save
        print()
        saved = save_enhanced_contagion(store, enhanced)
        for table, res in saved.items():
            print(f"    {table}: {res}")
        print(f"  Saved {len(saved)} tables")
    else:
        print("  FAILED — no enhanced contagion results")

    # Step 8: Parallel trends tests
    print()
    print("=" * 60)
    print("  Step 8: Parallel trends pre-tests")
    print("  (a) Treatment vs matched control (DiD validity)")
    print("  (b) Event firm vs industry peers (contagion validity)")
    print("=" * 60)

    # --- 8a: Treatment vs control parallel trends ---
    print()
    print("  (a) Treatment vs Matched Control — pre-event window [-10, -1]")
    pt_did = did_result.parallel_trends if did_result is not None else None
    if pt_did is not None:
        verdict = "PASS" if pt_did.passes else "FAIL"
        print(f"      Joint F = {pt_did.joint_f_stat:.3f},  p = {pt_did.joint_p_value:.4f}  "
              f"[{verdict}]")
        print(f"      Observations: {pt_did.n_observations},  Days: {pt_did.n_days}")
        if not pt_did.daily_coefficients.empty:
            print()
            print(f"      {'Day':>6} {'Coeff':>12} {'SE':>10} {'t-stat':>8} {'p':>8}")
            print("      " + "-" * 50)
            for _, row in pt_did.daily_coefficients.iterrows():
                sig = "*" if row['P_VALUE'] < 0.05 else ""
                print(f"      {int(row['DAY']):>+6d} {row['COEFFICIENT']:>+12.6f} "
                      f"{row['STD_ERROR']:>10.6f} {row['T_STAT']:>+8.2f} "
                      f"{row['P_VALUE']:>8.4f}{sig}")
        if pt_did.passes:
            print()
            print("      RESULT: Pre-event trends are parallel (p > 0.05).")
            print("      The DiD identifying assumption holds.")
        else:
            print()
            print("      WARNING: Pre-event trends are NOT parallel.")
            print("      DiD estimates should be interpreted with caution.")
    else:
        print("      SKIPPED — insufficient matched control data")

    # --- 8b: Event firm vs peers parallel trends ---
    print()
    print("  (b) Event Firm vs Industry Peers — pre-event window [-10, -1]")
    # Only run peer parallel trends if contagion data was available
    peer_pt = None
    if contagion_result is not None:
        peer_pt = peer_parallel_trends_test(store)
    else:
        logger.info("Skipping peer parallel trends — no contagion results")
    if peer_pt is not None:
        verdict = "PASS" if peer_pt.passes else "FAIL"
        print(f"      Joint F = {peer_pt.joint_f_stat:.3f},  p = {peer_pt.joint_p_value:.4f}  "
              f"[{verdict}]")
        print(f"      Observations: {peer_pt.n_observations},  Days: {peer_pt.n_days}")
        if not peer_pt.daily_coefficients.empty:
            print()
            print(f"      {'Day':>6} {'Coeff':>12} {'SE':>10} {'t-stat':>8} {'p':>8}")
            print("      " + "-" * 50)
            n_sig = 0
            for _, row in peer_pt.daily_coefficients.iterrows():
                sig = "*" if row['P_VALUE'] < 0.05 else ""
                if row['P_VALUE'] < 0.05:
                    n_sig += 1
                print(f"      {int(row['DAY']):>+6d} {row['COEFFICIENT']:>+12.6f} "
                      f"{row['STD_ERROR']:>10.6f} {row['T_STAT']:>+8.2f} "
                      f"{row['P_VALUE']:>8.4f}{sig}")
            print()
            print(f"      Individually significant days: {n_sig}/{peer_pt.n_days}")

        if peer_pt.passes:
            print()
            print("      RESULT: Event firms and peers trend together pre-event (p > 0.05).")
            print("      The contagion test's identifying assumption holds.")
        else:
            print()
            print("      WARNING: Event firms and peers do NOT trend together pre-event.")
            print("      Contagion results should be interpreted with caution.")

        # Save
        pt_df = peer_pt.daily_coefficients.copy()
        pt_df['JOINT_F_STAT'] = peer_pt.joint_f_stat
        pt_df['JOINT_P_VALUE'] = peer_pt.joint_p_value
        pt_df['PASSES'] = peer_pt.passes
        pt_df['TEST_TYPE'] = 'PEER_CONTAGION'
        ts = pd.Timestamp.now().isoformat()
        store.write_table(
            pt_df.assign(RUN_TIMESTAMP=ts),
            'ESSAY2_PEER_PARALLEL_TRENDS', replace=True,
        )
        print("      Saved to ESSAY2_PEER_PARALLEL_TRENDS")
    else:
        print("      SKIPPED — insufficient peer data")

    # ── Upload results to AWS (S3 + Glue) ──
    print()
    print("=" * 60)
    print("  Upload results to AWS")
    print("=" * 60)

    # Collect all ESSAY2_ tables written during this run
    _essay2_tables = [
        'ESSAY2_CAR_PANEL', 'ESSAY2_DID_COEFFICIENTS', 'ESSAY2_PARALLEL_TRENDS',
        'ESSAY2_MULTI_WINDOW_PANEL', 'ESSAY2_MULTI_WINDOW_SUMMARY',
        'ESSAY2_MULTI_WINDOW_TREAT_VS_CTRL', 'ESSAY2_MULTI_WINDOW_BY_LEAN',
        'ESSAY2_CONTAGION_PANEL', 'ESSAY2_CONTAGION_SUMMARY',
        'ESSAY2_CONTAGION_PEER_VS_NONPEER', 'ESSAY2_CONTAGION_BY_LEAN',
        'ESSAY2_CONTAGION_TIGHT_DIFF', 'ESSAY2_CONTAGION_BY_FACING',
        'ESSAY2_CONTAGION_CONS_VS_B2B', 'ESSAY2_CONTAGION_LEAN_MECH',
        'ESSAY2_CONTAGION_LEAN_PAIRWISE', 'ESSAY2_ENHANCED_CONTAGION_PANEL',
        'ESSAY2_PEER_PARALLEL_TRENDS',
        'ESSAY2_DIAG_PLACEBO', 'ESSAY2_DIAG_BOOTSTRAP_CI',
        'ESSAY2_DIAG_CLUSTER_ROBUST', 'ESSAY2_DIAG_NORMALITY',
        'ESSAY2_DIAG_HETEROSKEDASTICITY', 'ESSAY2_DIAG_VIF',
        'ESSAY2_DIAG_COVARIATE_BALANCE', 'ESSAY2_DIAG_AUTOCORRELATION',
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
