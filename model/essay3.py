"""Essay 3 — Insider Trading Around Culture War Events.

Structure mirrors the dissertation:
  4.1 Aggregate Tests (null characterized by TOST + power)
  4.2 Subgroup Analysis (signal-finding: C-suite, CAR severity, tight window)
  4.3 Robustness (DiD, placebo, 10b5-1 filter, bootstrap CIs)

Analyses:
  1.  Event-level insider trading panel (windows around each event)
  2.  Window summary statistics
  3.  Abnormal selling tests (pre-event vs benchmark)
  4.  Treatment vs control regression
  5.  Political leaning analysis
  6.  VIX regime interaction
  7.  Routine vs opportunistic insider classification
  8.  Placebo permutation test
  9.  Acceleration (Jonckheere-Terpstra) test
  10. CAR-insider regression
  11. Information gradient
  12. Difference-in-differences (pre/post × treatment/control)
  13. 10b5-1 plan filter (scheduled vs discretionary)
  14. Match quality validation (covariate balance)
  15. Event clustering adjustment (date-clustered SEs)
  16. Intensive vs extensive margin decomposition
  17. Abnormal volume ratio (normalized by benchmark volume)
  18. Firm fixed effects regression
  19. Cross-sectional determinants of abnormal selling
  20. Post-event reversal test (selling terciles × CAR)
  21. Bootstrap confidence intervals
  22. Fama-MacBeth regressions
  23. Short-swing profit rule (Section 16b) check
  24. TOST equivalence tests + power analysis (aggregate null characterization)
  25. Subgroup signal-finding (C-suite, high-CAR, tight window, normalized ratio)
  26. Volatility-shift identification strategy (firm-level vol spikes)
  27. Tail diagnostic: leaning × event type (who drives distributional diff?)
      a. Tail firm identification (top quintile abnormal sellers)
      b. Leaning distribution in tail (chi-square, KW)
      c. Planned vs reactive event type (two-sample + one-sample)
      d. Leaning × event type interaction (2×3 cells + OLS)
  28. Conservative × Planned deep dive (case-study table + power diagnostic)
  29. Industry-controlled logit for tail membership
  30. Propensity score matching for tail analysis
  31. Winsorized tail chi-square (1%, 5% clips)
  32. Size-stratified tail analysis (market-cap terciles)
  33. Insider-level tail concentration analysis
  34. Time-series tail decomposition (pre-DEI vs DEI era)
  35. Placebo test stratified by political leaning
  36. Within-firm temporal clustering of tail episodes
  37. Disclosure channel (10b5-1 × leaning in tail)
  38. Buy-side analysis (abnormal buy reduction)

Saves 41 tables to SQLite for visual.py and dashboard.py.
"""

import logging
import os
import re
import warnings

import numpy as np
import pandas as pd
from scipy import stats
import statsmodels.api as sm

from .datastore import DataStore

logger = logging.getLogger(__name__)

# ── Window definitions (days relative to event) ─────────────────────────
WINDOWS = {
    'BENCHMARK':  (-365, -181),
    'PRE_FAR':    (-180, -61),
    'PRE_MID':    (-60, -31),
    'PRE_NEAR':   (-30, -1),
    'PRE_FULL':   (-180, -1),
    'POST':       (0, 60),
}

# Transaction codes that represent sells vs buys
SELL_CODES = {'S'}        # S = open-market sale (F = tax withholding, excluded as non-discretionary)
BUY_CODES  = {'P'}        # P = open-market purchase


# ═══════════════════════════════════════════════════════════════════════
# HELPERS
# ═══════════════════════════════════════════════════════════════════════

def _load_form4(project_dir=None):
    """Load the combined Form 4 CSV."""
    if project_dir is None:
        project_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    csv_path = os.path.join(project_dir, 'sec_form4_data',
                            'form4_transactions_2000-01-01_to_2025-12-31.csv')
    if not os.path.exists(csv_path):
        logger.error("Form 4 CSV not found: %s", csv_path)
        return pd.DataFrame()
    df = pd.read_csv(csv_path)
    df['transaction_date'] = pd.to_datetime(df['transaction_date'], errors='coerce')
    df['filing_date'] = pd.to_datetime(df['filing_date'], errors='coerce')
    # Ensure numeric
    # shares/shares_owned_after: fill NaN with 0 (missing means no shares
    # transacted or reported; zeroing preserves row for event matching)
    for col in ['shares', 'shares_owned_after']:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors='coerce').fillna(0)
    for col in ['price_per_share', 'transaction_value']:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors='coerce')
    # Compute transaction_value if column missing; fill NaN rows from shares × price
    if 'transaction_value' not in df.columns:
        df['transaction_value'] = df['shares'] * df['price_per_share']
    else:
        missing = df['transaction_value'].isna()
        if missing.any():
            df.loc[missing, 'transaction_value'] = (
                df.loc[missing, 'shares'] * df.loc[missing, 'price_per_share']
            )
            logger.info("  Imputed %d missing transaction_value rows at load time",
                        missing.sum())
    logger.info("Loaded Form 4: %d transactions, %d tickers",
                len(df), df['ticker'].nunique())
    return df


def _classify_trade(code, acq_disp):
    """Classify a transaction as 'sell', 'buy', or 'other'.

    Following Cohen, Malloy & Pomorski (2012): only open-market sales (S)
    and purchases (P) are used. Derivative exercises (M) are excluded to
    avoid double-counting when a paired S transaction is also present.
    """
    code = str(code).upper().strip()
    acq_disp = str(acq_disp).upper().strip()
    if code in SELL_CODES:
        return 'sell'
    if code in BUY_CODES:
        return 'buy'
    # M (derivative exercise) excluded — often paired with S, causing
    # double-counting. F (tax withholding) is non-discretionary.
    return 'other'


def _is_routine_insider(form4, ticker, owner, event_date, lookback_years=2):
    """
    An insider is 'routine' if they traded in >= 3 of last 8 quarters
    before the event. Otherwise 'opportunistic'.
    """
    start = event_date - pd.DateOffset(years=lookback_years)
    mask = ((form4['ticker'] == ticker) &
            (form4['owner_name'] == owner) &
            (form4['transaction_date'] >= start) &
            (form4['transaction_date'] < event_date))
    trades = form4.loc[mask, 'transaction_date']
    if trades.empty:
        # No prior 2-year history → coded as None (unknown), NOT False (opportunistic).
        # Robustness: exclude None-history insiders to avoid thin-history firms
        # driving the opportunistic signal (see _is_routine_insider robustness row).
        return None
    quarters = trades.dt.to_period('Q').nunique()
    return quarters >= 3


def _compute_window_metrics(txns, window_days):
    """Compute trading metrics for a set of transactions in a window."""
    sells = txns[txns['_trade_type'] == 'sell']
    buys = txns[txns['_trade_type'] == 'buy']

    # transaction_value is imputed at load time in _load_form4;
    # no per-window imputation needed (avoids inconsistent imputation
    # between buy/sell subsets)

    n_transactions = len(txns)
    n_sells = len(sells)
    n_buys = len(buys)
    shares_sold = sells['shares'].sum()
    shares_bought = buys['shares'].sum()
    net_shares_sold = shares_sold - shares_bought
    dollar_sold = sells['transaction_value'].sum()
    dollar_bought = buys['transaction_value'].sum()
    net_dollar_sold = dollar_sold - dollar_bought
    total_dollar = dollar_sold + dollar_bought
    net_sell_ratio = net_dollar_sold / total_dollar if total_dollar > 0 else 0.0
    n_unique_insiders = txns['owner_name'].nunique() if len(txns) > 0 else 0
    # Explicit False = opportunistic (has 2yr history, <3 active quarters).
    # None = unknown history — excluded from both counts.
    n_opportunistic = txns[txns['_is_routine'] == False]['owner_name'].nunique() if len(txns) > 0 else 0  # noqa: E712
    n_routine = txns[txns['_is_routine'] == True]['owner_name'].nunique() if len(txns) > 0 else 0  # noqa: E712

    return {
        'N_TRANSACTIONS': n_transactions,
        'N_SELLS': n_sells,
        'N_BUYS': n_buys,
        'SHARES_SOLD': shares_sold,
        'SHARES_BOUGHT': shares_bought,
        'NET_SHARES_SOLD': net_shares_sold,
        'DOLLAR_SOLD': dollar_sold,
        'DOLLAR_BOUGHT': dollar_bought,
        'NET_DOLLAR_SOLD': net_dollar_sold,
        'NET_SELL_RATIO': net_sell_ratio,
        'N_UNIQUE_INSIDERS': n_unique_insiders,
        'N_OPPORTUNISTIC': n_opportunistic,
        'N_ROUTINE': n_routine,
    }


def _benjamini_hochberg(pvals, alpha=0.1):
    """Apply BH correction, return boolean array of significance."""
    pvals = np.array(pvals, dtype=float)
    n = len(pvals)
    if n == 0:
        return np.array([], dtype=bool)
    ranked = np.argsort(pvals)
    thresholds = alpha * (np.arange(1, n + 1)) / n
    significant = np.zeros(n, dtype=bool)
    sorted_p = pvals[ranked]
    # Find largest k where p(k) <= threshold(k)
    below = sorted_p <= thresholds
    if below.any():
        k = np.max(np.where(below)[0])
        significant[ranked[:k + 1]] = True
    return significant


# ═══════════════════════════════════════════════════════════════════════
# PANEL BUILDER
# ═══════════════════════════════════════════════════════════════════════

def build_insider_panel(form4, events, car_df, vix_regimes=None):
    """
    Build event-level insider trading panel.

    For each event (treatment + pseudo-events for controls), compute
    trading metrics in each window.

    Returns DataFrame with one row per event.
    """
    logger.info("Building insider trading panel...")

    # Build event list: treatment events + control pseudo-events
    event_rows = []
    for _, ev in events.iterrows():
        ticker = ev['TICKER']
        event_date = pd.Timestamp(ev['EVENT_DATE'])
        lean = ev.get('ESTIMATED_POLITICAL_LEANING', 'Unknown')
        event_id = f"{ticker}_{event_date.strftime('%Y%m%d')}"

        # Pair ID: shared by treatment and its matched control
        pair_id = f"{ticker}_{event_date.strftime('%Y%m%d')}"

        # Treatment event
        event_rows.append({
            'TICKER': ticker,
            'EVENT_ID': event_id,
            'EVENT_DATE': event_date,
            'IS_TREATMENT': True,
            'LEAN': lean,
            'PAIR_ID': pair_id,
        })

        # Control pseudo-event (same date, different ticker)
        ctrl_ticker = ev.get('CONTROL_TICKER')
        if pd.notna(ctrl_ticker) and ctrl_ticker:
            ctrl_id = f"{ctrl_ticker}_{event_date.strftime('%Y%m%d')}"
            event_rows.append({
                'TICKER': ctrl_ticker,
                'EVENT_ID': ctrl_id,
                'EVENT_DATE': event_date,
                'IS_TREATMENT': False,
                'LEAN': lean,
                'PAIR_ID': pair_id,
            })

    event_list = pd.DataFrame(event_rows)
    logger.info("  %d events (%d treatment, %d control)",
                len(event_list),
                event_list['IS_TREATMENT'].sum(),
                (~event_list['IS_TREATMENT']).sum())

    # Precompute routine/opportunistic classification per insider per event date
    logger.info("  Classifying routine vs opportunistic insiders...")
    routine_cache = {}
    for _, ev_row in event_list.iterrows():
        ticker = ev_row['TICKER']
        event_date = ev_row['EVENT_DATE']
        ticker_txns = form4[form4['ticker'] == ticker]
        for owner in ticker_txns['owner_name'].unique():
            key = (ticker, owner, event_date)
            if key not in routine_cache:
                routine_cache[key] = _is_routine_insider(form4, ticker, owner, event_date)

    # Ensure _trade_type is tagged (may already be done by caller)
    form4 = form4.copy()
    if '_trade_type' not in form4.columns:
        form4['_trade_type'] = form4.apply(
            lambda r: _classify_trade(r['transaction_code'], r['acquired_disposed']),
            axis=1
        )
    # _is_routine will be set per-event in the panel loop below
    # Default None (unknown history), NOT False — preserves three-valued logic
    form4['_is_routine'] = None

    # Build CAR lookup
    car_lookup = {}
    if car_df is not None and not car_df.empty:
        for _, row in car_df.iterrows():
            key = (row['TICKER'], pd.Timestamp(row['EVENT_DATE']).strftime('%Y%m%d'))
            car_lookup[key] = row.get('CAR', None)

    # Compute metrics for each event × window
    panel_rows = []
    n_events = len(event_list)
    for i, (idx, ev) in enumerate(event_list.iterrows()):
        ticker = ev['TICKER']
        event_date = ev['EVENT_DATE']
        ticker_txns = form4[form4['ticker'] == ticker].copy()

        # Set per-event routine classification
        # Default to None (unknown history) — not False (opportunistic).
        # Insiders with no 2-year history should be excluded from routine/
        # opportunistic counts, not silently lumped into opportunistic.
        ticker_txns['_is_routine'] = ticker_txns['owner_name'].apply(
            lambda owner: routine_cache.get((ticker, owner, event_date), None)
        )

        row = {
            'TICKER': ticker,
            'EVENT_ID': ev['EVENT_ID'],
            'EVENT_DATE': event_date,
            'IS_TREATMENT': ev['IS_TREATMENT'],
            'LEAN': ev['LEAN'],
            'PAIR_ID': ev.get('PAIR_ID', ev['EVENT_ID']),
        }

        # Look up CAR
        car_key = (ticker, event_date.strftime('%Y%m%d'))
        row['CAR_POST'] = car_lookup.get(car_key)

        # Compute each window
        has_data = False
        for window_name, (d_start, d_end) in WINDOWS.items():
            w_start = event_date + pd.Timedelta(days=d_start)
            w_end = event_date + pd.Timedelta(days=d_end)
            mask = ((ticker_txns['transaction_date'] >= w_start) &
                    (ticker_txns['transaction_date'] <= w_end))
            w_txns = ticker_txns[mask]
            metrics = _compute_window_metrics(w_txns, (d_start, d_end))

            for k, v in metrics.items():
                row[f'{window_name}_{k}'] = v

            if metrics['N_TRANSACTIONS'] > 0:
                has_data = True

        # Abnormal selling flag: pre-full daily > benchmark daily
        bench_days = abs(WINDOWS['BENCHMARK'][1] - WINDOWS['BENCHMARK'][0]) + 1
        pre_full_days = abs(WINDOWS['PRE_FULL'][1] - WINDOWS['PRE_FULL'][0]) + 1
        bench_daily = row.get('BENCHMARK_NET_DOLLAR_SOLD', 0) / bench_days
        pre_daily = row.get('PRE_FULL_NET_DOLLAR_SOLD', 0) / pre_full_days
        row['ABNORMAL_SELLING'] = 1 if (pre_daily > bench_daily and pre_daily > 0) else 0
        row['HAS_SUFFICIENT_DATA'] = 1 if has_data else 0

        panel_rows.append(row)

        if (i + 1) % 50 == 0:
            logger.info("  Processed %d/%d events", i + 1, n_events)

    panel = pd.DataFrame(panel_rows)
    logger.info("  Panel built: %d rows, %d with sufficient data",
                len(panel), panel['HAS_SUFFICIENT_DATA'].sum())
    return panel


# ═══════════════════════════════════════════════════════════════════════
# AGGREGATE ANALYSES
# ═══════════════════════════════════════════════════════════════════════

def compute_window_summary(panel):
    """Aggregate insider trading metrics by window, separately for treatment and control."""
    windows = ['BENCHMARK', 'PRE_FAR', 'PRE_MID', 'PRE_NEAR', 'PRE_FULL', 'POST']
    rows = []
    for group_name, sub in [('TREATMENT', panel[panel['IS_TREATMENT'] == True]),
                             ('CONTROL', panel[panel['IS_TREATMENT'] == False]),
                             ('ALL', panel)]:
        for w in windows:
            col_nds = f'{w}_NET_DOLLAR_SOLD'
            col_nsr = f'{w}_NET_SELL_RATIO'
            col_nt = f'{w}_N_TRANSACTIONS'
            col_no = f'{w}_N_OPPORTUNISTIC'
            if col_nds not in sub.columns:
                continue
            rows.append({
                'GROUP': group_name,
                'WINDOW': w,
                'N_EVENTS': len(sub),
                'MEAN_NET_DOLLAR_SOLD': sub[col_nds].mean(),
                'MEDIAN_NET_DOLLAR_SOLD': sub[col_nds].median(),
                'STD_NET_DOLLAR_SOLD': sub[col_nds].std(),
                'MEAN_NET_SELL_RATIO': sub[col_nsr].mean() if col_nsr in sub.columns else 0,
                'MEDIAN_NET_SELL_RATIO': sub[col_nsr].median() if col_nsr in sub.columns else 0,
                'MEAN_N_TRANSACTIONS': sub[col_nt].mean() if col_nt in sub.columns else 0,
                'TOTAL_TRANSACTIONS': int(sub[col_nt].sum()) if col_nt in sub.columns else 0,
                'MEAN_N_OPPORTUNISTIC': sub[col_no].mean() if col_no in sub.columns else 0,
            })
    return pd.DataFrame(rows)


def compute_abnormal_selling(panel):
    """Paired t-test: pre-event daily selling vs benchmark daily selling (treatment only)."""
    treatment = panel[panel['IS_TREATMENT'] == True]
    bench_days = abs(WINDOWS['BENCHMARK'][1] - WINDOWS['BENCHMARK'][0]) + 1
    windows = ['PRE_FAR', 'PRE_MID', 'PRE_NEAR', 'PRE_FULL']
    rows = []
    for w in windows:
        w_days = abs(WINDOWS[w][1] - WINDOWS[w][0]) + 1
        pre_col = f'{w}_NET_DOLLAR_SOLD'
        bench_col = 'BENCHMARK_NET_DOLLAR_SOLD'
        if pre_col not in treatment.columns or bench_col not in treatment.columns:
            continue

        pre_daily = treatment[pre_col] / w_days
        bench_daily = treatment[bench_col] / bench_days
        diff = pre_daily - bench_daily

        # Paired t-test
        t_stat, t_p = stats.ttest_rel(pre_daily, bench_daily)
        # Wilcoxon signed-rank (needs non-zero differences)
        nonzero_diff = diff[diff != 0]
        if len(nonzero_diff) > 10:
            w_stat, w_p = stats.wilcoxon(nonzero_diff)
        else:
            w_stat, w_p = np.nan, np.nan

        rows.append({
            'WINDOW': w,
            'N_PAIRS': len(treatment),
            'MEAN_PRE_DAILY': pre_daily.mean(),
            'MEAN_BENCH_DAILY': bench_daily.mean(),
            'MEAN_DIFF': diff.mean(),
            'T_STAT': t_stat,
            'T_PVALUE': t_p,
            'WILCOXON_STAT': w_stat if not np.isnan(w_stat) else 0.0,
            'WILCOXON_PVALUE': w_p if not np.isnan(w_p) else 1.0,
        })

    df = pd.DataFrame(rows)
    if not df.empty:
        df['T_BH_SIGNIFICANT'] = _benjamini_hochberg(df['T_PVALUE'].values).astype(int)
        df['WILCOXON_BH_SIGNIFICANT'] = _benjamini_hochberg(df['WILCOXON_PVALUE'].values).astype(int)
    return df


def compute_treatment_vs_control(panel):
    """OLS regression: net dollar sold ~ IS_TREATMENT + controls."""
    rows = []
    dep_var = 'PRE_FULL_NET_DOLLAR_SOLD'
    if dep_var not in panel.columns:
        return pd.DataFrame()

    pair_col = 'PAIR_ID' if 'PAIR_ID' in panel.columns else 'EVENT_ID'
    df = panel[[dep_var, 'IS_TREATMENT', 'LEAN', 'EVENT_ID', pair_col]].dropna(subset=[dep_var]).copy()
    df['IS_TREATMENT_INT'] = df['IS_TREATMENT'].astype(int)

    # Spec 1: OLS with clustering on matched pair (treatment + control share PAIR_ID)
    try:
        y = df[dep_var].astype(float)
        X = sm.add_constant(df[['IS_TREATMENT_INT']].astype(float))
        model = sm.OLS(y, X).fit(cov_type='cluster',
                                  cov_kwds={'groups': df[pair_col]})
        for var in model.params.index:
            rows.append({
                'SPECIFICATION': 'OLS_CLUSTER_EVENT',
                'VARIABLE': var,
                'COEFFICIENT': model.params[var],
                'STD_ERROR': model.bse[var],
                'T_STAT': model.tvalues[var],
                'P_VALUE': model.pvalues[var],
                'R_SQUARED': model.rsquared,
                'N_OBS': int(model.nobs),
            })
    except Exception as e:
        logger.warning("OLS treatment-vs-control failed: %s", e)

    # Spec 2: With lean dummies
    try:
        lean_dummies = pd.get_dummies(df['LEAN'], prefix='LEAN', drop_first=True).astype(float)
        X2 = sm.add_constant(pd.concat([df[['IS_TREATMENT_INT']].astype(float), lean_dummies], axis=1))
        model2 = sm.OLS(y, X2).fit(cov_type='cluster',
                                    cov_kwds={'groups': df[pair_col]})
        for var in model2.params.index:
            rows.append({
                'SPECIFICATION': 'OLS_CLUSTER_LEAN',
                'VARIABLE': var,
                'COEFFICIENT': model2.params[var],
                'STD_ERROR': model2.bse[var],
                'T_STAT': model2.tvalues[var],
                'P_VALUE': model2.pvalues[var],
                'R_SQUARED': model2.rsquared,
                'N_OBS': int(model2.nobs),
            })
    except Exception as e:
        logger.warning("OLS treatment-vs-control (lean) failed: %s", e)

    df_out = pd.DataFrame(rows)
    if not df_out.empty:
        df_out['BH_SIGNIFICANT'] = _benjamini_hochberg(df_out['P_VALUE'].values).astype(int)
    return df_out


def compute_leaning_analysis(panel):
    """Insider selling by political leaning."""
    treatment = panel[panel['IS_TREATMENT'] == True].copy()
    if treatment.empty:
        return pd.DataFrame()

    dep_var = 'PRE_FULL_NET_DOLLAR_SOLD'
    rows = []
    groups = []
    for lean, grp in treatment.groupby('LEAN'):
        vals = grp[dep_var].dropna()
        t_stat, t_p = (np.nan, np.nan)
        if len(vals) > 1 and vals.std() > 0:
            t_stat, t_p = stats.ttest_1samp(vals, 0)
        groups.append(vals.values)
        rows.append({
            'LEAN': lean,
            'N_EVENTS': len(grp),
            'MEAN_NET_DOLLAR_SOLD': vals.mean() if len(vals) > 0 else 0,
            'MEDIAN_NET_DOLLAR_SOLD': vals.median() if len(vals) > 0 else 0,
            'STD_NET_DOLLAR_SOLD': vals.std() if len(vals) > 0 else 0,
            'T_STAT_VS_ZERO': t_stat if not np.isnan(t_stat) else None,
            'P_VALUE_VS_ZERO': t_p if not np.isnan(t_p) else None,
        })

    # Kruskal-Wallis across leanings
    valid_groups = [g for g in groups if len(g) > 0]
    if len(valid_groups) >= 2:
        try:
            kw_stat, kw_p = stats.kruskal(*valid_groups)
        except Exception:
            kw_stat, kw_p = None, None
    else:
        kw_stat, kw_p = None, None

    df = pd.DataFrame(rows)
    df['KW_STAT'] = kw_stat
    df['KW_PVALUE'] = kw_p
    df['BH_SIGNIFICANT'] = 0
    if df['P_VALUE_VS_ZERO'].notna().any():
        pvals = df['P_VALUE_VS_ZERO'].fillna(1).values
        df['BH_SIGNIFICANT'] = _benjamini_hochberg(pvals).astype(int)
    return df


def compute_regime_interaction(panel, vix_df=None):
    """Test insider selling by VIX regime."""
    if vix_df is None or vix_df.empty:
        return pd.DataFrame()

    # Classify VIX regimes: Low (<15), Medium (15-25), High (>25)
    vix = vix_df[['DATE', 'VIX']].dropna().copy()
    vix['DATE'] = pd.to_datetime(vix['DATE'])
    vix['VIX_REGIME'] = pd.cut(vix['VIX'],
                                bins=[0, 15, 25, 100],
                                labels=['Low', 'Medium', 'High'])

    # Map each event to nearest prior trading day's VIX regime
    panel = panel.copy()
    panel['EVENT_DATE'] = pd.to_datetime(panel['EVENT_DATE'])
    vix = vix.sort_values('DATE')
    panel = panel.sort_values('EVENT_DATE')
    panel = pd.merge_asof(
        panel, vix[['DATE', 'VIX_REGIME']],
        left_on='EVENT_DATE', right_on='DATE',
        direction='backward', tolerance=pd.Timedelta(days=7)
    )
    panel = panel.drop(columns=['DATE'], errors='ignore')
    panel['VIX_MATCHED'] = panel['VIX_REGIME'].notna() & (panel['VIX_REGIME'].astype(str) != 'nan')
    panel['VIX_REGIME'] = panel['VIX_REGIME'].astype(str).replace('nan', 'Unknown')

    # Daily pre-event selling
    pre_days = abs(WINDOWS['PRE_FULL'][1] - WINDOWS['PRE_FULL'][0]) + 1
    panel['_pre_daily'] = panel['PRE_FULL_NET_DOLLAR_SOLD'] / pre_days

    rows = []
    tests = [
        ('NET_SELL_DAILY', '_pre_daily'),
    ]

    for test_name, col in tests:
        for regime, grp in panel.groupby('VIX_REGIME'):
            vals = grp[col].dropna()
            n = len(vals)
            if n < 2:
                t_stat, p_val = np.nan, np.nan
            else:
                t_stat, p_val = stats.ttest_1samp(vals, 0)
            rows.append({
                'TEST': f'REGIME_{regime}_VS_ZERO',
                'REGIME': regime,
                'N': n,
                'MEAN_NET_SELL_DAILY': vals.mean() if n > 0 else 0,
                'STD_NET_SELL_DAILY': vals.std() if n > 0 else 0,
                'T_STAT': t_stat if not np.isnan(t_stat) else None,
                'P_VALUE': p_val if not np.isnan(p_val) else None,
            })

    df = pd.DataFrame(rows)
    if not df.empty:
        pvals = df['P_VALUE'].fillna(1).values
        df['BH_SIGNIFICANT'] = _benjamini_hochberg(pvals).astype(int)
    return df


def compute_routine_vs_opportunistic(panel):
    """Compare routine vs opportunistic insider trading pre vs benchmark (treatment only)."""
    treatment = panel[panel['IS_TREATMENT'] == True]
    if treatment.empty:
        return pd.DataFrame()

    bench_days = abs(WINDOWS['BENCHMARK'][1] - WINDOWS['BENCHMARK'][0]) + 1
    pre_days = abs(WINDOWS['PRE_FULL'][1] - WINDOWS['PRE_FULL'][0]) + 1

    tests = [
        ('OPPORTUNISTIC_PRE_VS_BENCH', 'N_OPPORTUNISTIC'),
        ('ROUTINE_PRE_VS_BENCH', 'N_ROUTINE'),
        ('ALL_SELLS_PRE_VS_BENCH', 'NET_DOLLAR_SOLD'),
        ('OPP_DOLLAR_PRE_VS_BENCH', 'DOLLAR_SOLD'),
        ('SELL_RATIO_PRE_VS_BENCH', 'NET_SELL_RATIO'),
        ('INSIDERS_PRE_VS_BENCH', 'N_UNIQUE_INSIDERS'),
    ]

    rows = []
    for test_name, metric in tests:
        pre_col = f'PRE_FULL_{metric}'
        bench_col = f'BENCHMARK_{metric}'
        if pre_col not in treatment.columns or bench_col not in treatment.columns:
            continue

        pre_vals = treatment[pre_col].fillna(0)
        bench_vals = treatment[bench_col].fillna(0)

        if metric == 'NET_SELL_RATIO':
            # Ratio metric — already normalized, no day adjustment needed
            pre_daily = pre_vals
            bench_daily = bench_vals
        else:
            # Day-normalize all count and dollar metrics for apples-to-apples
            # comparison across windows of different lengths (PRE_FULL=180, BENCHMARK=185)
            pre_daily = pre_vals / pre_days
            bench_daily = bench_vals / bench_days

        diff = pre_daily - bench_daily
        t_stat, t_p = (np.nan, np.nan)
        if len(diff) > 1:
            t_stat, t_p = stats.ttest_rel(pre_daily, bench_daily)

        rows.append({
            'TEST': test_name,
            'N_PAIRS': len(treatment),
            'MEAN_PRE_DAILY': pre_daily.mean(),
            'MEAN_BENCH_DAILY': bench_daily.mean(),
            'MEAN_DIFF': diff.mean(),
            'T_STAT': t_stat if not np.isnan(t_stat) else None,
            'P_VALUE': t_p if not np.isnan(t_p) else None,
        })

    df = pd.DataFrame(rows)
    if not df.empty:
        pvals = df['P_VALUE'].fillna(1).values
        df['BH_SIGNIFICANT'] = _benjamini_hochberg(pvals).astype(int)
    return df


def compute_placebo_test(panel, n_iterations=500, seed=42):
    """Permutation test: shuffle treatment labels *within pairs*, compare selling.

    Treatment and control firms are matched pairs (PAIR_ID). We permute
    the treatment label within each pair so that the matched structure is
    preserved (each permuted draw has exactly one treatment and one control
    per pair). If PAIR_ID is unavailable, falls back to global permutation.
    """
    rng = np.random.RandomState(seed)
    pre_col = 'PRE_FULL_NET_DOLLAR_SOLD'
    bench_col = 'BENCHMARK_NET_DOLLAR_SOLD'
    if pre_col not in panel.columns or bench_col not in panel.columns:
        return pd.DataFrame()

    pre_days = abs(WINDOWS['PRE_FULL'][1] - WINDOWS['PRE_FULL'][0]) + 1
    bench_days = abs(WINDOWS['BENCHMARK'][1] - WINDOWS['BENCHMARK'][0]) + 1

    pre_daily = panel[pre_col] / pre_days
    bench_daily = panel[bench_col] / bench_days
    diff = (pre_daily - bench_daily).values

    treatment_mask = panel['IS_TREATMENT'].values
    observed = diff[treatment_mask].mean() - diff[~treatment_mask].mean()

    # Within-pair permutation: flip treatment label within each PAIR_ID
    has_pairs = 'PAIR_ID' in panel.columns
    if has_pairs:
        pair_ids = panel['PAIR_ID'].values
        unique_pairs = np.unique(pair_ids[~pd.isna(pair_ids)])
        if len(unique_pairs) == 0:
            logger.warning("Placebo: PAIR_ID column exists but no valid pairs; falling back to global")
            has_pairs = False

    placebo_stats = []
    for _ in range(n_iterations):
        if has_pairs and len(unique_pairs) > 0:
            shuffled = treatment_mask.copy()
            for pid in unique_pairs:
                pair_idx = np.where(pair_ids == pid)[0]
                if len(pair_idx) == 2:
                    if rng.random() < 0.5:
                        shuffled[pair_idx[0]], shuffled[pair_idx[1]] = \
                            shuffled[pair_idx[1]], shuffled[pair_idx[0]]
        else:
            shuffled = rng.permutation(treatment_mask)
        placebo_stat = diff[shuffled].mean() - diff[~shuffled].mean()
        placebo_stats.append(placebo_stat)

    placebo_stats = np.array(placebo_stats)
    percentile = (placebo_stats < observed).mean() * 100
    empirical_p = (np.abs(placebo_stats) >= np.abs(observed)).mean()

    return pd.DataFrame([{
        'TEST': 'PLACEBO_PERMUTATION',
        'OBSERVED_STAT': observed,
        'PLACEBO_MEAN': placebo_stats.mean(),
        'PLACEBO_STD': placebo_stats.std(),
        'PERCENTILE_ONE_SIDED': percentile,
        'EMPIRICAL_P_TWO_SIDED': empirical_p,
        'N_ITERATIONS': n_iterations,
        'N_FIRMS': panel['TICKER'].nunique(),
    }])


def compute_acceleration_test(panel):
    """Jonckheere-Terpstra trend test for monotonic increase far → mid → near.

    Note: JT assumes independent groups, but these are repeated measures on
    the same firms across consecutive windows. Results should be interpreted
    as approximate; a Page's trend test or mixed-effects model with linear
    time coefficient would be more appropriate for paired data.
    """
    windows = ['PRE_FAR', 'PRE_MID', 'PRE_NEAR']

    tests_config = [
        ('JT_TREND_ALL', 'NET_DOLLAR_SOLD', None),
        ('JT_TREND_TREATMENT', 'NET_DOLLAR_SOLD', True),
        ('JT_TREND_CONTROL', 'NET_DOLLAR_SOLD', False),
        ('JT_TREND_SELLS', 'N_SELLS', None),
        ('JT_TREND_DOLLAR_SOLD', 'DOLLAR_SOLD', None),
        ('JT_TREND_OPP', 'N_OPPORTUNISTIC', None),
        ('JT_TREND_INSIDERS', 'N_UNIQUE_INSIDERS', None),
        ('JT_TREND_SELL_RATIO', 'NET_SELL_RATIO', None),
        ('JT_TREND_CONSERVATIVE', 'NET_DOLLAR_SOLD', 'Conservative'),
        ('JT_TREND_LIBERAL', 'NET_DOLLAR_SOLD', 'Liberal'),
        ('JT_TREND_MIXED', 'NET_DOLLAR_SOLD', 'Mixed'),
        ('JT_TREND_NET_SHARES', 'NET_SHARES_SOLD', None),
    ]

    rows = []
    for test_name, metric, filter_val in tests_config:
        sub = panel.copy()
        if filter_val is True:
            sub = sub[sub['IS_TREATMENT'] == True]
        elif filter_val is False:
            sub = sub[sub['IS_TREATMENT'] == False]
        elif isinstance(filter_val, str):
            sub = sub[sub['LEAN'] == filter_val]

        cols = [f'{w}_{metric}' for w in windows]
        present = [c for c in cols if c in sub.columns]
        if len(present) < 3:
            continue

        # Normalize by window days
        day_counts = [abs(WINDOWS[w][1] - WINDOWS[w][0]) + 1 for w in windows]
        groups = []
        for col, days in zip(present, day_counts):
            if metric in ('NET_DOLLAR_SOLD', 'DOLLAR_SOLD', 'NET_SHARES_SOLD'):
                groups.append(sub[col].fillna(0) / days)
            else:
                groups.append(sub[col].fillna(0))

        # Jonckheere-Terpstra: test for ordered increase across independent groups
        # JT statistic = sum of Mann-Whitney U statistics for all ordered pairs
        try:
            jt_stat = 0
            for i_g in range(len(groups)):
                for j_g in range(i_g + 1, len(groups)):
                    u_stat, _ = stats.mannwhitneyu(
                        groups[i_g].values, groups[j_g].values, alternative='less'
                    )
                    jt_stat += u_stat
            # Normal approximation for p-value
            n_list = [len(g) for g in groups]
            n_total = sum(n_list)
            jt_mean = (n_total ** 2 - sum(n ** 2 for n in n_list)) / 4
            jt_var = (n_total ** 2 * (2 * n_total + 3) -
                      sum(n ** 2 * (2 * n + 3) for n in n_list)) / 72
            if jt_var > 0:
                jt_z = (jt_stat - jt_mean) / np.sqrt(jt_var)
                jt_p = 1 - stats.norm.cdf(jt_z)  # one-sided (increasing)
            else:
                jt_p = 1.0
        except Exception:
            # Fallback: Kruskal-Wallis as approximation
            try:
                jt_stat, jt_p = stats.kruskal(*[g.values for g in groups])
            except Exception:
                jt_stat, jt_p = 0, 0.5

        means = [g.mean() for g in groups]
        monotonic = all(means[i] <= means[i+1] for i in range(len(means)-1))

        rows.append({
            'TEST': test_name,
            'N': len(sub),
            'N_WINDOWS': len(present),
            'JT_STAT': jt_stat,
            'JT_PVALUE': jt_p,
            'MEAN_FAR': means[0],
            'MEAN_MID': means[1],
            'MEAN_NEAR': means[2],
            'MONOTONIC_INCREASE': monotonic,
        })

    df = pd.DataFrame(rows)
    if not df.empty:
        pvals = df['JT_PVALUE'].fillna(1).values
        df['BH_SIGNIFICANT'] = _benjamini_hochberg(pvals).astype(int)
    return df


def compute_car_insider_regression(panel):
    """Regress post-event CAR on pre-event insider selling."""
    if 'CAR_POST' not in panel.columns:
        return pd.DataFrame()

    valid = panel.dropna(subset=['CAR_POST', 'PRE_FULL_NET_DOLLAR_SOLD']).copy()
    if len(valid) < 10:
        return pd.DataFrame()

    rows = []
    # Cluster on PAIR_ID if available and complete (matched pairs share shocks)
    has_pairs = 'PAIR_ID' in valid.columns and valid['PAIR_ID'].notna().all()
    cluster_kwds = ({'groups': valid['PAIR_ID']} if has_pairs
                    else {'groups': valid['TICKER']})

    # Spec 1: CAR ~ selling
    try:
        y = valid['CAR_POST'].astype(float)
        X = sm.add_constant(valid[['PRE_FULL_NET_DOLLAR_SOLD']].astype(float))
        model = sm.OLS(y, X).fit(cov_type='cluster', cov_kwds=cluster_kwds)
        for var in model.params.index:
            rows.append({
                'SPECIFICATION': 'CAR_VS_SELLING',
                'VARIABLE': var,
                'COEFFICIENT': model.params[var],
                'STD_ERROR': model.bse[var],
                'T_STAT': model.tvalues[var],
                'P_VALUE': model.pvalues[var],
                'R_SQUARED': model.rsquared,
                'N_OBS': int(model.nobs),
            })
    except Exception as e:
        logger.warning("CAR regression failed: %s", e)

    # Spec 2: CAR ~ selling + treatment + lean
    try:
        valid2 = valid.copy().reset_index(drop=True)
        valid2['IS_TREATMENT_INT'] = valid2['IS_TREATMENT'].astype(int)
        lean_dummies = pd.get_dummies(valid2['LEAN'], prefix='LEAN', drop_first=True).astype(float)
        lean_dummies = lean_dummies.reset_index(drop=True)
        X2 = sm.add_constant(pd.concat([
            valid2[['PRE_FULL_NET_DOLLAR_SOLD', 'IS_TREATMENT_INT']].astype(float),
            lean_dummies
        ], axis=1))
        y2 = valid2['CAR_POST'].astype(float)
        cluster_kwds2 = ({'groups': valid2['PAIR_ID']} if has_pairs
                         else {'groups': valid2['TICKER']})
        model2 = sm.OLS(y2, X2).fit(cov_type='cluster', cov_kwds=cluster_kwds2)
        for var in model2.params.index:
            rows.append({
                'SPECIFICATION': 'CAR_VS_SELLING_CONTROLS',
                'VARIABLE': var,
                'COEFFICIENT': model2.params[var],
                'STD_ERROR': model2.bse[var],
                'T_STAT': model2.tvalues[var],
                'P_VALUE': model2.pvalues[var],
                'R_SQUARED': model2.rsquared,
                'N_OBS': int(model2.nobs),
            })
    except Exception as e:
        logger.warning("CAR regression (controls) failed: %s", e)

    return pd.DataFrame(rows)


def compute_information_gradient(panel):
    """Information gradient: how selling intensifies approaching the event.

    Reports gradient for ALL firms, TREATMENT only, and CONTROL only.
    The information-asymmetry story should be sharpest in treatment firms;
    the control gradient serves as a comparison baseline.
    """
    windows = ['BENCHMARK', 'PRE_FAR', 'PRE_MID', 'PRE_NEAR', 'POST']
    metrics = ['NET_DOLLAR_SOLD', 'N_SELLS', 'N_UNIQUE_INSIDERS', 'NET_SELL_RATIO']

    groups = [('ALL', panel)]
    if 'IS_TREATMENT' in panel.columns:
        groups.append(('TREATMENT', panel[panel['IS_TREATMENT'] == True]))
        groups.append(('CONTROL', panel[panel['IS_TREATMENT'] == False]))

    rows = []
    for group_name, sub in groups:
        for metric in metrics:
            for w in windows:
                col = f'{w}_{metric}'
                if col not in sub.columns:
                    continue
                days = abs(WINDOWS[w][1] - WINDOWS[w][0]) + 1
                vals = sub[col].fillna(0)
                if metric in ('NET_DOLLAR_SOLD', 'N_SELLS', 'N_UNIQUE_INSIDERS'):
                    daily = vals / days
                else:
                    daily = vals
                rows.append({
                    'GROUP': group_name,
                    'METRIC': metric,
                    'WINDOW': w,
                    'MEAN': daily.mean(),
                    'MEDIAN': daily.median(),
                    'STD': daily.std(),
                    'N': len(vals),
                })

    return pd.DataFrame(rows)


# ═══════════════════════════════════════════════════════════════════════
# ADDITIONAL ANALYSES (12–23)
# ═══════════════════════════════════════════════════════════════════════

def compute_diff_in_diff(panel):
    """Difference-in-differences: IS_TREATMENT × IS_POST interaction.

    Stacks the panel into long form with pre (PRE_FULL) and post (POST)
    windows, then regresses raw daily net dollar sold on treatment, post,
    and their interaction. The regression structure absorbs baseline
    differences via IS_TREATMENT and IS_POST main effects.
    """
    pre_days = abs(WINDOWS['PRE_FULL'][1] - WINDOWS['PRE_FULL'][0]) + 1
    post_days = abs(WINDOWS['POST'][1] - WINDOWS['POST'][0]) + 1

    required = ['PRE_FULL_NET_DOLLAR_SOLD', 'POST_NET_DOLLAR_SOLD',
                'IS_TREATMENT', 'TICKER', 'EVENT_DATE']
    if not all(c in panel.columns for c in required):
        return pd.DataFrame()

    # Stack pre and post into long form using raw daily selling (melt for speed)
    wide = panel[['TICKER', 'EVENT_DATE', 'IS_TREATMENT',
                   'PRE_FULL_NET_DOLLAR_SOLD', 'POST_NET_DOLLAR_SOLD']].copy()
    wide['IS_TREATMENT'] = wide['IS_TREATMENT'].astype(int)
    wide['PRE'] = wide['PRE_FULL_NET_DOLLAR_SOLD'] / pre_days
    wide['POST'] = wide['POST_NET_DOLLAR_SOLD'] / post_days
    long_df = wide.melt(
        id_vars=['TICKER', 'EVENT_DATE', 'IS_TREATMENT'],
        value_vars=['PRE', 'POST'],
        var_name='_period', value_name='NET_SELL_DAILY',
    )
    long_df['IS_POST'] = (long_df['_period'] == 'POST').astype(int)
    long_df = long_df.drop(columns=['_period']).dropna(subset=['NET_SELL_DAILY'])

    # Ensure balanced panel: each firm-event must have exactly 2 rows (pre + post)
    firm_event_counts = long_df.groupby(['TICKER', 'EVENT_DATE']).size()
    unbalanced = firm_event_counts[firm_event_counts != 2]
    if not unbalanced.empty:
        logger.warning("DiD: dropping %d unbalanced firm-events", len(unbalanced))
        balanced_idx = firm_event_counts[firm_event_counts == 2].index
        long_df = long_df.set_index(['TICKER', 'EVENT_DATE'])
        long_df = long_df.loc[long_df.index.isin(balanced_idx)].reset_index()

    if len(long_df) < 20:
        return pd.DataFrame()

    long_df['TREAT_X_POST'] = long_df['IS_TREATMENT'] * long_df['IS_POST']

    rows = []
    # Spec 1: Simple DiD
    try:
        y = long_df['NET_SELL_DAILY'].astype(float)
        X = sm.add_constant(long_df[['IS_TREATMENT', 'IS_POST', 'TREAT_X_POST']].astype(float))
        model = sm.OLS(y, X).fit(cov_type='cluster',
                                  cov_kwds={'groups': long_df['TICKER']})
        for var in model.params.index:
            rows.append({
                'SPECIFICATION': 'DID_SIMPLE',
                'VARIABLE': var,
                'COEFFICIENT': model.params[var],
                'STD_ERROR': model.bse[var],
                'T_STAT': model.tvalues[var],
                'P_VALUE': model.pvalues[var],
                'R_SQUARED': model.rsquared,
                'N_OBS': int(model.nobs),
            })
    except Exception as e:
        logger.warning("DiD simple failed: %s", e)

    # Spec 2: DiD with event-date fixed effects
    # With only 2 rows per firm (pre + post), firm FE collapses to a
    # before-after on treatment only. Event-date FE is more defensible:
    # it absorbs time-varying shocks common to all firms on a given date.
    n_dates = long_df['EVENT_DATE'].nunique()
    n_obs_did = len(long_df)
    # Guard: need at least 10 residual df after absorbing date dummies + 3 regressors + const
    if (n_dates - 1) + 4 >= n_obs_did - 10:
        logger.warning("DiD date FE: too many date dummies (%d) for %d obs; skipping", n_dates, n_obs_did)
    else:
        try:
            date_dummies = pd.get_dummies(
                long_df['EVENT_DATE'].astype(str), prefix='DATE', drop_first=True
            ).astype(float)
            X2 = sm.add_constant(pd.concat([
                long_df[['IS_TREATMENT', 'IS_POST', 'TREAT_X_POST']].astype(float),
                date_dummies
            ], axis=1))
            y2 = long_df['NET_SELL_DAILY'].astype(float)
            model2 = sm.OLS(y2, X2).fit(cov_type='cluster',
                                         cov_kwds={'groups': long_df['TICKER']})
            # Only report the main coefficients, not date dummies
            for var in ['const', 'IS_TREATMENT', 'IS_POST', 'TREAT_X_POST']:
                if var in model2.params.index:
                    rows.append({
                        'SPECIFICATION': 'DID_DATE_FE',
                        'VARIABLE': var,
                        'COEFFICIENT': model2.params[var],
                        'STD_ERROR': model2.bse[var],
                        'T_STAT': model2.tvalues[var],
                        'P_VALUE': model2.pvalues[var],
                        'R_SQUARED': model2.rsquared,
                        'N_OBS': int(model2.nobs),
                    })
        except Exception as e:
            logger.warning("DiD date FE failed: %s", e)

    df = pd.DataFrame(rows)
    if not df.empty:
        df['BH_SIGNIFICANT'] = _benjamini_hochberg(df['P_VALUE'].values).astype(int)
    return df


def compute_10b5_1_filter(panel, form4):
    """Sensitivity check: approximate 10b5-1 (scheduled) trades using regularity proxy.

    SEC Form 4 footnotes contain 10b5-1 plan disclosures, but they are not
    parsed from XML by our downloader. We approximate: an insider is flagged
    as likely 10b5-1 if they have regular, periodic transactions (trades in
    >= 4 of last 8 quarters with consistent timing, i.e., trades in similar
    calendar weeks). Results are reported with and without 10b5-1 trades.
    """
    pre_col = 'PRE_FULL_NET_DOLLAR_SOLD'
    bench_col = 'BENCHMARK_NET_DOLLAR_SOLD'
    if pre_col not in panel.columns or bench_col not in panel.columns:
        return pd.DataFrame()

    pre_days = abs(WINDOWS['PRE_FULL'][1] - WINDOWS['PRE_FULL'][0]) + 1
    bench_days = abs(WINDOWS['BENCHMARK'][1] - WINDOWS['BENCHMARK'][0]) + 1

    # Identify tickers with likely 10b5-1 insiders
    plan_tickers = set()
    for ticker in panel['TICKER'].unique():
        ticker_txns = form4[form4['ticker'] == ticker]
        for owner in ticker_txns['owner_name'].unique():
            trades = ticker_txns[ticker_txns['owner_name'] == owner]['transaction_date'].dropna()
            if len(trades) < 8:
                continue
            quarters = trades.dt.to_period('Q').nunique()
            if quarters < 4:
                continue
            # Check timing regularity: std of week-within-quarter < 3 weeks
            # Map each trade to its week offset within its quarter (0-12)
            q_start_month = ((trades.dt.month - 1) // 3) * 3 + 1
            day_of_quarter = (trades.dt.month - q_start_month) * 30 + trades.dt.day
            week_in_q = day_of_quarter // 7
            if week_in_q.std() < 3:
                plan_tickers.add(ticker)
                break

    panel = panel.copy()
    panel['LIKELY_10B5_1'] = panel['TICKER'].isin(plan_tickers).astype(int)

    rows = []
    for label, sub in [('ALL', panel),
                        ('EXCL_10B5_1', panel[panel['LIKELY_10B5_1'] == 0]),
                        ('ONLY_10B5_1', panel[panel['LIKELY_10B5_1'] == 1])]:
        treatment = sub[sub['IS_TREATMENT'] == True]
        if len(treatment) < 5:
            continue
        pre_daily = treatment[pre_col] / pre_days
        bench_daily = treatment[bench_col] / bench_days
        diff = pre_daily - bench_daily
        t_stat, t_p = (np.nan, np.nan)
        if len(diff) > 1:
            t_stat, t_p = stats.ttest_rel(pre_daily, bench_daily)
        rows.append({
            'FILTER': label,
            'N_EVENTS': len(treatment),
            'N_10B5_1_TICKERS': len(plan_tickers),
            'MEAN_DIFF': diff.mean(),
            'T_STAT': t_stat if not np.isnan(t_stat) else None,
            'P_VALUE': t_p if not np.isnan(t_p) else None,
        })

    df = pd.DataFrame(rows)
    if not df.empty and df['P_VALUE'].notna().any():
        df['BH_SIGNIFICANT'] = _benjamini_hochberg(df['P_VALUE'].fillna(1).values).astype(int)
    return df


def compute_match_quality(panel, events, stock_data=None):
    """Post-hoc validation of treatment-control balance on insider activity.

    This checks covariate balance on insider trading metrics in the
    benchmark period (not the original matching covariates like size,
    B/M, industry). Reports standardized mean differences (SMD).
    Threshold |SMD| < 0.25 per Rubin (2001); <0.1 is ideal.
    """
    treat = panel[panel['IS_TREATMENT'] == True]
    ctrl = panel[panel['IS_TREATMENT'] == False]

    if treat.empty or ctrl.empty:
        return pd.DataFrame()

    rows = []
    covariates = [
        ('BENCHMARK_NET_DOLLAR_SOLD', 'Benchmark Net Dollar Sold'),
        ('BENCHMARK_N_TRANSACTIONS', 'Benchmark N Transactions'),
        ('BENCHMARK_N_UNIQUE_INSIDERS', 'Benchmark N Unique Insiders'),
        ('BENCHMARK_NET_SELL_RATIO', 'Benchmark Net Sell Ratio'),
        ('BENCHMARK_N_OPPORTUNISTIC', 'Benchmark N Opportunistic'),
    ]

    for col, label in covariates:
        if col not in panel.columns:
            continue
        t_vals = treat[col].fillna(0)
        c_vals = ctrl[col].fillna(0)
        t_mean = t_vals.mean()
        c_mean = c_vals.mean()
        pooled_std = np.sqrt((t_vals.std() ** 2 + c_vals.std() ** 2) / 2)
        smd = (t_mean - c_mean) / pooled_std if pooled_std > 0 else 0
        # Two-sample t-test
        t_stat, t_p = stats.ttest_ind(t_vals, c_vals, equal_var=False)
        rows.append({
            'COVARIATE': label,
            'TREATMENT_MEAN': t_mean,
            'CONTROL_MEAN': c_mean,
            'TREATMENT_STD': t_vals.std(),
            'CONTROL_STD': c_vals.std(),
            'SMD': smd,
            'ABS_SMD': abs(smd),
            'BALANCED': 1 if abs(smd) < 0.25 else 0,
            'T_STAT': t_stat,
            'P_VALUE': t_p,
        })

    # Add daily volume if stock data available
    if stock_data is not None and not stock_data.empty and 'VOLUME' in stock_data.columns:
        vol_lookup = {}
        for ticker in panel['TICKER'].unique():
            t_stock = stock_data[stock_data['TICKER'] == ticker]
            if not t_stock.empty:
                vol_lookup[ticker] = t_stock['VOLUME'].mean()
        panel_vol = panel.copy()
        panel_vol['AVG_DAILY_VOLUME'] = panel_vol['TICKER'].map(vol_lookup)
        t_vol = panel_vol.loc[panel_vol['IS_TREATMENT'] == True, 'AVG_DAILY_VOLUME'].dropna()
        c_vol = panel_vol.loc[panel_vol['IS_TREATMENT'] == False, 'AVG_DAILY_VOLUME'].dropna()
        if len(t_vol) > 0 and len(c_vol) > 0:
            pooled = np.sqrt((t_vol.std() ** 2 + c_vol.std() ** 2) / 2)
            smd = (t_vol.mean() - c_vol.mean()) / pooled if pooled > 0 else 0
            t_stat, t_p = stats.ttest_ind(t_vol, c_vol, equal_var=False)
            rows.append({
                'COVARIATE': 'Avg Daily Volume',
                'TREATMENT_MEAN': t_vol.mean(),
                'CONTROL_MEAN': c_vol.mean(),
                'TREATMENT_STD': t_vol.std(),
                'CONTROL_STD': c_vol.std(),
                'SMD': smd,
                'ABS_SMD': abs(smd),
                'BALANCED': 1 if abs(smd) < 0.25 else 0,
                'T_STAT': t_stat,
                'P_VALUE': t_p,
            })

    df = pd.DataFrame(rows)
    if not df.empty:
        n_balanced = df['BALANCED'].sum()
        logger.info("  Match quality: %d/%d covariates balanced (|SMD| < 0.25)",
                    n_balanced, len(df))
    return df


def compute_event_clustering(panel):
    """Treatment vs control regression with date-clustered standard errors.

    Culture war events cluster in time. This re-runs the main specification
    with SEs clustered at the event-date level instead of event-ID.
    """
    dep_var = 'PRE_FULL_NET_DOLLAR_SOLD'
    if dep_var not in panel.columns:
        return pd.DataFrame()

    df = panel[[dep_var, 'IS_TREATMENT', 'EVENT_DATE', 'TICKER']].dropna(subset=[dep_var]).copy()
    df['IS_TREATMENT_INT'] = df['IS_TREATMENT'].astype(int)
    # Create date cluster: group events within 7 days of the cluster anchor date.
    # Anchor is the first event in each cluster (not chain-linked from prior event).
    df = df.sort_values('EVENT_DATE')
    df['EVENT_DATE_TS'] = pd.to_datetime(df['EVENT_DATE'])
    date_clusters = []
    cluster_id = 0
    anchor_date = None
    for _, row in df.iterrows():
        if anchor_date is None or (row['EVENT_DATE_TS'] - anchor_date).days > 7:
            cluster_id += 1
            anchor_date = row['EVENT_DATE_TS']
        date_clusters.append(cluster_id)
    df['DATE_CLUSTER'] = date_clusters

    rows = []
    cluster_types = [
        ('CLUSTER_EVENT_DATE', df['EVENT_DATE'].astype(str)),
        ('CLUSTER_DATE_7DAY', df['DATE_CLUSTER']),
        ('CLUSTER_TICKER', df['TICKER']),
        ('TWOWAY_DATE_TICKER', None),  # two-way clustering
    ]

    for spec_name, cluster_var in cluster_types:
        try:
            y = df[dep_var].astype(float)
            X = sm.add_constant(df[['IS_TREATMENT_INT']].astype(float))
            if spec_name == 'TWOWAY_DATE_TICKER':
                # Two-way clustering: Cameron, Gelbach & Miller (2011)
                # V = V_date + V_ticker - V_{date∩ticker}
                # Intersection of date × ticker is the individual observation
                model_d = sm.OLS(y, X).fit(cov_type='cluster',
                                            cov_kwds={'groups': df['EVENT_DATE'].astype(str)})
                model_t = sm.OLS(y, X).fit(cov_type='cluster',
                                            cov_kwds={'groups': df['TICKER']})
                # Intersection: each unique (date, ticker) pair
                intersection_id = df['EVENT_DATE'].astype(str) + '_' + df['TICKER']
                model_dt = sm.OLS(y, X).fit(cov_type='cluster',
                                             cov_kwds={'groups': intersection_id})
                # V_twoway = V_date + V_ticker - V_intersection
                var_twoway = model_d.cov_params() + model_t.cov_params() - model_dt.cov_params()
                # Enforce numerical symmetry before eigendecomposition
                var_twoway = (var_twoway + var_twoway.T) / 2
                # Ensure positive semi-definiteness (can fail with small clusters)
                eigvals, eigvecs = np.linalg.eigh(var_twoway)
                psd_adjusted = bool(np.any(eigvals < 0))
                if psd_adjusted:
                    # Use abs(eigvals) rather than max(eigvals, 0): the conservative
                    # convention per CGM (2011, eq. 2.13). Clipping to zero shrinks
                    # the variance matrix and inflates t-stats; abs preserves magnitude.
                    logger.warning("Two-way cluster VCV not PSD (min eigval=%.4e); "
                                   "using abs(eigenvalues) per CGM convention", eigvals.min())
                    var_twoway = eigvecs @ np.diag(np.abs(eigvals)) @ eigvecs.T
                se_twoway = np.sqrt(np.diag(var_twoway))
                for j, var in enumerate(model_d.params.index):
                    coef = model_d.params[var]
                    se = se_twoway[j]
                    t_val = coef / se if se > 0 else 0
                    p_val = 2 * (1 - stats.norm.cdf(abs(t_val)))
                    rows.append({
                        'SPECIFICATION': spec_name,
                        'VARIABLE': var,
                        'COEFFICIENT': coef,
                        'STD_ERROR': se,
                        'T_STAT': t_val,
                        'P_VALUE': p_val,
                        'N_CLUSTERS': f"{df['EVENT_DATE'].nunique()}+{df['TICKER'].nunique()}",
                        'N_OBS': int(len(df)),
                        'PSD_ADJUSTED': psd_adjusted,
                        'N_UNIQUE_TICKERS': int(df['TICKER'].nunique()),
                        'N_UNIQUE_DATES': int(df['EVENT_DATE'].nunique()),
                    })
            else:
                model = sm.OLS(y, X).fit(cov_type='cluster',
                                          cov_kwds={'groups': cluster_var})
                for var in model.params.index:
                    rows.append({
                        'SPECIFICATION': spec_name,
                        'VARIABLE': var,
                        'COEFFICIENT': model.params[var],
                        'STD_ERROR': model.bse[var],
                        'T_STAT': model.tvalues[var],
                        'P_VALUE': model.pvalues[var],
                        'N_CLUSTERS': str(cluster_var.nunique()),
                        'N_OBS': int(model.nobs),
                    })
        except Exception as e:
            logger.warning("Event clustering (%s) failed: %s", spec_name, e)

    df_out = pd.DataFrame(rows)
    if not df_out.empty:
        df_out['BH_SIGNIFICANT'] = _benjamini_hochberg(df_out['P_VALUE'].values).astype(int)
    return df_out


def compute_intensive_extensive_margin(panel):
    """Decompose abnormal selling into intensive and extensive margins.

    Extensive: do MORE insiders trade pre-event vs benchmark?
    Intensive: do the SAME insiders trade MORE ($ per insider)?
    """
    pre_days = abs(WINDOWS['PRE_FULL'][1] - WINDOWS['PRE_FULL'][0]) + 1
    bench_days = abs(WINDOWS['BENCHMARK'][1] - WINDOWS['BENCHMARK'][0]) + 1
    treatment = panel[panel['IS_TREATMENT'] == True].copy()

    if treatment.empty:
        return pd.DataFrame()

    rows = []

    # Extensive margin: N_UNIQUE_INSIDERS
    for metric, label in [('N_UNIQUE_INSIDERS', 'EXTENSIVE_INSIDERS'),
                           ('N_SELLS', 'EXTENSIVE_SELL_COUNT'),
                           ('N_OPPORTUNISTIC', 'EXTENSIVE_OPPORTUNISTIC')]:
        pre_col = f'PRE_FULL_{metric}'
        bench_col = f'BENCHMARK_{metric}'
        if pre_col not in treatment.columns or bench_col not in treatment.columns:
            continue
        pre_daily = treatment[pre_col] / pre_days
        bench_daily = treatment[bench_col] / bench_days
        diff = pre_daily - bench_daily
        t_stat, t_p = (np.nan, np.nan)
        if len(diff) > 1:
            t_stat, t_p = stats.ttest_rel(pre_daily, bench_daily)
        rows.append({
            'MARGIN': label,
            'N': len(treatment),
            'MEAN_PRE_DAILY': pre_daily.mean(),
            'MEAN_BENCH_DAILY': bench_daily.mean(),
            'MEAN_DIFF': diff.mean(),
            'PCT_INCREASE': (diff.mean() / bench_daily.mean() * 100) if bench_daily.mean() != 0 else np.nan,
            'T_STAT': t_stat if not np.isnan(t_stat) else None,
            'P_VALUE': t_p if not np.isnan(t_p) else None,
        })

    # Intensive margin: NET_DOLLAR_SOLD / N_UNIQUE_INSIDERS ($ per insider)
    for metric, label in [('NET_DOLLAR_SOLD', 'INTENSIVE_DOLLAR_PER_INSIDER'),
                           ('SHARES_SOLD', 'INTENSIVE_SHARES_PER_INSIDER')]:
        pre_col = f'PRE_FULL_{metric}'
        bench_col = f'BENCHMARK_{metric}'
        pre_ins = 'PRE_FULL_N_UNIQUE_INSIDERS'
        bench_ins = 'BENCHMARK_N_UNIQUE_INSIDERS'
        if not all(c in treatment.columns for c in [pre_col, bench_col, pre_ins, bench_ins]):
            continue
        # Per-insider, per-day
        pre_per_ins = np.where(treatment[pre_ins] > 0,
                               treatment[pre_col] / treatment[pre_ins] / pre_days, 0)
        bench_per_ins = np.where(treatment[bench_ins] > 0,
                                  treatment[bench_col] / treatment[bench_ins] / bench_days, 0)
        diff = pre_per_ins - bench_per_ins
        t_stat, t_p = (np.nan, np.nan)
        if len(diff) > 1:
            t_stat, t_p = stats.ttest_rel(pre_per_ins, bench_per_ins)
        rows.append({
            'MARGIN': label,
            'N': len(treatment),
            'MEAN_PRE_DAILY': float(np.mean(pre_per_ins)),
            'MEAN_BENCH_DAILY': float(np.mean(bench_per_ins)),
            'MEAN_DIFF': float(np.mean(diff)),
            'PCT_INCREASE': (np.mean(diff) / np.mean(bench_per_ins) * 100)
                            if np.mean(bench_per_ins) != 0 else np.nan,
            'T_STAT': t_stat if not np.isnan(t_stat) else None,
            'P_VALUE': t_p if not np.isnan(t_p) else None,
        })

    df = pd.DataFrame(rows)
    if not df.empty and df['P_VALUE'].notna().any():
        df['BH_SIGNIFICANT'] = _benjamini_hochberg(df['P_VALUE'].fillna(1).values).astype(int)
    return df


def compute_abnormal_volume_ratio(panel, stock_data=None):
    """Normalize insider selling by firm's average daily trading volume.

    Creates ABNORMAL_VOLUME_RATIO = pre-event daily insider $ sold /
    benchmark average daily stock volume (in $). This controls for firm
    size differences (Lakonishok & Lee, 2001).
    """
    if stock_data is None or stock_data.empty:
        return pd.DataFrame()

    pre_days = abs(WINDOWS['PRE_FULL'][1] - WINDOWS['PRE_FULL'][0]) + 1
    bench_days = abs(WINDOWS['BENCHMARK'][1] - WINDOWS['BENCHMARK'][0]) + 1

    # Compute benchmark-period average daily dollar volume per event
    if 'VOLUME' not in stock_data.columns or 'CLOSE' not in stock_data.columns:
        return pd.DataFrame()
    stock_data = stock_data.copy()
    stock_data['DATE'] = pd.to_datetime(stock_data['DATE'])
    stock_data['DOLLAR_VOL'] = stock_data['VOLUME'] * stock_data['CLOSE']

    vol_lookup = {}
    for ticker, grp in panel.groupby('TICKER'):
        t_stock = stock_data[stock_data['TICKER'] == ticker]
        if t_stock.empty:
            continue
        for _, row in grp.iterrows():
            event_date = pd.Timestamp(row['EVENT_DATE'])
            bench_start = event_date + pd.Timedelta(days=WINDOWS['BENCHMARK'][0])
            bench_end = event_date + pd.Timedelta(days=WINDOWS['BENCHMARK'][1])
            mask = (t_stock['DATE'] >= bench_start) & (t_stock['DATE'] <= bench_end)
            bench_vol = t_stock.loc[mask, 'DOLLAR_VOL']
            if not bench_vol.empty:
                vol_lookup[(ticker, event_date)] = bench_vol.mean()

    if not vol_lookup:
        logger.warning("No stock volume data available for abnormal volume ratio")
        return pd.DataFrame()

    treatment = panel[panel['IS_TREATMENT'] == True].copy()
    rows = []
    avr_values = []

    for _, row in treatment.iterrows():
        key = (row['TICKER'], pd.Timestamp(row['EVENT_DATE']))
        bench_vol = vol_lookup.get(key)
        if bench_vol is None or bench_vol <= 0:
            continue
        pre_raw = row.get('PRE_FULL_NET_DOLLAR_SOLD', 0)
        bench_raw = row.get('BENCHMARK_NET_DOLLAR_SOLD', 0)
        pre_daily = (0 if pd.isna(pre_raw) else pre_raw) / pre_days
        bench_daily_sell = (0 if pd.isna(bench_raw) else bench_raw) / bench_days
        avr_pre = pre_daily / bench_vol
        avr_bench = bench_daily_sell / bench_vol
        avr_values.append({
            'TICKER': row['TICKER'],
            'AVR_PRE': avr_pre,
            'AVR_BENCH': avr_bench,
            'AVR_DIFF': avr_pre - avr_bench,
            'BENCH_DOLLAR_VOL': bench_vol,
        })

    if not avr_values:
        return pd.DataFrame()

    avr_df = pd.DataFrame(avr_values)
    # Test if AVR is significantly different pre vs benchmark
    t_stat, t_p = stats.ttest_rel(avr_df['AVR_PRE'], avr_df['AVR_BENCH'])
    nonzero = avr_df['AVR_DIFF'][avr_df['AVR_DIFF'] != 0]
    w_stat, w_p = (np.nan, np.nan)
    if len(nonzero) > 10:
        w_stat, w_p = stats.wilcoxon(nonzero)

    rows.append({
        'TEST': 'AVR_PRE_VS_BENCH',
        'N': len(avr_df),
        'MEAN_AVR_PRE': avr_df['AVR_PRE'].mean(),
        'MEAN_AVR_BENCH': avr_df['AVR_BENCH'].mean(),
        'MEAN_AVR_DIFF': avr_df['AVR_DIFF'].mean(),
        'MEDIAN_AVR_DIFF': avr_df['AVR_DIFF'].median(),
        'T_STAT': t_stat,
        'T_PVALUE': t_p,
        'WILCOXON_STAT': w_stat if not np.isnan(w_stat) else None,
        'WILCOXON_PVALUE': w_p if not np.isnan(w_p) else None,
    })

    # By quintile of firm size (dollar volume)
    avr_df['SIZE_Q'] = pd.qcut(avr_df['BENCH_DOLLAR_VOL'], q=min(5, len(avr_df) // 5 or 1),
                                 labels=False, duplicates='drop') + 1
    for q, grp in avr_df.groupby('SIZE_Q'):
        t_s, t_pp = (np.nan, np.nan)
        if len(grp) > 2:
            t_s, t_pp = stats.ttest_rel(grp['AVR_PRE'], grp['AVR_BENCH'])
        rows.append({
            'TEST': f'AVR_SIZE_Q{int(q)}',
            'N': len(grp),
            'MEAN_AVR_PRE': grp['AVR_PRE'].mean(),
            'MEAN_AVR_BENCH': grp['AVR_BENCH'].mean(),
            'MEAN_AVR_DIFF': grp['AVR_DIFF'].mean(),
            'MEDIAN_AVR_DIFF': grp['AVR_DIFF'].median(),
            'T_STAT': t_s if not np.isnan(t_s) else None,
            'T_PVALUE': t_pp if not np.isnan(t_pp) else None,
            'WILCOXON_STAT': None,
            'WILCOXON_PVALUE': None,
        })

    return pd.DataFrame(rows)


def compute_firm_fixed_effects(panel):
    """OLS with firm fixed effects for firms appearing in multiple events."""
    dep_var = 'PRE_FULL_NET_DOLLAR_SOLD'
    if dep_var not in panel.columns:
        return pd.DataFrame()

    pre_days = abs(WINDOWS['PRE_FULL'][1] - WINDOWS['PRE_FULL'][0]) + 1
    df = panel[['TICKER', dep_var, 'IS_TREATMENT', 'LEAN']].dropna(subset=[dep_var]).copy()
    df['DAILY_SELL'] = df[dep_var] / pre_days
    df['IS_TREATMENT_INT'] = df['IS_TREATMENT'].astype(int)

    # Only include firms with >1 event (otherwise FE absorbs everything)
    firm_counts = df['TICKER'].value_counts()
    multi_event_firms = firm_counts[firm_counts > 1].index
    df_multi = df[df['TICKER'].isin(multi_event_firms)].copy()

    rows = []

    # Spec 1: Firm FE via dummies (for small panel)
    if len(df_multi) >= 10 and len(multi_event_firms) >= 3:
        try:
            firm_dummies = pd.get_dummies(df_multi['TICKER'], prefix='FE', drop_first=True).astype(float)
            X = sm.add_constant(pd.concat([
                df_multi[['IS_TREATMENT_INT']].astype(float).reset_index(drop=True),
                firm_dummies.reset_index(drop=True)
            ], axis=1))
            y = df_multi['DAILY_SELL'].astype(float).reset_index(drop=True)
            model = sm.OLS(y, X).fit(
                cov_type='cluster',
                cov_kwds={'groups': df_multi['TICKER'].reset_index(drop=True)})
            # Only report treatment coefficient and constant
            for var in ['const', 'IS_TREATMENT_INT']:
                if var in model.params.index:
                    rows.append({
                        'SPECIFICATION': 'FIRM_FE_MULTI_EVENT',
                        'VARIABLE': var,
                        'COEFFICIENT': model.params[var],
                        'STD_ERROR': model.bse[var],
                        'T_STAT': model.tvalues[var],
                        'P_VALUE': model.pvalues[var],
                        'R_SQUARED': model.rsquared,
                        'N_OBS': int(model.nobs),
                        'N_FIRMS': len(multi_event_firms),
                        'IS_TREATMENT_INCLUDED': True,
                    })
        except Exception as e:
            logger.warning("Firm FE (multi-event) failed: %s", e)

    # Spec 2: Within-firm demeaning (Mundlak)
    # Manual demeaning absorbs N_firms degrees of freedom that OLS doesn't
    # know about. Correct SEs by inflating by sqrt(n-k_ols) / sqrt(n-k_ols-N_firms).
    # NOTE: This dof correction on top of cluster-robust SEs is an approximation —
    # cluster SEs already include their own finite-cluster adjustment. The correction
    # is conservative (inflates SEs), so it won't produce false positives. For an
    # exact treatment, use linearmodels.PanelOLS with entity_effects=True. Spec 1
    # (explicit dummies) is the primary firm-FE result; this is a robustness check.
    if len(df_multi) >= 10:
        try:
            n_firms_dm = len(multi_event_firms)
            firm_means = df_multi.groupby('TICKER')['DAILY_SELL'].transform('mean')
            y_dm = (df_multi['DAILY_SELL'] - firm_means).astype(float).reset_index(drop=True)
            treat_dm = (df_multi['IS_TREATMENT_INT'] -
                        df_multi.groupby('TICKER')['IS_TREATMENT_INT'].transform('mean'))
            # Check whether demeaned treatment has variation; if firm FE
            # fully absorbs treatment (every firm is always-treated or
            # always-control), the demeaned treatment is zero and the
            # coefficient is meaningless.
            treatment_included = treat_dm.abs().sum() > 1e-10
            if not treatment_included:
                logger.info("Firm FE absorbs IS_TREATMENT entirely "
                            "(no within-firm variation); reporting intercept only")
                X_dm = sm.add_constant(
                    pd.Series(0.0, index=y_dm.index, name='IS_TREATMENT_INT'))
            else:
                X_dm = sm.add_constant(treat_dm.astype(float).reset_index(drop=True))
            model_dm = sm.OLS(y_dm, X_dm).fit(
                cov_type='cluster',
                cov_kwds={'groups': df_multi['TICKER'].reset_index(drop=True)})

            # Correct SEs for absorbed firm dummies (cluster SEs handle
            # within-firm correlation; dof correction handles absorbed FE)
            n = int(model_dm.nobs)
            k_ols = len(model_dm.params)
            dof_ols = n - k_ols
            dof_true = n - k_ols - n_firms_dm
            if dof_true > 0:
                se_correction = np.sqrt(dof_ols / dof_true)
            else:
                se_correction = np.nan

            for var in model_dm.params.index:
                corrected_se = model_dm.bse[var] * se_correction if not np.isnan(se_correction) else np.nan
                corrected_t = model_dm.params[var] / corrected_se if corrected_se and corrected_se > 0 else np.nan
                corrected_p = 2 * stats.t.sf(abs(corrected_t), dof_true) if not np.isnan(corrected_t) and dof_true > 0 else np.nan
                rows.append({
                    'SPECIFICATION': 'WITHIN_FIRM_DEMEANED',
                    'VARIABLE': var,
                    'COEFFICIENT': model_dm.params[var],
                    'STD_ERROR': corrected_se,
                    'T_STAT': corrected_t,
                    'P_VALUE': corrected_p,
                    'R_SQUARED': model_dm.rsquared,
                    'N_OBS': n,
                    'N_FIRMS': n_firms_dm,
                    'IS_TREATMENT_INCLUDED': treatment_included,
                })
        except Exception as e:
            logger.warning("Within-firm demeaned failed: %s", e)

    df_out = pd.DataFrame(rows)
    if not df_out.empty:
        df_out['BH_SIGNIFICANT'] = _benjamini_hochberg(df_out['P_VALUE'].values).astype(int)
    return df_out


def compute_cross_sectional_determinants(panel, events=None):
    """What predicts which firms see more abnormal selling?

    Regresses abnormal selling (pre daily - bench daily) on firm/event
    characteristics: political leaning, number of benchmark insiders,
    benchmark activity level, opportunistic share.
    """
    pre_days = abs(WINDOWS['PRE_FULL'][1] - WINDOWS['PRE_FULL'][0]) + 1
    bench_days = abs(WINDOWS['BENCHMARK'][1] - WINDOWS['BENCHMARK'][0]) + 1

    dep_var = 'PRE_FULL_NET_DOLLAR_SOLD'
    if dep_var not in panel.columns:
        return pd.DataFrame()

    treatment = panel[panel['IS_TREATMENT'] == True].copy()
    if len(treatment) < 15:
        return pd.DataFrame()

    treatment['ABN_SELL'] = (treatment[dep_var] / pre_days -
                              treatment['BENCHMARK_NET_DOLLAR_SOLD'] / bench_days)

    # Build regressors
    regressors = ['ABN_SELL']
    treat_df = treatment[['ABN_SELL', 'TICKER']].copy()

    if 'BENCHMARK_N_UNIQUE_INSIDERS' in treatment.columns:
        treat_df['BENCH_INSIDERS'] = treatment['BENCHMARK_N_UNIQUE_INSIDERS'].fillna(0)
        regressors.append('BENCH_INSIDERS')

    # N13 fix: BENCH_ACTIVITY removed — it is the subtrahend of ABN_SELL,
    # so including it as a regressor creates mechanical correlation with the DV.

    if 'BENCHMARK_N_OPPORTUNISTIC' in treatment.columns and 'BENCHMARK_N_ROUTINE' in treatment.columns:
        denom = (treatment['BENCHMARK_N_OPPORTUNISTIC'].fillna(0) +
                 treatment['BENCHMARK_N_ROUTINE'].fillna(0))
        treat_df['OPP_SHARE'] = np.where(
            denom > 0,
            treatment['BENCHMARK_N_OPPORTUNISTIC'].fillna(0) / denom,
            np.nan)
        regressors.append('OPP_SHARE')

    if 'LEAN' in treatment.columns:
        lean_dummies = pd.get_dummies(treatment['LEAN'], prefix='LEAN', drop_first=True).astype(float)
        lean_dummies = lean_dummies.reset_index(drop=True)
        treat_df = treat_df.reset_index(drop=True)
        treat_df = pd.concat([treat_df, lean_dummies], axis=1)
        regressors.extend(lean_dummies.columns.tolist())

    x_cols = [c for c in regressors if c != 'ABN_SELL' and c in treat_df.columns]
    if not x_cols:
        return pd.DataFrame()

    treat_df = treat_df.dropna(subset=['ABN_SELL'] + x_cols).reset_index(drop=True)
    if len(treat_df) < 15:
        return pd.DataFrame()

    rows = []
    try:
        y = treat_df['ABN_SELL'].astype(float)
        X = sm.add_constant(treat_df[x_cols].astype(float))
        model = sm.OLS(y, X).fit(cov_type='HC3')
        for var in model.params.index:
            rows.append({
                'VARIABLE': var,
                'COEFFICIENT': model.params[var],
                'STD_ERROR': model.bse[var],
                'T_STAT': model.tvalues[var],
                'P_VALUE': model.pvalues[var],
                'R_SQUARED': model.rsquared,
                'ADJ_R_SQUARED': model.rsquared_adj,
                'N_OBS': int(model.nobs),
            })
    except Exception as e:
        logger.warning("Cross-sectional determinants failed: %s", e)

    df = pd.DataFrame(rows)
    if not df.empty:
        df['BH_SIGNIFICANT'] = _benjamini_hochberg(df['P_VALUE'].values).astype(int)
    return df


def compute_post_event_reversal(panel):
    """Test if high pre-event insider selling predicts worse post-event CARs.

    Sorts treatment firms into terciles by pre-event abnormal selling,
    then compares post-event CAR distributions across terciles.
    """
    if 'CAR_POST' not in panel.columns:
        return pd.DataFrame()

    pre_days = abs(WINDOWS['PRE_FULL'][1] - WINDOWS['PRE_FULL'][0]) + 1
    bench_days = abs(WINDOWS['BENCHMARK'][1] - WINDOWS['BENCHMARK'][0]) + 1

    treatment = panel[panel['IS_TREATMENT'] == True].copy()
    valid = treatment.dropna(subset=['CAR_POST', 'PRE_FULL_NET_DOLLAR_SOLD']).copy()
    if len(valid) < 15:
        return pd.DataFrame()

    valid['ABN_SELL'] = (valid['PRE_FULL_NET_DOLLAR_SOLD'] / pre_days -
                          valid['BENCHMARK_NET_DOLLAR_SOLD'] / bench_days)

    # Terciles
    n_groups = min(3, len(valid) // 5 or 1)
    if n_groups < 2:
        return pd.DataFrame()
    valid['SELL_TERCILE'] = pd.qcut(valid['ABN_SELL'], q=n_groups,
                                      labels=[f'T{i+1}' for i in range(n_groups)],
                                      duplicates='drop')
    actual_groups = valid['SELL_TERCILE'].nunique()
    if actual_groups < 3:
        logger.warning("Post-event reversal: qcut produced %d groups (requested 3); "
                       "heavy zero-ties in ABN_SELL collapsed terciles", actual_groups)

    rows = []
    groups = []
    for tercile, grp in valid.groupby('SELL_TERCILE'):
        cars = grp['CAR_POST'].dropna()
        groups.append(cars.values)
        t_stat, t_p = (np.nan, np.nan)
        if len(cars) > 1:
            t_stat, t_p = stats.ttest_1samp(cars, 0)
        rows.append({
            'TERCILE': str(tercile),
            'N': len(grp),
            'MEAN_ABN_SELL': grp['ABN_SELL'].mean(),
            'MEAN_CAR': cars.mean() if len(cars) > 0 else np.nan,
            'MEDIAN_CAR': cars.median() if len(cars) > 0 else np.nan,
            'STD_CAR': cars.std() if len(cars) > 0 else np.nan,
            'T_STAT_VS_ZERO': t_stat if not np.isnan(t_stat) else None,
            'P_VALUE_VS_ZERO': t_p if not np.isnan(t_p) else None,
        })

    # Test monotonic relationship: highest selling tercile should have lowest CAR
    valid_groups = [g for g in groups if len(g) > 0]
    if len(valid_groups) >= 2:
        try:
            kw_stat, kw_p = stats.kruskal(*valid_groups)
        except Exception:
            kw_stat, kw_p = None, None
    else:
        kw_stat, kw_p = None, None

    # High vs Low tercile direct comparison
    # MEAN_CAR = (high-selling firms CAR) - (low-selling firms CAR)
    # Negative value supports reversal hypothesis (more selling → worse returns)
    if len(groups) >= 2 and len(groups[0]) > 1 and len(groups[-1]) > 1:
        try:
            hl_t, hl_p = stats.ttest_ind(groups[-1], groups[0], equal_var=False)
        except Exception:
            hl_t, hl_p = None, None
        rows.append({
            'TERCILE': 'HIGH_VS_LOW',
            'N': len(groups[-1]) + len(groups[0]),
            'MEAN_ABN_SELL': np.nan,
            'MEAN_CAR': np.mean(groups[-1]) - np.mean(groups[0]),
            'MEDIAN_CAR': np.nan,
            'STD_CAR': np.nan,
            'T_STAT_VS_ZERO': hl_t,
            'P_VALUE_VS_ZERO': hl_p,
        })

    df = pd.DataFrame(rows)
    df['KW_STAT'] = kw_stat
    df['KW_PVALUE'] = kw_p
    return df


def compute_bootstrap_ci(panel, n_bootstrap=1000, seed=42):
    """Bootstrap 95% confidence intervals for key statistics."""
    rng = np.random.RandomState(seed)
    pre_days = abs(WINDOWS['PRE_FULL'][1] - WINDOWS['PRE_FULL'][0]) + 1
    bench_days = abs(WINDOWS['BENCHMARK'][1] - WINDOWS['BENCHMARK'][0]) + 1

    pre_col = 'PRE_FULL_NET_DOLLAR_SOLD'
    bench_col = 'BENCHMARK_NET_DOLLAR_SOLD'
    if pre_col not in panel.columns or bench_col not in panel.columns:
        return pd.DataFrame()

    treatment = panel[panel['IS_TREATMENT'] == True].copy()
    control = panel[panel['IS_TREATMENT'] == False].copy()

    stats_config = [
        ('MEAN_ABN_SELL_TREATMENT', treatment, 'abn_sell'),
        ('MEAN_ABN_SELL_CONTROL', control, 'abn_sell'),
        ('DID_ESTIMATE', panel, 'did'),
    ]

    rows = []
    for stat_name, sub, stat_type in stats_config:
        if len(sub) < 5:
            continue

        if stat_type == 'abn_sell':
            values = (sub[pre_col] / pre_days - sub[bench_col] / bench_days).values
            observed = np.mean(values)
            boot_stats = []
            for _ in range(n_bootstrap):
                sample = rng.choice(values, size=len(values), replace=True)
                boot_stats.append(np.mean(sample))
        elif stat_type == 'did':
            # Resample matched pairs together to preserve pairing structure
            # Use PAIR_ID to properly align treatment and control
            pair_col = 'PAIR_ID' if 'PAIR_ID' in panel.columns else None
            if pair_col:
                dup_t = treatment[pair_col].duplicated().sum()
                dup_c = control[pair_col].duplicated().sum()
                if dup_t or dup_c:
                    raise ValueError(
                        f"Duplicate PAIR_IDs: {dup_t} in treatment, {dup_c} in control. "
                        "Join would produce cartesian product.")
                paired = treatment.set_index(pair_col)[[pre_col, bench_col]].join(
                    control.set_index(pair_col)[[pre_col, bench_col]],
                    lsuffix='_t', rsuffix='_c', how='inner'
                )
                t_vals = (paired[f'{pre_col}_t'] / pre_days - paired[f'{bench_col}_t'] / bench_days).values
                c_vals = (paired[f'{pre_col}_c'] / pre_days - paired[f'{bench_col}_c'] / bench_days).values
            elif 'CONTROL_TICKER' in panel.columns and 'EVENT_ID' in panel.columns:
                # Build explicit pair tuples from EVENT_ID → CONTROL_TICKER mapping
                pair_map = treatment.set_index('EVENT_ID')[['TICKER', pre_col, bench_col]].join(
                    control.set_index('EVENT_ID')[['TICKER', pre_col, bench_col]],
                    lsuffix='_t', rsuffix='_c', how='inner'
                )
                t_vals = (pair_map[f'{pre_col}_t'] / pre_days - pair_map[f'{bench_col}_t'] / bench_days).values
                c_vals = (pair_map[f'{pre_col}_c'] / pre_days - pair_map[f'{bench_col}_c'] / bench_days).values
            else:
                # No reliable pairing available — sort-based alignment is broken
                # (treatment/control tickers differ, so secondary sort scrambles pairs).
                # Fail loudly rather than produce silently misaligned bootstrap results.
                raise ValueError(
                    "Bootstrap DiD requires PAIR_ID or EVENT_ID+CONTROL_TICKER columns "
                    "to align treatment-control pairs. Sort-based fallback is unreliable."
                )
            n_pairs = min(len(t_vals), len(c_vals))
            t_vals = t_vals[:n_pairs]
            c_vals = c_vals[:n_pairs]
            observed = np.mean(t_vals) - np.mean(c_vals)
            boot_stats = []
            for _ in range(n_bootstrap):
                pair_idx = rng.choice(n_pairs, size=n_pairs, replace=True)
                boot_stats.append(np.mean(t_vals[pair_idx]) - np.mean(c_vals[pair_idx]))
        else:
            continue

        boot_stats = np.array(boot_stats)
        ci_lower = np.percentile(boot_stats, 2.5)
        ci_upper = np.percentile(boot_stats, 97.5)
        boot_se = boot_stats.std()

        rows.append({
            'STATISTIC': stat_name,
            'OBSERVED': observed,
            'BOOT_MEAN': boot_stats.mean(),
            'BOOT_SE': boot_se,
            'CI_LOWER_95': ci_lower,
            'CI_UPPER_95': ci_upper,
            'CI_LOWER_90': np.percentile(boot_stats, 5),
            'CI_UPPER_90': np.percentile(boot_stats, 95),
            'SIGNIFICANT_95': 1 if (ci_lower > 0 or ci_upper < 0) else 0,
            'N': len(sub) if stat_type == 'abn_sell' else len(panel),
            'N_BOOTSTRAP': n_bootstrap,
        })

    return pd.DataFrame(rows)


def compute_fama_macbeth(panel):
    """Fama-MacBeth (1973) cross-sectional regressions by event date.

    For each event date, run cross-sectional regression of daily net selling
    on IS_TREATMENT. Average coefficients across dates and compute
    Newey-West adjusted standard errors.
    """
    dep_var = 'PRE_FULL_NET_DOLLAR_SOLD'
    if dep_var not in panel.columns:
        return pd.DataFrame()

    pre_days = abs(WINDOWS['PRE_FULL'][1] - WINDOWS['PRE_FULL'][0]) + 1
    df = panel[['TICKER', 'EVENT_DATE', dep_var, 'IS_TREATMENT']].dropna(subset=[dep_var]).copy()
    df['DAILY_SELL'] = df[dep_var] / pre_days
    df['IS_TREATMENT_INT'] = df['IS_TREATMENT'].astype(int)

    # Cross-sectional regression for each event date
    date_coefs = []
    for date, grp in df.groupby('EVENT_DATE'):
        if len(grp) < 4 or grp['IS_TREATMENT_INT'].nunique() < 2:
            continue
        try:
            y = grp['DAILY_SELL'].astype(float)
            X = sm.add_constant(grp[['IS_TREATMENT_INT']].astype(float))
            model = sm.OLS(y, X).fit()
            const_pos = next((i for i, k in enumerate(model.params.index)
                              if k in ('const', 'Intercept')), 0)
            treat_pos = next((i for i, k in enumerate(model.params.index)
                              if k == 'IS_TREATMENT_INT'), None)
            date_coefs.append({
                'DATE': date,
                'CONST': model.params.iloc[const_pos],
                'IS_TREATMENT_INT': model.params.iloc[treat_pos] if treat_pos is not None else 0,
                'N': len(grp),
            })
        except Exception:
            continue

    if len(date_coefs) < 3:
        return pd.DataFrame()

    coef_df = pd.DataFrame(date_coefs)
    T = len(coef_df)
    n_per_date = coef_df['N']
    logger.info("FM: %d event-date cross-sections; N per date: "
                "median=%.0f, mean=%.1f, min=%d, max=%d",
                T, n_per_date.median(), n_per_date.mean(),
                int(n_per_date.min()), int(n_per_date.max()))

    rows = []
    for var in ['CONST', 'IS_TREATMENT_INT']:
        gamma = coef_df[var].values
        gamma_bar = gamma.mean()
        # Newey-West SE with lag = int(T^(1/3))
        max_lag = max(1, int(T ** (1 / 3)))
        gamma_dm = gamma - gamma_bar
        var_nw = np.sum(gamma_dm ** 2) / T
        for lag in range(1, max_lag + 1):
            weight = 1 - lag / (max_lag + 1)  # Bartlett kernel
            autocovar = np.sum(gamma_dm[lag:] * gamma_dm[:-lag]) / T
            var_nw += 2 * weight * autocovar
        if var_nw < 0:
            logger.warning("FM Newey-West variance negative for %s (%.4e); "
                           "falling back to simple variance", var, var_nw)
            var_nw = np.sum(gamma_dm ** 2) / T  # drop NW correction
            max_lag = 0  # signal that NW was not applied
        se_nw = np.sqrt(max(var_nw, 1e-30) / T)
        t_stat = gamma_bar / se_nw if se_nw > 0 else 0
        p_val = 2 * (1 - stats.norm.cdf(abs(t_stat)))

        rows.append({
            'VARIABLE': var,
            'FM_COEFFICIENT': gamma_bar,
            'FM_STD_ERROR': se_nw,
            'FM_T_STAT': t_stat,
            'FM_P_VALUE': p_val,
            'N_PERIODS': T,
            'MAX_LAG_NW': max_lag,
            'MEDIAN_N_PER_DATE': float(n_per_date.median()),
            'MEAN_N_PER_DATE': float(n_per_date.mean()),
        })

    df_out = pd.DataFrame(rows)
    if not df_out.empty:
        df_out['BH_SIGNIFICANT'] = _benjamini_hochberg(df_out['FM_P_VALUE'].values).astype(int)
    return df_out


def compute_short_swing_check(panel, form4):
    """Section 16(b) short-swing profit rule check.

    Flags insiders who bought within 6 months before selling in the pre-event
    window. These insiders face legal constraints on profit-taking, so their
    sells may be mechanically constrained rather than informationally motivated.
    Reports results with and without short-swing constrained insiders.
    """
    pre_col = 'PRE_FULL_NET_DOLLAR_SOLD'
    bench_col = 'BENCHMARK_NET_DOLLAR_SOLD'
    if pre_col not in panel.columns or bench_col not in panel.columns:
        return pd.DataFrame()

    pre_days = abs(WINDOWS['PRE_FULL'][1] - WINDOWS['PRE_FULL'][0]) + 1
    bench_days = abs(WINDOWS['BENCHMARK'][1] - WINDOWS['BENCHMARK'][0]) + 1

    # Identify tickers where insiders had buys within 6 months before pre-event sells
    short_swing_tickers = set()
    for _, row in panel[panel['IS_TREATMENT'] == True].iterrows():
        ticker = row['TICKER']
        event_date = pd.Timestamp(row['EVENT_DATE'])
        pre_start = event_date + pd.Timedelta(days=WINDOWS['PRE_FULL'][0])
        pre_end = event_date + pd.Timedelta(days=WINDOWS['PRE_FULL'][1])
        lookback_start = pre_start - pd.Timedelta(days=180)

        ticker_txns = form4[form4['ticker'] == ticker]
        # Find sellers in pre-event window
        pre_sellers = ticker_txns[
            (ticker_txns['transaction_date'] >= pre_start) &
            (ticker_txns['transaction_date'] <= pre_end) &
            (ticker_txns['_trade_type'] == 'sell')
        ]['owner_name'].unique()

        # Check if any of these sellers bought in the 6 months before their sell
        for seller in pre_sellers:
            buys = ticker_txns[
                (ticker_txns['owner_name'] == seller) &
                (ticker_txns['transaction_date'] >= lookback_start) &
                (ticker_txns['transaction_date'] < pre_start) &
                (ticker_txns['_trade_type'] == 'buy')
            ]
            if len(buys) > 0:
                short_swing_tickers.add(ticker)
                break

    panel = panel.copy()
    panel['SHORT_SWING_FLAG'] = panel['TICKER'].isin(short_swing_tickers).astype(int)

    rows = []
    for label, sub in [('ALL', panel),
                        ('EXCL_SHORT_SWING', panel[panel['SHORT_SWING_FLAG'] == 0]),
                        ('ONLY_SHORT_SWING', panel[panel['SHORT_SWING_FLAG'] == 1])]:
        treatment = sub[sub['IS_TREATMENT'] == True]
        if len(treatment) < 5:
            continue
        pre_daily = treatment[pre_col] / pre_days
        bench_daily = treatment[bench_col] / bench_days
        diff = pre_daily - bench_daily
        t_stat, t_p = (np.nan, np.nan)
        if len(diff) > 1:
            t_stat, t_p = stats.ttest_rel(pre_daily, bench_daily)
        rows.append({
            'FILTER': label,
            'N_EVENTS': len(treatment),
            'N_SHORT_SWING_TICKERS': len(short_swing_tickers),
            'MEAN_DIFF': diff.mean(),
            'T_STAT': t_stat if not np.isnan(t_stat) else None,
            'P_VALUE': t_p if not np.isnan(t_p) else None,
        })

    df = pd.DataFrame(rows)
    if not df.empty and df['P_VALUE'].notna().any():
        df['BH_SIGNIFICANT'] = _benjamini_hochberg(df['P_VALUE'].fillna(1).values).astype(int)
    return df


# ═══════════════════════════════════════════════════════════════════════
# 4.1 AGGREGATE NULL CHARACTERIZATION (TOST + POWER)
# ═══════════════════════════════════════════════════════════════════════

def compute_tost_equivalence(panel):
    """Two One-Sided Tests (TOST) for equivalence of pre-event vs benchmark selling.

    Tests whether aggregate insider selling is *equivalent* to benchmark within
    a practically meaningful margin (±δ). A significant TOST means we can
    confidently say the effect is negligibly small — characterizing the aggregate
    null rather than just failing to reject it.

    Equivalence margins (δ):
      - ±10% of benchmark mean daily selling (primary)
      - ±20% of benchmark mean (sensitivity)
      - ±0.5 SD of the difference distribution (Cohen's convention)

    Also reports post-hoc power of the paired t-test at conventional α=0.05
    for detecting small (d=0.2), medium (d=0.5), and the observed effect size.
    """
    pre_col = 'PRE_FULL_NET_DOLLAR_SOLD'
    bench_col = 'BENCHMARK_NET_DOLLAR_SOLD'
    if pre_col not in panel.columns or bench_col not in panel.columns:
        return pd.DataFrame()

    pre_days = abs(WINDOWS['PRE_FULL'][1] - WINDOWS['PRE_FULL'][0]) + 1
    bench_days = abs(WINDOWS['BENCHMARK'][1] - WINDOWS['BENCHMARK'][0]) + 1

    treatment = panel[panel['IS_TREATMENT'] == True].copy()
    if len(treatment) < 10:
        return pd.DataFrame()

    pre_daily = treatment[pre_col] / pre_days
    bench_daily = treatment[bench_col] / bench_days
    diff = pre_daily - bench_daily
    n = len(diff)
    diff_mean = diff.mean()
    diff_std = diff.std()
    diff_se = diff_std / np.sqrt(n)

    # Observed effect size (Cohen's d for paired data)
    observed_d = abs(diff_mean) / diff_std if diff_std > 0 else 0

    rows = []

    # ── TOST across multiple equivalence margins ──
    bench_mean_daily = bench_daily.mean()
    margins = [
        # 10%/20% of benchmark daily selling: pre-specified, external to the test
        ('10PCT_BENCH', abs(bench_mean_daily) * 0.10),
        ('20PCT_BENCH', abs(bench_mean_daily) * 0.20),
        # HALF_SD: sensitivity check only — margin is circular (wider when data
        # is noisy → equivalence easier to establish). Primary SESOI is benchmark-based.
        ('HALF_SD', diff_std * 0.5),
    ]

    for margin_name, delta in margins:
        if delta <= 0:
            continue
        # TOST: reject equivalence null if BOTH one-sided tests reject
        # H0_lower: diff <= -delta  =>  t_lower = (diff_mean + delta) / se
        # H0_upper: diff >= +delta  =>  t_upper = (diff_mean - delta) / se
        t_lower = (diff_mean + delta) / diff_se if diff_se > 0 else 0
        t_upper = (diff_mean - delta) / diff_se if diff_se > 0 else 0
        p_lower = 1 - stats.t.cdf(t_lower, df=n - 1)
        p_upper = stats.t.cdf(t_upper, df=n - 1)
        tost_p = max(p_lower, p_upper)  # both must reject
        equivalent = 1 if tost_p < 0.05 else 0

        # 90% CI for the difference (standard for equivalence testing)
        t_crit_90 = stats.t.ppf(0.95, df=n - 1)
        ci90_lower = diff_mean - t_crit_90 * diff_se
        ci90_upper = diff_mean + t_crit_90 * diff_se

        rows.append({
            'TEST': 'TOST',
            'MARGIN_NAME': margin_name,
            'DELTA': delta,
            'DIFF_MEAN': diff_mean,
            'DIFF_SE': diff_se,
            'T_LOWER': t_lower,
            'P_LOWER': p_lower,
            'T_UPPER': t_upper,
            'P_UPPER': p_upper,
            'TOST_P': tost_p,
            'EQUIVALENT': equivalent,
            'CI90_LOWER': ci90_lower,
            'CI90_UPPER': ci90_upper,
            'N': n,
            'OBSERVED_D': observed_d,
            'STATISTIC_VALUE': observed_d,  # Cohen's d for TOST rows
        })

    # ── Post-hoc power analysis ──
    # Power of paired t-test at α=0.05 for various effect sizes
    alpha = 0.05
    t_crit = stats.t.ppf(1 - alpha / 2, df=n - 1)
    for d_label, d_target in [('SMALL_D02', 0.2), ('MEDIUM_D05', 0.5),
                               ('OBSERVED', observed_d)]:
        if d_target == 0:
            power = alpha  # power equals α when effect is zero
        else:
            # Non-central t-distribution: ncp = d * sqrt(n)
            ncp = d_target * np.sqrt(n)
            # Power = P(reject H0 | H1 true)
            # Under H1, test stat ~ noncentral t(df=n-1, ncp)
            power = 1 - stats.nct.cdf(t_crit, df=n - 1, nc=ncp) + \
                    stats.nct.cdf(-t_crit, df=n - 1, nc=ncp)

        rows.append({
            'TEST': 'POWER',
            'MARGIN_NAME': d_label,
            'DELTA': d_target,
            'DIFF_MEAN': diff_mean,
            'DIFF_SE': diff_se,
            'T_LOWER': np.nan,
            'P_LOWER': np.nan,
            'T_UPPER': np.nan,
            'P_UPPER': np.nan,
            'TOST_P': np.nan,
            'EQUIVALENT': np.nan,
            'CI90_LOWER': np.nan,
            'CI90_UPPER': np.nan,
            'N': n,
            'OBSERVED_D': observed_d,
            'STATISTIC_VALUE': power,  # power (probability, 0-1)
        })

    # ── Minimum detectable effect (MDE) at 80% power ──
    # Solve: d * sqrt(n) = t_crit + z_0.80
    z80 = stats.norm.ppf(0.80)
    mde_d = (t_crit + z80) / np.sqrt(n) if n > 0 else np.nan
    mde_dollars = mde_d * diff_std if diff_std > 0 else np.nan

    rows.append({
        'TEST': 'MDE',
        'MARGIN_NAME': 'MDE_80PCT_POWER',
        'DELTA': mde_d,
        'DIFF_MEAN': mde_dollars,
        'DIFF_SE': diff_se,
        'T_LOWER': np.nan,
        'P_LOWER': np.nan,
        'T_UPPER': np.nan,
        'P_UPPER': np.nan,
        'TOST_P': np.nan,
        'EQUIVALENT': np.nan,
        'CI90_LOWER': np.nan,
        'CI90_UPPER': np.nan,
        'N': n,
        'OBSERVED_D': observed_d,
        'STATISTIC_VALUE': mde_d,  # MDE in Cohen's d units
    })

    return pd.DataFrame(rows)


# ═══════════════════════════════════════════════════════════════════════
# 4.2 SUBGROUP SIGNAL-FINDING
# ═════════════════════════════════════════════════════════��═════════════

def compute_subgroup_analysis(panel, form4):
    """Disaggregated subgroup tests for theoretically motivated heterogeneity.

    Tests whether insider selling is elevated in specific subgroups where
    information asymmetry or incentives are strongest:

    1. C-suite insiders (CEO, CFO, COO, CTO, President) vs all insiders
    2. High-CAR events (worst return quintile post-event)
    3. Tight window (PRE_NEAR: -30 to -1 days only)
    4. Normalized sell ratio (volume-adjusted, not raw dollars)
    5. C-suite × High-CAR interaction (most theoretically motivated subgroup)
    6. Opportunistic insiders only (exclude routine traders)
    """
    pre_col = 'PRE_FULL_NET_DOLLAR_SOLD'
    bench_col = 'BENCHMARK_NET_DOLLAR_SOLD'
    near_pre_col = 'PRE_NEAR_NET_DOLLAR_SOLD'
    near_bench_col = 'BENCHMARK_NET_DOLLAR_SOLD'
    if pre_col not in panel.columns or bench_col not in panel.columns:
        return pd.DataFrame()

    pre_days = abs(WINDOWS['PRE_FULL'][1] - WINDOWS['PRE_FULL'][0]) + 1
    bench_days = abs(WINDOWS['BENCHMARK'][1] - WINDOWS['BENCHMARK'][0]) + 1
    near_days = abs(WINDOWS['PRE_NEAR'][1] - WINDOWS['PRE_NEAR'][0]) + 1

    treatment = panel[panel['IS_TREATMENT'] == True].copy()
    if len(treatment) < 10:
        return pd.DataFrame()

    rows = []

    # ── Helper: run paired t-test on a subgroup and return row ──
    def _test_subgroup(label, sub, pre_c=pre_col, bench_c=bench_col,
                       pre_d=pre_days, bench_d=bench_days):
        if len(sub) < 5:
            return None
        pre_daily = sub[pre_c] / pre_d
        bench_daily = sub[bench_c] / bench_d
        diff = pre_daily - bench_daily
        t_stat, t_p = (np.nan, np.nan)
        if len(diff) > 1 and diff.std() > 0:
            t_stat, t_p = stats.ttest_rel(pre_daily, bench_daily)
        cohen_d = diff.mean() / diff.std() if diff.std() > 0 else (0 if diff.mean() == 0 else np.nan)
        return {
            'SUBGROUP': label,
            'N_EVENTS': len(sub),
            'MEAN_PRE_DAILY': pre_daily.mean(),
            'MEAN_BENCH_DAILY': bench_daily.mean(),
            'MEAN_DIFF': diff.mean(),
            'MEDIAN_DIFF': diff.median(),
            'COHEN_D': cohen_d,
            'T_STAT': t_stat if not np.isnan(t_stat) else None,
            'P_VALUE': t_p if not np.isnan(t_p) else None,
            'PCT_POSITIVE': (diff > 0).mean() * 100,
        }

    # 0. Baseline: all treatment events (for comparison)
    row = _test_subgroup('ALL_TREATMENT', treatment)
    if row:
        rows.append(row)

    # 1. C-suite insiders
    # Identify events where C-suite officers traded in the pre-event window
    csuite_titles = {'CEO', 'CFO', 'COO', 'CTO', 'PRESIDENT', 'CHIEF EXECUTIVE',
                     'CHIEF FINANCIAL', 'CHIEF OPERATING', 'CHIEF TECHNOLOGY'}
    csuite_events = set()
    for _, ev in treatment.iterrows():
        ticker = ev['TICKER']
        event_date = pd.Timestamp(ev['EVENT_DATE'])
        pre_start = event_date + pd.Timedelta(days=WINDOWS['PRE_FULL'][0])
        pre_end = event_date + pd.Timedelta(days=WINDOWS['PRE_FULL'][1])
        ticker_txns = form4[
            (form4['ticker'] == ticker) &
            (form4['transaction_date'] >= pre_start) &
            (form4['transaction_date'] <= pre_end)
        ]
        # Check insider_title (not owner_name) for C-suite presence
        if 'insider_title' in ticker_txns.columns:
            titles = ticker_txns['insider_title'].dropna().astype(str).str.upper()
            if titles.apply(lambda t: any(cs in t for cs in csuite_titles)).any():
                csuite_events.add(ev['EVENT_ID'])
        else:
            # Fallback: check owner_name if insider_title unavailable
            for owner in ticker_txns['owner_name'].unique():
                owner_upper = str(owner).upper()
                if any(title in owner_upper for title in csuite_titles):
                    csuite_events.add(ev['EVENT_ID'])
                    break

    csuite_mask = treatment['EVENT_ID'].isin(csuite_events)
    row = _test_subgroup('CSUITE_PRESENT', treatment[csuite_mask])
    if row:
        rows.append(row)
    row = _test_subgroup('NO_CSUITE', treatment[~csuite_mask])
    if row:
        rows.append(row)

    # 2. High-CAR events (worst quintile of post-event returns)
    if 'CAR_POST' in treatment.columns:
        valid_car = treatment.dropna(subset=['CAR_POST'])
        if len(valid_car) >= 20:
            car_q20 = valid_car['CAR_POST'].quantile(0.20)
            high_car_mask = valid_car['CAR_POST'] <= car_q20
            row = _test_subgroup('HIGH_CAR_SEVERITY', valid_car[high_car_mask])
            if row:
                rows.append(row)
            row = _test_subgroup('LOW_CAR_SEVERITY', valid_car[~high_car_mask])
            if row:
                rows.append(row)

    # 3. Tight window (PRE_NEAR: -30 to -1 days)
    if near_pre_col in treatment.columns:
        row = _test_subgroup('TIGHT_WINDOW_30D', treatment,
                             pre_c=near_pre_col, bench_c=near_bench_col,
                             pre_d=near_days, bench_d=bench_days)
        if row:
            rows.append(row)

    # 4. Normalized sell ratio (NET_SELL_RATIO instead of raw dollars)
    nsr_pre = 'PRE_FULL_NET_SELL_RATIO'
    nsr_bench = 'BENCHMARK_NET_SELL_RATIO'
    if nsr_pre in treatment.columns and nsr_bench in treatment.columns:
        sub = treatment.copy()
        pre_vals = sub[nsr_pre].fillna(0)
        bench_vals = sub[nsr_bench].fillna(0)
        diff = pre_vals - bench_vals
        t_stat, t_p = (np.nan, np.nan)
        if len(diff) > 1 and diff.std() > 0:
            t_stat, t_p = stats.ttest_rel(pre_vals, bench_vals)
        cohen_d = diff.mean() / diff.std() if diff.std() > 0 else (0 if diff.mean() == 0 else np.nan)
        rows.append({
            'SUBGROUP': 'NORMALIZED_SELL_RATIO',
            'N_EVENTS': len(sub),
            'MEAN_PRE_DAILY': pre_vals.mean(),
            'MEAN_BENCH_DAILY': bench_vals.mean(),
            'MEAN_DIFF': diff.mean(),
            'MEDIAN_DIFF': diff.median(),
            'COHEN_D': cohen_d,
            'T_STAT': t_stat if not np.isnan(t_stat) else None,
            'P_VALUE': t_p if not np.isnan(t_p) else None,
            'PCT_POSITIVE': (diff > 0).mean() * 100,
        })

    # 5. C-suite × High-CAR interaction
    if 'CAR_POST' in treatment.columns and len(csuite_events) > 0:
        valid_car = treatment.dropna(subset=['CAR_POST'])
        if len(valid_car) >= 20:
            car_q20 = valid_car['CAR_POST'].quantile(0.20)
            interaction_mask = (valid_car['EVENT_ID'].isin(csuite_events) &
                                (valid_car['CAR_POST'] <= car_q20))
            row = _test_subgroup('CSUITE_X_HIGH_CAR', valid_car[interaction_mask])
            if row:
                rows.append(row)

    # 6. Opportunistic insiders only
    opp_pre = 'PRE_FULL_N_OPPORTUNISTIC'
    opp_bench = 'BENCHMARK_N_OPPORTUNISTIC'
    if opp_pre in treatment.columns and opp_bench in treatment.columns:
        # Events where opportunistic insiders were active pre-event
        opp_active = treatment[treatment[opp_pre] > 0]
        row = _test_subgroup('FIRMS_W_OPPORTUNISTIC_ACTIVITY', opp_active)
        if row:
            rows.append(row)

    # 7. Large firms (more insiders = more potential for information-motivated trading)
    insiders_col = 'BENCHMARK_N_UNIQUE_INSIDERS'
    if insiders_col in treatment.columns:
        median_insiders = treatment[insiders_col].median()
        if median_insiders > 0:
            large_mask = treatment[insiders_col] >= median_insiders
            row = _test_subgroup('LARGE_INSIDER_BASE', treatment[large_mask])
            if row:
                rows.append(row)
            row = _test_subgroup('SMALL_INSIDER_BASE', treatment[~large_mask])
            if row:
                rows.append(row)

    df = pd.DataFrame(rows)
    if not df.empty and df['P_VALUE'].notna().any():
        pvals = df['P_VALUE'].fillna(1).values
        df['BH_SIGNIFICANT'] = _benjamini_hochberg(pvals).astype(int)
    return df


# ═══════════════════════════════════════════════════════════════════════
# 4.3 ROBUSTNESS — VOLATILITY-SHIFT IDENTIFICATION
# ═══════════════════════════════════════════════════════════════════════

def _identify_vol_spikes(stock_data, fast_window=5, slow_window=90,
                          threshold=1.5, min_gap_days=30):
    """Identify firm-specific realized volatility spikes.

    A spike occurs when the rolling fast-window realized volatility
    (std of returns) exceeds threshold x the rolling slow-window
    realized volatility. Spikes within min_gap_days of each other
    are deduplicated (keep earliest).

    Returns DataFrame with columns: TICKER, SPIKE_DATE, FAST_VOL, SLOW_VOL, RATIO.
    """
    if stock_data is None or stock_data.empty:
        return pd.DataFrame()

    required = ['TICKER', 'DATE', 'RETURN']
    alt_return = None
    if 'RETURN' not in stock_data.columns:
        for candidate in ['RET', 'DAILY_RETURN', 'LOG_RETURN']:
            if candidate in stock_data.columns:
                alt_return = candidate
                break
        if alt_return is None and 'CLOSE' in stock_data.columns:
            alt_return = '_COMPUTED_RETURN'
        if alt_return is None:
            return pd.DataFrame()

    sd = stock_data.copy()
    sd['DATE'] = pd.to_datetime(sd['DATE'])
    if alt_return == '_COMPUTED_RETURN':
        sd = sd.sort_values(['TICKER', 'DATE'])
        sd['RETURN'] = sd.groupby('TICKER')['CLOSE'].pct_change()
    elif alt_return:
        sd['RETURN'] = sd[alt_return]

    spikes = []
    for ticker, grp in sd.groupby('TICKER'):
        grp = grp.sort_values('DATE').copy()
        if len(grp) < slow_window + fast_window:
            continue

        # Realized vol = rolling std of returns (annualized not needed for ratio)
        grp['FAST_VOL'] = grp['RETURN'].rolling(fast_window, min_periods=fast_window).std()
        grp['SLOW_VOL'] = grp['RETURN'].rolling(slow_window, min_periods=slow_window).std()
        grp['VOL_RATIO'] = grp['FAST_VOL'] / grp['SLOW_VOL']
        grp = grp.dropna(subset=['VOL_RATIO'])

        # Find spike dates
        spike_dates = grp[grp['VOL_RATIO'] >= threshold].copy()
        if spike_dates.empty:
            continue

        # Deduplicate: keep first spike in each cluster
        spike_dates = spike_dates.sort_values('DATE')
        kept = []
        last_kept = None
        for _, row in spike_dates.iterrows():
            if last_kept is None or (row['DATE'] - last_kept).days >= min_gap_days:
                kept.append({
                    'TICKER': ticker,
                    'SPIKE_DATE': row['DATE'],
                    'FAST_VOL': row['FAST_VOL'],
                    'SLOW_VOL': row['SLOW_VOL'],
                    'VOL_RATIO': row['VOL_RATIO'],
                })
                last_kept = row['DATE']
        spikes.extend(kept)

    return pd.DataFrame(spikes)


def compute_volatility_shift_analysis(panel, form4, stock_data,
                                       vol_threshold=1.5, pre_window=30):
    """Alternative identification: anchor on volatility spikes instead of event dates.

    Tests whether insiders sell before firm-specific volatility spikes. Cross-
    references with culture war event panel to distinguish:
      - Vol spikes coinciding with culture war events (CW-driven)
      - Vol spikes with no associated event (non-CW control)

    If insiders sell before CW-driven spikes but not non-CW spikes → political
    foreknowledge. If they sell before both → general volatility-timing. If
    neither → no anticipatory trading.

    This provides a natural within-firm control that the event-date anchor lacks.

    Parameters
    ----------
    panel : DataFrame
        The existing insider trading panel (for cross-referencing event dates).
    form4 : DataFrame
        Raw Form 4 transactions.
    stock_data : DataFrame
        Daily stock data with TICKER, DATE, RETURN (or CLOSE).
    vol_threshold : float
        Fast vol / slow vol ratio to qualify as a spike (default 1.5).
    pre_window : int
        Days before spike to measure insider selling (default 30).
    """
    if stock_data is None or stock_data.empty:
        logger.warning("Volatility shift: no stock data available")
        return pd.DataFrame()

    # 1. Identify vol spikes across all firms
    spikes = _identify_vol_spikes(stock_data, threshold=vol_threshold)
    if spikes.empty:
        logger.warning("Volatility shift: no vol spikes found at threshold %.1f",
                        vol_threshold)
        return pd.DataFrame()

    logger.info("  Vol spikes identified: %d across %d tickers",
                len(spikes), spikes['TICKER'].nunique())

    # 2. Cross-reference with culture war event dates
    # A spike is "CW-driven" if it falls within ±7 days of a treatment event
    event_dates = {}
    if not panel.empty:
        treatment = panel[panel['IS_TREATMENT'] == True]
        for _, row in treatment.iterrows():
            ticker = row['TICKER']
            edate = pd.Timestamp(row['EVENT_DATE'])
            if ticker not in event_dates:
                event_dates[ticker] = []
            event_dates[ticker].append(edate)

    def _is_cw_spike(ticker, spike_date, tolerance_days=7):
        if ticker not in event_dates:
            return False
        for edate in event_dates[ticker]:
            if abs((spike_date - edate).days) <= tolerance_days:
                return True
        return False

    spikes['IS_CW_DRIVEN'] = spikes.apply(
        lambda r: _is_cw_spike(r['TICKER'], r['SPIKE_DATE']), axis=1
    )

    n_cw = spikes['IS_CW_DRIVEN'].sum()
    n_non_cw = (~spikes['IS_CW_DRIVEN']).sum()
    logger.info("  CW-driven spikes: %d, Non-CW spikes: %d", n_cw, n_non_cw)

    # 3. Measure insider selling in [-pre_window, -1] before each spike
    # and in benchmark [-365, -181] (same as main panel)
    bench_start_offset = -365
    bench_end_offset = -181
    bench_days = abs(bench_end_offset - bench_start_offset) + 1

    # Ensure form4 has trade type
    if '_trade_type' not in form4.columns:
        form4 = form4.copy()
        form4['_trade_type'] = form4.apply(
            lambda r: _classify_trade(r['transaction_code'], r['acquired_disposed']),
            axis=1
        )

    spike_rows = []
    for _, spike in spikes.iterrows():
        ticker = spike['TICKER']
        spike_date = pd.Timestamp(spike['SPIKE_DATE'])

        ticker_txns = form4[form4['ticker'] == ticker]
        if ticker_txns.empty:
            continue

        # Pre-spike window
        pre_start = spike_date - pd.Timedelta(days=pre_window)
        pre_end = spike_date - pd.Timedelta(days=1)
        pre_mask = ((ticker_txns['transaction_date'] >= pre_start) &
                    (ticker_txns['transaction_date'] <= pre_end))
        pre_txns = ticker_txns[pre_mask]

        # Benchmark window
        bench_start = spike_date + pd.Timedelta(days=bench_start_offset)
        bench_end = spike_date + pd.Timedelta(days=bench_end_offset)
        bench_mask = ((ticker_txns['transaction_date'] >= bench_start) &
                      (ticker_txns['transaction_date'] <= bench_end))
        bench_txns = ticker_txns[bench_mask]

        # Compute selling metrics
        def _net_dollar(txns):
            sells = txns[txns['_trade_type'] == 'sell']
            buys = txns[txns['_trade_type'] == 'buy']
            sell_val = sells['transaction_value'].sum() if 'transaction_value' in sells.columns else (sells['shares'] * sells['price_per_share']).sum()
            buy_val = buys['transaction_value'].sum() if 'transaction_value' in buys.columns else (buys['shares'] * buys['price_per_share']).sum()
            return sell_val - buy_val

        pre_net = _net_dollar(pre_txns)
        bench_net = _net_dollar(bench_txns)

        pre_daily = pre_net / pre_window
        bench_daily = bench_net / bench_days

        pre_sells = len(pre_txns[pre_txns['_trade_type'] == 'sell'])
        bench_sells = len(bench_txns[bench_txns['_trade_type'] == 'sell'])

        spike_rows.append({
            'TICKER': ticker,
            'SPIKE_DATE': spike_date,
            'IS_CW_DRIVEN': spike['IS_CW_DRIVEN'],
            'VOL_RATIO': spike['VOL_RATIO'],
            'PRE_NET_DOLLAR_DAILY': pre_daily,
            'BENCH_NET_DOLLAR_DAILY': bench_daily,
            'DIFF_DAILY': pre_daily - bench_daily,
            'PRE_N_SELLS': pre_sells,
            'BENCH_N_SELLS_DAILY': bench_sells / bench_days,
            'PRE_N_TRANSACTIONS': len(pre_txns),
            'HAS_PRE_DATA': 1 if len(pre_txns) > 0 else 0,
        })

    if not spike_rows:
        return pd.DataFrame()

    spike_panel = pd.DataFrame(spike_rows)

    # 4. Statistical tests
    rows = []

    # Test A: All vol spikes — do insiders sell before vol spikes generally?
    with_data = spike_panel[spike_panel['HAS_PRE_DATA'] == 1]
    for label, sub in [('ALL_SPIKES', with_data),
                        ('CW_DRIVEN', with_data[with_data['IS_CW_DRIVEN'] == True]),
                        ('NON_CW', with_data[with_data['IS_CW_DRIVEN'] == False])]:
        if len(sub) < 5:
            rows.append({
                'TEST': label,
                'N_SPIKES': len(sub),
                'N_TICKERS': sub['TICKER'].nunique() if len(sub) > 0 else 0,
                'MEAN_PRE_DAILY': np.nan,
                'MEAN_BENCH_DAILY': np.nan,
                'MEAN_DIFF': np.nan,
                'COHEN_D': np.nan,
                'T_STAT': np.nan,
                'P_VALUE': np.nan,
            })
            continue

        diff = sub['DIFF_DAILY']
        t_stat, t_p = (np.nan, np.nan)
        if len(diff) > 1 and diff.std() > 0:
            t_stat, t_p = stats.ttest_1samp(diff, 0)
        cohen_d = diff.mean() / diff.std() if diff.std() > 0 else (0 if diff.mean() == 0 else np.nan)

        rows.append({
            'TEST': label,
            'N_SPIKES': len(sub),
            'N_TICKERS': sub['TICKER'].nunique(),
            'MEAN_PRE_DAILY': sub['PRE_NET_DOLLAR_DAILY'].mean(),
            'MEAN_BENCH_DAILY': sub['BENCH_NET_DOLLAR_DAILY'].mean(),
            'MEAN_DIFF': diff.mean(),
            'COHEN_D': cohen_d,
            'T_STAT': t_stat if not np.isnan(t_stat) else None,
            'P_VALUE': t_p if not np.isnan(t_p) else None,
        })

    # Test B: CW-driven vs non-CW (two-sample comparison)
    cw = with_data[with_data['IS_CW_DRIVEN'] == True]['DIFF_DAILY']
    non_cw = with_data[with_data['IS_CW_DRIVEN'] == False]['DIFF_DAILY']
    if len(cw) >= 5 and len(non_cw) >= 5:
        t_stat_2s, t_p_2s = stats.ttest_ind(cw, non_cw, equal_var=False)
        pooled_std = np.sqrt((cw.std() ** 2 + non_cw.std() ** 2) / 2)
        cohen_d_2s = (cw.mean() - non_cw.mean()) / pooled_std if pooled_std > 0 else 0

        rows.append({
            'TEST': 'CW_VS_NON_CW',
            'N_SPIKES': len(cw) + len(non_cw),
            'N_TICKERS': with_data['TICKER'].nunique(),
            'MEAN_PRE_DAILY': cw.mean(),
            'MEAN_BENCH_DAILY': non_cw.mean(),
            'MEAN_DIFF': cw.mean() - non_cw.mean(),
            'COHEN_D': cohen_d_2s,
            'T_STAT': t_stat_2s,
            'P_VALUE': t_p_2s,
        })

        # Mann-Whitney U for robustness
        try:
            u_stat, u_p = stats.mannwhitneyu(cw, non_cw, alternative='two-sided')
            rows.append({
                'TEST': 'CW_VS_NON_CW_MANNWHITNEY',
                'N_SPIKES': len(cw) + len(non_cw),
                'N_TICKERS': with_data['TICKER'].nunique(),
                'MEAN_PRE_DAILY': cw.median(),
                'MEAN_BENCH_DAILY': non_cw.median(),
                'MEAN_DIFF': cw.median() - non_cw.median(),
                'COHEN_D': np.nan,
                'T_STAT': u_stat,
                'P_VALUE': u_p,
            })
        except Exception:
            pass

    # Test C: Regression — DIFF_DAILY ~ IS_CW_DRIVEN + VOL_RATIO
    if len(with_data) >= 20:
        try:
            reg_df = with_data[['DIFF_DAILY', 'IS_CW_DRIVEN', 'VOL_RATIO']].copy()
            reg_df['IS_CW_INT'] = reg_df['IS_CW_DRIVEN'].astype(int)
            y = reg_df['DIFF_DAILY'].astype(float)
            X = sm.add_constant(reg_df[['IS_CW_INT', 'VOL_RATIO']].astype(float))
            model = sm.OLS(y, X).fit(cov_type='HC1')
            for var in model.params.index:
                rows.append({
                    'TEST': f'OLS_{var}',
                    'N_SPIKES': int(model.nobs),
                    'N_TICKERS': with_data['TICKER'].nunique(),
                    'MEAN_PRE_DAILY': np.nan,
                    'MEAN_BENCH_DAILY': np.nan,
                    'MEAN_DIFF': model.params[var],
                    'COHEN_D': np.nan,
                    'T_STAT': model.tvalues[var],
                    'P_VALUE': model.pvalues[var],
                })
        except Exception as e:
            logger.warning("Vol-shift OLS failed: %s", e)

    # Test D: Transition test — selling in the N days before VIX crosses threshold
    # (within-firm: compare pre-spike selling at same firm on CW vs non-CW dates)
    firms_with_both = set()
    if not spike_panel.empty:
        for ticker, grp in spike_panel.groupby('TICKER'):
            if grp['IS_CW_DRIVEN'].any() and (~grp['IS_CW_DRIVEN']).any():
                firms_with_both.add(ticker)

    if len(firms_with_both) >= 3:
        within_firm = spike_panel[spike_panel['TICKER'].isin(firms_with_both)].copy()
        # Within-firm difference: CW spike selling - non-CW spike selling (same firm)
        firm_diffs = []
        for ticker in firms_with_both:
            t_data = within_firm[within_firm['TICKER'] == ticker]
            cw_mean = t_data[t_data['IS_CW_DRIVEN'] == True]['DIFF_DAILY'].mean()
            ncw_mean = t_data[t_data['IS_CW_DRIVEN'] == False]['DIFF_DAILY'].mean()
            if not np.isnan(cw_mean) and not np.isnan(ncw_mean):
                firm_diffs.append(cw_mean - ncw_mean)

        if len(firm_diffs) >= 3:
            firm_diffs = np.array(firm_diffs)
            t_stat_wf, t_p_wf = stats.ttest_1samp(firm_diffs, 0)
            rows.append({
                'TEST': 'WITHIN_FIRM_CW_VS_NON_CW',
                'N_SPIKES': len(firm_diffs),
                'N_TICKERS': len(firms_with_both),
                'MEAN_PRE_DAILY': np.nan,
                'MEAN_BENCH_DAILY': np.nan,
                'MEAN_DIFF': firm_diffs.mean(),
                'COHEN_D': firm_diffs.mean() / firm_diffs.std() if firm_diffs.std() > 0 else 0,
                'T_STAT': t_stat_wf,
                'P_VALUE': t_p_wf,
            })

    df = pd.DataFrame(rows)
    if not df.empty and df['P_VALUE'].notna().any():
        pvals = df['P_VALUE'].fillna(1).values
        df['BH_SIGNIFICANT'] = _benjamini_hochberg(pvals).astype(int)

    # Log interpretation guide
    if not df.empty:
        for _, r in df[df['TEST'].isin(['CW_DRIVEN', 'NON_CW', 'CW_VS_NON_CW'])].iterrows():
            logger.info("    %s: diff=%.0f, d=%.3f, p=%s",
                        r['TEST'], r.get('MEAN_DIFF', 0) or 0,
                        r.get('COHEN_D', 0) or 0,
                        f"{r['P_VALUE']:.4f}" if pd.notna(r.get('P_VALUE')) else 'N/A')

    return df


# ═══════════════════════════════════════════════════════════════════════
# TAIL DIAGNOSTIC — WHO'S DRIVING THE MANN-WHITNEY DISTRIBUTIONAL DIFF?
# ═══════════════════════════════════════════════════════════════════════

# Keywords for classifying culture-war events as planned vs reactive.
# Event descriptions typically describe both the corporate ACTION (planned)
# and the public REACTION, so we use position-weighted scoring: keywords
# appearing in the first half of the description (the action clause) get
# double weight, since "Nike launches Kaepernick campaign" is planned even
# though the description also mentions "backlash".

# Keywords use base-form stems with \w* to capture inflections (plurals,
# past tense, gerunds).  Short roots that would false-positive on common
# words (e.g. 'ad' → 'adventure', 'fire' → 'first') are enumerated as
# explicit inflection lists instead.
#
# Note: finditer counts *every* match, so a description mentioning "protest"
# three times scores higher than one mention — this is intentional (stronger
# evidence from repetition).  The old str.find() version counted only the
# first occurrence per keyword.

_PLANNED_KEYWORDS = [
    # Base-form stems (safe with \w* — long enough to avoid false positives)
    'partnership', 'campaign', 'announc', 'initiat', 'pledg',
    'introduc', 'releas', 'unveil', 'sponsor', 'endors',
    'donat', 'creat', 'expand',
    'promot', 'featur', 'launch', 'collect',
    'advertisement', 'commercial', 'marketing',
    # Explicit inflections — 'invest' enumerated to avoid colliding with
    # 'investigat' (REACTIVE); 'commit' enumerated to avoid 'committee'
    'invest', 'invested', 'investing', 'investment', 'investments',
    'investor', 'investors',
    'commit', 'committed', 'commitment', 'commitments', 'committing',
    # 'hire' as stem covers hire/hired/hires but NOT hiring (e-drop conjugation),
    # so 'hiring' is added explicitly. 'appoint' stem covers all inflections.
    'hire', 'hiring', 'appoint',
    'policy', 'policies',
    'decision', 'decisions',
    'ad', 'ads',  # exact-match only (short root)
]
# All keywords are stems except the explicit-inflection entries
_PLANNED_STEMS = {
    'partnership', 'campaign', 'announc', 'initiat', 'pledg',
    'introduc', 'releas', 'unveil', 'sponsor', 'endors',
    'donat', 'creat', 'expand',
    'promot', 'featur', 'launch', 'collect',
    'advertisement', 'commercial', 'marketing',
    'hire', 'appoint',
}

_REACTIVE_KEYWORDS = [
    # Base-form stems (safe with \w*)
    'boycott', 'backlash', 'controvers', 'critic',
    'scandal', 'accus', 'investigat',
    'condemn', 'apolog', 'walkout', 'petition',
    'whistleblow', 'outrag', 'fallout',
    # Explicit inflections for short / ambiguous roots
    # 'protest' enumerated to avoid matching 'protestant'
    'protest', 'protests', 'protested', 'protesting', 'protester', 'protesters',
    # 'fire' → 'first/firm/fireside' so enumerate
    'fired', 'fires', 'firing',
    'resigned', 'resigns', 'resigning', 'resignation',
    'sued', 'sues', 'suing', 'lawsuit',
    'leaked', 'leaks', 'leaking',
]
_REACTIVE_STEMS = {
    'boycott', 'backlash', 'controvers', 'critic',
    'scandal', 'accus', 'investigat',
    'condemn', 'apolog', 'walkout', 'petition',
    'whistleblow', 'outrag', 'fallout',
}


def _build_keyword_re(keywords, stems=frozenset()):
    """Build word-boundary regex. Stems get \\w* to match inflections."""
    parts = []
    seen = set()
    for kw in keywords:
        if kw in seen:
            continue
        seen.add(kw)
        escaped = re.escape(kw)
        if kw in stems:
            parts.append(escaped + r'\w*')
        else:
            parts.append(escaped)
    return re.compile(r'\b(?:' + '|'.join(parts) + r')\b', re.IGNORECASE)


# Compile once — avoids false positives from substring matches
# ('issued' no longer triggers 'sued'; 'broad' no longer triggers 'ad';
# 'protestant' no longer triggers 'protest').
_PLANNED_RE = _build_keyword_re(_PLANNED_KEYWORDS, _PLANNED_STEMS)
_REACTIVE_RE = _build_keyword_re(_REACTIVE_KEYWORDS, _REACTIVE_STEMS)


def _classify_event_type(description):
    """Classify a culture-war event as PLANNED, REACTIVE, or AMBIGUOUS.

    Uses position-weighted scoring: keywords in the first half of the
    description (the action clause) get 2x weight. This prevents events
    like "Nike launches Kaepernick campaign causing backlash" from being
    classified as REACTIVE when the corporate action was clearly planned.

    Word-boundary regex prevents false positives from substring matches
    (e.g. 'issued' no longer triggers 'sued'; 'protestant' no longer
    triggers 'protest').

    Note: repeated keyword mentions in a single description each contribute
    to the score (finditer counts every match). This is intentional — a
    description mentioning "protest" three times is stronger evidence than
    one mention.
    """
    desc = str(description)
    desc_lower = desc.lower()
    midpoint = len(desc_lower) // 2

    planned_score = 0
    for m in _PLANNED_RE.finditer(desc_lower):
        planned_score += 2 if m.start() < midpoint else 1

    reactive_score = 0
    for m in _REACTIVE_RE.finditer(desc_lower):
        reactive_score += 2 if m.start() < midpoint else 1

    if planned_score > reactive_score:
        return 'PLANNED'
    elif reactive_score > planned_score:
        return 'REACTIVE'
    return 'AMBIGUOUS'


def identify_tail_firms(panel, events, top_pct=0.20):
    """
    Identify treatment firms in the top quintile of abnormal pre-event
    selling relative to their own benchmark — the firms driving the
    Mann-Whitney distributional difference.

    Returns (tail_enriched DataFrame, threshold value).
    """
    pre_days = abs(WINDOWS['PRE_FULL'][1] - WINDOWS['PRE_FULL'][0]) + 1
    bench_days = abs(WINDOWS['BENCHMARK'][1] - WINDOWS['BENCHMARK'][0]) + 1

    treatment = panel[panel['IS_TREATMENT'] == True].copy()
    treatment['PRE_DAILY'] = treatment['PRE_FULL_NET_DOLLAR_SOLD'] / pre_days
    treatment['BENCH_DAILY'] = treatment['BENCHMARK_NET_DOLLAR_SOLD'] / bench_days
    treatment['ABN_SELL'] = treatment['PRE_DAILY'] - treatment['BENCH_DAILY']

    threshold = treatment['ABN_SELL'].quantile(1 - top_pct)
    tail_firms = treatment[treatment['ABN_SELL'] >= threshold].copy()

    # Enrich with event metadata
    event_meta = events[['TICKER', 'EVENT_DATE',
                          'ESTIMATED_POLITICAL_LEANING']].copy()
    if 'CULTURE_WAR_EVENT' in events.columns:
        event_meta['EVENT_TYPE'] = events['CULTURE_WAR_EVENT'].apply(
            _classify_event_type)
        event_meta['EVENT_DESCRIPTION'] = events['CULTURE_WAR_EVENT']

    tail_enriched = tail_firms.merge(
        event_meta, on=['TICKER', 'EVENT_DATE'], how='left',
        suffixes=('', '_evt')
    )

    logger.info("  Tail diagnostic: %d/%d events in top %.0f%% "
                "(threshold=$%.0f/day)",
                len(tail_enriched), len(treatment), top_pct * 100, threshold)

    return tail_enriched, threshold


def test_leaning_in_tail(panel, events, top_pct=0.20):
    """
    Chi-square test: is political leaning distributed differently in the
    top quintile of abnormal sellers vs the rest?

    Also includes Kruskal-Wallis across leanings and per-leaning stats.
    Returns DataFrame with test results.
    """
    pre_days = abs(WINDOWS['PRE_FULL'][1] - WINDOWS['PRE_FULL'][0]) + 1
    bench_days = abs(WINDOWS['BENCHMARK'][1] - WINDOWS['BENCHMARK'][0]) + 1

    treatment = panel[panel['IS_TREATMENT'] == True].copy()
    treatment['ABN_SELL'] = (
        treatment['PRE_FULL_NET_DOLLAR_SOLD'] / pre_days -
        treatment['BENCHMARK_NET_DOLLAR_SOLD'] / bench_days
    )
    threshold = treatment['ABN_SELL'].quantile(1 - top_pct)
    treatment['IN_TAIL'] = (treatment['ABN_SELL'] >= threshold).astype(int)

    # Merge leaning from events
    treatment = treatment.merge(
        events[['TICKER', 'EVENT_DATE', 'ESTIMATED_POLITICAL_LEANING']],
        on=['TICKER', 'EVENT_DATE'], how='left', suffixes=('', '_evt')
    )

    rows = []

    # Chi-square: leaning × tail membership
    contingency = pd.crosstab(
        treatment['ESTIMATED_POLITICAL_LEANING'],
        treatment['IN_TAIL']
    )
    if contingency.shape[0] >= 2 and contingency.shape[1] >= 2:
        chi2, chi2_p, dof, _ = stats.chi2_contingency(contingency)
        rows.append({
            'TEST': 'CHI2_LEANING_VS_TAIL',
            'N': len(treatment),
            'STAT': chi2,
            'P_VALUE': chi2_p,
            'DOF': dof,
        })

    # Mean abnormal selling by leaning
    for lean, grp in treatment.groupby('ESTIMATED_POLITICAL_LEANING'):
        vals = grp['ABN_SELL'].dropna()
        tail_share = grp['IN_TAIL'].mean() * 100
        t_stat, t_p = (np.nan, np.nan)
        if len(vals) > 1:
            t_stat, t_p = stats.ttest_1samp(vals, 0)
        rows.append({
            'TEST': f'LEANING_{lean.upper().replace(" ", "_")}',
            'N': len(grp),
            'MEAN_ABN_SELL': vals.mean() if len(vals) > 0 else np.nan,
            'MEDIAN_ABN_SELL': vals.median() if len(vals) > 0 else np.nan,
            'PCT_IN_TAIL': tail_share,
            'STAT': t_stat,
            'P_VALUE': t_p,
        })

    # Kruskal-Wallis across leanings
    groups = [
        treatment.loc[treatment['ESTIMATED_POLITICAL_LEANING'] == l,
                       'ABN_SELL'].dropna().values
        for l in treatment['ESTIMATED_POLITICAL_LEANING'].dropna().unique()
    ]
    valid_groups = [g for g in groups if len(g) >= 3]
    if len(valid_groups) >= 2:
        kw_stat, kw_p = stats.kruskal(*valid_groups)
        rows.append({
            'TEST': 'KW_LEANING_VS_ABN_SELL',
            'N': sum(len(g) for g in valid_groups),
            'STAT': kw_stat,
            'P_VALUE': kw_p,
        })

    df = pd.DataFrame(rows)
    if not df.empty and df['P_VALUE'].notna().any():
        df['BH_SIGNIFICANT'] = _benjamini_hochberg(
            df['P_VALUE'].fillna(1).values
        ).astype(int)
    return df


def test_event_type_vs_selling(panel, events):
    """
    Test whether insiders sell more before PLANNED events (where they had
    advance notice of corporate action) vs REACTIVE events (external
    pressure — less foreknowledge).

    Classifies events from CULTURE_WAR_EVENT text into PLANNED/REACTIVE/
    AMBIGUOUS using keyword matching.
    """
    pre_days = abs(WINDOWS['PRE_FULL'][1] - WINDOWS['PRE_FULL'][0]) + 1
    bench_days = abs(WINDOWS['BENCHMARK'][1] - WINDOWS['BENCHMARK'][0]) + 1

    treatment = panel[panel['IS_TREATMENT'] == True].copy()
    treatment['ABN_SELL'] = (
        treatment['PRE_FULL_NET_DOLLAR_SOLD'] / pre_days -
        treatment['BENCHMARK_NET_DOLLAR_SOLD'] / bench_days
    )

    # Classify events and merge
    evt = events[['TICKER', 'EVENT_DATE']].copy()
    if 'CULTURE_WAR_EVENT' in events.columns:
        evt['EVENT_TYPE'] = events['CULTURE_WAR_EVENT'].apply(
            _classify_event_type)
    else:
        logger.warning("No CULTURE_WAR_EVENT column — skipping event type test")
        return pd.DataFrame()

    treatment = treatment.merge(
        evt, on=['TICKER', 'EVENT_DATE'], how='left', suffixes=('', '_evt')
    )

    rows = []
    planned = treatment.loc[treatment['EVENT_TYPE'] == 'PLANNED',
                             'ABN_SELL'].dropna()
    reactive = treatment.loc[treatment['EVENT_TYPE'] == 'REACTIVE',
                              'ABN_SELL'].dropna()
    ambiguous = treatment.loc[treatment['EVENT_TYPE'] == 'AMBIGUOUS',
                               'ABN_SELL'].dropna()

    logger.info("  Event type classification: %d PLANNED, %d REACTIVE, "
                "%d AMBIGUOUS", len(planned), len(reactive), len(ambiguous))

    # Two-sample: PLANNED vs REACTIVE
    if len(planned) >= 5 and len(reactive) >= 5:
        t_stat, t_p = stats.ttest_ind(planned, reactive, equal_var=False)
        u_stat, u_p = stats.mannwhitneyu(planned, reactive,
                                          alternative='greater')
        pooled_sd = np.sqrt(
            (planned.std() ** 2 + reactive.std() ** 2) / 2)
        cohen_d = ((planned.mean() - reactive.mean()) / pooled_sd
                   if pooled_sd > 0 else 0)
        rows.append({
            'COMPARISON': 'PLANNED_VS_REACTIVE',
            'N_A': len(planned),
            'N_B': len(reactive),
            'MEAN_A': planned.mean(),
            'MEAN_B': reactive.mean(),
            'COHEN_D': cohen_d,
            'T_STAT': t_stat,
            'T_PVALUE': t_p,
            'MW_STAT': u_stat,
            'MW_PVALUE': u_p,
        })

    # One-sample tests: each type vs zero
    for label, vals in [('PLANNED', planned), ('REACTIVE', reactive),
                         ('AMBIGUOUS', ambiguous)]:
        if len(vals) >= 5:
            t_s, t_p = stats.ttest_1samp(vals, 0)
            d = vals.mean() / vals.std() if vals.std() > 0 else 0
            rows.append({
                'COMPARISON': f'{label}_VS_ZERO',
                'N_A': len(vals),
                'MEAN_A': vals.mean(),
                'COHEN_D': d,
                'T_STAT': t_s,
                'T_PVALUE': t_p,
            })

    df = pd.DataFrame(rows)
    if not df.empty:
        # BH on whichever p-value columns exist
        for pcol in ['T_PVALUE', 'MW_PVALUE']:
            if pcol in df.columns and df[pcol].notna().any():
                df[f'{pcol}_BH_SIG'] = _benjamini_hochberg(
                    df[pcol].fillna(1).values
                ).astype(int)
    return df


def test_leaning_x_event_type(panel, events):
    """
    2×3 interaction: Event Type (Planned/Reactive/Ambiguous) ×
    Political Leaning (Conservative/Liberal/Mixed).

    Theory: highest abnormal selling in CONSERVATIVE × PLANNED (insiders
    at firms making deliberate right-leaning political stands have the
    most foreknowledge). Lowest in LIBERAL × REACTIVE.

    Includes cell-level t-tests and OLS with interaction terms.
    """
    pre_days = abs(WINDOWS['PRE_FULL'][1] - WINDOWS['PRE_FULL'][0]) + 1
    bench_days = abs(WINDOWS['BENCHMARK'][1] - WINDOWS['BENCHMARK'][0]) + 1

    treatment = panel[panel['IS_TREATMENT'] == True].copy()
    treatment['ABN_SELL'] = (
        treatment['PRE_FULL_NET_DOLLAR_SOLD'] / pre_days -
        treatment['BENCHMARK_NET_DOLLAR_SOLD'] / bench_days
    )

    # Merge leaning + event type
    evt = events[['TICKER', 'EVENT_DATE',
                   'ESTIMATED_POLITICAL_LEANING']].copy()
    if 'CULTURE_WAR_EVENT' in events.columns:
        evt['EVENT_TYPE'] = events['CULTURE_WAR_EVENT'].apply(
            _classify_event_type)
    else:
        return pd.DataFrame()

    treatment = treatment.merge(
        evt, on=['TICKER', 'EVENT_DATE'], how='left', suffixes=('', '_evt')
    )

    rows = []
    for lean in treatment['ESTIMATED_POLITICAL_LEANING'].dropna().unique():
        for etype in ['PLANNED', 'REACTIVE', 'AMBIGUOUS']:
            cell = treatment.loc[
                (treatment['ESTIMATED_POLITICAL_LEANING'] == lean) &
                (treatment['EVENT_TYPE'] == etype),
                'ABN_SELL'
            ].dropna()

            if len(cell) < 3:
                continue

            t_stat, t_p = stats.ttest_1samp(cell, 0)
            d = cell.mean() / cell.std() if cell.std() > 0 else 0
            rows.append({
                'LEAN': lean,
                'EVENT_TYPE': etype,
                'N': len(cell),
                'MEAN_ABN_SELL': cell.mean(),
                'MEDIAN_ABN_SELL': cell.median(),
                'PCT_POSITIVE': (cell > 0).mean() * 100,
                'COHEN_D': d,
                'T_STAT': t_stat,
                'P_VALUE': t_p,
            })

    df = pd.DataFrame(rows)
    if not df.empty and df['P_VALUE'].notna().any():
        df['BH_SIGNIFICANT'] = _benjamini_hochberg(
            df['P_VALUE'].fillna(1).values
        ).astype(int)

    # OLS with interaction terms for formal test
    try:
        reg_df = treatment.dropna(
            subset=['ABN_SELL', 'ESTIMATED_POLITICAL_LEANING', 'EVENT_TYPE']
        ).copy()
        if len(reg_df) < 10:
            return df

        lean_dummies = pd.get_dummies(
            reg_df['ESTIMATED_POLITICAL_LEANING'],
            prefix='LEAN', drop_first=True
        ).astype(float)
        type_dummies = pd.get_dummies(
            reg_df['EVENT_TYPE'],
            prefix='TYPE', drop_first=True
        ).astype(float)

        # Interaction terms
        interaction_cols = {}
        for lc in lean_dummies.columns:
            for tc in type_dummies.columns:
                interaction_cols[f'{lc}_X_{tc}'] = (
                    lean_dummies[lc] * type_dummies[tc])
        interactions = pd.DataFrame(interaction_cols, index=reg_df.index)

        X = sm.add_constant(
            pd.concat([lean_dummies, type_dummies, interactions],
                      axis=1).astype(float)
        )
        y = reg_df['ABN_SELL'].astype(float)
        model = sm.OLS(y, X).fit()

        for var in model.params.index:
            if '_X_' in var or var == 'const':
                interaction_row = {
                    'LEAN': 'INTERACTION',
                    'EVENT_TYPE': var,
                    'N': int(model.nobs),
                    'MEAN_ABN_SELL': model.params[var],
                    'COHEN_D': np.nan,
                    'T_STAT': model.tvalues[var],
                    'P_VALUE': model.pvalues[var],
                }
                df = pd.concat([df, pd.DataFrame([interaction_row])],
                               ignore_index=True)

        # Store R-squared on last row
        if not df.empty:
            df.loc[df.index[-1], 'R_SQUARED'] = model.rsquared

    except Exception as e:
        logger.warning("Leaning × event type interaction OLS failed: %s", e)

    return df


def compute_tail_diagnostic(panel, events, form4, top_pct=0.20):
    """
    Master function: run all three tail-diagnostic tests plus the
    enriched tail-firm identification.

    Returns dict with keys:
      'tail_firms'   — enriched top-quintile firms
      'leaning_test' — chi-square / KW on leaning in tail
      'event_type'   — planned vs reactive selling comparison
      'interaction'  — leaning × event type 2×3 + OLS
    """
    logger.info("Running tail diagnostic (top %.0f%%)...", top_pct * 100)

    tail_enriched, threshold = identify_tail_firms(
        panel, events, top_pct=top_pct)

    logger.info("Test 1: Leaning distribution in tail...")
    leaning_test = test_leaning_in_tail(panel, events, top_pct=top_pct)

    logger.info("Test 2: Planned vs reactive event type...")
    event_type = test_event_type_vs_selling(panel, events)

    logger.info("Test 3: Leaning × event type interaction...")
    interaction = test_leaning_x_event_type(panel, events)

    # Log summary
    if not leaning_test.empty:
        chi2_row = leaning_test[leaning_test['TEST'] == 'CHI2_LEANING_VS_TAIL']
        if not chi2_row.empty:
            logger.info("  Chi-square (leaning vs tail): chi2=%.2f, p=%.4f",
                        chi2_row.iloc[0]['STAT'], chi2_row.iloc[0]['P_VALUE'])

    if not event_type.empty:
        pvr = event_type[event_type['COMPARISON'] == 'PLANNED_VS_REACTIVE']
        if not pvr.empty:
            logger.info("  Planned vs Reactive: d=%.3f, t-p=%.4f, MW-p=%.4f",
                        pvr.iloc[0]['COHEN_D'],
                        pvr.iloc[0]['T_PVALUE'],
                        pvr.iloc[0]['MW_PVALUE'])

    if not interaction.empty:
        cells = interaction[interaction['LEAN'] != 'INTERACTION']
        if not cells.empty:
            best = cells.loc[cells['MEAN_ABN_SELL'].idxmax()]
            logger.info("  Highest cell: %s × %s (d=%.3f, p=%.4f, n=%d)",
                        best.get('LEAN', '?'), best.get('EVENT_TYPE', '?'),
                        best.get('COHEN_D', 0), best.get('P_VALUE', 1),
                        int(best.get('N', 0)))

    return {
        'tail_firms': tail_enriched,
        'leaning_test': leaning_test,
        'event_type': event_type,
        'interaction': interaction,
    }


def compute_conservative_planned_deep_dive(panel, events, form4):
    """
    Deep dive on the Conservative × Planned cell (d=0.751).

    For each event in this cell, produces a case-study row with:
    - Individual firm abnormal selling magnitude
    - Pre-event and benchmark selling
    - CAR (market reaction severity)
    - C-suite insider presence and count
    - Insider trade timing (median days before event)
    - Transaction size relative to firm baseline
    - Power diagnostic: observed d, n needed for 80% power, actual n

    This table supports qualitative case-study discussion in the
    dissertation even when the cell is too small for inference.
    """
    pre_days = abs(WINDOWS['PRE_FULL'][1] - WINDOWS['PRE_FULL'][0]) + 1
    bench_days = abs(WINDOWS['BENCHMARK'][1] - WINDOWS['BENCHMARK'][0]) + 1

    treatment = panel[panel['IS_TREATMENT'] == True].copy()
    treatment['ABN_SELL'] = (
        treatment['PRE_FULL_NET_DOLLAR_SOLD'] / pre_days -
        treatment['BENCHMARK_NET_DOLLAR_SOLD'] / bench_days
    )
    treatment['PRE_DAILY'] = treatment['PRE_FULL_NET_DOLLAR_SOLD'] / pre_days
    treatment['BENCH_DAILY'] = treatment['BENCHMARK_NET_DOLLAR_SOLD'] / bench_days

    # Merge event metadata + classify event type
    evt = events[['TICKER', 'EVENT_DATE',
                   'ESTIMATED_POLITICAL_LEANING',
                   'COMPANY']].copy()
    if 'CULTURE_WAR_EVENT' in events.columns:
        evt['EVENT_TYPE'] = events['CULTURE_WAR_EVENT'].apply(
            _classify_event_type)
        evt['EVENT_DESCRIPTION'] = events['CULTURE_WAR_EVENT']
    else:
        return pd.DataFrame()

    merged = treatment.merge(
        evt, on=['TICKER', 'EVENT_DATE'], how='left', suffixes=('', '_evt')
    )

    cell = merged[
        (merged['ESTIMATED_POLITICAL_LEANING'] == 'Conservative') &
        (merged['EVENT_TYPE'] == 'PLANNED')
    ].copy()

    if cell.empty:
        logger.warning("  No Conservative × Planned events found")
        return pd.DataFrame()

    # Enrich with insider-level detail from form4
    case_rows = []
    for _, ev in cell.iterrows():
        ticker = ev['TICKER']
        event_date = pd.Timestamp(ev['EVENT_DATE'])
        pre_start = event_date + pd.Timedelta(days=WINDOWS['PRE_FULL'][0])
        pre_end = event_date + pd.Timedelta(days=WINDOWS['PRE_FULL'][1])

        # Get pre-event trades for this firm
        mask = (
            (form4['ticker'] == ticker) &
            (form4['transaction_date'] >= pre_start) &
            (form4['transaction_date'] <= pre_end)
        )
        pre_trades = form4.loc[mask]

        # C-suite detection
        csuite_titles = {'ceo', 'cfo', 'coo', 'cto', 'president',
                         'chief executive', 'chief financial',
                         'chief operating', 'chief technology'}
        n_csuite = 0
        csuite_present = False
        if 'insider_title' in pre_trades.columns:
            titles = pre_trades['insider_title'].dropna().str.lower()
            csuite_mask = titles.apply(
                lambda t: any(cs in t for cs in csuite_titles))
            n_csuite = csuite_mask.sum()
            csuite_present = n_csuite > 0

        # Median days before event
        if not pre_trades.empty and 'transaction_date' in pre_trades.columns:
            days_before = (event_date - pre_trades['transaction_date']).dt.days
            median_days = days_before.median()
        else:
            median_days = np.nan

        # Sell count in pre-event (use _trade_type consistent with panel builder)
        n_sells = len(pre_trades[pre_trades['_trade_type'] == 'sell']) \
            if '_trade_type' in pre_trades.columns else 0
        n_total = len(pre_trades)

        case_rows.append({
            'TICKER': ticker,
            'COMPANY': ev.get('COMPANY', ''),
            'EVENT_DATE': event_date.strftime('%Y-%m-%d'),
            'EVENT_DESCRIPTION': str(ev.get('EVENT_DESCRIPTION', ''))[:120],
            'ABN_SELL_DAILY': ev['ABN_SELL'],
            'PRE_DAILY_SELL': ev['PRE_DAILY'],
            'BENCH_DAILY_SELL': ev['BENCH_DAILY'],
            'PRE_FULL_NET_DOLLAR': ev.get('PRE_FULL_NET_DOLLAR_SOLD', np.nan),
            'CAR_POST': ev.get('CAR_POST', np.nan),
            'N_PRE_TRADES': n_total,
            'N_PRE_SELLS': n_sells,
            'N_CSUITE_TRADES': n_csuite,
            'CSUITE_PRESENT': csuite_present,
            'MEDIAN_DAYS_BEFORE': median_days,
        })

    case_df = pd.DataFrame(case_rows)
    case_df = case_df.sort_values('ABN_SELL_DAILY', ascending=False)

    # Add power diagnostic as summary row
    n_obs = len(case_df)
    cell_vals = cell['ABN_SELL'].dropna()
    if len(cell_vals) > 1:
        observed_d = cell_vals.mean() / cell_vals.std()
    else:
        observed_d = np.nan

    # Min n for 80% power at observed d
    if not np.isnan(observed_d) and observed_d > 0:
        z_a = stats.norm.ppf(1 - 0.05 / 2)
        z_b = stats.norm.ppf(0.80)
        n_needed = int(np.ceil(((z_a + z_b) / observed_d) ** 2))
    else:
        n_needed = np.nan

    # Append power diagnostic as metadata row
    power_row = {
        'TICKER': '_POWER_DIAGNOSTIC',
        'COMPANY': '',
        'EVENT_DATE': '',
        'EVENT_DESCRIPTION': (
            f'Cell n={n_obs}, d={observed_d:.3f}, '
            f'need n={n_needed} for 80% power'
            if not np.isnan(observed_d) else f'Cell n={n_obs}, d=N/A'
        ),
        'ABN_SELL_DAILY': cell_vals.mean() if len(cell_vals) > 0 else np.nan,
        'PRE_DAILY_SELL': np.nan,
        'BENCH_DAILY_SELL': np.nan,
        'PRE_FULL_NET_DOLLAR': np.nan,
        'CAR_POST': np.nan,
        'N_PRE_TRADES': n_obs,
        'N_PRE_SELLS': n_needed if not np.isnan(n_needed) else 0,
        'N_CSUITE_TRADES': 0,
        'CSUITE_PRESENT': False,
        'MEDIAN_DAYS_BEFORE': observed_d,
    }
    case_df = pd.concat([case_df, pd.DataFrame([power_row])],
                        ignore_index=True)

    logger.info("  Conservative × Planned deep dive: %d events, d=%.3f, "
                "need n=%s for 80%% power",
                n_obs,
                observed_d if not np.isnan(observed_d) else 0,
                str(n_needed) if not np.isnan(n_needed) else '?')

    return case_df


# ═══════════════════════════════════════════════════════════════════════
# ROBUSTNESS EXTENSIONS (29-38)
# ═══════════════════════════════════════════════════════════════════════

def _get_market_cap(stock_data, ticker, event_date, lookback_days=30):
    """Estimate market cap from stock data (price × shares outstanding proxy).

    Uses average ADJ_CLOSE in the 30 days before the event as a rough proxy.
    Returns NaN if no data.
    """
    if stock_data is None or stock_data.empty:
        return np.nan
    mask = (
        (stock_data['TICKER'] == ticker) &
        (stock_data['DATE'] >= event_date - pd.Timedelta(days=lookback_days)) &
        (stock_data['DATE'] < event_date)
    )
    sub = stock_data.loc[mask]
    if sub.empty or 'ADJ_CLOSE' not in sub.columns:
        return np.nan
    return sub['ADJ_CLOSE'].mean()


def compute_tail_logit(panel, events, stock_data=None, top_pct=0.20):
    """
    Analysis 29: Industry-controlled logit for tail membership.

    logit(IN_TAIL) ~ LEAN + NAICS_2DIGIT + log(PRICE_PROXY) + YEAR_FE

    Tests whether Liberal overrepresentation in the tail survives after
    controlling for industry, firm size, and year effects. If the Liberal
    coefficient disappears, the finding is industry-in-political-clothing.
    """
    logger.info("Computing tail-membership logit with industry controls...")

    pre_days = abs(WINDOWS['PRE_FULL'][1] - WINDOWS['PRE_FULL'][0]) + 1
    bench_days = abs(WINDOWS['BENCHMARK'][1] - WINDOWS['BENCHMARK'][0]) + 1

    treatment = panel[panel['IS_TREATMENT'] == True].copy()
    treatment['ABN_SELL'] = (
        treatment['PRE_FULL_NET_DOLLAR_SOLD'] / pre_days -
        treatment['BENCHMARK_NET_DOLLAR_SOLD'] / bench_days
    )
    threshold = treatment['ABN_SELL'].quantile(1 - top_pct)
    treatment['IN_TAIL'] = (treatment['ABN_SELL'] >= threshold).astype(int)

    # Merge event metadata (leaning, NAICS, year)
    evt = events[['TICKER', 'EVENT_DATE', 'ESTIMATED_POLITICAL_LEANING']].copy()
    if 'NAICS_CODE' in events.columns:
        evt['NAICS_2D'] = events['NAICS_CODE'].astype(str).str[:2]
    elif 'NAICS Code' in events.columns:
        evt['NAICS_2D'] = events['NAICS Code'].astype(str).str[:2]
    else:
        evt['NAICS_2D'] = '99'

    if 'INDUSTRY' in events.columns:
        evt['INDUSTRY'] = events['INDUSTRY']
    elif 'Industry' in events.columns:
        evt['INDUSTRY'] = events['Industry']
    else:
        evt['INDUSTRY'] = 'Unknown'

    treatment = treatment.merge(
        evt, on=['TICKER', 'EVENT_DATE'], how='left', suffixes=('', '_evt')
    )

    # Year from event date
    treatment['YEAR'] = treatment['EVENT_DATE'].dt.year

    # Price proxy for size (log of mean price in benchmark window)
    if stock_data is not None and not stock_data.empty:
        price_vals = []
        for _, row in treatment.iterrows():
            price_vals.append(_get_market_cap(stock_data, row['TICKER'], row['EVENT_DATE']))
        treatment['PRICE_PROXY'] = price_vals
    else:
        treatment['PRICE_PROXY'] = np.nan

    treatment['LOG_PRICE'] = np.log(treatment['PRICE_PROXY'].clip(lower=1))
    treatment.loc[treatment['PRICE_PROXY'].isna(), 'LOG_PRICE'] = np.nan

    rows = []

    # --- Model 1: Lean-only (replicates chi-square as logit) ---
    try:
        reg = treatment.dropna(subset=['IN_TAIL', 'ESTIMATED_POLITICAL_LEANING']).copy()
        lean_dummies = pd.get_dummies(
            reg['ESTIMATED_POLITICAL_LEANING'], prefix='LEAN', drop_first=True
        ).astype(float)
        X1 = sm.add_constant(lean_dummies)
        m1 = sm.Logit(reg['IN_TAIL'].astype(float), X1).fit(disp=0)
        for var in m1.params.index:
            rows.append({
                'MODEL': 'LEAN_ONLY',
                'VARIABLE': var,
                'COEFFICIENT': m1.params[var],
                'STD_ERROR': m1.bse[var],
                'Z_STAT': m1.tvalues[var],
                'P_VALUE': m1.pvalues[var],
                'N_OBS': int(m1.nobs),
                'PSEUDO_R2': m1.prsquared,
            })
    except Exception as e:
        logger.warning("Tail logit model 1 failed: %s", e)

    # --- Model 2: Lean + industry ---
    try:
        reg2 = treatment.dropna(
            subset=['IN_TAIL', 'ESTIMATED_POLITICAL_LEANING', 'NAICS_2D']
        ).copy()
        # Drop NAICS codes with < 3 observations to avoid perfect separation
        naics_counts = reg2['NAICS_2D'].value_counts()
        valid_naics = naics_counts[naics_counts >= 3].index
        reg2 = reg2[reg2['NAICS_2D'].isin(valid_naics)]

        lean_d = pd.get_dummies(
            reg2['ESTIMATED_POLITICAL_LEANING'], prefix='LEAN', drop_first=True
        ).astype(float)
        naics_d = pd.get_dummies(
            reg2['NAICS_2D'], prefix='NAICS', drop_first=True
        ).astype(float)

        X2 = sm.add_constant(pd.concat([lean_d, naics_d], axis=1))
        m2 = sm.Logit(reg2['IN_TAIL'].astype(float), X2).fit(disp=0, maxiter=100)
        for var in m2.params.index:
            rows.append({
                'MODEL': 'LEAN_INDUSTRY',
                'VARIABLE': var,
                'COEFFICIENT': m2.params[var],
                'STD_ERROR': m2.bse[var],
                'Z_STAT': m2.tvalues[var],
                'P_VALUE': m2.pvalues[var],
                'N_OBS': int(m2.nobs),
                'PSEUDO_R2': m2.prsquared,
            })
    except Exception as e:
        logger.warning("Tail logit model 2 failed: %s", e)

    # --- Model 3: Full (lean + industry + size + year FE) ---
    try:
        reg3 = treatment.dropna(
            subset=['IN_TAIL', 'ESTIMATED_POLITICAL_LEANING', 'NAICS_2D', 'LOG_PRICE']
        ).copy()
        naics_counts3 = reg3['NAICS_2D'].value_counts()
        valid_naics3 = naics_counts3[naics_counts3 >= 3].index
        reg3 = reg3[reg3['NAICS_2D'].isin(valid_naics3)]

        lean_d3 = pd.get_dummies(
            reg3['ESTIMATED_POLITICAL_LEANING'], prefix='LEAN', drop_first=True
        ).astype(float)
        naics_d3 = pd.get_dummies(
            reg3['NAICS_2D'], prefix='NAICS', drop_first=True
        ).astype(float)
        year_d3 = pd.get_dummies(
            reg3['YEAR'], prefix='YEAR', drop_first=True
        ).astype(float)

        X3 = sm.add_constant(
            pd.concat([lean_d3, naics_d3, reg3[['LOG_PRICE']].reset_index(drop=True),
                        year_d3], axis=1).reset_index(drop=True)
        )
        y3 = reg3['IN_TAIL'].astype(float).reset_index(drop=True)
        m3 = sm.Logit(y3, X3).fit(disp=0, maxiter=200)
        for var in m3.params.index:
            rows.append({
                'MODEL': 'FULL',
                'VARIABLE': var,
                'COEFFICIENT': m3.params[var],
                'STD_ERROR': m3.bse[var],
                'Z_STAT': m3.tvalues[var],
                'P_VALUE': m3.pvalues[var],
                'N_OBS': int(m3.nobs),
                'PSEUDO_R2': m3.prsquared,
            })
    except Exception as e:
        logger.warning("Tail logit model 3 (full) failed: %s", e)

    df = pd.DataFrame(rows)
    if not df.empty:
        logger.info("  Tail logit: %d models, %d coefficients",
                    df['MODEL'].nunique(), len(df))
        # Check if Liberal survives
        lib_full = df[(df['MODEL'] == 'FULL') & (df['VARIABLE'] == 'LEAN_Liberal')]
        lib_lean = df[(df['MODEL'] == 'LEAN_ONLY') & (df['VARIABLE'] == 'LEAN_Liberal')]
        if not lib_lean.empty:
            logger.info("    Liberal (lean-only): coef=%.3f, p=%.4f",
                        lib_lean.iloc[0]['COEFFICIENT'], lib_lean.iloc[0]['P_VALUE'])
        if not lib_full.empty:
            logger.info("    Liberal (full): coef=%.3f, p=%.4f",
                        lib_full.iloc[0]['COEFFICIENT'], lib_full.iloc[0]['P_VALUE'])
    return df


def compute_propensity_score_matching(panel, events, form4, stock_data=None, top_pct=0.20):
    """
    Analysis 30: Propensity score matching for tail analysis.

    Construct a propensity score for 'firm likely to experience a CW event'
    using industry + size + prior insider activity, then re-do the tail
    analysis conditional on matched pairs with similar propensity.
    """
    logger.info("Computing propensity score matching...")

    pre_days = abs(WINDOWS['PRE_FULL'][1] - WINDOWS['PRE_FULL'][0]) + 1
    bench_days = abs(WINDOWS['BENCHMARK'][1] - WINDOWS['BENCHMARK'][0]) + 1

    treatment = panel[panel['IS_TREATMENT'] == True].copy()
    control = panel[panel['IS_TREATMENT'] == False].copy()
    full = pd.concat([treatment, control], ignore_index=True)

    full['ABN_SELL'] = (
        full['PRE_FULL_NET_DOLLAR_SOLD'] / pre_days -
        full['BENCHMARK_NET_DOLLAR_SOLD'] / bench_days
    )

    # Add covariates for propensity model
    evt = events[['TICKER', 'EVENT_DATE', 'ESTIMATED_POLITICAL_LEANING']].copy()
    if 'NAICS Code' in events.columns:
        evt['NAICS_2D'] = events['NAICS Code'].astype(str).str[:2]
    elif 'NAICS_CODE' in events.columns:
        evt['NAICS_2D'] = events['NAICS_CODE'].astype(str).str[:2]
    else:
        evt['NAICS_2D'] = '99'

    full = full.merge(evt, on=['TICKER', 'EVENT_DATE'], how='left', suffixes=('', '_evt'))

    # Benchmark activity as covariate
    full['BENCH_DAILY'] = full['BENCHMARK_NET_DOLLAR_SOLD'] / bench_days
    full['N_BENCH_TXN'] = full.get('BENCHMARK_N_TRANSACTIONS', 0)
    full['YEAR'] = full['EVENT_DATE'].dt.year

    # Price proxy
    if stock_data is not None and not stock_data.empty:
        prices = []
        for _, row in full.iterrows():
            prices.append(_get_market_cap(stock_data, row['TICKER'], row['EVENT_DATE']))
        full['LOG_PRICE'] = np.log(pd.Series(prices).clip(lower=1))
    else:
        full['LOG_PRICE'] = np.nan

    rows = []

    # Estimate propensity score: P(treatment | covariates)
    try:
        ps_df = full.dropna(subset=['IS_TREATMENT', 'BENCH_DAILY', 'LOG_PRICE']).copy()
        if len(ps_df) < 20:
            logger.warning("Propensity score: too few observations (%d)", len(ps_df))
            return pd.DataFrame()

        naics_d = pd.get_dummies(ps_df['NAICS_2D'], prefix='NAICS', drop_first=True).astype(float)
        # Drop sparse categories
        naics_d = naics_d.loc[:, naics_d.sum() >= 3]

        X_ps = sm.add_constant(
            pd.concat([
                naics_d.reset_index(drop=True),
                ps_df[['BENCH_DAILY', 'LOG_PRICE']].reset_index(drop=True),
            ], axis=1)
        )
        y_ps = ps_df['IS_TREATMENT'].astype(float).reset_index(drop=True)

        ps_model = sm.Logit(y_ps, X_ps).fit(disp=0, maxiter=100)
        ps_df['PSCORE'] = ps_model.predict(X_ps)

        # Nearest-neighbor matching (1:1, caliper = 0.1)
        treat_ps = ps_df[ps_df['IS_TREATMENT'] == True].copy()
        ctrl_ps = ps_df[ps_df['IS_TREATMENT'] == False].copy()

        matched_pairs = []
        used_ctrl = set()
        for idx, t_row in treat_ps.iterrows():
            diffs = (ctrl_ps['PSCORE'] - t_row['PSCORE']).abs()
            diffs = diffs[~diffs.index.isin(used_ctrl)]
            if diffs.empty:
                continue
            best_idx = diffs.idxmin()
            if diffs[best_idx] <= 0.1:
                matched_pairs.append((idx, best_idx))
                used_ctrl.add(best_idx)

        logger.info("  Propensity matching: %d/%d treatment events matched (caliper=0.1)",
                    len(matched_pairs), len(treat_ps))

        if len(matched_pairs) < 10:
            rows.append({
                'TEST': 'PS_MATCHING_FAILED',
                'N_MATCHED': len(matched_pairs),
                'N_TREATMENT': len(treat_ps),
                'NOTE': 'Too few matched pairs for analysis',
            })
        else:
            # Re-do tail analysis on matched sample
            matched_t_idx = [p[0] for p in matched_pairs]
            matched_c_idx = [p[1] for p in matched_pairs]
            matched_treat = ps_df.loc[matched_t_idx]
            matched_ctrl = ps_df.loc[matched_c_idx]

            # Tail test on matched treatment
            threshold = matched_treat['ABN_SELL'].quantile(1 - top_pct)
            matched_treat = matched_treat.copy()
            matched_treat['IN_TAIL'] = (matched_treat['ABN_SELL'] >= threshold).astype(int)

            # Merge leaning
            lean_col = 'ESTIMATED_POLITICAL_LEANING'
            if lean_col not in matched_treat.columns:
                matched_treat = matched_treat.merge(
                    events[['TICKER', 'EVENT_DATE', lean_col]],
                    on=['TICKER', 'EVENT_DATE'], how='left'
                )

            # Chi-square on matched sample
            contingency = pd.crosstab(
                matched_treat[lean_col],
                matched_treat['IN_TAIL']
            )
            if contingency.shape[0] >= 2 and contingency.shape[1] >= 2:
                chi2, chi2_p, dof, _ = stats.chi2_contingency(contingency)
                rows.append({
                    'TEST': 'PS_CHI2_LEANING_VS_TAIL',
                    'STAT': chi2,
                    'P_VALUE': chi2_p,
                    'N_MATCHED': len(matched_pairs),
                    'DOF': dof,
                })

            # Per-leaning tail share
            for lean, grp in matched_treat.groupby(lean_col):
                rows.append({
                    'TEST': f'PS_LEANING_{lean.upper().replace(" ", "_")}',
                    'N_MATCHED': len(grp),
                    'MEAN_ABN_SELL': grp['ABN_SELL'].mean(),
                    'PCT_IN_TAIL': grp['IN_TAIL'].mean() * 100,
                })

            # Pscore balance check (mean difference in pscore)
            t_ps_vals = ps_df.loc[matched_t_idx, 'PSCORE']
            c_ps_vals = ps_df.loc[matched_c_idx, 'PSCORE']
            rows.append({
                'TEST': 'PS_BALANCE',
                'STAT': abs(t_ps_vals.mean() - c_ps_vals.mean()),
                'N_MATCHED': len(matched_pairs),
                'NOTE': f'Mean pscore: T={t_ps_vals.mean():.3f}, C={c_ps_vals.mean():.3f}',
            })

    except Exception as e:
        logger.warning("Propensity score matching failed: %s", e)
        rows.append({'TEST': 'PS_MATCHING_ERROR', 'NOTE': str(e)})

    return pd.DataFrame(rows)


def compute_winsorized_tail(panel, events, top_pct=0.20):
    """
    Analysis 31: Winsorize abnormal selling and re-run tail chi-square.

    Clips at 1% and 5% to test whether the Liberal overrepresentation
    is driven by a few mega-trades or is a population pattern.
    """
    logger.info("Computing winsorized tail analysis...")

    pre_days = abs(WINDOWS['PRE_FULL'][1] - WINDOWS['PRE_FULL'][0]) + 1
    bench_days = abs(WINDOWS['BENCHMARK'][1] - WINDOWS['BENCHMARK'][0]) + 1

    treatment = panel[panel['IS_TREATMENT'] == True].copy()
    treatment['ABN_SELL'] = (
        treatment['PRE_FULL_NET_DOLLAR_SOLD'] / pre_days -
        treatment['BENCHMARK_NET_DOLLAR_SOLD'] / bench_days
    )

    # Merge leaning
    treatment = treatment.merge(
        events[['TICKER', 'EVENT_DATE', 'ESTIMATED_POLITICAL_LEANING']],
        on=['TICKER', 'EVENT_DATE'], how='left', suffixes=('', '_evt')
    )

    rows = []

    for clip_pct in [0.0, 0.01, 0.05]:
        label = f'WINSOR_{int(clip_pct * 100)}PCT' if clip_pct > 0 else 'RAW'

        if clip_pct > 0:
            lo = treatment['ABN_SELL'].quantile(clip_pct)
            hi = treatment['ABN_SELL'].quantile(1 - clip_pct)
            abn = treatment['ABN_SELL'].clip(lower=lo, upper=hi)
        else:
            abn = treatment['ABN_SELL']

        threshold = abn.quantile(1 - top_pct)
        in_tail = (abn >= threshold).astype(int)

        # Chi-square: leaning × tail
        contingency = pd.crosstab(
            treatment['ESTIMATED_POLITICAL_LEANING'], in_tail
        )
        if contingency.shape[0] >= 2 and contingency.shape[1] >= 2:
            chi2, chi2_p, dof, _ = stats.chi2_contingency(contingency)
            rows.append({
                'WINSORIZATION': label,
                'TEST': 'CHI2',
                'STAT': chi2,
                'P_VALUE': chi2_p,
                'N': len(treatment),
                'CLIP_PCT': clip_pct,
            })

        # Per-leaning tail share
        for lean in treatment['ESTIMATED_POLITICAL_LEANING'].dropna().unique():
            mask = treatment['ESTIMATED_POLITICAL_LEANING'] == lean
            lean_tail = in_tail[mask]
            rows.append({
                'WINSORIZATION': label,
                'TEST': f'LEAN_{lean.upper().replace(" ", "_")}',
                'PCT_IN_TAIL': lean_tail.mean() * 100 if len(lean_tail) > 0 else np.nan,
                'MEAN_ABN_SELL': abn[mask].mean(),
                'N': int(mask.sum()),
                'CLIP_PCT': clip_pct,
            })

    df = pd.DataFrame(rows)
    if not df.empty:
        chi2_rows = df[df['TEST'] == 'CHI2']
        for _, r in chi2_rows.iterrows():
            logger.info("  %s chi2=%.2f, p=%.4f",
                        r['WINSORIZATION'], r['STAT'], r['P_VALUE'])
    return df


def compute_size_stratified_tail(panel, events, stock_data=None, top_pct=0.20):
    """
    Analysis 32: Split tail analysis by firm size terciles.

    If Liberal-tail only exists in top tercile → big-tech finding.
    If it holds across all three → more robust political pattern.
    """
    logger.info("Computing size-stratified tail analysis...")

    pre_days = abs(WINDOWS['PRE_FULL'][1] - WINDOWS['PRE_FULL'][0]) + 1
    bench_days = abs(WINDOWS['BENCHMARK'][1] - WINDOWS['BENCHMARK'][0]) + 1

    treatment = panel[panel['IS_TREATMENT'] == True].copy()
    treatment['ABN_SELL'] = (
        treatment['PRE_FULL_NET_DOLLAR_SOLD'] / pre_days -
        treatment['BENCHMARK_NET_DOLLAR_SOLD'] / bench_days
    )

    # Merge leaning
    treatment = treatment.merge(
        events[['TICKER', 'EVENT_DATE', 'ESTIMATED_POLITICAL_LEANING']],
        on=['TICKER', 'EVENT_DATE'], how='left', suffixes=('', '_evt')
    )

    # Size proxy: benchmark-window dollar volume as proxy for firm size
    # (larger firms have more trading activity)
    treatment['SIZE_PROXY'] = treatment['BENCHMARK_NET_DOLLAR_SOLD'].abs()

    # If stock_data available, use price as better size proxy
    if stock_data is not None and not stock_data.empty:
        prices = []
        for _, row in treatment.iterrows():
            prices.append(_get_market_cap(stock_data, row['TICKER'], row['EVENT_DATE']))
        price_series = pd.Series(prices, index=treatment.index)
        if price_series.notna().sum() >= 10:
            treatment['SIZE_PROXY'] = price_series

    rows = []

    # Create size terciles
    try:
        treatment['SIZE_TERCILE'] = pd.qcut(
            treatment['SIZE_PROXY'].rank(method='first'),
            q=3, labels=['SMALL', 'MEDIUM', 'LARGE']
        )
    except ValueError:
        logger.warning("Size stratification: could not create 3 equal groups")
        treatment['SIZE_TERCILE'] = pd.qcut(
            treatment['SIZE_PROXY'].rank(method='first'),
            q=3, labels=['SMALL', 'MEDIUM', 'LARGE'],
            duplicates='drop'
        )

    for tercile in ['SMALL', 'MEDIUM', 'LARGE', 'ALL']:
        if tercile == 'ALL':
            sub = treatment
        else:
            sub = treatment[treatment['SIZE_TERCILE'] == tercile]

        if len(sub) < 10:
            continue

        threshold = sub['ABN_SELL'].quantile(1 - top_pct)
        in_tail = (sub['ABN_SELL'] >= threshold).astype(int)

        # Chi-square
        contingency = pd.crosstab(sub['ESTIMATED_POLITICAL_LEANING'], in_tail)
        if contingency.shape[0] >= 2 and contingency.shape[1] >= 2:
            chi2, chi2_p, dof, _ = stats.chi2_contingency(contingency)
            rows.append({
                'SIZE_TERCILE': tercile,
                'TEST': 'CHI2',
                'STAT': chi2,
                'P_VALUE': chi2_p,
                'N': len(sub),
            })

        # Per-leaning tail share
        for lean in sub['ESTIMATED_POLITICAL_LEANING'].dropna().unique():
            mask = sub['ESTIMATED_POLITICAL_LEANING'] == lean
            lean_tail = in_tail[mask]
            rows.append({
                'SIZE_TERCILE': tercile,
                'TEST': f'LEAN_{lean.upper().replace(" ", "_")}',
                'PCT_IN_TAIL': lean_tail.mean() * 100 if len(lean_tail) > 0 else np.nan,
                'MEAN_ABN_SELL': sub.loc[mask, 'ABN_SELL'].mean(),
                'N': int(mask.sum()),
            })

    df = pd.DataFrame(rows)
    if not df.empty:
        for tercile in ['SMALL', 'MEDIUM', 'LARGE']:
            chi2_r = df[(df['SIZE_TERCILE'] == tercile) & (df['TEST'] == 'CHI2')]
            if not chi2_r.empty:
                logger.info("  %s tercile: chi2=%.2f, p=%.4f, n=%d",
                            tercile, chi2_r.iloc[0]['STAT'],
                            chi2_r.iloc[0]['P_VALUE'], chi2_r.iloc[0]['N'])
    return df


def compute_insider_level_analysis(panel, events, form4, top_pct=0.20):
    """
    Analysis 33: Insider-level analysis — are specific people driving the tail?

    For each insider who sold before a CW event, check if they appear in
    multiple events. If 3-4 insiders at big Liberal firms generate most of
    the tail, that's a weaker finding than dispersed activity.
    """
    logger.info("Computing insider-level tail analysis...")

    pre_days = abs(WINDOWS['PRE_FULL'][1] - WINDOWS['PRE_FULL'][0]) + 1
    bench_days = abs(WINDOWS['BENCHMARK'][1] - WINDOWS['BENCHMARK'][0]) + 1

    treatment = panel[panel['IS_TREATMENT'] == True].copy()
    treatment['ABN_SELL'] = (
        treatment['PRE_FULL_NET_DOLLAR_SOLD'] / pre_days -
        treatment['BENCHMARK_NET_DOLLAR_SOLD'] / bench_days
    )
    threshold = treatment['ABN_SELL'].quantile(1 - top_pct)
    treatment['IN_TAIL'] = (treatment['ABN_SELL'] >= threshold).astype(int)

    # Merge leaning
    treatment = treatment.merge(
        events[['TICKER', 'EVENT_DATE', 'ESTIMATED_POLITICAL_LEANING']],
        on=['TICKER', 'EVENT_DATE'], how='left', suffixes=('', '_evt')
    )

    rows = []

    # For each tail event, find which insiders sold in the pre-event window
    tail_events = treatment[treatment['IN_TAIL'] == 1]
    insider_sells = []
    for _, ev in tail_events.iterrows():
        ticker = ev['TICKER']
        event_date = ev['EVENT_DATE']
        pre_start = event_date + pd.Timedelta(days=WINDOWS['PRE_FULL'][0])
        pre_end = event_date + pd.Timedelta(days=WINDOWS['PRE_FULL'][1])

        pre_txns = form4[
            (form4['ticker'] == ticker) &
            (form4['_trade_type'] == 'sell') &
            (form4['transaction_date'] >= pre_start) &
            (form4['transaction_date'] <= pre_end)
        ]
        for _, txn in pre_txns.iterrows():
            insider_sells.append({
                'TICKER': ticker,
                'EVENT_DATE': event_date,
                'LEAN': ev.get('ESTIMATED_POLITICAL_LEANING', 'Unknown'),
                'OWNER_NAME': txn['owner_name'],
                'DOLLAR_SOLD': txn.get('transaction_value', 0),
            })

    if not insider_sells:
        return pd.DataFrame([{'TEST': 'NO_TAIL_SELLS', 'N': 0}])

    sells_df = pd.DataFrame(insider_sells)

    # Concentration: how many unique insiders drive the tail?
    n_unique_insiders = sells_df['OWNER_NAME'].nunique()
    n_tail_events = len(tail_events)
    total_dollar = sells_df['DOLLAR_SOLD'].sum()

    # Top insiders by total selling
    insider_totals = sells_df.groupby('OWNER_NAME').agg(
        TOTAL_SOLD=('DOLLAR_SOLD', 'sum'),
        N_EVENTS=('EVENT_DATE', 'nunique'),
        TICKERS=('TICKER', lambda x: ','.join(sorted(x.unique()))),
        LEANS=('LEAN', lambda x: ','.join(sorted(x.unique()))),
    ).sort_values('TOTAL_SOLD', ascending=False)

    rows.append({
        'TEST': 'TAIL_CONCENTRATION',
        'N_TAIL_EVENTS': n_tail_events,
        'N_UNIQUE_INSIDERS': n_unique_insiders,
        'TOTAL_DOLLAR_SOLD': total_dollar,
        'TOP1_PCT': insider_totals.iloc[0]['TOTAL_SOLD'] / total_dollar * 100
            if total_dollar > 0 and len(insider_totals) > 0 else np.nan,
        'TOP5_PCT': insider_totals.head(5)['TOTAL_SOLD'].sum() / total_dollar * 100
            if total_dollar > 0 and len(insider_totals) >= 5 else np.nan,
    })

    # How many insiders appear in multiple tail events?
    multi_event = insider_totals[insider_totals['N_EVENTS'] > 1]
    rows.append({
        'TEST': 'MULTI_EVENT_INSIDERS',
        'N_MULTI': len(multi_event),
        'N_UNIQUE_INSIDERS': n_unique_insiders,
        'MULTI_PCT': len(multi_event) / n_unique_insiders * 100
            if n_unique_insiders > 0 else 0,
    })

    # By leaning: number of unique insiders in tail
    for lean, grp in sells_df.groupby('LEAN'):
        rows.append({
            'TEST': f'INSIDER_LEAN_{lean.upper().replace(" ", "_")}',
            'N_UNIQUE_INSIDERS': grp['OWNER_NAME'].nunique(),
            'TOTAL_DOLLAR_SOLD': grp['DOLLAR_SOLD'].sum(),
            'N_TAIL_EVENTS': grp['EVENT_DATE'].nunique(),
        })

    # Top 10 insiders detail
    for i, (owner, row_data) in enumerate(insider_totals.head(10).iterrows()):
        rows.append({
            'TEST': f'TOP_INSIDER_{i + 1}',
            'OWNER_NAME': owner,
            'TOTAL_DOLLAR_SOLD': row_data['TOTAL_SOLD'],
            'N_TAIL_EVENTS': row_data['N_EVENTS'],
            'TICKERS': row_data['TICKERS'],
            'LEANS': row_data['LEANS'],
        })

    return pd.DataFrame(rows)


def compute_time_series_tail(panel, events, top_pct=0.20):
    """
    Analysis 34: Time-series decomposition of tail membership by leaning.

    Are CW events concentrated in 2019-2024 (DEI era)? Plot tail membership
    over time by leaning. If Liberal-tail is a 2020-2023 phenomenon, it's a
    period effect. If stable across 2010-2024, it's durable.
    """
    logger.info("Computing time-series tail decomposition...")

    pre_days = abs(WINDOWS['PRE_FULL'][1] - WINDOWS['PRE_FULL'][0]) + 1
    bench_days = abs(WINDOWS['BENCHMARK'][1] - WINDOWS['BENCHMARK'][0]) + 1

    treatment = panel[panel['IS_TREATMENT'] == True].copy()
    treatment['ABN_SELL'] = (
        treatment['PRE_FULL_NET_DOLLAR_SOLD'] / pre_days -
        treatment['BENCHMARK_NET_DOLLAR_SOLD'] / bench_days
    )
    threshold = treatment['ABN_SELL'].quantile(1 - top_pct)
    treatment['IN_TAIL'] = (treatment['ABN_SELL'] >= threshold).astype(int)

    # Merge leaning
    treatment = treatment.merge(
        events[['TICKER', 'EVENT_DATE', 'ESTIMATED_POLITICAL_LEANING']],
        on=['TICKER', 'EVENT_DATE'], how='left', suffixes=('', '_evt')
    )

    treatment['YEAR'] = treatment['EVENT_DATE'].dt.year

    rows = []

    # Overall year distribution
    for year, grp in treatment.groupby('YEAR'):
        rows.append({
            'LEAN': 'ALL',
            'YEAR': int(year),
            'N_EVENTS': len(grp),
            'N_TAIL': int(grp['IN_TAIL'].sum()),
            'PCT_IN_TAIL': grp['IN_TAIL'].mean() * 100,
            'MEAN_ABN_SELL': grp['ABN_SELL'].mean(),
        })

    # By leaning × year
    for lean in treatment['ESTIMATED_POLITICAL_LEANING'].dropna().unique():
        lean_sub = treatment[treatment['ESTIMATED_POLITICAL_LEANING'] == lean]
        for year, grp in lean_sub.groupby('YEAR'):
            rows.append({
                'LEAN': lean,
                'YEAR': int(year),
                'N_EVENTS': len(grp),
                'N_TAIL': int(grp['IN_TAIL'].sum()),
                'PCT_IN_TAIL': grp['IN_TAIL'].mean() * 100,
                'MEAN_ABN_SELL': grp['ABN_SELL'].mean(),
            })

    # Period comparison: pre-DEI (<2019) vs DEI era (2019-2024)
    for lean in treatment['ESTIMATED_POLITICAL_LEANING'].dropna().unique():
        lean_sub = treatment[treatment['ESTIMATED_POLITICAL_LEANING'] == lean]
        pre_dei = lean_sub[lean_sub['YEAR'] < 2019]
        dei_era = lean_sub[(lean_sub['YEAR'] >= 2019) & (lean_sub['YEAR'] <= 2024)]

        if len(pre_dei) >= 3 and len(dei_era) >= 3:
            rows.append({
                'LEAN': lean,
                'YEAR': -1,  # sentinel for period comparison
                'N_EVENTS': len(pre_dei) + len(dei_era),
                'N_TAIL': -1,
                'PCT_IN_TAIL': np.nan,
                'MEAN_ABN_SELL': np.nan,
                'PRE_DEI_PCT_TAIL': pre_dei['IN_TAIL'].mean() * 100,
                'DEI_ERA_PCT_TAIL': dei_era['IN_TAIL'].mean() * 100,
                'PRE_DEI_N': len(pre_dei),
                'DEI_ERA_N': len(dei_era),
            })

    df = pd.DataFrame(rows)
    if not df.empty:
        period_rows = df[df['YEAR'] == -1]
        for _, r in period_rows.iterrows():
            logger.info("  %s: pre-DEI tail=%.1f%% (n=%s), DEI-era tail=%.1f%% (n=%s)",
                        r['LEAN'],
                        r.get('PRE_DEI_PCT_TAIL', 0), r.get('PRE_DEI_N', 0),
                        r.get('DEI_ERA_PCT_TAIL', 0), r.get('DEI_ERA_N', 0))
    return df


def compute_placebo_stratified_by_lean(panel, events, n_iterations=500, seed=42):
    """
    Analysis 35: Placebo test stratified by political leaning.

    Run the permutation test separately within Conservative, Liberal, and
    Mixed subsamples. If Liberal firms show tail-like behavior on random
    dates too, they just have higher baseline volatility.
    """
    logger.info("Computing lean-stratified placebo test...")

    pre_col = 'PRE_FULL_NET_DOLLAR_SOLD'
    bench_col = 'BENCHMARK_NET_DOLLAR_SOLD'
    if pre_col not in panel.columns or bench_col not in panel.columns:
        return pd.DataFrame()

    pre_days = abs(WINDOWS['PRE_FULL'][1] - WINDOWS['PRE_FULL'][0]) + 1
    bench_days = abs(WINDOWS['BENCHMARK'][1] - WINDOWS['BENCHMARK'][0]) + 1

    # Merge leaning
    merged = panel.merge(
        events[['TICKER', 'EVENT_DATE', 'ESTIMATED_POLITICAL_LEANING']],
        on=['TICKER', 'EVENT_DATE'], how='left', suffixes=('', '_evt')
    )

    rows = []
    rng = np.random.RandomState(seed)

    for lean in ['Conservative', 'Liberal', 'Mixed', 'ALL']:
        if lean == 'ALL':
            sub = merged
        else:
            sub = merged[merged['ESTIMATED_POLITICAL_LEANING'] == lean]

        if len(sub) < 10:
            continue

        pre_daily = sub[pre_col] / pre_days
        bench_daily = sub[bench_col] / bench_days
        diff = (pre_daily - bench_daily).values
        treatment_mask = sub['IS_TREATMENT'].values

        if treatment_mask.sum() < 3 or (~treatment_mask).sum() < 3:
            continue

        observed = diff[treatment_mask].mean() - diff[~treatment_mask].mean()

        placebo_stats = []
        for _ in range(n_iterations):
            shuffled = rng.permutation(treatment_mask)
            p_stat = diff[shuffled].mean() - diff[~shuffled].mean()
            placebo_stats.append(p_stat)

        placebo_stats = np.array(placebo_stats)
        empirical_p = (np.abs(placebo_stats) >= np.abs(observed)).mean()

        rows.append({
            'LEAN': lean,
            'OBSERVED_STAT': observed,
            'PLACEBO_MEAN': placebo_stats.mean(),
            'PLACEBO_STD': placebo_stats.std(),
            'EMPIRICAL_P_TWO_SIDED': empirical_p,
            'N_TREATMENT': int(treatment_mask.sum()),
            'N_CONTROL': int((~treatment_mask).sum()),
            'N_ITERATIONS': n_iterations,
        })

    df = pd.DataFrame(rows)
    if not df.empty:
        for _, r in df.iterrows():
            logger.info("  %s: observed=%.0f, placebo_p=%.4f",
                        r['LEAN'], r['OBSERVED_STAT'], r['EMPIRICAL_P_TWO_SIDED'])
    return df


def compute_within_firm_temporal(panel, events, form4, top_pct=0.20):
    """
    Analysis 36: Within-firm temporal clustering.

    For each firm with multiple events, ask whether tail-selling episodes
    cluster around CW events specifically or occur uniformly across the
    firm's history.
    """
    logger.info("Computing within-firm temporal clustering...")

    pre_days = abs(WINDOWS['PRE_FULL'][1] - WINDOWS['PRE_FULL'][0]) + 1
    bench_days = abs(WINDOWS['BENCHMARK'][1] - WINDOWS['BENCHMARK'][0]) + 1

    treatment = panel[panel['IS_TREATMENT'] == True].copy()
    treatment['ABN_SELL'] = (
        treatment['PRE_FULL_NET_DOLLAR_SOLD'] / pre_days -
        treatment['BENCHMARK_NET_DOLLAR_SOLD'] / bench_days
    )
    threshold = treatment['ABN_SELL'].quantile(1 - top_pct)
    treatment['IN_TAIL'] = (treatment['ABN_SELL'] >= threshold).astype(int)

    # Merge leaning
    treatment = treatment.merge(
        events[['TICKER', 'EVENT_DATE', 'ESTIMATED_POLITICAL_LEANING']],
        on=['TICKER', 'EVENT_DATE'], how='left', suffixes=('', '_evt')
    )

    # Firms with multiple events
    firm_event_counts = treatment.groupby('TICKER').size()
    multi_event_firms = firm_event_counts[firm_event_counts >= 2].index

    rows = []

    for ticker in multi_event_firms:
        firm_events = treatment[treatment['TICKER'] == ticker]
        n_events = len(firm_events)
        n_tail = int(firm_events['IN_TAIL'].sum())
        lean = firm_events['ESTIMATED_POLITICAL_LEANING'].mode()
        lean = lean.iloc[0] if len(lean) > 0 else 'Unknown'

        # Check if tail episodes cluster on specific events
        rows.append({
            'TICKER': ticker,
            'LEAN': lean,
            'N_EVENTS': n_events,
            'N_TAIL': n_tail,
            'TAIL_RATE': n_tail / n_events if n_events > 0 else 0,
            'MEAN_ABN_SELL': firm_events['ABN_SELL'].mean(),
            'STD_ABN_SELL': firm_events['ABN_SELL'].std() if n_events > 1 else np.nan,
        })

    df = pd.DataFrame(rows)
    if not df.empty:
        # Summary by leaning
        for lean in df['LEAN'].unique():
            lean_sub = df[df['LEAN'] == lean]
            logger.info("  %s: %d multi-event firms, mean tail rate=%.1f%%",
                        lean, len(lean_sub), lean_sub['TAIL_RATE'].mean() * 100)

        # Fisher exact test: do Liberal multi-event firms have higher tail rates?
        if len(df) >= 10:
            df['HIGH_TAIL'] = (df['TAIL_RATE'] > 0).astype(int)
            liberal_high = df[df['LEAN'] == 'Liberal']['HIGH_TAIL'].sum()
            liberal_n = len(df[df['LEAN'] == 'Liberal'])
            other_high = df[df['LEAN'] != 'Liberal']['HIGH_TAIL'].sum()
            other_n = len(df[df['LEAN'] != 'Liberal'])
            if liberal_n > 0 and other_n > 0:
                table = np.array([[liberal_high, liberal_n - liberal_high],
                                  [other_high, other_n - other_high]])
                if table.min() >= 0:
                    _, fisher_p = stats.fisher_exact(table)
                    summary_row = pd.DataFrame([{
                        'TICKER': '_FISHER_TEST',
                        'LEAN': 'SUMMARY',
                        'N_EVENTS': len(df),
                        'N_TAIL': -1,
                        'TAIL_RATE': fisher_p,
                        'MEAN_ABN_SELL': liberal_high / liberal_n if liberal_n > 0 else 0,
                        'STD_ABN_SELL': other_high / other_n if other_n > 0 else 0,
                    }])
                    df = pd.concat([df, summary_row], ignore_index=True)

    return df


def compute_disclosure_channel(panel, events, form4, top_pct=0.20):
    """
    Analysis 37: 10b5-1 disclosure channel by political leaning.

    Did Liberal firms in the tail file 10b5-1 plans more or less than
    Conservative firms? If Liberal-tail trades were scheduled, the
    foreknowledge story is dead.
    """
    logger.info("Computing disclosure channel analysis (10b5-1 × leaning)...")

    pre_days = abs(WINDOWS['PRE_FULL'][1] - WINDOWS['PRE_FULL'][0]) + 1
    bench_days = abs(WINDOWS['BENCHMARK'][1] - WINDOWS['BENCHMARK'][0]) + 1

    treatment = panel[panel['IS_TREATMENT'] == True].copy()
    treatment['ABN_SELL'] = (
        treatment['PRE_FULL_NET_DOLLAR_SOLD'] / pre_days -
        treatment['BENCHMARK_NET_DOLLAR_SOLD'] / bench_days
    )
    threshold = treatment['ABN_SELL'].quantile(1 - top_pct)
    treatment['IN_TAIL'] = (treatment['ABN_SELL'] >= threshold).astype(int)

    # Merge leaning
    treatment = treatment.merge(
        events[['TICKER', 'EVENT_DATE', 'ESTIMATED_POLITICAL_LEANING']],
        on=['TICKER', 'EVENT_DATE'], how='left', suffixes=('', '_evt')
    )

    # Identify likely 10b5-1 tickers (reuse logic from compute_10b5_1_filter)
    plan_tickers = set()
    for ticker in treatment['TICKER'].unique():
        ticker_txns = form4[form4['ticker'] == ticker]
        for owner in ticker_txns['owner_name'].unique():
            trades = ticker_txns[ticker_txns['owner_name'] == owner]['transaction_date'].dropna()
            if len(trades) < 8:
                continue
            quarters = trades.dt.to_period('Q').nunique()
            if quarters < 4:
                continue
            q_start_month = ((trades.dt.month - 1) // 3) * 3 + 1
            day_of_quarter = (trades.dt.month - q_start_month) * 30 + trades.dt.day
            week_in_q = day_of_quarter // 7
            if week_in_q.std() < 3:
                plan_tickers.add(ticker)
                break

    treatment['LIKELY_10B5_1'] = treatment['TICKER'].isin(plan_tickers).astype(int)

    rows = []

    # By leaning: 10b5-1 rate in tail vs non-tail
    for lean in treatment['ESTIMATED_POLITICAL_LEANING'].dropna().unique():
        lean_sub = treatment[treatment['ESTIMATED_POLITICAL_LEANING'] == lean]
        tail_sub = lean_sub[lean_sub['IN_TAIL'] == 1]
        nontail_sub = lean_sub[lean_sub['IN_TAIL'] == 0]

        rows.append({
            'LEAN': lean,
            'GROUP': 'TAIL',
            'N': len(tail_sub),
            'N_10B5_1': int(tail_sub['LIKELY_10B5_1'].sum()),
            'PCT_10B5_1': tail_sub['LIKELY_10B5_1'].mean() * 100 if len(tail_sub) > 0 else np.nan,
            'MEAN_ABN_SELL': tail_sub['ABN_SELL'].mean() if len(tail_sub) > 0 else np.nan,
        })
        rows.append({
            'LEAN': lean,
            'GROUP': 'NON_TAIL',
            'N': len(nontail_sub),
            'N_10B5_1': int(nontail_sub['LIKELY_10B5_1'].sum()),
            'PCT_10B5_1': nontail_sub['LIKELY_10B5_1'].mean() * 100 if len(nontail_sub) > 0 else np.nan,
            'MEAN_ABN_SELL': nontail_sub['ABN_SELL'].mean() if len(nontail_sub) > 0 else np.nan,
        })

    # Fisher exact: tail × 10b5-1 across all firms
    tail_all = treatment[treatment['IN_TAIL'] == 1]
    nontail_all = treatment[treatment['IN_TAIL'] == 0]
    a = int(tail_all['LIKELY_10B5_1'].sum())
    b = len(tail_all) - a
    c = int(nontail_all['LIKELY_10B5_1'].sum())
    d = len(nontail_all) - c
    if min(a + b, c + d) > 0:
        _, fisher_p = stats.fisher_exact(np.array([[a, b], [c, d]]))
        rows.append({
            'LEAN': 'ALL',
            'GROUP': 'FISHER_TEST',
            'N': len(treatment),
            'N_10B5_1': a + c,
            'PCT_10B5_1': fisher_p,  # stores p-value
            'MEAN_ABN_SELL': np.nan,
        })

    df = pd.DataFrame(rows)
    if not df.empty:
        tail_rows = df[df['GROUP'] == 'TAIL']
        for _, r in tail_rows.iterrows():
            logger.info("  %s tail: %d events, %.0f%% likely 10b5-1",
                        r['LEAN'], r['N'], r['PCT_10B5_1'] if pd.notna(r['PCT_10B5_1']) else 0)
    return df


def compute_buy_analysis(panel, form4, events, top_pct=0.20):
    """
    Analysis 38: Buy-side analysis — do insiders reduce buying before CW events?

    An informed insider with negative information sells (already tested).
    They also reduce buying. Check whether buy reduction correlates with
    sell increase by leaning.
    """
    logger.info("Computing buy-side analysis...")

    pre_days = abs(WINDOWS['PRE_FULL'][1] - WINDOWS['PRE_FULL'][0]) + 1
    bench_days = abs(WINDOWS['BENCHMARK'][1] - WINDOWS['BENCHMARK'][0]) + 1

    treatment = panel[panel['IS_TREATMENT'] == True].copy()

    # Abnormal selling (existing)
    treatment['ABN_SELL'] = (
        treatment['PRE_FULL_NET_DOLLAR_SOLD'] / pre_days -
        treatment['BENCHMARK_NET_DOLLAR_SOLD'] / bench_days
    )

    # Abnormal buying (mirror of sell logic using dollar_bought columns)
    buy_pre_col = 'PRE_FULL_DOLLAR_BOUGHT'
    buy_bench_col = 'BENCHMARK_DOLLAR_BOUGHT'

    if buy_pre_col in treatment.columns and buy_bench_col in treatment.columns:
        treatment['ABN_BUY'] = (
            treatment[buy_pre_col] / pre_days -
            treatment[buy_bench_col] / bench_days
        )
    else:
        # If separate buy columns don't exist, compute from net and gross
        # Gross sold = (NET + GROSS) / 2 ... but we may not have GROSS
        # Fall back: compute buys directly from form4
        logger.info("  Computing buy volumes directly from Form 4...")
        buy_rows = []
        for _, ev in treatment.iterrows():
            ticker = ev['TICKER']
            event_date = ev['EVENT_DATE']
            bench_start = event_date + pd.Timedelta(days=WINDOWS['BENCHMARK'][0])
            bench_end = event_date + pd.Timedelta(days=WINDOWS['BENCHMARK'][1])
            pre_start = event_date + pd.Timedelta(days=WINDOWS['PRE_FULL'][0])
            pre_end = event_date + pd.Timedelta(days=WINDOWS['PRE_FULL'][1])

            ticker_buys = form4[
                (form4['ticker'] == ticker) &
                (form4['_trade_type'] == 'buy')
            ]

            bench_buys = ticker_buys[
                (ticker_buys['transaction_date'] >= bench_start) &
                (ticker_buys['transaction_date'] <= bench_end)
            ]['transaction_value'].sum()

            pre_buys = ticker_buys[
                (ticker_buys['transaction_date'] >= pre_start) &
                (ticker_buys['transaction_date'] <= pre_end)
            ]['transaction_value'].sum()

            buy_rows.append({
                'TICKER': ticker,
                'EVENT_DATE': event_date,
                'PRE_BUY_DAILY': pre_buys / pre_days,
                'BENCH_BUY_DAILY': bench_buys / bench_days,
            })

        buy_df = pd.DataFrame(buy_rows)
        treatment = treatment.merge(buy_df, on=['TICKER', 'EVENT_DATE'], how='left')
        treatment['ABN_BUY'] = treatment['PRE_BUY_DAILY'] - treatment['BENCH_BUY_DAILY']

    # Merge leaning
    treatment = treatment.merge(
        events[['TICKER', 'EVENT_DATE', 'ESTIMATED_POLITICAL_LEANING']],
        on=['TICKER', 'EVENT_DATE'], how='left', suffixes=('', '_evt')
    )

    rows = []

    # Aggregate: do buys decrease before CW events?
    abn_buy = treatment['ABN_BUY'].dropna()
    if len(abn_buy) > 1:
        t_stat, t_p = stats.ttest_1samp(abn_buy, 0)
        d = abn_buy.mean() / abn_buy.std() if abn_buy.std() > 0 else 0
        rows.append({
            'TEST': 'ABN_BUY_VS_ZERO',
            'LEAN': 'ALL',
            'N': len(abn_buy),
            'MEAN_ABN_BUY': abn_buy.mean(),
            'COHEN_D': d,
            'T_STAT': t_stat,
            'P_VALUE': t_p,
        })

    # By leaning
    for lean in treatment['ESTIMATED_POLITICAL_LEANING'].dropna().unique():
        lean_sub = treatment[treatment['ESTIMATED_POLITICAL_LEANING'] == lean]
        abn = lean_sub['ABN_BUY'].dropna()
        if len(abn) > 1:
            t_s, t_p = stats.ttest_1samp(abn, 0)
            d = abn.mean() / abn.std() if abn.std() > 0 else 0
            rows.append({
                'TEST': 'ABN_BUY_VS_ZERO',
                'LEAN': lean,
                'N': len(abn),
                'MEAN_ABN_BUY': abn.mean(),
                'COHEN_D': d,
                'T_STAT': t_s,
                'P_VALUE': t_p,
            })

    # Correlation between sell increase and buy decrease
    valid = treatment[['ABN_SELL', 'ABN_BUY']].dropna()
    if len(valid) >= 10:
        corr, corr_p = stats.pearsonr(valid['ABN_SELL'], valid['ABN_BUY'])
        rows.append({
            'TEST': 'SELL_BUY_CORRELATION',
            'LEAN': 'ALL',
            'N': len(valid),
            'MEAN_ABN_BUY': corr,
            'T_STAT': corr_p,
            'P_VALUE': corr_p,
            'COHEN_D': np.nan,
        })

    # By leaning: sell-buy asymmetry
    for lean in treatment['ESTIMATED_POLITICAL_LEANING'].dropna().unique():
        lean_sub = treatment[treatment['ESTIMATED_POLITICAL_LEANING'] == lean].dropna(
            subset=['ABN_SELL', 'ABN_BUY'])
        if len(lean_sub) >= 5:
            rows.append({
                'TEST': 'SELL_BUY_ASYMMETRY',
                'LEAN': lean,
                'N': len(lean_sub),
                'MEAN_ABN_BUY': lean_sub['ABN_BUY'].mean(),
                'COHEN_D': lean_sub['ABN_SELL'].mean(),  # store sell for comparison
                'T_STAT': np.nan,
                'P_VALUE': np.nan,
            })

    df = pd.DataFrame(rows)
    if not df.empty:
        agg = df[(df['TEST'] == 'ABN_BUY_VS_ZERO') & (df['LEAN'] == 'ALL')]
        if not agg.empty:
            logger.info("  Aggregate abnormal buy: mean=$%.0f/day, d=%.3f, p=%.4f",
                        agg.iloc[0]['MEAN_ABN_BUY'], agg.iloc[0]['COHEN_D'],
                        agg.iloc[0]['P_VALUE'])
    return df


# ═══════════════════════════════════════════════════════════════════════
# SAVE RESULTS
# ═══════════════════════════════════════════════════════════════════════

def save_essay3_results(store, results_dict):
    """Save all Essay 3 tables to the database."""
    timestamp = pd.Timestamp.now().isoformat() + 'Z'
    tables = {
        'ESSAY3_INSIDER_PANEL': results_dict.get('panel'),
        'ESSAY3_WINDOW_SUMMARY': results_dict.get('window_summary'),
        'ESSAY3_ABNORMAL_SELLING': results_dict.get('abnormal'),
        'ESSAY3_TREATMENT_VS_CONTROL': results_dict.get('treatment_ctrl'),
        'ESSAY3_LEANING_ANALYSIS': results_dict.get('leaning'),
        'ESSAY3_REGIME_INTERACTION': results_dict.get('regime'),
        'ESSAY3_ROUTINE_VS_OPPORTUNISTIC': results_dict.get('rvo'),
        'ESSAY3_PLACEBO_TEST': results_dict.get('placebo'),
        'ESSAY3_ACCELERATION_TEST': results_dict.get('accel'),
        'ESSAY3_CAR_INSIDER_REGRESSION': results_dict.get('car_reg'),
        'ESSAY3_INFORMATION_GRADIENT': results_dict.get('gradient'),
        'ESSAY3_DIFF_IN_DIFF': results_dict.get('did'),
        'ESSAY3_10B5_1_FILTER': results_dict.get('plan_filter'),
        'ESSAY3_MATCH_QUALITY': results_dict.get('match_quality'),
        'ESSAY3_EVENT_CLUSTERING': results_dict.get('event_clustering'),
        'ESSAY3_INTENSIVE_EXTENSIVE': results_dict.get('margins'),
        'ESSAY3_ABNORMAL_VOLUME_RATIO': results_dict.get('avr'),
        'ESSAY3_FIRM_FIXED_EFFECTS': results_dict.get('firm_fe'),
        'ESSAY3_CROSS_SECTIONAL': results_dict.get('cross_section'),
        'ESSAY3_POST_EVENT_REVERSAL': results_dict.get('reversal'),
        'ESSAY3_BOOTSTRAP_CI': results_dict.get('bootstrap'),
        'ESSAY3_FAMA_MACBETH': results_dict.get('fama_macbeth'),
        'ESSAY3_SHORT_SWING': results_dict.get('short_swing'),
        'ESSAY3_TOST_EQUIVALENCE': results_dict.get('tost'),
        'ESSAY3_SUBGROUP_ANALYSIS': results_dict.get('subgroup'),
        'ESSAY3_VOL_SHIFT': results_dict.get('vol_shift'),
        'ESSAY3_TAIL_FIRMS': results_dict.get('tail_firms'),
        'ESSAY3_TAIL_LEANING': results_dict.get('tail_leaning'),
        'ESSAY3_TAIL_EVENT_TYPE': results_dict.get('tail_event_type'),
        'ESSAY3_TAIL_INTERACTION': results_dict.get('tail_interaction'),
        'ESSAY3_CONS_PLANNED': results_dict.get('cons_planned'),
        # Robustness extensions (29-38)
        'ESSAY3_TAIL_LOGIT': results_dict.get('tail_logit'),
        'ESSAY3_PROPENSITY_MATCH': results_dict.get('propensity_match'),
        'ESSAY3_WINSORIZED_TAIL': results_dict.get('winsorized_tail'),
        'ESSAY3_SIZE_STRATIFIED_TAIL': results_dict.get('size_stratified'),
        'ESSAY3_INSIDER_LEVEL': results_dict.get('insider_level'),
        'ESSAY3_TIME_SERIES_TAIL': results_dict.get('time_series_tail'),
        'ESSAY3_PLACEBO_STRATIFIED': results_dict.get('placebo_stratified'),
        'ESSAY3_WITHIN_FIRM_TEMPORAL': results_dict.get('within_firm_temporal'),
        'ESSAY3_DISCLOSURE_CHANNEL': results_dict.get('disclosure_channel'),
        'ESSAY3_BUY_ANALYSIS': results_dict.get('buy_analysis'),
    }

    for table_name, df in tables.items():
        if table_name == 'ESSAY3_INSIDER_PANEL' and df is not None:
            mb = df.memory_usage(deep=True).sum() / (1024 * 1024)
            logger.info("  Panel size: %d rows x %d cols (%.1f MB in memory)",
                        len(df), len(df.columns), mb)
        if df is not None and not df.empty:
            df = df.copy()
            df['RUN_TIMESTAMP'] = timestamp
            store.write_table(df, table_name, replace=True)
            logger.info("  Saved %s: %d rows", table_name, len(df))
        else:
            logger.warning("  Skipped %s (empty)", table_name)


# ═══════════════════════════════════════════════════════════════════════
# MAIN ENTRY POINT
# ═══════════════════════════════════════════════════════════════════════

def run_essay3(store=None):
    """Run the complete Essay 3 insider trading analysis."""
    logger.info("=" * 60)
    logger.info("  Essay 3: Insider Trading Around Culture War Events")
    logger.info("=" * 60)

    # Connect to data
    if store is None:
        store = DataStore()

    # Load Form 4 data
    form4 = _load_form4()
    if form4.empty:
        logger.error("No Form 4 data available. Run the Form 4 downloader first.")
        return

    # Load events
    events = store.events
    if events.empty:
        logger.error("No events data available.")
        return

    # Load CAR results from Event Study
    car_df = store.read_table('EVENT_STUDY_RESULTS')

    # Load VIX for regime analysis
    vix_df = store.vix

    # Load stock data for volume-based analyses
    stock_data = store.stocks if hasattr(store, 'stocks') and store.stocks is not None else None

    # Tag form4 with trade type for downstream analyses (short-swing, 10b5-1)
    form4['_trade_type'] = form4.apply(
        lambda r: _classify_trade(r['transaction_code'], r['acquired_disposed']),
        axis=1
    )

    # 1. Build panel
    panel = build_insider_panel(form4, events, car_df, vix_df)

    # 2. Window summary
    logger.info("Computing window summary...")
    window_summary = compute_window_summary(panel)

    # 3. Abnormal selling
    logger.info("Computing abnormal selling tests...")
    abnormal = compute_abnormal_selling(panel)

    # 4. Treatment vs control
    logger.info("Computing treatment vs control regression...")
    treatment_ctrl = compute_treatment_vs_control(panel)

    # 5. Leaning analysis
    logger.info("Computing leaning analysis...")
    leaning = compute_leaning_analysis(panel)

    # 6. Regime interaction
    logger.info("Computing regime interaction...")
    regime = compute_regime_interaction(panel, vix_df)

    # 7. Routine vs opportunistic
    logger.info("Computing routine vs opportunistic...")
    rvo = compute_routine_vs_opportunistic(panel)

    # 8. Placebo test
    logger.info("Computing placebo permutation test...")
    placebo = compute_placebo_test(panel)

    # 9. Acceleration test
    logger.info("Computing acceleration tests...")
    accel = compute_acceleration_test(panel)

    # 10. CAR-insider regression
    logger.info("Computing CAR-insider regression...")
    car_reg = compute_car_insider_regression(panel)

    # 11. Information gradient
    logger.info("Computing information gradient...")
    gradient = compute_information_gradient(panel)

    # 12. Difference-in-differences
    logger.info("Computing difference-in-differences...")
    did = compute_diff_in_diff(panel)

    # 13. 10b5-1 plan filter
    logger.info("Computing 10b5-1 plan filter...")
    plan_filter = compute_10b5_1_filter(panel, form4)

    # 14. Match quality validation
    logger.info("Computing match quality validation...")
    match_quality = compute_match_quality(panel, events, stock_data)

    # 15. Event clustering adjustment
    logger.info("Computing event clustering adjustment...")
    event_clustering = compute_event_clustering(panel)

    # 16. Intensive vs extensive margin
    logger.info("Computing intensive vs extensive margin...")
    margins = compute_intensive_extensive_margin(panel)

    # 17. Abnormal volume ratio
    logger.info("Computing abnormal volume ratio...")
    avr = compute_abnormal_volume_ratio(panel, stock_data)

    # 18. Firm fixed effects
    logger.info("Computing firm fixed effects...")
    firm_fe = compute_firm_fixed_effects(panel)

    # 19. Cross-sectional determinants
    logger.info("Computing cross-sectional determinants...")
    cross_section = compute_cross_sectional_determinants(panel, events)

    # 20. Post-event reversal test
    logger.info("Computing post-event reversal test...")
    reversal = compute_post_event_reversal(panel)

    # 21. Bootstrap confidence intervals
    logger.info("Computing bootstrap confidence intervals...")
    bootstrap = compute_bootstrap_ci(panel)

    # 22. Fama-MacBeth regressions
    logger.info("Computing Fama-MacBeth regressions...")
    fama_macbeth = compute_fama_macbeth(panel)

    # 23. Short-swing profit rule check
    logger.info("Computing short-swing profit rule check...")
    short_swing = compute_short_swing_check(panel, form4)

    # 24. TOST equivalence tests + power analysis
    logger.info("Computing TOST equivalence tests + power analysis...")
    tost = compute_tost_equivalence(panel)

    # 25. Subgroup signal-finding
    logger.info("Computing subgroup signal-finding analysis...")
    subgroup = compute_subgroup_analysis(panel, form4)

    # 26. Volatility-shift identification strategy
    logger.info("Computing volatility-shift identification strategy...")
    vol_shift = compute_volatility_shift_analysis(panel, form4, stock_data)

    # 27. Tail diagnostic — who drives the Mann-Whitney distributional diff?
    logger.info("Computing tail diagnostic (leaning × event type)...")
    tail_diag = compute_tail_diagnostic(panel, events, form4)

    # 28. Conservative × Planned deep dive (case-study table + power diagnostic)
    logger.info("Computing Conservative × Planned deep dive...")
    cons_planned = compute_conservative_planned_deep_dive(panel, events, form4)

    # 29. Tail logit with industry controls
    logger.info("Computing tail logit with industry controls...")
    tail_logit = compute_tail_logit(panel, events, stock_data)

    # 30. Propensity score matching
    logger.info("Computing propensity score matching...")
    propensity_match = compute_propensity_score_matching(panel, events, form4, stock_data)

    # 31. Winsorized tail analysis
    logger.info("Computing winsorized tail analysis...")
    winsorized_tail = compute_winsorized_tail(panel, events)

    # 32. Size-stratified tail analysis
    logger.info("Computing size-stratified tail analysis...")
    size_stratified = compute_size_stratified_tail(panel, events, stock_data)

    # 33. Insider-level analysis
    logger.info("Computing insider-level tail analysis...")
    insider_level = compute_insider_level_analysis(panel, events, form4)

    # 34. Time-series tail decomposition
    logger.info("Computing time-series tail decomposition...")
    time_series_tail = compute_time_series_tail(panel, events)

    # 35. Placebo test stratified by leaning
    logger.info("Computing lean-stratified placebo test...")
    placebo_stratified = compute_placebo_stratified_by_lean(panel, events)

    # 36. Within-firm temporal clustering
    logger.info("Computing within-firm temporal clustering...")
    within_firm_temporal = compute_within_firm_temporal(panel, events, form4)

    # 37. Disclosure channel (10b5-1 × leaning)
    logger.info("Computing disclosure channel analysis...")
    disclosure_channel = compute_disclosure_channel(panel, events, form4)

    # 38. Buy-side analysis
    logger.info("Computing buy-side analysis...")
    buy_analysis = compute_buy_analysis(panel, form4, events)

    # Collect all results
    results = {
        'panel': panel,
        'window_summary': window_summary,
        'abnormal': abnormal,
        'treatment_ctrl': treatment_ctrl,
        'leaning': leaning,
        'regime': regime,
        'rvo': rvo,
        'placebo': placebo,
        'accel': accel,
        'car_reg': car_reg,
        'gradient': gradient,
        'did': did,
        'plan_filter': plan_filter,
        'match_quality': match_quality,
        'event_clustering': event_clustering,
        'margins': margins,
        'avr': avr,
        'firm_fe': firm_fe,
        'cross_section': cross_section,
        'reversal': reversal,
        'bootstrap': bootstrap,
        'fama_macbeth': fama_macbeth,
        'short_swing': short_swing,
        'tost': tost,
        'subgroup': subgroup,
        'vol_shift': vol_shift,
        'tail_firms': tail_diag.get('tail_firms'),
        'tail_leaning': tail_diag.get('leaning_test'),
        'tail_event_type': tail_diag.get('event_type'),
        'tail_interaction': tail_diag.get('interaction'),
        'cons_planned': cons_planned,
        # Robustness extensions (29-38)
        'tail_logit': tail_logit,
        'propensity_match': propensity_match,
        'winsorized_tail': winsorized_tail,
        'size_stratified': size_stratified,
        'insider_level': insider_level,
        'time_series_tail': time_series_tail,
        'placebo_stratified': placebo_stratified,
        'within_firm_temporal': within_firm_temporal,
        'disclosure_channel': disclosure_channel,
        'buy_analysis': buy_analysis,
    }

    # Save all results
    logger.info("Saving results...")
    save_essay3_results(store, results)

    # Summary
    logger.info("=" * 60)
    logger.info("  Essay 3 complete!")
    logger.info("  Panel: %d events (%d with data)",
                len(panel), panel['HAS_SUFFICIENT_DATA'].sum())
    logger.info("  Treatment: %d, Control: %d",
                panel['IS_TREATMENT'].sum(), (~panel['IS_TREATMENT']).sum())
    if not abnormal.empty:
        sig = abnormal[abnormal['T_BH_SIGNIFICANT'] == 1]
        logger.info("  Abnormal selling: %d/%d windows significant (BH)",
                    len(sig), len(abnormal))
    if not did.empty:
        did_treat = did[(did['SPECIFICATION'] == 'DID_SIMPLE') &
                        (did['VARIABLE'] == 'TREAT_X_POST')]
        if not did_treat.empty:
            logger.info("  DiD interaction: coef=%.4f, p=%.4f",
                        did_treat.iloc[0]['COEFFICIENT'],
                        did_treat.iloc[0]['P_VALUE'])
    if not match_quality.empty:
        n_bal = match_quality['BALANCED'].sum()
        logger.info("  Match quality: %d/%d covariates balanced",
                    n_bal, len(match_quality))
    if not tost.empty:
        tost_rows = tost[tost['TEST'] == 'TOST']
        n_equiv = tost_rows['EQUIVALENT'].sum() if not tost_rows.empty else 0
        logger.info("  TOST equivalence: %d/%d margins show equivalence",
                    int(n_equiv), len(tost_rows))
        power_obs = tost[tost['MARGIN_NAME'] == 'OBSERVED']
        if not power_obs.empty:
            logger.info("  Power at observed effect: %.1f%%",
                        power_obs.iloc[0]['STATISTIC_VALUE'] * 100)
        mde = tost[tost['MARGIN_NAME'] == 'MDE_80PCT_POWER']
        if not mde.empty:
            logger.info("  MDE (80%% power): d=%.3f ($%.0f/day)",
                        mde.iloc[0]['DELTA'], mde.iloc[0]['DIFF_MEAN'])
    if not subgroup.empty:
        sig_subs = subgroup[subgroup.get('BH_SIGNIFICANT', pd.Series(dtype=int)) == 1]
        logger.info("  Subgroup analysis: %d/%d subgroups significant (BH)",
                    len(sig_subs), len(subgroup))
        for _, sub_row in sig_subs.iterrows():
            logger.info("    → %s: d=%.3f, p=%.4f, n=%d",
                        sub_row['SUBGROUP'], sub_row['COHEN_D'],
                        sub_row['P_VALUE'], sub_row['N_EVENTS'])
    if not vol_shift.empty:
        n_tests = len(vol_shift)
        sig_vs = vol_shift[vol_shift.get('P_VALUE', pd.Series(dtype=float)) < 0.05]
        logger.info("  Volatility-shift: %d tests, %d significant (p<.05)",
                    n_tests, len(sig_vs))
        cw_row = vol_shift[vol_shift['TEST'] == 'CW_VS_NON_CW']
        if not cw_row.empty:
            logger.info("    CW vs non-CW: diff=%.4f, p=%.4f",
                        cw_row.iloc[0].get('DIFF_MEAN', 0),
                        cw_row.iloc[0].get('P_VALUE', 1))
    # Tail diagnostic summary
    tail_leaning = tail_diag.get('leaning_test', pd.DataFrame())
    tail_event_type = tail_diag.get('event_type', pd.DataFrame())
    tail_interaction = tail_diag.get('interaction', pd.DataFrame())
    if not tail_leaning.empty:
        chi2_r = tail_leaning[tail_leaning['TEST'] == 'CHI2_LEANING_VS_TAIL']
        if not chi2_r.empty:
            logger.info("  Tail leaning chi2: p=%.4f", chi2_r.iloc[0]['P_VALUE'])
    if not tail_event_type.empty:
        pvr = tail_event_type[tail_event_type['COMPARISON'] == 'PLANNED_VS_REACTIVE']
        if not pvr.empty:
            logger.info("  Planned vs Reactive: d=%.3f, t-p=%.4f",
                        pvr.iloc[0]['COHEN_D'], pvr.iloc[0]['T_PVALUE'])
    if not tail_interaction.empty:
        cells = tail_interaction[tail_interaction['LEAN'] != 'INTERACTION']
        if not cells.empty:
            best = cells.loc[cells['MEAN_ABN_SELL'].idxmax()]
            logger.info("  Highest interaction cell: %s × %s (d=%.3f, p=%.4f)",
                        best.get('LEAN', '?'), best.get('EVENT_TYPE', '?'),
                        best.get('COHEN_D', 0), best.get('P_VALUE', 1))
    if not cons_planned.empty:
        actual = cons_planned[cons_planned['TICKER'] != '_POWER_DIAGNOSTIC']
        power = cons_planned[cons_planned['TICKER'] == '_POWER_DIAGNOSTIC']
        logger.info("  Conservative × Planned: %d case-study events", len(actual))
        if not power.empty:
            logger.info("    %s", power.iloc[0].get('EVENT_DESCRIPTION', ''))
    logger.info("=" * 60)

    return results


# Keep for backward compatibility
def classify_inflation_regime(store: DataStore, low=2.0, high=4.0):
    """Classify months into inflation regimes using Core PCE YoY."""
    if store.inflation.empty:
        return pd.DataFrame()

    col = None
    for candidate in ['CORE_PCE_YOY', 'CORE_CPI_YOY', 'CPI_YOY']:
        if candidate in store.inflation.columns:
            col = candidate
            break

    if col is None:
        logger.warning("No inflation YoY column found")
        return pd.DataFrame()

    inf = store.inflation[['DATE', col]].dropna().copy()
    inf['INFLATION_REGIME'] = pd.cut(
        inf[col],
        bins=[-np.inf, low, high, np.inf],
        labels=['Low', 'Moderate', 'High'],
    )
    return inf[['DATE', 'INFLATION_REGIME']]


# ── CLI entry point ──────────────────────────────────────────────────
if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO,
                        format='%(asctime)s %(levelname)s %(message)s')
    run_essay3()
