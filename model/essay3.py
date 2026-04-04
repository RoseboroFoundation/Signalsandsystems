"""Essay 3 — Insider Trading Around Culture War Events.

Analyses:
  1. Event-level insider trading panel (windows around each event)
  2. Window summary statistics
  3. Abnormal selling tests (pre-event vs benchmark)
  4. Treatment vs control regression
  5. Political leaning analysis
  6. VIX regime interaction
  7. Routine vs opportunistic insider classification
  8. Placebo permutation test
  9. Acceleration (Jonckheere-Terpstra) test
 10. CAR-insider regression
 11. Information gradient

Saves 11 tables to SQLite for visual.py and dashboard.py.
"""

import logging
import os
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
SELL_CODES = {'S', 'F'}   # S = open-market sale, F = tax withholding
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
    for col in ['shares', 'price_per_share', 'transaction_value', 'shares_owned_after']:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors='coerce').fillna(0)
    logger.info("Loaded Form 4: %d transactions, %d tickers",
                len(df), df['ticker'].nunique())
    return df


def _classify_trade(code, acq_disp):
    """Classify a transaction as 'sell', 'buy', or 'other'."""
    code = str(code).upper().strip()
    acq_disp = str(acq_disp).upper().strip()
    if code in SELL_CODES or (code == 'M' and acq_disp == 'D'):
        return 'sell'
    if code in BUY_CODES:
        return 'buy'
    # Use acquired/disposed as fallback
    if acq_disp == 'D':
        return 'sell'
    if acq_disp == 'A':
        return 'buy'
    return 'other'


def _is_routine_insider(form4, ticker, owner, event_date, lookback_years=2):
    """
    An insider is 'routine' if they traded in >= 3 of last 8 quarters
    before the event. Otherwise 'opportunistic'.
    """
    start = event_date - pd.Timedelta(days=lookback_years * 365)
    mask = ((form4['ticker'] == ticker) &
            (form4['owner_name'] == owner) &
            (form4['transaction_date'] >= start) &
            (form4['transaction_date'] < event_date))
    trades = form4.loc[mask, 'transaction_date']
    if trades.empty:
        return False  # opportunistic by default
    quarters = trades.dt.to_period('Q').nunique()
    return quarters >= 3


def _compute_window_metrics(txns, window_days):
    """Compute trading metrics for a set of transactions in a window."""
    n_days = abs(window_days[1] - window_days[0]) + 1

    sells = txns[txns['_trade_type'] == 'sell']
    buys = txns[txns['_trade_type'] == 'buy']

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
    n_opportunistic = txns[txns['_is_routine'] == False]['owner_name'].nunique() if len(txns) > 0 else 0
    n_routine = txns[txns['_is_routine'] == True]['owner_name'].nunique() if len(txns) > 0 else 0

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

        # Treatment event
        event_rows.append({
            'TICKER': ticker,
            'EVENT_ID': event_id,
            'EVENT_DATE': event_date,
            'IS_TREATMENT': True,
            'LEAN': lean,
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
            })

    event_list = pd.DataFrame(event_rows)
    logger.info("  %d events (%d treatment, %d control)",
                len(event_list),
                event_list['IS_TREATMENT'].sum(),
                (~event_list['IS_TREATMENT']).sum())

    # Precompute routine/opportunistic classification per insider
    # (cache to avoid recomputing for every window)
    logger.info("  Classifying routine vs opportunistic insiders...")
    routine_cache = {}
    for ticker in event_list['TICKER'].unique():
        ticker_txns = form4[form4['ticker'] == ticker]
        for owner in ticker_txns['owner_name'].unique():
            # Use the earliest event date for this ticker as reference
            ref_dates = event_list.loc[event_list['TICKER'] == ticker, 'EVENT_DATE']
            if ref_dates.empty:
                continue
            ref_date = ref_dates.min()
            key = (ticker, owner)
            routine_cache[key] = _is_routine_insider(form4, ticker, owner, ref_date)

    # Tag transactions with trade type and routine flag
    form4 = form4.copy()
    form4['_trade_type'] = form4.apply(
        lambda r: _classify_trade(r['transaction_code'], r['acquired_disposed']),
        axis=1
    )
    form4['_is_routine'] = form4.apply(
        lambda r: routine_cache.get((r['ticker'], r['owner_name']), False),
        axis=1
    )

    # Build CAR lookup
    car_lookup = {}
    if car_df is not None and not car_df.empty:
        for _, row in car_df.iterrows():
            key = (row['TICKER'], pd.Timestamp(row['EVENT_DATE']).strftime('%Y%m%d'))
            car_lookup[key] = row.get('CAR', None)

    # Compute metrics for each event × window
    panel_rows = []
    n_events = len(event_list)
    for idx, ev in event_list.iterrows():
        ticker = ev['TICKER']
        event_date = ev['EVENT_DATE']
        ticker_txns = form4[form4['ticker'] == ticker].copy()

        row = {
            'TICKER': ticker,
            'EVENT_ID': ev['EVENT_ID'],
            'EVENT_DATE': event_date,
            'IS_TREATMENT': ev['IS_TREATMENT'],
            'LEAN': ev['LEAN'],
        }

        # Look up CAR
        car_key = (ticker, event_date.strftime('%Y%m%d'))
        car_val = car_lookup.get(car_key)
        row['CAR_POST'] = car_val
        row['CAR_PRE'] = None
        row['CAR_FULL'] = None

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

        if (idx + 1) % 50 == 0:
            logger.info("  Processed %d/%d events", idx + 1, n_events)

    panel = pd.DataFrame(panel_rows)
    logger.info("  Panel built: %d rows, %d with sufficient data",
                len(panel), panel['HAS_SUFFICIENT_DATA'].sum())
    return panel


# ═══════════════════════════════════════════════════════════════════════
# AGGREGATE ANALYSES
# ═══════════════════════════════════════════════════════════════════════

def compute_window_summary(panel):
    """Aggregate insider trading metrics by window."""
    windows = ['BENCHMARK', 'PRE_FAR', 'PRE_MID', 'PRE_NEAR', 'PRE_FULL', 'POST']
    rows = []
    for w in windows:
        col_nds = f'{w}_NET_DOLLAR_SOLD'
        col_nsr = f'{w}_NET_SELL_RATIO'
        col_nt = f'{w}_N_TRANSACTIONS'
        col_no = f'{w}_N_OPPORTUNISTIC'
        if col_nds not in panel.columns:
            continue
        rows.append({
            'WINDOW': w,
            'N_EVENTS': len(panel),
            'MEAN_NET_DOLLAR_SOLD': panel[col_nds].mean(),
            'MEDIAN_NET_DOLLAR_SOLD': panel[col_nds].median(),
            'STD_NET_DOLLAR_SOLD': panel[col_nds].std(),
            'MEAN_NET_SELL_RATIO': panel[col_nsr].mean() if col_nsr in panel.columns else 0,
            'MEDIAN_NET_SELL_RATIO': panel[col_nsr].median() if col_nsr in panel.columns else 0,
            'MEAN_N_TRANSACTIONS': panel[col_nt].mean() if col_nt in panel.columns else 0,
            'TOTAL_TRANSACTIONS': int(panel[col_nt].sum()) if col_nt in panel.columns else 0,
            'MEAN_N_OPPORTUNISTIC': panel[col_no].mean() if col_no in panel.columns else 0,
        })
    return pd.DataFrame(rows)


def compute_abnormal_selling(panel):
    """Paired t-test: pre-event daily selling vs benchmark daily selling."""
    bench_days = abs(WINDOWS['BENCHMARK'][1] - WINDOWS['BENCHMARK'][0]) + 1
    windows = ['PRE_FAR', 'PRE_MID', 'PRE_NEAR', 'PRE_FULL']
    rows = []
    for w in windows:
        w_days = abs(WINDOWS[w][1] - WINDOWS[w][0]) + 1
        pre_col = f'{w}_NET_DOLLAR_SOLD'
        bench_col = 'BENCHMARK_NET_DOLLAR_SOLD'
        if pre_col not in panel.columns or bench_col not in panel.columns:
            continue

        pre_daily = panel[pre_col] / w_days
        bench_daily = panel[bench_col] / bench_days
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
            'N_PAIRS': len(panel),
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

    df = panel[[dep_var, 'IS_TREATMENT', 'LEAN', 'EVENT_ID']].dropna(subset=[dep_var]).copy()
    df['IS_TREATMENT_INT'] = df['IS_TREATMENT'].astype(int)

    # Spec 1: Simple OLS with clustering on EVENT_ID
    try:
        y = df[dep_var].astype(float)
        X = sm.add_constant(df[['IS_TREATMENT_INT']].astype(float))
        model = sm.OLS(y, X).fit(cov_type='cluster',
                                  cov_kwds={'groups': df['EVENT_ID']})
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
                                    cov_kwds={'groups': df['EVENT_ID']})
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

    # Map each event to its VIX regime (event date VIX)
    panel = panel.copy()
    panel['EVENT_DATE'] = pd.to_datetime(panel['EVENT_DATE'])
    panel = panel.merge(
        vix[['DATE', 'VIX_REGIME']].rename(columns={'DATE': 'EVENT_DATE'}),
        on='EVENT_DATE', how='left'
    )
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
    """Compare routine vs opportunistic insider trading pre vs benchmark."""
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
        if pre_col not in panel.columns or bench_col not in panel.columns:
            continue

        pre_vals = panel[pre_col].fillna(0)
        bench_vals = panel[bench_col].fillna(0)

        if metric in ('NET_DOLLAR_SOLD', 'DOLLAR_SOLD'):
            pre_daily = pre_vals / pre_days
            bench_daily = bench_vals / bench_days
        else:
            pre_daily = pre_vals
            bench_daily = bench_vals

        diff = pre_daily - bench_daily
        t_stat, t_p = (np.nan, np.nan)
        if len(diff) > 1:
            t_stat, t_p = stats.ttest_rel(pre_daily, bench_daily)

        rows.append({
            'TEST': test_name,
            'N_PAIRS': len(panel),
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
    """Permutation test: shuffle treatment labels, compare selling."""
    rng = np.random.RandomState(seed)
    pre_col = 'PRE_FULL_NET_DOLLAR_SOLD'
    bench_col = 'BENCHMARK_NET_DOLLAR_SOLD'
    if pre_col not in panel.columns or bench_col not in panel.columns:
        return pd.DataFrame()

    pre_days = abs(WINDOWS['PRE_FULL'][1] - WINDOWS['PRE_FULL'][0]) + 1
    bench_days = abs(WINDOWS['BENCHMARK'][1] - WINDOWS['BENCHMARK'][0]) + 1

    pre_daily = panel[pre_col] / pre_days
    bench_daily = panel[bench_col] / bench_days
    diff = pre_daily - bench_daily

    observed = diff.mean()

    # Permutation: shuffle differences
    placebo_stats = []
    vals = diff.values.copy()
    for _ in range(n_iterations):
        signs = rng.choice([-1, 1], size=len(vals))
        placebo_stats.append((vals * signs).mean())

    placebo_stats = np.array(placebo_stats)
    percentile = (placebo_stats < observed).mean() * 100
    empirical_p = (np.abs(placebo_stats) >= np.abs(observed)).mean()

    return pd.DataFrame([{
        'TEST': 'PLACEBO_PERMUTATION',
        'OBSERVED_STAT': observed,
        'PLACEBO_MEAN': placebo_stats.mean(),
        'PLACEBO_STD': placebo_stats.std(),
        'PERCENTILE': percentile,
        'EMPIRICAL_P': empirical_p,
        'N_ITERATIONS': n_iterations,
        'N_FIRMS': panel['TICKER'].nunique(),
    }])


def compute_acceleration_test(panel):
    """Jonckheere-Terpstra test for monotonic increase far → mid → near."""
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

        # Jonckheere-Terpstra: test for ordered increase
        try:
            # Use scipy's JT test (available in recent scipy)
            jt_result = stats.page_trend_test(
                np.column_stack(groups),
                predicted_ranks=[1, 2, 3],
                method='exact' if len(sub) < 20 else 'asymptotic'
            )
            jt_stat = jt_result.statistic
            jt_p = jt_result.pvalue
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
    # Spec 1: CAR ~ selling
    try:
        y = valid['CAR_POST'].astype(float)
        X = sm.add_constant(valid[['PRE_FULL_NET_DOLLAR_SOLD']].astype(float))
        model = sm.OLS(y, X).fit()
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
        valid2 = valid.copy()
        valid2['IS_TREATMENT_INT'] = valid2['IS_TREATMENT'].astype(int)
        lean_dummies = pd.get_dummies(valid2['LEAN'], prefix='LEAN', drop_first=True).astype(float)
        X2 = sm.add_constant(pd.concat([
            valid2[['PRE_FULL_NET_DOLLAR_SOLD', 'IS_TREATMENT_INT']].astype(float),
            lean_dummies
        ], axis=1))
        model2 = sm.OLS(y[:len(X2)], X2).fit()
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
    """Information gradient: how selling intensifies approaching the event."""
    windows = ['BENCHMARK', 'PRE_FAR', 'PRE_MID', 'PRE_NEAR', 'POST']
    metrics = ['NET_DOLLAR_SOLD', 'N_SELLS', 'N_UNIQUE_INSIDERS', 'NET_SELL_RATIO']

    rows = []
    for metric in metrics:
        for w in windows:
            col = f'{w}_{metric}'
            if col not in panel.columns:
                continue
            days = abs(WINDOWS[w][1] - WINDOWS[w][0]) + 1
            vals = panel[col].fillna(0)
            if metric in ('NET_DOLLAR_SOLD',):
                daily = vals / days
            else:
                daily = vals
            rows.append({
                'METRIC': metric,
                'WINDOW': w,
                'MEAN': daily.mean(),
                'MEDIAN': daily.median(),
                'STD': daily.std(),
                'N': len(vals),
            })

    return pd.DataFrame(rows)


# ═══════════════════════════════════════════════════════════════════════
# SAVE RESULTS
# ═══════════════════════════════════════════════════════════════════════

def save_essay3_results(store, panel, window_summary, abnormal, treatment_ctrl,
                         leaning, regime, rvo, placebo, accel, car_reg,
                         gradient):
    """Save all Essay 3 tables to the database."""
    timestamp = pd.Timestamp.now().isoformat() + 'Z'
    tables = {
        'ESSAY3_INSIDER_PANEL': panel,
        'ESSAY3_WINDOW_SUMMARY': window_summary,
        'ESSAY3_ABNORMAL_SELLING': abnormal,
        'ESSAY3_TREATMENT_VS_CONTROL': treatment_ctrl,
        'ESSAY3_LEANING_ANALYSIS': leaning,
        'ESSAY3_REGIME_INTERACTION': regime,
        'ESSAY3_ROUTINE_VS_OPPORTUNISTIC': rvo,
        'ESSAY3_PLACEBO_TEST': placebo,
        'ESSAY3_ACCELERATION_TEST': accel,
        'ESSAY3_CAR_INSIDER_REGRESSION': car_reg,
        'ESSAY3_INFORMATION_GRADIENT': gradient,
    }

    for table_name, df in tables.items():
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

    # Save all results
    logger.info("Saving results...")
    save_essay3_results(store, panel, window_summary, abnormal, treatment_ctrl,
                        leaning, regime, rvo, placebo, accel, car_reg, gradient)

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
    logger.info("=" * 60)

    return {
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
    }


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
