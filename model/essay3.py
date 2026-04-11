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

Saves 23 tables to SQLite for visual.py and dashboard.py.
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
    for col in ['shares', 'shares_owned_after']:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors='coerce').fillna(0)
    for col in ['price_per_share', 'transaction_value']:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors='coerce')
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
        return False  # opportunistic by default (no prior trading history)
    quarters = trades.dt.to_period('Q').nunique()
    return quarters >= 3


def _compute_window_metrics(txns, window_days):
    """Compute trading metrics for a set of transactions in a window."""
    n_days = abs(window_days[1] - window_days[0]) + 1

    sells = txns[txns['_trade_type'] == 'sell'].copy()
    buys = txns[txns['_trade_type'] == 'buy'].copy()

    # Impute missing transaction_value from shares × price where possible
    for subset in [sells, buys]:
        missing = (subset['transaction_value'].isna() &
                   subset['shares'].notna() &
                   subset['price_per_share'].notna())
        if missing.any():
            subset.loc[missing, 'transaction_value'] = (
                subset.loc[missing, 'shares'] * subset.loc[missing, 'price_per_share']
            )

    # Log transactions where value could not be imputed
    for label, subset in [('sell', sells), ('buy', buys)]:
        n_missing = subset['transaction_value'].isna().sum()
        if n_missing > 0:
            logger.debug("  %d %s transactions have no imputable value "
                         "(shares or price missing)", n_missing, label)

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

    # Tag transactions with trade type
    form4 = form4.copy()
    form4['_trade_type'] = form4.apply(
        lambda r: _classify_trade(r['transaction_code'], r['acquired_disposed']),
        axis=1
    )
    # _is_routine will be set per-event in the panel loop below
    form4['_is_routine'] = False

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
        ticker_txns['_is_routine'] = ticker_txns['owner_name'].apply(
            lambda owner: routine_cache.get((ticker, owner, event_date), False)
        )

        row = {
            'TICKER': ticker,
            'EVENT_ID': ev['EVENT_ID'],
            'EVENT_DATE': event_date,
            'IS_TREATMENT': ev['IS_TREATMENT'],
            'LEAN': ev['LEAN'],
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

    # Map each event to nearest prior trading day's VIX regime
    panel = panel.copy()
    panel['EVENT_DATE'] = pd.to_datetime(panel['EVENT_DATE'])
    vix = vix.sort_values('DATE')
    panel = panel.sort_values('EVENT_DATE')
    panel = pd.merge_asof(
        panel, vix[['DATE', 'VIX_REGIME']],
        left_on='EVENT_DATE', right_on='DATE',
        direction='backward', tolerance=pd.Timedelta(days=5)
    )
    panel = panel.drop(columns=['DATE'], errors='ignore')
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
    diff = (pre_daily - bench_daily).values

    treatment_mask = panel['IS_TREATMENT'].values
    observed = diff[treatment_mask].mean() - diff[~treatment_mask].mean()

    # Permutation: shuffle treatment labels
    placebo_stats = []
    for _ in range(n_iterations):
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
        valid2 = valid.copy().reset_index(drop=True)
        valid2['IS_TREATMENT_INT'] = valid2['IS_TREATMENT'].astype(int)
        lean_dummies = pd.get_dummies(valid2['LEAN'], prefix='LEAN', drop_first=True).astype(float)
        lean_dummies = lean_dummies.reset_index(drop=True)
        X2 = sm.add_constant(pd.concat([
            valid2[['PRE_FULL_NET_DOLLAR_SOLD', 'IS_TREATMENT_INT']].astype(float),
            lean_dummies
        ], axis=1))
        y2 = valid2['CAR_POST'].astype(float)
        model2 = sm.OLS(y2, X2).fit()
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
            if metric in ('NET_DOLLAR_SOLD', 'N_SELLS', 'N_UNIQUE_INSIDERS'):
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

    # Stack pre and post into long form using raw daily selling
    long_rows = []
    for _, row in panel.iterrows():
        base = {
            'TICKER': row['TICKER'],
            'EVENT_DATE': row['EVENT_DATE'],
            'IS_TREATMENT': int(row['IS_TREATMENT']),
        }
        pre_daily = row['PRE_FULL_NET_DOLLAR_SOLD'] / pre_days
        post_daily = row['POST_NET_DOLLAR_SOLD'] / post_days
        long_rows.append({**base, 'IS_POST': 0, 'NET_SELL_DAILY': pre_daily})
        long_rows.append({**base, 'IS_POST': 1, 'NET_SELL_DAILY': post_daily})

    long_df = pd.DataFrame(long_rows).dropna(subset=['NET_SELL_DAILY'])

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

    # Spec 2: DiD with firm fixed effects (full within transformation)
    # Demean ALL variables by firm. IS_TREATMENT is constant within firm
    # and will be absorbed by the FE (correctly dropped from the model).
    try:
        demeaned = long_df.copy()
        for col in ['NET_SELL_DAILY', 'IS_TREATMENT', 'IS_POST', 'TREAT_X_POST']:
            firm_mean = demeaned.groupby('TICKER')[col].transform('mean')
            demeaned[f'{col}_DM'] = demeaned[col] - firm_mean

        y2 = demeaned['NET_SELL_DAILY_DM'].astype(float)
        # IS_TREATMENT_DM will be ~0 (constant within firm → absorbed by FE)
        # Only IS_POST_DM and TREAT_X_POST_DM have within-firm variation
        x_cols = ['IS_POST_DM', 'TREAT_X_POST_DM']
        # Include IS_TREATMENT_DM only if it has variation (mixed firms)
        if demeaned['IS_TREATMENT_DM'].abs().sum() > 1e-10:
            x_cols = ['IS_TREATMENT_DM'] + x_cols
        # No constant after within transformation (intercept is zero by construction)
        X2 = demeaned[x_cols].astype(float)
        model2 = sm.OLS(y2, X2).fit(cov_type='cluster',
                                     cov_kwds={'groups': demeaned['TICKER']})
        for var in model2.params.index:
            # Map demeaned variable names back for readability
            var_label = var.replace('_DM', '')
            rows.append({
                'SPECIFICATION': 'DID_FIRM_FE',
                'VARIABLE': var_label,
                'COEFFICIENT': model2.params[var],
                'STD_ERROR': model2.bse[var],
                'T_STAT': model2.tvalues[var],
                'P_VALUE': model2.pvalues[var],
                'R_SQUARED': model2.rsquared,
                'N_OBS': int(model2.nobs),
            })
    except Exception as e:
        logger.warning("DiD firm FE failed: %s", e)

    df = pd.DataFrame(rows)
    if not df.empty:
        df['BH_SIGNIFICANT'] = _benjamini_hochberg(df['P_VALUE'].values).astype(int)
    return df


def compute_10b5_1_filter(panel, form4):
    """Flag likely 10b5-1 (scheduled) trades using routine trading heuristic.

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
                eigvals = np.linalg.eigvalsh(var_twoway)
                if np.any(eigvals < 0):
                    logger.warning("Two-way cluster VCV not PSD (min eigval=%.4e); "
                                   "applying ridge correction", eigvals.min())
                    var_twoway += np.eye(len(var_twoway)) * (abs(eigvals.min()) * 1.1 + 1e-10)
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
            model = sm.OLS(y, X).fit()
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
                    })
        except Exception as e:
            logger.warning("Firm FE (multi-event) failed: %s", e)

    # Spec 2: Within-firm demeaning (Mundlak)
    if len(df_multi) >= 10:
        try:
            firm_means = df_multi.groupby('TICKER')['DAILY_SELL'].transform('mean')
            y_dm = (df_multi['DAILY_SELL'] - firm_means).astype(float).reset_index(drop=True)
            treat_dm = (df_multi['IS_TREATMENT_INT'] -
                        df_multi.groupby('TICKER')['IS_TREATMENT_INT'].transform('mean'))
            X_dm = sm.add_constant(treat_dm.astype(float).reset_index(drop=True))
            model_dm = sm.OLS(y_dm, X_dm).fit()
            for var in model_dm.params.index:
                rows.append({
                    'SPECIFICATION': 'WITHIN_FIRM_DEMEANED',
                    'VARIABLE': var,
                    'COEFFICIENT': model_dm.params[var],
                    'STD_ERROR': model_dm.bse[var],
                    'T_STAT': model_dm.tvalues[var],
                    'P_VALUE': model_dm.pvalues[var],
                    'R_SQUARED': model_dm.rsquared,
                    'N_OBS': int(model_dm.nobs),
                    'N_FIRMS': len(multi_event_firms),
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

    if 'BENCHMARK_NET_DOLLAR_SOLD' in treatment.columns:
        treat_df['BENCH_ACTIVITY'] = treatment['BENCHMARK_NET_DOLLAR_SOLD'].fillna(0) / bench_days
        regressors.append('BENCH_ACTIVITY')

    if 'BENCHMARK_N_OPPORTUNISTIC' in treatment.columns and 'BENCHMARK_N_UNIQUE_INSIDERS' in treatment.columns:
        treat_df['OPP_SHARE'] = np.where(
            treatment['BENCHMARK_N_UNIQUE_INSIDERS'] > 0,
            treatment['BENCHMARK_N_OPPORTUNISTIC'] / treatment['BENCHMARK_N_UNIQUE_INSIDERS'],
            0)
        regressors.append('OPP_SHARE')

    if 'LEAN' in treatment.columns:
        lean_dummies = pd.get_dummies(treatment['LEAN'], prefix='LEAN', drop_first=True).astype(float)
        lean_dummies = lean_dummies.reset_index(drop=True)
        treat_df = treat_df.reset_index(drop=True)
        treat_df = pd.concat([treat_df, lean_dummies], axis=1)
        regressors.extend(lean_dummies.columns.tolist())

    treat_df = treat_df.dropna(subset=['ABN_SELL']).reset_index(drop=True)
    if len(treat_df) < 15:
        return pd.DataFrame()

    x_cols = [c for c in regressors if c != 'ABN_SELL' and c in treat_df.columns]
    if not x_cols:
        return pd.DataFrame()

    rows = []
    try:
        y = treat_df['ABN_SELL'].astype(float)
        X = sm.add_constant(treat_df[x_cols].astype(float))
        model = sm.OLS(y, X).fit()
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
            # Pairs share EVENT_DATE; align by sorting
            t_sorted = treatment.sort_values(['EVENT_DATE', 'TICKER'])
            c_sorted = control.sort_values(['EVENT_DATE', 'TICKER'])
            t_vals = (t_sorted[pre_col] / pre_days - t_sorted[bench_col] / bench_days).values
            c_vals = (c_sorted[pre_col] / pre_days - c_sorted[bench_col] / bench_days).values
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
            params_idx = model.params.index.str.lower()
            const_pos = next((i for i, k in enumerate(params_idx)
                              if k in ('const', 'intercept')), 0)
            treat_pos = next((i for i, k in enumerate(params_idx)
                              if 'treatment' in k), None)
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
