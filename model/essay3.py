"""Essay 3 — Informed Insider Trading Around Political Decisions:
Foreknowledge, Profits, and the Limits of Regulatory Architecture.

Research question: Do insiders trade on advance knowledge of political
decisions that move firm equity values?

Core argument (Roseboro 2026):
  In the 180 days before political decisions that move firm equity values,
  insiders trade in directions that significantly predict the event-day price
  reaction. Pre-event sells are ~56% directionally accurate (vs 50% null,
  p < 1e-66) compared to ~53% in matched non-political windows (premium
  ~+3pp, p < 0.001). The aggregate dollar magnitude is billions in informed-
  trading profits over 2000-2025, concentrated in the 30 days before events
  with large negative CARs.

  The buy/sell asymmetry and proximity-to-event pattern suggest informed
  trading operates through both opportunistic and structural channels.
  Post-2023 cooling-off amendments have not attenuated informed trading;
  the directional-accuracy premium is highest in the most recent regulatory
  period, raising the question of whether ex-ante commitment requirements
  may institutionalize rather than disrupt informed pre-event trading.

  Connection to Essay 2: the dollar profits documented here account for
  a substantial fraction of the price discovery around political events.
  The persistent mispricing in Essay 2 reflects, in part, systematic
  extraction of value by informed insiders — a microfoundation linking
  political-economic risk to corporate valuation through an identifiable
  channel.

Methodology:
  Headline tests (§12):
    1. Event-CAR-based directional profitability: political vs control
    2. By trade direction (buy/sell asymmetry)
    3. By proximity to event (0-30, 31-60, 61-90, 91-180 days)
    4. By event-CAR severity (−5%, −10%, −15% thresholds)
    5. Dollar magnitudes by cut
    6. Pre/post regulatory period comparison

  Supporting evidence (CG size-accuracy attenuation):
    7. Trade-size decile accuracy table (political vs matched control)
    8. LPM: Pr(accurate) ~ log(size) × political interaction
    9. Pooled and within-firm (ticker FE) specifications

  Distributional evidence (§1-§11):
   10. Family-level Wilcoxon correction (Holm + BH)
   11. Insider-level concentration metrics (Gini, HHI, top-K%)
   12. Active subset characterization
   13. Quantile/median regression, trimmed robustness, bootstrap
   14. Insider fixed-effects panel

Output tables (27, orphaned tables from prior runs are auto-dropped):
  ESSAY3_PANEL                  Event-level panel (carried forward)
  ESSAY3_STRATIFICATION         Year x activity-tercile strata
  ESSAY3_MEAN_VS_DISTRIBUTIONAL Side-by-side t-test vs Wilcoxon across all cuts
  ESSAY3_WILCOXON_FAMILY        Family-level correction (Holm + BH)
  ESSAY3_INSIDER_CONCENTRATION  Gini, HHI, top-K% profit shares
  ESSAY3_ACTIVE_SUBSET          Observable characteristics of active insiders
  ESSAY3_QUANTILE_REGRESSION    Median + quantile regressions
  ESSAY3_TRIMMED_ROBUSTNESS     Re-test after trimming 1% tails
  ESSAY3_BOOTSTRAP_WILCOXON     Bootstrapped Wilcoxon (resampled under H0)
  ESSAY3_INSIDER_PANEL          Within-insider variation (PanelOLS entity FE)
  ESSAY3_CONCENTRATION_CUTS     Concentration by event type, policy area, connection
  ESSAY3_REPEAT_TRADERS         Repeat-trader analysis
  ESSAY3_TOST                   TOST equivalence (carried forward)
  ESSAY3_PLACEBO                Permutation test (carried forward)
  ESSAY3_BOOTSTRAP_CI           Bootstrap confidence intervals (carried forward)
  ESSAY3_CRSP_PROFITS           Trade-level abnormal returns (CRSP profit analysis)
  ESSAY3_CRSP_SUMMARY           Summary of abnormal returns by cut
  ESSAY3_INFORMED_TRADING       Headline: directional accuracy by cut (headline);
                                also contains MAGNITUDE_GATED, WEIGHTED_PROPORTION,
                                and SELLS_PREMIUM_TRADE_CAR_FAR (far-window only)
  ESSAY3_INFORMED_PROXIMITY     Accuracy by proximity-to-event window (headline)
  ESSAY3_INFORMED_DOLLARS       Dollar magnitudes by proximity and severity (headline)
  ESSAY3_SIZE_ACCURACY          Accuracy by trade-size decile × sample (supporting);
                                political uses EVENT_CAR construct (same as headline),
                                control uses TRADE_CAR_30 (ACCURACY_CONSTRUCT column)
  ESSAY3_SIZE_ACCURACY_SLOPES   Slope of accuracy on log(median_tv) per sample
  ESSAY3_REVERSAL_REGRESSION    LPM: accuracy ~ log(size) × political (supporting);
                                political DV = EVENT_PROFITABLE, control = PROFITABLE_30
  ESSAY3_CONTROL_TRADES         Non-political matched control trades with CARs
  ESSAY3_CONTROL_ATTRITION      Retained vs dropped control trades after CAR
                                computation: N, size, direction by group (Lambert
                                review 2026-07-06: ~20% attrition, size-selective)
  ESSAY3_CLUSTER_INFERENCE      Event- & ticker-clustered wild bootstrap CIs,
                                Pesaran-Timmermann conditional-null variant
  ESSAY3_DIRECTIONAL_PLACEBO    Date-randomization placebo on directional accuracy
  ESSAY3_CAR_NET_SLOPE          OLS of event CAR on net sell direction (one row
                                per event, HC3 SE); base-rate-immune
  ESSAY3_JOINT_PT               Joint Pesaran-Timmermann 2x2 sign test (buys +
                                sells against independence null)
"""

import bisect
import logging
import os
import warnings

import numpy as np
import pandas as pd
from scipy import stats
import statsmodels.api as sm

from .datastore import DataStore
from .essay2_did import compute_car

logger = logging.getLogger(__name__)

# ── Constants ─────────────────────────────────────────────────────────
WINDOWS = {
    'BENCHMARK':  (-365, -181),
    'PRE_FAR':    (-180, -61),
    'PRE_MID':    (-60, -31),
    'PRE_NEAR':   (-30, -1),
    'PRE_FULL':   (-180, -1),
    'POST':       (0, 60),
}

SELL_CODES = {'S', 'D', 'F'}   # Sale, Disposition, Tax withholding via share surrender
BUY_CODES  = {'P', 'A'}       # Purchase, Acquisition (option exercise); M handled separately
SESOI_D = 0.20

REGULATORY_PERIODS = {
    'PRE_SOX':         (pd.Timestamp('2000-01-01'), pd.Timestamp('2002-12-31')),
    'SOX_ERA':         (pd.Timestamp('2003-01-01'), pd.Timestamp('2009-12-31')),
    'DODD_FRANK':      (pd.Timestamp('2010-01-01'), pd.Timestamp('2022-12-31')),
    'POST_AMENDMENTS': (pd.Timestamp('2023-01-01'), pd.Timestamp('2030-12-31')),
}


# ═════════════════════════════════════════════════════════════════════
# HELPERS
# ═════════════════════════════════════════════════════════════════════

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
    for col in ['shares', 'shares_owned_after']:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors='coerce').fillna(0)
    for col in ['price_per_share', 'transaction_value']:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors='coerce')
    if 'transaction_value' not in df.columns:
        df['transaction_value'] = df['shares'] * df['price_per_share']
    else:
        missing = df['transaction_value'].isna()
        if missing.any():
            df.loc[missing, 'transaction_value'] = (
                df.loc[missing, 'shares'] * df.loc[missing, 'price_per_share']
            )

    # Sanity-filter: drop transactions where value is implausible.
    # Two checks:
    #   1. Mechanical consistency: if shares and price are both present,
    #      flag rows where transaction_value differs by >10x from
    #      shares * price (catches data-entry errors without removing
    #      legitimate large trades like Bezos/Musk sales >$1B).
    #   2. Absolute cap at $10B as a backstop (no single Form 4
    #      transaction should exceed this).
    TV_ABS_CAP = 10_000_000_000
    n_abs = (df['transaction_value'].abs() > TV_ABS_CAP).sum()
    if n_abs > 0:
        df.loc[df['transaction_value'].abs() > TV_ABS_CAP, 'transaction_value'] = np.nan
        logger.info("  Dropped %d transactions with |value| > $10B", n_abs)

    # Mechanical consistency check: shares * price vs reported value
    # Use full-length series to avoid fragile index alignment
    has_both = (df['shares'] > 0) & df['price_per_share'].notna() & df['transaction_value'].notna()
    if has_both.any():
        computed = pd.Series(np.nan, index=df.index)
        computed.loc[has_both] = (df.loc[has_both, 'shares']
                                  * df.loc[has_both, 'price_per_share'])
        ratio = pd.Series(np.nan, index=df.index)
        ratio.loc[has_both] = (df.loc[has_both, 'transaction_value']
                               / computed.loc[has_both])
        ratio = ratio.replace([np.inf, -np.inf], np.nan)
        bad = ratio.notna() & ((ratio > 10) | (ratio < 0.1))
        n_bad = bad.sum()
        if n_bad > 0:
            df.loc[bad, 'transaction_value'] = computed.loc[bad]
            logger.info("  Fixed %d transactions where value differed >10x "
                        "from shares*price", n_bad)

    logger.info("Loaded Form 4: %d transactions, %d tickers",
                len(df), df['ticker'].nunique())
    return df


def _classify_trade(code, acq_disp):
    """Classify a transaction as 'sell', 'buy', or 'other'.

    For code 'M' (exercise/conversion), the acq_disp field determines
    whether the underlying was acquired ('A' → buy) or disposed ('D' → sell).
    """
    code = str(code).upper().strip()
    if code in SELL_CODES:
        return 'sell'
    if code == 'M':
        # Exercise/conversion: direction depends on whether underlying was acquired or disposed
        ad = str(acq_disp).upper().strip()
        return 'sell' if ad == 'D' else 'buy'
    if code in BUY_CODES:
        return 'buy'
    return 'other'


def _build_routine_cache(form4):
    """Precompute (ticker, owner) -> (month, year_set) for CMP routine classification.

    Cohen-Malloy-Pomorski (2012): an insider is 'routine' for a given calendar
    month if they traded in that same calendar month in >= 3 of the prior 5 years.
    We precompute per (ticker, owner, month) the set of years they traded in.
    """
    df = form4[['ticker', 'owner_name', 'transaction_date']].dropna().copy()
    df['month'] = df['transaction_date'].dt.month
    df['year'] = df['transaction_date'].dt.year
    # Group by (ticker, owner, month) -> set of years
    cache = {}
    grouped = df.groupby(['ticker', 'owner_name', 'month'])['year'].apply(set).reset_index()
    for _, row in grouped.iterrows():
        key = (row['ticker'], row['owner_name'], row['month'])
        cache[key] = row['year']
    return cache


def _is_routine_from_cache(cache, ticker, owner, event_date, lookback_years=5,
                           data_start_year=2000):
    """Month-of-trade routine classification, modified from CMP (2012).

    Cohen-Malloy-Pomorski define routine at the insider-year level; our version
    is finer-grained — evaluated per (ticker, owner, event_month).  An insider
    is routine for this month if they traded in the same calendar month in >= 3
    of the prior 5 years.

    If the full lookback window extends before `data_start_year`, fewer than
    `lookback_years` years of history are available.  When fewer than 3 years
    are observable we cannot distinguish routine from opportunistic and return
    None rather than risk systematic misclassification of early-sample insiders.
    """
    event_month = event_date.month
    event_year = event_date.year
    lookback_start = event_year - lookback_years
    # Guard: require at least 3 observable years in the lookback window.
    available_years = event_year - max(lookback_start, data_start_year)
    if available_years < 3:
        return None
    years_traded = cache.get((ticker, owner, event_month))
    if not years_traded:
        return None
    # Count years in the lookback window [event_year - lookback_years, event_year - 1]
    n_years = sum(1 for y in years_traded if lookback_start <= y < event_year)
    return n_years >= 3


def _compute_window_metrics(txns, window_days):
    """Compute trading metrics for transactions in a window."""
    sells = txns[txns['_trade_type'] == 'sell']
    buys = txns[txns['_trade_type'] == 'buy']

    n_transactions = len(txns)
    dollar_sold = sells['transaction_value'].sum()
    dollar_bought = buys['transaction_value'].sum()
    net_dollar = dollar_sold - dollar_bought
    total_dollar = dollar_sold + dollar_bought
    net_sell_ratio = net_dollar / total_dollar if total_dollar > 0 else 0.0
    n_unique_insiders = txns['owner_name'].nunique() if len(txns) > 0 else 0
    n_opportunistic = txns[txns['_is_routine'] == False]['owner_name'].nunique() if len(txns) > 0 else 0  # noqa: E712
    n_routine = txns[txns['_is_routine'] == True]['owner_name'].nunique() if len(txns) > 0 else 0  # noqa: E712

    return {
        'N_TRANSACTIONS': n_transactions,
        'N_SELLS': len(sells),
        'N_BUYS': len(buys),
        'SHARES_SOLD': sells['shares'].sum(),
        'SHARES_BOUGHT': buys['shares'].sum(),
        'DOLLAR_SOLD': dollar_sold,
        'DOLLAR_BOUGHT': dollar_bought,
        'NET_SELLING': net_dollar,
        'NET_SELL_RATIO': net_sell_ratio,
        'N_UNIQUE_INSIDERS': n_unique_insiders,
        'N_OPPORTUNISTIC': n_opportunistic,
        'N_ROUTINE': n_routine,
    }


def _holm_bonferroni(pvals, alpha=0.05):
    """Holm-Bonferroni correction. Returns boolean array of significance."""
    pvals = np.array(pvals, dtype=float)
    n = len(pvals)
    if n == 0:
        return np.array([], dtype=bool)
    ranked = np.argsort(pvals)
    significant = np.zeros(n, dtype=bool)
    for i, idx in enumerate(ranked):
        if pvals[idx] <= alpha / (n - i):
            significant[idx] = True
        else:
            break
    return significant


def _benjamini_hochberg(pvals, alpha=0.05):
    """Benjamini-Hochberg FDR correction. Returns boolean array."""
    pvals = np.array(pvals, dtype=float)
    n = len(pvals)
    if n == 0:
        return np.array([], dtype=bool)
    ranked = np.argsort(pvals)
    significant = np.zeros(n, dtype=bool)
    # Work backwards from largest p-value
    for i in range(n - 1, -1, -1):
        idx = ranked[i]
        threshold = alpha * (i + 1) / n
        if pvals[idx] <= threshold:
            # This and all smaller p-values are significant
            for j in range(i + 1):
                significant[ranked[j]] = True
            break
    return significant


def _daily_rate(value, window_name):
    """Convert a window total to a per-trading-day rate.

    Uses ~252 trading days per 365 calendar days (ratio ≈ 0.69).
    """
    start, end = WINDOWS[window_name]
    cal_days = abs(end - start) + 1
    trading_days = max(1, int(round(cal_days * 252 / 365)))
    return value / trading_days if trading_days > 0 else 0.0


def _two_sample_test(vals_a, vals_b, label_a='A', label_b='B'):
    """Standard two-sample comparison returning a result dict."""
    if len(vals_a) < 5 or len(vals_b) < 5:
        return None
    t_stat, t_pval = stats.ttest_ind(vals_a, vals_b, equal_var=False)
    u_stat, u_pval = stats.mannwhitneyu(vals_a, vals_b, alternative='two-sided')
    pooled_std = np.sqrt(
        ((len(vals_a) - 1) * np.std(vals_a, ddof=1)**2 +
         (len(vals_b) - 1) * np.std(vals_b, ddof=1)**2) /
        (len(vals_a) + len(vals_b) - 2)
    )
    cohen_d = (np.mean(vals_a) - np.mean(vals_b)) / pooled_std if pooled_std > 0 else 0
    return {
        'N_A': len(vals_a), 'N_B': len(vals_b),
        'MEAN_A': vals_a.mean(), 'MEAN_B': vals_b.mean(),
        'DIFF_MEAN': vals_a.mean() - vals_b.mean(),
        'T_STAT': t_stat, 'P_VALUE': t_pval,
        'U_STAT': u_stat, 'U_PVALUE': u_pval,
        'COHEN_D': cohen_d,
    }


def _one_sample_test(vals):
    """One-sample t-test and Wilcoxon against zero."""
    if len(vals) < 5:
        return None
    t_stat, t_pval = stats.ttest_1samp(vals, 0)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        w_stat, w_pval = (stats.wilcoxon(vals, alternative='two-sided', zero_method='pratt')
                           if len(vals) >= 10 else (np.nan, np.nan))
    return {
        'N': len(vals),
        'MEAN': vals.mean(), 'MEDIAN': vals.median(), 'STD': vals.std(),
        'T_STAT': t_stat, 'P_VALUE': t_pval,
        'WILCOXON_STAT': w_stat, 'WILCOXON_PVALUE': w_pval,
        'COHEN_D': np.mean(vals) / np.std(vals, ddof=1) if np.std(vals, ddof=1) > 0 else 0,
        'PCT_POSITIVE': (vals > 0).mean(),
    }


def _assign_regulatory_period(event_date):
    """Assign a regulatory period label to an event date."""
    for period, (start, end) in REGULATORY_PERIODS.items():
        if start <= event_date <= end:
            return period
    return 'PRE_SOX'


def _winsorize_panel(panel, percentile=1):
    """Winsorize dollar columns within each EVENT_CATEGORY separately."""
    dollar_cols = [c for c in panel.columns
                   if any(kw in c for kw in ['DOLLAR', 'NET_TRADING', 'NET_SELLING'])]
    lower = percentile / 100
    upper = 1 - lower
    n_clipped = 0
    for cat in panel['EVENT_CATEGORY'].unique():
        mask = panel['EVENT_CATEGORY'] == cat
        for col in dollar_cols:
            vals = panel.loc[mask, col].dropna()
            if len(vals) < 10:
                continue
            lo, hi = vals.quantile(lower), vals.quantile(upper)
            before = panel.loc[mask, col].copy()
            panel.loc[mask, col] = panel.loc[mask, col].clip(lower=lo, upper=hi)
            clipped = (before != panel.loc[mask, col]).sum()
            n_clipped += clipped
    logger.info("  Winsorized %d dollar columns at %d/%d%% within-category (%d values clipped)",
                len(dollar_cols), percentile, 100 - percentile, n_clipped)
    return panel


def _compute_tost(vals, sesoi_d):
    """TOST equivalence test against zero."""
    mean_val = vals.mean()
    std_val = vals.std()
    n = len(vals)
    se = std_val / np.sqrt(n)
    delta = sesoi_d * std_val

    t_upper = (mean_val - delta) / se if se > 0 else np.nan
    t_lower = (mean_val + delta) / se if se > 0 else np.nan
    p_upper = stats.t.cdf(t_upper, df=n - 1)
    p_lower = 1 - stats.t.cdf(t_lower, df=n - 1)
    p_tost = max(p_upper, p_lower)

    ncp = mean_val / se if se > 0 else 0
    power = 1 - stats.t.cdf(
        stats.t.ppf(0.975, df=n - 1), df=n - 1, loc=abs(ncp)
    ) if se > 0 else np.nan

    z_alpha = stats.norm.ppf(0.975)
    z_beta = stats.norm.ppf(0.80)
    mde_raw = (z_alpha + z_beta) * se if se > 0 else np.nan
    mde_d = mde_raw / std_val if std_val > 0 and not np.isnan(mde_raw) else np.nan

    return {
        'N': n, 'MEAN': mean_val, 'STD': std_val, 'SE': se,
        'SESOI_D': sesoi_d, 'DELTA_RAW': delta,
        'P_UPPER': p_upper, 'P_LOWER': p_lower, 'P_TOST': p_tost,
        'EQUIVALENT': p_tost < 0.05,
        'POWER_AT_OBSERVED': power, 'MDE_80_D': mde_d, 'MDE_80_RAW': mde_raw,
    }


# ═════════════════════════════════════════════════════════════════════
# FUNDAMENTAL CAR COMPUTATION
# ═════════════════════════════════════════════════════════════════════

def _compute_fundamental_cars(car_df, political_events, form4, store):
    """Compute CARs for fundamental political events and merge with existing car_df."""
    if political_events is None or political_events.empty:
        return car_df

    classified = political_events[
        political_events['POLICY_AREA'].notna() &
        (political_events['POLICY_AREA'] != 'unknown') &
        (political_events['POLICY_AREA'] != '')
    ]
    MAX_POLITICAL_EVENTS = 500
    if len(classified) > MAX_POLITICAL_EVENTS:
        rng = np.random.RandomState(42)
        classified = classified.sample(n=MAX_POLITICAL_EVENTS, random_state=rng)

    form4_tickers = form4['ticker'].unique()
    pairs = set()
    for _, ev in classified.iterrows():
        event_date = pd.Timestamp(ev['EVENT_DATE'])
        affected_naics = str(ev.get('AFFECTED_NAICS', ''))
        naics_list = ([n.strip() for n in affected_naics.split(',')]
                      if affected_naics and affected_naics != 'nan' else [])
        matched = _match_tickers_to_naics(
            form4_tickers, naics_list, form4, event_date, store=store
        )
        for ticker in matched:
            pairs.add((ticker, event_date))

    existing = set()
    if car_df is not None and not car_df.empty:
        for _, row in car_df.iterrows():
            existing.add((row['TICKER'], pd.Timestamp(row['EVENT_DATE'])))
    new_pairs = pairs - existing
    if not new_pairs:
        logger.info("All fundamental CARs already computed (%d pairs)", len(pairs))
        return car_df

    logger.info("Computing CARs for %d fundamental (ticker, event_date) pairs...",
                len(new_pairs))

    new_rows = []
    done = 0
    for ticker, event_date in new_pairs:
        result = compute_car(
            ticker=ticker, store=store,
            event_id=f"POL_{ticker}_{event_date.strftime('%Y%m%d')}",
            event_date=event_date, is_treatment=True, regime='Unknown',
        )
        if result is not None:
            new_rows.append({
                'TICKER': ticker, 'EVENT_DATE': event_date,
                'CAR': result.car_post, 'CAR_PRE': result.car_pre,
                'CAR_FULL': result.car_full, 'N_OBS': result.n_event_obs,
                'R_SQUARED': result.r_squared, 'STATUS': 'OK',
            })
        done += 1
        if done % 500 == 0:
            logger.info("  CARs computed: %d/%d (%d successful)",
                        done, len(new_pairs), len(new_rows))

    logger.info("  Computed %d CARs from %d pairs (%.0f%% coverage)",
                len(new_rows), len(new_pairs),
                100 * len(new_rows) / len(new_pairs) if new_pairs else 0)

    if not new_rows:
        return car_df

    new_car_df = pd.DataFrame(new_rows)
    if car_df is not None and not car_df.empty:
        car_df = pd.concat([car_df, new_car_df], ignore_index=True)
    else:
        car_df = new_car_df
    return car_df


# ═════════════════════════════════════════════════════════════════════
# PANEL BUILDER
# ═════════════════════════════════════════════════════════════════════

def build_insider_panel(form4, culture_events, political_events,
                        political_exposure=None, car_df=None, store=None):
    """Build unified event-level insider trading panel.

    Returns a DataFrame with one row per event-firm pair, tagged with
    EVENT_CATEGORY (CULTURAL/FUNDAMENTAL), REGULATORY_PERIOD, stratification
    variables, and per-window trading metrics.
    """
    logger.info("Building unified insider trading panel...")

    form4 = form4.copy()
    if '_trade_type' not in form4.columns:
        form4['_trade_type'] = form4.apply(
            lambda r: _classify_trade(r['transaction_code'], r['acquired_disposed']),
            axis=1
        )
    form4['_is_routine'] = None

    # ── Cultural events ──────────────────────────────────────────────
    cultural_rows = []
    for _, ev in culture_events.iterrows():
        ticker = ev['TICKER']
        event_date = pd.Timestamp(ev['EVENT_DATE'])
        event_id = f"CW_{ticker}_{event_date.strftime('%Y%m%d')}"
        pair_id = f"{ticker}_{event_date.strftime('%Y%m%d')}"
        cultural_rows.append({
            'TICKER': ticker, 'EVENT_ID': event_id,
            'EVENT_DATE': event_date, 'EVENT_CATEGORY': 'CULTURAL',
            'EVENT_TYPE': 'CULTURE_WAR', 'IS_TREATMENT': True,
            'LEAN': ev.get('ESTIMATED_POLITICAL_LEANING', 'Unknown'),
            'PAIR_ID': pair_id, 'POLICY_AREA': 'cultural',
            'DESCRIPTION': ev.get('CULTURE_WAR_EVENT', ''),
        })
        ctrl_ticker = ev.get('CONTROL_TICKER')
        if pd.notna(ctrl_ticker) and ctrl_ticker:
            cultural_rows.append({
                'TICKER': ctrl_ticker,
                'EVENT_ID': f"CW_CTRL_{ctrl_ticker}_{event_date.strftime('%Y%m%d')}",
                'EVENT_DATE': event_date, 'EVENT_CATEGORY': 'CULTURAL',
                'EVENT_TYPE': 'CULTURE_WAR', 'IS_TREATMENT': False,
                'LEAN': ev.get('ESTIMATED_POLITICAL_LEANING', 'Unknown'),
                'PAIR_ID': pair_id, 'POLICY_AREA': 'cultural',
                'DESCRIPTION': '',
            })

    # ── Fundamental political events ─────────────────────────────────
    fundamental_rows = []
    if political_events is not None and not political_events.empty:
        classified = political_events[
            political_events['POLICY_AREA'].notna() &
            (political_events['POLICY_AREA'] != 'unknown') &
            (political_events['POLICY_AREA'] != '')
        ]
        logger.info("  Filtered political events: %d classified / %d total",
                     len(classified), len(political_events))

        MAX_POLITICAL_EVENTS = 500
        if len(classified) > MAX_POLITICAL_EVENTS:
            rng = np.random.RandomState(42)
            classified = classified.sample(n=MAX_POLITICAL_EVENTS, random_state=rng)
            logger.info("  Sampled %d political events", len(classified))

        form4_tickers = form4['ticker'].unique()
        for _, ev in classified.iterrows():
            event_date = pd.Timestamp(ev['EVENT_DATE'])
            affected_naics = str(ev.get('AFFECTED_NAICS', ''))
            naics_list = ([n.strip() for n in affected_naics.split(',')]
                          if affected_naics and affected_naics != 'nan' else [])
            affected_tickers = _match_tickers_to_naics(
                form4_tickers, naics_list, form4, event_date, store=store
            )
            for ticker in affected_tickers:
                event_id_base = ev.get('EVENT_ID', f"POL_{event_date.strftime('%Y%m%d')}")
                fundamental_rows.append({
                    'TICKER': ticker,
                    # Include event_date in the ID so different votes in the
                    # same chamber-year get distinct IDs.  Without the date,
                    # e.g. all 2017 House votes on WFC share "vote_H_2017_WFC",
                    # causing a 26% fan-out when §12b merges on EVENT_ID.
                    'EVENT_ID': f"{event_id_base}_{event_date.strftime('%Y%m%d')}_{ticker}",
                    'EVENT_DATE': event_date, 'EVENT_CATEGORY': 'FUNDAMENTAL',
                    'EVENT_TYPE': ev.get('EVENT_TYPE', 'POLITICAL'),
                    'IS_TREATMENT': True, 'LEAN': 'N/A',
                    'PAIR_ID': f"{event_id_base}_{ticker}",
                    'POLICY_AREA': ev.get('POLICY_AREA', 'unknown'),
                    'DESCRIPTION': ev.get('DESCRIPTION', ''),
                    'IS_CLOSE_VOTE': ev.get('IS_CLOSE_VOTE', False),
                    'VOTE_MARGIN': ev.get('VOTE_MARGIN', np.nan),
                })

    all_event_rows = cultural_rows + fundamental_rows
    if not all_event_rows:
        logger.error("No events to process")
        return pd.DataFrame()

    event_list = pd.DataFrame(all_event_rows)
    n_cultural = (event_list['EVENT_CATEGORY'] == 'CULTURAL').sum()
    n_fundamental = (event_list['EVENT_CATEGORY'] == 'FUNDAMENTAL').sum()
    logger.info("  %d total events: %d cultural, %d fundamental",
                len(event_list), n_cultural, n_fundamental)

    # Precompute routine classification (vectorized)
    logger.info("  Building routine insider cache (vectorized)...")
    quarter_cache = _build_routine_cache(form4)
    logger.info("  Cache built: %d (ticker, owner) pairs", len(quarter_cache))

    routine_cache = {}
    unique_events = event_list[['TICKER', 'EVENT_DATE']].drop_duplicates()
    for _, ev_row in unique_events.iterrows():
        ticker = ev_row['TICKER']
        event_date = ev_row['EVENT_DATE']
        ticker_owners = form4.loc[form4['ticker'] == ticker, 'owner_name'].unique()
        for owner in ticker_owners:
            key = (ticker, owner, event_date)
            if key not in routine_cache:
                routine_cache[key] = _is_routine_from_cache(
                    quarter_cache, ticker, owner, event_date
                )

    # Build lookups
    car_lookup = {}
    if car_df is not None and not car_df.empty:
        for _, row in car_df.iterrows():
            key = (row['TICKER'], pd.Timestamp(row['EVENT_DATE']).strftime('%Y%m%d'))
            car_lookup[key] = row.get('CAR', None)

    exposure_lookup = {}
    if political_exposure is not None and not political_exposure.empty:
        for _, row in political_exposure.iterrows():
            key = (row['TICKER'], int(row['YEAR']))
            exposure_lookup[key] = {
                'LOBBYING_TOTAL': row.get('LOBBYING_TOTAL', 0),
                'PAC_TOTAL': row.get('PAC_TOTAL', 0),
                'POLITICAL_CONNECTION_SCORE': row.get('POLITICAL_CONNECTION_SCORE', np.nan),
            }

    # ── Compute metrics per event ────────────────────────────────────
    panel_rows = []
    n_events = len(event_list)
    for i, (_, ev) in enumerate(event_list.iterrows()):
        ticker = ev['TICKER']
        event_date = ev['EVENT_DATE']
        ticker_txns = form4[form4['ticker'] == ticker].copy()
        ticker_txns['_is_routine'] = ticker_txns['owner_name'].apply(
            lambda owner: routine_cache.get((ticker, owner, event_date), None)
        )

        row = {
            'TICKER': ticker, 'EVENT_ID': ev['EVENT_ID'],
            'EVENT_DATE': event_date, 'EVENT_CATEGORY': ev['EVENT_CATEGORY'],
            'EVENT_TYPE': ev['EVENT_TYPE'], 'IS_TREATMENT': ev['IS_TREATMENT'],
            'LEAN': ev.get('LEAN', 'N/A'),
            'PAIR_ID': ev.get('PAIR_ID', ev['EVENT_ID']),
            'POLICY_AREA': ev.get('POLICY_AREA', 'unknown'),
            'IS_CLOSE_VOTE': ev.get('IS_CLOSE_VOTE', False),
            'VOTE_MARGIN': ev.get('VOTE_MARGIN', np.nan),
            'REGULATORY_PERIOD': _assign_regulatory_period(event_date),
        }

        # CAR + exposure lookups
        car_key = (ticker, event_date.strftime('%Y%m%d'))
        row['CAR_POST'] = car_lookup.get(car_key)
        exp = exposure_lookup.get((ticker, event_date.year), {})
        row['LOBBYING_TOTAL'] = exp.get('LOBBYING_TOTAL', 0)
        row['PAC_TOTAL'] = exp.get('PAC_TOTAL', 0)
        row['POLITICAL_CONNECTION_SCORE'] = exp.get('POLITICAL_CONNECTION_SCORE', np.nan)
        row['HIGH_POLITICAL_CONNECTION'] = (
            row['POLITICAL_CONNECTION_SCORE'] >= 0.5
            if not np.isnan(row.get('POLITICAL_CONNECTION_SCORE', np.nan))
            else False
        )

        # Compute each window
        has_data = False
        for window_name, (d_start, d_end) in WINDOWS.items():
            w_start = event_date + pd.Timedelta(days=d_start)
            w_end = event_date + pd.Timedelta(days=d_end)
            mask = ((ticker_txns['transaction_date'] >= w_start) &
                    (ticker_txns['transaction_date'] <= w_end))
            metrics = _compute_window_metrics(ticker_txns[mask], (d_start, d_end))
            for k, v in metrics.items():
                row[f'{window_name}_{k}'] = v
            if metrics['N_TRANSACTIONS'] > 0:
                has_data = True

        bench_daily = _daily_rate(row.get('BENCHMARK_NET_SELLING', 0), 'BENCHMARK')
        pre_daily = _daily_rate(row.get('PRE_FULL_NET_SELLING', 0), 'PRE_FULL')
        row['ABNORMAL_NET_TRADING'] = pre_daily - bench_daily
        row['ABNORMAL_SELLING'] = 1 if (pre_daily > bench_daily and pre_daily > 0) else 0
        row['HAS_SUFFICIENT_DATA'] = 1 if has_data else 0

        # Four-condition proxies (kept for descriptive use)
        row['C1_FIRM_VALUE'] = int(abs(row.get('CAR_POST') or 0) > 0.02)
        pre_n = row.get('PRE_NEAR_N_TRANSACTIONS', 0)
        post_n = row.get('POST_N_TRANSACTIONS', 0)
        if post_n == 0:
            row['C2_INFO_ASYMMETRY'] = int(pre_n > 0)  # max asymmetry if pre but no post
        else:
            row['C2_INFO_ASYMMETRY'] = int(pre_n > post_n * 1.5)
        row['C3_TRADEABLE'] = int(row.get('BENCHMARK_N_TRANSACTIONS', 0) >= 3)
        row['C4_ABNORMAL_RETURN'] = int(abs(row['ABNORMAL_NET_TRADING']) > 0)
        row['CONDITIONS_MET'] = (row['C1_FIRM_VALUE'] + row['C2_INFO_ASYMMETRY'] +
                                  row['C3_TRADEABLE'] + row['C4_ABNORMAL_RETURN'])

        # Opportunistic fraction for this event
        opp_n = row.get('PRE_FULL_N_OPPORTUNISTIC', 0)
        rout_n = row.get('PRE_FULL_N_ROUTINE', 0)
        total_classified = opp_n + rout_n
        row['OPP_FRACTION'] = opp_n / total_classified if total_classified > 0 else np.nan
        row['HIGH_OPP'] = row['OPP_FRACTION'] >= 0.5 if not np.isnan(row['OPP_FRACTION']) else False

        panel_rows.append(row)
        if (i + 1) % 500 == 0:
            logger.info("  Processed %d/%d events", i + 1, n_events)

    panel = pd.DataFrame(panel_rows)

    # Winsorize within-category
    panel = _winsorize_panel(panel, percentile=1)

    # Recompute DV after winsorization
    panel['ABNORMAL_NET_TRADING'] = panel.apply(
        lambda r: _daily_rate(r.get('PRE_FULL_NET_SELLING', 0), 'PRE_FULL')
                  - _daily_rate(r.get('BENCHMARK_NET_SELLING', 0), 'BENCHMARK'),
        axis=1
    )
    panel['ABNORMAL_SELLING'] = (panel['ABNORMAL_NET_TRADING'] > 0).astype(int)

    assert panel['ABNORMAL_NET_TRADING'].abs().sum() > 0, (
        "ABNORMAL_NET_TRADING is identically zero — column lookup failed "
        "(check _compute_window_metrics keys vs build_insider_panel lookups)"
    )

    # Stratify
    panel = _stratify_panel(panel)

    logger.info("  Panel built: %d rows (%d cultural, %d fundamental), %d with data",
                len(panel),
                (panel['EVENT_CATEGORY'] == 'CULTURAL').sum(),
                (panel['EVENT_CATEGORY'] == 'FUNDAMENTAL').sum(),
                panel['HAS_SUFFICIENT_DATA'].sum())
    return panel


def _build_ticker_naics_map(store):
    """Build ticker -> 3-digit NAICS prefix mapping from database tables.

    Combines CULTURE_WAR_COMPANIES and CONTROL_COMPANIES NAICS codes,
    plus manual assignments for tickers without database coverage.
    """
    naics_map = {}

    # From CULTURE_WAR_COMPANIES
    try:
        cwc = store.read_table('CULTURE_WAR_COMPANIES')
        if not cwc.empty and 'NAICS_CODE' in cwc.columns:
            for _, r in cwc[['TICKER', 'NAICS_CODE']].drop_duplicates('TICKER').iterrows():
                code = int(r['NAICS_CODE']) if pd.notna(r['NAICS_CODE']) else 0
                if code != 999999 and code > 0:
                    naics_map[r['TICKER']] = str(code)[:3]
    except Exception:
        pass

    # From CONTROL_COMPANIES
    try:
        cc = store.read_table('CONTROL_COMPANIES')
        if not cc.empty and 'NAICS_CODE' in cc.columns:
            col = 'CONTROL_TICKER' if 'CONTROL_TICKER' in cc.columns else 'TICKER'
            for _, r in cc[[col, 'NAICS_CODE']].drop_duplicates(col).iterrows():
                code = int(r['NAICS_CODE']) if pd.notna(r['NAICS_CODE']) else 0
                ticker = r[col]
                if code != 999999 and code > 0 and ticker not in naics_map:
                    naics_map[ticker] = str(code)[:3]
    except Exception:
        pass

    # Manual assignments for tickers missing from DB tables
    # Based on publicly available SIC/NAICS classifications
    manual = {
        'AXP': '522',   # credit card issuing
        'BBBY': '442',  # home furnishings
        'BBWI': '453',  # specialty retail (Bath & Body Works)
        'CAR': '532',   # car rental (Avis)
        'CBRL': '722',  # restaurants (Cracker Barrel)
        'CPB': '311',   # food manufacturing (Campbell's)
        'DELL': '334',  # computer manufacturing
        'DPZ': '722',   # restaurants (Domino's)
        'ELF': '325',   # cosmetics
        'EPC': '325',   # specialty chemicals
        'HPE': '334',   # computer equipment
        'HPQ': '334',   # computer equipment
        'HSY': '311',   # food/candy (Hershey)
        'HTZ': '532',   # car rental
        'IBM': '541',   # IT services
        'LULU': '315',  # apparel
        'LYFT': '485',  # rideshare/transit
        'MA': '522',    # payment processing
        'MCD': '722',   # restaurants
        'MDLZ': '311',  # food manufacturing (Mondelez)
        'MET': '524',   # insurance
        'PG': '325',    # consumer goods/chemicals (P&G)
        'PTON': '339',  # fitness equipment
        'PZZA': '722',  # restaurants (Papa John's)
        'SONY': '334',  # electronics
        'TSLA': '336',  # motor vehicles
        'TWLO': '517',  # telecom/cloud
        'ULTA': '446',  # health/beauty retail
        'V': '522',     # payment processing
        'W': '454',     # e-commerce (Wayfair)
        'YUM': '722',   # restaurants
    }
    for ticker, naics3 in manual.items():
        if ticker not in naics_map:
            naics_map[ticker] = naics3

    return naics_map


# Module-level cache for the NAICS map
_ticker_naics_map = None


def _match_tickers_to_naics(tickers, naics_list, form4, event_date,
                            max_per_event=15, store=None):
    """Match Form 4 tickers to affected NAICS sectors.

    Uses 3-digit NAICS prefix matching: a ticker is matched to an event
    if the ticker's NAICS prefix appears in the event's AFFECTED_NAICS list.
    Falls back to returning tickers active in the prior year only if no
    NAICS codes are available for the event.
    """
    global _ticker_naics_map
    if _ticker_naics_map is None and store is not None:
        _ticker_naics_map = _build_ticker_naics_map(store)
        logger.info("  Built ticker-NAICS map: %d tickers", len(_ticker_naics_map))

    # Filter to tickers active in the prior year (baseline requirement)
    window_start = event_date - pd.Timedelta(days=365)
    active = set(form4[
        (form4['transaction_date'] >= window_start) &
        (form4['transaction_date'] <= event_date)
    ]['ticker'].unique())

    # If no NAICS list or no NAICS map, cannot match — return empty
    if not naics_list or _ticker_naics_map is None:
        return []

    # Parse event NAICS to 3-digit prefixes
    event_naics_3 = set()
    for code in naics_list:
        code = str(code).strip()
        if len(code) >= 3:
            event_naics_3.add(code[:3])

    if not event_naics_3:
        return []

    # Match: ticker's 3-digit NAICS prefix must appear in the event's NAICS list
    matched = []
    for ticker in active:
        ticker_naics3 = _ticker_naics_map.get(ticker)
        if ticker_naics3 and ticker_naics3 in event_naics_3:
            matched.append(ticker)

    # Cap at max_per_event using deterministic seed
    if len(matched) > max_per_event:
        seed = int(event_date.strftime('%Y%m%d'))
        rng = np.random.RandomState(seed)
        matched = list(rng.choice(matched, max_per_event, replace=False))

    return matched


def _stratify_panel(panel):
    """Add stratification variables and create a matched subsample."""
    panel = panel.copy()
    panel['EVENT_YEAR'] = pd.to_datetime(panel['EVENT_DATE']).dt.year

    bench_total = panel['BENCHMARK_DOLLAR_SOLD'].fillna(0) + panel['BENCHMARK_DOLLAR_BOUGHT'].fillna(0)
    panel['BENCHMARK_TOTAL_VOLUME'] = bench_total
    panel['ACTIVITY_TERCILE'] = 0
    for cat in panel['EVENT_CATEGORY'].unique():
        mask = panel['EVENT_CATEGORY'] == cat
        vals = panel.loc[mask, 'BENCHMARK_TOTAL_VOLUME']
        try:
            panel.loc[mask, 'ACTIVITY_TERCILE'] = pd.qcut(
                vals, q=3, labels=[1, 2, 3], duplicates='drop'
            ).astype(int)
        except (ValueError, TypeError):
            panel.loc[mask, 'ACTIVITY_TERCILE'] = 2

    # Matched subsample: 3:1 fundamental-to-cultural within strata
    panel['MATCHED'] = False
    cultural_mask = panel['EVENT_CATEGORY'] == 'CULTURAL'
    panel.loc[cultural_mask, 'MATCHED'] = True

    cultural_strata = (
        panel.loc[cultural_mask]
        .groupby(['EVENT_YEAR', 'ACTIVITY_TERCILE'])
        .size().to_dict()
    )

    fund_mask = panel['EVENT_CATEGORY'] == 'FUNDAMENTAL'
    rng = np.random.RandomState(42)
    matched_indices = []
    for (year, tercile), n_cultural in cultural_strata.items():
        stratum_mask = (fund_mask &
                        (panel['EVENT_YEAR'] == year) &
                        (panel['ACTIVITY_TERCILE'] == tercile))
        available = panel.loc[stratum_mask].index.tolist()
        n_draw = min(len(available), n_cultural * 3)
        if n_draw > 0:
            chosen = rng.choice(available, size=n_draw, replace=False)
            matched_indices.extend(chosen)
    panel.loc[matched_indices, 'MATCHED'] = True

    n_mf = panel.loc[fund_mask & panel['MATCHED']].shape[0]
    n_mc = panel.loc[cultural_mask & panel['MATCHED']].shape[0]
    logger.info("  Stratified: %d strata, matched subsample: %d cultural + %d fundamental",
                len(cultural_strata), n_mc, n_mf)
    return panel


# ═════════════════════════════════════════════════════════════════════
# §1 MEAN VS DISTRIBUTIONAL: THE CORE DIVERGENCE
# ═════════════════════════════════════════════════════════════════════

def compute_mean_vs_distributional(panel):
    """Run t-test AND Wilcoxon side-by-side across every analytical cut.

    This is the core table that motivates the paper: mean tests (t) are null
    everywhere, but distributional tests (Wilcoxon) are significant in specific
    cells — revealing a narrow, concentrated channel.
    """
    if panel.empty:
        return pd.DataFrame()

    df = panel[panel['HAS_SUFFICIENT_DATA'] == 1].copy()
    dv = 'ABNORMAL_NET_TRADING'
    results = []

    def _add_cut(label, subset_label, vals):
        if len(vals) < 10:
            return
        t_stat, t_pval = stats.ttest_1samp(vals, 0)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            w_stat, w_pval = stats.wilcoxon(vals, alternative='two-sided', zero_method='pratt')
        results.append({
            'CUT': label, 'SUBSET': subset_label,
            'N': len(vals), 'MEAN': vals.mean(), 'MEDIAN': vals.median(),
            'STD': vals.std(), 'PCT_POSITIVE': (vals > 0).mean(),
            'T_STAT': t_stat, 'T_PVALUE': t_pval,
            'WILCOXON_STAT': w_stat, 'WILCOXON_PVALUE': w_pval,
            'COHEN_D': np.mean(vals) / np.std(vals, ddof=1) if np.std(vals, ddof=1) > 0 else 0,
            'DIVERGENCE': 'YES' if (t_pval > 0.10 and w_pval < 0.05) else 'NO',
        })

    # Overall by category
    for cat in ['FUNDAMENTAL', 'CULTURAL']:
        vals = df.loc[df['EVENT_CATEGORY'] == cat, dv].dropna()
        _add_cut('CATEGORY', cat, vals)

    # By event type (fundamental only)
    fund = df[df['EVENT_CATEGORY'] == 'FUNDAMENTAL']
    for etype in fund['EVENT_TYPE'].unique():
        vals = fund.loc[fund['EVENT_TYPE'] == etype, dv].dropna()
        _add_cut('EVENT_TYPE', etype, vals)

    # By policy area
    for area in fund['POLICY_AREA'].unique():
        vals = fund.loc[fund['POLICY_AREA'] == area, dv].dropna()
        _add_cut('POLICY_AREA', area, vals)

    # By political connection
    high = fund[fund['HIGH_POLITICAL_CONNECTION'] == True][dv].dropna()  # noqa: E712
    low = fund[fund['HIGH_POLITICAL_CONNECTION'] == False][dv].dropna()  # noqa: E712
    _add_cut('POLITICAL_CONNECTION', 'HIGH', high)
    _add_cut('POLITICAL_CONNECTION', 'LOW', low)

    # By insider type
    high_opp = fund[fund['HIGH_OPP'] == True][dv].dropna()  # noqa: E712
    low_opp = fund[fund['HIGH_OPP'] == False][dv].dropna()  # noqa: E712
    _add_cut('INSIDER_TYPE', 'HIGH_OPP', high_opp)
    _add_cut('INSIDER_TYPE', 'LOW_OPP', low_opp)

    # By regulatory period
    for period in ['PRE_SOX', 'SOX_ERA', 'DODD_FRANK', 'POST_AMENDMENTS']:
        vals = fund.loc[fund['REGULATORY_PERIOD'] == period, dv].dropna()
        _add_cut('REGULATORY_PERIOD', period, vals)

    # By conditions met
    for n_cond in range(5):
        vals = fund.loc[fund['CONDITIONS_MET'] == n_cond, dv].dropna()
        _add_cut('CONDITIONS_MET', str(n_cond), vals)

    # By vote closeness (congressional only)
    cong = fund[fund['EVENT_TYPE'] == 'CONGRESSIONAL_VOTE']
    close = cong[cong['IS_CLOSE_VOTE'] == True][dv].dropna()  # noqa: E712
    decisive = cong[cong['IS_CLOSE_VOTE'] == False][dv].dropna()  # noqa: E712
    _add_cut('VOTE_CLOSENESS', 'CLOSE', close)
    _add_cut('VOTE_CLOSENESS', 'DECISIVE', decisive)

    # Cultural subsets
    cult = df[df['EVENT_CATEGORY'] == 'CULTURAL']
    cult_high_opp = cult[cult['HIGH_OPP'] == True][dv].dropna()  # noqa: E712
    cult_low_opp = cult[cult['HIGH_OPP'] == False][dv].dropna()  # noqa: E712
    _add_cut('CULTURAL_INSIDER_TYPE', 'HIGH_OPP', cult_high_opp)
    _add_cut('CULTURAL_INSIDER_TYPE', 'LOW_OPP', cult_low_opp)

    return pd.DataFrame(results)


# ═════════════════════════════════════════════════════════════════════
# §2 FAMILY-LEVEL WILCOXON CORRECTION (GATING TEST)
# ═════════════════════════════════════════════════════════════════════

def compute_wilcoxon_family(mean_vs_dist):
    """Apply Holm-Bonferroni and Benjamini-Hochberg to the full Wilcoxon family.

    This is the gating test for the paper. If fewer than 3 cells survive
    at alpha=0.05 corrected, the distributional finding is noise.
    """
    if mean_vs_dist.empty:
        return pd.DataFrame()

    df = mean_vs_dist.copy()
    w_pvals = df['WILCOXON_PVALUE'].values

    # Replace NaN with 1.0 for correction
    w_pvals_clean = np.where(np.isnan(w_pvals), 1.0, w_pvals)

    holm_sig = _holm_bonferroni(w_pvals_clean, alpha=0.05)
    bh_sig = _benjamini_hochberg(w_pvals_clean, alpha=0.05)

    df['HOLM_SIGNIFICANT'] = holm_sig.astype(int)
    df['BH_SIGNIFICANT'] = bh_sig.astype(int)
    df['N_TOTAL_TESTS'] = len(w_pvals_clean)

    # Summary stats
    n_holm = holm_sig.sum()
    n_bh = bh_sig.sum()
    n_nominal = (w_pvals_clean < 0.05).sum()
    n_divergent = (df['DIVERGENCE'] == 'YES').sum()
    logger.info("  Wilcoxon family correction: %d tests, %d nominal p<0.05, "
                "%d Holm-significant, %d BH-significant, %d t/W divergent",
                len(w_pvals_clean), n_nominal, n_holm, n_bh, n_divergent)

    return df


# ═════════════════════════════════════════════════════════════════════
# §3 INSIDER CONCENTRATION (Cziraki-Gider 2021 style)
# ═════════════════════════════════════════════════════════════════════

def compute_insider_concentration(panel, form4, insider_profits=None):
    """Compute Gini, HHI, and top-K% profit shares at the insider level.

    Matches insider-level trading to the event panel to measure whether
    the dollar volume around political events is concentrated in a few hands.
    """
    if panel.empty:
        return pd.DataFrame()

    df = panel[panel['HAS_SUFFICIENT_DATA'] == 1].copy()
    if insider_profits is None:
        insider_profits = _compute_insider_profits(df, form4)
    results = []

    for cat in ['FUNDAMENTAL', 'CULTURAL', 'ALL']:
        if cat == 'ALL':
            sub_profits = insider_profits
        else:
            sub_profits = insider_profits[insider_profits['EVENT_CATEGORY'] == cat]
        if sub_profits.empty:
            continue

        profits = sub_profits['ABNORMAL_NET_SELLING'].values
        abs_profits = np.abs(profits)

        if len(abs_profits) < 5:
            continue

        # Sort descending by absolute profit
        sorted_abs = np.sort(abs_profits)[::-1]
        total_abs = sorted_abs.sum()

        if total_abs == 0:
            continue

        n = len(sorted_abs)

        # Top-K% shares
        for k_pct in [1, 5, 10, 20, 50]:
            k = max(1, int(np.ceil(n * k_pct / 100)))
            share = sorted_abs[:k].sum() / total_abs
            results.append({
                'EVENT_CATEGORY': cat, 'METRIC': f'TOP_{k_pct}PCT_SHARE',
                'VALUE': share, 'K': k, 'N_INSIDERS': n,
                'TOTAL_ABS_NET_SELLING': total_abs,
            })

        # Gini coefficient
        gini = _gini(abs_profits)
        results.append({
            'EVENT_CATEGORY': cat, 'METRIC': 'GINI',
            'VALUE': gini, 'K': np.nan, 'N_INSIDERS': n,
            'TOTAL_ABS_NET_SELLING': total_abs,
        })

        # HHI
        shares = abs_profits / total_abs
        hhi = (shares ** 2).sum()
        results.append({
            'EVENT_CATEGORY': cat, 'METRIC': 'HHI',
            'VALUE': hhi, 'K': np.nan, 'N_INSIDERS': n,
            'TOTAL_ABS_NET_SELLING': total_abs,
        })

        # Summary stats
        results.append({
            'EVENT_CATEGORY': cat, 'METRIC': 'MEAN_ABS_NET_SELLING',
            'VALUE': abs_profits.mean(), 'K': np.nan, 'N_INSIDERS': n,
            'TOTAL_ABS_NET_SELLING': total_abs,
        })
        results.append({
            'EVENT_CATEGORY': cat, 'METRIC': 'MEDIAN_ABS_NET_SELLING',
            'VALUE': np.median(abs_profits), 'K': np.nan, 'N_INSIDERS': n,
            'TOTAL_ABS_NET_SELLING': total_abs,
        })

    return pd.DataFrame(results)


def _compute_insider_profits(panel_sub, form4):
    """Compute per-insider abnormal net selling around events in the panel subset.

    Note: only insiders with at least one pre-event trade are included.
    Insiders who withdrew (had benchmark activity but no pre-event activity)
    are excluded — this is intentional: we measure the trading behavior of
    active pre-event insiders, not the decision to abstain.
    """
    rows = []
    for _, ev in panel_sub.iterrows():
        ticker = ev['TICKER']
        event_date = ev['EVENT_DATE']
        pre_start = event_date + pd.Timedelta(days=WINDOWS['PRE_FULL'][0])
        pre_end = event_date + pd.Timedelta(days=WINDOWS['PRE_FULL'][1])
        bench_start = event_date + pd.Timedelta(days=WINDOWS['BENCHMARK'][0])
        bench_end = event_date + pd.Timedelta(days=WINDOWS['BENCHMARK'][1])

        tkr_txns = form4[form4['ticker'] == ticker]

        pre_txns = tkr_txns[
            (tkr_txns['transaction_date'] >= pre_start) &
            (tkr_txns['transaction_date'] <= pre_end)
        ]
        bench_txns = tkr_txns[
            (tkr_txns['transaction_date'] >= bench_start) &
            (tkr_txns['transaction_date'] <= bench_end)
        ]

        # Use trading-day normalization (252/365) consistent with panel DV
        pre_cal_days = abs(WINDOWS['PRE_FULL'][1] - WINDOWS['PRE_FULL'][0]) + 1
        bench_cal_days = abs(WINDOWS['BENCHMARK'][1] - WINDOWS['BENCHMARK'][0]) + 1
        n_pre_tdays = max(1, int(round(pre_cal_days * 252 / 365)))
        n_bench_tdays = max(1, int(round(bench_cal_days * 252 / 365)))

        for owner in pre_txns['owner_name'].unique():
            owner_pre = pre_txns[pre_txns['owner_name'] == owner]
            owner_bench = bench_txns[bench_txns['owner_name'] == owner]

            pre_net = _owner_net_dollar(owner_pre)
            bench_net = _owner_net_dollar(owner_bench)

            abnormal = pre_net / n_pre_tdays - bench_net / n_bench_tdays
            rows.append({
                'OWNER': owner, 'TICKER': ticker,
                'EVENT_ID': ev['EVENT_ID'],
                'EVENT_DATE': event_date,
                'EVENT_CATEGORY': ev['EVENT_CATEGORY'],
                'EVENT_TYPE': ev['EVENT_TYPE'],
                'POLICY_AREA': ev['POLICY_AREA'],
                'REGULATORY_PERIOD': ev['REGULATORY_PERIOD'],
                'HIGH_POLITICAL_CONNECTION': ev['HIGH_POLITICAL_CONNECTION'],
                'HIGH_OPP': ev.get('HIGH_OPP', False),
                'ABNORMAL_NET_SELLING': abnormal,
                'PRE_NET_SELLING': pre_net,
                'BENCH_NET_SELLING': bench_net,
                'N_PRE_TXNS': len(owner_pre),
            })

    return pd.DataFrame(rows) if rows else pd.DataFrame()


def _owner_net_dollar(txns):
    """Net dollar: sells - buys for a set of transactions."""
    sells = txns[txns['_trade_type'] == 'sell']['transaction_value'].sum()
    buys = txns[txns['_trade_type'] == 'buy']['transaction_value'].sum()
    return sells - buys


def _gini(values):
    """Compute Gini coefficient for an array of non-negative values."""
    values = np.sort(np.abs(values))
    n = len(values)
    if n == 0 or values.sum() == 0:
        return 0.0
    index = np.arange(1, n + 1)
    return (2 * (index * values).sum()) / (n * values.sum()) - (n + 1) / n


# ═════════════════════════════════════════════════════════════════════
# §4 ACTIVE SUBSET CHARACTERIZATION
# ═════════════════════════════════════════════════════════════════════

def compute_active_subset(panel, form4, insider_profits=None):
    """Identify and characterize the insiders driving the Wilcoxon results.

    'Active' = insiders whose abnormal net trading is in the top or bottom
    decile of the distribution (the tails that shift the Wilcoxon).
    """
    if panel.empty:
        return pd.DataFrame()

    df = panel[panel['HAS_SUFFICIENT_DATA'] == 1].copy()
    if insider_profits is None:
        insider_profits = _compute_insider_profits(df, form4)
    if insider_profits.empty:
        return pd.DataFrame()

    profits = insider_profits['ABNORMAL_NET_SELLING']
    q10 = profits.quantile(0.10)
    q90 = profits.quantile(0.90)

    insider_profits['IS_ACTIVE'] = (
        (profits <= q10) | (profits >= q90)
    )

    results = []

    # Characterize active vs inactive
    for active_label, mask in [('ACTIVE', True), ('INACTIVE', False)]:
        sub = insider_profits[insider_profits['IS_ACTIVE'] == mask]
        if sub.empty:
            continue

        row = {
            'GROUP': active_label,
            'N_INSIDER_EVENTS': len(sub),
            'N_UNIQUE_INSIDERS': sub['OWNER'].nunique(),
            'N_UNIQUE_TICKERS': sub['TICKER'].nunique(),
            'MEAN_ABS_NET_SELLING': sub['ABNORMAL_NET_SELLING'].abs().mean(),
            'MEDIAN_ABS_NET_SELLING': sub['ABNORMAL_NET_SELLING'].abs().median(),
            'TOTAL_ABS_NET_SELLING': sub['ABNORMAL_NET_SELLING'].abs().sum(),
            'PCT_NET_SELLERS': (sub['ABNORMAL_NET_SELLING'] > 0).mean(),
        }

        # Event type distribution
        for etype in ['CONGRESSIONAL_VOTE', 'EXECUTIVE_ORDER', 'COURT_DECISION', 'CULTURE_WAR']:
            n_type = (sub['EVENT_TYPE'] == etype).sum()
            row[f'PCT_{etype}'] = n_type / len(sub) if len(sub) > 0 else 0

        # High political connection rate
        row['PCT_HIGH_CONN'] = sub['HIGH_POLITICAL_CONNECTION'].mean()

        # High opportunistic rate
        row['PCT_HIGH_OPP'] = sub['HIGH_OPP'].mean()

        # Category distribution
        row['PCT_FUNDAMENTAL'] = (sub['EVENT_CATEGORY'] == 'FUNDAMENTAL').mean()
        row['PCT_CULTURAL'] = (sub['EVENT_CATEGORY'] == 'CULTURAL').mean()

        results.append(row)

    # Repeat trader analysis within active
    active = insider_profits[insider_profits['IS_ACTIVE']]
    if not active.empty:
        owner_counts = active.groupby('OWNER').size()
        results.append({
            'GROUP': 'ACTIVE_REPEAT_STATS',
            'N_INSIDER_EVENTS': len(active),
            'N_UNIQUE_INSIDERS': active['OWNER'].nunique(),
            'N_UNIQUE_TICKERS': active['TICKER'].nunique(),
            'MEAN_ABS_NET_SELLING': owner_counts.mean(),   # mean events per active insider
            'MEDIAN_ABS_NET_SELLING': owner_counts.median(),
            'TOTAL_ABS_NET_SELLING': (owner_counts > 1).sum(),  # N repeat traders
            'PCT_NET_SELLERS': (owner_counts > 1).mean(),   # pct repeat
            'PCT_HIGH_CONN': np.nan, 'PCT_HIGH_OPP': np.nan,
            'PCT_FUNDAMENTAL': np.nan, 'PCT_CULTURAL': np.nan,
        })

    return pd.DataFrame(results)


# ═════════════════════════════════════════════════════════════════════
# §5 QUANTILE REGRESSION
# ═════════════════════════════════════════════════════════════════════

def compute_quantile_regression(panel):
    """Median and quantile regressions: IS_FUNDAMENTAL on ABNORMAL_NET_TRADING.

    If the channel operates through the tails, quantile regressions at
    tau=0.10 and tau=0.90 should show significant coefficients even when
    the median (tau=0.50) and mean (OLS) do not.
    """
    if panel.empty:
        return pd.DataFrame()

    df = panel[panel['HAS_SUFFICIENT_DATA'] == 1].copy()
    dv = 'ABNORMAL_NET_TRADING'
    results = []

    # Within fundamental only: test whether observable characteristics predict
    # abnormal trading at different quantiles
    fund = df[df['EVENT_CATEGORY'] == 'FUNDAMENTAL'].copy()
    fund['HIGH_CONN_INT'] = fund['HIGH_POLITICAL_CONNECTION'].astype(int)
    fund['HIGH_OPP_INT'] = fund['HIGH_OPP'].astype(int)
    fund['IS_COURT'] = (fund['EVENT_TYPE'] == 'COURT_DECISION').astype(int)

    X_cols = ['HIGH_CONN_INT', 'HIGH_OPP_INT', 'IS_COURT']
    valid = fund[[dv] + X_cols].dropna()
    if len(valid) < 30:
        return pd.DataFrame()

    X = sm.add_constant(valid[X_cols].astype(float))
    y = valid[dv].astype(float)

    # Use cluster-bootstrap SEs by resampling tickers (not observations)
    # to account for within-ticker correlation
    unique_tickers = fund.loc[valid.index, 'TICKER'].unique()
    n_boot = 200

    for tau in [0.10, 0.25, 0.50, 0.75, 0.90]:
        try:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                model = sm.QuantReg(y, X).fit(q=tau, max_iter=5000)
            # Cluster-bootstrap: resample tickers, refit, collect coefficients
            rng_qr = np.random.RandomState(42)
            boot_coefs = {var: [] for var in X_cols}
            for _ in range(n_boot):
                boot_tickers = rng_qr.choice(unique_tickers, len(unique_tickers), replace=True)
                boot_idx = []
                for t in boot_tickers:
                    t_idx = valid.index[fund.loc[valid.index, 'TICKER'] == t].tolist()
                    boot_idx.extend(t_idx)
                if len(boot_idx) < 30:
                    continue
                X_b = sm.add_constant(valid.loc[boot_idx, X_cols].astype(float))
                y_b = valid.loc[boot_idx, dv].astype(float)
                try:
                    with warnings.catch_warnings():
                        warnings.simplefilter("ignore")
                        m_b = sm.QuantReg(y_b, X_b).fit(q=tau, max_iter=2000)
                    for var in X_cols:
                        boot_coefs[var].append(m_b.params.get(var, np.nan))
                except Exception:
                    continue

            for var in X_cols:
                coef = model.params.get(var, np.nan)
                bc = np.array(boot_coefs[var])
                bc = bc[~np.isnan(bc)]
                if len(bc) >= 50:
                    boot_se = np.std(bc, ddof=1)
                    boot_t = coef / boot_se if boot_se > 0 else np.nan
                    boot_p = 2 * (1 - stats.t.cdf(abs(boot_t), df=len(bc) - 1)) if not np.isnan(boot_t) else np.nan
                else:
                    boot_se = model.bse.get(var, np.nan)
                    boot_t = model.tvalues.get(var, np.nan)
                    boot_p = model.pvalues.get(var, np.nan)
                results.append({
                    'QUANTILE': tau, 'VARIABLE': var,
                    'COEFFICIENT': coef,
                    'STD_ERROR': boot_se,
                    'T_STAT': boot_t,
                    'P_VALUE': boot_p,
                    'N_OBS': int(model.nobs),
                })
        except Exception as e:
            logger.warning("QuantReg at tau=%.2f failed: %s", tau, e)

    # Also OLS for comparison
    try:
        model = sm.OLS(y, X).fit(
            cov_type='cluster',
            cov_kwds={'groups': fund.loc[valid.index, 'TICKER']},
        )
        for var in X_cols:
            results.append({
                'QUANTILE': 0.0,  # sentinel for OLS
                'VARIABLE': var,
                'COEFFICIENT': model.params.get(var, np.nan),
                'STD_ERROR': model.bse.get(var, np.nan),
                'T_STAT': model.tvalues.get(var, np.nan),
                'P_VALUE': model.pvalues.get(var, np.nan),
                'N_OBS': int(model.nobs),
            })
    except Exception as e:
        logger.warning("OLS comparison failed: %s", e)

    return pd.DataFrame(results)


# ═════════════════════════════════════════════════════════════════════
# §6 TRIMMED ROBUSTNESS
# ═════════════════════════════════════════════════════════════════════

def compute_trimmed_robustness(panel):
    """Re-run the core tests after trimming the top and bottom 1% of trades.

    If the Wilcoxon result is driven by a handful of outlier observations,
    trimming kills the finding. If it survives, the distributional shift
    is real and not an artifact of extreme values.
    """
    if panel.empty:
        return pd.DataFrame()

    df = panel[panel['HAS_SUFFICIENT_DATA'] == 1].copy()
    dv = 'ABNORMAL_NET_TRADING'

    results = []

    for trim_pct in [1, 5]:
        vals_all = df[dv].dropna()
        lo = vals_all.quantile(trim_pct / 100)
        hi = vals_all.quantile(1 - trim_pct / 100)
        trimmed = df[(df[dv] >= lo) & (df[dv] <= hi)].copy()

        # Overall fundamental
        fund_vals = trimmed.loc[trimmed['EVENT_CATEGORY'] == 'FUNDAMENTAL', dv].dropna()
        if len(fund_vals) >= 10:
            t_stat, t_pval = stats.ttest_1samp(fund_vals, 0)
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                w_stat, w_pval = stats.wilcoxon(fund_vals, alternative='two-sided', zero_method='pratt')
            results.append({
                'TRIM_PCT': trim_pct, 'CUT': 'FUNDAMENTAL_ALL',
                'N': len(fund_vals), 'MEAN': fund_vals.mean(),
                'MEDIAN': fund_vals.median(),
                'T_STAT': t_stat, 'T_PVALUE': t_pval,
                'WILCOXON_STAT': w_stat, 'WILCOXON_PVALUE': w_pval,
                'PCT_POSITIVE': (fund_vals > 0).mean(),
            })

        # Court decisions
        court_vals = trimmed.loc[
            (trimmed['EVENT_CATEGORY'] == 'FUNDAMENTAL') &
            (trimmed['EVENT_TYPE'] == 'COURT_DECISION'),
            dv
        ].dropna()
        if len(court_vals) >= 10:
            t_stat, t_pval = stats.ttest_1samp(court_vals, 0)
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                w_stat, w_pval = stats.wilcoxon(court_vals, alternative='two-sided', zero_method='pratt')
            results.append({
                'TRIM_PCT': trim_pct, 'CUT': 'COURT_DECISION',
                'N': len(court_vals), 'MEAN': court_vals.mean(),
                'MEDIAN': court_vals.median(),
                'T_STAT': t_stat, 'T_PVALUE': t_pval,
                'WILCOXON_STAT': w_stat, 'WILCOXON_PVALUE': w_pval,
                'PCT_POSITIVE': (court_vals > 0).mean(),
            })

        # High political connection
        high_conn = trimmed.loc[
            (trimmed['EVENT_CATEGORY'] == 'FUNDAMENTAL') &
            (trimmed['HIGH_POLITICAL_CONNECTION'] == True),  # noqa: E712
            dv
        ].dropna()
        if len(high_conn) >= 10:
            t_stat, t_pval = stats.ttest_1samp(high_conn, 0)
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                w_stat, w_pval = stats.wilcoxon(high_conn, alternative='two-sided', zero_method='pratt')
            results.append({
                'TRIM_PCT': trim_pct, 'CUT': 'HIGH_POLITICAL_CONNECTION',
                'N': len(high_conn), 'MEAN': high_conn.mean(),
                'MEDIAN': high_conn.median(),
                'T_STAT': t_stat, 'T_PVALUE': t_pval,
                'WILCOXON_STAT': w_stat, 'WILCOXON_PVALUE': w_pval,
                'PCT_POSITIVE': (high_conn > 0).mean(),
            })

        # High opportunistic
        high_opp = trimmed.loc[
            (trimmed['EVENT_CATEGORY'] == 'FUNDAMENTAL') &
            (trimmed['HIGH_OPP'] == True),  # noqa: E712
            dv
        ].dropna()
        if len(high_opp) >= 10:
            t_stat, t_pval = stats.ttest_1samp(high_opp, 0)
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                w_stat, w_pval = stats.wilcoxon(high_opp, alternative='two-sided', zero_method='pratt')
            results.append({
                'TRIM_PCT': trim_pct, 'CUT': 'HIGH_OPP',
                'N': len(high_opp), 'MEAN': high_opp.mean(),
                'MEDIAN': high_opp.median(),
                'T_STAT': t_stat, 'T_PVALUE': t_pval,
                'WILCOXON_STAT': w_stat, 'WILCOXON_PVALUE': w_pval,
                'PCT_POSITIVE': (high_opp > 0).mean(),
            })

        # 4 conditions met
        cond4 = trimmed.loc[
            (trimmed['EVENT_CATEGORY'] == 'FUNDAMENTAL') &
            (trimmed['CONDITIONS_MET'] == 4),
            dv
        ].dropna()
        if len(cond4) >= 10:
            t_stat, t_pval = stats.ttest_1samp(cond4, 0)
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                w_stat, w_pval = stats.wilcoxon(cond4, alternative='two-sided', zero_method='pratt')
            results.append({
                'TRIM_PCT': trim_pct, 'CUT': 'CONDITIONS_MET_4',
                'N': len(cond4), 'MEAN': cond4.mean(),
                'MEDIAN': cond4.median(),
                'T_STAT': t_stat, 'T_PVALUE': t_pval,
                'WILCOXON_STAT': w_stat, 'WILCOXON_PVALUE': w_pval,
                'PCT_POSITIVE': (cond4 > 0).mean(),
            })

    return pd.DataFrame(results)


# ═════════════════════════════════════════════════════════════════════
# §7 BOOTSTRAPPED WILCOXON
# ═════════════════════════════════════════════════════════════════════

def compute_bootstrap_wilcoxon(panel, n_bootstrap=1000, seed=42):
    """Bootstrap the Wilcoxon test statistic to verify parametric inference.

    For each Wilcoxon-significant cut, resample with replacement and
    recompute the statistic. Report the empirical p-value distribution.
    """
    if panel.empty:
        return pd.DataFrame()

    df = panel[panel['HAS_SUFFICIENT_DATA'] == 1].copy()
    dv = 'ABNORMAL_NET_TRADING'
    rng = np.random.RandomState(seed)
    results = []

    cuts = {
        'FUNDAMENTAL_ALL': df.loc[df['EVENT_CATEGORY'] == 'FUNDAMENTAL', dv].dropna(),
        'COURT_DECISION': df.loc[
            (df['EVENT_CATEGORY'] == 'FUNDAMENTAL') &
            (df['EVENT_TYPE'] == 'COURT_DECISION'),
            dv
        ].dropna(),
        'HIGH_CONN': df.loc[
            (df['EVENT_CATEGORY'] == 'FUNDAMENTAL') &
            (df['HIGH_POLITICAL_CONNECTION'] == True),  # noqa: E712
            dv
        ].dropna(),
        'HIGH_OPP': df.loc[
            (df['EVENT_CATEGORY'] == 'FUNDAMENTAL') &
            (df['HIGH_OPP'] == True),  # noqa: E712
            dv
        ].dropna(),
        'CONDITIONS_4': df.loc[
            (df['EVENT_CATEGORY'] == 'FUNDAMENTAL') &
            (df['CONDITIONS_MET'] == 4),
            dv
        ].dropna(),
    }

    for label, vals in cuts.items():
        if len(vals) < 10:
            continue

        # Observed Wilcoxon
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            obs_stat, obs_pval = stats.wilcoxon(vals, alternative='two-sided', zero_method='pratt')

        # Bootstrap under H0 via sign-flipping: Wilcoxon tests symmetry
        # around zero.  Randomly flipping signs imposes that null without
        # assuming the distribution is symmetric (median-centering preserves
        # skew, so resampling from a skewed centered distribution doesn't
        # actually impose the null).
        abs_vals = np.abs(vals.values)
        boot_stats = []
        boot_pvals = []
        for _ in range(n_bootstrap):
            signs = rng.choice([-1, 1], size=len(abs_vals))
            sample = signs * abs_vals
            try:
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore")
                    bs, bp = stats.wilcoxon(sample, alternative='two-sided', zero_method='pratt')
                boot_stats.append(bs)
                boot_pvals.append(bp)
            except Exception:
                continue

        if not boot_stats:
            continue

        boot_stats = np.array(boot_stats)
        boot_pvals = np.array(boot_pvals)

        results.append({
            'CUT': label, 'N': len(vals),
            'OBSERVED_STAT': obs_stat, 'OBSERVED_PVALUE': obs_pval,
            'BOOT_STAT_MEAN': boot_stats.mean(),
            'BOOT_STAT_STD': boot_stats.std(),
            'BOOT_PVALUE_MEAN': boot_pvals.mean(),
            'BOOT_PVALUE_MEDIAN': np.median(boot_pvals),
            'BOOT_PCT_SIG_005': (boot_pvals < 0.05).mean(),
            'BOOT_PCT_SIG_010': (boot_pvals < 0.10).mean(),
            'N_BOOTSTRAP': n_bootstrap,
        })

    return pd.DataFrame(results)


# ═════════════════════════════════════════════════════════════════════
# §8 INSIDER FIXED EFFECTS
# ═════════════════════════════════════════════════════════════════════

def compute_insider_panel(panel, form4, insider_profits=None):
    """Within-insider variation: do the same insiders trade differently
    around political vs non-political periods?

    Uses insider fixed effects (demeaning) to absorb time-invariant
    insider characteristics and test whether the *within-insider* shift
    around political events is significant.
    """
    if panel.empty:
        return pd.DataFrame()

    df = panel[panel['HAS_SUFFICIENT_DATA'] == 1].copy()
    if insider_profits is None:
        insider_profits = _compute_insider_profits(df, form4)
    if insider_profits.empty or len(insider_profits) < 30:
        return pd.DataFrame()

    results = []

    # Add dummies
    ip = insider_profits.copy()
    ip['IS_FUNDAMENTAL'] = (ip['EVENT_CATEGORY'] == 'FUNDAMENTAL').astype(int)
    ip['IS_COURT'] = (ip['EVENT_TYPE'] == 'COURT_DECISION').astype(int)
    ip['HIGH_CONN_INT'] = ip['HIGH_POLITICAL_CONNECTION'].astype(int)
    ip['HIGH_OPP_INT'] = ip['HIGH_OPP'].astype(int)

    # Only keep insiders with >= 2 events (needed for within-insider variation)
    owner_counts = ip.groupby('OWNER').size()
    repeat_owners = owner_counts[owner_counts >= 2].index
    ip_repeat = ip[ip['OWNER'].isin(repeat_owners)].copy()

    if len(ip_repeat) < 20 or ip_repeat['OWNER'].nunique() < 10:
        logger.warning("Insufficient repeat insiders for FE analysis")
        return pd.DataFrame()

    # Insider FE via linearmodels.PanelOLS (proper dof adjustment)
    X_cols_fe = ['IS_FUNDAMENTAL', 'IS_COURT']
    valid_fe = ip_repeat[['OWNER', 'EVENT_DATE', 'ABNORMAL_NET_SELLING'] + X_cols_fe].dropna()
    if len(valid_fe) < 20:
        return pd.DataFrame()

    valid_fe = valid_fe.copy()

    try:
        from linearmodels.panel import PanelOLS
        # Use EVENT_DATE binned to month (as integer YYYYMM) for the time
        # dimension so entity effects absorb time-invariant insider heterogeneity
        event_dt = pd.to_datetime(valid_fe['EVENT_DATE'])
        valid_fe['_time'] = event_dt.dt.year * 100 + event_dt.dt.month
        valid_fe = valid_fe.drop(columns=['EVENT_DATE'])
        # Average duplicate (OWNER, month) pairs — PanelOLS needs unique index
        if valid_fe.duplicated(subset=['OWNER', '_time']).any():
            valid_fe = valid_fe.groupby(['OWNER', '_time'], as_index=False).mean()
        valid_fe = valid_fe.set_index(['OWNER', '_time'])

        y = valid_fe['ABNORMAL_NET_SELLING']
        X_fe = valid_fe[X_cols_fe].astype(float)
        model = PanelOLS(y, X_fe, entity_effects=True).fit(
            cov_type='clustered', cluster_entity=True
        )
        for var in X_cols_fe:
            results.append({
                'SPECIFICATION': 'INSIDER_FE',
                'VARIABLE': var,
                'COEFFICIENT': model.params.get(var, np.nan),
                'STD_ERROR': model.std_errors.get(var, np.nan),
                'T_STAT': model.tstats.get(var, np.nan),
                'P_VALUE': model.pvalues.get(var, np.nan),
                'N_OBS': int(model.nobs),
                'N_INSIDERS': ip_repeat['OWNER'].nunique(),
                'R_SQUARED': model.rsquared_within,
            })
    except Exception as e:
        logger.warning("PanelOLS insider FE failed: %s", e)
        # Fallback to demeaned OLS if linearmodels fails
        for col in ['ABNORMAL_NET_SELLING'] + X_cols_fe:
            owner_means = ip_repeat.groupby('OWNER')[col].transform('mean')
            ip_repeat[f'{col}_DM'] = ip_repeat[col] - owner_means
        valid_dm = ip_repeat[['ABNORMAL_NET_SELLING_DM'] + [f'{c}_DM' for c in X_cols_fe]].dropna()
        if len(valid_dm) >= 20:
            X = sm.add_constant(valid_dm[[f'{c}_DM' for c in X_cols_fe]].astype(float))
            y_dm = valid_dm['ABNORMAL_NET_SELLING_DM'].astype(float)
            model_dm = sm.OLS(y_dm, X).fit(
                cov_type='cluster',
                cov_kwds={'groups': ip_repeat.loc[valid_dm.index, 'OWNER']},
            )
            for var in X_cols_fe:
                var_dm = f'{var}_DM'
                results.append({
                    'SPECIFICATION': 'INSIDER_FE_DEMEANED',
                    'VARIABLE': var,
                    'COEFFICIENT': model_dm.params.get(var_dm, np.nan),
                    'STD_ERROR': model_dm.bse.get(var_dm, np.nan),
                    'T_STAT': model_dm.tvalues.get(var_dm, np.nan),
                    'P_VALUE': model_dm.pvalues.get(var_dm, np.nan),
                    'N_OBS': int(model_dm.nobs),
                    'N_INSIDERS': ip_repeat['OWNER'].nunique(),
                    'R_SQUARED': model_dm.rsquared,
                })

    # Pooled OLS for comparison
    try:
        X_cols_pooled = ['IS_FUNDAMENTAL', 'IS_COURT', 'HIGH_CONN_INT', 'HIGH_OPP_INT']
        valid_p = ip[[col for col in ['ABNORMAL_NET_SELLING'] + X_cols_pooled]].dropna()
        X = sm.add_constant(valid_p[X_cols_pooled].astype(float))
        y = valid_p['ABNORMAL_NET_SELLING'].astype(float)
        model = sm.OLS(y, X).fit(
            cov_type='cluster',
            cov_kwds={'groups': ip.loc[valid_p.index, 'OWNER']},
        )
        for var in X_cols_pooled:
            results.append({
                'SPECIFICATION': 'POOLED_OLS',
                'VARIABLE': var,
                'COEFFICIENT': model.params.get(var, np.nan),
                'STD_ERROR': model.bse.get(var, np.nan),
                'T_STAT': model.tvalues.get(var, np.nan),
                'P_VALUE': model.pvalues.get(var, np.nan),
                'N_OBS': int(model.nobs),
                'N_INSIDERS': ip['OWNER'].nunique(),
                'R_SQUARED': model.rsquared,
            })
    except Exception as e:
        logger.warning("Pooled OLS failed: %s", e)

    return pd.DataFrame(results)


# ═════════════════════════════════════════════════════════════════════
# §9 CONCENTRATION BY DIMENSION
# ═════════════════════════════════════════════════════════════════════

def compute_concentration_cuts(panel, form4, insider_profits=None):
    """Concentration metrics broken out by event type, policy area, connection."""
    if panel.empty:
        return pd.DataFrame()

    df = panel[
        (panel['HAS_SUFFICIENT_DATA'] == 1) &
        (panel['EVENT_CATEGORY'] == 'FUNDAMENTAL')
    ].copy()

    # Compute or filter insider profits to fundamental events only
    if insider_profits is not None:
        fund_event_ids = set(df['EVENT_ID'])
        fund_profits = insider_profits[insider_profits['EVENT_ID'].isin(fund_event_ids)]
    else:
        fund_profits = _compute_insider_profits(df, form4)

    results = []

    # Build cuts keyed by column filters on the insider profits (not panel)
    cut_filters = {}
    for etype in df['EVENT_TYPE'].unique():
        sub_ids = set(df.loc[df['EVENT_TYPE'] == etype, 'EVENT_ID'])
        if len(sub_ids) >= 20:
            cut_filters[f'EVENT_TYPE:{etype}'] = fund_profits[fund_profits['EVENT_ID'].isin(sub_ids)]

    cut_filters['CONN:HIGH'] = fund_profits[fund_profits['HIGH_POLITICAL_CONNECTION'] == True]  # noqa: E712
    cut_filters['CONN:LOW'] = fund_profits[fund_profits['HIGH_POLITICAL_CONNECTION'] == False]  # noqa: E712
    cut_filters['OPP:HIGH'] = fund_profits[fund_profits['HIGH_OPP'] == True]  # noqa: E712
    cut_filters['OPP:LOW'] = fund_profits[fund_profits['HIGH_OPP'] == False]  # noqa: E712

    for area in ['tax', 'defense', 'healthcare', 'finance', 'energy', 'environment']:
        sub_ids = set(df.loc[df['POLICY_AREA'] == area, 'EVENT_ID'])
        if len(sub_ids) >= 20:
            cut_filters[f'POLICY:{area}'] = fund_profits[fund_profits['EVENT_ID'].isin(sub_ids)]

    for label, sub_profits in cut_filters.items():
        if sub_profits.empty or len(sub_profits) < 5:
            continue

        abs_profits = np.abs(sub_profits['ABNORMAL_NET_SELLING'].values)
        total = abs_profits.sum()
        if total == 0:
            continue

        n = len(abs_profits)
        sorted_abs = np.sort(abs_profits)[::-1]
        top10_k = max(1, int(np.ceil(n * 0.10)))
        top10_share = sorted_abs[:top10_k].sum() / total

        gini = _gini(abs_profits)

        results.append({
            'CUT': label, 'N_INSIDER_EVENTS': n,
            'N_UNIQUE_INSIDERS': sub_profits['OWNER'].nunique(),
            'TOTAL_ABS_NET_SELLING': total,
            'TOP_10PCT_SHARE': top10_share,
            'GINI': gini,
            'MEAN_ABS_NET_SELLING': abs_profits.mean(),
            'MEDIAN_ABS_NET_SELLING': np.median(abs_profits),
        })

    return pd.DataFrame(results)


# ═════════════════════════════════════════════════════════════════════
# §10 REPEAT TRADERS
# ═════════════════════════════════════════════════════════════════════

def compute_repeat_traders(panel, form4, insider_profits=None):
    """Analyze whether repeat traders (same insider across multiple events)
    drive the distributional results disproportionately.
    """
    if panel.empty:
        return pd.DataFrame()

    df = panel[panel['HAS_SUFFICIENT_DATA'] == 1].copy()
    if insider_profits is None:
        insider_profits = _compute_insider_profits(df, form4)
    if insider_profits.empty:
        return pd.DataFrame()

    owner_counts = insider_profits.groupby('OWNER').size().reset_index(name='N_EVENTS')
    owner_stats = insider_profits.groupby('OWNER').agg(
        TOTAL_ABS_NET_SELLING=('ABNORMAL_NET_SELLING', lambda x: x.abs().sum()),
        MEAN_NET_SELLING=('ABNORMAL_NET_SELLING', 'mean'),
        N_TICKERS=('TICKER', 'nunique'),
        PCT_NET_SELLERS=('ABNORMAL_NET_SELLING', lambda x: (x > 0).mean()),
    ).reset_index()
    owner_stats = owner_stats.merge(owner_counts, on='OWNER')

    results = []

    # Overall stats
    results.append({
        'GROUP': 'ALL_INSIDERS',
        'N_INSIDERS': len(owner_stats),
        'MEAN_EVENTS': owner_stats['N_EVENTS'].mean(),
        'MEDIAN_EVENTS': owner_stats['N_EVENTS'].median(),
        'MAX_EVENTS': owner_stats['N_EVENTS'].max(),
        'N_REPEAT': (owner_stats['N_EVENTS'] > 1).sum(),
        'PCT_REPEAT': (owner_stats['N_EVENTS'] > 1).mean(),
        'TOTAL_ABS_NET_SELLING': owner_stats['TOTAL_ABS_NET_SELLING'].sum(),
        'REPEAT_SHARE_OF_NET_SELLING': np.nan,
    })

    # Repeat vs one-time
    repeat = owner_stats[owner_stats['N_EVENTS'] > 1]
    onetime = owner_stats[owner_stats['N_EVENTS'] == 1]
    total_profit = owner_stats['TOTAL_ABS_NET_SELLING'].sum()

    if total_profit > 0 and not repeat.empty:
        results.append({
            'GROUP': 'REPEAT_TRADERS',
            'N_INSIDERS': len(repeat),
            'MEAN_EVENTS': repeat['N_EVENTS'].mean(),
            'MEDIAN_EVENTS': repeat['N_EVENTS'].median(),
            'MAX_EVENTS': repeat['N_EVENTS'].max(),
            'N_REPEAT': len(repeat),
            'PCT_REPEAT': 1.0,
            'TOTAL_ABS_NET_SELLING': repeat['TOTAL_ABS_NET_SELLING'].sum(),
            'REPEAT_SHARE_OF_NET_SELLING': repeat['TOTAL_ABS_NET_SELLING'].sum() / total_profit,
        })

    if not onetime.empty:
        results.append({
            'GROUP': 'ONE_TIME_TRADERS',
            'N_INSIDERS': len(onetime),
            'MEAN_EVENTS': 1.0,
            'MEDIAN_EVENTS': 1.0,
            'MAX_EVENTS': 1,
            'N_REPEAT': 0,
            'PCT_REPEAT': 0.0,
            'TOTAL_ABS_NET_SELLING': onetime['TOTAL_ABS_NET_SELLING'].sum(),
            'REPEAT_SHARE_OF_NET_SELLING': onetime['TOTAL_ABS_NET_SELLING'].sum() / total_profit if total_profit > 0 else 0,
        })

    # Wilcoxon on repeat vs one-time profits
    repeat_ids = repeat['OWNER'].values if not repeat.empty else []
    onetime_ids = onetime['OWNER'].values if not onetime.empty else []
    repeat_profits = insider_profits[insider_profits['OWNER'].isin(repeat_ids)]['ABNORMAL_NET_SELLING']
    onetime_profits = insider_profits[insider_profits['OWNER'].isin(onetime_ids)]['ABNORMAL_NET_SELLING']

    if len(repeat_profits) >= 10 and len(onetime_profits) >= 10:
        u_stat, u_pval = stats.mannwhitneyu(repeat_profits, onetime_profits, alternative='two-sided')
        t_stat, t_pval = stats.ttest_ind(repeat_profits, onetime_profits, equal_var=False)
        results.append({
            'GROUP': 'REPEAT_VS_ONETIME_TEST',
            'N_INSIDERS': len(repeat_profits) + len(onetime_profits),
            'U_STAT': u_stat,
            'U_PVALUE': u_pval,
            'T_STAT': t_stat,
            'T_PVALUE': t_pval,
            'REPEAT_MEAN_NET_SELLING': repeat_profits.mean(),
            'ONETIME_MEAN_NET_SELLING': onetime_profits.mean(),
            'DIFF_MEAN_NET_SELLING': repeat_profits.mean() - onetime_profits.mean(),
        })

    return pd.DataFrame(results)


# ═════════════════════════════════════════════════════════════════════
# §11 CRSP PROFIT ANALYSIS
# ═════════════════════════════════════════════════════════════════════

def compute_crsp_profits(panel, form4, store, insider_profits=None):
    """Compute forward-looking abnormal returns for insider trades.

    For each insider trade in the pre-event window [-180, -1], compute the
    FF5-adjusted abnormal return over [+1, +30] and [+1, +60] trading days
    after the trade date. A trade is 'directionally profitable' if:
      - Buy and post-trade CAR > 0
      - Sell and post-trade CAR < 0

    Then compare active-subset trades (top/bottom decile of abnormal net
    trading) vs inactive trades across event type, connection, regulatory
    period, and insider type.

    This is the test that determines economic significance: do active-subset
    trades actually earn abnormal returns, or is the distributional shift
    economically empty?
    """
    if panel.empty:
        return pd.DataFrame(), pd.DataFrame()

    df = panel[panel['HAS_SUFFICIENT_DATA'] == 1].copy()
    logger.info("  CRSP profit analysis: %d panel events with data", len(df))

    # Build insider-level profits to identify active subset
    if insider_profits is None:
        insider_profits = _compute_insider_profits(df, form4)
    if insider_profits.empty:
        logger.warning("  No insider profits to compute CRSP returns for")
        return pd.DataFrame(), pd.DataFrame()

    # Identify active insiders (top/bottom decile of abnormal net trading)
    profits = insider_profits['ABNORMAL_NET_SELLING']
    q10 = profits.quantile(0.10)
    q90 = profits.quantile(0.90)
    active_owners = insider_profits[
        (profits <= q10) | (profits >= q90)
    ][['OWNER', 'TICKER', 'EVENT_ID']].copy()
    active_keys = set(
        zip(active_owners['OWNER'], active_owners['TICKER'], active_owners['EVENT_ID'])
    )
    logger.info("  Active subset: %d insider-event pairs (q10=%.0f, q90=%.0f)",
                len(active_keys), q10, q90)

    # Build insider-level routine cache (10b5-1 proxy per Cohen-Malloy-Pomorski)
    quarter_cache = _build_routine_cache(form4)

    # Build per-insider prior-buy index for Section 16(b) short-swing flag:
    # (owner, ticker) → sorted list of buy dates
    _buy_dates_by_insider = {}
    buys_only = form4[form4['_trade_type'] == 'buy']
    for (owner, ticker), grp in buys_only.groupby(['owner_name', 'ticker']):
        _buy_dates_by_insider[(owner, ticker)] = sorted(grp['transaction_date'].dropna())

    # Collect all individual trades in the pre-event window
    trade_rows = []
    done = 0
    n_events = len(df)
    for _, ev in df.iterrows():
        ticker = ev['TICKER']
        event_date = ev['EVENT_DATE']
        event_id = ev['EVENT_ID']
        pre_start = event_date + pd.Timedelta(days=WINDOWS['PRE_FULL'][0])
        pre_end = event_date + pd.Timedelta(days=WINDOWS['PRE_FULL'][1])

        tkr_txns = form4[
            (form4['ticker'] == ticker) &
            (form4['transaction_date'] >= pre_start) &
            (form4['transaction_date'] <= pre_end)
        ]

        for _, txn in tkr_txns.iterrows():
            trade_type = txn.get('_trade_type', 'other')
            if trade_type not in ('buy', 'sell'):
                continue

            owner = txn['owner_name']
            is_active = (owner, ticker, event_id) in active_keys
            trade_value = txn.get('transaction_value', 0)
            if pd.isna(trade_value) or trade_value == 0:
                continue

            # Insider-level routine classification (10b5-1 proxy)
            is_routine = _is_routine_from_cache(
                quarter_cache, ticker, owner, event_date
            )

            # Section 16(b) short-swing flag: did this insider buy the
            # same stock within the prior 6 months?  Sellers accepting
            # disgorgement risk are more informationally motivated.
            has_prior_buy_6m = False
            if trade_type == 'sell':
                trade_date = txn['transaction_date']
                buy_dates = _buy_dates_by_insider.get((owner, ticker), [])
                cutoff = trade_date - pd.DateOffset(months=6)
                idx = bisect.bisect_left(buy_dates, cutoff)
                has_prior_buy_6m = any(
                    cutoff <= d < trade_date for d in buy_dates[idx:]
                )

            trade_rows.append({
                'OWNER': owner,
                'TICKER': ticker,
                'EVENT_ID': event_id,
                'EVENT_DATE': event_date,
                'TRADE_DATE': txn['transaction_date'],
                'TRADE_TYPE': trade_type,
                'TRANSACTION_CODE': txn.get('transaction_code', ''),
                'TRADE_VALUE': trade_value,
                'SHARES': txn.get('shares', 0),
                'PRICE': txn.get('price_per_share', np.nan),
                'IS_ACTIVE': is_active,
                'EVENT_CATEGORY': ev['EVENT_CATEGORY'],
                'EVENT_TYPE': ev['EVENT_TYPE'],
                'POLICY_AREA': ev.get('POLICY_AREA', 'unknown'),
                'REGULATORY_PERIOD': ev['REGULATORY_PERIOD'],
                'HIGH_POLITICAL_CONNECTION': ev['HIGH_POLITICAL_CONNECTION'],
                'HIGH_OPP': ev.get('HIGH_OPP', False),
                'IS_ROUTINE': is_routine,
                'HAS_PRIOR_BUY_6M': has_prior_buy_6m,
                'IS_10B5_1_PLAN': txn.get('is_10b5_1_plan', None),
            })

        done += 1
        if done % 500 == 0:
            logger.info("  Collected trades: %d/%d events (%d trades so far)",
                        done, n_events, len(trade_rows))

    if not trade_rows:
        logger.warning("  No trades found in pre-event windows")
        return pd.DataFrame(), pd.DataFrame()

    trades = pd.DataFrame(trade_rows)
    logger.info("  Total pre-event trades (pre-dedup): %d", len(trades))

    # Deduplicate: same (OWNER, TICKER, TRADE_DATE, TRADE_TYPE) across events.
    # When a trade falls in multiple event windows, keep the closest event
    # (smallest |EVENT_DATE - TRADE_DATE|) so EVENT_CAR attribution is
    # meaningful.  IS_ACTIVE is true if active in ANY associated event.
    n_before = len(trades)
    trades['TRADE_ID'] = (trades['OWNER'] + '|' + trades['TICKER'] + '|' +
                          trades['TRADE_DATE'].astype(str) + '|' + trades['TRADE_TYPE'])
    # For IS_ACTIVE: a trade is active if active in ANY event (conservative)
    active_by_trade = trades.groupby('TRADE_ID')['IS_ACTIVE'].any().reset_index()
    active_by_trade.columns = ['TRADE_ID', 'IS_ACTIVE_ANY']
    # Sort by proximity to event so drop_duplicates keeps the closest event
    trades['_days_to_event'] = (trades['EVENT_DATE'] - trades['TRADE_DATE']).dt.days.abs()
    trades = trades.sort_values('_days_to_event')
    trades = trades.drop_duplicates(subset='TRADE_ID', keep='first')
    trades = trades.drop(columns=['IS_ACTIVE', '_days_to_event']).merge(
        active_by_trade, on='TRADE_ID'
    ).rename(columns={'IS_ACTIVE_ANY': 'IS_ACTIVE'})
    logger.info("  After dedup: %d trades (%d removed), %d active, %d inactive",
                len(trades), n_before - len(trades),
                trades['IS_ACTIVE'].sum(), (~trades['IS_ACTIVE']).sum())

    # Compute forward-looking abnormal returns for each trade
    from .essay2_did import _estimate_normal_returns, _FF5_ALL

    # Note: trade-value truncation is handled upstream by _load_form4's $1B
    # cap.  No additional winsorization here — applying 1%/99% clipping to
    # political trades but not control trades creates an asymmetry that
    # distorts the pooled decile thresholds in §12.

    # Compute post-trade CARs for unique (ticker, trade_date) pairs
    unique_trades = trades[['TICKER', 'TRADE_DATE']].drop_duplicates()
    logger.info("  Computing post-trade CARs for %d unique (ticker, trade_date) pairs...",
                len(unique_trades))

    car_cache = {}
    done = 0
    for _, row in unique_trades.iterrows():
        ticker = row['TICKER']
        trade_date = pd.Timestamp(row['TRADE_DATE'])
        key = (ticker, trade_date)

        estimate = _estimate_normal_returns(ticker, store, trade_date)
        if estimate is None:
            car_cache[key] = (np.nan, np.nan)
            done += 1
            continue

        merged = estimate.event_data
        ols_fit = estimate.fit

        cars = {}
        for horizon_label, horizon_end in [('CAR_30', 30), ('CAR_60', 60)]:
            post_obs = merged[
                (merged['TD_OFFSET'] >= 1) &
                (merged['TD_OFFSET'] <= horizon_end)
            ]
            if len(post_obs) < 5:
                cars[horizon_label] = np.nan
                continue

            X_event = sm.add_constant(post_obs[_FF5_ALL])
            expected = ols_fit.predict(X_event)
            ar = post_obs['EXCESS_RETURN'].values - expected.values
            cars[horizon_label] = ar.sum()

        car_cache[key] = (cars.get('CAR_30', np.nan), cars.get('CAR_60', np.nan))
        done += 1
        if done % 500 == 0:
            logger.info("  Post-trade CARs: %d/%d pairs computed",
                        done, len(unique_trades))

    # Merge CARs back to trades
    trades['CAR_30'] = trades.apply(
        lambda r: car_cache.get((r['TICKER'], pd.Timestamp(r['TRADE_DATE'])), (np.nan, np.nan))[0],
        axis=1
    )
    trades['CAR_60'] = trades.apply(
        lambda r: car_cache.get((r['TICKER'], pd.Timestamp(r['TRADE_DATE'])), (np.nan, np.nan))[1],
        axis=1
    )

    # Compute directional profitability
    # Buy is profitable if CAR > 0 (price went up after buying)
    # Sell is profitable if CAR < 0 (price went down after selling)
    # Trades with NaN CARs get NaN profitability (excluded from stats)
    trades['PROFITABLE_30'] = np.where(
        trades['CAR_30'].isna(), np.nan,
        np.where(trades['TRADE_TYPE'] == 'buy',
                 (trades['CAR_30'] > 0).astype(float),
                 (trades['CAR_30'] < 0).astype(float))
    )
    trades['PROFITABLE_60'] = np.where(
        trades['CAR_60'].isna(), np.nan,
        np.where(trades['TRADE_TYPE'] == 'buy',
                 (trades['CAR_60'] > 0).astype(float),
                 (trades['CAR_60'] < 0).astype(float))
    )

    # Signed profit: CAR * trade_value (positive = profitable direction)
    trades['SIGNED_PROFIT_30'] = np.where(
        trades['TRADE_TYPE'] == 'buy',
        trades['CAR_30'] * trades['TRADE_VALUE'],
        -trades['CAR_30'] * trades['TRADE_VALUE']   # sells profit when CAR < 0
    )
    trades['SIGNED_PROFIT_60'] = np.where(
        trades['TRADE_TYPE'] == 'buy',
        trades['CAR_60'] * trades['TRADE_VALUE'],
        -trades['CAR_60'] * trades['TRADE_VALUE']
    )

    n_with_car = trades['CAR_30'].notna().sum()
    logger.info("  Trades with valid CAR_30: %d / %d (%.1f%%)",
                n_with_car, len(trades), 100 * n_with_car / len(trades) if len(trades) > 0 else 0)

    # Build summary table
    summary_rows = []

    def _summarize_group(sub, label):
        """Summarize a group of trades for the CRSP summary table."""
        valid_30 = sub['CAR_30'].dropna()
        valid_60 = sub['CAR_60'].dropna()
        prof_30 = sub['PROFITABLE_30'].dropna()
        prof_60 = sub['PROFITABLE_60'].dropna()
        sp_30 = sub['SIGNED_PROFIT_30'].dropna()
        sp_60 = sub['SIGNED_PROFIT_60'].dropna()

        row = {
            'CUT': label,
            'N_TRADES': len(sub),
            'N_WITH_CAR': len(valid_30),
        }

        # CAR stats
        if len(valid_30) >= 5:
            row['MEAN_CAR_30'] = valid_30.mean()
            row['MEDIAN_CAR_30'] = valid_30.median()
            t_stat, t_pval = stats.ttest_1samp(valid_30, 0)
            row['CAR_30_TSTAT'] = t_stat
            row['CAR_30_PVAL'] = t_pval
        else:
            row['MEAN_CAR_30'] = np.nan
            row['MEDIAN_CAR_30'] = np.nan
            row['CAR_30_TSTAT'] = np.nan
            row['CAR_30_PVAL'] = np.nan

        if len(valid_60) >= 5:
            row['MEAN_CAR_60'] = valid_60.mean()
            row['MEDIAN_CAR_60'] = valid_60.median()
            t_stat, t_pval = stats.ttest_1samp(valid_60, 0)
            row['CAR_60_TSTAT'] = t_stat
            row['CAR_60_PVAL'] = t_pval
        else:
            row['MEAN_CAR_60'] = np.nan
            row['MEDIAN_CAR_60'] = np.nan
            row['CAR_60_TSTAT'] = np.nan
            row['CAR_60_PVAL'] = np.nan

        # Directional profitability
        if len(prof_30) >= 5:
            row['PCT_PROFITABLE_30'] = prof_30.mean()
            # Binomial test against 50%
            n_prof = int(prof_30.sum())
            n_total = len(prof_30)
            row['BINOM_PVAL_30'] = stats.binomtest(n_prof, n_total, 0.5).pvalue
        else:
            row['PCT_PROFITABLE_30'] = np.nan
            row['BINOM_PVAL_30'] = np.nan

        if len(prof_60) >= 5:
            row['PCT_PROFITABLE_60'] = prof_60.mean()
            n_prof = int(prof_60.sum())
            n_total = len(prof_60)
            row['BINOM_PVAL_60'] = stats.binomtest(n_prof, n_total, 0.5).pvalue
        else:
            row['PCT_PROFITABLE_60'] = np.nan
            row['BINOM_PVAL_60'] = np.nan

        # Dollar-weighted profits
        if len(sp_30) >= 5:
            row['MEAN_SIGNED_PROFIT_30'] = sp_30.mean()
            row['TOTAL_SIGNED_PROFIT_30'] = sp_30.sum()
        else:
            row['MEAN_SIGNED_PROFIT_30'] = np.nan
            row['TOTAL_SIGNED_PROFIT_30'] = np.nan

        if len(sp_60) >= 5:
            row['MEAN_SIGNED_PROFIT_60'] = sp_60.mean()
            row['TOTAL_SIGNED_PROFIT_60'] = sp_60.sum()
        else:
            row['MEAN_SIGNED_PROFIT_60'] = np.nan
            row['TOTAL_SIGNED_PROFIT_60'] = np.nan

        return row

    # Active vs inactive (headline comparison)
    for is_active, label in [(True, 'ACTIVE'), (False, 'INACTIVE')]:
        sub = trades[trades['IS_ACTIVE'] == is_active]
        if len(sub) >= 5:
            summary_rows.append(_summarize_group(sub, label))

    # Active only: by trade direction
    active_trades = trades[trades['IS_ACTIVE']]
    for ttype, label in [('buy', 'ACTIVE_BUYS'), ('sell', 'ACTIVE_SELLS')]:
        sub = active_trades[active_trades['TRADE_TYPE'] == ttype]
        if len(sub) >= 5:
            summary_rows.append(_summarize_group(sub, label))

    # Fundamental only
    fund_trades = trades[trades['EVENT_CATEGORY'] == 'FUNDAMENTAL']
    active_fund = fund_trades[fund_trades['IS_ACTIVE']]

    if len(active_fund) >= 5:
        summary_rows.append(_summarize_group(active_fund, 'ACTIVE_FUNDAMENTAL'))

    # By event type (active only)
    for etype in active_fund['EVENT_TYPE'].unique():
        sub = active_fund[active_fund['EVENT_TYPE'] == etype]
        if len(sub) >= 5:
            summary_rows.append(_summarize_group(sub, f'ACTIVE_ETYPE:{etype}'))

    # By political connection (active only)
    for conn, label in [(True, 'ACTIVE_HIGH_CONN'), (False, 'ACTIVE_LOW_CONN')]:
        sub = active_fund[active_fund['HIGH_POLITICAL_CONNECTION'] == conn]
        if len(sub) >= 5:
            summary_rows.append(_summarize_group(sub, label))

    # By insider type (active only)
    for opp, label in [(True, 'ACTIVE_HIGH_OPP'), (False, 'ACTIVE_LOW_OPP')]:
        sub = active_fund[active_fund['HIGH_OPP'] == opp]
        if len(sub) >= 5:
            summary_rows.append(_summarize_group(sub, label))

    # By regulatory period (active only)
    for period in active_fund['REGULATORY_PERIOD'].unique():
        sub = active_fund[active_fund['REGULATORY_PERIOD'] == period]
        if len(sub) >= 5:
            summary_rows.append(_summarize_group(sub, f'ACTIVE_REG:{period}'))

    # Active vs inactive two-sample comparison (abs CARs at both horizons)
    active_cars_30 = trades.loc[trades['IS_ACTIVE'], 'CAR_30'].dropna()
    inactive_cars_30 = trades.loc[~trades['IS_ACTIVE'], 'CAR_30'].dropna()
    active_cars_60 = trades.loc[trades['IS_ACTIVE'], 'CAR_60'].dropna()
    inactive_cars_60 = trades.loc[~trades['IS_ACTIVE'], 'CAR_60'].dropna()
    if len(active_cars_30) >= 10 and len(inactive_cars_30) >= 10:
        t30, p30 = stats.ttest_ind(
            active_cars_30.abs(), inactive_cars_30.abs(), equal_var=False
        )
        t60, p60 = (
            stats.ttest_ind(active_cars_60.abs(), inactive_cars_60.abs(), equal_var=False)
            if len(active_cars_60) >= 10 and len(inactive_cars_60) >= 10
            else (np.nan, np.nan)
        )
        summary_rows.append({
            'CUT': 'ACTIVE_VS_INACTIVE_ABS_CAR',
            'N_TRADES': len(active_cars_30) + len(inactive_cars_30),
            'N_WITH_CAR': len(active_cars_30) + len(inactive_cars_30),
            'MEAN_CAR_30': active_cars_30.abs().mean(),
            'MEDIAN_CAR_30': active_cars_30.abs().median(),
            'CAR_30_TSTAT': t30, 'CAR_30_PVAL': p30,
            'MEAN_CAR_60': active_cars_60.abs().mean() if len(active_cars_60) >= 5 else np.nan,
            'MEDIAN_CAR_60': active_cars_60.abs().median() if len(active_cars_60) >= 5 else np.nan,
            'CAR_60_TSTAT': t60, 'CAR_60_PVAL': p60,
            'PCT_PROFITABLE_30': np.nan, 'BINOM_PVAL_30': np.nan,
            'PCT_PROFITABLE_60': np.nan, 'BINOM_PVAL_60': np.nan,
            'MEAN_SIGNED_PROFIT_30': np.nan, 'TOTAL_SIGNED_PROFIT_30': np.nan,
            'MEAN_SIGNED_PROFIT_60': np.nan, 'TOTAL_SIGNED_PROFIT_60': np.nan,
        })

    crsp_summary = pd.DataFrame(summary_rows) if summary_rows else pd.DataFrame()

    # Trade-level table: save a compact version.
    # EVENT_DATE is included so §12b can merge on (EVENT_ID, TICKER, EVENT_DATE)
    # and avoid fan-out when EVENT_IDs are not yet fully date-stamped.
    save_cols = [
        'OWNER', 'TICKER', 'EVENT_ID', 'EVENT_DATE', 'TRADE_DATE', 'TRADE_TYPE',
        'TRANSACTION_CODE', 'TRADE_VALUE', 'IS_ACTIVE',
        'EVENT_CATEGORY', 'EVENT_TYPE',
        'REGULATORY_PERIOD', 'HIGH_POLITICAL_CONNECTION', 'HIGH_OPP',
        'IS_ROUTINE', 'HAS_PRIOR_BUY_6M', 'IS_10B5_1_PLAN',
        'CAR_30', 'CAR_60', 'PROFITABLE_30', 'PROFITABLE_60',
        'SIGNED_PROFIT_30', 'SIGNED_PROFIT_60',
    ]
    # Only include columns that exist (backwards-compatible)
    save_cols = [c for c in save_cols if c in trades.columns]
    crsp_profits = trades[save_cols].copy()

    logger.info("  CRSP profit analysis complete: %d trades, %d summary rows",
                len(crsp_profits), len(crsp_summary))

    return crsp_profits, crsp_summary


# ═════════════════════════════════════════════════════════════════════
# §12 DIRECTIONAL ACCURACY REVERSAL (Cziraki-Gider 2021 test)
# ═════════════════════════════════════════════════════════════════════

def _build_control_trades(panel, form4, store, political_trades):
    """Build matched non-political control trades with forward CARs.

    For each political-event trade, draws a same-firm trade outside any
    political-event window, matched on ticker only (ticker-quarter matching
    was dropped because the 425-day exclusion window removes most quarters
    with political activity, yielding degenerate samples; year FE in the
    LPM specs absorb time-period variation). Computes FF5-adjusted forward
    CARs for matched controls.
    """
    if political_trades.empty:
        return pd.DataFrame()

    # Exclude the full event footprint: [benchmark_start, post_end] = [-365, +60]
    # This prevents control trades from containing:
    #   - post-event reactive trades (day 0 to +60)
    #   - benchmark-window trades used to calibrate active/inactive status
    #   - pre-event window trades already in the political sample
    ticker_windows = {}
    for _, ev in panel.iterrows():
        ticker = ev['TICKER']
        ed = pd.Timestamp(ev['EVENT_DATE'])
        w_start = ed + pd.Timedelta(days=WINDOWS['BENCHMARK'][0])   # -365
        w_end = ed + pd.Timedelta(days=WINDOWS['POST'][1])          # +60
        ticker_windows.setdefault(ticker, []).append((w_start, w_end))

    # Filter Form 4 to relevant tickers, buys/sells with value
    political_tickers = set(political_trades['TICKER'].unique())
    f4 = form4[
        form4['ticker'].isin(political_tickers) &
        form4['_trade_type'].isin(['buy', 'sell']) &
        form4['transaction_value'].notna() &
        (form4['transaction_value'] > 0)
    ].copy()

    if f4.empty:
        return pd.DataFrame()

    # Mark trades inside any political window (vectorized)
    in_window = pd.Series(False, index=f4.index)
    for ticker, windows in ticker_windows.items():
        tkr_mask = f4['ticker'] == ticker
        if not tkr_mask.any():
            continue
        for ws, we in windows:
            in_window |= (tkr_mask &
                          (f4['transaction_date'] >= ws) &
                          (f4['transaction_date'] <= we))

    # Control pool: trades NOT in any political window
    pool = f4[~in_window].copy()
    pool = pool.rename(columns={
        'ticker': 'TICKER', 'owner_name': 'OWNER',
        'transaction_date': 'TRADE_DATE', '_trade_type': 'TRADE_TYPE',
        'transaction_code': 'TRANSACTION_CODE',
        'transaction_value': 'TRADE_VALUE', 'shares': 'SHARES',
        'price_per_share': 'PRICE',
    })[['TICKER', 'OWNER', 'TRADE_DATE', 'TRADE_TYPE', 'TRANSACTION_CODE',
        'TRADE_VALUE', 'SHARES', 'PRICE']]

    logger.info("  Control pool: %d non-political trades, %d tickers",
                len(pool), pool['TICKER'].nunique())

    # Match by ticker: sample N control trades per firm, proportional
    # to political trade count.  Ticker-quarter matching produced a
    # degenerate sample (N<100) because the 425-day exclusion window
    # [-365, +60] removes most quarters with political activity.
    # Ticker-only matching gives a large control sample; year FE in
    # the LPM specifications absorb time-period variation.
    pol_counts = political_trades.groupby('TICKER').size()
    pool_grouped = {k: v for k, v in pool.groupby('TICKER')}

    rng = np.random.RandomState(42)
    matched_parts = []
    for ticker, n_needed in pol_counts.items():
        cand = pool_grouped.get(ticker)
        if cand is None or cand.empty:
            continue
        n_draw = min(len(cand), n_needed)
        chosen_idx = rng.choice(cand.index, size=n_draw, replace=False)
        matched_parts.append(pool.loc[chosen_idx])

    if not matched_parts:
        logger.warning("  No matched control trades found")
        return pd.DataFrame()

    control = pd.concat(matched_parts, ignore_index=True)

    # Cap to keep CAR computation tractable
    MAX_CONTROL = 40000
    if len(control) > MAX_CONTROL:
        control = control.sample(n=MAX_CONTROL, random_state=rng).reset_index(drop=True)
        logger.info("  Capped control sample to %d trades", MAX_CONTROL)

    logger.info("  Matched %d control trades (target: %d)",
                len(control), len(political_trades))

    # Compute forward-looking CARs for control trades
    from .essay2_did import _estimate_normal_returns, _FF5_ALL

    unique_ct = control[['TICKER', 'TRADE_DATE']].drop_duplicates()
    logger.info("  Computing CARs for %d unique control (ticker, date) pairs...",
                len(unique_ct))

    car_cache = {}
    done = 0
    for _, row in unique_ct.iterrows():
        ticker = row['TICKER']
        trade_date = pd.Timestamp(row['TRADE_DATE'])
        key = (ticker, trade_date)

        estimate = _estimate_normal_returns(ticker, store, trade_date)
        if estimate is None:
            car_cache[key] = (np.nan, np.nan)
        else:
            merged = estimate.event_data
            ols_fit = estimate.fit
            cars = {}
            for hl, he in [('CAR_30', 30), ('CAR_60', 60)]:
                post = merged[
                    (merged['TD_OFFSET'] >= 1) & (merged['TD_OFFSET'] <= he)
                ]
                if len(post) < 5:
                    cars[hl] = np.nan
                    continue
                X_ev = sm.add_constant(post[_FF5_ALL])
                expected = ols_fit.predict(X_ev)
                cars[hl] = (post['EXCESS_RETURN'].values - expected.values).sum()
            car_cache[key] = (cars.get('CAR_30', np.nan), cars.get('CAR_60', np.nan))

        done += 1
        if done % 500 == 0:
            logger.info("  Control CARs: %d/%d", done, len(unique_ct))

    # Merge CARs back
    control['CAR_30'] = control.apply(
        lambda r: car_cache.get(
            (r['TICKER'], pd.Timestamp(r['TRADE_DATE'])), (np.nan, np.nan)
        )[0], axis=1
    )
    control['CAR_60'] = control.apply(
        lambda r: car_cache.get(
            (r['TICKER'], pd.Timestamp(r['TRADE_DATE'])), (np.nan, np.nan)
        )[1], axis=1
    )

    # Directional profitability
    control['PROFITABLE_30'] = np.where(
        control['CAR_30'].isna(), np.nan,
        np.where(control['TRADE_TYPE'] == 'buy',
                 (control['CAR_30'] > 0).astype(float),
                 (control['CAR_30'] < 0).astype(float)))
    control['PROFITABLE_60'] = np.where(
        control['CAR_60'].isna(), np.nan,
        np.where(control['TRADE_TYPE'] == 'buy',
                 (control['CAR_60'] > 0).astype(float),
                 (control['CAR_60'] < 0).astype(float)))

    n_car = control['CAR_30'].notna().sum()
    logger.info("  Control trades with CAR: %d/%d (%.1f%%)",
                n_car, len(control),
                100 * n_car / len(control) if len(control) > 0 else 0)

    return control


def compute_directional_accuracy_reversal(panel, form4, store, crsp_profits=None):
    """Test the Cziraki-Gider (2021) trade-size/accuracy reversal.

    Headline test: does directional accuracy decrease with trade size around
    political events, reversing the unconditional CG regularity?

    Runs:
      1. Trade-size decile × sample accuracy table
      2. LPM: Pr(accurate) ~ log(trade_value) + controls, separately on
         political and control samples, then pooled with interaction
      3. Pre/post 2023 amendments interaction (regulatory architecture test)
      4. Ticker FE specification (absorbs firm-level variation)

    Returns (size_accuracy, size_accuracy_slopes, reversal_regression,
             control_trades, control_attrition).

    Lambert review 2026-07-06:
    - SIZE_ACCURACY for political uses EVENT_PROFITABLE (event-CAR construct,
      same as the headline directional accuracy) rather than PROFITABLE_30
      (30-day forward-CAR, which straddles the event drop for near-event trades).
    - control_attrition documents the ~20% of matched control trades that lose
      valid CARs (compared by size, year, direction).
    """
    empty = pd.DataFrame(), pd.DataFrame(), pd.DataFrame(), pd.DataFrame(), pd.DataFrame()
    if crsp_profits is None or crsp_profits.empty:
        logger.warning("  No CRSP profits for directional accuracy analysis")
        return empty

    political = crsp_profits.copy()
    logger.info("  §12 Directional accuracy reversal: %d political trades",
                len(political))

    # ── Step 1: Build matched control sample ──────────────────────────
    control = _build_control_trades(panel, form4, store, political)

    # ── Step 2: Trade-size decile classification ──────────────────────
    # Pool thresholds across both samples so decile N means the same
    # dollar range in political and control (comparability requirement)
    political['SAMPLE'] = 'POLITICAL'

    if not control.empty:
        control['SAMPLE'] = 'CONTROL'

        pooled_tv = pd.concat([political['TRADE_VALUE'], control['TRADE_VALUE']])
        try:
            _, bins = pd.qcut(pooled_tv, 10, retbins=True, duplicates='drop')
            bins[0] = 0           # include minimum
            bins[-1] = np.inf     # include maximum
            political['SIZE_DECILE'] = pd.cut(
                political['TRADE_VALUE'], bins=bins, labels=False
            ) + 1
            control['SIZE_DECILE'] = pd.cut(
                control['TRADE_VALUE'], bins=bins, labels=False
            ) + 1
        except ValueError:
            political['SIZE_DECILE'] = 1
            control['SIZE_DECILE'] = 1
    else:
        try:
            political['SIZE_DECILE'] = pd.qcut(
                political['TRADE_VALUE'], 10, labels=False, duplicates='drop'
            ) + 1
        except ValueError:
            political['SIZE_DECILE'] = 1

    # ── Step 3: Accuracy by trade-size decile ─────────────────────────
    # Lambert review 2026-07-06: political sample uses EVENT_PROFITABLE
    # (event-CAR construct, identical to the headline test) so both directional
    # accuracy numbers refer to the same quantity.  Control uses PROFITABLE_30
    # (forward-CAR) since control trades have no event-date anchor.
    size_rows = []
    slope_rows = []
    for sample_label, df in [('POLITICAL', political), ('CONTROL', control)]:
        if df.empty:
            continue
        accuracy_col = (
            'EVENT_PROFITABLE'
            if sample_label == 'POLITICAL' and 'EVENT_PROFITABLE' in df.columns
            else 'PROFITABLE_30'
        )
        valid = df[df[accuracy_col].notna()]
        for decile in sorted(valid['SIZE_DECILE'].unique()):
            dec = valid[valid['SIZE_DECILE'] == decile]
            n = len(dec)
            if n < 5:
                continue
            pct = dec[accuracy_col].mean()
            n_acc = int(dec[accuracy_col].sum())
            size_rows.append({
                'SAMPLE': sample_label,
                'SIZE_DECILE': int(decile),
                'N_TRADES': n,
                'MEAN_TRADE_VALUE': dec['TRADE_VALUE'].mean(),
                'MEDIAN_TRADE_VALUE': dec['TRADE_VALUE'].median(),
                'PCT_ACCURATE': pct,
                'N_ACCURATE': n_acc,
                'BINOM_PVAL': stats.binomtest(n_acc, n, 0.5).pvalue,
                'MEAN_CAR_30': dec['CAR_30'].mean() if 'CAR_30' in dec.columns else np.nan,
                'ACCURACY_CONSTRUCT': 'EVENT_CAR' if accuracy_col == 'EVENT_PROFITABLE' else 'TRADE_CAR_30',
            })

        # Slope test: regress decile-level accuracy on log(median_tv)
        # (not on decile rank, which isn't comparable across samples)
        if len(valid['SIZE_DECILE'].unique()) >= 3:
            decile_data = valid.groupby('SIZE_DECILE').agg(
                accuracy=(accuracy_col, 'mean'),
                log_tv=('TRADE_VALUE', lambda x: np.log(x.median())),
            )
            slope, _, r_val, p_val, _ = stats.linregress(
                decile_data['log_tv'].values, decile_data['accuracy'].values
            )
            slope_rows.append({
                'SAMPLE': sample_label,
                'SLOPE': slope,
                'SLOPE_PVAL': p_val,
                'SLOPE_R2': r_val ** 2,
                'N_TRADES': len(valid),
                'N_DECILES': len(decile_data),
                'ACCURACY_CONSTRUCT': 'EVENT_CAR' if accuracy_col == 'EVENT_PROFITABLE' else 'TRADE_CAR_30',
            })

    size_accuracy = pd.DataFrame(size_rows)
    size_accuracy_slopes = pd.DataFrame(slope_rows)

    # ── Control attrition table ────────────────────────────────────────
    # Lambert review 2026-07-06: ~20% of matched control trades drop out when
    # the 30-day forward CAR cannot be computed.  Document size, year, and
    # direction differences between retained and dropped trades so the reviewer
    # can assess whether attrition is size-selective (which would bias the
    # accuracy premium upward on the control side).
    attrition_rows = []
    if not control.empty and 'PROFITABLE_30' in control.columns:
        for grp_label, grp in [
            ('RETAINED', control[control['PROFITABLE_30'].notna()]),
            ('DROPPED',  control[control['PROFITABLE_30'].isna()]),
        ]:
            if grp.empty:
                continue
            td_g = pd.to_datetime(grp['TRADE_DATE'])
            attrition_rows.append({
                'GROUP':               grp_label,
                'N':                   len(grp),
                'N_SELLS':             int((grp['TRADE_TYPE'] == 'sell').sum()),
                'N_BUYS':              int((grp['TRADE_TYPE'] == 'buy').sum()),
                'MEAN_TRADE_VALUE':    grp['TRADE_VALUE'].mean(),
                'MEDIAN_TRADE_VALUE':  grp['TRADE_VALUE'].median(),
                'P90_TRADE_VALUE':     grp['TRADE_VALUE'].quantile(0.90),
                'YEAR_MEAN':           td_g.dt.year.mean(),
                'YEAR_MIN':            int(td_g.dt.year.min()),
                'YEAR_MAX':            int(td_g.dt.year.max()),
            })
    control_attrition = pd.DataFrame(attrition_rows)

    # ── Step 4: LPM regressions ───────────────────────────────────────
    regression_rows = []

    # Ensure NAICS map for industry FE
    global _ticker_naics_map
    if _ticker_naics_map is None and store is not None:
        _ticker_naics_map = _build_ticker_naics_map(store)

    # Prepare combined dataset
    if not control.empty:
        combined = pd.concat([political, control], ignore_index=True)
    else:
        combined = political.copy()

    # LPM dependent variable: EVENT_PROFITABLE for political (event-CAR construct,
    # same as headline), PROFITABLE_30 for control (forward-CAR).
    # This ensures both size-analysis tables and LPM specs use the same accuracy
    # measure as the headline directional test (Lambert review 2026-07-06 §5).
    if 'EVENT_PROFITABLE' in combined.columns:
        combined['_ACCURATE'] = np.where(
            combined['SAMPLE'] == 'POLITICAL',
            combined['EVENT_PROFITABLE'],
            combined['PROFITABLE_30'],
        )
    else:
        combined['_ACCURATE'] = combined['PROFITABLE_30']

    valid = combined[combined['_ACCURATE'].notna()].copy()
    if len(valid) < 30:
        logger.warning("  Too few valid trades for LPM regression")
        return size_accuracy, size_accuracy_slopes, pd.DataFrame(), control

    valid['IS_POLITICAL'] = (valid['SAMPLE'] == 'POLITICAL').astype(int)
    valid['LOG_TV'] = np.log(valid['TRADE_VALUE'])
    valid['IS_BUY'] = (valid['TRADE_TYPE'] == 'buy').astype(int)
    valid['LOG_TV_x_POL'] = valid['LOG_TV'] * valid['IS_POLITICAL']

    td = pd.to_datetime(valid['TRADE_DATE'])
    valid['YEAR'] = td.dt.year
    valid['POST_2023'] = (valid['YEAR'] >= 2023).astype(int)
    valid['LOG_TV_x_POST2023'] = valid['LOG_TV'] * valid['POST_2023']

    # Insider-type controls (available in political trades from CRSP profits)
    for col in ['HIGH_POLITICAL_CONNECTION', 'HIGH_OPP']:
        if col in valid.columns:
            valid[f'{col}_INT'] = valid[col].fillna(False).astype(int)
        else:
            valid[f'{col}_INT'] = 0

    # Year dummies for time FE (drop first to avoid collinearity)
    year_dummies = pd.get_dummies(valid['YEAR'], prefix='Y', drop_first=True,
                                  dtype=float)
    year_dummy_cols = list(year_dummies.columns)
    valid = pd.concat([valid, year_dummies], axis=1)

    def _run_lpm(subset, spec_name, x_cols, results_list, include_year_fe=True):
        """Run LPM with ticker-clustered SEs and year FE, append results."""
        if len(subset) < 30:
            return
        # Drop columns with no variation
        x_use = [c for c in x_cols if c in subset.columns and subset[c].nunique() > 1]
        if not x_use:
            return
        # Add year FE (drop any year columns with no variation in subset)
        if include_year_fe:
            yr_use = [c for c in year_dummy_cols
                      if c in subset.columns and subset[c].nunique() > 1]
            x_all = x_use + yr_use
        else:
            x_all = x_use
        X = sm.add_constant(subset[x_all].astype(float))
        y = subset['_ACCURATE'].astype(float)
        try:
            model = sm.OLS(y, X).fit(
                cov_type='cluster',
                cov_kwds={'groups': subset['TICKER']},
            )
            # Report coefficients for main variables (not year dummies)
            for var in x_use:
                results_list.append({
                    'SPECIFICATION': spec_name,
                    'VARIABLE': var,
                    'COEFFICIENT': model.params.get(var, np.nan),
                    'STD_ERROR': model.bse.get(var, np.nan),
                    'T_STAT': model.tvalues.get(var, np.nan),
                    'P_VALUE': model.pvalues.get(var, np.nan),
                    'N_OBS': int(model.nobs),
                    'R_SQUARED': model.rsquared,
                })
        except Exception as e:
            logger.warning("  LPM '%s' failed: %s", spec_name, e)

    # ── Spec 1: Political-only ────────────────────────────────────────
    pol = valid[valid['IS_POLITICAL'] == 1]
    _run_lpm(pol, 'POLITICAL_ONLY',
             ['LOG_TV', 'IS_BUY', 'HIGH_POLITICAL_CONNECTION_INT', 'HIGH_OPP_INT'],
             regression_rows)

    # ── Spec 1b/1c: Political buys-only and sells-only ───────────────
    # Buy/sell asymmetry: 10b5-1 and blackout periods bind hardest on
    # sells, so the reversal may be sell-driven
    pol_buys = pol[pol['IS_BUY'] == 1]
    pol_sells = pol[pol['IS_BUY'] == 0]
    _run_lpm(pol_buys, 'POLITICAL_BUYS_ONLY',
             ['LOG_TV', 'HIGH_POLITICAL_CONNECTION_INT', 'HIGH_OPP_INT'],
             regression_rows)
    _run_lpm(pol_sells, 'POLITICAL_SELLS_ONLY',
             ['LOG_TV', 'HIGH_POLITICAL_CONNECTION_INT', 'HIGH_OPP_INT'],
             regression_rows)

    # ── Spec 2: Control-only (Cziraki-Gider replication) ─────────────
    ctrl = valid[valid['IS_POLITICAL'] == 0]
    _run_lpm(ctrl, 'CONTROL_ONLY',
             ['LOG_TV', 'IS_BUY'],
             regression_rows)

    # ── Spec 2b/2c: Control buys-only and sells-only ─────────────────
    # Mirror the political buy/sell split for symmetric comparison
    if len(ctrl) >= 30:
        ctrl_buys = ctrl[ctrl['IS_BUY'] == 1]
        ctrl_sells = ctrl[ctrl['IS_BUY'] == 0]
        _run_lpm(ctrl_buys, 'CONTROL_BUYS_ONLY', ['LOG_TV'], regression_rows)
        _run_lpm(ctrl_sells, 'CONTROL_SELLS_ONLY', ['LOG_TV'], regression_rows)

    # ── Spec 3: Pooled with interaction ───────────────────────────────
    if not control.empty:
        _run_lpm(valid, 'POOLED_INTERACTION',
                 ['LOG_TV', 'IS_POLITICAL', 'LOG_TV_x_POL', 'IS_BUY'],
                 regression_rows)

    # ── Spec 3b: Pooled with ticker + year FE ─────────────────────────
    # If LOG_TV_x_POL survives ticker FE absorption, the interaction is
    # real.  If it goes to zero, the pooled result is a between-firm
    # composition artifact.
    if not control.empty and valid['TICKER'].nunique() >= 10:
        fe_x_cols = ['LOG_TV', 'IS_POLITICAL', 'LOG_TV_x_POL', 'IS_BUY']
        yr_fe_cols = [c for c in year_dummy_cols
                      if c in valid.columns and valid[c].nunique() > 1]
        all_fe_x = fe_x_cols + yr_fe_cols
        pooled_fe = valid[['TICKER', 'TRADE_DATE', 'PROFITABLE_30'] + all_fe_x].dropna().copy()
        pooled_fe = pooled_fe.reset_index(drop=True)
        pooled_fe['_row_id'] = range(len(pooled_fe))
        try:
            from linearmodels.panel import PanelOLS as _PanelOLS
            pooled_fe_idx = pooled_fe.set_index(['TICKER', '_row_id'])
            y_fe = pooled_fe_idx['PROFITABLE_30']
            X_fe = pooled_fe_idx[all_fe_x].astype(float)
            model = _PanelOLS(y_fe, X_fe, entity_effects=True).fit(
                cov_type='clustered', cluster_entity=True
            )
            for var in fe_x_cols:
                regression_rows.append({
                    'SPECIFICATION': 'POOLED_TICKER_FE',
                    'VARIABLE': var,
                    'COEFFICIENT': model.params.get(var, np.nan),
                    'STD_ERROR': model.std_errors.get(var, np.nan),
                    'T_STAT': model.tstats.get(var, np.nan),
                    'P_VALUE': model.pvalues.get(var, np.nan),
                    'N_OBS': int(model.nobs),
                    'R_SQUARED': model.rsquared_within,
                })
        except Exception as e:
            logger.warning("  PanelOLS pooled ticker FE failed: %s", e)

    # ── Spec 4: Political with post-2023 interaction ──────────────────
    if (len(pol) >= 30 and pol['POST_2023'].sum() >= 20 and
            (1 - pol['POST_2023']).sum() >= 20):
        _run_lpm(pol, 'POLITICAL_POST2023',
                 ['LOG_TV', 'POST_2023', 'LOG_TV_x_POST2023', 'IS_BUY'],
                 regression_rows)

    # ── Spec 5: Ticker FE via PanelOLS (proper DOF adjustment) ─────────
    # Includes year dummies for consistency with other specs
    if len(pol) >= 50 and pol['TICKER'].nunique() >= 10:
        fe_x_cols = ['LOG_TV', 'IS_BUY']
        # Include year dummies (same as other specs)
        yr_fe_cols = [c for c in year_dummy_cols
                      if c in pol.columns and pol[c].nunique() > 1]
        all_fe_x = fe_x_cols + yr_fe_cols
        pol_fe = pol[['TICKER', 'TRADE_DATE', 'PROFITABLE_30'] + all_fe_x].dropna().copy()
        # Need unique (entity, time) index — use trade-level row ID as time
        pol_fe = pol_fe.reset_index(drop=True)
        pol_fe['_row_id'] = range(len(pol_fe))
        try:
            from linearmodels.panel import PanelOLS as _PanelOLS
            pol_fe_idx = pol_fe.set_index(['TICKER', '_row_id'])
            y_fe = pol_fe_idx['PROFITABLE_30']
            X_fe = pol_fe_idx[all_fe_x].astype(float)
            model = _PanelOLS(y_fe, X_fe, entity_effects=True).fit(
                cov_type='clustered', cluster_entity=True
            )
            # Report main variables only (not year dummies)
            for var in fe_x_cols:
                regression_rows.append({
                    'SPECIFICATION': 'POLITICAL_TICKER_FE',
                    'VARIABLE': var,
                    'COEFFICIENT': model.params.get(var, np.nan),
                    'STD_ERROR': model.std_errors.get(var, np.nan),
                    'T_STAT': model.tstats.get(var, np.nan),
                    'P_VALUE': model.pvalues.get(var, np.nan),
                    'N_OBS': int(model.nobs),
                    'R_SQUARED': model.rsquared_within,
                })
        except Exception as e:
            logger.warning("  PanelOLS ticker FE failed: %s", e)

    # ── Spec 5b: Control ticker FE (symmetric to POLITICAL_TICKER_FE) ─
    # If CONTROL_TICKER_FE β >> POLITICAL_TICKER_FE β, you have a
    # within-firm reversal.  If they're similar, the pooled flattening
    # is between-firm composition.
    if len(ctrl) >= 50 and ctrl['TICKER'].nunique() >= 10:
        fe_x_cols = ['LOG_TV', 'IS_BUY']
        yr_fe_cols = [c for c in year_dummy_cols
                      if c in ctrl.columns and ctrl[c].nunique() > 1]
        all_fe_x = fe_x_cols + yr_fe_cols
        ctrl_fe = ctrl[['TICKER', 'TRADE_DATE', 'PROFITABLE_30'] + all_fe_x].dropna().copy()
        ctrl_fe = ctrl_fe.reset_index(drop=True)
        ctrl_fe['_row_id'] = range(len(ctrl_fe))
        try:
            from linearmodels.panel import PanelOLS as _PanelOLS
            ctrl_fe_idx = ctrl_fe.set_index(['TICKER', '_row_id'])
            y_fe = ctrl_fe_idx['PROFITABLE_30']
            X_fe = ctrl_fe_idx[all_fe_x].astype(float)
            model = _PanelOLS(y_fe, X_fe, entity_effects=True).fit(
                cov_type='clustered', cluster_entity=True
            )
            for var in fe_x_cols:
                regression_rows.append({
                    'SPECIFICATION': 'CONTROL_TICKER_FE',
                    'VARIABLE': var,
                    'COEFFICIENT': model.params.get(var, np.nan),
                    'STD_ERROR': model.std_errors.get(var, np.nan),
                    'T_STAT': model.tstats.get(var, np.nan),
                    'P_VALUE': model.pvalues.get(var, np.nan),
                    'N_OBS': int(model.nobs),
                    'R_SQUARED': model.rsquared_within,
                })
        except Exception as e:
            logger.warning("  PanelOLS control ticker FE failed: %s", e)

    # ── Spec 6: P/S-only robustness (exclude mechanical codes F, A, M) ──
    # Mechanical trades (F=tax withholding, A=grant, M=exercise) are bigger
    # on average and less informationally accurate by construction.  If the
    # reversal survives on discretionary Purchase/Sale codes only, it's not
    # an artifact of trade-code definition.
    if 'TRANSACTION_CODE' in pol.columns:
        ps_only = pol[pol['TRANSACTION_CODE'].isin(['P', 'S'])]
        if len(ps_only) >= 30:
            _run_lpm(ps_only, 'POLITICAL_PS_ONLY',
                     ['LOG_TV', 'IS_BUY'], regression_rows)
            logger.info("  Spec 6 (P/S-only): %d discretionary trades "
                        "(of %d total)", len(ps_only), len(pol))
        else:
            logger.info("  Spec 6 (P/S-only): only %d discretionary trades — "
                        "skipped (need ≥30)", len(ps_only))
    else:
        logger.info("  Spec 6 (P/S-only): TRANSACTION_CODE not available — skipped")

    # Control-side P/S-only for symmetric comparison
    if 'TRANSACTION_CODE' in ctrl.columns:
        ctrl_ps = ctrl[ctrl['TRANSACTION_CODE'].isin(['P', 'S'])]
        if len(ctrl_ps) >= 30:
            _run_lpm(ctrl_ps, 'CONTROL_PS_ONLY',
                     ['LOG_TV', 'IS_BUY'], regression_rows)
            logger.info("  Spec 6 control (P/S-only): %d discretionary trades "
                        "(of %d total)", len(ctrl_ps), len(ctrl))
        else:
            logger.info("  Spec 6 control (P/S-only): only %d trades — "
                        "skipped (need ≥30)", len(ctrl_ps))

    reversal_regression = pd.DataFrame(regression_rows)

    logger.info("  §12 complete: %d accuracy rows, %d slope rows, "
                "%d regression rows, %d control trades, %d attrition rows",
                len(size_accuracy), len(size_accuracy_slopes),
                len(reversal_regression), len(control), len(control_attrition))

    return size_accuracy, size_accuracy_slopes, reversal_regression, control, control_attrition


# ═════════════════════════════════════════════════════════════════════
# CLUSTERED INFERENCE HELPERS
# ═════════════════════════════════════════════════════════════════════

def _effective_cluster_count(cluster_labels):
    """Effective cluster count G* (MacKinnon-Webb 2017).

    G* = N^2 / sum(n_g^2).  When G* < 12, Webb 6-point weights should
    replace 2-point Rademacher weights in wild cluster bootstrap.
    Returns (G_raw, G_eff).
    """
    unique, counts = np.unique(cluster_labels, return_counts=True)
    G = len(unique)
    sq_sum = float((counts ** 2).sum())
    g_eff = float(counts.sum()) ** 2 / sq_sum if sq_sum > 0 else float(G)
    return G, g_eff


def _wild_cluster_bootstrap_proportion(outcomes, cluster_labels,
                                        p0=0.5, n_boot=1000, seed=42):
    """Wild cluster bootstrap CI and test for a proportion.

    Treats each unique value in cluster_labels as a cluster.  Uses 2-point
    Rademacher weights (±1 with equal probability) unless G_eff < 12, in
    which case Webb (2014) 6-point weights are substituted.

    Cameron-Gelbach-Miller (2011, JBE); MacKinnon-Webb (2017, 2018).

    Returns dict with: point_est, cluster_se, ci_lo_95, ci_hi_95,
                       n_obs, g_raw, g_eff, p_value_vs_p0, p0, weights_used.
    Returns None if fewer than 10 valid observations.
    """
    outcomes = np.asarray(outcomes, dtype=float)
    cluster_labels = np.asarray(cluster_labels)
    valid = ~np.isnan(outcomes)
    outcomes = outcomes[valid]
    cluster_labels = cluster_labels[valid]
    n = len(outcomes)
    if n < 10:
        return None

    clusters = np.unique(cluster_labels)
    G = len(clusters)
    _, g_eff = _effective_cluster_count(cluster_labels)

    p_hat = outcomes.mean()

    # Cluster-level quantities
    n_g = np.array([np.sum(cluster_labels == g) for g in clusters], dtype=float)
    y_bar_g = np.array([outcomes[cluster_labels == g].mean() for g in clusters])
    w_g = n_g / n  # cluster weights

    # Choose weight distribution
    use_webb = g_eff < 12
    if use_webb:
        weight_pool = np.array([-np.sqrt(1.5), -1.0, -np.sqrt(0.5),
                                  np.sqrt(0.5),  1.0,  np.sqrt(1.5)])
        weights_used = 'webb6'
    else:
        weight_pool = np.array([-1.0, 1.0])
        weights_used = 'rademacher2'

    rng = np.random.RandomState(seed)

    # Bootstrap distribution for CI (centred at p_hat)
    boot_props = np.empty(n_boot)
    for b in range(n_boot):
        eps = rng.choice(weight_pool, size=G)
        boot_props[b] = p_hat + np.dot(w_g * (y_bar_g - p_hat), eps)

    cluster_se = boot_props.std(ddof=1)
    ci_lo_95 = np.percentile(boot_props, 2.5)
    ci_hi_95 = np.percentile(boot_props, 97.5)

    # Bootstrap test vs p0 (centred at p0, use fresh seed)
    t_obs = abs(p_hat - p0)
    rng2 = np.random.RandomState(seed + 1)
    t_star = np.empty(n_boot)
    for b in range(n_boot):
        eps = rng2.choice(weight_pool, size=G)
        t_star[b] = abs(np.dot(w_g * (y_bar_g - p0), eps))
    p_value = (t_star >= t_obs).mean()

    return {
        'point_est': p_hat,
        'cluster_se': cluster_se,
        'ci_lo_95': ci_lo_95,
        'ci_hi_95': ci_hi_95,
        'n_obs': n,
        'g_raw': G,
        'g_eff': g_eff,
        'p_value_vs_p0': p_value,
        'p0': p0,
        'weights_used': weights_used,
    }


def _conditional_sign_base_rate(event_cars):
    """Pesaran-Timmermann (1992) conditional base rate for sign tests.

    For a sell-accuracy test the natural null is not 50% but the fraction of
    events in the sample that have a negative CAR.  Events in this paper are
    selected for large adverse reactions, so p_c >> 0.5 by construction, and
    testing against 50% mechanically inflates significance.

    Returns float p_c (fraction of provided CARs that are < 0).
    """
    cars = np.asarray(event_cars, dtype=float)
    valid = ~np.isnan(cars)
    if valid.sum() == 0:
        return 0.5
    return float((cars[valid] < 0).mean())


def _date_randomization_placebo_directional(pol_df, n_perms=1000, seed=42):
    """Date-randomization placebo for directional accuracy.

    Randomly shuffles EVENT_CAR labels across events, recomputes directional
    accuracy each time.  The permutation distribution should centre near the
    conditional base rate under H0 (no informed trading).  An empirical
    p-value is computed as the fraction of permutations at least as accurate
    as the observed value.

    pol_df must contain columns: TRADE_TYPE, EVENT_CAR, EVENT_ID.
    Returns a single-row DataFrame (or empty if inputs are insufficient).
    """
    if pol_df.empty or 'EVENT_CAR' not in pol_df.columns:
        return pd.DataFrame()

    valid = pol_df[
        pol_df['EVENT_CAR'].notna() &
        pol_df['TRADE_TYPE'].isin(['buy', 'sell'])
    ].copy()

    if len(valid) < 10:
        return pd.DataFrame()

    def _compute_accuracy(df):
        sells = df[df['TRADE_TYPE'] == 'sell']
        buys  = df[df['TRADE_TYPE'] == 'buy']
        correct = int((sells['EVENT_CAR'] < 0).sum()) + int((buys['EVENT_CAR'] > 0).sum())
        total   = len(sells) + len(buys)
        return correct / total if total > 0 else np.nan

    observed_acc = _compute_accuracy(valid)
    if np.isnan(observed_acc):
        return pd.DataFrame()

    # Map of event_id -> event CAR (one value per event)
    event_car_map = valid.groupby('EVENT_ID')['EVENT_CAR'].first()
    event_ids  = event_car_map.index.values
    car_values = event_car_map.values

    rng = np.random.RandomState(seed)
    perm_accs = []
    for _ in range(n_perms):
        shuffled = rng.permutation(car_values)
        car_shuffle = dict(zip(event_ids, shuffled))
        perm_df = valid.copy()
        perm_df['EVENT_CAR'] = perm_df['EVENT_ID'].map(car_shuffle)
        perm_accs.append(_compute_accuracy(perm_df))

    perm_accs = np.array(perm_accs)
    p_value = float((perm_accs >= observed_acc).mean())

    return pd.DataFrame([{
        'TEST': 'DATE_RANDOMIZATION_PLACEBO_DIRECTIONAL',
        'OBSERVED_ACCURACY': observed_acc,
        'PERM_MEAN': perm_accs.mean(),
        'PERM_STD': perm_accs.std(),
        'PERM_CI_2_5': np.percentile(perm_accs, 2.5),
        'PERM_CI_97_5': np.percentile(perm_accs, 97.5),
        'P_VALUE_VS_PERMUTATION': p_value,
        'N_OBS': len(valid),
        'N_EVENTS': len(event_ids),
        'N_PERMS': n_perms,
    }])


# ═════════════════════════════════════════════════════════════════════
# §12b INFORMED TRADING TEST (HEADLINE)
# ═════════════════════════════════════════════════════════════════════

def compute_informed_trading_test(panel, crsp_profits, control_trades):
    """Headline test: do insiders trade on advance knowledge of political decisions?

    Uses event-window CAR (panel CAR_POST) to define event-profitability,
    avoiding the mechanical overlap between trade-window CARs and event
    windows for trades near the event date.

    A trade is "event-profitable" if:
      - Sell and CAR_POST < 0 (sold before price drop)
      - Buy and CAR_POST > 0 (bought before price rise)

    Returns (informed_trading, informed_proximity, informed_dollars,
             cluster_inference, directional_placebo, car_net_slope, joint_pt).
    cluster_inference (ESSAY3_CLUSTER_INFERENCE): event- and ticker-clustered
      wild bootstrap CIs and p-values for the headline proportion, plus
      Pesaran-Timmermann conditional-null variant.
    directional_placebo (ESSAY3_DIRECTIONAL_PLACEBO): date-randomization
      placebo that shuffles EVENT_CAR labels across events.
    car_net_slope (ESSAY3_CAR_NET_SLOPE): OLS of event-level CAR on net sell
      direction (sign of $ sells minus $ buys), one row per event, HC3 SE.
      Base-rate-immune: selection shifts mean CAR, not its covariance with
      what insiders did.
    joint_pt (ESSAY3_JOINT_PT): joint Pesaran-Timmermann 2x2 sign test
      (buys + sells against independence null).
    """
    empty = (pd.DataFrame(), pd.DataFrame(), pd.DataFrame(),
             pd.DataFrame(), pd.DataFrame(), pd.DataFrame(), pd.DataFrame())
    if crsp_profits is None or crsp_profits.empty:
        logger.warning("  No CRSP profits for informed trading test")
        return empty

    # ── Merge event-level CAR and EVENT_DATE onto trades ──────────────
    # Merge on (EVENT_ID, TICKER, EVENT_DATE) to prevent fan-out when multiple
    # events share the same EVENT_ID (e.g. several 2017 House votes before the
    # upstream ID was made date-specific).  drop_duplicates ensures the lookup
    # table is at most one row per key, so the merge is always many-to-one.
    # Rename CAR_POST only; keep EVENT_DATE as the merge key and alias it to
    # EVENT_DATE_PANEL after the join.
    event_info = (
        panel[['EVENT_ID', 'TICKER', 'EVENT_DATE', 'CAR_POST']]
        .drop_duplicates(subset=['EVENT_ID', 'TICKER', 'EVENT_DATE'])
        .rename(columns={'CAR_POST': 'EVENT_CAR'})
        .copy()
    )

    pol = crsp_profits.merge(
        event_info, on=['EVENT_ID', 'TICKER', 'EVENT_DATE'], how='left'
    )
    # Alias EVENT_DATE to EVENT_DATE_PANEL (the name the rest of §12b expects).
    pol['EVENT_DATE_PANEL'] = pol['EVENT_DATE']
    pol['EVENT_DATE_PANEL'] = pd.to_datetime(pol['EVENT_DATE_PANEL'])
    pol['TRADE_DATE'] = pd.to_datetime(pol['TRADE_DATE'])

    # Compute proximity
    pol['DAYS_BEFORE_EVENT'] = (pol['EVENT_DATE_PANEL'] - pol['TRADE_DATE']).dt.days

    # Event-profitability (uses EVENT CAR, not trade-window CAR)
    pol['EVENT_CAR'] = pd.to_numeric(pol['EVENT_CAR'], errors='coerce')
    has_event_car = pol['EVENT_CAR'].notna()
    pol['EVENT_PROFITABLE'] = np.nan
    pol.loc[has_event_car & (pol['TRADE_TYPE'] == 'sell'),
            'EVENT_PROFITABLE'] = (pol.loc[has_event_car & (pol['TRADE_TYPE'] == 'sell'),
                                           'EVENT_CAR'] < 0).astype(float)
    pol.loc[has_event_car & (pol['TRADE_TYPE'] == 'buy'),
            'EVENT_PROFITABLE'] = (pol.loc[has_event_car & (pol['TRADE_TYPE'] == 'buy'),
                                           'EVENT_CAR'] > 0).astype(float)

    # Event-level signed profit: CAR_event * trade_value (direction-adjusted)
    pol['EVENT_PROFIT'] = np.where(
        pol['TRADE_TYPE'] == 'sell',
        -pol['EVENT_CAR'] * pol['TRADE_VALUE'],  # sell profits when CAR < 0
        pol['EVENT_CAR'] * pol['TRADE_VALUE'],   # buy profits when CAR > 0
    )

    valid_pol = pol[pol['EVENT_PROFITABLE'].notna()].copy()
    logger.info("  Informed trading test: %d political trades with event CAR "
                "(of %d total)", len(valid_pol), len(crsp_profits))

    # Compute conditional base rate NOW from the analysis events (not full panel).
    # Using valid_pol['EVENT_CAR'] ensures the base rate matches the exact events
    # that appear in the analysis — avoiding a subtle population mismatch.
    _analysis_event_cars = valid_pol.groupby('EVENT_ID')['EVENT_CAR'].first().dropna()
    p_cond_sell = _conditional_sign_base_rate(_analysis_event_cars)
    # p_cond_buy = P(CAR >= 0) = 1 - P(CAR < 0).  EVENT_PROFITABLE for buys
    # is defined as (EVENT_CAR > 0), so informed buying implies accuracy >
    # p_cond_buy.  The one-tailed test is H1: buy accuracy > p_cond_buy (is_sell=True).
    p_cond_buy  = 1.0 - p_cond_sell
    logger.info("  P-T conditional null (analysis events): sell=%.4f buy=%.4f",
                p_cond_sell, p_cond_buy)

    # ── Helper: summarize a subset ────────────────────────────────────
    def _summarize(sub, label, sample='POLITICAL', p_cond=None):
        """Summarize directional accuracy for a trade subset.

        p_cond: if provided, also report METRIC_PVAL_COND (vs conditional null).
                Should be p_cond_sell for sell subsets, p_cond_buy for buy subsets.
        """
        n = len(sub)
        if n < 5:
            return None
        pct = sub['EVENT_PROFITABLE'].mean()
        n_prof = int(sub['EVENT_PROFITABLE'].sum())
        binom_p_50 = stats.binomtest(n_prof, n, 0.5).pvalue
        binom_p_cond = (
            stats.binomtest(n_prof, n, p_cond).pvalue
            if p_cond is not None else np.nan
        )
        sp = sub['EVENT_PROFIT'].dropna()
        tv = sub['TRADE_VALUE'].dropna()
        return {
            'CUT': label,
            'SAMPLE': sample,
            'METRIC_TYPE': 'PROPORTION',
            'N_TRADES': n,
            'METRIC_VALUE': pct,
            'METRIC_PVAL': binom_p_50,       # naive vs 50% — DO NOT use as headline
            'METRIC_PVAL_COND': binom_p_cond, # honest vs conditional base rate
            'COND_NULL': p_cond if p_cond is not None else np.nan,
            'MEAN_TRADE_VALUE': tv.mean() if len(tv) > 0 else np.nan,
            'MEAN_EVENT_PROFIT': sp.mean() if len(sp) > 0 else np.nan,
            'TOTAL_EVENT_PROFIT': sp.sum() if len(sp) > 0 else np.nan,
            'MEAN_EVENT_CAR': sub['EVENT_CAR'].mean()
            if 'EVENT_CAR' in sub.columns and sub['EVENT_CAR'].notna().any()
            else np.nan,
        }

    # ── Table 1: INFORMED_TRADING — headline summary by cut ──────────
    summary_rows = []

    # All pre-event trades
    r = _summarize(valid_pol, 'ALL')
    if r:
        summary_rows.append(r)

    # By direction — pass conditional null to sell and buy subsets
    sells = valid_pol[valid_pol['TRADE_TYPE'] == 'sell']
    buys = valid_pol[valid_pol['TRADE_TYPE'] == 'buy']
    for sub, label, pc in [(sells, 'SELLS_ONLY', p_cond_sell),
                           (buys, 'BUYS_ONLY', p_cond_buy)]:
        r = _summarize(sub, label, p_cond=pc)
        if r:
            summary_rows.append(r)

    # Discretionary trades only (Purchase/Sale codes P and S).  Excludes
    # M (option exercise), D (disposition), F (tax withholding on RSU vest),
    # A (non-purchase acquisition).  Matches the §12 POLITICAL_PS_ONLY filter.
    if 'TRANSACTION_CODE' in valid_pol.columns:
        ps_only = valid_pol[valid_pol['TRANSACTION_CODE'].isin(['P', 'S'])]
        ps_sells = ps_only[ps_only['TRADE_TYPE'] == 'sell']
        ps_buys = ps_only[ps_only['TRADE_TYPE'] == 'buy']
        for sub, label, pc in [(ps_only, 'ALL_PS_ONLY', None),
                               (ps_sells, 'SELLS_PS_ONLY', p_cond_sell),
                               (ps_buys, 'BUYS_PS_ONLY', p_cond_buy)]:
            r = _summarize(sub, label, p_cond=pc)
            if r:
                summary_rows.append(r)

    # 10b5-1 proxy: insider-level routine vs opportunistic (CMP classification)
    # The headline rests on sells being informationally motivated rather than
    # scheduled.  Splitting at the insider level (not ticker) per Lambert.
    if 'IS_ROUTINE' in valid_pol.columns:
        opp_sells = sells[sells['IS_ROUTINE'] == False]  # noqa: E712
        rout_sells = sells[sells['IS_ROUTINE'] == True]  # noqa: E712
        for sub, label in [(opp_sells, 'SELLS_OPPORTUNISTIC'),
                           (rout_sells, 'SELLS_ROUTINE')]:
            r = _summarize(sub, label, p_cond=p_cond_sell)
            if r:
                summary_rows.append(r)

    # Post-2023 10b5-1 plan split (Lambert review 2026-07-06):
    # Split post-2023 sells into plan-flagged vs non-plan to test whether the
    # accuracy premium is driven by discretionary trades or pre-scheduled ones.
    # Amendment effective 2023-02-27 — use that exact cutoff, not Jan 1.
    if 'IS_10B5_1_PLAN' in sells.columns:
        sells_post23 = sells[sells['TRADE_DATE'] >= pd.Timestamp('2023-02-27')]
        if len(sells_post23) >= 10:
            plan_sells = sells_post23[sells_post23['IS_10B5_1_PLAN'] == True]   # noqa: E712
            noplan_sells = sells_post23[sells_post23['IS_10B5_1_PLAN'] == False]  # noqa: E712
            for sub, label in [(sells_post23, 'SELLS_POST2023'),
                               (plan_sells, 'SELLS_10B5_PLAN'),
                               (noplan_sells, 'SELLS_NOT_10B5_PLAN')]:
                r = _summarize(sub, label, p_cond=p_cond_sell)
                if r:
                    summary_rows.append(r)

    # Section 16(b) short-swing cut: sells where the insider bought the same
    # stock within the prior 6 months.  These sellers accept disgorgement risk,
    # implying stronger informational motivation — not weaker — because the
    # expected gain must exceed the 16(b) clawback.
    if 'HAS_PRIOR_BUY_6M' in sells.columns:
        s16b = sells[sells['HAS_PRIOR_BUY_6M'] == True]  # noqa: E712
        s_no16b = sells[sells['HAS_PRIOR_BUY_6M'] == False]  # noqa: E712
        for sub, label in [(s16b, 'SELLS_16B_ELIGIBLE'),
                           (s_no16b, 'SELLS_NO_16B')]:
            r = _summarize(sub, label, p_cond=p_cond_sell)
            if r:
                summary_rows.append(r)

    # Control sample comparison (uses trade-window CAR, not event CAR)
    if control_trades is not None and not control_trades.empty:
        ctrl = control_trades[control_trades['PROFITABLE_30'].notna()].copy()
        if len(ctrl) >= 5:
            n_ctrl = len(ctrl)
            pct_ctrl = ctrl['PROFITABLE_30'].mean()
            n_prof_ctrl = int(ctrl['PROFITABLE_30'].sum())
            binom_ctrl = stats.binomtest(n_prof_ctrl, n_ctrl, 0.5).pvalue
            summary_rows.append({
                'CUT': 'ALL', 'SAMPLE': 'CONTROL',
                'METRIC_TYPE': 'PROPORTION',
                'N_TRADES': n_ctrl,
                'METRIC_VALUE': pct_ctrl,
                'METRIC_PVAL': binom_ctrl,
                'METRIC_PVAL_COND': np.nan,
                'COND_NULL': np.nan,
                'MEAN_TRADE_VALUE': ctrl['TRADE_VALUE'].mean(),
                'MEAN_EVENT_PROFIT': np.nan,
                'TOTAL_EVENT_PROFIT': np.nan,
                'MEAN_EVENT_CAR': np.nan,
            })
            # Control by direction
            for ttype, label in [('sell', 'SELLS_ONLY'), ('buy', 'BUYS_ONLY')]:
                csub = ctrl[ctrl['TRADE_TYPE'] == ttype]
                if len(csub) >= 5:
                    n_cs = len(csub)
                    pct_cs = csub['PROFITABLE_30'].mean()
                    summary_rows.append({
                        'CUT': label, 'SAMPLE': 'CONTROL',
                        'METRIC_TYPE': 'PROPORTION',
                        'N_TRADES': n_cs,
                        'METRIC_VALUE': pct_cs,
                        'METRIC_PVAL': stats.binomtest(
                            int(csub['PROFITABLE_30'].sum()), n_cs, 0.5
                        ).pvalue,
                        'METRIC_PVAL_COND': np.nan,
                        'COND_NULL': np.nan,
                        'MEAN_TRADE_VALUE': csub['TRADE_VALUE'].mean(),
                        'MEAN_EVENT_PROFIT': np.nan,
                        'TOTAL_EVENT_PROFIT': np.nan,
                        'MEAN_EVENT_CAR': np.nan,
                    })

    # Political vs control difference (sells) — apples-to-apples via PROFITABLE_30
    # Both sides use 30-day post-trade CAR profitability (same window, same definition).
    # NOTE: earlier versions used EVENT_PROFITABLE for political vs PROFITABLE_30 for
    # control — that was methodologically inconsistent and has been removed.
    if ('PROFITABLE_30' in valid_pol.columns and
            control_trades is not None and not control_trades.empty):
        pol_sells_30 = sells[sells['PROFITABLE_30'].notna()]
        ctrl_sells_30 = control_trades[
            (control_trades['TRADE_TYPE'] == 'sell') &
            control_trades['PROFITABLE_30'].notna()
        ]
        if len(pol_sells_30) >= 5 and len(ctrl_sells_30) >= 5:
            # Fix inflated-N: aggregate to cluster level before computing SE.
            # Political sells cluster within events → collapse to event-level means.
            # Control sells cluster within tickers → collapse to ticker-level means.
            # Welch t-test on the resulting independent cluster-level proportions.
            pol_ev_acc = (
                pol_sells_30.groupby('EVENT_ID')['PROFITABLE_30'].mean()
                if 'EVENT_ID' in pol_sells_30.columns
                else pol_sells_30['PROFITABLE_30']
            )
            ctrl_tkr_acc = (
                ctrl_sells_30.groupby('TICKER')['PROFITABLE_30'].mean()
                if 'TICKER' in ctrl_sells_30.columns
                else ctrl_sells_30['PROFITABLE_30']
            )
            n1_eff, n2_eff = len(pol_ev_acc), len(ctrl_tkr_acc)
            if n1_eff >= 5 and n2_eff >= 5:
                premium = pol_ev_acc.mean() - ctrl_tkr_acc.mean()
                _, z_pval = stats.ttest_ind(
                    pol_ev_acc.values, ctrl_tkr_acc.values, equal_var=False
                )
                summary_rows.append({
                    'CUT': 'SELLS_PREMIUM', 'SAMPLE': 'DIFFERENCE',
                    'METRIC_TYPE': 'DIFFERENCE',
                    'METRIC_PVAL': z_pval,
                    'METRIC_PVAL_COND': np.nan,
                    'COND_NULL': np.nan,
                    'N_TRADES': len(pol_sells_30) + len(ctrl_sells_30),
                    'METRIC_VALUE': premium,
                    'MEAN_TRADE_VALUE': np.nan,
                    'MEAN_EVENT_PROFIT': np.nan,
                    'TOTAL_EVENT_PROFIT': np.nan,
                    'MEAN_EVENT_CAR': np.nan,
                })

    # Apples-to-apples comparison using trade-window CAR_30 for both samples
    if ('PROFITABLE_30' in crsp_profits.columns and
            control_trades is not None and not control_trades.empty):
        pol_sells_pt = crsp_profits[
            (crsp_profits['TRADE_TYPE'] == 'sell') &
            crsp_profits['PROFITABLE_30'].notna()
        ]
        ctrl_sells_pt = control_trades[
            (control_trades['TRADE_TYPE'] == 'sell') &
            control_trades['PROFITABLE_30'].notna()
        ]
        if len(pol_sells_pt) >= 5 and len(ctrl_sells_pt) >= 5:
            # Fix inflated-N: same cluster-level aggregation as SELLS_PREMIUM.
            pol_ev_acc_pt = (
                pol_sells_pt.groupby('EVENT_ID')['PROFITABLE_30'].mean()
                if 'EVENT_ID' in pol_sells_pt.columns
                else pol_sells_pt['PROFITABLE_30']
            )
            ctrl_tkr_acc_pt = (
                ctrl_sells_pt.groupby('TICKER')['PROFITABLE_30'].mean()
                if 'TICKER' in ctrl_sells_pt.columns
                else ctrl_sells_pt['PROFITABLE_30']
            )
            n1_eff_pt, n2_eff_pt = len(pol_ev_acc_pt), len(ctrl_tkr_acc_pt)
            if n1_eff_pt >= 5 and n2_eff_pt >= 5:
                premium_pt = pol_ev_acc_pt.mean() - ctrl_tkr_acc_pt.mean()
                _, z_p = stats.ttest_ind(
                    pol_ev_acc_pt.values, ctrl_tkr_acc_pt.values, equal_var=False
                )
                summary_rows.append({
                    'CUT': 'SELLS_PREMIUM_TRADE_CAR', 'SAMPLE': 'DIFFERENCE',
                    'METRIC_TYPE': 'DIFFERENCE',
                    'N_TRADES': len(pol_sells_pt) + len(ctrl_sells_pt),
                    'METRIC_VALUE': premium_pt,
                    'METRIC_PVAL': z_p,
                    'MEAN_TRADE_VALUE': np.nan,
                    'MEAN_EVENT_PROFIT': np.nan,
                    'TOTAL_EVENT_PROFIT': np.nan,
                    'MEAN_EVENT_CAR': np.nan,
                })

    # By regulatory period (min 30 trades for meaningful inference)
    for period in sorted(valid_pol['REGULATORY_PERIOD'].dropna().unique()):
        sub = valid_pol[valid_pol['REGULATORY_PERIOD'] == period]
        if len(sub) >= 30:
            r = _summarize(sub, f'REG:{period}')
            if r:
                summary_rows.append(r)
        # Sells only by period
        sub_sells = sub[sub['TRADE_TYPE'] == 'sell']
        if len(sub_sells) >= 30:
            r = _summarize(sub_sells, f'REG_SELLS:{period}')
            if r:
                summary_rows.append(r)

    # Control by regulatory period (for within-period comparison)
    if control_trades is not None and not control_trades.empty:
        ctrl_valid = control_trades[control_trades['PROFITABLE_30'].notna()].copy()
        if len(ctrl_valid) > 0:
            ctrl_valid['TRADE_DATE'] = pd.to_datetime(ctrl_valid['TRADE_DATE'])
            ctrl_valid['REG_PERIOD'] = ctrl_valid['TRADE_DATE'].apply(
                _assign_regulatory_period)
            for period in sorted(ctrl_valid['REG_PERIOD'].unique()):
                csub = ctrl_valid[ctrl_valid['REG_PERIOD'] == period]
                if len(csub) >= 30:
                    pct = csub['PROFITABLE_30'].mean()
                    n_p = int(csub['PROFITABLE_30'].sum())
                    summary_rows.append({
                        'CUT': f'REG:{period}', 'SAMPLE': 'CONTROL',
                        'METRIC_TYPE': 'PROPORTION',
                        'N_TRADES': len(csub),
                        'METRIC_VALUE': pct,
                        'METRIC_PVAL': stats.binomtest(n_p, len(csub), 0.5).pvalue,
                        'MEAN_TRADE_VALUE': csub['TRADE_VALUE'].mean(),
                        'MEAN_EVENT_PROFIT': np.nan,
                        'TOTAL_EVENT_PROFIT': np.nan,
                        'MEAN_EVENT_CAR': np.nan,
                    })
                # Control sells by period
                csub_sells = csub[csub['TRADE_TYPE'] == 'sell']
                if len(csub_sells) >= 30:
                    pct = csub_sells['PROFITABLE_30'].mean()
                    n_p = int(csub_sells['PROFITABLE_30'].sum())
                    summary_rows.append({
                        'CUT': f'REG_SELLS:{period}', 'SAMPLE': 'CONTROL',
                        'METRIC_TYPE': 'PROPORTION',
                        'N_TRADES': len(csub_sells),
                        'METRIC_VALUE': pct,
                        'METRIC_PVAL': stats.binomtest(
                            n_p, len(csub_sells), 0.5).pvalue,
                        'MEAN_TRADE_VALUE': csub_sells['TRADE_VALUE'].mean(),
                        'MEAN_EVENT_PROFIT': np.nan,
                        'TOTAL_EVENT_PROFIT': np.nan,
                        'MEAN_EVENT_CAR': np.nan,
                    })

    # By event category
    for cat in valid_pol['EVENT_CATEGORY'].dropna().unique():
        sub = valid_pol[valid_pol['EVENT_CATEGORY'] == cat]
        r = _summarize(sub, f'CAT:{cat}')
        if r:
            summary_rows.append(r)

    # Year distribution for diagnostic
    valid_pol['YEAR'] = valid_pol['TRADE_DATE'].dt.year
    for year in sorted(valid_pol['YEAR'].unique()):
        sub = valid_pol[valid_pol['YEAR'] == year]
        r = _summarize(sub, f'YEAR:{year}')
        if r:
            summary_rows.append(r)

    # ── Table 2: INFORMED_PROXIMITY — by days-before-event window ────
    proximity_rows = []
    # PRE_FULL extraction ends at event_date−1, so DAYS_BEFORE_EVENT=0 trades
    # cannot enter the analysis set via the normal pipeline.  Windows start at 0
    # so they span [0,30], [31,60], [61,90], [91,180] as defined.
    windows = [(0, 30), (31, 60), (61, 90), (91, 180)]
    for w_start, w_end in windows:
        sub = valid_pol[
            (valid_pol['DAYS_BEFORE_EVENT'] >= w_start) &
            (valid_pol['DAYS_BEFORE_EVENT'] <= w_end)
        ]
        label = f'{w_start}-{w_end}d'
        r = _summarize(sub, label)
        if r:
            r['WINDOW_START'] = w_start
            r['WINDOW_END'] = w_end
            proximity_rows.append(r)
        # Sells only
        sub_sells = sub[sub['TRADE_TYPE'] == 'sell']
        r = _summarize(sub_sells, f'{label}_SELLS')
        if r:
            r['WINDOW_START'] = w_start
            r['WINDOW_END'] = w_end
            proximity_rows.append(r)
        # Buys only
        sub_buys = sub[sub['TRADE_TYPE'] == 'buy']
        r = _summarize(sub_buys, f'{label}_BUYS')
        if r:
            r['WINDOW_START'] = w_start
            r['WINDOW_END'] = w_end
            proximity_rows.append(r)

    # Control reference rows (no event date, so unconditional baseline)
    if control_trades is not None and not control_trades.empty:
        ctrl_valid = control_trades[control_trades['PROFITABLE_30'].notna()].copy()
        for ttype, suffix in [('sell', '_SELLS'), ('buy', '_BUYS'), (None, '')]:
            csub = ctrl_valid if ttype is None else ctrl_valid[ctrl_valid['TRADE_TYPE'] == ttype]
            if len(csub) >= 5:
                pct = csub['PROFITABLE_30'].mean()
                n_p = int(csub['PROFITABLE_30'].sum())
                proximity_rows.append({
                    'CUT': f'CONTROL_ALL{suffix}',
                    'SAMPLE': 'CONTROL',
                    'METRIC_TYPE': 'PROPORTION',
                    'N_TRADES': len(csub),
                    'METRIC_VALUE': pct,
                    'METRIC_PVAL': stats.binomtest(n_p, len(csub), 0.5).pvalue,
                    'MEAN_TRADE_VALUE': csub['TRADE_VALUE'].mean(),
                    'MEAN_EVENT_PROFIT': np.nan,
                    'TOTAL_EVENT_PROFIT': np.nan,
                    'MEAN_EVENT_CAR': np.nan,
                    'WINDOW_START': 0,
                    'WINDOW_END': 180,
                })

    informed_proximity = pd.DataFrame(proximity_rows) if proximity_rows else pd.DataFrame()

    # Bonferroni correction for the 4 proximity windows (multiple-testing adjustment)
    # Applies to POLITICAL PROPORTION rows only (not control, not DIFFERENCE rows).
    if not informed_proximity.empty and 'METRIC_PVAL' in informed_proximity.columns:
        n_windows = len(windows)  # 4
        mask = (
            (informed_proximity['SAMPLE'] == 'POLITICAL') &
            (informed_proximity['METRIC_TYPE'] == 'PROPORTION')
        )
        informed_proximity.loc[mask, 'METRIC_PVAL_BONF'] = (
            informed_proximity.loc[mask, 'METRIC_PVAL'] * n_windows
        ).clip(upper=1.0)
        informed_proximity['METRIC_PVAL_BONF'] = informed_proximity.get(
            'METRIC_PVAL_BONF', np.nan)

    # ── Table 3: INFORMED_DOLLARS — by severity and proximity ────────
    dollar_rows = []

    # Sells by proximity window (dollar magnitudes)
    for w_start, w_end in windows:
        sub_sells = valid_pol[
            (valid_pol['TRADE_TYPE'] == 'sell') &
            (valid_pol['DAYS_BEFORE_EVENT'] >= w_start) &
            (valid_pol['DAYS_BEFORE_EVENT'] <= w_end)
        ]
        if len(sub_sells) >= 5:
            sp = sub_sells['EVENT_PROFIT'].dropna()
            dollar_rows.append({
                'CUT': f'SELLS_{w_start}-{w_end}d',
                'N_SELLS': len(sub_sells),
                'MEAN_EVENT_CAR': sub_sells['EVENT_CAR'].mean(),
                'MEAN_TRADE_VALUE': sub_sells['TRADE_VALUE'].mean(),
                'MEAN_PROFIT': sp.mean() if len(sp) > 0 else np.nan,
                'TOTAL_PROFIT': sp.sum() if len(sp) > 0 else np.nan,
                'METRIC_VALUE': sub_sells['EVENT_PROFITABLE'].mean(),
            })

    # Sells before events with negative CARs (severity cuts).
    for car_threshold in [-0.05, -0.10, -0.15]:
        for w_start, w_end in [(0, 30), (0, 60), (0, 180)]:
            sub = valid_pol[
                (valid_pol['TRADE_TYPE'] == 'sell') &
                (valid_pol['EVENT_CAR'] <= car_threshold) &
                (valid_pol['DAYS_BEFORE_EVENT'] >= w_start) &
                (valid_pol['DAYS_BEFORE_EVENT'] <= w_end)
            ]
            if len(sub) >= 5:
                sp = sub['EVENT_PROFIT'].dropna()
                dollar_rows.append({
                    'CUT': f'SELLS_{w_start}-{w_end}d_CAR<={int(car_threshold*100)}pct',
                    'N_SELLS': len(sub),
                    'MEAN_EVENT_CAR': sub['EVENT_CAR'].mean(),
                    'MEAN_TRADE_VALUE': sub['TRADE_VALUE'].mean(),
                    'MEAN_PROFIT': sp.mean() if len(sp) > 0 else np.nan,
                    'TOTAL_PROFIT': sp.sum() if len(sp) > 0 else np.nan,
                    # PCT_EVENT_PROFITABLE omitted: mechanically 1.0
                    # (sells with CAR ≤ threshold < 0 are always event-profitable)
                    'METRIC_VALUE': np.nan,
                })

    # Aggregate totals
    all_sells = valid_pol[valid_pol['TRADE_TYPE'] == 'sell']
    if len(all_sells) >= 5:
        sp = all_sells['EVENT_PROFIT'].dropna()
        # Restrict to directionally-correct sells for "informed profit" total
        correct_sells = all_sells[all_sells['EVENT_PROFITABLE'] == 1.0]
        correct_sp = correct_sells['EVENT_PROFIT'].dropna()
        dollar_rows.append({
            'CUT': 'ALL_SELLS_TOTAL',
            'N_SELLS': len(all_sells),
            'MEAN_EVENT_CAR': all_sells['EVENT_CAR'].mean(),
            'MEAN_TRADE_VALUE': all_sells['TRADE_VALUE'].mean(),
            'MEAN_PROFIT': sp.mean() if len(sp) > 0 else np.nan,
            'TOTAL_PROFIT': sp.sum() if len(sp) > 0 else np.nan,
            'METRIC_VALUE': all_sells['EVENT_PROFITABLE'].mean(),
        })
        if len(correct_sells) >= 5:
            dollar_rows.append({
                'CUT': 'CORRECT_SELLS_TOTAL',
                'N_SELLS': len(correct_sells),
                'MEAN_EVENT_CAR': correct_sells['EVENT_CAR'].mean(),
                'MEAN_TRADE_VALUE': correct_sells['TRADE_VALUE'].mean(),
                'MEAN_PROFIT': correct_sp.mean() if len(correct_sp) > 0 else np.nan,
                'TOTAL_PROFIT': correct_sp.sum() if len(correct_sp) > 0 else np.nan,
                'METRIC_VALUE': np.nan,  # mechanically 1.0 by definition
            })

    informed_dollars = pd.DataFrame(dollar_rows) if dollar_rows else pd.DataFrame()

    # ── Far-window apples-to-apples (days 61-180; no event-return overlap) ──
    # Placed outside the EVENT_ID cluster guard so it is always computed when
    # crsp_profits and control_trades are available.  Uses cluster-level Welch
    # t-test (same pattern as SELLS_PREMIUM / SELLS_PREMIUM_TRADE_CAR).
    if ('PROFITABLE_30' in crsp_profits.columns and
            control_trades is not None and not control_trades.empty):
        pol_far = crsp_profits[
            crsp_profits['TRADE_TYPE'].isin(['sell']) &
            crsp_profits['PROFITABLE_30'].notna()
        ].copy()
        if 'EVENT_DATE_PANEL' in pol_far.columns:
            pol_far['DAYS_BEFORE'] = (
                pd.to_datetime(pol_far['EVENT_DATE_PANEL']) -
                pd.to_datetime(pol_far['TRADE_DATE'])
            ).dt.days
        elif 'DAYS_BEFORE_EVENT' in pol_far.columns:
            pol_far['DAYS_BEFORE'] = pol_far['DAYS_BEFORE_EVENT']
        else:
            pol_far = pd.DataFrame()

        if not pol_far.empty:
            pol_far = pol_far[pol_far['DAYS_BEFORE'].between(61, 180)]
        ctrl_far = (
            control_trades[control_trades['TRADE_TYPE'] == 'sell']
            if not control_trades.empty else pd.DataFrame()
        )
        if len(pol_far) >= 5 and len(ctrl_far) >= 5:
            ctrl_far_v = ctrl_far[ctrl_far['PROFITABLE_30'].notna()]
            # Fix inflated-N: aggregate to cluster level before computing SE.
            pol_ev_acc_f = (
                pol_far.groupby('EVENT_ID')['PROFITABLE_30'].mean()
                if 'EVENT_ID' in pol_far.columns
                else pol_far['PROFITABLE_30']
            )
            ctrl_tkr_acc_f = (
                ctrl_far_v.groupby('TICKER')['PROFITABLE_30'].mean()
                if 'TICKER' in ctrl_far_v.columns
                else ctrl_far_v['PROFITABLE_30']
            )
            n1f_eff, n2f_eff = len(pol_ev_acc_f), len(ctrl_tkr_acc_f)
            if n1f_eff >= 5 and n2f_eff >= 5:
                premium_f = pol_ev_acc_f.mean() - ctrl_tkr_acc_f.mean()
                _, z_pf = stats.ttest_ind(
                    pol_ev_acc_f.values, ctrl_tkr_acc_f.values, equal_var=False
                )
                summary_rows.append({
                    'CUT': 'SELLS_PREMIUM_TRADE_CAR_FAR',
                    'SAMPLE': 'DIFFERENCE',
                    'METRIC_TYPE': 'DIFFERENCE',
                    'N_TRADES': len(pol_far) + len(ctrl_far_v),
                    'METRIC_VALUE': premium_f,
                    'METRIC_PVAL': z_pf,
                    'MEAN_TRADE_VALUE': np.nan,
                    'MEAN_EVENT_PROFIT': np.nan,
                    'TOTAL_EVENT_PROFIT': np.nan,
                    'MEAN_EVENT_CAR': np.nan,
                })

    # ── Table 4: CLUSTER_INFERENCE — event-clustered CIs and tests ────
    # Lambert (2026-07-06): the exact-binomial test treats 26,817 sells as
    # independent but they are nested in ~160 events (within-event outcome
    # correlation ≈ 1).  Wild cluster bootstrap on EVENT_ID inflates the SE
    # by ~sqrt(N_trades/N_events) ≈ 13x and yields the honest CI.
    # The conditional sign test (Pesaran-Timmermann 1992) replaces the fixed
    # 50% null with the realized fraction of events having CAR < 0.
    cluster_rows = []

    if len(valid_pol) >= 10 and 'EVENT_ID' in valid_pol.columns:
        # p_cond_sell and p_cond_buy were computed earlier (from analysis events)

        # Magnitude-gated sell accuracy (|CAR| thresholds)
        for car_thr in [0.02, 0.05, 0.10, 0.15]:
            mag_sells = valid_pol[
                (valid_pol['TRADE_TYPE'] == 'sell') &
                (valid_pol['EVENT_CAR'].abs() >= car_thr)
            ]
            if len(mag_sells) >= 10:
                r = _summarize(mag_sells, f'SELLS_|CAR|>={int(car_thr*100)}pct')
                if r:
                    r['METRIC_TYPE'] = 'MAGNITUDE_GATED'
                    summary_rows.append(r)

        # |CAR|-weighted accuracy on all sells.
        # Bootstrap p-value: permute EVENT_PROFITABLE labels within each event
        # (preserving within-event structure) and compute the null distribution
        # of the weighted accuracy.
        wt_sells = valid_pol[
            (valid_pol['TRADE_TYPE'] == 'sell') &
            valid_pol['EVENT_PROFITABLE'].notna() &
            valid_pol['EVENT_CAR'].notna()
        ]
        if len(wt_sells) >= 10:
            weights = wt_sells['EVENT_CAR'].abs()
            if weights.sum() > 0:
                wt_acc = (wt_sells['EVENT_PROFITABLE'] * weights).sum() / weights.sum()
                # Bootstrap null: shuffle EVENT_PROFITABLE across all trades
                _rng_wt = np.random.RandomState(77)
                _ep = wt_sells['EVENT_PROFITABLE'].values
                _wt = weights.values
                _boot_wt = np.array([
                    (_rng_wt.permutation(_ep) * _wt).sum() / _wt.sum()
                    for _ in range(2000)
                ])
                wt_pval = (np.abs(_boot_wt - 0.5) >= np.abs(wt_acc - 0.5)).mean()
                summary_rows.append({
                    'CUT': 'SELLS_CAR_WEIGHTED',
                    'SAMPLE': 'POLITICAL',
                    'METRIC_TYPE': 'WEIGHTED_PROPORTION',
                    'N_TRADES': len(wt_sells),
                    'METRIC_VALUE': wt_acc,
                    'METRIC_PVAL': wt_pval,
                    'MEAN_TRADE_VALUE': wt_sells['TRADE_VALUE'].mean(),
                    'MEAN_EVENT_PROFIT': np.nan,
                    'TOTAL_EVENT_PROFIT': np.nan,
                    'MEAN_EVENT_CAR': wt_sells['EVENT_CAR'].mean(),
                })

        # Rebuild informed_trading now that we've appended more rows
        informed_trading = pd.DataFrame(summary_rows) if summary_rows else pd.DataFrame()

        # ── Benjamini-Hochberg FDR correction across all PROPORTION cuts ──
        # The large number of direction × period × year cuts inflates Type I
        # error. BH is applied across POLITICAL PROPORTION rows only; CONTROL,
        # DIFFERENCE, MAGNITUDE_GATED, and WEIGHTED_PROPORTION rows are excluded
        # (they either use independent two-sample tests or are descriptive).
        if not informed_trading.empty and 'METRIC_PVAL' in informed_trading.columns:
            _mask_prop = (
                (informed_trading['SAMPLE'] == 'POLITICAL') &
                (informed_trading['METRIC_TYPE'] == 'PROPORTION') &
                informed_trading['METRIC_PVAL'].notna()
            )
            _prop_pvals = np.where(
                np.isnan(informed_trading.loc[_mask_prop, 'METRIC_PVAL'].values),
                1.0,
                informed_trading.loc[_mask_prop, 'METRIC_PVAL'].values,
            )
            _bh_sig = _benjamini_hochberg(_prop_pvals, alpha=0.05)
            informed_trading['BH_SIGNIFICANT'] = np.nan
            informed_trading.loc[_mask_prop, 'BH_SIGNIFICANT'] = _bh_sig.astype(int)
            # Store BH-adjusted p (Benjamini-Hochberg step-up adjusted value)
            _n_tests = int(_mask_prop.sum())
            _sorted_idx = np.argsort(_prop_pvals)
            _bh_adj = np.ones(len(_prop_pvals))
            _running_min = 1.0
            for _rank in range(_n_tests - 1, -1, -1):
                _i = _sorted_idx[_rank]
                _bh_adj[_i] = min(_running_min,
                                   _prop_pvals[_i] * _n_tests / (_rank + 1))
                _running_min = _bh_adj[_i]
            informed_trading['METRIC_PVAL_BH'] = np.nan
            informed_trading.loc[_mask_prop, 'METRIC_PVAL_BH'] = _bh_adj

        # ── Event-clustered wild cluster bootstrap ──────────────────
        cut_specs = [
            # (label, subset, null_p0, is_sell)
            # is_sell=True → one-tailed (accuracy > p0); False → one-tailed (accuracy < p0);
            # None → two-tailed
            ('ALL_SELLS',  valid_pol[valid_pol['TRADE_TYPE'] == 'sell'], 0.5,         True),
            # is_sell=True for buys: EVENT_PROFITABLE=(CAR>0), so informed buys
            # imply accuracy > p0, the same direction as sells.
            ('ALL_BUYS',   valid_pol[valid_pol['TRADE_TYPE'] == 'buy'],  0.5,         True),
            ('ALL_TRADES', valid_pol,                                    0.5,         None),
            # Conditional-null rows: correct for base-rate selection.
            # p_cond_buy = P(CAR>=0) = 1 - p_cond_sell; H1: buy accuracy > p_cond_buy.
            ('SELLS_COND', valid_pol[valid_pol['TRADE_TYPE'] == 'sell'], p_cond_sell, True),
            ('BUYS_COND',  valid_pol[valid_pol['TRADE_TYPE'] == 'buy'],  p_cond_buy,  True),
        ]
        # Post-2023 plan/non-plan split: add cluster bootstrap rows if available
        if 'IS_10B5_1_PLAN' in valid_pol.columns and 'TRADE_DATE' in valid_pol.columns:
            _s23 = valid_pol[
                (valid_pol['TRADE_TYPE'] == 'sell') &
                (valid_pol['TRADE_DATE'] >= pd.Timestamp('2023-02-27'))
            ]
            _plan_ci   = _s23[_s23['IS_10B5_1_PLAN'] == True]   # noqa: E712
            _noplan_ci = _s23[_s23['IS_10B5_1_PLAN'] == False]  # noqa: E712
            for _lbl, _sub in [('SELLS_10B5_PLAN', _plan_ci),
                                ('SELLS_NOT_10B5_PLAN', _noplan_ci)]:
                if len(_sub) >= 10:
                    cut_specs.append((_lbl, _sub, 0.5, True))
        for cut_label, sub_df, p0, is_sell in cut_specs:
            if len(sub_df) < 10:
                continue
            sub_valid = sub_df[sub_df['EVENT_PROFITABLE'].notna()].copy()
            if len(sub_valid) < 10:
                continue

            # Event-clustered bootstrap
            ev_boot = _wild_cluster_bootstrap_proportion(
                sub_valid['EVENT_PROFITABLE'].values,
                sub_valid['EVENT_ID'].values,
                p0=p0, n_boot=1000, seed=42,
            )
            # Ticker-clustered bootstrap (robustness)
            tkr_boot = None
            if 'TICKER' in sub_valid.columns:
                tkr_boot = _wild_cluster_bootstrap_proportion(
                    sub_valid['EVENT_PROFITABLE'].values,
                    sub_valid['TICKER'].values,
                    p0=p0, n_boot=1000, seed=43,
                )

            for boot_result, cluster_type in [(ev_boot, 'EVENT'), (tkr_boot, 'TICKER')]:
                if boot_result is None:
                    continue
                # Convert two-tailed bootstrap p to one-tailed for directional H1.
                # sells: H1 = accuracy > p0; buys: H1 = accuracy < p0.
                # Two-tailed bootstrap already uses abs() so p_two = 2*p_one when
                # the observed effect is in the right direction.
                p_two = boot_result['p_value_vs_p0']
                p_hat = boot_result['point_est']
                if is_sell is True:
                    p_clust = p_two / 2 if p_hat > p0 else 1.0 - p_two / 2
                elif is_sell is False:
                    p_clust = p_two / 2 if p_hat < p0 else 1.0 - p_two / 2
                else:
                    p_clust = p_two  # ALL_TRADES: two-tailed
                cluster_rows.append({
                    'CUT': cut_label,
                    'CLUSTER_TYPE': cluster_type,
                    'P0': p0,
                    'CONDITIONAL_NULL': cut_label in ('SELLS_COND', 'BUYS_COND'),
                    'N_OBS': boot_result['n_obs'],
                    'POINT_EST': boot_result['point_est'],
                    'CLUSTER_SE': boot_result['cluster_se'],
                    'CI_LO_95': boot_result['ci_lo_95'],
                    'CI_HI_95': boot_result['ci_hi_95'],
                    'G_RAW': boot_result['g_raw'],
                    'G_EFF': boot_result['g_eff'],
                    'P_VALUE_CLUSTERED': p_clust,
                    'P_VALUE_TWO_TAILED': p_two,
                    'WEIGHTS_USED': boot_result['weights_used'],
                    'NAIVE_BINOMIAL_P': stats.binomtest(
                        int(sub_valid['EVENT_PROFITABLE'].sum()),
                        len(sub_valid), p0
                    ).pvalue if len(sub_valid) > 0 else np.nan,
                })

            # ── Event-level aggregation test (equal-weight, independent) ──────
            # One observation per event = mean sell accuracy in that event.
            # Events are independent by construction; this avoids the inflated
            # variance from large-event domination in the weighted bootstrap.
            # Method: t-test + bootstrap CI across event-level means.
            if 'EVENT_ID' in sub_valid.columns:
                event_acc = (
                    sub_valid.groupby('EVENT_ID')['EVENT_PROFITABLE']
                    .mean().dropna()
                )
                n_ev = len(event_acc)
                if n_ev >= 10:
                    ev_mean = event_acc.mean()
                    ev_se   = event_acc.std(ddof=1) / np.sqrt(n_ev)
                    # One-tailed t-test: H1 = accuracy > p0 for sells, < p0 for buys
                    t_stat, t_two = stats.ttest_1samp(event_acc, p0)
                    if is_sell is True:
                        t_pval = t_two / 2 if t_stat > 0 else 1 - t_two / 2
                    elif is_sell is False:
                        t_pval = t_two / 2 if t_stat < 0 else 1 - t_two / 2
                    else:
                        t_pval = t_two  # two-tailed for mixed
                    # Bootstrap CI on event-level means
                    rng_ev = np.random.RandomState(99)
                    boot_ev = np.array([
                        rng_ev.choice(event_acc.values, size=n_ev, replace=True).mean()
                        for _ in range(2000)
                    ])
                    cluster_rows.append({
                        'CUT': cut_label,
                        'CLUSTER_TYPE': 'EVENT_AGG',
                        'P0': p0,
                        'CONDITIONAL_NULL': (cut_label == 'SELLS_COND'),
                        'N_OBS': n_ev,
                        'POINT_EST': ev_mean,
                        'CLUSTER_SE': ev_se,
                        'CI_LO_95': np.percentile(boot_ev, 2.5),
                        'CI_HI_95': np.percentile(boot_ev, 97.5),
                        'G_RAW': n_ev,
                        'G_EFF': float(n_ev),
                        'P_VALUE_CLUSTERED': t_pval,
                        'WEIGHTS_USED': 'event_ttest',
                        'NAIVE_BINOMIAL_P': np.nan,
                    })

    cluster_inference = pd.DataFrame(cluster_rows) if cluster_rows else pd.DataFrame()

    # ── Table 5: DIRECTIONAL_PLACEBO — date-randomization placebo ────
    # Shuffles EVENT_CAR assignments across events; accuracy should centre
    # near the conditional base rate under H0.
    directional_placebo = _date_randomization_placebo_directional(
        valid_pol, n_perms=1000, seed=42
    ) if len(valid_pol) >= 10 else pd.DataFrame()

    # ── Table 6: CAR_NET_SLOPE — OLS event CAR on net sell direction ──
    # One row per event; base-rate-immune because selection shifts mean CAR
    # not its covariance with what insiders did.
    car_net_slope = pd.DataFrame()
    try:
        from statsmodels.api import OLS, add_constant
        from statsmodels.stats.sandwich_covariance import cov_hc3
        if not valid_pol.empty and 'EVENT_CAR' in valid_pol.columns:
            ev_grp = valid_pol.copy()
            _dollar_col = 'TRADE_VALUE' if 'TRADE_VALUE' in ev_grp.columns else 'SHARES'
            ev_grp['_SELL_USD'] = np.where(
                ev_grp['TRADE_TYPE'] == 'sell',
                ev_grp[_dollar_col].fillna(1.0).abs(),
                0.0)
            ev_grp['_BUY_USD'] = np.where(
                ev_grp['TRADE_TYPE'] == 'buy',
                ev_grp[_dollar_col].fillna(1.0).abs(),
                0.0)
            ev_agg = ev_grp.groupby('EVENT_ID').agg(
                EVENT_CAR=('EVENT_CAR', 'first'),
                NET_SELL_USD=('_SELL_USD', 'sum'),
                NET_BUY_USD=('_BUY_USD', 'sum'),
            ).dropna(subset=['EVENT_CAR']).reset_index()
            ev_agg['NET_DIRECTION'] = np.sign(
                ev_agg['NET_SELL_USD'] - ev_agg['NET_BUY_USD'])
            ev_fit = ev_agg.dropna(subset=['EVENT_CAR', 'NET_DIRECTION'])
            ev_fit = ev_fit[ev_fit['NET_DIRECTION'] != 0]
            if len(ev_fit) >= 10:
                X = add_constant(ev_fit['NET_DIRECTION'].values)
                y = ev_fit['EVENT_CAR'].values
                model = OLS(y, X).fit()
                hc3 = cov_hc3(model)
                se_hc3 = np.sqrt(np.diag(hc3))
                t_stat = model.params[1] / se_hc3[1]
                p_two = 2 * (1 - stats.t.cdf(abs(t_stat), df=len(y) - 2))
                car_net_slope = pd.DataFrame([{
                    'N_EVENTS': len(ev_fit),
                    'COEF_NET_DIRECTION': model.params[1],
                    'SE_HC3': se_hc3[1],
                    'T_STAT': t_stat,
                    'P_VALUE_TWO': p_two,
                    'INTERCEPT': model.params[0],
                    'R_SQUARED': model.rsquared,
                    'INTERPRETATION': (
                        'Negative coef = events where insiders net-sold had '
                        'more negative CARs; positive = net-bought events '
                        'had positive CARs. Independent of base-rate.'),
                }])
                logger.info(
                    "  CAR-on-net-direction: coef=%.4f SE=%.4f p=%.4f (n=%d events)",
                    model.params[1], se_hc3[1], p_two, len(ev_fit))
    except Exception as _e:
        logger.warning("  CAR net slope failed: %s", _e)

    # ── Table 7: JOINT_PT — joint Pesaran-Timmermann 2x2 sign test ───
    # Tests buy+sell direction against independence null.
    # P_null = P(sell)*P(neg_car) + P(buy)*P(pos_car)
    # Uses Pesaran-Timmermann (1992) eq. 7 variance.
    joint_pt = pd.DataFrame()
    try:
        if not valid_pol.empty and 'EVENT_CAR' in valid_pol.columns and 'TRADE_TYPE' in valid_pol.columns:
            # Aggregate to event level: one observation per event.
            # PT (1992) assumes independent predictions; multiple trades from the
            # same event share an identical EVENT_CAR (within-event correlation =1),
            # so the trade-level N severely understates the variance.  Use net
            # direction (majority trade-type by count) as the single prediction
            # per event; exclude tied events (equal buy and sell counts).
            jp_raw = valid_pol[['TRADE_TYPE', 'EVENT_CAR', 'EVENT_ID']].dropna()
            jp_ev = (
                jp_raw.groupby('EVENT_ID')
                .agg(
                    EVENT_CAR=('EVENT_CAR', 'first'),
                    N_SELLS=('TRADE_TYPE', lambda x: (x == 'sell').sum()),
                    N_BUYS=('TRADE_TYPE', lambda x: (x == 'buy').sum()),
                )
                .reset_index()
            )
            # Net direction: +1 = net sell, -1 = net buy, 0 = tie (excluded)
            jp_ev['NET_DIR'] = np.sign(jp_ev['N_SELLS'] - jp_ev['N_BUYS'])
            jp_ev = jp_ev[jp_ev['NET_DIR'] != 0].copy()
            jp_ev['TRADE_DIRECTION'] = np.where(jp_ev['NET_DIR'] > 0, 'sell', 'buy')

            if len(jp_ev) >= 10:
                n_tot = len(jp_ev)
                p_sell = (jp_ev['TRADE_DIRECTION'] == 'sell').mean()
                p_buy = 1.0 - p_sell
                p_neg_car = (jp_ev['EVENT_CAR'] < 0).mean()
                p_pos_car = 1.0 - p_neg_car
                # Joint correct = sell+negCAR OR buy+posCAR
                correct = (
                    ((jp_ev['TRADE_DIRECTION'] == 'sell') & (jp_ev['EVENT_CAR'] < 0)) |
                    ((jp_ev['TRADE_DIRECTION'] == 'buy') & (jp_ev['EVENT_CAR'] > 0))
                )
                obs_acc = correct.mean()
                # Independence null accuracy
                p_star = p_sell * p_neg_car + p_buy * p_pos_car
                # PT (1992) eq. 7 variance
                var_p_star = (
                    (2 * p_neg_car - 1) ** 2 * p_sell * (1 - p_sell) / n_tot +
                    (2 * p_sell - 1) ** 2 * p_neg_car * (1 - p_neg_car) / n_tot +
                    4 * p_sell * p_neg_car * (1 - p_sell) * (1 - p_neg_car) / n_tot ** 2
                )
                var_obs = obs_acc * (1 - obs_acc) / n_tot
                # PT (1992) eq. 7: Cov(P_hat, P*) = Var(P*), so
                # Var(P_hat - P*) = Var(P_hat) + Var(P*) - 2*Cov = Var(P_hat) - Var(P*)
                var_diff = var_obs - var_p_star
                z_pt = (obs_acc - p_star) / np.sqrt(var_diff) if var_diff > 0 else np.nan
                p_pt = 2 * (1 - stats.norm.cdf(abs(z_pt))) if not np.isnan(z_pt) else np.nan
                joint_pt = pd.DataFrame([{
                    'N_EVENTS': n_tot,
                    'N_TRADES_RAW': len(jp_raw),
                    'P_SELL': p_sell,
                    'P_NEG_CAR': p_neg_car,
                    'NULL_ACCURACY': p_star,
                    'OBS_ACCURACY': obs_acc,
                    'EXCESS_ACCURACY': obs_acc - p_star,
                    'PT_Z_STAT': z_pt,
                    'PT_P_VALUE': p_pt,
                    'INTERPRETATION': (
                        'Joint PT (event-level): one obs per event, net trade '
                        'direction vs independence null '
                        '(P_null=P(net-sell)*P(neg_car)+P(net-buy)*P(pos_car)). '
                        'Base-rate-aware two-sided test.'),
                }])
                logger.info(
                    "  Joint PT (event-level): n_ev=%d obs_acc=%.4f null=%.4f "
                    "excess=%.4f z=%.3f p=%.4f",
                    n_tot, obs_acc, p_star, obs_acc - p_star, z_pt, p_pt)
    except Exception as _e:
        logger.warning("  Joint PT failed: %s", _e)

    logger.info("  Informed trading test complete: %d summary rows, "
                "%d proximity rows, %d dollar rows, "
                "%d cluster inference rows, directional placebo: %s, "
                "car_net_slope: %d rows, joint_pt: %d rows",
                len(informed_trading), len(informed_proximity),
                len(informed_dollars), len(cluster_inference),
                'yes' if not directional_placebo.empty else 'no',
                len(car_net_slope), len(joint_pt))

    return (informed_trading, informed_proximity, informed_dollars,
            cluster_inference, directional_placebo, car_net_slope, joint_pt)


# ═════════════════════════════════════════════════════════════════════
# CARRIED-FORWARD ROBUSTNESS
# ═════════════════════════════════════════════════════════════════════

def compute_tost_equivalence(panel):
    """TOST equivalence tests for both categories."""
    if panel.empty:
        return pd.DataFrame()
    df = panel[panel['HAS_SUFFICIENT_DATA'] == 1]
    dv = 'ABNORMAL_NET_TRADING'
    results = []
    for cat in ['CULTURAL', 'FUNDAMENTAL']:
        vals = df.loc[df['EVENT_CATEGORY'] == cat, dv].dropna()
        if len(vals) >= 10:
            t = _compute_tost(vals, SESOI_D)
            t.update({'EVENT_CATEGORY': cat, 'TEST': 'TOST'})
            results.append(t)
    matched_fund = df[(df['EVENT_CATEGORY'] == 'FUNDAMENTAL') &
                      (df['MATCHED'] == True)]  # noqa: E712
    vals = matched_fund[dv].dropna()
    if len(vals) >= 10:
        t = _compute_tost(vals, SESOI_D)
        t.update({'EVENT_CATEGORY': 'FUNDAMENTAL_MATCHED', 'TEST': 'TOST'})
        results.append(t)
    return pd.DataFrame(results)


def compute_placebo_test(panel, n_iterations=500, seed=42):
    """Permutation test: shuffle event category labels."""
    if panel.empty:
        return pd.DataFrame()
    df = panel[panel['HAS_SUFFICIENT_DATA'] == 1]
    dv = 'ABNORMAL_NET_TRADING'
    fund = df.loc[df['EVENT_CATEGORY'] == 'FUNDAMENTAL', dv].dropna()
    cult = df.loc[df['EVENT_CATEGORY'] == 'CULTURAL', dv].dropna()
    if len(fund) < 5 or len(cult) < 5:
        return pd.DataFrame()
    observed = fund.mean() - cult.mean()
    rng = np.random.RandomState(seed)
    all_vals = pd.concat([fund, cult]).values
    n_fund = len(fund)
    null_diffs = np.array([
        (lambda s: s[:n_fund].mean() - s[n_fund:].mean())(rng.permutation(all_vals))
        for _ in range(n_iterations)
    ])
    p_value = (np.abs(null_diffs) >= np.abs(observed)).mean()
    return pd.DataFrame([{
        'TEST': 'PLACEBO_PERMUTATION',
        'OBSERVED_DIFF': observed, 'NULL_MEAN': null_diffs.mean(),
        'NULL_STD': null_diffs.std(), 'P_VALUE': p_value,
        'N_ITERATIONS': n_iterations,
        'CI_2_5': np.percentile(null_diffs, 2.5),
        'CI_97_5': np.percentile(null_diffs, 97.5),
    }])


def compute_bootstrap_ci(panel, n_bootstrap=1000, seed=42):
    """Bootstrap confidence intervals for the fundamental-cultural contrast."""
    if panel.empty:
        return pd.DataFrame()
    df = panel[panel['HAS_SUFFICIENT_DATA'] == 1]
    dv = 'ABNORMAL_NET_TRADING'
    fund = df.loc[df['EVENT_CATEGORY'] == 'FUNDAMENTAL', dv].dropna().values
    cult = df.loc[df['EVENT_CATEGORY'] == 'CULTURAL', dv].dropna().values
    if len(fund) < 5 or len(cult) < 5:
        return pd.DataFrame()
    rng = np.random.RandomState(seed)
    boot_diffs = np.array([
        rng.choice(fund, len(fund), replace=True).mean() -
        rng.choice(cult, len(cult), replace=True).mean()
        for _ in range(n_bootstrap)
    ])
    observed = fund.mean() - cult.mean()
    return pd.DataFrame([{
        'TEST': 'BOOTSTRAP_CI', 'OBSERVED_DIFF': observed,
        'BOOT_MEAN': boot_diffs.mean(), 'BOOT_STD': boot_diffs.std(),
        'CI_2_5': np.percentile(boot_diffs, 2.5),
        'CI_97_5': np.percentile(boot_diffs, 97.5),
        'CI_5': np.percentile(boot_diffs, 5),
        'CI_95': np.percentile(boot_diffs, 95),
        'N_BOOTSTRAP': n_bootstrap,
        'ZERO_IN_95CI': int(np.percentile(boot_diffs, 2.5) <= 0 <= np.percentile(boot_diffs, 97.5)),
    }])


# ═════════════════════════════════════════════════════════════════════
# RESULTS PERSISTENCE
# ═════════════════════════════════════════════════════════════════════

def save_essay3_results(store, results_dict):
    """Save all Essay 3 results to database and drop orphaned tables."""
    table_map = {
        'panel': 'ESSAY3_PANEL',
        'stratification': 'ESSAY3_STRATIFICATION',
        'mean_vs_distributional': 'ESSAY3_MEAN_VS_DISTRIBUTIONAL',
        'wilcoxon_family': 'ESSAY3_WILCOXON_FAMILY',
        'insider_concentration': 'ESSAY3_INSIDER_CONCENTRATION',
        'active_subset': 'ESSAY3_ACTIVE_SUBSET',
        'quantile_regression': 'ESSAY3_QUANTILE_REGRESSION',
        'trimmed_robustness': 'ESSAY3_TRIMMED_ROBUSTNESS',
        'bootstrap_wilcoxon': 'ESSAY3_BOOTSTRAP_WILCOXON',
        'insider_panel': 'ESSAY3_INSIDER_PANEL',
        'concentration_cuts': 'ESSAY3_CONCENTRATION_CUTS',
        'repeat_traders': 'ESSAY3_REPEAT_TRADERS',
        'tost': 'ESSAY3_TOST',
        'placebo': 'ESSAY3_PLACEBO',
        'bootstrap_ci': 'ESSAY3_BOOTSTRAP_CI',
        'crsp_profits': 'ESSAY3_CRSP_PROFITS',
        'crsp_summary': 'ESSAY3_CRSP_SUMMARY',
        'informed_trading': 'ESSAY3_INFORMED_TRADING',
        'informed_proximity': 'ESSAY3_INFORMED_PROXIMITY',
        'informed_dollars': 'ESSAY3_INFORMED_DOLLARS',
        'size_accuracy': 'ESSAY3_SIZE_ACCURACY',
        'size_accuracy_slopes': 'ESSAY3_SIZE_ACCURACY_SLOPES',
        'reversal_regression': 'ESSAY3_REVERSAL_REGRESSION',
        'control_trades': 'ESSAY3_CONTROL_TRADES',
        'control_attrition': 'ESSAY3_CONTROL_ATTRITION',
        'cluster_inference': 'ESSAY3_CLUSTER_INFERENCE',
        'directional_placebo': 'ESSAY3_DIRECTIONAL_PLACEBO',
        'car_net_slope': 'ESSAY3_CAR_NET_SLOPE',
        'joint_pt': 'ESSAY3_JOINT_PT',
    }
    current_tables = set(table_map.values())

    # Drop orphaned ESSAY3_* tables from prior schema versions
    if store._conn is not None:
        try:
            cursor = store._conn.cursor()
            cursor.execute("SELECT name FROM sqlite_master WHERE type='table' AND name LIKE 'ESSAY3_%'")
            existing = {r[0] for r in cursor.fetchall()}
            orphaned = existing - current_tables
            for orphan in sorted(orphaned):
                # Use bracket quoting with escaped brackets; names come from
                # sqlite_master so there is no external injection path, but
                # bracket-escaping is more robust than double-quote f-strings.
                safe_name = orphan.replace(']', ']]')
                cursor.execute(f'DROP TABLE IF EXISTS [{safe_name}]')
                logger.info("  Dropped orphaned table: %s", orphan)
            if orphaned:
                store._conn.commit()
        except Exception as e:
            logger.warning("  Could not clean orphaned tables: %s", e)

    for key, table_name in table_map.items():
        df = results_dict.get(key)
        if df is None or (isinstance(df, pd.DataFrame) and df.empty):
            continue
        if not isinstance(df, pd.DataFrame):
            continue
        try:
            save_df = df.drop(columns=['DESCRIPTION'], errors='ignore') if key == 'panel' else df
            store._loader.write_table(save_df, table_name, replace=True)
            logger.info("  Saved %s: %d rows", table_name, len(save_df))
        except Exception as e:
            logger.warning("  Failed to save %s: %s", table_name, e)


# ═════════════════════════════════════════════════════════════════════
# MAIN RUNNER
# ═════════════════════════════════════════════════════════════════════

def run_essay3(store=None):
    """Run the complete Essay 3 analysis.

    Informed Insider Trading Around Political Decisions: tests whether
    insiders trade on advance knowledge of political decisions that move
    firm equity values. Headline claim (§12b): pre-event sells are ~56%
    directionally accurate vs ~53% in matched non-political windows,
    with billions in aggregate informed-trading profits.

    Supporting evidence: CG size-accuracy attenuation (§12), distributional
    tests (§1-§11), concentration metrics, insider fixed effects.
    """
    logger.info("=" * 60)
    logger.info("  Essay 3: Informed Insider Trading Around Political Decisions")
    logger.info("  Foreknowledge, Profits, and the Limits of Regulatory Architecture")
    logger.info("=" * 60)

    if store is None:
        store = DataStore()

    # Reset module-level caches for clean run
    global _ticker_naics_map
    _ticker_naics_map = None

    # ── Load data ────────────────────────────────────────────────────
    form4 = _load_form4()
    if form4.empty:
        logger.error("No Form 4 data.")
        return
    form4['_trade_type'] = form4.apply(
        lambda r: _classify_trade(r['transaction_code'], r['acquired_disposed']),
        axis=1
    )

    culture_events = store.events
    if culture_events.empty:
        logger.error("No culture war events data.")
        return

    political_events = store.read_table('POLITICAL_EVENTS', parse_dates=['EVENT_DATE'])
    if political_events.empty:
        logger.warning("No political events data.")

    political_exposure = store.read_table('POLITICAL_EXPOSURE')
    car_df = store.read_table('EVENT_STUDY_RESULTS')

    # ── Compute CARs for fundamental political events ─────────────
    car_df = _compute_fundamental_cars(
        car_df, political_events, form4, store
    )

    # ── Build panel ──────────────────────────────────────────────────
    panel = build_insider_panel(
        form4, culture_events, political_events,
        political_exposure, car_df, store=store,
    )
    if panel.empty:
        logger.error("Panel is empty.")
        return

    # ── §1 Mean vs Distributional (core table) ──────────────────────
    logger.info("§1 Mean vs distributional analysis...")
    mean_vs_dist = compute_mean_vs_distributional(panel)

    # ── §2 Family-level Wilcoxon correction (GATING TEST) ───────────
    logger.info("§2 Wilcoxon family correction (gating test)...")
    wilcoxon_family = compute_wilcoxon_family(mean_vs_dist)

    n_bh = wilcoxon_family['BH_SIGNIFICANT'].sum() if not wilcoxon_family.empty else 0
    n_holm = wilcoxon_family['HOLM_SIGNIFICANT'].sum() if not wilcoxon_family.empty else 0
    logger.info("  GATING TEST: %d BH-significant, %d Holm-significant", n_bh, n_holm)

    # ── Pre-compute insider profits once (used by §3, §4, §8, §9, §10, §11) ──
    logger.info("Pre-computing insider-level profits...")
    df_with_data = panel[panel['HAS_SUFFICIENT_DATA'] == 1].copy()
    insider_profits = _compute_insider_profits(df_with_data, form4)

    # ── §3 Insider concentration ────────────────────────────────────
    logger.info("§3 Insider concentration (Cziraki-Gider style)...")
    insider_concentration = compute_insider_concentration(panel, form4, insider_profits=insider_profits)

    # ── §4 Active subset characterization ───────────────────────────
    logger.info("§4 Active subset characterization...")
    active_subset = compute_active_subset(panel, form4, insider_profits=insider_profits)

    # ── §5 Quantile regression ──────────────────────────────────────
    logger.info("§5 Quantile regression...")
    quantile_regression = compute_quantile_regression(panel)

    # ── §6 Trimmed robustness ───────────────────────────────────────
    logger.info("§6 Trimmed robustness (1% and 5% tails)...")
    trimmed_robustness = compute_trimmed_robustness(panel)

    # ── §7 Bootstrap Wilcoxon ───────────────────────────────────────
    logger.info("§7 Bootstrap Wilcoxon inference...")
    bootstrap_wilcoxon = compute_bootstrap_wilcoxon(panel)

    # ── §8 Insider fixed effects ────────────────────────────────────
    logger.info("§8 Insider fixed-effects panel...")
    insider_panel_results = compute_insider_panel(panel, form4, insider_profits=insider_profits)

    # ── §9 Concentration by dimension ───────────────────────────────
    logger.info("§9 Concentration by dimension...")
    concentration_cuts = compute_concentration_cuts(panel, form4, insider_profits=insider_profits)

    # ── §10 Repeat traders ──────────────────────────────────────────
    logger.info("§10 Repeat trader analysis...")
    repeat_traders = compute_repeat_traders(panel, form4, insider_profits=insider_profits)

    # ── §11 CRSP profit analysis ──────────────────────────────────────
    logger.info("§11 CRSP profit analysis (forward-looking abnormal returns)...")
    crsp_profits, crsp_summary = compute_crsp_profits(panel, form4, store, insider_profits=insider_profits)

    # ── §12 CG size-accuracy attenuation (supporting) ─────────────────
    logger.info("§12 CG size-accuracy attenuation (supporting evidence)...")
    # Merge event-CAR onto crsp_profits before the size analysis so that
    # ESSAY3_SIZE_ACCURACY uses the same construct as the headline directional
    # accuracy (EVENT_PROFITABLE) rather than the 30-day forward-CAR
    # (PROFITABLE_30), which straddles the event drop for near-event trades.
    # Lambert review 2026-07-06 §5.
    crsp_profits_with_event_car = crsp_profits.copy()
    if not crsp_profits_with_event_car.empty and 'EVENT_ID' in crsp_profits_with_event_car.columns:
        # Merge on (EVENT_ID, TICKER) so each trade gets the CAR for its own firm.
        # Cultural events share one EVENT_ID across multiple tickers; a plain
        # EVENT_ID merge would fan out.  Fundamental events already encode
        # the ticker in the EVENT_ID, so the (EVENT_ID, TICKER) key is
        # always 1:1 in the panel.
        ev_cars = (panel[['EVENT_ID', 'TICKER', 'CAR_POST']]
                   .rename(columns={'CAR_POST': 'EVENT_CAR'})
                   .drop_duplicates(subset=['EVENT_ID', 'TICKER'])
                   .copy())
        ev_cars['EVENT_CAR'] = pd.to_numeric(ev_cars['EVENT_CAR'], errors='coerce')
        crsp_profits_with_event_car = crsp_profits_with_event_car.merge(
            ev_cars, on=['EVENT_ID', 'TICKER'], how='left')
        has_ec = crsp_profits_with_event_car['EVENT_CAR'].notna()
        crsp_profits_with_event_car['EVENT_PROFITABLE'] = np.nan
        crsp_profits_with_event_car.loc[
            has_ec & (crsp_profits_with_event_car['TRADE_TYPE'] == 'sell'),
            'EVENT_PROFITABLE'
        ] = (crsp_profits_with_event_car.loc[
            has_ec & (crsp_profits_with_event_car['TRADE_TYPE'] == 'sell'),
            'EVENT_CAR'] < 0).astype(float)
        crsp_profits_with_event_car.loc[
            has_ec & (crsp_profits_with_event_car['TRADE_TYPE'] == 'buy'),
            'EVENT_PROFITABLE'
        ] = (crsp_profits_with_event_car.loc[
            has_ec & (crsp_profits_with_event_car['TRADE_TYPE'] == 'buy'),
            'EVENT_CAR'] > 0).astype(float)

    size_accuracy, size_accuracy_slopes, reversal_regression, control_trades, control_attrition = (
        compute_directional_accuracy_reversal(
            panel, form4, store, crsp_profits=crsp_profits_with_event_car)
    )

    # ── §12b Informed trading test (HEADLINE) ─────────────────────────
    logger.info("§12b Informed trading test (headline)...")
    (informed_trading, informed_proximity, informed_dollars,
     cluster_inference, directional_placebo,
     car_net_slope, joint_pt) = (
        compute_informed_trading_test(panel, crsp_profits, control_trades)
    )

    # ── Carried-forward robustness ──────────────────────────────────
    logger.info("Carried-forward robustness (TOST, placebo, bootstrap CI)...")
    tost = compute_tost_equivalence(panel)
    placebo = compute_placebo_test(panel)
    bootstrap_ci = compute_bootstrap_ci(panel)

    # ── Collect results ──────────────────────────────────────────────
    results = {
        'panel': panel,
        'stratification': _build_stratification_summary(panel),
        'mean_vs_distributional': mean_vs_dist,
        'wilcoxon_family': wilcoxon_family,
        'insider_concentration': insider_concentration,
        'active_subset': active_subset,
        'quantile_regression': quantile_regression,
        'trimmed_robustness': trimmed_robustness,
        'bootstrap_wilcoxon': bootstrap_wilcoxon,
        'insider_panel': insider_panel_results,
        'concentration_cuts': concentration_cuts,
        'repeat_traders': repeat_traders,
        'tost': tost,
        'placebo': placebo,
        'bootstrap_ci': bootstrap_ci,
        'crsp_profits': crsp_profits,
        'crsp_summary': crsp_summary,
        'informed_trading': informed_trading,
        'informed_proximity': informed_proximity,
        'informed_dollars': informed_dollars,
        'size_accuracy': size_accuracy,
        'size_accuracy_slopes': size_accuracy_slopes,
        'reversal_regression': reversal_regression,
        'control_trades': control_trades,
        'control_attrition': control_attrition,
        'cluster_inference': cluster_inference,
        'directional_placebo': directional_placebo,
        'car_net_slope': car_net_slope,
        'joint_pt': joint_pt,
    }

    logger.info("Saving results...")
    save_essay3_results(store, results)

    # ── Summary ──────────────────────────────────────────────────────
    _print_summary(panel, results)

    return results


def _build_stratification_summary(panel):
    """Build stratification summary table."""
    rows = []
    for cat in ['CULTURAL', 'FUNDAMENTAL']:
        sub = panel[panel['EVENT_CATEGORY'] == cat]
        matched = sub[sub['MATCHED'] == True]  # noqa: E712
        for year in sorted(panel['EVENT_YEAR'].dropna().unique()):
            for tercile in sorted(panel['ACTIVITY_TERCILE'].unique()):
                stratum = sub[(sub['EVENT_YEAR'] == year) &
                              (sub['ACTIVITY_TERCILE'] == tercile)]
                m_stratum = matched[(matched['EVENT_YEAR'] == year) &
                                    (matched['ACTIVITY_TERCILE'] == tercile)]
                if len(stratum) == 0 and len(m_stratum) == 0:
                    continue
                dv = stratum['ABNORMAL_NET_TRADING'].dropna()
                rows.append({
                    'EVENT_CATEGORY': cat, 'EVENT_YEAR': year,
                    'ACTIVITY_TERCILE': tercile, 'N_EVENTS': len(stratum),
                    'N_MATCHED': len(m_stratum),
                    'MEAN_ABNORMAL': dv.mean() if len(dv) > 0 else np.nan,
                    'MEAN_BENCH_VOLUME': stratum['BENCHMARK_TOTAL_VOLUME'].mean()
                    if 'BENCHMARK_TOTAL_VOLUME' in stratum.columns else np.nan,
                })
    return pd.DataFrame(rows)


def _print_summary(panel, results):
    """Print summary to log."""
    logger.info("=" * 60)
    logger.info("  RESULTS SUMMARY")
    logger.info("=" * 60)

    n_cult = (panel['EVENT_CATEGORY'] == 'CULTURAL').sum()
    n_fund = (panel['EVENT_CATEGORY'] == 'FUNDAMENTAL').sum()
    logger.info("  Full panel: %d events (%d cultural, %d fundamental)",
                len(panel), n_cult, n_fund)

    # Gating test result
    wf = results.get('wilcoxon_family')
    if wf is not None and not wf.empty:
        n_tests = len(wf)
        n_nominal = (wf['WILCOXON_PVALUE'] < 0.05).sum()
        n_holm = wf['HOLM_SIGNIFICANT'].sum()
        n_bh = wf['BH_SIGNIFICANT'].sum()
        n_div = (wf['DIVERGENCE'] == 'YES').sum()
        logger.info("  ── GATING TEST (Wilcoxon Family) ──")
        logger.info("    %d tests | %d nominal p<.05 | %d Holm-sig | %d BH-sig | %d t/W divergent",
                    n_tests, n_nominal, n_holm, n_bh, n_div)

        # Show BH-significant cells
        bh_cells = wf[wf['BH_SIGNIFICANT'] == 1]
        if not bh_cells.empty:
            logger.info("    BH-significant cells:")
            for _, row in bh_cells.iterrows():
                logger.info("      %s/%s: Wilcoxon p=%.4f, t p=%.4f, d=%.3f, N=%d",
                            row['CUT'], row['SUBSET'], row['WILCOXON_PVALUE'],
                            row['T_PVALUE'], row['COHEN_D'], row['N'])

    # Concentration
    ic = results.get('insider_concentration')
    if ic is not None and not ic.empty:
        logger.info("  ── CONCENTRATION ──")
        for _, row in ic.iterrows():
            if row['METRIC'] in ['GINI', 'TOP_5PCT_SHARE', 'TOP_10PCT_SHARE']:
                logger.info("    %s [%s]: %.3f", row['METRIC'],
                            row['EVENT_CATEGORY'], row['VALUE'])

    # Active subset
    act = results.get('active_subset')
    if act is not None and not act.empty:
        logger.info("  ── ACTIVE SUBSET ──")
        for _, row in act.iterrows():
            if row['GROUP'] in ['ACTIVE', 'INACTIVE']:
                logger.info("    %s: %d insider-events, %d unique insiders, "
                            "pct_high_conn=%.2f, pct_high_opp=%.2f, pct_court=%.2f",
                            row['GROUP'], row['N_INSIDER_EVENTS'],
                            row['N_UNIQUE_INSIDERS'],
                            row.get('PCT_HIGH_CONN', 0),
                            row.get('PCT_HIGH_OPP', 0),
                            row.get('PCT_COURT_DECISION', 0))

    # Bootstrap Wilcoxon
    bw = results.get('bootstrap_wilcoxon')
    if bw is not None and not bw.empty:
        logger.info("  ── BOOTSTRAP WILCOXON ──")
        for _, row in bw.iterrows():
            logger.info("    %s: obs_p=%.4f, boot_pct_sig=%.1f%%, boot_pval_median=%.4f",
                        row['CUT'], row['OBSERVED_PVALUE'],
                        row['BOOT_PCT_SIG_005'] * 100, row['BOOT_PVALUE_MEDIAN'])

    # Trimmed robustness
    tr = results.get('trimmed_robustness')
    if tr is not None and not tr.empty:
        logger.info("  ── TRIMMED ROBUSTNESS ──")
        for _, row in tr.iterrows():
            logger.info("    [trim %d%%] %s: t_p=%.4f, W_p=%.4f, N=%d",
                        row['TRIM_PCT'], row['CUT'],
                        row['T_PVALUE'], row['WILCOXON_PVALUE'], row['N'])

    # CRSP profit analysis
    cs = results.get('crsp_summary')
    if cs is not None and not cs.empty:
        logger.info("  ── CRSP PROFIT ANALYSIS ──")
        for _, row in cs.iterrows():
            cut = row['CUT']
            if cut in ('ACTIVE', 'INACTIVE', 'ACTIVE_BUYS', 'ACTIVE_SELLS',
                       'ACTIVE_FUNDAMENTAL'):
                pct_prof = row.get('PCT_PROFITABLE_30', np.nan)
                binom_p = row.get('BINOM_PVAL_30', np.nan)
                mean_car = row.get('MEAN_CAR_30', np.nan)
                logger.info("    %s: N=%d, mean_CAR30=%.4f, pct_profitable=%.1f%%, binom_p=%.4f",
                            cut, row['N_TRADES'],
                            mean_car if not np.isnan(mean_car) else 0,
                            (pct_prof * 100) if not np.isnan(pct_prof) else 0,
                            binom_p if not np.isnan(binom_p) else 1.0)

    # ── INFORMED TRADING TEST (headline) ──────────────────────────────
    it = results.get('informed_trading')
    if it is not None and not it.empty:
        logger.info("  ── INFORMED TRADING TEST (Headline) ──")
        # Main political vs control comparison
        for _, row in it.iterrows():
            if row['CUT'] in ('ALL', 'SELLS_ONLY', 'BUYS_ONLY',
                              'SELLS_PREMIUM', 'SELLS_PREMIUM_TRADE_CAR'):
                sample = row.get('SAMPLE', 'POLITICAL')
                pct = row['METRIC_VALUE']
                if row['CUT'] in ('SELLS_PREMIUM', 'SELLS_PREMIUM_TRADE_CAR'):
                    tag = 'event-CAR' if row['CUT'] == 'SELLS_PREMIUM' else 'trade-CAR'
                    logger.info("    SELLS PREMIUM (%s): +%.2f pp (p=%.4f, N=%d)",
                                tag, pct * 100, row['METRIC_PVAL'], row['N_TRADES'])
                else:
                    logger.info("    [%s] %s: %.2f%% profitable (p=%.2e, N=%d)",
                                sample, row['CUT'], pct * 100,
                                row['METRIC_PVAL'], row['N_TRADES'])
        # Regulatory periods (sells only) — political then control
        for sample_label in ['POLITICAL', 'CONTROL']:
            for _, row in it.iterrows():
                if (row['CUT'].startswith('REG_SELLS:') and
                        row.get('SAMPLE', 'POLITICAL') == sample_label):
                    period = row['CUT'].replace('REG_SELLS:', '')
                    logger.info("    [%s %s] Sells: %.2f%% (N=%d)",
                                sample_label, period,
                                row['METRIC_VALUE'] * 100,
                                row['N_TRADES'])

    ip = results.get('informed_proximity')
    if ip is not None and not ip.empty:
        logger.info("  ── PROXIMITY TO EVENT ──")
        for _, row in ip.iterrows():
            if row['CUT'].endswith('_SELLS'):
                logger.info("    %s: %.2f%% profitable (N=%d)",
                            row['CUT'], row['METRIC_VALUE'] * 100,
                            row['N_TRADES'])

    id_ = results.get('informed_dollars')
    if id_ is not None and not id_.empty:
        logger.info("  ── DOLLAR MAGNITUDES ──")
        for _, row in id_.iterrows():
            if row['CUT'].startswith('SELLS_') and 'CAR' not in row['CUT']:
                logger.info("    %s: N=%d, mean_profit=$%.0f, total=$%.0fM",
                            row['CUT'], row['N_SELLS'],
                            row['MEAN_PROFIT'],
                            row['TOTAL_PROFIT'] / 1e6)
        # Severity cuts (sells near large-CAR events)
        for _, row in id_.iterrows():
            if 'CAR<=' in row['CUT'] and '0-30d' in row['CUT']:
                logger.info("    %s: N=%d, mean_CAR=%.1f%%, mean_profit=$%.0f, "
                            "total=$%.0fM",
                            row['CUT'], row['N_SELLS'],
                            row['MEAN_EVENT_CAR'] * 100,
                            row['MEAN_PROFIT'],
                            row['TOTAL_PROFIT'] / 1e6)
        # Totals
        for _, row in id_.iterrows():
            if row['CUT'] in ('ALL_SELLS_TOTAL', 'CORRECT_SELLS_TOTAL'):
                logger.info("    %s: N=%d, total=$%.0fM",
                            row['CUT'], row['N_SELLS'],
                            row['TOTAL_PROFIT'] / 1e6)

    # ── CG SIZE-ACCURACY ATTENUATION (supporting) ─────────────────────
    sa = results.get('size_accuracy')
    sa_slopes = results.get('size_accuracy_slopes')
    if sa is not None and not sa.empty:
        logger.info("  ── CG SIZE-ACCURACY (Supporting) ──")
        if sa_slopes is not None and not sa_slopes.empty:
            for _, row in sa_slopes.iterrows():
                logger.info("    %s slope(log_tv): %.5f (p=%.4f, R²=%.3f, N=%d)",
                            row['SAMPLE'], row['SLOPE'],
                            row['SLOPE_PVAL'], row['SLOPE_R2'],
                            row['N_TRADES'])

    rr = results.get('reversal_regression')
    if rr is not None and not rr.empty:
        logger.info("  ── LPM REGRESSION (Supporting) ──")
        for _, row in rr.iterrows():
            if row['VARIABLE'] == 'LOG_TV':
                logger.info("    [%s] LOG_TV: β=%.5f (t=%.2f, p=%.4f, N=%d)",
                            row['SPECIFICATION'], row['COEFFICIENT'],
                            row['T_STAT'], row['P_VALUE'], row['N_OBS'])
        for spec in ['POOLED_INTERACTION', 'POOLED_TICKER_FE']:
            interaction = rr[(rr['SPECIFICATION'] == spec) &
                             (rr['VARIABLE'] == 'LOG_TV_x_POL')]
            if not interaction.empty:
                r = interaction.iloc[0]
                logger.info("    [%s] LOG_TV×POL: β=%.5f (t=%.2f, p=%.4f)",
                            spec, r['COEFFICIENT'], r['T_STAT'], r['P_VALUE'])

    logger.info("=" * 60)
