"""Essay 3 — Do Legislators Profit from Legislative Foreknowledge?
Congressional Trading Around Roll-Call Votes Under the STOCK Act.

Research question: Do members of Congress trade in directions that
predict the market impact of votes they subsequently cast?

Core argument (Roseboro 2026):
  Legislators possess advance knowledge of legislative outcomes through
  committee membership, caucus negotiations, and vote-whipping. The
  STOCK Act (2012) requires disclosure but does not prohibit trading.
  We test whether congressional trades in the 90 days before a roll-call
  vote are directionally aligned with the vote's CAR, and whether
  committee members — who see bills first — show higher accuracy.

Methodology:
  1. Match STOCK Act trades to congressional votes via (chamber, date
     proximity, NAICS sector overlap).
  2. Compute CARs around vote dates for traded tickers.
  3. Directional accuracy: did the legislator's trade predict the CAR?
     - Buy before positive CAR → accurate
     - Sell before negative CAR → accurate
  4. Conditional null: accuracy must beat the base rate (share of
     negative CARs for sells, positive for buys), not 50%.
  5. Committee advantage: members of the relevant committee vs others.
  6. Filing latency: late filers may be hiding informed trades.
  7. Dollar magnitude: aggregate informed-trading profits.

Output tables:
  ESSAY3_SA_PANEL               Trade-vote matched panel
  ESSAY3_SA_DIRECTIONAL         Directional accuracy by cut
  ESSAY3_SA_COMMITTEE           Committee vs non-committee accuracy
  ESSAY3_SA_TIMING              Accuracy by trade-to-vote proximity
  ESSAY3_SA_LATENCY             Late vs on-time filer accuracy
  ESSAY3_SA_MAGNITUDE           Dollar magnitudes of informed trades
  ESSAY3_SA_REGULATORY          Pre vs post STOCK Act comparison
  ESSAY3_SA_MEMBER_FE           Member fixed-effects panel
  ESSAY3_SA_PLACEBO             Date-permutation placebo test
  ESSAY3_SA_SUMMARY             One-row headline summary
"""

import logging
import os
import warnings

import numpy as np
import pandas as pd
from scipy import stats

from .datastore import DataStore

logger = logging.getLogger(__name__)

# ── Price data cache (bounded to prevent memory leaks on large runs) ──
_PRICE_CACHE_MAX = 500
_price_cache = {}


def _compute_car_yf(ticker, event_date, pre_window=(-5, -1),
                    post_window=(0, 5)):
    """Compute market-adjusted CAR using yfinance price data.

    Uses a simple market-model: AR = r_stock - r_SPY.
    Downloads and caches price data per ticker.

    Returns
    -------
    dict or None
        {'car_pre', 'car_post', 'car_full', 'n_obs'} or None on failure.
    """
    import yfinance as yf

    start_dl = event_date - pd.Timedelta(days=30)
    end_dl = event_date + pd.Timedelta(days=30)

    # Cache key
    cache_key = (ticker, start_dl.strftime('%Y-%m'), end_dl.strftime('%Y-%m'))

    if len(_price_cache) >= _PRICE_CACHE_MAX:
        # Evict oldest entries when cache is full
        keys_to_drop = list(_price_cache.keys())[:len(_price_cache) // 2]
        for k in keys_to_drop:
            del _price_cache[k]

    if cache_key not in _price_cache:
        try:
            data = yf.download(
                [ticker, 'SPY'],
                start=start_dl - pd.Timedelta(days=10),
                end=end_dl + pd.Timedelta(days=10),
                progress=False,
                auto_adjust=True,
            )
            if data.empty:
                _price_cache[cache_key] = None
            else:
                close = data['Close'] if 'Close' in data.columns else data
                _price_cache[cache_key] = close
        except Exception:
            _price_cache[cache_key] = None

    close = _price_cache[cache_key]
    if close is None:
        return None

    # Get returns
    try:
        if isinstance(close.columns, pd.MultiIndex):
            stock_close = close[(ticker,)].dropna() if (ticker,) in close.columns else close[ticker].dropna()
            spy_close = close[('SPY',)].dropna() if ('SPY',) in close.columns else close['SPY'].dropna()
        else:
            stock_close = close[ticker].dropna()
            spy_close = close['SPY'].dropna()
    except (KeyError, TypeError):
        return None

    stock_ret = stock_close.pct_change().dropna()
    spy_ret = spy_close.pct_change().dropna()

    # Align
    common = stock_ret.index.intersection(spy_ret.index)
    if len(common) < 5:
        return None

    stock_ret = stock_ret.loc[common]
    spy_ret = spy_ret.loc[common]
    ar = stock_ret - spy_ret  # abnormal return (market-adjusted)

    # Compute trading-day offsets from event date
    all_dates = ar.index.normalize()
    event_dt = pd.Timestamp(event_date).normalize()

    # Find event date position (or nearest)
    if event_dt in all_dates:
        event_idx = list(all_dates).index(event_dt)
    else:
        diffs = abs(all_dates - event_dt)
        event_idx = diffs.argmin()

    offsets = pd.Series(range(len(ar)), index=ar.index) - event_idx

    pre_mask = (offsets >= pre_window[0]) & (offsets <= pre_window[1])
    post_mask = (offsets >= post_window[0]) & (offsets <= post_window[1])
    full_mask = (offsets >= pre_window[0]) & (offsets <= post_window[1])

    car_pre = ar[pre_mask].sum() if pre_mask.any() else np.nan
    car_post = ar[post_mask].sum() if post_mask.any() else np.nan
    car_full = ar[full_mask].sum() if full_mask.any() else np.nan

    return {
        'car_pre': car_pre,
        'car_post': car_post,
        'car_full': car_full,
        'n_obs': int(full_mask.sum()),
    }


# ── Constants ─────────────────────────────────────────────────────────

# Trade-to-vote matching windows (calendar days before vote)
MATCHING_WINDOWS = {
    'NEAR':  (1, 30),
    'MID':   (31, 60),
    'FAR':   (61, 90),
    'FULL':  (1, 90),
}

# CAR event windows (trading days around vote date)
PRE_VOTE_WINDOW = (-5, -1)
POST_VOTE_WINDOW = (0, 5)

# STOCK Act effective date
STOCK_ACT_DATE = pd.Timestamp('2012-04-04')



# ═════════════════════════════════════════════════════════════════════
# §1  BUILD TRADE-VOTE PANEL
# ═════════════════════════════════════════════════════════════════════

def _load_stock_act_trades():
    """Load STOCK Act trades from the clean module."""
    from clean.stock_act import load_stock_act_trades
    df = load_stock_act_trades(source='combined', cache=True)
    if df.empty:
        logger.error("No STOCK Act trades loaded.")
        return df

    # Keep equity trades only
    if 'IS_EQUITY' in df.columns:
        df = df[df['IS_EQUITY'] == True].copy()  # noqa: E712

    # Require a ticker and transaction date
    df = df.dropna(subset=['TICKER', 'TRANSACTION_DATE'])
    df['TRANSACTION_DATE'] = pd.to_datetime(df['TRANSACTION_DATE'],
                                            errors='coerce')
    df = df.dropna(subset=['TRANSACTION_DATE'])

    # Standardize trade type
    if 'TRADE_TYPE_CLEAN' in df.columns:
        df['DIRECTION'] = df['TRADE_TYPE_CLEAN'].map(
            {'buy': 'BUY', 'sell': 'SELL'})
        df = df.dropna(subset=['DIRECTION'])

    logger.info("STOCK Act trades: %d equity trades with tickers, %d members",
                len(df), df['MEMBER'].nunique())
    return df



def build_trade_vote_panel(trades, votes, store, max_gap_days=90):
    """Match legislator trades to congressional votes.

    A trade is matched to a vote when:
      1. The trade occurs 1-90 calendar days before the vote.
      2. The legislator's chamber matches the vote chamber.

    Following Ziobrowski et al. (2004), we do NOT filter by sector —
    legislators may trade any stock in anticipation of market-wide
    effects of legislation.  Each (trade, vote) pair is deduplicated
    to keep only the closest upcoming vote per (member, ticker, trade).

    Parameters
    ----------
    trades : pd.DataFrame
        STOCK Act trades from clean.stock_act.
    votes : pd.DataFrame
        Congressional votes from POLITICAL_EVENTS.
    store : DataStore
        For CAR computation (price data).
    max_gap_days : int
        Maximum calendar days between trade and vote.

    Returns
    -------
    pd.DataFrame
        Panel with one row per (trade, vote) match.
    """
    # Prepare votes
    votes = votes.copy()
    votes['EVENT_DATE'] = pd.to_datetime(votes['EVENT_DATE'], errors='coerce')
    votes = votes.dropna(subset=['EVENT_DATE'])

    # Fix non-unique EVENT_IDs by creating synthetic unique IDs
    if votes['EVENT_ID'].nunique() < len(votes):
        votes = votes.reset_index(drop=True)
        votes['EVENT_ID'] = (votes['EVENT_ID'].astype(str) + '_'
                             + votes.index.astype(str))

    logger.info("Matching %d trades to %d votes...",
                len(trades), len(votes))

    matches = []

    # Group trades by chamber for faster matching
    for chamber in ['Senate', 'House']:
        chamber_trades = trades[trades['CHAMBER'] == chamber]
        chamber_votes = votes[votes['CHAMBER'] == chamber]

        if chamber_trades.empty or chamber_votes.empty:
            continue

        chamber_votes = chamber_votes.sort_values('EVENT_DATE').reset_index(
            drop=True)

        for _, trade in chamber_trades.iterrows():
            t_date = trade['TRANSACTION_DATE']
            ticker = trade['TICKER']

            # Find votes in [t_date + 1 day, t_date + max_gap_days]
            window_start = t_date + pd.Timedelta(days=1)
            window_end = t_date + pd.Timedelta(days=max_gap_days)

            mask = ((chamber_votes['EVENT_DATE'] >= window_start) &
                    (chamber_votes['EVENT_DATE'] <= window_end))
            candidate_votes = chamber_votes[mask]

            for _, vote in candidate_votes.iterrows():
                gap_days = (vote['EVENT_DATE'] - t_date).days

                if 1 <= gap_days <= 30:
                    proximity = 'NEAR'
                elif 31 <= gap_days <= 60:
                    proximity = 'MID'
                elif 61 <= gap_days <= 90:
                    proximity = 'FAR'
                else:
                    continue

                matches.append({
                    'MEMBER': trade['MEMBER'],
                    'CHAMBER': chamber,
                    'PARTY': trade.get('PARTY', ''),
                    'STATE': trade.get('STATE', ''),
                    'TICKER': ticker,
                    'DIRECTION': trade['DIRECTION'],
                    'TRADE_DATE': t_date,
                    'AMOUNT_MIDPOINT': trade.get('AMOUNT_MIDPOINT', np.nan),
                    'OWNER': trade.get('OWNER', 'Self'),
                    'IS_LATE_FILING': trade.get('IS_LATE_FILING', False),
                    'FILING_LATENCY_DAYS': trade.get('FILING_LATENCY_DAYS',
                                                     np.nan),
                    'VOTE_DATE': vote['EVENT_DATE'],
                    'VOTE_ID': vote['EVENT_ID'],
                    'BILL_NUMBER': vote.get('BILL_NUMBER', ''),
                    'POLICY_AREA': vote.get('POLICY_AREA', 'unknown'),
                    'VOTE_RESULT': vote.get('RESULT', ''),
                    'IS_CLOSE_VOTE': vote.get('IS_CLOSE_VOTE', 0),
                    'GAP_DAYS': gap_days,
                    'PROXIMITY': proximity,
                })

    panel = pd.DataFrame(matches)
    if panel.empty:
        logger.warning("No trade-vote matches found.")
        return panel

    logger.info("Raw matches: %d (trade, vote) pairs, %d unique members",
                len(panel), panel['MEMBER'].nunique())

    # Deduplicate: keep closest vote for each (member, ticker, trade_date)
    panel = (panel.sort_values('GAP_DAYS')
             .drop_duplicates(subset=['MEMBER', 'TICKER', 'TRADE_DATE'],
                              keep='first')
             .reset_index(drop=True))

    logger.info("After dedup (closest vote): %d pairs", len(panel))

    # ── Compute CARs ──────────────────────────────────────────────────
    unique_pairs = panel[['TICKER', 'VOTE_ID', 'VOTE_DATE']].drop_duplicates(
        subset=['TICKER', 'VOTE_ID'])
    logger.info("Computing CARs for %d unique (ticker, vote) pairs...",
                len(unique_pairs))

    car_cache = {}
    n_car_computed = 0
    n_car_failed = 0

    for _, row in unique_pairs.iterrows():
        cache_key = (row['TICKER'], row['VOTE_ID'])
        car = _compute_car_yf(
            row['TICKER'], row['VOTE_DATE'],
            pre_window=PRE_VOTE_WINDOW,
            post_window=POST_VOTE_WINDOW,
        )
        car_cache[cache_key] = car
        if car is not None:
            n_car_computed += 1
        else:
            n_car_failed += 1

        if (n_car_computed + n_car_failed) % 100 == 0:
            logger.info("  CARs: %d computed, %d failed",
                        n_car_computed, n_car_failed)

    car_results = []
    for key, car in car_cache.items():
        if car is not None:
            car_results.append({
                'TICKER': key[0],
                'VOTE_ID': key[1],
                'CAR_PRE': car['car_pre'],
                'CAR_POST': car['car_post'],
                'CAR_FULL': car['car_full'],
                'N_EVENT_OBS': car['n_obs'],
            })

    logger.info("CARs: %d computed, %d failed", n_car_computed, n_car_failed)

    if car_results:
        car_df = pd.DataFrame(car_results).drop_duplicates(
            subset=['TICKER', 'VOTE_ID'])
        panel = panel.merge(car_df, on=['TICKER', 'VOTE_ID'], how='left')
    else:
        panel['CAR_PRE'] = np.nan
        panel['CAR_POST'] = np.nan
        panel['CAR_FULL'] = np.nan
        panel['N_EVENT_OBS'] = 0

    # ── Directional accuracy ──────────────────────────────────────────
    has_car = panel['CAR_POST'].notna()
    panel['ACCURATE'] = np.nan

    # Buy before positive CAR → accurate
    buy_mask = has_car & (panel['DIRECTION'] == 'BUY')
    panel.loc[buy_mask, 'ACCURATE'] = (
        panel.loc[buy_mask, 'CAR_POST'] > 0).astype(float)

    # Sell before negative CAR → accurate
    sell_mask = has_car & (panel['DIRECTION'] == 'SELL')
    panel.loc[sell_mask, 'ACCURATE'] = (
        panel.loc[sell_mask, 'CAR_POST'] < 0).astype(float)

    panel['PRE_STOCK_ACT'] = (panel['TRADE_DATE'] < STOCK_ACT_DATE).astype(int)

    n_with_car = has_car.sum()
    n_accurate = panel['ACCURATE'].sum()
    accuracy = n_accurate / n_with_car if n_with_car > 0 else np.nan

    logger.info("Panel: %d matched pairs, %d with CARs, accuracy=%.3f",
                len(panel), n_with_car, accuracy)

    return panel


# ═════════════════════════════════════════════════════════════════════
# §2  DIRECTIONAL ACCURACY TESTS
# ═════════════════════════════════════════════════════════════════════

def _binomial_test(n_accurate, n_total, null_rate=0.5):
    """Two-sided binomial test against a null rate."""
    if n_total == 0:
        return {'accuracy': np.nan, 'n': 0, 'p_value': np.nan,
                'null_rate': null_rate, 'excess': np.nan}
    accuracy = n_accurate / n_total
    p = stats.binomtest(int(n_accurate), int(n_total), null_rate,
                        alternative='greater').pvalue
    return {
        'accuracy': accuracy,
        'n': n_total,
        'p_value': p,
        'null_rate': null_rate,
        'excess': accuracy - null_rate,
    }


def compute_directional_accuracy(panel):
    """Test directional accuracy across multiple cuts.

    Tests against both 50% null and the conditional null (base rate
    of negative CARs for sells, positive CARs for buys).

    Returns
    -------
    pd.DataFrame
        One row per (cut, subset) combination with accuracy, p-values.
    """
    valid = panel[panel['ACCURATE'].notna()].copy()
    if valid.empty:
        return pd.DataFrame()

    # Compute conditional nulls
    sell_base = (valid.loc[valid['DIRECTION'] == 'SELL', 'CAR_POST'] < 0).mean()
    buy_base = (valid.loc[valid['DIRECTION'] == 'BUY', 'CAR_POST'] > 0).mean()

    rows = []

    def _add_row(cut, subset, data, null_50=0.5, cond_null=None):
        n_acc = data['ACCURATE'].sum()
        n_tot = len(data)
        r50 = _binomial_test(n_acc, n_tot, 0.5)
        rc = _binomial_test(n_acc, n_tot, cond_null) if cond_null else {
            'p_value': np.nan, 'excess': np.nan}
        rows.append({
            'CUT': cut,
            'SUBSET': subset,
            'N': n_tot,
            'N_ACCURATE': int(n_acc),
            'ACCURACY': r50['accuracy'],
            'P_VALUE_50': r50['p_value'],
            'COND_NULL': cond_null if cond_null else np.nan,
            'P_VALUE_COND': rc['p_value'],
            'EXCESS_VS_COND': rc['excess'],
        })

    # Overall
    _add_row('ALL', 'ALL', valid, cond_null=None)

    # By direction
    for d in ['BUY', 'SELL']:
        sub = valid[valid['DIRECTION'] == d]
        cn = buy_base if d == 'BUY' else sell_base
        _add_row('DIRECTION', d, sub, cond_null=cn)

    # By chamber
    for ch in ['Senate', 'House']:
        sub = valid[valid['CHAMBER'] == ch]
        _add_row('CHAMBER', ch, sub)

    # By party
    for party in valid['PARTY'].dropna().unique():
        sub = valid[valid['PARTY'] == party]
        if len(sub) >= 20:
            _add_row('PARTY', party, sub)

    # By proximity
    for prox in ['NEAR', 'MID', 'FAR']:
        sub = valid[valid['PROXIMITY'] == prox]
        _add_row('PROXIMITY', prox, sub)

    # By policy area
    for pa in valid['POLICY_AREA'].value_counts().head(8).index:
        sub = valid[valid['POLICY_AREA'] == pa]
        if len(sub) >= 20:
            _add_row('POLICY_AREA', pa, sub)

    # Close votes only
    close = valid[valid['IS_CLOSE_VOTE'] == 1]
    if len(close) >= 20:
        _add_row('VOTE_TYPE', 'CLOSE_VOTE', close)

    # By direction × proximity (key interaction)
    for d in ['BUY', 'SELL']:
        cn = buy_base if d == 'BUY' else sell_base
        for prox in ['NEAR', 'MID', 'FAR']:
            sub = valid[(valid['DIRECTION'] == d) &
                        (valid['PROXIMITY'] == prox)]
            if len(sub) >= 10:
                _add_row('DIR_x_PROX', f'{d}_{prox}', sub, cond_null=cn)

    result = pd.DataFrame(rows)
    logger.info("Directional accuracy: %d test rows", len(result))
    return result


# ═════════════════════════════════════════════════════════════════════
# §3  COMMITTEE ADVANTAGE
# ═════════════════════════════════════════════════════════════════════

def compute_committee_advantage(panel):
    """Test accuracy by policy area as a proxy for committee relevance.

    Legislators on relevant committees (e.g., Finance committee members
    trading financial stocks before finance votes) should show higher
    accuracy. We use policy area as a rough proxy.

    Returns
    -------
    pd.DataFrame
    """
    valid = panel[panel['ACCURATE'].notna()].copy()
    if valid.empty:
        return pd.DataFrame()

    rows = []
    # Compare accuracy across policy areas
    for pa in valid['POLICY_AREA'].value_counts().head(8).index:
        sub = valid[valid['POLICY_AREA'] == pa]
        if len(sub) >= 10:
            acc = sub['ACCURATE'].mean()
            p = stats.binomtest(int(sub['ACCURATE'].sum()), len(sub), 0.5,
                                alternative='greater').pvalue
            rows.append({
                'GROUP': f'POLICY_{pa.upper()}',
                'N': len(sub),
                'ACCURACY': acc,
                'P_VALUE': p,
            })

    # Close vs non-close votes (close votes = more uncertainty = more scope
    # for insider advantage)
    for label, sub in [
        ('CLOSE_VOTE', valid[valid['IS_CLOSE_VOTE'] == 1]),
        ('NOT_CLOSE', valid[valid['IS_CLOSE_VOTE'] == 0]),
    ]:
        if len(sub) >= 10:
            rows.append({
                'GROUP': label,
                'N': len(sub),
                'ACCURACY': sub['ACCURATE'].mean(),
                'P_VALUE': stats.binomtest(
                    int(sub['ACCURATE'].sum()), len(sub), 0.5,
                    alternative='greater').pvalue,
            })

    return pd.DataFrame(rows)


# ═════════════════════════════════════════════════════════════════════
# §4  FILING LATENCY ANALYSIS
# ═════════════════════════════════════════════════════════════════════

def compute_filing_latency(panel):
    """Test whether late filers show higher accuracy (hiding trades).

    The STOCK Act requires filing within 45 days. Late filers may be
    strategically delaying disclosure of informed trades.

    Returns
    -------
    pd.DataFrame
    """
    valid = panel[panel['ACCURATE'].notna()].copy()
    if valid.empty:
        return pd.DataFrame()

    rows = []

    # Late vs on-time
    if 'IS_LATE_FILING' in valid.columns:
        for late_val, label in [(True, 'LATE'), (False, 'ON_TIME')]:
            sub = valid[valid['IS_LATE_FILING'] == late_val]
            if len(sub) >= 10:
                acc = sub['ACCURATE'].mean()
                p = stats.binomtest(int(sub['ACCURATE'].sum()), len(sub),
                                    0.5, alternative='greater').pvalue
                rows.append({
                    'GROUP': label, 'N': len(sub), 'ACCURACY': acc,
                    'P_VALUE': p,
                    'MEAN_LATENCY': sub['FILING_LATENCY_DAYS'].mean(),
                })

    # Latency quartiles
    if 'FILING_LATENCY_DAYS' in valid.columns:
        latency = valid['FILING_LATENCY_DAYS'].dropna()
        if len(latency) >= 40:
            valid['LATENCY_Q'] = pd.qcut(valid['FILING_LATENCY_DAYS'],
                                         4, labels=['Q1', 'Q2', 'Q3', 'Q4'],
                                         duplicates='drop')
            for q in ['Q1', 'Q2', 'Q3', 'Q4']:
                sub = valid[valid['LATENCY_Q'] == q]
                if len(sub) >= 5:
                    rows.append({
                        'GROUP': f'LATENCY_{q}',
                        'N': len(sub),
                        'ACCURACY': sub['ACCURATE'].mean(),
                        'P_VALUE': stats.binomtest(
                            int(sub['ACCURATE'].sum()), len(sub), 0.5,
                            alternative='greater').pvalue,
                        'MEAN_LATENCY': sub['FILING_LATENCY_DAYS'].mean(),
                    })

    return pd.DataFrame(rows)


# ═════════════════════════════════════════════════════════════════════
# §5  DOLLAR MAGNITUDE
# ═════════════════════════════════════════════════════════════════════

def compute_dollar_magnitude(panel):
    """Estimate aggregate informed-trading profits.

    Uses AMOUNT_MIDPOINT × |CAR| as a proxy for profit on each trade.
    Only counts trades where direction was accurate.

    Returns
    -------
    pd.DataFrame
    """
    valid = panel[panel['ACCURATE'].notna() &
                  panel['AMOUNT_MIDPOINT'].notna()].copy()
    if valid.empty:
        return pd.DataFrame()

    valid['EST_PROFIT'] = np.where(
        valid['ACCURATE'] == 1,
        valid['AMOUNT_MIDPOINT'] * valid['CAR_POST'].abs(),
        -valid['AMOUNT_MIDPOINT'] * valid['CAR_POST'].abs()
    )

    rows = []
    for cut, label, sub in [
        ('ALL', 'ALL', valid),
        ('DIRECTION', 'BUY', valid[valid['DIRECTION'] == 'BUY']),
        ('DIRECTION', 'SELL', valid[valid['DIRECTION'] == 'SELL']),
        ('PROXIMITY', 'NEAR', valid[valid['PROXIMITY'] == 'NEAR']),
        ('PROXIMITY', 'MID', valid[valid['PROXIMITY'] == 'MID']),
        ('PROXIMITY', 'FAR', valid[valid['PROXIMITY'] == 'FAR']),
    ]:
        if len(sub) < 5:
            continue
        rows.append({
            'CUT': cut,
            'SUBSET': label,
            'N_TRADES': len(sub),
            'TOTAL_NOTIONAL': sub['AMOUNT_MIDPOINT'].sum(),
            'TOTAL_EST_PROFIT': sub['EST_PROFIT'].sum(),
            'MEAN_EST_PROFIT': sub['EST_PROFIT'].mean(),
            'MEDIAN_EST_PROFIT': sub['EST_PROFIT'].median(),
            'FRAC_ACCURATE': sub['ACCURATE'].mean(),
        })

    return pd.DataFrame(rows)


# ═════════════════════════════════════════════════════════════════════
# §6  REGULATORY PERIOD COMPARISON
# ═════════════════════════════════════════════════════════════════════

def compute_regulatory_comparison(panel):
    """Compare accuracy before vs after the STOCK Act (2012).

    Returns
    -------
    pd.DataFrame
    """
    valid = panel[panel['ACCURATE'].notna()].copy()
    if valid.empty:
        return pd.DataFrame()

    rows = []
    for label, sub in [
        ('PRE_STOCK_ACT', valid[valid['PRE_STOCK_ACT'] == 1]),
        ('POST_STOCK_ACT', valid[valid['PRE_STOCK_ACT'] == 0]),
    ]:
        if len(sub) >= 10:
            acc = sub['ACCURATE'].mean()
            p = stats.binomtest(int(sub['ACCURATE'].sum()), len(sub),
                                0.5, alternative='greater').pvalue
            rows.append({
                'PERIOD': label,
                'N': len(sub),
                'ACCURACY': acc,
                'P_VALUE': p,
                'MEAN_CAR': sub['CAR_POST'].mean(),
            })

    return pd.DataFrame(rows)


# ═════════════════════════════════════════════════════════════════════
# §7  MEMBER FIXED EFFECTS
# ═════════════════════════════════════════════════════════════════════

def compute_member_fixed_effects(panel):
    """Within-member variation in accuracy using LPM with member FE.

    Pr(ACCURATE) = alpha_i + beta_1 * NEAR + beta_2 * SELL
                 + beta_3 * CLOSE_VOTE + beta_4 * log(AMOUNT)
                 + epsilon

    Returns
    -------
    pd.DataFrame
        Regression results.
    """
    import statsmodels.api as sm

    valid = panel[panel['ACCURATE'].notna()].copy()
    if valid.empty or valid['MEMBER'].nunique() < 5:
        return pd.DataFrame()

    # Build regressors
    valid['IS_NEAR'] = (valid['PROXIMITY'] == 'NEAR').astype(int)
    valid['IS_SELL'] = (valid['DIRECTION'] == 'SELL').astype(int)
    valid['IS_CLOSE'] = valid['IS_CLOSE_VOTE'].fillna(0).astype(int)
    valid['LOG_AMOUNT'] = np.log1p(valid['AMOUNT_MIDPOINT'].fillna(0))

    # Member dummies (absorb fixed effects)
    member_dummies = pd.get_dummies(valid['MEMBER'], prefix='FE',
                                    drop_first=True, dtype=float)

    X = pd.concat([
        valid[['IS_NEAR', 'IS_SELL', 'IS_CLOSE', 'LOG_AMOUNT']],
        member_dummies
    ], axis=1)
    X = sm.add_constant(X)
    y = valid['ACCURATE']

    # Drop any remaining NaN
    mask = X.notna().all(axis=1) & y.notna()
    X, y = X[mask], y[mask]

    if len(y) < X.shape[1] + 5:
        logger.warning("Not enough observations for member FE regression.")
        return pd.DataFrame()

    try:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            model = sm.OLS(y, X).fit(cov_type='HC1')

        # Extract key coefficients (not FE dummies)
        key_vars = ['const', 'IS_NEAR', 'IS_SELL', 'IS_CLOSE', 'LOG_AMOUNT']
        rows = []
        for var in key_vars:
            if var in model.params.index:
                rows.append({
                    'VARIABLE': var,
                    'COEF': model.params[var],
                    'SE': model.bse[var],
                    'T_STAT': model.tvalues[var],
                    'P_VALUE': model.pvalues[var],
                })

        result = pd.DataFrame(rows)
        result['N_OBS'] = int(model.nobs)
        result['N_MEMBERS'] = valid['MEMBER'].nunique()
        result['R_SQUARED'] = model.rsquared
        return result

    except Exception as e:
        logger.warning("Member FE regression failed: %s", e)
        return pd.DataFrame()


# ═════════════════════════════════════════════════════════════════════
# §8  PLACEBO TEST
# ═════════════════════════════════════════════════════════════════════

def compute_placebo(panel, n_permutations=1000, seed=42):
    """Direction-permutation placebo: shuffle trade directions, re-compute accuracy.

    If accuracy is driven by genuine foreknowledge, randomly assigning
    BUY/SELL directions should produce ~50% accuracy (base rate).
    We permute the DIRECTION column and recompute accuracy each iteration
    to build a null distribution.

    Returns
    -------
    pd.DataFrame
        Permutation distribution summary.
    """
    valid = panel[panel['ACCURATE'].notna()].copy()
    if len(valid) < 20:
        return pd.DataFrame()

    observed_accuracy = valid['ACCURATE'].mean()
    rng = np.random.RandomState(seed)

    # Pre-compute post-event CARs for reuse across permutations
    car_post = valid['CAR_POST'].values
    directions = valid['DIRECTION'].values

    null_accuracies = []
    for _ in range(n_permutations):
        # Shuffle trade directions to break any foreknowledge signal
        shuffled_dir = rng.permutation(directions)
        perm_accurate = np.where(
            shuffled_dir == 'BUY', (car_post > 0).astype(float),
            np.where(shuffled_dir == 'SELL', (car_post < 0).astype(float), np.nan)
        )
        null_accuracies.append(np.nanmean(perm_accurate))

    null_arr = np.array(null_accuracies)
    p_value = (null_arr >= observed_accuracy).mean()

    return pd.DataFrame([{
        'OBSERVED_ACCURACY': observed_accuracy,
        'NULL_MEAN': null_arr.mean(),
        'NULL_STD': null_arr.std(),
        'NULL_P05': np.percentile(null_arr, 5),
        'NULL_P95': np.percentile(null_arr, 95),
        'P_VALUE': p_value,
        'N_PERMUTATIONS': n_permutations,
        'N_TRADES': len(valid),
    }])


# ═════════════════════════════════════════════════════════════════════
# ORCHESTRATION
# ═════════════════════════════════════════════════════════════════════

def save_results(store, results):
    """Save all Essay 3 STOCK Act results to the datastore."""
    table_map = {
        'panel':        'ESSAY3_SA_PANEL',
        'directional':  'ESSAY3_SA_DIRECTIONAL',
        'committee':    'ESSAY3_SA_COMMITTEE',
        'timing':       'ESSAY3_SA_TIMING',
        'latency':      'ESSAY3_SA_LATENCY',
        'magnitude':    'ESSAY3_SA_MAGNITUDE',
        'regulatory':   'ESSAY3_SA_REGULATORY',
        'member_fe':    'ESSAY3_SA_MEMBER_FE',
        'placebo':      'ESSAY3_SA_PLACEBO',
        'summary':      'ESSAY3_SA_SUMMARY',
    }
    for key, table_name in table_map.items():
        df = results.get(key)
        if df is not None and not df.empty:
            store.write_table(df, table_name)
            logger.info("  Saved %s: %d rows", table_name, len(df))


def run_essay3_stock_act(store=None):
    """Run the complete Essay 3 STOCK Act analysis.

    Do Legislators Profit from Legislative Foreknowledge?
    Congressional Trading Around Roll-Call Votes Under the STOCK Act.
    """
    logger.info("=" * 60)
    logger.info("  Essay 3: Congressional Trading Under the STOCK Act")
    logger.info("  Do Legislators Profit from Legislative Foreknowledge?")
    logger.info("=" * 60)

    if store is None:
        store = DataStore()

    # ── Load data ────────────────────────────────────────────────────
    trades = _load_stock_act_trades()
    if trades.empty:
        logger.error("No STOCK Act trades. Aborting.")
        return None

    political_events = store.read_table('POLITICAL_EVENTS',
                                        parse_dates=['EVENT_DATE'])
    votes = political_events[
        political_events['EVENT_TYPE'] == 'CONGRESSIONAL_VOTE'
    ].copy()
    logger.info("Congressional votes: %d", len(votes))

    # ── §1 Build panel ───────────────────────────────────────────────
    logger.info("§1 Building trade-vote panel...")
    panel = build_trade_vote_panel(trades, votes, store)
    if panel.empty:
        logger.error("Empty panel. Aborting.")
        return None

    # ── §2 Directional accuracy ──────────────────────────────────────
    logger.info("§2 Directional accuracy tests...")
    directional = compute_directional_accuracy(panel)

    # ── §3 Committee advantage ───────────────────────────────────────
    logger.info("§3 Committee / sector advantage...")
    committee = compute_committee_advantage(panel)

    # ── §4 Filing latency ────────────────────────────────────────────
    logger.info("§4 Filing latency analysis...")
    latency = compute_filing_latency(panel)

    # ── §5 Dollar magnitude ──────────────────────────────────────────
    logger.info("§5 Dollar magnitude estimates...")
    magnitude = compute_dollar_magnitude(panel)

    # ── §6 Regulatory comparison ─────────────────────────────────────
    logger.info("§6 Pre/post STOCK Act comparison...")
    regulatory = compute_regulatory_comparison(panel)

    # ── §7 Member fixed effects ──────────────────────────────────────
    logger.info("§7 Member fixed-effects regression...")
    member_fe = compute_member_fixed_effects(panel)

    # ── §8 Placebo ───────────────────────────────────────────────────
    logger.info("§8 Placebo permutation test...")
    placebo = compute_placebo(panel)

    # ── Timing (extract from directional for separate table) ─────────
    timing = directional[directional['CUT'].isin(
        ['PROXIMITY', 'DIR_x_PROX'])].copy() if not directional.empty else pd.DataFrame()

    # ── Summary ──────────────────────────────────────────────────────
    valid = panel[panel['ACCURATE'].notna()]
    summary = pd.DataFrame([{
        'N_TRADES_RAW': len(trades),
        'N_MEMBERS': trades['MEMBER'].nunique(),
        'N_MATCHED_PAIRS': len(panel),
        'N_WITH_CAR': len(valid),
        'N_UNIQUE_VOTES': panel['VOTE_ID'].nunique(),
        'N_UNIQUE_TICKERS': panel['TICKER'].nunique(),
        'OVERALL_ACCURACY': valid['ACCURATE'].mean() if len(valid) > 0 else np.nan,
        'BUY_ACCURACY': (valid.loc[valid['DIRECTION'] == 'BUY', 'ACCURATE'].mean()
                         if (valid['DIRECTION'] == 'BUY').any() else np.nan),
        'SELL_ACCURACY': (valid.loc[valid['DIRECTION'] == 'SELL', 'ACCURATE'].mean()
                          if (valid['DIRECTION'] == 'SELL').any() else np.nan),
        'SELL_BASE_RATE': ((valid.loc[valid['DIRECTION'] == 'SELL', 'CAR_POST'] < 0).mean()
                           if (valid['DIRECTION'] == 'SELL').any() else np.nan),
        'BUY_BASE_RATE': ((valid.loc[valid['DIRECTION'] == 'BUY', 'CAR_POST'] > 0).mean()
                          if (valid['DIRECTION'] == 'BUY').any() else np.nan),
        'MEAN_GAP_DAYS': panel['GAP_DAYS'].mean(),
        'MEAN_CAR_POST': valid['CAR_POST'].mean() if len(valid) > 0 else np.nan,
        'DATE_RANGE_START': str(panel['TRADE_DATE'].min().date()),
        'DATE_RANGE_END': str(panel['TRADE_DATE'].max().date()),
    }])

    # ── Print summary ────────────────────────────────────────────────
    logger.info("=" * 60)
    logger.info("  RESULTS SUMMARY")
    logger.info("=" * 60)
    if not summary.empty:
        s = summary.iloc[0]
        logger.info("  Trades: %d raw → %d matched → %d with CARs",
                    int(s['N_TRADES_RAW']), int(s['N_MATCHED_PAIRS']),
                    int(s['N_WITH_CAR']))
        logger.info("  Members: %d, Votes: %d, Tickers: %d",
                    int(s['N_MEMBERS']), int(s['N_UNIQUE_VOTES']),
                    int(s['N_UNIQUE_TICKERS']))
        logger.info("  Overall accuracy: %.3f", s['OVERALL_ACCURACY'])
        logger.info("  Buy accuracy: %.3f (base rate: %.3f)",
                    s['BUY_ACCURACY'], s['BUY_BASE_RATE'])
        logger.info("  Sell accuracy: %.3f (base rate: %.3f)",
                    s['SELL_ACCURACY'], s['SELL_BASE_RATE'])

    if not directional.empty:
        logger.info("  ── Directional Accuracy by Cut ──")
        for _, row in directional.head(15).iterrows():
            sig = '*' if row.get('P_VALUE_50', 1) < 0.05 else ''
            cond_sig = ('†' if row.get('P_VALUE_COND', 1) < 0.05 else '')
            logger.info("    %s/%s: %.3f (N=%d) p50=%.4f%s pcond=%.4f%s",
                        row['CUT'], row['SUBSET'], row['ACCURACY'],
                        int(row['N']), row['P_VALUE_50'], sig,
                        row.get('P_VALUE_COND', np.nan), cond_sig)

    # ── Save ─────────────────────────────────────────────────────────
    results = {
        'panel': panel,
        'directional': directional,
        'committee': committee,
        'timing': timing,
        'latency': latency,
        'magnitude': magnitude,
        'regulatory': regulatory,
        'member_fe': member_fe,
        'placebo': placebo,
        'summary': summary,
    }

    logger.info("Saving results...")
    save_results(store, results)

    return results
