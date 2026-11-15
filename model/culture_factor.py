"""Culture Misalignment Factor — systematic risk tests.

Constructs a long-short portfolio (CMF) from firm-level political stance
alignment, then tests whether culture misalignment is a priced systematic
risk factor beyond Fama-French 5.

Tests:
  1. Spanning regression: CMF = a + b'FF5 + e  (alpha != 0?)
  2. Fama-MacBeth (1973) two-pass: time-series betas → cross-sectional
     pricing with Newey-West (1987) standard errors. Falls back to
     Brennan-Chordia-Subrahmanyam (1998) characteristic-based test when
     insufficient data for full two-pass.
  3. Regime conditioning: does CMF premium vary with VIX regime?

References
----------
Fama, E.F. & MacBeth, J.D. (1973). Risk, return, and equilibrium:
    Empirical tests. Journal of Political Economy, 81(3).
Brennan, M.J., Chordia, T. & Subrahmanyam, A. (1998). Alternative
    factor specifications, security characteristics, and the
    cross-section of expected stock returns. JFE, 49(3).
Newey, W.K. & West, K.D. (1987). A simple, positive semi-definite,
    heteroskedasticity and autocorrelation consistent covariance matrix.
    Econometrica, 55(3).
Li, K., Mai, F., Shen, R. & Yan, X. (2021). Measuring Corporate Culture
    Using Machine Learning. Review of Financial Studies, 34(7).
"""

import logging
import warnings
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import statsmodels.api as sm
from scipy import stats

logger = logging.getLogger(__name__)

_FF5_FACTORS = ['MKT_RF', 'SMB', 'HML', 'RMW', 'CMA']
_HAC_MAXLAGS = 6


# ═══════════════════════════════════════════════════════════════════════
# ALIGNMENT SCORING
# ═══════════════════════════════════════════════════════════════════════

def compute_alignment(
    stance_scores: pd.DataFrame,
    surrounding_culture: pd.Series,
) -> pd.DataFrame:
    """Compute firm-level alignment with surrounding culture.

    Parameters
    ----------
    stance_scores : pd.DataFrame
        From StanceScorer.run(). Must have TICKER, YEAR, STANCE_SCORE.
    surrounding_culture : pd.Series
        Time series of surrounding culture index (monthly or yearly).
        Higher = more liberal environment.

    Returns
    -------
    pd.DataFrame with added columns:
        SURROUNDING_CULTURE, ALIGNMENT (score - culture),
        MISALIGNMENT (absolute), DELIBERATE_STANCE (residual).
    """
    df = stance_scores.copy()

    # Match surrounding culture to the PRIOR year to avoid look-ahead bias.
    # A firm's filing in year t should be compared to culture from year t-1
    # (events that had already occurred when the filing was written).
    if hasattr(surrounding_culture.index, 'year'):
        culture_by_year = surrounding_culture.groupby(
            surrounding_culture.index.year
        ).mean()
    else:
        culture_by_year = surrounding_culture

    # Standardize surrounding culture to match stance scale [-1, 1]
    if culture_by_year.std() > 0:
        culture_std = (culture_by_year - culture_by_year.mean()) / culture_by_year.std()
    else:
        culture_std = culture_by_year * 0

    # Lag by 1 year: firm filing in year t uses culture from year t-1
    culture_lagged = {yr + 1: val for yr, val in culture_std.to_dict().items()}
    df['SURROUNDING_CULTURE'] = df['YEAR'].map(culture_lagged)
    df['SURROUNDING_CULTURE'] = df['SURROUNDING_CULTURE'].fillna(0)

    # Alignment: positive = firm aligns with environment
    # Firm liberal + environment liberal → positive alignment
    df['ALIGNMENT'] = df['STANCE_SCORE'] * df['SURROUNDING_CULTURE']
    df['MISALIGNMENT'] = (df['STANCE_SCORE'] - df['SURROUNDING_CULTURE']).abs()

    # Orthogonalize: regress STANCE on SURROUNDING_CULTURE
    # Residual = DELIBERATE_STANCE (the firm's active choice)
    valid = df.dropna(subset=['STANCE_SCORE', 'SURROUNDING_CULTURE'])
    if len(valid) > 10 and valid['SURROUNDING_CULTURE'].std() > 1e-10:
        y = valid['STANCE_SCORE']
        X = sm.add_constant(valid['SURROUNDING_CULTURE'])
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            fit = sm.OLS(y, X).fit()

        df['DELIBERATE_STANCE'] = np.nan
        df.loc[valid.index, 'DELIBERATE_STANCE'] = fit.resid.values

        logger.info(
            "Stance orthogonalization: R²=%.4f (%.1f%% explained by environment)",
            fit.rsquared, fit.rsquared * 100,
        )
    else:
        df['DELIBERATE_STANCE'] = df['STANCE_SCORE']
        logger.info("Stance orthogonalization: insufficient variation, using raw stance")

    return df


# ═══════════════════════════════════════════════════════════════════════
# CMF CONSTRUCTION (Long-Short Portfolio)
# ═══════════════════════════════════════════════════════════════════════

def build_cmf(
    stance_scores: pd.DataFrame,
    stock_returns: pd.DataFrame,
    n_quantiles: int = 3,
    weighting: str = 'equal',
) -> pd.DataFrame:
    """Construct the Culture Misalignment Factor (CMF).

    Long aligned firms, short misaligned firms, rebalanced annually.

    Parameters
    ----------
    stance_scores : pd.DataFrame
        Must have TICKER, YEAR, ALIGNMENT (or STANCE_SCORE).
    stock_returns : pd.DataFrame
        Daily returns with TICKER, DATE, RETURN.
    n_quantiles : int
        Number of portfolios (3 = terciles, 5 = quintiles).
    weighting : str
        'equal' or 'value' weighted.

    Returns
    -------
    pd.DataFrame
        DATE, CMF (long-short return), ALIGNED_RET, MISALIGNED_RET,
        N_ALIGNED, N_MISALIGNED.
    """
    sort_col = 'ALIGNMENT' if 'ALIGNMENT' in stance_scores.columns else 'STANCE_SCORE'

    # Assign portfolios per year
    stance = stance_scores[['TICKER', 'YEAR', sort_col]].dropna().copy()
    stance['PORTFOLIO'] = stance.groupby('YEAR')[sort_col].transform(
        lambda x: pd.qcut(x, n_quantiles, labels=False, duplicates='drop')
    )

    # Map portfolio assignments to daily returns
    returns = stock_returns[['TICKER', 'DATE', 'RETURN']].dropna().copy()
    returns['DATE'] = pd.to_datetime(returns['DATE'])
    returns['YEAR'] = returns['DATE'].dt.year

    # Use prior year's stance to avoid look-ahead
    stance_lagged = stance.copy()
    stance_lagged['YEAR'] = stance_lagged['YEAR'] + 1

    merged = returns.merge(
        stance_lagged[['TICKER', 'YEAR', 'PORTFOLIO']],
        on=['TICKER', 'YEAR'],
        how='inner',
    )

    if merged.empty:
        logger.warning("No overlap between stance scores and returns")
        return pd.DataFrame()

    max_port = merged['PORTFOLIO'].max()

    # Aligned = top portfolio (high alignment), Misaligned = bottom
    aligned = merged[merged['PORTFOLIO'] == max_port]
    misaligned = merged[merged['PORTFOLIO'] == 0]

    # Equal-weighted daily returns
    aligned_ret = aligned.groupby('DATE')['RETURN'].mean()
    misaligned_ret = misaligned.groupby('DATE')['RETURN'].mean()
    aligned_n = aligned.groupby('DATE')['TICKER'].nunique()
    misaligned_n = misaligned.groupby('DATE')['TICKER'].nunique()

    cmf = pd.DataFrame({
        'ALIGNED_RET': aligned_ret,
        'MISALIGNED_RET': misaligned_ret,
        'N_ALIGNED': aligned_n,
        'N_MISALIGNED': misaligned_n,
    })
    # Drop days where either portfolio has zero stocks (avoid NaN propagation)
    cmf = cmf[(cmf['N_ALIGNED'] > 0) & (cmf['N_MISALIGNED'] > 0)]
    cmf['CMF'] = cmf['ALIGNED_RET'] - cmf['MISALIGNED_RET']
    cmf.index.name = 'DATE'
    cmf = cmf.dropna()

    logger.info(
        "CMF: %d trading days, mean=%.4f bps/day, t=%.2f",
        len(cmf),
        cmf['CMF'].mean() * 10000,
        cmf['CMF'].mean() / (cmf['CMF'].std() / np.sqrt(len(cmf)))
        if cmf['CMF'].std() > 0 else 0,
    )

    return cmf.reset_index()


# ═══════════════════════════════════════════════════════════════════════
# TEST 1: SPANNING REGRESSION
# ═══════════════════════════════════════════════════════════════════════

@dataclass
class SpanningResult:
    alpha: float = np.nan
    alpha_t: float = np.nan
    alpha_p: float = np.nan
    r_squared: float = np.nan
    adj_r_squared: float = np.nan
    n_obs: int = 0
    factor_loadings: pd.DataFrame = field(default_factory=pd.DataFrame)
    annualized_alpha_bps: float = np.nan


def spanning_test(
    cmf: pd.DataFrame,
    factors: pd.DataFrame,
    factor_cols: List[str] = None,
) -> SpanningResult:
    """Test if CMF alpha is spanned by existing factors.

    CMF_t = alpha + beta'FF5_t + epsilon_t

    If alpha != 0, CMF captures something FF5 does not.

    Parameters
    ----------
    cmf : pd.DataFrame
        Must have DATE and CMF columns.
    factors : pd.DataFrame
        Must have DATE and factor columns.
    factor_cols : list, optional
        Default: MKT_RF, SMB, HML, RMW, CMA.

    Returns
    -------
    SpanningResult
    """
    if factor_cols is None:
        factor_cols = _FF5_FACTORS

    # Merge on date
    cmf_df = cmf[['DATE', 'CMF']].copy()
    cmf_df['DATE'] = pd.to_datetime(cmf_df['DATE'])

    factors_df = factors.copy()
    factors_df['DATE'] = pd.to_datetime(factors_df['DATE'])

    merged = cmf_df.merge(factors_df[['DATE'] + factor_cols], on='DATE', how='inner')
    merged = merged.dropna()

    if len(merged) < 30:
        logger.warning("Spanning test: insufficient observations (%d)", len(merged))
        return SpanningResult()

    # Factor normalization (percent→decimal) is handled by DataStore at load
    # time.  No heuristic conversion here.

    y = merged['CMF']
    X = sm.add_constant(merged[factor_cols])

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        fit = sm.OLS(y, X).fit(
            cov_type='HAC', cov_kwds={'maxlags': _HAC_MAXLAGS})

    # Build factor loading table
    rows = []
    for var in ['const'] + factor_cols:
        rows.append({
            'FACTOR': var if var != 'const' else 'ALPHA',
            'BETA': fit.params[var],
            'T_STAT': fit.tvalues[var],
            'P_VALUE': fit.pvalues[var],
        })

    result = SpanningResult(
        alpha=fit.params['const'],
        alpha_t=fit.tvalues['const'],
        alpha_p=fit.pvalues['const'],
        r_squared=fit.rsquared,
        adj_r_squared=fit.rsquared_adj,
        n_obs=int(fit.nobs),
        factor_loadings=pd.DataFrame(rows),
        annualized_alpha_bps=fit.params['const'] * 252 * 10000,
    )

    logger.info(
        "Spanning test: alpha=%.4f bps/day (t=%.2f, p=%.4f), R²=%.4f, N=%d",
        result.alpha * 10000, result.alpha_t, result.alpha_p,
        result.r_squared, result.n_obs,
    )
    if result.alpha_p < 0.05:
        logger.info("  → CMF alpha is SIGNIFICANT: not spanned by FF5")
    else:
        logger.info("  → CMF alpha is NOT significant: may be spanned by FF5")

    return result


# ═══════════════════════════════════════════════════════════════════════
# TEST 2: FAMA-MACBETH CROSS-SECTIONAL PRICING
# ═══════════════════════════════════════════════════════════════════════

@dataclass
class FamaMacBethResult:
    gamma_cmf: float = np.nan
    gamma_cmf_t: float = np.nan
    gamma_cmf_p: float = np.nan
    n_months: int = 0
    n_stocks_avg: float = np.nan
    avg_r_squared: float = np.nan
    all_gammas: pd.DataFrame = field(default_factory=pd.DataFrame)


def fama_macbeth(
    stock_returns: pd.DataFrame,
    factors: pd.DataFrame,
    cmf: pd.DataFrame,
    stance_scores: pd.DataFrame,
    rolling_window: int = 60,
    min_obs: int = 24,
) -> FamaMacBethResult:
    """Fama-MacBeth (1973) two-pass cross-sectional pricing test.

    Pass 1 (time-series): For each stock i, estimate betas from a rolling
        regression of excess returns on FF5 + CMF factors.
    Pass 2 (cross-section): Each month t, regress cross-section of stock
        returns on estimated betas to obtain risk-price estimates (gammas).
    Inference: Time-series average of monthly gammas with Newey-West (1987)
        standard errors (lag = 4*(T/100)^0.29, Andrews 1991).

    Parameters
    ----------
    stock_returns : pd.DataFrame
        TICKER, DATE, RETURN.
    factors : pd.DataFrame
        DATE, MKT_RF, SMB, HML, RMW, CMA, RF.
    cmf : pd.DataFrame
        DATE, CMF.
    stance_scores : pd.DataFrame
        TICKER, YEAR, STANCE_SCORE (used as an additional characteristic).
    rolling_window : int
        Months of daily data for first-pass beta estimation (default 60).
    min_obs : int
        Minimum daily observations required per rolling window.

    Returns
    -------
    FamaMacBethResult

    References
    ----------
    Fama, E.F. & MacBeth, J.D. (1973). Risk, return, and equilibrium:
        Empirical tests. Journal of Political Economy, 81(3).
    Newey, W.K. & West, K.D. (1987). A simple, positive semi-definite,
        heteroskedasticity and autocorrelation consistent covariance matrix.
        Econometrica, 55(3).
    """
    # ── Prepare data ──────────────────────────────────────────────────
    ret = stock_returns[['TICKER', 'DATE', 'RETURN']].dropna().copy()
    ret['DATE'] = pd.to_datetime(ret['DATE'])

    fact = factors.copy()
    fact['DATE'] = pd.to_datetime(fact['DATE'])
    # Factor normalization (percent→decimal) is handled by DataStore

    # Merge CMF into factors
    cmf_df = cmf[['DATE', 'CMF']].copy()
    cmf_df['DATE'] = pd.to_datetime(cmf_df['DATE'])
    fact = fact.merge(cmf_df, on='DATE', how='inner')

    ff_cols = [c for c in _FF5_FACTORS if c in fact.columns]
    all_factor_cols = ff_cols + ['CMF']

    # Need RF for excess returns
    if 'RF' not in fact.columns:
        fact['RF'] = 0.0

    # Merge returns with factors
    merged = ret.merge(fact[['DATE', 'RF'] + all_factor_cols], on='DATE', how='inner')
    merged = merged.dropna()
    merged['EXCESS_RETURN'] = merged['RETURN'] - merged['RF']
    merged['MONTH'] = merged['DATE'].dt.to_period('M')

    # ── Pass 1: Rolling time-series betas ─────────────────────────────
    months = sorted(merged['MONTH'].unique())
    if len(months) < rolling_window + 6:
        logger.warning("Fama-MacBeth: insufficient months (%d) for rolling window (%d)",
                        len(months), rolling_window)
        # Fall back to characteristic-based approach
        return _fama_macbeth_characteristic(stock_returns, stance_scores)

    betas_by_month = {}  # month -> {ticker -> {factor: beta}}

    # For each test month, use prior rolling_window months of daily data
    for m_idx in range(rolling_window, len(months)):
        test_month = months[m_idx]
        window_months = months[m_idx - rolling_window:m_idx]

        # Get daily data in the estimation window
        window_data = merged[merged['MONTH'].isin(window_months)]

        month_betas = {}
        for ticker, grp in window_data.groupby('TICKER'):
            if len(grp) < min_obs:
                continue
            y = grp['EXCESS_RETURN'].values
            X = sm.add_constant(grp[all_factor_cols].values)
            try:
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore")
                    fit = sm.OLS(y, X).fit()
                # Store betas (skip constant at index 0)
                month_betas[ticker] = {
                    col: fit.params[i + 1]
                    for i, col in enumerate(all_factor_cols)
                }
            except Exception:
                continue

        if month_betas:
            betas_by_month[test_month] = month_betas

    # ── Pass 2: Monthly cross-sectional regressions ───────────────────
    # Also incorporate stance score as a firm characteristic (lagged 1 year)
    stance = stance_scores[['TICKER', 'YEAR', 'STANCE_SCORE']].dropna().copy()
    stance_lagged = stance.copy()
    stance_lagged['YEAR'] = stance_lagged['YEAR'] + 1

    gammas = []
    for test_month, betas_dict in betas_by_month.items():
        # Get realized returns for the test month
        month_ret = merged[merged['MONTH'] == test_month]
        monthly_ret = month_ret.groupby('TICKER').agg(
            RETURN=('RETURN', lambda x: (1 + x).prod() - 1),
        ).reset_index()

        # Build cross-sectional regressors: estimated betas + stance
        rows = []
        for _, r in monthly_ret.iterrows():
            ticker = r['TICKER']
            if ticker not in betas_dict:
                continue
            row = {'TICKER': ticker, 'RETURN': r['RETURN']}
            row.update({f'BETA_{c}': betas_dict[ticker][c] for c in all_factor_cols})

            # Add stance as characteristic
            year = test_month.start_time.year
            stance_match = stance_lagged[
                (stance_lagged['TICKER'] == ticker) &
                (stance_lagged['YEAR'] == year)
            ]
            if not stance_match.empty:
                row['STANCE_SCORE'] = stance_match['STANCE_SCORE'].iloc[0]
            rows.append(row)

        if len(rows) < 10:
            continue

        cross_df = pd.DataFrame(rows)
        cross_df = cross_df.dropna()
        if len(cross_df) < 10:
            continue

        y = cross_df['RETURN']
        x_cols = [f'BETA_{c}' for c in all_factor_cols]
        if 'STANCE_SCORE' in cross_df.columns and cross_df['STANCE_SCORE'].std() > 1e-10:
            x_cols.append('STANCE_SCORE')
        X = sm.add_constant(cross_df[x_cols])

        try:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                fit = sm.OLS(y, X).fit()

            gamma_row = {
                'MONTH': str(test_month),
                'N_STOCKS': len(cross_df),
                'R_SQUARED': fit.rsquared,
                'GAMMA_CONST': fit.params['const'],
            }
            for col in x_cols:
                gamma_row[f'GAMMA_{col}'] = fit.params[col]
            # Store stance gamma under canonical name
            if 'STANCE_SCORE' in x_cols:
                gamma_row['GAMMA_STANCE'] = fit.params['STANCE_SCORE']
            else:
                gamma_row['GAMMA_STANCE'] = np.nan
            gammas.append(gamma_row)
        except Exception:
            continue

    if not gammas:
        logger.warning("Fama-MacBeth: no valid cross-sections; falling back to characteristic approach")
        return _fama_macbeth_characteristic(stock_returns, stance_scores)

    gamma_df = pd.DataFrame(gammas)

    # ── Inference: Newey-West standard errors on gamma time series ─────
    gamma_series = gamma_df['GAMMA_STANCE'].dropna()
    if len(gamma_series) < 3:
        logger.warning("Fama-MacBeth: too few months with STANCE gamma (%d)", len(gamma_series))
        return _fama_macbeth_characteristic(stock_returns, stance_scores)

    mean_gamma = gamma_series.mean()
    n_months = len(gamma_series)

    # Newey-West SE on the gamma time series (Andrews 1991 lag selection)
    nw_lags = max(1, int(4 * (n_months / 100) ** 0.29))
    try:
        gamma_ols = sm.OLS(gamma_series.values, np.ones(n_months)).fit(
            cov_type='HAC', cov_kwds={'maxlags': nw_lags})
        se_gamma = gamma_ols.bse[0]
        t_gamma = gamma_ols.tvalues[0]
        p_gamma = gamma_ols.pvalues[0]
    except Exception:
        se_gamma = gamma_series.std() / np.sqrt(n_months)
        t_gamma = mean_gamma / se_gamma if se_gamma > 0 else 0
        p_gamma = 2 * (1 - stats.t.cdf(abs(t_gamma), df=n_months - 1))

    result = FamaMacBethResult(
        gamma_cmf=mean_gamma,
        gamma_cmf_t=t_gamma,
        gamma_cmf_p=p_gamma,
        n_months=n_months,
        n_stocks_avg=gamma_df['N_STOCKS'].mean(),
        avg_r_squared=gamma_df['R_SQUARED'].mean(),
        all_gammas=gamma_df,
    )

    logger.info(
        "Fama-MacBeth (2-pass, NW): gamma_stance=%.6f (t=%.2f, p=%.4f), "
        "%d months, %.0f avg stocks, NW lags=%d",
        mean_gamma, t_gamma, p_gamma, n_months, result.n_stocks_avg, nw_lags,
    )

    return result


def _fama_macbeth_characteristic(
    stock_returns: pd.DataFrame,
    stance_scores: pd.DataFrame,
) -> FamaMacBethResult:
    """Fallback: characteristic-based cross-sectional test (BCS 1998).

    When insufficient data for the full two-pass procedure, uses stance
    score directly as a firm characteristic in monthly cross-sections.
    Newey-West standard errors on the gamma time series.

    References
    ----------
    Brennan, M.J., Chordia, T. & Subrahmanyam, A. (1998). Alternative
        factor specifications, security characteristics, and the
        cross-section of expected stock returns. JFE, 49(3).
    """
    ret = stock_returns[['TICKER', 'DATE', 'RETURN']].dropna().copy()
    ret['DATE'] = pd.to_datetime(ret['DATE'])
    ret['MONTH'] = ret['DATE'].dt.to_period('M')

    stance = stance_scores[['TICKER', 'YEAR', 'STANCE_SCORE']].dropna().copy()
    stance_lagged = stance.copy()
    stance_lagged['YEAR'] = stance_lagged['YEAR'] + 1

    ret['YEAR'] = ret['DATE'].dt.year
    ret = ret.merge(stance_lagged, on=['TICKER', 'YEAR'], how='inner')

    monthly = ret.groupby(['TICKER', 'MONTH']).agg(
        RETURN=('RETURN', lambda x: (1 + x).prod() - 1),
        STANCE_SCORE=('STANCE_SCORE', 'first'),
    ).reset_index()

    months = sorted(monthly['MONTH'].unique())
    gammas = []

    for month in months:
        cross = monthly[monthly['MONTH'] == month]
        if len(cross) < 10:
            continue

        y = cross['RETURN']
        X = sm.add_constant(cross[['STANCE_SCORE']])

        try:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                fit = sm.OLS(y, X).fit()

            gammas.append({
                'MONTH': str(month),
                'GAMMA_CONST': fit.params['const'],
                'GAMMA_STANCE': fit.params['STANCE_SCORE'],
                'N_STOCKS': len(cross),
                'R_SQUARED': fit.rsquared,
            })
        except Exception:
            continue

    if not gammas:
        logger.warning("BCS characteristic test: no valid cross-sections")
        return FamaMacBethResult()

    gamma_df = pd.DataFrame(gammas)

    gamma_series = gamma_df['GAMMA_STANCE'].dropna()
    mean_gamma = gamma_series.mean()
    n_months = len(gamma_series)

    # Newey-West SE
    nw_lags = max(1, int(4 * (n_months / 100) ** 0.29))
    try:
        gamma_ols = sm.OLS(gamma_series.values, np.ones(n_months)).fit(
            cov_type='HAC', cov_kwds={'maxlags': nw_lags})
        t_gamma = gamma_ols.tvalues[0]
        p_gamma = gamma_ols.pvalues[0]
    except Exception:
        se_gamma = gamma_series.std() / np.sqrt(n_months)
        t_gamma = mean_gamma / se_gamma if se_gamma > 0 else 0
        p_gamma = 2 * (1 - stats.t.cdf(abs(t_gamma), df=n_months - 1))

    result = FamaMacBethResult(
        gamma_cmf=mean_gamma,
        gamma_cmf_t=t_gamma,
        gamma_cmf_p=p_gamma,
        n_months=n_months,
        n_stocks_avg=gamma_df['N_STOCKS'].mean(),
        avg_r_squared=gamma_df['R_SQUARED'].mean(),
        all_gammas=gamma_df,
    )

    logger.info(
        "BCS characteristic test (NW): gamma_stance=%.6f (t=%.2f, p=%.4f), "
        "%d months, %.0f avg stocks",
        mean_gamma, t_gamma, p_gamma, n_months, result.n_stocks_avg,
    )

    return result


# ═══════════════════════════════════════════════════════════════════════
# TEST 3: REGIME-CONDITIONED ALPHA
# ═══════════════════════════════════════════════════════════════════════

def regime_spanning_test(
    cmf: pd.DataFrame,
    factors: pd.DataFrame,
    vix: pd.DataFrame,
    n_regimes: int = 3,
) -> pd.DataFrame:
    """Test CMF alpha across VIX-based volatility regimes.

    Parameters
    ----------
    cmf : pd.DataFrame
        DATE, CMF.
    factors : pd.DataFrame
        DATE + FF5 columns.
    vix : pd.DataFrame
        DATE, VIX (or CLOSE column).
    n_regimes : int
        Number of VIX quantile buckets.

    Returns
    -------
    pd.DataFrame with spanning results per regime.
    """
    vix_df = vix.copy()
    vix_df['DATE'] = pd.to_datetime(vix_df['DATE'])
    vix_col = 'VIX' if 'VIX' in vix_df.columns else 'CLOSE'
    if vix_col not in vix_df.columns:
        for c in vix_df.columns:
            if c != 'DATE':
                vix_col = c
                break

    # Assign regime by VIX level
    vix_df['REGIME'] = pd.qcut(
        vix_df[vix_col], n_regimes,
        labels=[f'Low_VIX', 'Normal_VIX', 'High_VIX'][:n_regimes],
        duplicates='drop',
    )

    # Merge
    cmf_df = cmf[['DATE', 'CMF']].copy()
    cmf_df['DATE'] = pd.to_datetime(cmf_df['DATE'])

    factors_df = factors.copy()
    factors_df['DATE'] = pd.to_datetime(factors_df['DATE'])

    merged = cmf_df.merge(factors_df, on='DATE', how='inner')
    merged = merged.merge(vix_df[['DATE', 'REGIME']], on='DATE', how='inner')
    merged = merged.dropna()

    factor_cols = [c for c in _FF5_FACTORS if c in merged.columns]

    # Factor normalization (percent→decimal) is handled by DataStore at load
    # time.  No heuristic conversion here.

    results = []
    for regime in sorted(merged['REGIME'].unique()):
        sub = merged[merged['REGIME'] == regime]
        if len(sub) < 30:
            continue

        y = sub['CMF']
        X = sm.add_constant(sub[factor_cols])

        try:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                fit = sm.OLS(y, X).fit(
                    cov_type='HAC', cov_kwds={'maxlags': _HAC_MAXLAGS})

            results.append({
                'REGIME': regime,
                'N_OBS': int(fit.nobs),
                'ALPHA': fit.params['const'],
                'ALPHA_T': fit.tvalues['const'],
                'ALPHA_P': fit.pvalues['const'],
                'ALPHA_ANN_BPS': fit.params['const'] * 252 * 10000,
                'R_SQUARED': fit.rsquared,
            })
        except Exception as e:
            logger.warning("Regime %s spanning failed: %s", regime, e)

    return pd.DataFrame(results)


# ═══════════════════════════════════════════════════════════════════════
# ORTHOGONALIZATION: STANCE vs POLITICAL SPENDING
# ═══════════════════════════════════════════════════════════════════════

def orthogonalize_stance_vs_spending(
    stance_scores: pd.DataFrame,
    political_exposure: pd.DataFrame,
) -> pd.DataFrame:
    """Decompose stance into spending-explained vs. culture residual.

    STANCE = a + b*POLITICAL_CONNECTION_SCORE + e
    e = STANCE_CULTURE (pure culture component)

    Parameters
    ----------
    stance_scores : pd.DataFrame
        TICKER, YEAR, STANCE_SCORE.
    political_exposure : pd.DataFrame
        Must have TICKER and POLITICAL_CONNECTION_SCORE (or similar).

    Returns
    -------
    pd.DataFrame with added: STANCE_SPENDING (fitted), STANCE_CULTURE (residual),
                             SPENDING_R2.
    """
    df = stance_scores.copy()

    # Standardize political exposure columns
    exp = political_exposure.copy()
    exp.columns = [c.upper() for c in exp.columns]

    # Find the score column
    score_col = None
    for c in ['POLITICAL_CONNECTION_SCORE', 'CONNECTION_SCORE', 'TOTAL_SPENDING']:
        if c in exp.columns:
            score_col = c
            break

    if score_col is None:
        logger.warning("No political connection score found in exposure data")
        df['STANCE_CULTURE'] = df['STANCE_SCORE']
        df['STANCE_SPENDING'] = 0.0
        df['SPENDING_R2'] = 0.0
        return df

    # Merge
    merge_cols = ['TICKER']
    if 'YEAR' in exp.columns:
        merge_cols.append('YEAR')

    merged = df.merge(
        exp[merge_cols + [score_col]].drop_duplicates(),
        on=merge_cols, how='left',
    )
    merged[score_col] = merged[score_col].fillna(0)

    # Regress
    valid = merged.dropna(subset=['STANCE_SCORE', score_col])
    if len(valid) > 10 and valid[score_col].std() > 1e-10:
        y = valid['STANCE_SCORE']
        X = sm.add_constant(valid[score_col])
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            fit = sm.OLS(y, X).fit()

        merged['STANCE_SPENDING'] = np.nan
        merged['STANCE_CULTURE'] = np.nan
        merged.loc[valid.index, 'STANCE_SPENDING'] = fit.fittedvalues.values
        merged.loc[valid.index, 'STANCE_CULTURE'] = fit.resid.values
        merged['SPENDING_R2'] = fit.rsquared

        logger.info(
            "Stance vs spending: R²=%.4f (%.1f%% of language explained by money)",
            fit.rsquared, fit.rsquared * 100,
        )
    else:
        merged['STANCE_CULTURE'] = merged['STANCE_SCORE']
        merged['STANCE_SPENDING'] = 0.0
        merged['SPENDING_R2'] = 0.0

    return merged


# ═══════════════════════════════════════════════════════════════════════
# CROSS-SECTIONAL VALUE IMPACT
# ═══════════════════════════════════════════════════════════════════════

def value_impact_regression(
    stance_scores: pd.DataFrame,
    stock_returns: pd.DataFrame,
    windows: List[int] = None,
) -> pd.DataFrame:
    """Test whether alignment predicts post-period returns.

    For each year, compute forward returns over various windows and
    regress on stance/alignment measures.

    Parameters
    ----------
    stance_scores : pd.DataFrame
        TICKER, YEAR, STANCE_SCORE, ALIGNMENT, DELIBERATE_STANCE.
    stock_returns : pd.DataFrame
        TICKER, DATE, RETURN.
    windows : list of int
        Forward return windows in trading days.

    Returns
    -------
    pd.DataFrame of regression results.
    """
    if windows is None:
        windows = [21, 63, 126, 252]  # 1m, 3m, 6m, 12m

    ret = stock_returns[['TICKER', 'DATE', 'RETURN']].dropna().copy()
    ret['DATE'] = pd.to_datetime(ret['DATE'])
    ret = ret.sort_values(['TICKER', 'DATE'])

    # Compute forward cumulative returns from year start
    results = []
    for window in windows:
        # For each ticker-year, compute cumulative return over window
        # starting from the first trading day of the year
        fwd_returns = []
        for ticker in ret['TICKER'].unique():
            t_ret = ret[ret['TICKER'] == ticker].copy()
            t_ret['YEAR'] = t_ret['DATE'].dt.year

            for year in t_ret['YEAR'].unique():
                yr_ret = t_ret[t_ret['YEAR'] == year].head(window)
                if len(yr_ret) >= window * 0.8:  # require 80% of days
                    cum_ret = (1 + yr_ret['RETURN']).prod() - 1
                    fwd_returns.append({
                        'TICKER': ticker,
                        'YEAR': year,
                        'FWD_RETURN': cum_ret,
                    })

        if not fwd_returns:
            continue

        fwd_df = pd.DataFrame(fwd_returns)

        # Use prior year stance to predict current year returns
        stance = stance_scores.copy()
        stance['YEAR'] = stance['YEAR'] + 1

        test_vars = [c for c in ['STANCE_SCORE', 'ALIGNMENT', 'DELIBERATE_STANCE',
                                  'MISALIGNMENT', 'STANCE_CULTURE']
                     if c in stance.columns]

        merged = fwd_df.merge(stance, on=['TICKER', 'YEAR'], how='inner')

        for var in test_vars:
            valid = merged.dropna(subset=['FWD_RETURN', var])
            if len(valid) < 20:
                continue

            y = valid['FWD_RETURN']
            X = sm.add_constant(valid[[var]])

            try:
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore")
                    fit = sm.OLS(y, X).fit(cov_type='HC1')

                results.append({
                    'WINDOW_DAYS': window,
                    'WINDOW_LABEL': f'{window}d (~{window // 21}m)',
                    'VARIABLE': var,
                    'BETA': fit.params[var],
                    'T_STAT': fit.tvalues[var],
                    'P_VALUE': fit.pvalues[var],
                    'R_SQUARED': fit.rsquared,
                    'N_OBS': int(fit.nobs),
                })
            except Exception:
                continue

    return pd.DataFrame(results)
