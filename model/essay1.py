"""
Essay 1 — Volatility Regimes and the Fama-French Five-Factor Model.

Identifies market volatility regimes via Markov regime-switching on the
VIX (Hamilton 1989) and tests whether FF5 factor premia and loadings
differ significantly across regimes.

References
----------
Hamilton, J.D. (1989). A new approach to the economic analysis of
    nonstationary time series and the business cycle. Econometrica, 57(2).
Ang, A. & Bekaert, G. (2002). International asset allocation with
    regime shifts. Review of Financial Studies, 15(4).
Guidolin, M. & Timmermann, A. (2008). Size and value anomalies under
    regime shifts. Journal of Financial Econometrics, 6(1).
Fama, E.F. & French, K.R. (2015). A five-factor model. Journal of
    Financial Economics, 116(1).
"""

import logging
import re
import time
import warnings
from dataclasses import dataclass, field
from typing import Optional

import numpy as np
import pandas as pd
import statsmodels.api as sm
from numpy.linalg import cond
from scipy import stats
from statsmodels.tsa.regime_switching.markov_regression import MarkovRegression

from .datastore import DataStore

logger = logging.getLogger(__name__)

# Regressors for factor-level spanning regressions (dep var = MKT_RF).
# MKT_RF is excluded because it IS the dependent variable.
_FF5_REGRESSORS = ['SMB', 'HML', 'RMW', 'CMA']

# All five FF5 factor columns — used as regressors in individual stock
# pricing regressions (R_i - RF ~ MKT_RF + SMB + HML + RMW + CMA)
# and for premia reporting.
_FF5_ALL = ['MKT_RF', 'SMB', 'HML', 'RMW', 'CMA']

# Default Newey-West lag length
_HAC_MAXLAGS = 5

# Minimum observations per regime for regression estimation
_MIN_REGIME_OBS = 30

# Trading days per year (for annualising daily premia)
_TRADING_DAYS_PER_YEAR = 252


# =========================================================================
# MARKOV REGIME-SWITCHING ON VIX
# =========================================================================

@dataclass
class RegimeResult:
    """Result from Markov regime-switching estimation on VIX."""
    n_regimes: int
    regime_means: dict              # {regime_label: mean VIX}
    regime_variances: dict          # {regime_label: variance}
    transition_matrix: np.ndarray   # k x k transition probabilities
    expected_durations: dict        # {regime_label: expected days}
    smoothed_probabilities: pd.DataFrame  # DATE + P(regime) columns
    regime_assignments: pd.DataFrame      # DATE, VIX, REGIME, REGIME_LABEL
    regime_summary: pd.DataFrame          # summary stats per regime
    model_fit: object = field(repr=False, default=None)
    aic: float = None
    bic: float = None
    log_likelihood: float = None

    # Percentile-based robustness check
    percentile_assignments: pd.DataFrame = field(repr=False, default=None)


def estimate_vix_regimes(
    store: DataStore,
    n_regimes: int = 3,
    switching_variance: bool = True,
) -> Optional[RegimeResult]:
    """
    Identify volatility regimes using a Markov-switching model on VIX levels.

    Fits a Markov regime-switching model (Hamilton 1989) to the VIX series,
    allowing both the mean and (optionally) variance of VIX to differ
    across regimes.

    Parameters
    ----------
    store : DataStore
        Must have VIX data loaded.
    n_regimes : int
        Number of hidden states (default 3: low, normal, high).
    switching_variance : bool
        If True, variance also switches across regimes.

    Returns
    -------
    RegimeResult or None
    """
    if store.vix.empty or 'VIX' not in store.vix.columns:
        logger.error("No VIX data available")
        return None

    vix = store.vix[['DATE', 'VIX']].dropna().copy()
    vix = vix.sort_values('DATE').reset_index(drop=True)

    if len(vix) < 100:
        logger.error("Insufficient VIX observations (%d)", len(vix))
        return None

    # Fit Markov regime-switching model
    # VIX_t = mu_s(t) + epsilon_t, where s(t) is the hidden state
    logger.info("Fitting %d-regime Markov switching model on VIX (%d obs)...",
                n_regimes, len(vix))

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        mod = MarkovRegression(
            vix['VIX'],
            k_regimes=n_regimes,
            switching_variance=switching_variance,
        )
        fit = mod.fit(maxiter=500, disp=False)

    # Extract regime parameters
    # Means for each regime
    raw_means = {}
    for i in range(n_regimes):
        raw_means[i] = fit.params[f'const[{i}]']

    # Sort regimes by mean VIX (low -> high)
    sorted_regimes = sorted(raw_means.keys(), key=lambda r: raw_means[r])
    label_map = {}
    labels = _regime_labels(n_regimes)
    for rank, regime_idx in enumerate(sorted_regimes):
        label_map[regime_idx] = labels[rank]

    regime_means = {label_map[i]: raw_means[i] for i in range(n_regimes)}

    # Variances
    regime_variances = {}
    for i in range(n_regimes):
        if switching_variance:
            regime_variances[label_map[i]] = fit.params[f'sigma2[{i}]']
        else:
            regime_variances[label_map[i]] = fit.params['sigma2']

    # Transition matrix — statsmodels stores P[to|from] column-wise,
    # so columns sum to 1. Squeeze trailing dim and transpose to get
    # the conventional row-stochastic form where P[i,j] = P(to j | from i).
    raw_trans = np.squeeze(fit.regime_transition)  # (k, k)
    transition_matrix = raw_trans.T                # now rows sum to 1
    row_sums = transition_matrix.sum(axis=1)
    assert np.allclose(row_sums, 1.0, atol=1e-4), \
        f"Transition matrix rows don't sum to 1: {row_sums}"

    # Expected duration in each regime (statsmodels provides this directly)
    raw_durations = fit.expected_durations
    expected_durations = {label_map[i]: raw_durations[i] for i in range(n_regimes)}

    # Smoothed probabilities (full-sample inference)
    smooth_probs = pd.DataFrame({'DATE': vix['DATE'].values})
    for i in range(n_regimes):
        smooth_probs[f'P_{label_map[i]}'] = fit.smoothed_marginal_probabilities.iloc[:, i].values

    # Regime assignments (most probable regime per date)
    regime_series = fit.smoothed_marginal_probabilities.values.argmax(axis=1)
    assignments = pd.DataFrame({
        'DATE': vix['DATE'].values,
        'VIX': vix['VIX'].values,
        'REGIME': regime_series,
        'REGIME_LABEL': [label_map[r] for r in regime_series],
    })

    # Summary statistics per regime
    regime_summary = (
        assignments.groupby('REGIME_LABEL')['VIX']
        .agg(['mean', 'std', 'min', 'max', 'count'])
        .reset_index()
    )
    regime_summary.columns = ['REGIME', 'MEAN_VIX', 'STD_VIX',
                               'MIN_VIX', 'MAX_VIX', 'N_DAYS']
    regime_summary['PCT_DAYS'] = regime_summary['N_DAYS'] / len(assignments) * 100

    # Percentile-based robustness check
    pct_assignments = _percentile_regimes(vix, n_regimes)

    logger.info("Markov regime estimation complete:")
    for label in labels:
        m = regime_means[label]
        n = regime_summary.loc[regime_summary['REGIME'] == label, 'N_DAYS'].values[0]
        dur = expected_durations[label]
        logger.info("  %s: mean VIX=%.1f (%d days, E[duration]=%.0f days)",
                    label, m, n, dur)

    return RegimeResult(
        n_regimes=n_regimes,
        regime_means=regime_means,
        regime_variances=regime_variances,
        transition_matrix=transition_matrix,
        expected_durations=expected_durations,
        smoothed_probabilities=smooth_probs,
        regime_assignments=assignments,
        regime_summary=regime_summary,
        model_fit=fit,
        aic=fit.aic,
        bic=fit.bic,
        log_likelihood=fit.llf,
        percentile_assignments=pct_assignments,
    )


# =========================================================================
# REGIME MODEL SELECTION (K=2,3,4,...)
# =========================================================================

def select_n_regimes(
    store: DataStore,
    k_range: range = range(2, 5),
    switching_variance: bool = True,
) -> pd.DataFrame:
    """
    Compare Markov regime-switching models across different numbers of regimes.

    Fits the model for each K in k_range and returns AIC, BIC, and
    log-likelihood for model selection.

    Parameters
    ----------
    store : DataStore
    k_range : range
        Range of regime counts to test (default 2-4).
    switching_variance : bool
        If True, variance switches across regimes.

    Returns
    -------
    pd.DataFrame with columns: K, AIC, BIC, LOG_LIKELIHOOD, N_PARAMS.
    """
    rows = []
    t0 = time.time()
    for k in k_range:
        logger.info("Fitting K=%d regime model...", k)
        result = estimate_vix_regimes(
            store, n_regimes=k, switching_variance=switching_variance,
        )
        if result is None:
            continue
        rows.append({
            'K': k,
            'AIC': result.aic,
            'BIC': result.bic,
            'LOG_LIKELIHOOD': result.log_likelihood,
            'N_PARAMS': len(result.model_fit.params),
        })

    elapsed = time.time() - t0
    df = pd.DataFrame(rows)
    if not df.empty:
        best_aic = df.loc[df['AIC'].idxmin(), 'K']
        best_bic = df.loc[df['BIC'].idxmin(), 'K']
        logger.info("Model selection: best AIC → K=%d, best BIC → K=%d (%.1fs total)",
                    best_aic, best_bic, elapsed)
    return df


# =========================================================================
# HELPERS
# =========================================================================

def _regime_labels(n: int) -> list:
    """Generate ordered regime labels for n regimes."""
    if n == 2:
        return ['Low Volatility', 'High Volatility']
    elif n == 3:
        return ['Low Volatility', 'Normal', 'High Volatility']
    elif n == 4:
        return ['Low Volatility', 'Below Average', 'Above Average', 'High Volatility']
    else:
        return [f'Regime {i+1}' for i in range(n)]


def _percentile_regimes(vix: pd.DataFrame, n_regimes: int) -> pd.DataFrame:
    """Assign regimes using equal-frequency percentile bins (robustness check)."""
    labels = _regime_labels(n_regimes)
    vix = vix.copy()
    try:
        vix['REGIME_LABEL'] = pd.qcut(
            vix['VIX'], q=n_regimes, labels=labels,
        )
    except ValueError:
        # Duplicate bin edges occur when VIX distribution has ties at a
        # percentile boundary (e.g., extended flat-volatility periods).
        # Fall back to equal-width bins.
        logger.warning(
            "_percentile_regimes: duplicate bin edges for q=%d, "
            "falling back to equal-width bins", n_regimes,
        )
        vix['REGIME_LABEL'] = pd.cut(
            vix['VIX'], bins=n_regimes, labels=labels,
        )
    return vix[['DATE', 'VIX', 'REGIME_LABEL']]


def _safe_col(label: str) -> str:
    """Convert a regime label to a safe column name."""
    return re.sub(r'\W+', '_', label).strip('_')


# =========================================================================
# FF5 FACTOR ANALYSIS BY REGIME
# =========================================================================

@dataclass
class RegimeFactorResult:
    """FF5 estimation results conditional on volatility regime."""
    regime: str
    n_obs: int
    alpha: float
    alpha_t: float
    alpha_p: float
    betas: dict                # {factor_name: coefficient}
    beta_tstats: dict          # {factor_name: t-stat}
    beta_pvalues: dict         # {factor_name: p-value}
    r_squared: float
    adj_r_squared: float
    factor_premia: dict        # {factor_name: mean return in this regime}
    summary: object = field(repr=False, default=None)


@dataclass
class FF5RegimeAnalysis:
    """Complete FF5-by-regime analysis."""
    regime_result: RegimeResult
    regime_regressions: dict          # {regime_label: RegimeFactorResult}
    pooled_regression: object         # Full-sample FF5 for comparison
    chow_test: dict                   # F-stat, p-value for structural break
    interaction_model: object         # Regime-interacted FF5 results
    factor_premia_comparison: pd.DataFrame  # Side-by-side factor means
    coefficient_comparison: pd.DataFrame    # Side-by-side betas


def ff5_by_regime(
    store: DataStore,
    regime_result: RegimeResult = None,
    n_regimes: int = 3,
    macro_controls: pd.DataFrame = None,  # TODO: add macro controls to regressions (Essay 3)
) -> Optional[FF5RegimeAnalysis]:
    """
    Test FF5 factor spanning and premia across volatility regimes.

    Runs a spanning regression (MKT_RF ~ SMB + HML + RMW + CMA) to test
    whether the market factor is explained by the other four FF5 factors,
    and whether this relationship differs across regimes.

    For each regime identified by the Markov model:
      1. Subset FF5 factor data to dates in that regime.
      2. Compute mean factor premia per regime.
      3. Run a regime-interacted spanning regression to test whether
         factor loadings differ significantly across regimes.
      4. Chow test for structural break across regimes.

    Parameters
    ----------
    store : DataStore
    regime_result : RegimeResult, optional
        Pre-computed regime assignments. If None, estimates them.
    n_regimes : int
        Number of regimes (used only if regime_result is None).
    macro_controls : pd.DataFrame, optional
        Reserved for Essay 3 macro-augmented regressions.

    Returns
    -------
    FF5RegimeAnalysis or None
    """
    if store.ff5.empty:
        logger.error("No FF5 data available")
        return None

    # Step 0: Get regimes
    if regime_result is None:
        regime_result = estimate_vix_regimes(store, n_regimes=n_regimes)
        if regime_result is None:
            return None

    # Prepare factor data (convert percent to decimal).
    # Use median of absolute values — max > 1 fires on legitimate returns.
    factors = store.ff5[['DATE'] + _FF5_ALL + ['RF']].dropna().copy()
    for col in _FF5_ALL + ['RF']:
        if factors[col].abs().median() > 0.1:
            factors[col] = factors[col] / 100

    # Merge with regime assignments
    merged = factors.merge(
        regime_result.regime_assignments[['DATE', 'REGIME_LABEL']],
        on='DATE', how='inner',
    )

    if len(merged) < 100:
        logger.error("Insufficient merged observations (%d)", len(merged))
        return None

    logger.info("FF5 x regime analysis: %d observations across %d regimes",
                len(merged), regime_result.n_regimes)

    # Step 1: Pooled (full-sample) FF5
    y_pooled = merged['MKT_RF']
    X_pooled = sm.add_constant(merged[_FF5_REGRESSORS])
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        pooled_fit = sm.OLS(y_pooled, X_pooled).fit(
            cov_type='HAC', cov_kwds={'maxlags': _HAC_MAXLAGS},
        )

    # Step 2: Per-regime FF5 regressions + factor premia
    regime_regressions = {}
    labels = sorted(merged['REGIME_LABEL'].unique(),
                    key=lambda x: regime_result.regime_means.get(x, 0))

    for label in labels:
        sub = merged[merged['REGIME_LABEL'] == label]
        if len(sub) < _MIN_REGIME_OBS:
            logger.warning("Skipping regime '%s': only %d obs", label, len(sub))
            continue

        # Factor premia in this regime (all five factors)
        premia = {col: sub[col].mean() for col in _FF5_ALL}

        # Regression: MKT_RF ~ SMB + HML + RMW + CMA
        y = sub['MKT_RF']
        X = sm.add_constant(sub[_FF5_REGRESSORS])

        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            fit = sm.OLS(y, X).fit(
                cov_type='HAC', cov_kwds={'maxlags': _HAC_MAXLAGS},
            )

        betas = {col: fit.params[col] for col in _FF5_REGRESSORS}
        tstats = {col: fit.tvalues[col] for col in _FF5_REGRESSORS}
        pvals = {col: fit.pvalues[col] for col in _FF5_REGRESSORS}

        regime_regressions[label] = RegimeFactorResult(
            regime=label,
            n_obs=int(fit.nobs),
            alpha=fit.params['const'],
            alpha_t=fit.tvalues['const'],
            alpha_p=fit.pvalues['const'],
            betas=betas,
            beta_tstats=tstats,
            beta_pvalues=pvals,
            r_squared=fit.rsquared,
            adj_r_squared=fit.rsquared_adj,
            factor_premia=premia,
            summary=fit,
        )

    # Step 3: Interaction model — test regime effect on factor loadings
    # Create regime dummies (baseline = lowest volatility regime)
    interaction_data = merged.copy()

    interact_cols = []
    for label in labels[1:]:
        dummy_name = f'D_{_safe_col(label)}'
        interaction_data[dummy_name] = (interaction_data['REGIME_LABEL'] == label).astype(int)

        # Interaction terms: dummy * each factor
        for col in _FF5_REGRESSORS:
            interact_name = f'{dummy_name}_x_{col}'
            interaction_data[interact_name] = interaction_data[dummy_name] * interaction_data[col]
            interact_cols.append(interact_name)

    dummy_cols = [f'D_{_safe_col(l)}' for l in labels[1:]]
    all_regressors = _FF5_REGRESSORS + dummy_cols + interact_cols

    y_int = interaction_data['MKT_RF']
    X_int = sm.add_constant(interaction_data[all_regressors])

    cn = cond(X_int.values)
    if cn > 30:
        logger.warning("Interaction model condition number = %.1f (collinearity concern)", cn)

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        interaction_fit = sm.OLS(y_int, X_int).fit(
            cov_type='HAC', cov_kwds={'maxlags': _HAC_MAXLAGS},
        )

    # Step 4: Chow test — are the regime-specific models significantly different?
    chow = _chow_test(merged, factor_cols=_FF5_REGRESSORS,
                       dep_var='MKT_RF', regime_col='REGIME_LABEL')

    # Step 5: Build comparison tables
    factor_premia_rows = []
    coeff_rows = []
    for label in labels:
        if label not in regime_regressions:
            continue
        rr = regime_regressions[label]
        premia_row = {'REGIME': label, 'N_DAYS': rr.n_obs}
        coeff_row = {'REGIME': label, 'ALPHA': rr.alpha, 'ALPHA_P': rr.alpha_p}
        for col in _FF5_ALL:
            premia_row[f'{col}_MEAN'] = rr.factor_premia[col]
            premia_row[f'{col}_MEAN_ANN'] = rr.factor_premia[col] * _TRADING_DAYS_PER_YEAR
        for col in _FF5_REGRESSORS:
            coeff_row[f'{col}_BETA'] = rr.betas[col]
            coeff_row[f'{col}_T'] = rr.beta_tstats[col]
            coeff_row[f'{col}_P'] = rr.beta_pvalues[col]
        coeff_row['R_SQUARED'] = rr.r_squared
        factor_premia_rows.append(premia_row)
        coeff_rows.append(coeff_row)

    return FF5RegimeAnalysis(
        regime_result=regime_result,
        regime_regressions=regime_regressions,
        pooled_regression=pooled_fit,
        chow_test=chow,
        interaction_model=interaction_fit,
        factor_premia_comparison=pd.DataFrame(factor_premia_rows),
        coefficient_comparison=pd.DataFrame(coeff_rows),
    )


def _chow_test(
    data: pd.DataFrame,
    factor_cols: list,
    dep_var: str,
    regime_col: str,
) -> dict:
    """
    Chow test for structural break across regimes.

    Compares RSS of pooled model vs. sum of regime-specific RSS.
    Only regimes with sufficient observations are included; the pooled
    model is restricted to those same rows to avoid RSS inflation bias.

    Note: Uses classical OLS (not HAC) for the F-statistic. The Chow test
    compares residual sums of squares for structural break detection, not
    coefficient significance — classical OLS RSS comparison is standard
    here even though per-regime regressions elsewhere use HAC standard
    errors for inference on individual coefficients.
    """
    k = len(factor_cols) + 1  # including constant

    # Identify regimes with enough observations to fit
    valid_labels = [
        label for label, sub in data.groupby(regime_col)
        if len(sub) >= k + 1
    ]
    if len(valid_labels) < 2:
        return {'f_stat': np.nan, 'p_value': np.nan, 'significant_005': False}

    # Restrict both pooled and subgroup models to the same rows
    data = data[data[regime_col].isin(valid_labels)]
    n_total = len(data)

    y_all = data[dep_var]
    X_all = sm.add_constant(data[factor_cols])
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        pooled = sm.OLS(y_all, X_all).fit()
    rss_pooled = pooled.ssr

    rss_sub = 0.0
    for label in valid_labels:
        sub = data[data[regime_col] == label]
        y_sub = sub[dep_var]
        X_sub = sm.add_constant(sub[factor_cols])
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            fit_sub = sm.OLS(y_sub, X_sub).fit()
        rss_sub += fit_sub.ssr

    n_regimes = len(valid_labels)
    df_num = (n_regimes - 1) * k
    df_den = n_total - n_regimes * k

    if df_den <= 0 or rss_sub == 0:
        return {'f_stat': np.nan, 'p_value': np.nan, 'significant_005': False}

    f_stat = ((rss_pooled - rss_sub) / df_num) / (rss_sub / df_den)
    p_value = 1 - stats.f.cdf(f_stat, df_num, df_den)

    return {
        'f_stat': f_stat,
        'p_value': p_value,
        'df_numerator': df_num,
        'df_denominator': df_den,
        'n_regimes_tested': n_regimes,
        'significant_005': p_value < 0.05,
        'significant_001': p_value < 0.01,
    }


# =========================================================================
# MULTIPLE TESTING CORRECTION
# =========================================================================

def benjamini_hochberg(pvals: list, q: float = 0.05) -> list:
    """
    Benjamini-Hochberg procedure for controlling FDR.

    Parameters
    ----------
    pvals : list of float
        Raw p-values.
    q : float
        Target false discovery rate (default 0.05).

    Returns
    -------
    list of bool
        True where the null is rejected at FDR = q.
    """
    n = len(pvals)
    if n == 0:
        return []

    indexed = sorted(enumerate(pvals), key=lambda x: x[1])
    rejected = [False] * n
    cutoff_rank = -1

    for rank, (orig_idx, p) in enumerate(indexed, start=1):
        threshold = q * rank / n
        if p <= threshold:
            cutoff_rank = rank

    if cutoff_rank > 0:
        for rank, (orig_idx, p) in enumerate(indexed, start=1):
            if rank <= cutoff_rank:
                rejected[orig_idx] = True

    return rejected


# =========================================================================
# CULTURE WAR STOCK ANALYSIS BY REGIME
# =========================================================================

@dataclass
class CultureWarRegimeAnalysis:
    """Results from culture war stock analysis conditioned on regimes."""
    n_stocks: int
    n_failed: int
    stock_results: dict = field(default_factory=dict)
    summary: pd.DataFrame = field(default_factory=pd.DataFrame)


def culture_war_by_regime(
    store: DataStore,
    regime_result: RegimeResult = None,
    ff5_analysis: FF5RegimeAnalysis = None,  # TODO: use for regime-level benchmarks (Essay 3)
    macro_controls: pd.DataFrame = None,     # TODO: add macro controls to regressions (Essay 3)
) -> Optional[CultureWarRegimeAnalysis]:
    """
    Analyse culture war stocks' FF5 loadings across volatility regimes.

    Parameters
    ----------
    store : DataStore
    regime_result : RegimeResult, optional
    ff5_analysis : FF5RegimeAnalysis, optional
        Reserved for regime-level benchmark comparison.
    macro_controls : pd.DataFrame, optional
        Reserved for Essay 3 macro-augmented regressions.

    Returns
    -------
    CultureWarRegimeAnalysis or None
    """
    tickers = store.get_event_tickers()
    if not tickers:
        logger.warning("No event tickers found")
        return None

    if regime_result is None:
        regime_result = estimate_vix_regimes(store)
        if regime_result is None:
            return None

    stock_results = {}
    n_failed = 0

    for ticker in tickers:
        returns = store.get_ticker_returns(ticker)
        if returns.empty or 'RETURN' not in returns.columns:
            n_failed += 1
            continue

        merged = returns[['DATE', 'RETURN']].merge(
            store.ff5[['DATE'] + _FF5_ALL + ['RF']].dropna(),
            on='DATE', how='inner',
        )
        merged = merged.merge(
            regime_result.regime_assignments[['DATE', 'REGIME_LABEL']],
            on='DATE', how='inner',
        )

        if len(merged) < 60:
            n_failed += 1
            continue

        # Convert percent to decimal if needed (median-based check)
        for col in _FF5_ALL + ['RF']:
            if col in merged.columns and merged[col].abs().median() > 0.1:
                merged[col] = merged[col] / 100

        merged['EXCESS_RETURN'] = merged['RETURN'] - merged['RF']

        regime_betas = {}
        for label, sub in merged.groupby('REGIME_LABEL'):
            if len(sub) < _MIN_REGIME_OBS:
                continue
            # Individual stock pricing regression: R_i - RF ~ MKT_RF + SMB + HML + RMW + CMA
            # Uses _FF5_ALL (all five factors) unlike ff5_by_regime which uses _FF5_REGRESSORS
            # (four factors) because there MKT_RF is the dependent variable.
            y = sub['EXCESS_RETURN']
            X = sm.add_constant(sub[_FF5_ALL])
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                fit = sm.OLS(y, X).fit(
                    cov_type='HAC', cov_kwds={'maxlags': _HAC_MAXLAGS},
                )
            regime_betas[label] = {
                'alpha': fit.params['const'],
                'alpha_p': fit.pvalues['const'],
                'betas': {c: fit.params[c] for c in _FF5_ALL},
                'n_obs': int(fit.nobs),
                'r_squared': fit.rsquared,
            }

        # Only count as success if at least one regime was estimable
        if regime_betas:
            stock_results[ticker] = regime_betas
        else:
            logger.warning("No estimable regimes for %s (all regimes < %d obs)",
                          ticker, _MIN_REGIME_OBS)
            n_failed += 1

    # Build summary
    rows = []
    for ticker, regimes in stock_results.items():
        for regime, res in regimes.items():
            row = {'TICKER': ticker, 'REGIME': regime,
                   'ALPHA': res['alpha'], 'ALPHA_P': res['alpha_p'],
                   'N_OBS': res['n_obs'], 'R_SQUARED': res['r_squared']}
            for factor, beta in res['betas'].items():
                row[f'{factor}_BETA'] = beta
            rows.append(row)

    if rows:
        summary = pd.DataFrame(rows)
        bh_rejected = benjamini_hochberg(summary['ALPHA_P'].tolist(), q=0.05)
        summary['ALPHA_SIGNIFICANT_BH'] = bh_rejected
    else:
        summary = pd.DataFrame()

    logger.info("Culture war analysis: %d stocks, %d failed",
                len(stock_results), n_failed)

    return CultureWarRegimeAnalysis(
        n_stocks=len(stock_results),
        n_failed=n_failed,
        stock_results=stock_results,
        summary=summary,
    )


# =========================================================================
# MACRO CONTROLS
# =========================================================================

def assemble_macro_controls(store: DataStore) -> pd.DataFrame:
    """
    Build a panel of macro control variables aligned to daily dates.

    Merges inflation, rates, employment, and GDP data. Monthly/quarterly
    series are forward-filled to daily frequency.

    Parameters
    ----------
    store : DataStore

    Returns
    -------
    pd.DataFrame with DATE index and macro control columns.
    """
    sources = [
        ('inflation', store.inflation),
        ('rates', store.rates),
        ('employment', store.employment),
        ('gdp', store.gdp),
        ('macro', store.macro),
    ]

    frames = []
    names = []
    for name, df in sources:
        if df is not None and not df.empty and 'DATE' in df.columns:
            frames.append(df.copy())
            names.append(name)

    if not frames:
        logger.warning("No macro data available for controls")
        return pd.DataFrame(columns=['DATE'])

    result = frames[0].rename(columns={
        c: f'{c}_{names[0]}' for c in frames[0].columns if c != 'DATE'
    })
    for name, df in zip(names[1:], frames[1:]):
        df = df.rename(columns={c: f'{c}_{name}' for c in df.columns if c != 'DATE'})
        result = result.merge(df, on='DATE', how='outer')

    result = result.sort_values('DATE').reset_index(drop=True)
    # Forward-fill monthly/quarterly series
    result = result.ffill()

    logger.info("Assembled macro controls: %d rows, %d columns",
                len(result), len(result.columns) - 1)
    return result


# =========================================================================
# RESULT PERSISTENCE
# =========================================================================

def save_results(
    store: DataStore,
    ff5_analysis: FF5RegimeAnalysis = None,
    cw_analysis: CultureWarRegimeAnalysis = None,
) -> dict:
    """
    Save Essay 1 results to the database.

    Parameters
    ----------
    store : DataStore
    ff5_analysis : FF5RegimeAnalysis, optional
    cw_analysis : CultureWarRegimeAnalysis, optional

    Returns
    -------
    dict mapping table name to write result.
    """
    results = {}

    if ff5_analysis is not None:
        # Save coefficient comparison
        if not ff5_analysis.coefficient_comparison.empty:
            res = store.write_table(
                ff5_analysis.coefficient_comparison,
                'ESSAY1_FF5_COEFFICIENTS', replace=True,
            )
            results['ESSAY1_FF5_COEFFICIENTS'] = res

        # Save factor premia comparison
        if not ff5_analysis.factor_premia_comparison.empty:
            res = store.write_table(
                ff5_analysis.factor_premia_comparison,
                'ESSAY1_FACTOR_PREMIA', replace=True,
            )
            results['ESSAY1_FACTOR_PREMIA'] = res

    if cw_analysis is not None and not cw_analysis.summary.empty:
        res = store.write_table(
            cw_analysis.summary,
            'ESSAY1_CW_STOCK_RESULTS', replace=True,
        )
        results['ESSAY1_CW_STOCK_RESULTS'] = res

    logger.info("Saved Essay 1 results: %s", list(results.keys()))
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

    print("Dissertation Essay 1 — Volatility Regimes & FF5")
    print(f"Started at {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print()

    store = DataStore()

    # Step 1: Estimate VIX regimes
    print("=" * 60)
    print("  Step 1: Markov regime-switching on VIX")
    print("=" * 60)
    regime_result = estimate_vix_regimes(store, n_regimes=3)

    if regime_result is not None:
        print(f"  Regimes: {regime_result.n_regimes}")
        for label, mean in regime_result.regime_means.items():
            n = regime_result.regime_summary.loc[
                regime_result.regime_summary['REGIME'] == label, 'N_DAYS'
            ].values[0]
            dur = regime_result.expected_durations[label]
            print(f"    {label}: mean VIX={mean:.1f} ({n} days, E[dur]={dur:.0f}d)")
        print(f"  AIC={regime_result.aic:.1f}, BIC={regime_result.bic:.1f}")
    else:
        print("  FAILED — no VIX data or estimation error")

    # Step 2: Model selection
    print()
    print("=" * 60)
    print("  Step 2: Model selection (K=2,3,4)")
    print("=" * 60)
    selection = select_n_regimes(store)
    if not selection.empty:
        for _, row in selection.iterrows():
            print(f"    K={int(row['K'])}: AIC={row['AIC']:.1f}, "
                  f"BIC={row['BIC']:.1f}, LL={row['LOG_LIKELIHOOD']:.1f}")

    # Step 3: FF5 by regime
    print()
    print("=" * 60)
    print("  Step 3: FF5 factor analysis by regime")
    print("=" * 60)
    ff5_analysis = ff5_by_regime(store, regime_result=regime_result)

    if ff5_analysis is not None:
        chow = ff5_analysis.chow_test
        print(f"  Chow test: F={chow['f_stat']:.2f}, p={chow['p_value']:.4f}")
        print(f"  Structural break: {'YES' if chow['significant_005'] else 'NO'} (p<0.05)")
        if not ff5_analysis.coefficient_comparison.empty:
            print("  Coefficient comparison:")
            print(ff5_analysis.coefficient_comparison.to_string(index=False))
    else:
        print("  FAILED — insufficient data")

    # Step 4: Macro controls
    print()
    print("=" * 60)
    print("  Step 4: Macro controls")
    print("=" * 60)
    macro = assemble_macro_controls(store)
    print(f"  Macro panel: {len(macro)} rows, {len(macro.columns)-1} variables")

    # Step 5: Culture war stocks by regime
    print()
    print("=" * 60)
    print("  Step 5: Culture war stocks by regime")
    print("=" * 60)
    cw = culture_war_by_regime(store, regime_result=regime_result)
    if cw is not None:
        print(f"  Stocks analysed: {cw.n_stocks}")
        print(f"  Stocks failed: {cw.n_failed}")
        if not cw.summary.empty:
            print(f"  Summary rows: {len(cw.summary)}")
    else:
        print("  FAILED — no event tickers or data")

    # Step 6: Benjamini-Hochberg sanity check
    print()
    print("=" * 60)
    print("  Step 6: Benjamini-Hochberg sanity check")
    print("=" * 60)
    test_pvals = [0.001, 0.008, 0.039, 0.041, 0.23, 0.45, 0.89]
    bh = benjamini_hochberg(test_pvals, q=0.05)
    print(f"  Input p-values: {test_pvals}")
    print(f"  Rejected (FDR=5%): {bh}")

    # Step 7: Save results to database
    print()
    print("=" * 60)
    print("  Step 7: Save results to database")
    print("=" * 60)
    saved = save_results(store, ff5_analysis=ff5_analysis, cw_analysis=cw)
    if saved:
        for table, res in saved.items():
            print(f"    {table}: {res}")
        print(f"  Saved {len(saved)} tables")
    else:
        print("  Nothing to save (no results produced)")

    store.close()
    print()
    print(f"Completed at {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
