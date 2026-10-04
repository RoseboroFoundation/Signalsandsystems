"""Orthogonalize the culture factor with respect to political variables.

Implements the Baker-Wurgler (2006) purging approach recommended by Dr. Hu:

    Culture_t = alpha + beta' * Political_t + epsilon_t

where epsilon_hat_t is the "systematic" culture factor, orthogonal to
political affiliation by construction.

Political proxies:
  1. Presidential party (D=1, R=0)
  2. Senate majority party (D=1, R=0)
  3. House majority party (D=1, R=0)
  4. Unified government dummy (president + both chambers same party)
  5. Economic Policy Uncertainty (EPU) index (Baker, Bloom, Davis 2016)
  6. Partisan conflict index (Federal Reserve Bank of Philadelphia)

Estimation modes:
  - Full-sample (for descriptive analysis)
  - Rolling window (for predictive tests, avoids look-ahead bias per Pagan 1984)
  - Expanding window (alternative to rolling)

References
----------
Baker, M. & Wurgler, J. (2006). Investor sentiment and the cross-section
    of stock returns. Journal of Finance, 61(4).
Baker, S.R., Bloom, N. & Davis, S.J. (2016). Measuring economic policy
    uncertainty. Quarterly Journal of Economics, 131(4).
Pagan, A. (1984). Econometric issues in the analysis of regressions with
    generated regressors. International Economic Review, 25(1).
"""

import logging
import os
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import statsmodels.api as sm

from .config import logger

# ═══════════════════════════════════════════════════════════════════════
# POLITICAL PROXY DATA
# ═══════════════════════════════════════════════════════════════════════

# Party control of the presidency (inauguration dates)
_PRESIDENT_PARTY: List[Tuple[str, str, int]] = [
    # (start_date, end_date, is_democrat)
    ('1993-01-20', '2001-01-19', 1),   # Clinton (D)
    ('2001-01-20', '2009-01-19', 0),   # Bush (R)
    ('2009-01-20', '2017-01-19', 1),   # Obama (D)
    ('2017-01-20', '2021-01-19', 0),   # Trump (R)
    ('2021-01-20', '2025-01-19', 1),   # Biden (D)
    ('2025-01-20', '2029-01-19', 0),   # Trump (R)
]

# Senate majority party by Congress (start of each Congress in January)
_SENATE_MAJORITY: List[Tuple[str, str, int]] = [
    ('2000-01-01', '2001-01-02', 0),   # 106th: R majority
    ('2001-01-03', '2001-06-05', 0),   # 107th initially R (50-50, Cheney tiebreak)
    ('2001-06-06', '2003-01-02', 1),   # Jeffords switch → D majority
    ('2003-01-03', '2007-01-03', 0),   # 108th-109th: R
    ('2007-01-04', '2011-01-04', 1),   # 110th-111th: D
    ('2011-01-05', '2015-01-05', 1),   # 112th-113th: D
    ('2015-01-06', '2021-01-19', 0),   # 114th-116th: R
    ('2021-01-20', '2023-01-02', 1),   # 117th: D (50-50 + Harris)
    ('2023-01-03', '2025-01-02', 1),   # 118th: D (kept slim majority)
    ('2025-01-03', '2027-01-02', 0),   # 119th: R
]

# House majority party
_HOUSE_MAJORITY: List[Tuple[str, str, int]] = [
    ('2000-01-01', '2007-01-03', 0),   # R majority through 109th
    ('2007-01-04', '2011-01-04', 1),   # 110th-111th: D (Pelosi)
    ('2011-01-05', '2019-01-02', 0),   # 112th-115th: R (Boehner/Ryan)
    ('2019-01-03', '2023-01-02', 1),   # 116th-117th: D (Pelosi)
    ('2023-01-03', '2025-01-02', 0),   # 118th: R (McCarthy/Johnson)
    ('2025-01-03', '2027-01-02', 0),   # 119th: R
]

# NBER recession dates (for plotting)
NBER_RECESSIONS: List[Tuple[str, str]] = [
    ('2001-03-01', '2001-11-30'),
    ('2007-12-01', '2009-06-30'),
    ('2020-02-01', '2020-04-30'),
]

# Election dates (presidential)
ELECTION_DATES: List[str] = [
    '2000-11-07', '2004-11-02', '2008-11-04', '2012-11-06',
    '2016-11-08', '2020-11-03', '2024-11-05',
]


def _normalize_freq(freq: str) -> str:
    """Map user-friendly frequency codes to pandas-compatible ones."""
    return {'M': 'MS', 'Q': 'QS', 'D': 'D'}.get(freq, freq)


def _build_party_series(
    intervals: List[Tuple[str, str, int]],
    freq: str = 'M',
    start: str = '2000-01-01',
    end: str = '2025-12-31',
) -> pd.Series:
    """Convert (start, end, value) intervals into a time series."""
    idx = pd.date_range(start, end, freq=_normalize_freq(freq))
    series = pd.Series(np.nan, index=idx, dtype=float)
    for s, e, val in intervals:
        mask = (series.index >= pd.Timestamp(s)) & (series.index <= pd.Timestamp(e))
        series.loc[mask] = val
    # Forward-fill any gaps
    series = series.ffill().bfill()
    return series


def build_political_proxies(
    freq: str = 'M',
    start: str = '2000-01-01',
    end: str = '2025-12-31',
    epu_path: Optional[str] = None,
    partisan_conflict_path: Optional[str] = None,
    fred_api_key: Optional[str] = None,
) -> pd.DataFrame:
    """Build a DataFrame of political proxy variables at the given frequency.

    Parameters
    ----------
    freq : str
        'M' for monthly (recommended), 'Q' for quarterly, 'D' for daily.
    start, end : str
        Date range.
    epu_path : str, optional
        Path to EPU index CSV (policyuncertainty.com). If None, attempts
        FRED download.
    partisan_conflict_path : str, optional
        Path to partisan conflict CSV. If None, attempts FRED download.
    fred_api_key : str, optional
        FRED API key for downloading EPU/partisan conflict.

    Returns
    -------
    pd.DataFrame
        Columns: PRES_DEM, SENATE_DEM, HOUSE_DEM, UNIFIED_GOV,
                 EPU (if available), PARTISAN_CONFLICT (if available).
    """
    pd_freq = _normalize_freq(freq)
    pres = _build_party_series(_PRESIDENT_PARTY, freq, start, end)
    senate = _build_party_series(_SENATE_MAJORITY, freq, start, end)
    house = _build_party_series(_HOUSE_MAJORITY, freq, start, end)

    df = pd.DataFrame({
        'PRES_DEM': pres,
        'SENATE_DEM': senate,
        'HOUSE_DEM': house,
    })
    df.index.name = 'DATE'

    # Unified government: president + both chambers same party
    df['UNIFIED_GOV'] = (
        (df['PRES_DEM'] == df['SENATE_DEM']) &
        (df['PRES_DEM'] == df['HOUSE_DEM'])
    ).astype(float)

    # Try to load EPU index
    epu = _load_epu(epu_path, fred_api_key, start, end, freq)
    if epu is not None:
        # Normalize EPU index to month-start for alignment
        epu.index = epu.index.to_period('M').to_timestamp()
        df = df.join(epu, how='left')
        df['EPU'] = df['EPU'].interpolate(method='time')

    # Try to load Partisan Conflict index
    pc = _load_partisan_conflict(partisan_conflict_path, fred_api_key, start, end, freq)
    if pc is not None:
        pc.index = pc.index.to_period('M').to_timestamp()
        df = df.join(pc, how='left')
        df['PARTISAN_CONFLICT'] = df['PARTISAN_CONFLICT'].interpolate(method='time')

    logger.info("Political proxies: %d rows, %d columns (%s–%s)",
                len(df), len(df.columns), df.index.min().date(), df.index.max().date())
    return df


def _load_epu(
    path: Optional[str],
    fred_api_key: Optional[str],
    start: str, end: str, freq: str,
) -> Optional[pd.Series]:
    """Load Economic Policy Uncertainty index."""
    # Try local file first
    if path and os.path.exists(path):
        try:
            df = pd.read_csv(path)
            # EPU CSV from policyuncertainty.com has Year, Month, columns
            if 'Year' in df.columns and 'Month' in df.columns:
                df['DATE'] = pd.to_datetime(
                    df['Year'].astype(str) + '-' + df['Month'].astype(str) + '-01')
                # Use the overall US EPU column
                epu_col = [c for c in df.columns if 'EPU' in c.upper() or 'news' in c.lower()]
                if epu_col:
                    series = df.set_index('DATE')[epu_col[0]].sort_index()
                    series.name = 'EPU'
                    series = series.loc[start:end]
                    if freq != 'M':
                        series = series.resample(freq).mean()
                    logger.info("  Loaded EPU from %s (%d obs)", path, len(series))
                    return series
        except Exception as e:
            logger.warning("  EPU load from file failed: %s", e)

    # Try FRED (series: USEPUINDXD for daily, USEPUINDXM for monthly)
    api_key = fred_api_key or os.environ.get('FRED_API_KEY', '')
    if api_key:
        try:
            import pandas_datareader as pdr
            series_id = 'USEPUINDXM' if freq in ('M', 'Q') else 'USEPUINDXD'
            epu = pdr.DataReader(series_id, 'fred', start=start, end=end)
            epu = epu.iloc[:, 0]
            epu.name = 'EPU'
            if freq == 'Q':
                epu = epu.resample('Q').mean()
            logger.info("  Downloaded EPU from FRED (%d obs)", len(epu))
            return epu
        except Exception as e:
            logger.debug("  EPU FRED download failed: %s", e)

    logger.info("  EPU index not available (optional — analysis proceeds without it)")
    return None


def _load_partisan_conflict(
    path: Optional[str],
    fred_api_key: Optional[str],
    start: str, end: str, freq: str,
) -> Optional[pd.Series]:
    """Load Partisan Conflict Index (Philadelphia Fed)."""
    if path and os.path.exists(path):
        try:
            df = pd.read_csv(path, parse_dates=['DATE'], index_col='DATE')
            col = df.columns[0]
            series = df[col].sort_index().loc[start:end]
            series.name = 'PARTISAN_CONFLICT'
            if freq != 'M':
                series = series.resample(freq).mean()
            logger.info("  Loaded Partisan Conflict from %s (%d obs)", path, len(series))
            return series
        except Exception as e:
            logger.warning("  Partisan Conflict load failed: %s", e)

    api_key = fred_api_key or os.environ.get('FRED_API_KEY', '')
    if api_key:
        try:
            import pandas_datareader as pdr
            pc = pdr.DataReader('PARTISAN', 'fred', start=start, end=end)
            pc = pc.iloc[:, 0]
            pc.name = 'PARTISAN_CONFLICT'
            if freq == 'Q':
                pc = pc.resample('Q').mean()
            logger.info("  Downloaded Partisan Conflict from FRED (%d obs)", len(pc))
            return pc
        except Exception as e:
            logger.debug("  Partisan Conflict FRED download failed: %s", e)

    logger.info("  Partisan Conflict index not available (optional)")
    return None


# ═══════════════════════════════════════════════════════════════════════
# CULTURE INDEX CONSTRUCTION
# ═══════════════════════════════════════════════════════════════════════

def build_culture_index(
    events_df: pd.DataFrame,
    freq: str = 'M',
    method: str = 'net_score',
) -> pd.Series:
    """Construct a time-series culture index from company-level events.

    Converts the cross-sectional company culture war events into a
    time-series measure suitable for orthogonalization.

    Parameters
    ----------
    events_df : pd.DataFrame
        Culture war events with columns:
        - 'Event Date' (datetime)
        - 'Estimated Political Leaning' (Liberal/Conservative/Mixed)
    freq : str
        Output frequency ('M' monthly, 'Q' quarterly).
    method : str
        Index construction method:
        - 'net_score': Liberal=+1, Conservative=-1, Mixed=0, sum per period
        - 'count_weighted': Count of events weighted by leaning direction
        - 'intensity': Rolling 12-month event count (captures salience)

    Returns
    -------
    pd.Series
        Culture index, named 'CULTURE_RAW'. Higher values = more
        liberal-leaning culture war activity; lower = more conservative.
    """
    df = events_df.copy()

    # Standardize column names
    date_col = None
    lean_col = None
    for c in df.columns:
        if 'event' in c.lower() and 'date' in c.lower():
            date_col = c
        # Match "Estimated Political Leaning" but NOT "Political Leaning Justifications"
        if 'estimated' in c.lower() and 'political' in c.lower():
            lean_col = c
    # Fallback: any column with 'political' and 'leaning' (not 'justification')
    if lean_col is None:
        for c in df.columns:
            cl = c.lower()
            if 'political' in cl and 'lean' in cl and 'justif' not in cl:
                lean_col = c
                break

    if date_col is None or lean_col is None:
        raise ValueError(
            f"Cannot find Event Date or Political Leaning columns. "
            f"Available: {df.columns.tolist()}"
        )

    df['_date'] = pd.to_datetime(df[date_col], errors='coerce')
    df = df.dropna(subset=['_date'])

    # Map political leaning to numeric score
    lean_map = {
        'liberal': 1.0,
        'conservative': -1.0,
        'mixed': 0.0,
    }
    df['_score'] = df[lean_col].str.strip().str.lower().map(lean_map).fillna(0.0)

    # Set period index
    df['_period'] = df['_date'].dt.to_period(freq)

    if method == 'net_score':
        # Sum of signed scores per period
        culture = df.groupby('_period')['_score'].sum()
    elif method == 'count_weighted':
        # Count * direction per period
        culture = df.groupby('_period').apply(
            lambda g: len(g) * g['_score'].mean() if len(g) > 0 else 0.0
        )
    elif method == 'intensity':
        # Total event count (unsigned) — captures cultural salience
        culture = df.groupby('_period')['_score'].count().astype(float)
    else:
        raise ValueError(f"Unknown method: {method}")

    # Convert to timestamp index (month-start) and fill gaps
    culture.index = culture.index.to_timestamp()
    full_idx = pd.date_range(
        culture.index.min(), culture.index.max(), freq=_normalize_freq(freq))
    culture = culture.reindex(full_idx, fill_value=0.0)
    culture.index.name = 'DATE'
    culture.name = 'CULTURE_RAW'

    logger.info("Culture index (%s): %d periods, mean=%.2f, std=%.2f",
                method, len(culture), culture.mean(), culture.std())
    return culture


# ═══════════════════════════════════════════════════════════════════════
# ORTHOGONALIZATION (BAKER-WURGLER PURGING)
# ═══════════════════════════════════════════════════════════════════════

@dataclass
class OrthogonalizationResult:
    """Result from orthogonalizing culture factor against political proxies.

    Attributes
    ----------
    culture_raw : pd.Series
        Original (raw) culture index.
    culture_orthogonal : pd.Series
        Residuals from the first-stage regression — the component of
        culture that is orthogonal to politics by construction.
    first_stage_r2 : float
        R-squared from the full-sample first-stage regression.
        Low R2 means the culture factor is largely independent of
        politics (most variance survives orthogonalization).
    first_stage_adj_r2 : float
        Adjusted R-squared.
    first_stage_coefficients : pd.DataFrame
        Coefficients, t-stats, p-values for each political proxy.
    political_proxies_used : List[str]
        Names of political variables in the first-stage regression.
    estimation_method : str
        'full_sample', 'rolling', or 'expanding'.
    window_size : Optional[int]
        Rolling window size (None for full-sample).
    n_obs : int
        Number of observations used.
    diagnostics : Dict
        Additional diagnostics (DW stat, F-stat, etc.).
    """
    culture_raw: pd.Series
    culture_orthogonal: pd.Series
    first_stage_r2: float
    first_stage_adj_r2: float
    first_stage_coefficients: pd.DataFrame
    political_proxies_used: List[str]
    estimation_method: str
    window_size: Optional[int]
    n_obs: int
    diagnostics: Dict = field(default_factory=dict)


def orthogonalize_culture(
    culture: pd.Series,
    political: pd.DataFrame,
    method: str = 'full_sample',
    window: int = 60,
    min_obs: int = 24,
    hac_maxlags: int = 4,
) -> OrthogonalizationResult:
    """Orthogonalize culture factor against political proxies.

    Regresses Culture_t on Political_t and returns residuals as the
    purged "systematic" culture factor (Baker & Wurgler 2006 approach).

    Parameters
    ----------
    culture : pd.Series
        Raw culture index (from build_culture_index).
    political : pd.DataFrame
        Political proxy variables (from build_political_proxies).
    method : str
        - 'full_sample': Single regression over the entire sample.
          Use for descriptive analysis and reporting R2.
        - 'rolling': Rolling window OLS. Use for predictive tests
          to avoid look-ahead bias (Pagan 1984).
        - 'expanding': Expanding window OLS (alternative to rolling).
    window : int
        Window size in periods (months) for rolling estimation.
        Default 60 (5 years).
    min_obs : int
        Minimum observations required for estimation.
    hac_maxlags : int
        Newey-West max lags for HAC standard errors.

    Returns
    -------
    OrthogonalizationResult
    """
    # Align culture and political proxies
    merged = pd.DataFrame({'CULTURE': culture}).join(political, how='inner')
    merged = merged.dropna()

    if len(merged) < min_obs:
        raise ValueError(
            f"Insufficient observations after alignment: {len(merged)} < {min_obs}")

    y = merged['CULTURE']
    proxy_cols = [c for c in political.columns if c in merged.columns]
    X = sm.add_constant(merged[proxy_cols], has_constant='add')

    logger.info("Orthogonalizing culture: %d obs, %d political proxies (%s), method=%s",
                len(merged), len(proxy_cols), ', '.join(proxy_cols), method)

    if method == 'full_sample':
        residuals, result = _orthogonalize_full_sample(y, X, proxy_cols, hac_maxlags)
    elif method == 'rolling':
        residuals, result = _orthogonalize_rolling(
            y, X, proxy_cols, window, min_obs, hac_maxlags)
    elif method == 'expanding':
        residuals, result = _orthogonalize_expanding(
            y, X, proxy_cols, min_obs, hac_maxlags)
    else:
        raise ValueError(f"Unknown method: {method}. Use 'full_sample', 'rolling', or 'expanding'.")

    residuals.name = 'CULTURE_ORTHOGONAL'

    return OrthogonalizationResult(
        culture_raw=culture,
        culture_orthogonal=residuals,
        first_stage_r2=result['r2'],
        first_stage_adj_r2=result['adj_r2'],
        first_stage_coefficients=result['coefficients'],
        political_proxies_used=proxy_cols,
        estimation_method=method,
        window_size=window if method == 'rolling' else None,
        n_obs=len(merged),
        diagnostics=result.get('diagnostics', {}),
    )


def _orthogonalize_full_sample(
    y: pd.Series,
    X: pd.DataFrame,
    proxy_cols: List[str],
    hac_maxlags: int,
) -> Tuple[pd.Series, Dict]:
    """Full-sample OLS orthogonalization."""
    import warnings
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        fit = sm.OLS(y, X).fit(
            cov_type='HAC', cov_kwds={'maxlags': hac_maxlags})

    residuals = fit.resid
    residuals.index = y.index

    # Build coefficient table
    coef_rows = []
    for var in ['const'] + proxy_cols:
        coef_rows.append({
            'VARIABLE': var,
            'COEFFICIENT': fit.params[var],
            'STD_ERROR': fit.bse[var],
            'T_STAT': fit.tvalues[var],
            'P_VALUE': fit.pvalues[var],
        })
    coef_df = pd.DataFrame(coef_rows)

    logger.info("  Full-sample R2=%.4f, Adj-R2=%.4f, F=%.2f (p=%.4f), DW=%.3f",
                fit.rsquared, fit.rsquared_adj, fit.fvalue, fit.f_pvalue,
                _durbin_watson(fit.resid))

    return residuals, {
        'r2': fit.rsquared,
        'adj_r2': fit.rsquared_adj,
        'coefficients': coef_df,
        'diagnostics': {
            'f_stat': float(fit.fvalue),
            'f_pvalue': float(fit.f_pvalue),
            'durbin_watson': _durbin_watson(fit.resid),
            'aic': fit.aic,
            'bic': fit.bic,
            'n_obs': int(fit.nobs),
        },
    }


def _orthogonalize_rolling(
    y: pd.Series,
    X: pd.DataFrame,
    proxy_cols: List[str],
    window: int,
    min_obs: int,
    hac_maxlags: int,
) -> Tuple[pd.Series, Dict]:
    """Rolling-window OLS orthogonalization (no look-ahead bias)."""
    import warnings
    residuals = pd.Series(np.nan, index=y.index)
    n = len(y)

    for i in range(window, n):
        start_idx = i - window
        y_win = y.iloc[start_idx:i]
        X_win = X.iloc[start_idx:i]

        if len(y_win) < min_obs:
            continue

        try:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                fit = sm.OLS(y_win, X_win).fit()
            # Predict for the CURRENT observation using in-sample coefficients
            # (out-of-sample residual for observation i)
            y_pred = X.iloc[i:i+1].dot(fit.params)
            residuals.iloc[i] = y.iloc[i] - y_pred.iloc[0]
        except Exception:
            continue

    # Also run full-sample for coefficient reporting
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        full_fit = sm.OLS(y, X).fit(
            cov_type='HAC', cov_kwds={'maxlags': hac_maxlags})

    coef_rows = []
    for var in ['const'] + proxy_cols:
        coef_rows.append({
            'VARIABLE': var,
            'COEFFICIENT': full_fit.params[var],
            'STD_ERROR': full_fit.bse[var],
            'T_STAT': full_fit.tvalues[var],
            'P_VALUE': full_fit.pvalues[var],
        })

    valid = residuals.dropna()
    logger.info("  Rolling (w=%d): %d/%d valid residuals, full-sample R2=%.4f",
                window, len(valid), n, full_fit.rsquared)

    return residuals, {
        'r2': full_fit.rsquared,
        'adj_r2': full_fit.rsquared_adj,
        'coefficients': pd.DataFrame(coef_rows),
        'diagnostics': {
            'f_stat': float(full_fit.fvalue),
            'f_pvalue': float(full_fit.f_pvalue),
            'durbin_watson': _durbin_watson(full_fit.resid),
            'window_size': window,
            'valid_residuals': len(valid),
            'total_obs': n,
        },
    }


def _orthogonalize_expanding(
    y: pd.Series,
    X: pd.DataFrame,
    proxy_cols: List[str],
    min_obs: int,
    hac_maxlags: int,
) -> Tuple[pd.Series, Dict]:
    """Expanding-window OLS orthogonalization."""
    import warnings
    residuals = pd.Series(np.nan, index=y.index)
    n = len(y)

    for i in range(min_obs, n):
        y_win = y.iloc[:i]
        X_win = X.iloc[:i]

        try:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                fit = sm.OLS(y_win, X_win).fit()
            y_pred = X.iloc[i:i+1].dot(fit.params)
            residuals.iloc[i] = y.iloc[i] - y_pred.iloc[0]
        except Exception:
            continue

    # Full-sample for reporting
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        full_fit = sm.OLS(y, X).fit(
            cov_type='HAC', cov_kwds={'maxlags': hac_maxlags})

    coef_rows = []
    for var in ['const'] + proxy_cols:
        coef_rows.append({
            'VARIABLE': var,
            'COEFFICIENT': full_fit.params[var],
            'STD_ERROR': full_fit.bse[var],
            'T_STAT': full_fit.tvalues[var],
            'P_VALUE': full_fit.pvalues[var],
        })

    valid = residuals.dropna()
    logger.info("  Expanding (min=%d): %d/%d valid residuals, full-sample R2=%.4f",
                min_obs, len(valid), n, full_fit.rsquared)

    return residuals, {
        'r2': full_fit.rsquared,
        'adj_r2': full_fit.rsquared_adj,
        'coefficients': pd.DataFrame(coef_rows),
        'diagnostics': {
            'f_stat': float(full_fit.fvalue),
            'f_pvalue': float(full_fit.f_pvalue),
            'durbin_watson': _durbin_watson(full_fit.resid),
            'min_obs': min_obs,
            'valid_residuals': len(valid),
            'total_obs': n,
        },
    }


def _durbin_watson(resid: pd.Series) -> float:
    """Compute Durbin-Watson statistic."""
    diff = np.diff(resid.values)
    denom = np.sum(resid.values**2)
    if denom == 0:
        return np.nan
    return float(np.sum(diff**2) / denom)


# ═══════════════════════════════════════════════════════════════════════
# HORSE-RACE AND COMPARISON UTILITIES
# ═══════════════════════════════════════════════════════════════════════

def compare_raw_vs_orthogonal(
    raw: pd.Series,
    orthogonal: pd.Series,
    returns: pd.Series,
    lags: int = 1,
    controls: Optional[pd.DataFrame] = None,
    hac_maxlags: int = 4,
) -> pd.DataFrame:
    """Horse-race: compare raw vs orthogonalized culture factor in
    predicting aggregate market returns.

    If the raw factor predicts but the orthogonal does not, the result
    is driven by politics. If the orthogonal retains power, the culture
    contribution is distinct.

    Parameters
    ----------
    raw : pd.Series
        Raw culture index.
    orthogonal : pd.Series
        Orthogonalized culture index.
    returns : pd.Series
        Market returns (e.g., MKT_RF).
    lags : int
        Number of periods to lag the culture factors.
    controls : pd.DataFrame, optional
        Additional control variables.
    hac_maxlags : int
        Newey-West lags.

    Returns
    -------
    pd.DataFrame
        Model comparison with R2, coefficients, t-stats for each spec.
    """
    import warnings

    df = pd.DataFrame({
        'RETURNS': returns,
        'CULTURE_RAW': raw.shift(lags),
        'CULTURE_ORTHO': orthogonal.shift(lags),
    }).dropna()

    if controls is not None:
        df = df.join(controls, how='inner').dropna()

    results = []

    # Spec 1: Raw culture only
    y = df['RETURNS']
    X1 = sm.add_constant(df[['CULTURE_RAW']])
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        fit1 = sm.OLS(y, X1).fit(cov_type='HAC', cov_kwds={'maxlags': hac_maxlags})
    results.append({
        'SPECIFICATION': '1. Raw culture only',
        'R_SQUARED': fit1.rsquared,
        'ADJ_R_SQUARED': fit1.rsquared_adj,
        'CULTURE_COEF': fit1.params['CULTURE_RAW'],
        'CULTURE_T': fit1.tvalues['CULTURE_RAW'],
        'CULTURE_P': fit1.pvalues['CULTURE_RAW'],
        'N_OBS': int(fit1.nobs),
    })

    # Spec 2: Orthogonalized culture only
    X2 = sm.add_constant(df[['CULTURE_ORTHO']])
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        fit2 = sm.OLS(y, X2).fit(cov_type='HAC', cov_kwds={'maxlags': hac_maxlags})
    results.append({
        'SPECIFICATION': '2. Orthogonalized culture only',
        'R_SQUARED': fit2.rsquared,
        'ADJ_R_SQUARED': fit2.rsquared_adj,
        'CULTURE_COEF': fit2.params['CULTURE_ORTHO'],
        'CULTURE_T': fit2.tvalues['CULTURE_ORTHO'],
        'CULTURE_P': fit2.pvalues['CULTURE_ORTHO'],
        'N_OBS': int(fit2.nobs),
    })

    # Spec 3: Both (horse race)
    X3 = sm.add_constant(df[['CULTURE_RAW', 'CULTURE_ORTHO']])
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        fit3 = sm.OLS(y, X3).fit(cov_type='HAC', cov_kwds={'maxlags': hac_maxlags})
    results.append({
        'SPECIFICATION': '3. Horse race (both)',
        'R_SQUARED': fit3.rsquared,
        'ADJ_R_SQUARED': fit3.rsquared_adj,
        'CULTURE_COEF': fit3.params.get('CULTURE_RAW', np.nan),
        'CULTURE_T': fit3.tvalues.get('CULTURE_RAW', np.nan),
        'CULTURE_P': fit3.pvalues.get('CULTURE_RAW', np.nan),
        'N_OBS': int(fit3.nobs),
    })

    comparison = pd.DataFrame(results)
    logger.info("Horse race: Raw R2=%.4f, Ortho R2=%.4f, Both R2=%.4f",
                fit1.rsquared, fit2.rsquared, fit3.rsquared)
    return comparison


# ═══════════════════════════════════════════════════════════════════════
# PLOTTING
# ═══════════════════════════════════════════════════════════════════════

def plot_culture_factors(
    result: OrthogonalizationResult,
    output_path: str = 'culture_factor_comparison.png',
) -> Optional[str]:
    """Plot raw vs orthogonalized culture factor with election dates
    and NBER recessions marked.

    Returns path to saved figure, or None if matplotlib unavailable.
    """
    try:
        import matplotlib.pyplot as plt
        import matplotlib.dates as mdates
    except ImportError:
        logger.info("matplotlib not available, skipping plot")
        return None

    fig, axes = plt.subplots(3, 1, figsize=(14, 10), sharex=True)

    raw = result.culture_raw.dropna()
    ortho = result.culture_orthogonal.dropna()

    # Panel A: Raw culture factor
    ax = axes[0]
    ax.plot(raw.index, raw.values, color='steelblue', linewidth=1.2, label='Raw Culture Factor')
    ax.set_ylabel('Culture Index')
    ax.set_title('Panel A: Raw Culture Factor (confounded with politics)')
    ax.legend(loc='upper left')
    _add_event_markers(ax)
    ax.grid(True, alpha=0.3)

    # Panel B: Orthogonalized culture factor
    ax = axes[1]
    ax.plot(ortho.index, ortho.values, color='darkred', linewidth=1.2,
            label='Orthogonalized Culture Factor')
    ax.axhline(y=0, color='black', linewidth=0.5, linestyle='--')
    ax.set_ylabel('Residual')
    ax.set_title(f'Panel B: Orthogonalized Culture Factor (R² = {result.first_stage_r2:.3f})')
    ax.legend(loc='upper left')
    _add_event_markers(ax)
    ax.grid(True, alpha=0.3)

    # Panel C: Overlay
    ax = axes[2]
    # Standardize for comparison
    raw_std = (raw - raw.mean()) / raw.std() if raw.std() > 0 else raw
    ortho_std = (ortho - ortho.mean()) / ortho.std() if ortho.std() > 0 else ortho
    ax.plot(raw_std.index, raw_std.values, color='steelblue', alpha=0.6,
            linewidth=1, label='Raw (standardized)')
    ax.plot(ortho_std.index, ortho_std.values, color='darkred', alpha=0.8,
            linewidth=1.2, label='Orthogonalized (standardized)')
    ax.axhline(y=0, color='black', linewidth=0.5, linestyle='--')
    ax.set_ylabel('Std. Units')
    ax.set_title('Panel C: Comparison (standardized)')
    ax.legend(loc='upper left')
    _add_event_markers(ax)
    ax.grid(True, alpha=0.3)

    ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y'))
    ax.xaxis.set_major_locator(mdates.YearLocator(2))

    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    logger.info("Saved culture factor plot to %s", output_path)
    return output_path


def _add_event_markers(ax):
    """Add election dates and NBER recessions to an axes."""
    # NBER recessions (gray bands)
    for start, end in NBER_RECESSIONS:
        ax.axvspan(pd.Timestamp(start), pd.Timestamp(end),
                   alpha=0.15, color='gray', label='')

    # Election dates (vertical lines)
    for ed in ELECTION_DATES:
        ts = pd.Timestamp(ed)
        ax.axvline(ts, color='green', alpha=0.4, linewidth=0.8, linestyle=':')


# ═══════════════════════════════════════════════════════════════════════
# CONVENIENCE: RUN FULL ORTHOGONALIZATION PIPELINE
# ═══════════════════════════════════════════════════════════════════════

def run_orthogonalization_pipeline(
    events_df: pd.DataFrame,
    freq: str = 'M',
    culture_method: str = 'net_score',
    ortho_method: str = 'full_sample',
    rolling_window: int = 60,
    epu_path: Optional[str] = None,
    output_dir: str = './output',
) -> OrthogonalizationResult:
    """Run the complete culture factor orthogonalization pipeline.

    Steps:
    1. Build political proxy time series
    2. Construct raw culture index from events
    3. Orthogonalize culture w.r.t. political proxies
    4. Save results and plots

    Parameters
    ----------
    events_df : pd.DataFrame
        Culture war companies data.
    freq : str
        Time series frequency.
    culture_method : str
        How to construct the raw culture index.
    ortho_method : str
        'full_sample', 'rolling', or 'expanding'.
    rolling_window : int
        Window for rolling/expanding estimation.
    epu_path : str, optional
        Path to EPU index CSV.
    output_dir : str
        Output directory for plots and CSVs.

    Returns
    -------
    OrthogonalizationResult
    """
    os.makedirs(output_dir, exist_ok=True)

    # Step 1: Political proxies
    logger.info("Step 1: Building political proxy variables...")
    political = build_political_proxies(
        freq=freq, epu_path=epu_path)

    # Step 2: Culture index
    logger.info("Step 2: Constructing raw culture index...")
    culture = build_culture_index(events_df, freq=freq, method=culture_method)

    # Step 3: Orthogonalize
    logger.info("Step 3: Orthogonalizing culture factor...")
    result = orthogonalize_culture(
        culture, political,
        method=ortho_method,
        window=rolling_window,
    )

    # Step 4: Report and save
    logger.info("=" * 60)
    logger.info("ORTHOGONALIZATION RESULTS")
    logger.info("=" * 60)
    logger.info("  First-stage R² = %.4f  (%.1f%% of culture variance is political)",
                result.first_stage_r2, result.first_stage_r2 * 100)
    logger.info("  Adjusted R²   = %.4f", result.first_stage_adj_r2)
    logger.info("  Method: %s", result.estimation_method)
    logger.info("  N observations: %d", result.n_obs)
    logger.info("  Political proxies: %s", ', '.join(result.political_proxies_used))
    logger.info("")
    logger.info("  First-stage coefficients:")
    for _, row in result.first_stage_coefficients.iterrows():
        sig = '*' if row['P_VALUE'] < 0.05 else ''
        logger.info("    %-20s  β=%.4f  (t=%.2f, p=%.4f)%s",
                     row['VARIABLE'], row['COEFFICIENT'],
                     row['T_STAT'], row['P_VALUE'], sig)

    # Save to CSV
    culture_df = pd.DataFrame({
        'DATE': result.culture_raw.index,
        'CULTURE_RAW': result.culture_raw.values,
    })
    ortho_vals = result.culture_orthogonal.reindex(result.culture_raw.index)
    culture_df['CULTURE_ORTHOGONAL'] = ortho_vals.values
    culture_df.to_csv(os.path.join(output_dir, 'culture_factors.csv'), index=False)

    result.first_stage_coefficients.to_csv(
        os.path.join(output_dir, 'first_stage_coefficients.csv'), index=False)

    # Plot
    plot_path = os.path.join(output_dir, 'culture_factor_comparison.png')
    plot_culture_factors(result, output_path=plot_path)

    return result
