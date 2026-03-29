"""
Essay 2 — Culture War Event Study with Regime Conditioning.

Consumes Essay 1's regime assignments and matched-control results.
Nothing is re-estimated here — regimes, factor loadings, and control
deltas all flow in from essay1.py and essay1_matched.py.

The event study logic will be added incrementally on top of this
foundation.
"""

import logging
import warnings
from dataclasses import dataclass, field
from typing import Dict, List, Optional

import pandas as pd
import statsmodels.api as sm

from .datastore import DataStore
from .essay1 import (
    RegimeResult,
    estimate_vix_regimes,
    FF5RegimeAnalysis,
    ff5_by_regime,
    CultureWarRegimeAnalysis,
    culture_war_by_regime,
    _FF5_ALL,
    _HAC_MAXLAGS,
    benjamini_hochberg,
)
from .essay1_matched import (
    MatchedControlResult,
    ff5_matched_control_analysis,
)

logger = logging.getLogger(__name__)


# ─── Backward Compatibility ──────────────────────────────────────────

@dataclass
class FactorModelResult:
    """Result from a factor model regression."""
    ticker: str
    model_name: str
    alpha: float
    alpha_t: float
    alpha_p: float
    betas: dict
    r_squared: float
    adj_r_squared: float
    n_obs: int
    residuals: pd.Series = field(repr=False)
    summary: object = field(repr=False, default=None)


def factor_model(
    store: DataStore,
    ticker: str,
    model: str = 'FF5',
    start_date: str = None,
    end_date: str = None,
) -> Optional[FactorModelResult]:
    """
    Estimate a Fama-French factor model for a given ticker.
    """
    returns = store.get_ticker_returns(ticker)
    if returns.empty or 'RETURN' not in returns.columns:
        logger.warning("No return data for %s", ticker)
        return None

    if model == 'FF3':
        factor_cols = ['MKT_RF', 'SMB', 'HML']
        fdata = store.ff3.copy()
    elif model == 'FF5':
        factor_cols = ['MKT_RF', 'SMB', 'HML', 'RMW', 'CMA']
        fdata = store.ff5.copy()
    elif model == 'FF5+MOM':
        factor_cols = ['MKT_RF', 'SMB', 'HML', 'RMW', 'CMA', 'MOM']
        fdata = store.factors.copy()
    else:
        raise ValueError(f"Unknown model: {model}. Use 'FF3', 'FF5', or 'FF5+MOM'.")

    if fdata.empty:
        logger.warning("No factor data available for model %s", model)
        return None

    for col in factor_cols + ['RF']:
        if col in fdata.columns and fdata[col].abs().max() > 1:
            fdata[col] = fdata[col] / 100

    merged = returns[['DATE', 'RETURN']].merge(fdata, on='DATE', how='inner')

    if start_date:
        merged = merged[merged['DATE'] >= pd.Timestamp(start_date)]
    if end_date:
        merged = merged[merged['DATE'] <= pd.Timestamp(end_date)]

    merged = merged.dropna(subset=['RETURN'] + factor_cols + ['RF'])
    if len(merged) < 30:
        logger.warning("Insufficient observations for %s (%d)", ticker, len(merged))
        return None

    merged['EXCESS_RETURN'] = merged['RETURN'] - merged['RF']

    y = merged['EXCESS_RETURN']
    X = sm.add_constant(merged[factor_cols], has_constant='add')

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        result = sm.OLS(y, X).fit(cov_type='HC1')

    betas = {col: result.params[col] for col in factor_cols}

    return FactorModelResult(
        ticker=ticker,
        model_name=model,
        alpha=result.params['const'],
        alpha_t=result.tvalues['const'],
        alpha_p=result.pvalues['const'],
        betas=betas,
        r_squared=result.rsquared,
        adj_r_squared=result.rsquared_adj,
        n_obs=result.nobs,
        residuals=pd.Series(result.resid.values, index=merged['DATE'].values),
        summary=result,
    )
