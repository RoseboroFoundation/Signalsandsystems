"""Essay 3 — Systematic Risk: Macro regime effects on event impacts.

In progress — to be developed after Essays 1 and 2.
Inflation regime classification kept for future use.
"""

import logging
import warnings

import numpy as np
import pandas as pd

from .datastore import DataStore

warnings.warn(
    "essay3 is under active development; the API may change.",
    FutureWarning,
    stacklevel=2,
)

logger = logging.getLogger(__name__)


def classify_inflation_regime(store: DataStore, low=2.0, high=4.0):
    """
    Classify months into inflation regimes using Core PCE YoY.

    Returns a DataFrame with DATE and INFLATION_REGIME columns.
    """
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
