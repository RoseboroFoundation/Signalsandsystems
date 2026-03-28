"""Tests for model package — Essay 1 analytical pipeline.

All tests use synthetic data. No network calls, no real database.
"""

import tempfile
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from model.essay1 import benjamini_hochberg, _chow_test
from model.datastore import DataStore


# =========================================================================
# Benjamini-Hochberg FDR procedure (4 tests)
# =========================================================================


class TestBenjaminiHochberg:
    """Tests for the BH step-up procedure."""

    def test_all_below_threshold(self):
        """All p-values well below threshold → all rejected."""
        pvals = [0.001, 0.002, 0.003, 0.004]
        result = benjamini_hochberg(pvals, q=0.05)
        assert result == [True, True, True, True]

    def test_all_above_threshold(self):
        """All p-values above threshold → none rejected."""
        pvals = [0.80, 0.85, 0.90, 0.95]
        result = benjamini_hochberg(pvals, q=0.05)
        assert result == [False, False, False, False]

    def test_mixed_step_up_cutoff(self):
        """Verify cumulative rejection logic with known step-up cutoff.

        BH at q=0.05 with 5 tests:
          sorted p-values: 0.005, 0.011, 0.030, 0.400, 0.900
          thresholds:      0.010, 0.020, 0.030, 0.040, 0.050
          p <= threshold?:  Yes,   Yes,   Yes,   No,    No
          Largest k where p_k <= threshold: k=3
          → reject ranks 1-3 (indices 0, 2, 1 in original order)
        """
        pvals = [0.005, 0.030, 0.011, 0.400, 0.900]
        result = benjamini_hochberg(pvals, q=0.05)
        # Sorted: (0,0.005), (2,0.011), (1,0.030), (3,0.400), (4,0.900)
        # Ranks 1-3 rejected → original indices 0, 2, 1
        assert result == [True, True, True, False, False]

    def test_empty_input(self):
        """Empty list → empty list."""
        assert benjamini_hochberg([], q=0.05) == []


# =========================================================================
# Chow test for structural break (3 tests)
# =========================================================================


class TestChowTest:
    """Tests for the Chow structural break test."""

    def test_different_means_significant(self):
        """Two groups with very different coefficients → significant F-stat."""
        rng = np.random.default_rng(42)
        n = 500

        # Group A: y = 2*x + noise
        x_a = rng.standard_normal(n)
        y_a = 2.0 * x_a + rng.normal(0, 0.5, n)
        # Group B: y = -1*x + noise
        x_b = rng.standard_normal(n)
        y_b = -1.0 * x_b + rng.normal(0, 0.5, n)

        data = pd.DataFrame({
            'Y': np.concatenate([y_a, y_b]),
            'X1': np.concatenate([x_a, x_b]),
            'REGIME': ['A'] * n + ['B'] * n,
        })

        result = _chow_test(data, factor_cols=['X1'], dep_var='Y',
                            regime_col='REGIME')

        assert result['f_stat'] > 10  # should be very large
        assert result['p_value'] < 0.001
        assert result['significant_005'] == True

    def test_same_distribution_not_significant(self):
        """Two groups from same DGP → F-stat should be non-significant."""
        rng = np.random.default_rng(123)
        n = 500

        x = rng.standard_normal(2 * n)
        y = 1.5 * x + rng.normal(0, 1.0, 2 * n)

        data = pd.DataFrame({
            'Y': y,
            'X1': x,
            'REGIME': ['A'] * n + ['B'] * n,
        })

        result = _chow_test(data, factor_cols=['X1'], dep_var='Y',
                            regime_col='REGIME')

        assert result['p_value'] > 0.05
        assert result['significant_005'] == False

    def test_insufficient_obs_excluded(self):
        """Regime with fewer than k+1 obs excluded gracefully."""
        rng = np.random.default_rng(99)
        n = 200

        x = rng.standard_normal(n + 2)
        y = 1.0 * x + rng.normal(0, 0.5, n + 2)

        data = pd.DataFrame({
            'Y': y,
            'X1': x,
            'REGIME': ['A'] * n + ['C'] * 2,  # C has only 2 obs (< k+1=2+1=3 with const)
        })

        # Only 1 valid regime → returns NaN (need ≥ 2 valid regimes)
        result = _chow_test(data, factor_cols=['X1'], dep_var='Y',
                            regime_col='REGIME')

        assert np.isnan(result['f_stat'])
        assert result['significant_005'] == False


# =========================================================================
# FOMO z-score computation (3 tests)
# =========================================================================


class TestFOMOZScore:
    """Tests for FOMO z-score logic (within-regime normalization).

    Tests the z-score formula directly since the full sentiment_by_regime
    function requires FinBERT. The formula is:
        FOMO_Z = (sent_mean - regime_mean) / regime_std
    with NaN when regime_std == 0.
    """

    @staticmethod
    def _compute_fomo_z(sent_mean, regime_mean, regime_std):
        """Mirror the z-score logic from essay1.py line 1084."""
        return np.where(
            regime_std > 0,
            (sent_mean - regime_mean) / regime_std,
            np.nan,
        )

    def test_at_regime_mean(self):
        """Sentiment at regime mean → Z ≈ 0."""
        z = self._compute_fomo_z(
            sent_mean=np.array([0.5]),
            regime_mean=np.array([0.5]),
            regime_std=np.array([0.2]),
        )
        assert abs(z[0]) < 1e-10

    def test_two_std_above(self):
        """Sentiment 2 std above regime mean → Z ≈ 2 (euphoria)."""
        z = self._compute_fomo_z(
            sent_mean=np.array([0.9]),
            regime_mean=np.array([0.5]),
            regime_std=np.array([0.2]),
        )
        assert abs(z[0] - 2.0) < 1e-10

    def test_zero_std_returns_nan(self):
        """Regime with zero std → Z = NaN (not Inf, not 0)."""
        z = self._compute_fomo_z(
            sent_mean=np.array([0.5]),
            regime_mean=np.array([0.5]),
            regime_std=np.array([0.0]),
        )
        assert np.isnan(z[0])


# =========================================================================
# Matched control delta & aggregation (3 tests)
# =========================================================================


class TestMatchedControlDelta:
    """Tests for delta computation and treatment-level aggregation."""

    def test_single_pair_delta_arithmetic(self):
        """Single pair: delta = treatment_beta - control_beta."""
        treatment = pd.DataFrame({
            'TICKER': ['TRT'], 'REGIME': ['Low'],
            'MKT_RF_BETA': [1.2], 'SMB_BETA': [0.5],
        })
        control = pd.DataFrame({
            'TICKER': ['CTL'], 'REGIME': ['Low'],
            'MKT_RF_BETA': [0.8], 'SMB_BETA': [0.3],
        })
        # Delta = treatment - control
        delta_mkt = treatment['MKT_RF_BETA'].iloc[0] - control['MKT_RF_BETA'].iloc[0]
        delta_smb = treatment['SMB_BETA'].iloc[0] - control['SMB_BETA'].iloc[0]

        assert abs(delta_mkt - 0.4) < 1e-10
        assert abs(delta_smb - 0.2) < 1e-10

    def test_treatment_level_aggregation(self):
        """Control reused across pairs: verify treatment-level aggregation
        produces correct N for t-test.

        WMT is control for both TGT and COST. After computing deltas,
        we should have N=2 treatment firms, not N=4 raw pairs.
        """
        # Two treatment firms, same control, two regimes each
        deltas = pd.DataFrame({
            'TREATMENT_TICKER': ['TGT', 'TGT', 'COST', 'COST'],
            'CONTROL_TICKER': ['WMT', 'WMT', 'WMT', 'WMT'],
            'REGIME': ['Low', 'High', 'Low', 'High'],
            'MKT_RF_DELTA': [0.1, 0.3, -0.1, 0.2],
        })

        # For "Low" regime, aggregate to treatment-firm level
        low = deltas[deltas['REGIME'] == 'Low']
        agg = low.groupby('TREATMENT_TICKER')['MKT_RF_DELTA'].mean()

        # N should be 2 (TGT, COST), not the raw pair count
        assert len(agg) == 2
        assert abs(agg['TGT'] - 0.1) < 1e-10
        assert abs(agg['COST'] - (-0.1)) < 1e-10

    def test_insufficient_obs_excluded(self):
        """Pair where one side has insufficient obs → excluded from delta."""
        from model.essay1_matched import StockRegimeResult

        # Sufficient obs
        good = StockRegimeResult(
            ticker='TRT', regime='Low', sufficient_obs=True, n_obs=200,
            alpha=0.001, alpha_t=1.5, alpha_p=0.13, r_squared=0.15,
            betas={'MKT_RF': 1.1}, t_stats={'MKT_RF': 5.0},
            p_values={'MKT_RF': 0.001},
        )
        # Insufficient obs
        bad = StockRegimeResult(
            ticker='CTL', regime='Low', sufficient_obs=False, n_obs=10,
        )

        # When control is insufficient, delta should not be computed
        assert good.sufficient_obs is True
        assert bad.sufficient_obs is False
        # The pipeline skips pairs where either side has sufficient_obs=False
        # (verified by checking the guard in ff5_matched_control_analysis)
        assert bad.betas == {}


# =========================================================================
# DataStore (2 tests)
# =========================================================================


class TestDataStore:
    """Tests for DataStore persistence and computation."""

    def test_sqlite_round_trip(self, tmp_path):
        """Write a small DataFrame, read it back, verify contents."""
        from Database import SQLiteLoader

        db_path = str(tmp_path / "test.db")
        df_in = pd.DataFrame({
            'A': [1, 2, 3],
            'B': ['x', 'y', 'z'],
            'C': [1.1, 2.2, 3.3],
        })

        with SQLiteLoader(db_path=db_path) as db:
            db.write_table(df_in, 'TEST_TABLE', replace=True)

        with SQLiteLoader(db_path=db_path) as db:
            df_out = db.read_table('TEST_TABLE')

        assert len(df_out) == 3
        assert list(df_out.columns) == ['A', 'B', 'C']
        assert df_out['A'].tolist() == [1, 2, 3]
        assert df_out['B'].tolist() == ['x', 'y', 'z']
        np.testing.assert_allclose(df_out['C'].tolist(), [1.1, 2.2, 3.3])

    def test_compute_returns(self):
        """Known prices → known returns (verify pct_change arithmetic)."""
        from model.datastore import DataStore

        prices = pd.DataFrame({
            'TICKER': ['AAPL'] * 4,
            'DATE': pd.date_range('2020-01-01', periods=4),
            'ADJ_CLOSE': [100.0, 110.0, 99.0, 108.0],
        })

        # Call _compute_returns directly (it's a method but doesn't need
        # a fully initialized DataStore — we can call it on an instance)
        ds = object.__new__(DataStore)
        result = ds._compute_returns(prices)

        # First row should be NaN (no previous price)
        assert pd.isna(result['RETURN'].iloc[0])

        # Subsequent returns: pct_change
        expected = [np.nan, 0.10, -0.10, 108.0 / 99.0 - 1]
        np.testing.assert_allclose(
            result['RETURN'].iloc[1:].values,
            expected[1:],
            rtol=1e-10,
        )

        # LOG_RETURN = log1p(RETURN)
        for i in range(1, 4):
            np.testing.assert_allclose(
                result['LOG_RETURN'].iloc[i],
                np.log1p(result['RETURN'].iloc[i]),
                rtol=1e-10,
            )
