"""Tests for clean_dataframe() — all four fill methods plus edge cases."""

import numpy as np
import pandas as pd
import pytest

from clean.orchestration import clean_dataframe


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def ts_with_gaps():
    """Time-series DataFrame with scattered NaN gaps of varying length."""
    dates = pd.date_range("2020-01-01", periods=20, freq="D")
    data = pd.DataFrame({"A": range(20), "B": range(100, 120)}, index=dates, dtype=float)
    # Single NaN
    data.iloc[3, 0] = np.nan
    # Gap of 3
    data.iloc[7:10, 1] = np.nan
    # Gap of 6 (exceeds default max_gap=5)
    data.iloc[12:18, 0] = np.nan
    return data


@pytest.fixture
def unsorted_ts():
    """Time-series with a shuffled DatetimeIndex."""
    dates = pd.date_range("2020-01-01", periods=10, freq="D")
    data = pd.DataFrame({"X": range(10)}, index=dates, dtype=float)
    return data.sample(frac=1, random_state=42)


@pytest.fixture
def string_date_index():
    """DataFrame whose index is date-like strings (not DatetimeIndex)."""
    idx = ["2021-06-01", "2021-06-02", "2021-06-03", "2021-06-04"]
    return pd.DataFrame({"V": [1.0, np.nan, 3.0, np.nan]}, index=idx)


# ---------------------------------------------------------------------------
# method="ffill"
# ---------------------------------------------------------------------------

class TestFfill:
    def test_single_nan_filled(self, ts_with_gaps):
        result = clean_dataframe(ts_with_gaps, method="ffill")
        # Row 3 col A was NaN; ffill should propagate row 2's value
        assert result.iloc[3, 0] == pytest.approx(2.0)

    def test_gap_within_limit_filled(self, ts_with_gaps):
        result = clean_dataframe(ts_with_gaps, method="ffill")
        # Rows 7-9 col B (gap of 3, within default max_gap=5) should all be filled
        assert result.iloc[7:10, 1].notna().all()

    def test_gap_exceeding_limit_partially_unfilled(self, ts_with_gaps):
        result = clean_dataframe(ts_with_gaps, method="ffill")
        # Rows 12-17 col A (gap of 6, max_gap=5): first 5 filled, last 1 still NaN
        filled = result.iloc[12:17, 0]
        assert filled.notna().all()
        assert pd.isna(result.iloc[17, 0])

    def test_custom_max_gap(self, ts_with_gaps):
        result = clean_dataframe(ts_with_gaps, method="ffill", max_gap=2)
        # Gap of 3 in col B should NOT be fully filled with max_gap=2
        assert pd.isna(result.iloc[9, 1])


# ---------------------------------------------------------------------------
# method="bfill"
# ---------------------------------------------------------------------------

class TestBfill:
    def test_single_nan_filled_backward(self, ts_with_gaps):
        result = clean_dataframe(ts_with_gaps, method="bfill")
        # Row 3 col A was NaN; bfill should propagate row 4's value (4.0)
        assert result.iloc[3, 0] == pytest.approx(4.0)

    def test_gap_within_limit_filled(self, ts_with_gaps):
        result = clean_dataframe(ts_with_gaps, method="bfill")
        assert result.iloc[7:10, 1].notna().all()


# ---------------------------------------------------------------------------
# method="interpolate"
# ---------------------------------------------------------------------------

class TestInterpolate:
    def test_single_nan_interpolated(self, ts_with_gaps):
        result = clean_dataframe(ts_with_gaps, method="interpolate")
        # Row 3 col A: between 2.0 and 4.0 → should be ~3.0
        assert result.iloc[3, 0] == pytest.approx(3.0, abs=0.5)

    def test_gap_within_limit_interpolated(self, ts_with_gaps):
        result = clean_dataframe(ts_with_gaps, method="interpolate")
        assert result.iloc[7:10, 1].notna().all()


# ---------------------------------------------------------------------------
# method="drop"
# ---------------------------------------------------------------------------

class TestDrop:
    def test_rows_with_nan_removed(self, ts_with_gaps):
        result = clean_dataframe(ts_with_gaps, method="drop")
        assert result.notna().all().all()
        # Original had 20 rows; some should be gone
        assert len(result) < 20

    def test_no_nan_rows_survive(self, ts_with_gaps):
        result = clean_dataframe(ts_with_gaps, method="drop")
        assert result.isnull().sum().sum() == 0


# ---------------------------------------------------------------------------
# Edge cases
# ---------------------------------------------------------------------------

class TestEdgeCases:
    def test_none_input_returns_none(self):
        assert clean_dataframe(None) is None

    def test_empty_dataframe_returns_empty(self):
        empty = pd.DataFrame()
        result = clean_dataframe(empty)
        assert result.empty

    def test_all_nan_rows_dropped(self):
        """Rows that are entirely NaN should be removed regardless of method."""
        dates = pd.date_range("2020-01-01", periods=5, freq="D")
        data = pd.DataFrame({"A": [1.0, np.nan, 3.0, np.nan, 5.0],
                              "B": [10.0, np.nan, 30.0, np.nan, 50.0]},
                             index=dates)
        # Use drop method to clear NaN rows, then verify all-NaN rows are gone
        result = clean_dataframe(data, method="drop")
        assert len(result) == 3
        assert result["A"].tolist() == [1.0, 3.0, 5.0]

    def test_datetime_index_sorted(self, unsorted_ts):
        result = clean_dataframe(unsorted_ts, method="ffill")
        assert result.index.is_monotonic_increasing

    def test_string_index_converted_to_datetime(self, string_date_index):
        result = clean_dataframe(string_date_index, method="ffill")
        assert isinstance(result.index, pd.DatetimeIndex)

    def test_non_datetime_index_preserved(self):
        """Integer index should pass through without conversion."""
        data = pd.DataFrame({"A": [1.0, np.nan, 3.0]}, index=[10, 20, 30])
        result = clean_dataframe(data, method="ffill")
        assert list(result.index) == [10, 20, 30]

    def test_original_not_mutated(self, ts_with_gaps):
        original_copy = ts_with_gaps.copy()
        clean_dataframe(ts_with_gaps, method="ffill")
        pd.testing.assert_frame_equal(ts_with_gaps, original_copy)
