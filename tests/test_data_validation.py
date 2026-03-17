"""Basic data validation helpers and tests."""

import pandas as pd
import numpy as np
import pytest

from clean.cache import _save_cache, _load_cache


def validate_dataframe(df, name="", min_rows=1, expected_date_index=True):
    """Validate a DataFrame meets basic quality checks.

    Parameters
    ----------
    df : pd.DataFrame
        DataFrame to validate.
    name : str
        Label for error messages.
    min_rows : int
        Minimum expected row count.
    expected_date_index : bool
        Whether the index should be DatetimeIndex.

    Returns
    -------
    list[str] : List of validation issues (empty = all good).
    """
    issues = []

    if df is None:
        issues.append(f"{name}: DataFrame is None")
        return issues

    if not isinstance(df, pd.DataFrame):
        issues.append(f"{name}: not a DataFrame (got {type(df).__name__})")
        return issues

    if len(df) < min_rows:
        issues.append(f"{name}: only {len(df)} rows (expected >= {min_rows})")

    # Check for all-null columns
    all_null_cols = df.columns[df.isnull().all()].tolist()
    if all_null_cols:
        issues.append(f"{name}: all-null columns: {all_null_cols}")

    # Check date index
    if expected_date_index and not isinstance(df.index, pd.DatetimeIndex):
        issues.append(f"{name}: index is {type(df.index).__name__}, expected DatetimeIndex")

    return issues


class TestValidateDataFrame:
    """Tests for the validate_dataframe helper."""

    def test_valid_dataframe_passes(self):
        df = pd.DataFrame(
            {"A": [1, 2, 3]},
            index=pd.date_range("2020-01-01", periods=3),
        )
        assert validate_dataframe(df, "test") == []

    def test_none_detected(self):
        issues = validate_dataframe(None, "test")
        assert len(issues) == 1
        assert "None" in issues[0]

    def test_all_null_column_detected(self):
        df = pd.DataFrame(
            {"good": [1, 2, 3], "bad": [None, None, None]},
            index=pd.date_range("2020-01-01", periods=3),
        )
        issues = validate_dataframe(df, "test")
        assert any("all-null" in i for i in issues)
        assert "bad" in issues[0]

    def test_too_few_rows_detected(self):
        df = pd.DataFrame(
            {"A": [1]},
            index=pd.date_range("2020-01-01", periods=1),
        )
        issues = validate_dataframe(df, "test", min_rows=10)
        assert any("rows" in i for i in issues)

    def test_wrong_index_type_detected(self):
        df = pd.DataFrame({"A": [1, 2, 3]}, index=["a", "b", "c"])
        issues = validate_dataframe(df, "test", expected_date_index=True)
        assert any("DatetimeIndex" in i for i in issues)

    def test_non_date_index_ok_when_not_required(self):
        df = pd.DataFrame({"A": [1, 2, 3]}, index=["a", "b", "c"])
        issues = validate_dataframe(df, "test", expected_date_index=False)
        assert issues == []


class TestCacheDataIntegrity:
    """Validate that cached data maintains quality after round-trip."""

    def test_no_all_null_columns_after_round_trip(self, sample_dataframe, tmp_cache_dir):
        _save_cache(sample_dataframe, tmp_cache_dir)
        loaded = _load_cache(tmp_cache_dir)

        issues = validate_dataframe(loaded, "round-trip")
        assert issues == []

    def test_date_range_preserved(self, sample_dataframe, tmp_cache_dir):
        _save_cache(sample_dataframe, tmp_cache_dir)
        loaded = _load_cache(tmp_cache_dir)

        assert loaded.index.min() == sample_dataframe.index.min()
        assert loaded.index.max() == sample_dataframe.index.max()

    def test_no_data_loss(self, sample_dataframe, tmp_cache_dir):
        _save_cache(sample_dataframe, tmp_cache_dir)
        loaded = _load_cache(tmp_cache_dir)

        assert loaded.shape == sample_dataframe.shape
        assert not loaded.isnull().any().any()
