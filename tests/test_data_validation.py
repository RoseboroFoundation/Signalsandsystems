"""Tests for clean/validation.py and cache data integrity."""

import pandas as pd
import numpy as np
import pytest

from clean.cache import _save_cache, _load_cache
from clean.validation import validate_dataframe


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
