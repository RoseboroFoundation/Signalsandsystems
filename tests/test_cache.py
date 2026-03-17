"""Tests for clean/cache.py — parquet-based cache round-trip."""

import json
import os

import numpy as np
import pandas as pd
import pytest

from clean.cache import _save_cache, _load_cache


class TestSingleDataFrame:
    """Round-trip a plain DataFrame."""

    def test_round_trip(self, sample_dataframe, tmp_cache_dir):
        _save_cache(sample_dataframe, tmp_cache_dir)
        loaded = _load_cache(tmp_cache_dir)

        pd.testing.assert_frame_equal(loaded, sample_dataframe, check_freq=False)

    def test_creates_manifest(self, sample_dataframe, tmp_cache_dir):
        _save_cache(sample_dataframe, tmp_cache_dir)

        manifest_path = os.path.join(tmp_cache_dir, "_manifest.json")
        assert os.path.exists(manifest_path)

        with open(manifest_path) as f:
            manifest = json.load(f)
        assert manifest["type"] == "dataframe"

    def test_creates_parquet_file(self, sample_dataframe, tmp_cache_dir):
        _save_cache(sample_dataframe, tmp_cache_dir)
        assert os.path.exists(os.path.join(tmp_cache_dir, "_data.parquet"))


class TestDictOfDataFrames:
    """Round-trip a dict of DataFrames (like load_inflation_data returns)."""

    def test_round_trip(self, sample_dict_of_dataframes, tmp_cache_dir):
        _save_cache(sample_dict_of_dataframes, tmp_cache_dir)
        loaded = _load_cache(tmp_cache_dir)

        assert set(loaded.keys()) == set(sample_dict_of_dataframes.keys())
        for key in sample_dict_of_dataframes:
            pd.testing.assert_frame_equal(loaded[key], sample_dict_of_dataframes[key], check_freq=False)

    def test_manifest_tracks_keys(self, sample_dict_of_dataframes, tmp_cache_dir):
        _save_cache(sample_dict_of_dataframes, tmp_cache_dir)

        with open(os.path.join(tmp_cache_dir, "_manifest.json")) as f:
            manifest = json.load(f)

        assert manifest["type"] == "dict"
        saved_names = {e["name"] for e in manifest["keys"]}
        assert saved_names == set(sample_dict_of_dataframes.keys())


class TestSummaryStats:
    """Round-trip summary_stats with Timestamps and numpy scalars."""

    def test_timestamp_preserved(self, sample_dataframe, tmp_cache_dir):
        data = {
            "raw": sample_dataframe,
            "summary_stats": {
                "inflation": {
                    "start_date": pd.Timestamp("2020-01-01"),
                    "end_date": pd.Timestamp("2020-12-01"),
                    "mean": np.float64(105.5),
                }
            },
        }

        _save_cache(data, tmp_cache_dir)
        loaded = _load_cache(tmp_cache_dir)

        stats = loaded["summary_stats"]["inflation"]
        assert stats["start_date"] == pd.Timestamp("2020-01-01")
        assert stats["end_date"] == pd.Timestamp("2020-12-01")
        # numpy scalar should be converted to native Python type
        assert isinstance(stats["mean"], (int, float))
        assert stats["mean"] == pytest.approx(105.5)


class TestNestedDicts:
    """Round-trip nested dict structures."""

    def test_nested_dict_of_dataframes(self, sample_dataframe, tmp_cache_dir):
        data = {
            "level1": {
                "inner_df": sample_dataframe,
            }
        }

        _save_cache(data, tmp_cache_dir)
        loaded = _load_cache(tmp_cache_dir)

        pd.testing.assert_frame_equal(loaded["level1"]["inner_df"], sample_dataframe, check_freq=False)


class TestNoneValues:
    """None values in dict should survive round-trip."""

    def test_none_preserved(self, sample_dataframe, tmp_cache_dir):
        data = {
            "raw": sample_dataframe,
            "missing": None,
        }

        _save_cache(data, tmp_cache_dir)
        loaded = _load_cache(tmp_cache_dir)

        assert loaded["missing"] is None
        pd.testing.assert_frame_equal(loaded["raw"], sample_dataframe, check_freq=False)


class TestLoadCacheEdgeCases:
    """Edge cases for _load_cache."""

    def test_returns_none_for_missing_dir(self, tmp_path):
        assert _load_cache(str(tmp_path / "nonexistent")) is None

    def test_returns_none_for_dir_without_manifest(self, tmp_path):
        # dir exists but no manifest
        assert _load_cache(str(tmp_path)) is None
