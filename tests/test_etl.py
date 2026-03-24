"""Tests for ETL dispatch logic, dependency resolution, and loader routing."""

import sys
import types
from unittest.mock import MagicMock, patch

import pandas as pd
import pytest

# ---------------------------------------------------------------------------
# We import ETL functions directly; the heavy clean.py imports happen at
# module level, so we patch them only where needed.
# ---------------------------------------------------------------------------
sys.path.insert(0, ".")

from ETL import (
    DATA_DICTIONARY,
    _DATA_DICT_LOADERS,
    _KWARGS_ONLY_LOADERS,
    _load_single,
    _resolve_dependencies,
)


# =====================================================================
# 1.  _resolve_dependencies
# =====================================================================

class TestResolveDependencies:
    """Dependency resolver correctly orders and auto-adds transitive deps."""

    def test_no_dependencies(self):
        """Keys with no deps come back unchanged."""
        result = _resolve_dependencies(["vixdata"])
        assert result == ["vixdata"]

    def test_direct_dependency_added(self):
        """Requesting 'stockdata' auto-adds its dep 'culturewardata' first."""
        result = _resolve_dependencies(["stockdata"])
        assert "culturewardata" in result
        assert result.index("culturewardata") < result.index("stockdata")

    def test_transitive_dependencies(self):
        """Requesting 'sec' category keys pulls culturewardata transitively."""
        result = _resolve_dependencies(["form4data", "sec_fundamentals"])
        assert "culturewardata" in result
        assert result.index("culturewardata") < result.index("form4data")
        assert result.index("culturewardata") < result.index("sec_fundamentals")

    def test_no_duplicates(self):
        """Deps shared by multiple keys appear only once."""
        result = _resolve_dependencies(
            ["stockdata", "form4data", "controlcompanies"]
        )
        assert result.count("culturewardata") == 1

    def test_unknown_keys_ignored(self):
        """Keys not in DATA_DICTIONARY are silently dropped."""
        result = _resolve_dependencies(["nonexistent_key", "vixdata"])
        assert "nonexistent_key" not in result
        assert "vixdata" in result


# =====================================================================
# 2.  _load_single — routing tests with mock loaders
# =====================================================================

class TestLoadSingleRouting:
    """_load_single dispatches to the correct calling convention."""

    def test_data_dict_loader_receives_data_dict(self):
        """Loaders in _DATA_DICT_LOADERS get data_dict as first arg."""
        mock_loader = MagicMock(return_value=pd.DataFrame({"x": [1]}))
        # Temporarily add to the set so _load_single recognises it
        _DATA_DICT_LOADERS.add(mock_loader)
        try:
            entry = {
                "loader": mock_loader,
                "args": (),
                "kwargs": {"start_date": "2020-01-01", "end_date": "2025-01-01"},
                "depends_on": [],
            }
            existing = {"culturewardata": pd.DataFrame()}
            _load_single("test_key", entry, existing, None, None, False)

            mock_loader.assert_called_once()
            # First positional arg must be the data_dict
            assert mock_loader.call_args[0][0] is existing
        finally:
            _DATA_DICT_LOADERS.discard(mock_loader)

    def test_kwargs_only_loader_no_positional_args(self):
        """Loaders in _KWARGS_ONLY_LOADERS get only **kwargs."""
        mock_loader = MagicMock(return_value=pd.DataFrame({"x": [1]}))
        _KWARGS_ONLY_LOADERS.add(mock_loader)
        try:
            entry = {
                "loader": mock_loader,
                "args": (),
                "kwargs": {},
                "depends_on": [],
            }
            _load_single("test_key", entry, {}, None, None, False)

            mock_loader.assert_called_once()
            # No positional args
            assert mock_loader.call_args[0] == ()
        finally:
            _KWARGS_ONLY_LOADERS.discard(mock_loader)

    def test_standard_loader_gets_args_and_kwargs(self):
        """Standard FRED-style loaders receive *args and **kwargs."""
        mock_loader = MagicMock(return_value=pd.DataFrame({"x": [1]}))
        entry = {
            "loader": mock_loader,
            "args": ("arg1",),
            "kwargs": {"start_date": "2020-01-01", "cache_path": "./data"},
            "depends_on": [],
        }
        _load_single("test_key", entry, {}, None, None, False)

        mock_loader.assert_called_once_with(
            "arg1", start_date="2020-01-01", cache_path="./data"
        )


# =====================================================================
# 3.  Wrapper return-None-on-missing-dependency behaviour
# =====================================================================

class TestWrapperMissingDependencies:
    """Wrapper functions return None (not raise) when deps are absent."""

    def test_load_stockdata_returns_none_without_culturewardata(self):
        from ETL import _load_stockdata

        result = _load_stockdata({})
        assert result is None

    def test_load_controlcompanies_returns_none_without_culturewardata(self):
        from ETL import _load_controlcompanies

        result = _load_controlcompanies({})
        assert result is None

    def test_load_form4_returns_none_without_culturewardata(self):
        from ETL import _load_form4

        result = _load_form4({})
        assert result is None

    def test_load_sec_fundamentals_returns_none_without_culturewardata(self):
        from ETL import _load_sec_fundamentals

        result = _load_sec_fundamentals({})
        assert result is None
