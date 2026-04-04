"""Tests for clean/fred_loaders.py and clean/config.py FRED helpers."""

from unittest.mock import patch, MagicMock

import pandas as pd
import pytest

from clean.config import _download_fred_series, _validate_fred_api_key


class TestValidateFredApiKey:
    """Test API key validation."""

    def test_raises_without_key(self):
        with patch("clean.config.API_KEY", None):
            with pytest.raises(ValueError, match="FRED API key not found"):
                _validate_fred_api_key()

    def test_passes_with_key(self):
        with patch("clean.config.API_KEY", "test_key_123"):
            _validate_fred_api_key()  # should not raise


class TestDownloadFredSeries:
    """Test _download_fred_series with mocked DataReader."""

    @pytest.fixture
    def mock_fred_data(self):
        """Mock DataReader to return predictable data."""
        dates = pd.date_range("2020-01-01", periods=6, freq="MS")
        cpi = pd.DataFrame({"CPIAUCSL": [100, 101, 102, 103, 104, 105]}, index=dates)
        core = pd.DataFrame({"CPILFESL": [200, 201, 202, 203, 204, 205]}, index=dates)
        return {"CPIAUCSL": cpi, "CPILFESL": core}

    def test_returns_dataframe_with_correct_columns(self, mock_fred_data):
        def side_effect(code, source, start, end):
            return mock_fred_data[code]

        with patch("clean.config.API_KEY", "test_key"):
            with patch("pandas_datareader.DataReader", side_effect=side_effect):
                result = _download_fred_series(
                    {"CPI": "CPIAUCSL", "Core_CPI": "CPILFESL"},
                    "2020-01-01",
                    "2020-06-01",
                )

        assert isinstance(result, pd.DataFrame)
        assert list(result.columns) == ["CPI", "Core_CPI"]
        assert len(result) == 6

    def test_handles_failed_series_gracefully(self, mock_fred_data):
        def side_effect(code, source, start, end):
            if code == "CPILFESL":
                raise Exception("API error")
            return mock_fred_data[code]

        with patch("clean.config.API_KEY", "test_key"):
            with patch("pandas_datareader.DataReader", side_effect=side_effect):
                result = _download_fred_series(
                    {"CPI": "CPIAUCSL", "Core_CPI": "CPILFESL"},
                    "2020-01-01",
                    "2020-06-01",
                )

        # Should still return the successful series
        assert "CPI" in result.columns
        assert "Core_CPI" not in result.columns

    def test_raises_without_api_key(self):
        with patch("clean.config.API_KEY", None):
            with pytest.raises(ValueError):
                _download_fred_series({"CPI": "CPIAUCSL"}, "2020-01-01", "2020-06-01")


class TestInflationDataStructure:
    """Test load_inflation_data returns the expected dict structure."""

    def test_returns_expected_keys(self, tmp_path):
        dates = pd.date_range("2020-01-01", periods=24, freq="MS")
        mock_df = pd.DataFrame(
            {code: range(100, 124) for code in ["CPIAUCSL", "CPILFESL", "PCEPI", "PCEPILFE", "PPIACO", "GDPDEF"]},
            index=dates,
        )

        def side_effect(code, source, start, end):
            return pd.DataFrame({code: mock_df[code]})

        with patch("clean.config.API_KEY", "test_key"):
            with patch("pandas_datareader.DataReader", side_effect=side_effect):
                from clean.fred_loaders import load_inflation_data

                result = load_inflation_data(
                    start_date="2020-01-01",
                    end_date="2021-12-01",
                    cache_path=str(tmp_path / "fred"),
                )

        assert result is not None
        assert set(result.keys()) == {"raw", "yoy", "mom", "combined"}
        assert isinstance(result["raw"], pd.DataFrame)
        assert isinstance(result["yoy"], pd.DataFrame)

    def test_yoy_columns_have_suffix(self, tmp_path):
        dates = pd.date_range("2020-01-01", periods=24, freq="MS")
        mock_df = pd.DataFrame(
            {code: range(100, 124) for code in ["CPIAUCSL", "CPILFESL", "PCEPI", "PCEPILFE", "PPIACO", "GDPDEF"]},
            index=dates,
        )

        def side_effect(code, source, start, end):
            return pd.DataFrame({code: mock_df[code]})

        with patch("clean.config.API_KEY", "test_key"):
            with patch("pandas_datareader.DataReader", side_effect=side_effect):
                from clean.fred_loaders import load_inflation_data

                result = load_inflation_data(
                    start_date="2020-01-01",
                    end_date="2021-12-01",
                    cache_path=str(tmp_path / "fred2"),
                )

        assert all(col.endswith("_YoY") for col in result["yoy"].columns)
        assert all(col.endswith("_MoM") for col in result["mom"].columns)


class TestInflationMathCorrectness:
    """Verify YoY and MoM calculations produce mathematically correct values.

    Uses a hand-constructed CPI series where expected results can be
    computed on paper, so any regression in the pct_change logic is caught.
    """

    @pytest.fixture
    def inflation_result(self, tmp_path):
        """Run load_inflation_data with a known CPI series: 100, 101, ..., 123."""
        dates = pd.date_range("2020-01-01", periods=24, freq="MS")
        # Only provide CPI to keep the math simple
        values = list(range(100, 124))  # 100..123

        def side_effect(code, source, start, end):
            return pd.DataFrame({code: values}, index=dates)

        series_map = {
            "CPIAUCSL": "CPIAUCSL",
            "CPILFESL": "CPILFESL",
            "PCEPI": "PCEPI",
            "PCEPILFE": "PCEPILFE",
            "PPIACO": "PPIACO",
            "GDPDEF": "GDPDEF",
        }

        with patch("clean.config.API_KEY", "test_key"):
            with patch("pandas_datareader.DataReader", side_effect=side_effect):
                from clean.fred_loaders import load_inflation_data

                return load_inflation_data(
                    start_date="2020-01-01",
                    end_date="2021-12-01",
                    cache_path=str(tmp_path / "fred_math"),
                )

    def test_yoy_formula(self, inflation_result):
        """YoY = (value / value_12_months_ago - 1) * 100.

        At month index 12 (2021-01): value=112, 12 months prior=100.
        Expected YoY = (112/100 - 1)*100 = 12.0%
        """
        yoy = inflation_result["yoy"]
        # First 12 rows should be NaN (no prior year)
        assert yoy.iloc[:12].isnull().all().all()
        # Month 12: CPI went from 100 to 112 → 12%
        assert yoy["CPI_YoY"].iloc[12] == pytest.approx(12.0)
        # Month 23: CPI went from 111 to 123 → (123/111 - 1)*100 ≈ 10.81%
        assert yoy["CPI_YoY"].iloc[23] == pytest.approx((123 / 111 - 1) * 100)

    def test_mom_formula(self, inflation_result):
        """MoM (annualized) = (value / prev_value - 1) * 100 * 12.

        At month index 1 (2020-02): value=101, prior=100.
        Expected MoM = (101/100 - 1)*100*12 = 12.0%
        """
        mom = inflation_result["mom"]
        # First row should be NaN
        assert mom.iloc[0].isnull().all()
        # Month 1: 100→101, MoM annualized = (1/100)*100*12 = 12.0%
        assert mom["CPI_MoM"].iloc[1] == pytest.approx(12.0)
        # Month 10: 109→110, MoM annualized = (1/109)*100*12
        assert mom["CPI_MoM"].iloc[10] == pytest.approx((1 / 109) * 100 * 12)
