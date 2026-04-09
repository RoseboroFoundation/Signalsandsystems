"""Shared fixtures for the test suite."""

import os
import sys
import tempfile
from unittest.mock import MagicMock

import pandas as pd
import pytest

# Inject a fake pandas_datareader into sys.modules BEFORE any test imports
# trigger the real one.  The real package is broken on pandas 2.2+
# (deprecate_kwarg signature change).  Individual tests that need DataReader
# behaviour set side_effect on this mock's DataReader attribute.
if "pandas_datareader" not in sys.modules:
    sys.modules["pandas_datareader"] = MagicMock()


@pytest.fixture
def sample_dataframe():
    """A small DataFrame typical of FRED time-series data."""
    dates = pd.date_range("2020-01-01", periods=12, freq="MS")
    return pd.DataFrame(
        {"CPI": range(100, 112), "Core_CPI": range(200, 212)},
        index=dates,
    )


@pytest.fixture
def sample_dict_of_dataframes(sample_dataframe):
    """A dict-of-DataFrames like load_inflation_data() returns."""
    yoy = sample_dataframe.pct_change(periods=12) * 100
    yoy.columns = [f"{c}_YoY" for c in yoy.columns]
    return {
        "raw": sample_dataframe,
        "yoy": yoy,
        "combined": pd.concat([sample_dataframe, yoy], axis=1),
    }


@pytest.fixture
def tmp_cache_dir(tmp_path):
    """Temporary directory for cache round-trip tests."""
    return str(tmp_path / "cache")


@pytest.fixture
def form4_xml():
    """Minimal SEC Form 4 XML document."""
    return """<?xml version="1.0" encoding="UTF-8"?>
<ownershipDocument>
    <reportingOwner>
        <reportingOwnerId>
            <rptOwnerCik>0001234567</rptOwnerCik>
            <rptOwnerName>DOE JOHN</rptOwnerName>
        </reportingOwnerId>
    </reportingOwner>
    <nonDerivativeTable>
        <nonDerivativeTransaction>
            <transactionDate><value>2024-06-15</value></transactionDate>
            <transactionCoding>
                <transactionCode>S</transactionCode>
            </transactionCoding>
            <transactionAmounts>
                <transactionShares><value>5000</value></transactionShares>
                <transactionPricePerShare><value>150.25</value></transactionPricePerShare>
                <transactionAcquiredDisposedCode><value>D</value></transactionAcquiredDisposedCode>
            </transactionAmounts>
            <postTransactionAmounts>
                <sharesOwnedFollowingTransaction><value>45000</value></sharesOwnedFollowingTransaction>
            </postTransactionAmounts>
        </nonDerivativeTransaction>
        <nonDerivativeTransaction>
            <transactionDate><value>2024-06-16</value></transactionDate>
            <transactionCoding>
                <transactionCode>P</transactionCode>
            </transactionCoding>
            <transactionAmounts>
                <transactionShares><value>2000</value></transactionShares>
                <transactionPricePerShare><value>148.00</value></transactionPricePerShare>
                <transactionAcquiredDisposedCode><value>A</value></transactionAcquiredDisposedCode>
            </transactionAmounts>
            <postTransactionAmounts>
                <sharesOwnedFollowingTransaction><value>47000</value></sharesOwnedFollowingTransaction>
            </postTransactionAmounts>
        </nonDerivativeTransaction>
    </nonDerivativeTable>
</ownershipDocument>"""


@pytest.fixture
def form4_xml_missing_fields():
    """Form 4 XML with missing optional fields."""
    return """<?xml version="1.0" encoding="UTF-8"?>
<ownershipDocument>
    <nonDerivativeTable>
        <nonDerivativeTransaction>
            <transactionDate><value>2024-07-01</value></transactionDate>
            <transactionAmounts>
                <transactionShares><value>1000</value></transactionShares>
            </transactionAmounts>
        </nonDerivativeTransaction>
    </nonDerivativeTable>
</ownershipDocument>"""


@pytest.fixture
def form4_filing_html():
    """Minimal SEC filing index page with an XML link."""
    return """<html><body>
<table>
    <tr><td><a href="/Archives/edgar/data/123/filing.xml">filing.xml</a></td></tr>
    <tr><td><a href="/Archives/edgar/data/123/primary_doc.xml">primary_doc.xml</a></td></tr>
</table>
</body></html>"""
