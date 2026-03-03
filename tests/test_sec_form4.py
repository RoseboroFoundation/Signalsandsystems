"""Tests for clean/sec_form4.py — XML parsing logic."""

from unittest.mock import patch, MagicMock

import pytest
from bs4 import BeautifulSoup

from clean.sec_form4 import Form4Downloader


class TestParseForm4XML:
    """Test the XML parsing extracted from parse_form4_xml.

    We test the BeautifulSoup parsing logic directly using fixtures,
    bypassing the HTTP requests.
    """

    def _parse_xml(self, xml_string):
        """Helper: parse XML string the same way Form4Downloader does."""
        soup = BeautifulSoup(xml_string, "xml")

        reporting_owner = soup.find("reportingOwner")
        if reporting_owner:
            owner_name_tag = reporting_owner.find("rptOwnerName")
            owner_name = owner_name_tag.text if owner_name_tag else "Unknown"
        else:
            owner_name = "Unknown"

        transactions = []
        for trans in soup.find_all("nonDerivativeTransaction"):
            trans_date_tag = trans.find("transactionDate")
            trans_code_tag = trans.find("transactionCode")
            shares_tag = trans.find("transactionShares")
            price_tag = trans.find("transactionPricePerShare")
            acq_disp_tag = trans.find("transactionAcquiredDisposedCode")
            shares_owned_tag = trans.find("sharesOwnedFollowingTransaction")

            transaction = {
                "owner_name": owner_name,
                "transaction_date": (
                    trans_date_tag.find("value").text
                    if trans_date_tag and trans_date_tag.find("value")
                    else None
                ),
                "transaction_code": (
                    trans_code_tag.text if trans_code_tag else None
                ),
                "shares": (
                    float(shares_tag.find("value").text)
                    if shares_tag and shares_tag.find("value")
                    else 0
                ),
                "price_per_share": (
                    float(price_tag.find("value").text)
                    if price_tag and price_tag.find("value")
                    else None
                ),
                "acquired_disposed": (
                    acq_disp_tag.find("value").text
                    if acq_disp_tag and acq_disp_tag.find("value")
                    else None
                ),
                "shares_owned_after": (
                    float(shares_owned_tag.find("value").text)
                    if shares_owned_tag and shares_owned_tag.find("value")
                    else None
                ),
            }
            transactions.append(transaction)

        return transactions

    def test_parses_two_transactions(self, form4_xml):
        txns = self._parse_xml(form4_xml)
        assert len(txns) == 2

    def test_owner_name_extracted(self, form4_xml):
        txns = self._parse_xml(form4_xml)
        assert txns[0]["owner_name"] == "DOE JOHN"
        assert txns[1]["owner_name"] == "DOE JOHN"

    def test_sale_transaction_fields(self, form4_xml):
        txns = self._parse_xml(form4_xml)
        sale = txns[0]

        assert sale["transaction_date"] == "2024-06-15"
        assert sale["transaction_code"] == "S"
        assert sale["shares"] == 5000.0
        assert sale["price_per_share"] == pytest.approx(150.25)
        assert sale["acquired_disposed"] == "D"
        assert sale["shares_owned_after"] == 45000.0

    def test_purchase_transaction_fields(self, form4_xml):
        txns = self._parse_xml(form4_xml)
        purchase = txns[1]

        assert purchase["transaction_date"] == "2024-06-16"
        assert purchase["transaction_code"] == "P"
        assert purchase["shares"] == 2000.0
        assert purchase["price_per_share"] == pytest.approx(148.00)
        assert purchase["acquired_disposed"] == "A"
        assert purchase["shares_owned_after"] == 47000.0

    def test_missing_owner_defaults_to_unknown(self, form4_xml_missing_fields):
        txns = self._parse_xml(form4_xml_missing_fields)
        assert txns[0]["owner_name"] == "Unknown"

    def test_missing_optional_fields(self, form4_xml_missing_fields):
        txns = self._parse_xml(form4_xml_missing_fields)
        txn = txns[0]

        assert txn["transaction_date"] == "2024-07-01"
        assert txn["shares"] == 1000.0
        # Missing fields default to None
        assert txn["transaction_code"] is None
        assert txn["price_per_share"] is None
        assert txn["acquired_disposed"] is None
        assert txn["shares_owned_after"] is None

    def test_empty_xml_returns_no_transactions(self):
        xml = '<?xml version="1.0"?><ownershipDocument></ownershipDocument>'
        txns = self._parse_xml(xml)
        assert txns == []


class TestParseForm4XMLIntegration:
    """Test parse_form4_xml with mocked HTTP responses."""

    def test_full_parse_with_mocked_requests(self, form4_filing_html, form4_xml):
        downloader = Form4Downloader(output_dir="/tmp/test_form4")

        filing_response = MagicMock()
        filing_response.status_code = 200
        filing_response.text = form4_filing_html

        xml_response = MagicMock()
        xml_response.status_code = 200
        xml_response.content = form4_xml.encode("utf-8")

        with patch("clean.sec_form4.requests.get") as mock_get:
            mock_get.side_effect = [filing_response, xml_response]
            txns = downloader.parse_form4_xml(
                "https://www.sec.gov/Archives/edgar/data/123/index.html"
            )

        assert len(txns) == 2
        assert txns[0]["owner_name"] == "DOE JOHN"
        assert txns[0]["shares"] == 5000.0

    def test_returns_empty_on_404(self):
        downloader = Form4Downloader(output_dir="/tmp/test_form4")

        response = MagicMock()
        response.status_code = 404

        with patch("clean.sec_form4.requests.get", return_value=response):
            txns = downloader.parse_form4_xml("https://www.sec.gov/fake")

        assert txns == []

    def test_returns_empty_when_no_xml_link(self):
        downloader = Form4Downloader(output_dir="/tmp/test_form4")

        response = MagicMock()
        response.status_code = 200
        response.text = "<html><body><p>No links here</p></body></html>"

        with patch("clean.sec_form4.requests.get", return_value=response):
            txns = downloader.parse_form4_xml("https://www.sec.gov/fake")

        assert txns == []
