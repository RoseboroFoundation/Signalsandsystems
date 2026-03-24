"""Tests for clean/sec_form4.py — XML parsing logic."""

from unittest.mock import patch, MagicMock

import pytest

from clean.sec_form4 import Form4Downloader


class TestParseForm4XML:
    """Test Form4Downloader.parse_form4_xml() with mocked HTTP responses.

    parse_form4_xml takes a filing URL string, fetches the index page,
    finds an XML link, fetches the XML, and parses transactions.
    """

    def _parse_with_xml(self, xml_string):
        """Mock both HTTP calls (index page → XML link → XML doc) and parse."""
        downloader = Form4Downloader(output_dir="/tmp/test_form4")

        index_html = (
            '<html><body><table>'
            '<tr><td><a href="/Archives/edgar/data/123/filing.xml">filing.xml</a></td></tr>'
            '</table></body></html>'
        )

        index_response = MagicMock()
        index_response.status_code = 200
        index_response.text = index_html

        xml_response = MagicMock()
        xml_response.status_code = 200
        xml_response.content = xml_string.encode("utf-8")

        with patch("clean.sec_form4.requests.get", side_effect=[index_response, xml_response]):
            return downloader.parse_form4_xml(
                "https://www.sec.gov/Archives/edgar/data/123/0001234567-24-000001-index.htm"
            )

    def test_parses_two_transactions(self, form4_xml):
        txns = self._parse_with_xml(form4_xml)
        assert len(txns) == 2

    def test_owner_name_extracted(self, form4_xml):
        txns = self._parse_with_xml(form4_xml)
        assert txns[0]["owner_name"] == "DOE JOHN"
        assert txns[1]["owner_name"] == "DOE JOHN"

    def test_sale_transaction_fields(self, form4_xml):
        txns = self._parse_with_xml(form4_xml)
        sale = txns[0]

        assert sale["transaction_date"] == "2024-06-15"
        assert sale["transaction_code"] == "S"
        assert sale["shares"] == 5000.0
        assert sale["price_per_share"] == pytest.approx(150.25)
        assert sale["acquired_disposed"] == "D"
        assert sale["shares_owned_after"] == 45000.0

    def test_purchase_transaction_fields(self, form4_xml):
        txns = self._parse_with_xml(form4_xml)
        purchase = txns[1]

        assert purchase["transaction_date"] == "2024-06-16"
        assert purchase["transaction_code"] == "P"
        assert purchase["shares"] == 2000.0
        assert purchase["price_per_share"] == pytest.approx(148.00)
        assert purchase["acquired_disposed"] == "A"
        assert purchase["shares_owned_after"] == 47000.0

    def test_missing_owner_defaults_to_unknown(self, form4_xml_missing_fields):
        txns = self._parse_with_xml(form4_xml_missing_fields)
        assert txns[0]["owner_name"] == "Unknown"

    def test_missing_optional_fields(self, form4_xml_missing_fields):
        txns = self._parse_with_xml(form4_xml_missing_fields)
        txn = txns[0]

        assert txn["transaction_date"] == "2024-07-01"
        assert txn["shares"] == 1000.0
        assert txn["transaction_code"] is None
        assert txn["price_per_share"] is None
        assert txn["acquired_disposed"] is None
        assert txn["shares_owned_after"] is None

    def test_empty_xml_returns_no_transactions(self):
        txns = self._parse_with_xml(
            '<?xml version="1.0"?><ownershipDocument></ownershipDocument>'
        )
        assert txns == []


class TestParseForm4XMLIntegration:
    """Test parse_form4_xml edge cases with mocked HTTP."""

    def test_returns_empty_on_failed_request(self):
        downloader = Form4Downloader(output_dir="/tmp/test_form4")

        with patch("clean.sec_form4.requests.get", side_effect=Exception("network error")):
            txns = downloader.parse_form4_xml("https://www.sec.gov/some/filing")

        assert txns == []

    def test_returns_empty_on_non_200(self):
        downloader = Form4Downloader(output_dir="/tmp/test_form4")

        response = MagicMock()
        response.status_code = 404

        with patch("clean.sec_form4.requests.get", return_value=response):
            txns = downloader.parse_form4_xml("https://www.sec.gov/some/filing")

        assert txns == []

    def test_returns_empty_when_no_xml_link(self):
        downloader = Form4Downloader(output_dir="/tmp/test_form4")

        response = MagicMock()
        response.status_code = 200
        response.text = "<html><body><p>No links here</p></body></html>"

        with patch("clean.sec_form4.requests.get", return_value=response):
            txns = downloader.parse_form4_xml("https://www.sec.gov/some/filing")

        assert txns == []
