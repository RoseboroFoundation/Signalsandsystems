"""Tests for clean/sec_form4.py — XML parsing logic."""

from unittest.mock import patch, MagicMock

import pytest

from clean.sec_form4 import Form4Downloader


class TestParseForm4XML:
    """Test Form4Downloader._parse_xml_content() directly with XML fixtures."""

    def _parse(self, xml_string):
        """Call the real source function with XML bytes."""
        return Form4Downloader(output_dir="/tmp/test_form4")._parse_xml_content(
            xml_string.encode("utf-8")
        )

    def test_parses_two_transactions(self, form4_xml):
        txns = self._parse(form4_xml)
        assert len(txns) == 2

    def test_owner_name_extracted(self, form4_xml):
        txns = self._parse(form4_xml)
        assert txns[0]["owner_name"] == "DOE JOHN"
        assert txns[1]["owner_name"] == "DOE JOHN"

    def test_sale_transaction_fields(self, form4_xml):
        txns = self._parse(form4_xml)
        sale = txns[0]

        assert sale["transaction_date"] == "2024-06-15"
        assert sale["transaction_code"] == "S"
        assert sale["shares"] == 5000.0
        assert sale["price_per_share"] == pytest.approx(150.25)
        assert sale["acquired_disposed"] == "D"
        assert sale["shares_owned_after"] == 45000.0

    def test_purchase_transaction_fields(self, form4_xml):
        txns = self._parse(form4_xml)
        purchase = txns[1]

        assert purchase["transaction_date"] == "2024-06-16"
        assert purchase["transaction_code"] == "P"
        assert purchase["shares"] == 2000.0
        assert purchase["price_per_share"] == pytest.approx(148.00)
        assert purchase["acquired_disposed"] == "A"
        assert purchase["shares_owned_after"] == 47000.0

    def test_missing_owner_defaults_to_unknown(self, form4_xml_missing_fields):
        txns = self._parse(form4_xml_missing_fields)
        assert txns[0]["owner_name"] == "Unknown"

    def test_missing_optional_fields(self, form4_xml_missing_fields):
        txns = self._parse(form4_xml_missing_fields)
        txn = txns[0]

        assert txn["transaction_date"] == "2024-07-01"
        assert txn["shares"] == 1000.0
        assert txn["transaction_code"] is None
        assert txn["price_per_share"] is None
        assert txn["acquired_disposed"] is None
        assert txn["shares_owned_after"] is None

    def test_empty_xml_returns_no_transactions(self):
        xml = '<?xml version="1.0"?><ownershipDocument></ownershipDocument>'
        txns = self._parse(xml)
        assert txns == []


class TestParseForm4XMLIntegration:
    """Test parse_form4_xml with mocked HTTP responses.

    parse_form4_xml expects a filing dict with keys:
    filing_url, primary_document, accession_number, cik.
    """

    def _make_filing(self, primary_document="form4.xml"):
        return {
            "filing_url": "https://www.sec.gov/Archives/edgar/data/123/0001234567-24-000001/form4.xml",
            "primary_document": primary_document,
            "accession_number": "0001234567-24-000001",
            "cik": "123",
        }

    def test_full_parse_with_xml_primary_doc(self, form4_xml):
        downloader = Form4Downloader(output_dir="/tmp/test_form4")

        xml_response = MagicMock()
        xml_response.status_code = 200
        xml_response.content = form4_xml.encode("utf-8")

        with patch.object(downloader, "_request", return_value=xml_response):
            txns = downloader.parse_form4_xml(self._make_filing())

        assert len(txns) == 2
        assert txns[0]["owner_name"] == "DOE JOHN"
        assert txns[0]["shares"] == 5000.0

    def test_returns_empty_on_failed_request(self):
        downloader = Form4Downloader(output_dir="/tmp/test_form4")

        with patch.object(downloader, "_request", return_value=None):
            txns = downloader.parse_form4_xml(self._make_filing())

        assert txns == []

    def test_returns_empty_when_no_xml_found(self):
        downloader = Form4Downloader(output_dir="/tmp/test_form4")

        # Primary doc is HTML, and index page has no XML links
        html_response = MagicMock()
        html_response.status_code = 200
        html_response.text = "<html><body><p>No links here</p></body></html>"

        with patch.object(downloader, "_request", return_value=html_response):
            txns = downloader.parse_form4_xml(
                self._make_filing(primary_document="filing.htm")
            )

        assert txns == []
