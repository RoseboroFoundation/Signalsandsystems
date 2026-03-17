"""Tests for clean/sec_filings.py — 10-K/10-Q filing parsing."""

from unittest.mock import patch, MagicMock
from datetime import datetime

import pandas as pd
import pytest

from clean.sec_filings import SECFilingDownloader


# ── Fixtures ─────────────────────────────────────────────────────────────

@pytest.fixture
def downloader(tmp_path):
    return SECFilingDownloader(output_dir=str(tmp_path / "sec"))


@pytest.fixture
def edgar_batch():
    """Minimal EDGAR submissions batch with 4 filings."""
    return {
        'form': ['10-K', '10-Q', '10-K/A', '8-K', '10-Q'],
        'filingDate': [
            '2022-02-15', '2022-05-10', '2022-03-01',
            '2022-06-01', '2021-11-05',
        ],
        'accessionNumber': [
            '0001234567-22-000001',
            '0001234567-22-000002',
            '0001234567-22-000003',
            '0001234567-22-000004',
            '0001234567-21-000005',
        ],
        'primaryDocument': [
            'aapl-20211231.htm',
            'aapl-20220331.htm',
            'aapl-20211231a.htm',
            'aapl-8k.htm',
            'aapl-20210930.htm',
        ],
        'reportDate': [
            '2021-12-31', '2022-03-31', '2021-12-31',
            '', '2021-09-30',
        ],
    }


@pytest.fixture
def companyfacts_json():
    """Minimal XBRL companyfacts JSON with Revenue and NetIncome."""
    return {
        'facts': {
            'us-gaap': {
                'Revenues': {
                    'units': {
                        'USD': [
                            {
                                'form': '10-K',
                                'end': '2022-12-31',
                                'filed': '2023-02-15',
                                'val': 100_000_000,
                            },
                            {
                                'form': '10-Q',
                                'end': '2022-09-30',
                                'filed': '2022-11-01',
                                'val': 25_000_000,
                            },
                            {
                                'form': '8-K',  # should be skipped
                                'end': '2022-06-30',
                                'filed': '2022-07-01',
                                'val': 999,
                            },
                        ],
                    },
                },
                'NetIncomeLoss': {
                    'units': {
                        'USD': [
                            {
                                'form': '10-K',
                                'end': '2022-12-31',
                                'filed': '2023-02-15',
                                'val': 10_000_000,
                            },
                        ],
                    },
                },
                'EarningsPerShareBasic': {
                    'units': {
                        'USD/shares': [
                            {
                                'form': '10-K',
                                'end': '2022-12-31',
                                'filed': '2023-02-15',
                                'val': 5.25,
                            },
                        ],
                    },
                },
            },
        },
    }


# ── _collect_filings_from_batch ──────────────────────────────────────────

class TestCollectFilingsFromBatch:
    """Test the pure parsing of EDGAR submission batches."""

    def test_filters_to_target_forms(self, downloader, edgar_batch):
        filings = downloader._collect_filings_from_batch(
            edgar_batch,
            target_forms=('10-K',),
            start_dt=datetime(2020, 1, 1),
            end_dt=datetime(2025, 12, 31),
            ticker='AAPL',
            cik='0000320193',
        )
        assert len(filings) == 1
        assert filings[0]['form_type'] == '10-K'

    def test_filters_by_date_range(self, downloader, edgar_batch):
        filings = downloader._collect_filings_from_batch(
            edgar_batch,
            target_forms=('10-K', '10-Q', '10-K/A', '10-Q/A'),
            start_dt=datetime(2022, 1, 1),
            end_dt=datetime(2022, 12, 31),
            ticker='AAPL',
            cik='0000320193',
        )
        # 2021-11-05 10-Q is out of range
        assert all(f['filing_date'] >= '2022-01-01' for f in filings)

    def test_excludes_non_target_forms(self, downloader, edgar_batch):
        filings = downloader._collect_filings_from_batch(
            edgar_batch,
            target_forms=('10-K', '10-Q', '10-K/A', '10-Q/A'),
            start_dt=datetime(2020, 1, 1),
            end_dt=datetime(2025, 12, 31),
            ticker='AAPL',
            cik='0000320193',
        )
        form_types = [f['form_type'] for f in filings]
        assert '8-K' not in form_types

    def test_filing_dict_structure(self, downloader, edgar_batch):
        filings = downloader._collect_filings_from_batch(
            edgar_batch,
            target_forms=('10-K',),
            start_dt=datetime(2020, 1, 1),
            end_dt=datetime(2025, 12, 31),
            ticker='AAPL',
            cik='0000320193',
        )
        f = filings[0]
        assert f['ticker'] == 'AAPL'
        assert f['cik'] == '0000320193'
        assert f['form_type'] == '10-K'
        assert f['filing_date'] == '2022-02-15'
        assert f['report_date'] == '2021-12-31'
        assert f['accession_number'] == '0001234567-22-000001'
        assert f['primary_document'] == 'aapl-20211231.htm'
        assert 'sec.gov' in f['filing_url']

    def test_filing_url_format(self, downloader, edgar_batch):
        filings = downloader._collect_filings_from_batch(
            edgar_batch,
            target_forms=('10-K',),
            start_dt=datetime(2020, 1, 1),
            end_dt=datetime(2025, 12, 31),
            ticker='AAPL',
            cik='0000320193',
        )
        url = filings[0]['filing_url']
        # CIK stripped of leading zeros, accession without dashes
        assert '/320193/' in url
        assert '000123456722000001' in url
        assert url.endswith('aapl-20211231.htm')

    def test_empty_batch(self, downloader):
        filings = downloader._collect_filings_from_batch(
            {'form': [], 'filingDate': [], 'accessionNumber': [],
             'primaryDocument': [], 'reportDate': []},
            target_forms=('10-K',),
            start_dt=datetime(2020, 1, 1),
            end_dt=datetime(2025, 12, 31),
            ticker='AAPL',
            cik='0000320193',
        )
        assert filings == []


# ── extract_financial_metrics ────────────────────────────────────────────

class TestExtractFinancialMetrics:
    """Test XBRL companyfacts extraction via the real source function."""

    def test_extracts_revenue_and_net_income(self, downloader, companyfacts_json):
        with patch.object(downloader, '_get_company_facts', return_value=companyfacts_json):
            df = downloader.extract_financial_metrics('0000320193', ticker='AAPL')

        assert not df.empty
        annual = df[df['FORM_TYPE'] == '10-K']
        assert len(annual) == 1
        assert annual.iloc[0]['REVENUE'] == 100_000_000
        assert annual.iloc[0]['NET_INCOME'] == 10_000_000

    def test_extracts_eps_from_usd_shares_units(self, downloader, companyfacts_json):
        with patch.object(downloader, '_get_company_facts', return_value=companyfacts_json):
            df = downloader.extract_financial_metrics('0000320193', ticker='AAPL')

        annual = df[df['FORM_TYPE'] == '10-K']
        assert annual.iloc[0]['EPS_BASIC'] == 5.25

    def test_skips_8k_filings(self, downloader, companyfacts_json):
        with patch.object(downloader, '_get_company_facts', return_value=companyfacts_json):
            df = downloader.extract_financial_metrics('0000320193', ticker='AAPL')

        # 8-K entry with val=999 should not appear
        assert 999 not in df['REVENUE'].values

    def test_includes_quarterly_data(self, downloader, companyfacts_json):
        with patch.object(downloader, '_get_company_facts', return_value=companyfacts_json):
            df = downloader.extract_financial_metrics('0000320193', ticker='AAPL')

        quarterly = df[df['FORM_TYPE'] == '10-Q']
        assert len(quarterly) == 1
        assert quarterly.iloc[0]['REVENUE'] == 25_000_000

    def test_period_end_is_datetime(self, downloader, companyfacts_json):
        with patch.object(downloader, '_get_company_facts', return_value=companyfacts_json):
            df = downloader.extract_financial_metrics('0000320193', ticker='AAPL')

        assert pd.api.types.is_datetime64_any_dtype(df['PERIOD_END'])

    def test_returns_empty_when_no_facts(self, downloader):
        with patch.object(downloader, '_get_company_facts', return_value=None):
            df = downloader.extract_financial_metrics('0000320193', ticker='AAPL')

        assert df.empty

    def test_returns_empty_when_no_us_gaap(self, downloader):
        facts = {'facts': {}}
        with patch.object(downloader, '_get_company_facts', return_value=facts):
            df = downloader.extract_financial_metrics('0000320193', ticker='AAPL')

        assert df.empty

    def test_amendment_merged_with_base_form(self, downloader):
        """10-K/A entries should be keyed as 10-K (merged, not duplicated)."""
        facts = {
            'facts': {
                'us-gaap': {
                    'Revenues': {
                        'units': {
                            'USD': [
                                {'form': '10-K/A', 'end': '2022-12-31',
                                 'filed': '2023-04-01', 'val': 105_000_000},
                            ],
                        },
                    },
                },
            },
        }
        with patch.object(downloader, '_get_company_facts', return_value=facts):
            df = downloader.extract_financial_metrics('0000320193', ticker='AAPL')

        assert len(df) == 1
        assert df.iloc[0]['FORM_TYPE'] == '10-K'
        assert df.iloc[0]['REVENUE'] == 105_000_000
