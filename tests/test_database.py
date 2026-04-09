"""Tests for Database.py — covers the blocking items from Dr. Lambert's review."""

import logging
import time
from datetime import datetime, timezone
from io import BytesIO
from unittest.mock import MagicMock, patch, call

import pandas as pd
import pytest

from Database import (
    AthenaLoader,
    BaseLoader,
    SQLiteLoader,
    _extract_dataframe,
    _prepare_dataframe,
    _retry,
)


# =============================================================================
# _prepare_dataframe
# =============================================================================
class TestPrepareDataframe:
    def test_dedup_keeps_original_renames_duplicates(self):
        """Original column kept clean, first duplicate gets _2, second gets _3."""
        df = pd.DataFrame([[1, 2, 3, 4]], columns=['A', 'A', 'B', 'A'])
        result = _prepare_dataframe(df)
        assert list(result.columns) == ['A', 'A_2', 'B', 'A_3']

    def test_datetime_index_resets_to_date_column(self):
        dates = pd.date_range('2020-01-01', periods=3)
        df = pd.DataFrame({'val': [1, 2, 3]}, index=dates)
        result = _prepare_dataframe(df)
        assert 'DATE' in result.columns
        assert not isinstance(result.index, pd.DatetimeIndex)

    def test_timezone_aware_to_naive_utc(self):
        dates = pd.to_datetime(['2020-01-01', '2020-01-02']).tz_localize('US/Eastern')
        df = pd.DataFrame({'TS': dates, 'VAL': [1, 2]})
        result = _prepare_dataframe(df)
        assert result['TS'].dt.tz is None
        # Verify the UTC offset was actually applied (not just stripped)
        expected = dates.tz_convert('UTC').tz_localize(None)
        pd.testing.assert_series_equal(result['TS'], pd.Series(expected, name='TS'))

    def test_period_index_columns_to_timestamps(self):
        periods = pd.period_range('2020-01', periods=3, freq='M')
        df = pd.DataFrame({'PERIOD': periods, 'VAL': [1, 2, 3]})
        result = _prepare_dataframe(df)
        assert pd.api.types.is_datetime64_any_dtype(result['PERIOD'])


# =============================================================================
# _extract_dataframe
# =============================================================================
class TestExtractDataframe:
    def test_dataframe_mode(self):
        df = pd.DataFrame({'A': [1]})
        result = _extract_dataframe('key', df, 'dataframe')
        assert result is df

    def test_combined_mode(self):
        inner = pd.DataFrame({'X': [1, 2]})
        result = _extract_dataframe('key', {'combined': inner}, 'combined')
        assert result is inner

    def test_concat_mode(self):
        d = {
            'a': pd.DataFrame({'X': [1]}),
            'b': pd.DataFrame({'X': [2]}),
        }
        result = _extract_dataframe('key', d, 'concat')
        assert len(result) == 2
        assert list(result['X']) == [1, 2]

    def test_nested_mode(self):
        inner = pd.DataFrame({'Z': [10]})
        result = _extract_dataframe('key', {'combined': inner}, 'nested')
        assert result is inner

    def test_unknown_mode_returns_none(self):
        result = _extract_dataframe('key', pd.DataFrame(), 'bogus_mode')
        assert result is None


# =============================================================================
# _retry
# =============================================================================
class TestRetry:
    def test_succeeds_after_failures(self):
        counter = {'n': 0}

        def flaky():
            counter['n'] += 1
            if counter['n'] < 3:
                exc = Exception("throttle")
                exc.response = {'Error': {'Code': 'ThrottlingException'}}
                raise exc
            return 'ok'

        with patch('Database.time.sleep'):
            result = _retry(flaky, max_attempts=3, base_delay=0.01)
        assert result == 'ok'
        assert counter['n'] == 3

    def test_non_retryable_raises_immediately(self):
        counter = {'n': 0}

        def bad():
            counter['n'] += 1
            exc = Exception("access denied")
            exc.response = {'Error': {'Code': 'AccessDeniedException'}}
            raise exc

        with pytest.raises(Exception, match="access denied"):
            _retry(bad, max_attempts=5, base_delay=0.01)
        assert counter['n'] == 1


# =============================================================================
# AthenaLoader._write_table (mocked boto3)
# =============================================================================
class TestAthenaWriteTable:
    def _make_loader(self):
        loader = AthenaLoader.__new__(AthenaLoader)
        loader.database = 'test_db'
        loader.workgroup = 'test_wg'
        loader.s3_bucket = 'test-bucket'
        loader.s3_prefix = 'test_prefix'
        loader._s3 = MagicMock()
        loader._glue = MagicMock()
        loader._athena = MagicMock()
        loader._connected = True
        loader._glue.get_table.return_value = {'Table': {'StorageDescriptor': {'Columns': []}}}
        return loader

    def test_write_table_happy_path(self):
        loader = self._make_loader()
        df = pd.DataFrame({'date': pd.date_range('2020-01-01', periods=3), 'val': [1, 2, 3]})

        loader._s3.list_objects_v2.return_value = {'Contents': [], 'IsTruncated': False}

        result = loader._write_table(df, 'TEST_TABLE', replace=True, etl_key='test', verbose=False)

        assert result['status'] == 'SUCCESS'
        assert result['rows'] == 3
        loader._s3.upload_fileobj.assert_called_once()
        upload_args = loader._s3.upload_fileobj.call_args
        assert upload_args[0][1] == 'test-bucket'
        assert 'test_table/data.parquet' in upload_args[0][2]

    def test_replace_clears_stale_s3_files(self):
        loader = self._make_loader()
        df = pd.DataFrame({'val': [1]})

        loader._s3.list_objects_v2.return_value = {
            'Contents': [
                {'Key': 'test_prefix/test_table/data_20260324_151500.parquet'},
                {'Key': 'test_prefix/test_table/data_20260323_100000.parquet'},
            ],
            'IsTruncated': False,
        }

        result = loader._write_table(df, 'TEST_TABLE', replace=True, etl_key='test', verbose=False)

        assert result['status'] == 'SUCCESS'
        loader._s3.delete_objects.assert_called_once()
        deleted_keys = loader._s3.delete_objects.call_args[1]['Delete']['Objects']
        assert len(deleted_keys) == 2

    def test_clear_s3_prefix_paginates_beyond_1000(self):
        """list_objects_v2 returns max 1000 keys per call — verify we loop."""
        loader = self._make_loader()

        page1_objects = [{'Key': f'test_prefix/big_table/file_{i}.parquet'} for i in range(1000)]
        page2_objects = [{'Key': f'test_prefix/big_table/file_{i}.parquet'} for i in range(1000, 1200)]

        loader._s3.list_objects_v2.side_effect = [
            {
                'Contents': page1_objects,
                'IsTruncated': True,
                'NextContinuationToken': 'token123',
            },
            {
                'Contents': page2_objects,
                'IsTruncated': False,
            },
        ]

        loader._clear_s3_prefix('big_table')

        assert loader._s3.list_objects_v2.call_count == 2
        assert loader._s3.delete_objects.call_count == 2
        second_list_call = loader._s3.list_objects_v2.call_args_list[1]
        assert second_list_call[1]['ContinuationToken'] == 'token123'

    def test_orphan_message_only_when_s3_uploaded(self, caplog):
        """Orphan warning fires only when S3 succeeded but Glue failed."""
        loader = self._make_loader()
        df = pd.DataFrame({'val': [1]})

        # S3 upload succeeds, Glue registration raises
        loader._s3.list_objects_v2.return_value = {'Contents': [], 'IsTruncated': False}
        loader._glue.exceptions = MagicMock()
        loader._glue.exceptions.EntityNotFoundException = type('EntityNotFoundException', (Exception,), {})
        loader._glue.update_table.side_effect = Exception("Glue boom")
        loader._glue.create_table.side_effect = Exception("Glue boom")

        with caplog.at_level(logging.ERROR, logger='Database'):
            result = loader._write_table(df, 'TEST_TABLE', replace=True, etl_key='test', verbose=False)

        assert result['status'] == 'FAILED'
        assert 'S3 data orphaned' in caplog.text

    def test_no_orphan_message_when_s3_failed(self, caplog):
        """No orphan warning when S3 upload itself failed."""
        loader = self._make_loader()
        df = pd.DataFrame({'val': [1]})

        loader._s3.list_objects_v2.return_value = {'Contents': [], 'IsTruncated': False}
        loader._s3.upload_fileobj.side_effect = Exception("S3 boom")

        with caplog.at_level(logging.ERROR, logger='Database'):
            result = loader._write_table(df, 'TEST_TABLE', replace=True, etl_key='test', verbose=False)

        assert result['status'] == 'FAILED'
        assert 'orphaned' not in caplog.text


# =============================================================================
# AthenaLoader.read_table type coercion
# =============================================================================
class TestAthenaReadTableCoercion:
    def test_coerces_timestamp_and_numeric_columns(self):
        loader = AthenaLoader.__new__(AthenaLoader)
        loader.database = 'test_db'
        loader._glue = MagicMock()
        loader._glue.get_table.return_value = {
            'Table': {
                'StorageDescriptor': {
                    'Columns': [
                        {'Name': 'date', 'Type': 'timestamp'},
                        {'Name': 'price', 'Type': 'double'},
                        {'Name': 'volume', 'Type': 'bigint'},
                        {'Name': 'ticker', 'Type': 'string'},
                    ]
                }
            }
        }

        csv_df = pd.DataFrame({
            'date': ['2020-01-01', '2020-01-02'],
            'price': ['100.5', '101.3'],
            'volume': ['1000', '2000'],
            'ticker': ['AAPL', 'AAPL'],
        })

        result = loader._coerce_types(csv_df, 'test_table')

        assert pd.api.types.is_datetime64_any_dtype(result['date'])
        assert pd.api.types.is_float_dtype(result['price'])
        assert result['volume'].dtype.name == 'Int64'
        assert pd.api.types.is_string_dtype(result['ticker'])


# =============================================================================
# SQLiteLoader round-trip
# =============================================================================
class TestSQLiteLoaderRoundTrip:
    def test_write_and_read_in_memory(self):
        """SQLiteLoader can write and read back a DataFrame via :memory:."""
        loader = SQLiteLoader(db_path=':memory:')
        loader.connect()
        try:
            df = pd.DataFrame({
                'DATE': pd.date_range('2020-01-01', periods=3),
                'VALUE': [1.1, 2.2, 3.3],
                'LABEL': ['a', 'b', 'c'],
            })
            result = loader._write_table(df, 'TEST_TABLE', replace=True, etl_key='test', verbose=False)
            assert result['status'] == 'SUCCESS'
            assert result['rows'] == 3

            read_back = loader.read_table('TEST_TABLE')
            assert len(read_back) == 3
            assert list(read_back.columns) == ['DATE', 'VALUE', 'LABEL']
        finally:
            loader.close()


# =============================================================================
# BaseLoader.load_etl with replace=True
# =============================================================================
class TestBaseLoaderLoadEtl:
    def test_load_etl_replace_calls_write_table(self):
        loader = AthenaLoader.__new__(AthenaLoader)
        loader.database = 'test_db'
        loader.workgroup = 'test_wg'
        loader.s3_bucket = 'test-bucket'
        loader.s3_prefix = 'test_prefix'
        loader._s3 = MagicMock()
        loader._glue = MagicMock()
        loader._athena = MagicMock()
        loader._connected = True
        loader._s3.list_objects_v2.return_value = {'Contents': []}

        df = pd.DataFrame({'close': [100.0, 101.0]})
        data_dict = {'vixdata': df}

        with patch.object(loader, '_write_table', return_value={
            'rows': 2, 'status': 'SUCCESS', 'duration': 0.1, 'error': None, 'etl_key': 'vixdata',
        }) as mock_write:
            results = loader.load_etl(data_dict, replace=True, verbose=False)

        mock_write.assert_called_once()
        args = mock_write.call_args
        assert args[0][1] == 'VIX_DATA'
        assert args[0][2] is True
        assert 'VIX_DATA' in results

    def test_public_write_table_uses_table_name_as_etl_key(self):
        """write_table() should pass the table name as etl_key, not a hardcoded string."""
        loader = AthenaLoader.__new__(AthenaLoader)
        loader.database = 'test_db'
        loader.workgroup = 'test_wg'
        loader.s3_bucket = 'test-bucket'
        loader.s3_prefix = 'test_prefix'
        loader._s3 = MagicMock()
        loader._glue = MagicMock()
        loader._athena = MagicMock()
        loader._connected = True

        df = pd.DataFrame({'val': [1, 2]})

        with patch.object(loader, '_write_table', return_value={
            'rows': 2, 'status': 'SUCCESS', 'duration': 0.1, 'error': None, 'etl_key': 'my_results',
        }) as mock_write:
            loader.write_table(df, 'MY_RESULTS', replace=True)

        args, kwargs = mock_write.call_args
        etl_key_passed = kwargs.get('etl_key') or args[3]
        assert etl_key_passed == 'my_results'
