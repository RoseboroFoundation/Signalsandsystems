"""Caching utilities for serializing/deserializing data to parquet."""

import os
import json

import pandas as pd

from .config import logger


def _make_json_serializable(obj):
    """Convert Timestamps/numpy types for JSON serialization."""
    if isinstance(obj, dict):
        return {k: _make_json_serializable(v) for k, v in obj.items()}
    if isinstance(obj, pd.Timestamp):
        return obj.isoformat()
    if hasattr(obj, 'item'):  # numpy scalar
        return obj.item()
    return obj


def _restore_summary_stats(data):
    """Restore Timestamps from ISO strings in loaded summary_stats."""
    if not isinstance(data, dict):
        return data
    for key, value in data.items():
        if isinstance(value, dict):
            for k, v in value.items():
                if k in ('start_date', 'end_date') and isinstance(v, str):
                    try:
                        value[k] = pd.Timestamp(v)
                    except Exception:
                        pass
    return data


def _save_cache(result, cache_dir):
    """Serialize DataFrame or dict-of-DataFrames to parquet-based cache."""
    os.makedirs(cache_dir, exist_ok=True)
    manifest = {'type': None, 'keys': []}

    if isinstance(result, pd.DataFrame):
        result.to_parquet(os.path.join(cache_dir, '_data.parquet'))
        manifest['type'] = 'dataframe'
    elif isinstance(result, dict):
        manifest['type'] = 'dict'
        for key, value in result.items():
            if isinstance(value, pd.DataFrame):
                value.to_parquet(os.path.join(cache_dir, f'{key}.parquet'))
                manifest['keys'].append({'name': key, 'kind': 'dataframe'})
            elif key == 'summary_stats' and isinstance(value, dict):
                with open(os.path.join(cache_dir, '_summary_stats.json'), 'w') as f:
                    json.dump(_make_json_serializable(value), f)
                manifest['keys'].append({'name': key, 'kind': 'json'})
            elif isinstance(value, dict):
                _save_cache(value, os.path.join(cache_dir, key))
                manifest['keys'].append({'name': key, 'kind': 'subdir'})
            elif value is None:
                manifest['keys'].append({'name': key, 'kind': 'none'})

    with open(os.path.join(cache_dir, '_manifest.json'), 'w') as f:
        json.dump(manifest, f)


def _load_cache(cache_dir):
    """Load cached data from parquet-based cache directory."""
    manifest_path = os.path.join(cache_dir, '_manifest.json')
    if not os.path.exists(manifest_path):
        return None

    with open(manifest_path, 'r') as f:
        manifest = json.load(f)

    if manifest['type'] == 'dataframe':
        return pd.read_parquet(os.path.join(cache_dir, '_data.parquet'))
    elif manifest['type'] == 'dict':
        result = {}
        for entry in manifest['keys']:
            name = entry['name']
            kind = entry['kind']
            if kind == 'dataframe':
                result[name] = pd.read_parquet(
                    os.path.join(cache_dir, f'{name}.parquet')
                )
            elif kind == 'json':
                with open(os.path.join(cache_dir, '_summary_stats.json'), 'r') as f:
                    result[name] = _restore_summary_stats(json.load(f))
            elif kind == 'subdir':
                result[name] = _load_cache(os.path.join(cache_dir, name))
            elif kind == 'none':
                result[name] = None
        return result
    return None
