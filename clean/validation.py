"""Data validation utilities for the clean package."""

import pandas as pd


def validate_dataframe(df, name="", min_rows=1, expected_date_index=True):
    """Validate a DataFrame meets basic quality checks.

    Parameters
    ----------
    df : pd.DataFrame
        DataFrame to validate.
    name : str
        Label for error messages.
    min_rows : int
        Minimum expected row count.
    expected_date_index : bool
        Whether the index should be DatetimeIndex.

    Returns
    -------
    list[str] : List of validation issues (empty = all good).
    """
    issues = []

    if df is None:
        issues.append(f"{name}: DataFrame is None")
        return issues

    if not isinstance(df, pd.DataFrame):
        issues.append(f"{name}: not a DataFrame (got {type(df).__name__})")
        return issues

    if len(df) < min_rows:
        issues.append(f"{name}: only {len(df)} rows (expected >= {min_rows})")

    # Check for all-null columns
    all_null_cols = df.columns[df.isnull().all()].tolist()
    if all_null_cols:
        issues.append(f"{name}: all-null columns: {all_null_cols}")

    # Check date index
    if expected_date_index and not isinstance(df.index, pd.DatetimeIndex):
        issues.append(f"{name}: index is {type(df.index).__name__}, expected DatetimeIndex")

    return issues
