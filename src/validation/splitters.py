from __future__ import annotations

from typing import Tuple

import pandas as pd


def chronological_train_validation_split(
    df: pd.DataFrame,
    date_col: str = "date",
    train_ratio: float = 0.8,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Split a panel dataset chronologically by unique dates.

    All rows from earlier dates go to train, later dates go to validation.
    This avoids leakage from future information.
    """
    if df.empty:
        raise ValueError("Input dataframe is empty.")

    if date_col not in df.columns:
        raise ValueError(f"Missing required date column: {date_col}")

    if not 0 < train_ratio < 1:
        raise ValueError("train_ratio must be between 0 and 1.")

    df = df.copy()
    df[date_col] = pd.to_datetime(df[date_col], errors="coerce")
    df = df.sort_values([date_col, "ticker"]).reset_index(drop=True)

    unique_dates = sorted(df[date_col].dropna().unique())
    if len(unique_dates) < 2:
        raise ValueError("Not enough unique dates for splitting.")

    split_idx = int(len(unique_dates) * train_ratio)
    split_idx = max(1, min(split_idx, len(unique_dates) - 1))

    train_dates = unique_dates[:split_idx]
    val_dates = unique_dates[split_idx:]

    train_df = df[df[date_col].isin(train_dates)].copy()
    val_df = df[df[date_col].isin(val_dates)].copy()

    if train_df.empty or val_df.empty:
        raise ValueError("Train or validation split is empty.")

    return train_df, val_df


def chronological_train_val_test_split(
    df: pd.DataFrame,
    date_col: str = "date",
    train_end: str = "2022-12-31",
    val_end: str = "2023-12-31",
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Split panel data into chronological train / validation / test sets.

    Default split:
    - Train: dates <= 2022-12-31
    - Validation: 2023-01-01 to 2023-12-31
    - Test: dates >= 2024-01-01
    """
    if df.empty:
        raise ValueError("Input dataframe is empty.")

    if date_col not in df.columns:
        raise ValueError(f"Missing required date column: {date_col}")

    df = df.copy()
    df[date_col] = pd.to_datetime(df[date_col], errors="coerce")
    df = df.dropna(subset=[date_col]).sort_values([date_col, "ticker"]).reset_index(drop=True)

    train_end_dt = pd.Timestamp(train_end)
    val_end_dt = pd.Timestamp(val_end)

    if train_end_dt >= val_end_dt:
        raise ValueError("train_end must be earlier than val_end.")

    train_df = df[df[date_col] <= train_end_dt].copy()
    val_df = df[(df[date_col] > train_end_dt) & (df[date_col] <= val_end_dt)].copy()
    test_df = df[df[date_col] > val_end_dt].copy()

    if train_df.empty or val_df.empty or test_df.empty:
        raise ValueError("One of train/validation/test splits is empty.")

    return train_df, val_df, test_df