from __future__ import annotations

from typing import List, Tuple

import pandas as pd
from sklearn.linear_model import LinearRegression


def train_linear_regression(
    train_df: pd.DataFrame,
    feature_cols: List[str],
    target_col: str,
) -> LinearRegression:
    """
    Train a simple linear regression model.
    """
    missing_features = [col for col in feature_cols if col not in train_df.columns]
    if missing_features:
        raise ValueError(f"Missing feature columns: {missing_features}")

    if target_col not in train_df.columns:
        raise ValueError(f"Missing target column: {target_col}")

    X_train = train_df[feature_cols]
    y_train = train_df[target_col]

    model = LinearRegression()
    model.fit(X_train, y_train)
    return model


def predict_linear_regression(
    model: LinearRegression,
    df: pd.DataFrame,
    feature_cols: List[str],
) -> pd.Series:
    """
    Generate predictions from a trained linear regression model.
    """
    missing_features = [col for col in feature_cols if col not in df.columns]
    if missing_features:
        raise ValueError(f"Missing feature columns: {missing_features}")

    preds = model.predict(df[feature_cols])
    return pd.Series(preds, index=df.index, name="prediction")