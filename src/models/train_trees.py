from __future__ import annotations

from typing import Dict, List, Optional

import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.tree import DecisionTreeRegressor


def train_decision_tree_regressor(
    train_df: pd.DataFrame,
    feature_cols: List[str],
    target_col: str,
    model_params: Optional[Dict] = None,
) -> DecisionTreeRegressor:
    """
    Train a decision tree regressor on the provided dataframe.
    """
    if train_df.empty:
        raise ValueError("Training dataframe is empty.")

    missing_features = [col for col in feature_cols if col not in train_df.columns]
    if missing_features:
        raise ValueError(f"Missing feature columns: {missing_features}")

    if target_col not in train_df.columns:
        raise ValueError(f"Missing target column: {target_col}")

    model_df = train_df[feature_cols + [target_col]].dropna().copy()
    if model_df.empty:
        raise ValueError("No valid rows remain after dropping NaNs.")

    X_train = model_df[feature_cols]
    y_train = model_df[target_col]

    default_params = {
        "max_depth": 5,
        "min_samples_split": 20,
        "min_samples_leaf": 10,
        "random_state": 42,
    }

    if model_params is not None:
        default_params.update(model_params)

    model = DecisionTreeRegressor(**default_params)
    model.fit(X_train, y_train)
    return model


def train_random_forest_regressor(
    train_df: pd.DataFrame,
    feature_cols: List[str],
    target_col: str,
    model_params: Optional[Dict] = None,
) -> RandomForestRegressor:
    """
    Train a random forest regressor on the provided dataframe.
    """
    if train_df.empty:
        raise ValueError("Training dataframe is empty.")

    missing_features = [col for col in feature_cols if col not in train_df.columns]
    if missing_features:
        raise ValueError(f"Missing feature columns: {missing_features}")

    if target_col not in train_df.columns:
        raise ValueError(f"Missing target column: {target_col}")

    model_df = train_df[feature_cols + [target_col]].dropna().copy()
    if model_df.empty:
        raise ValueError("No valid rows remain after dropping NaNs.")

    X_train = model_df[feature_cols]
    y_train = model_df[target_col]

    default_params = {
        "n_estimators": 100,
        "max_depth": 8,
        "min_samples_split": 20,
        "min_samples_leaf": 10,
        "random_state": 42,
        "n_jobs": -1,
    }

    if model_params is not None:
        default_params.update(model_params)

    model = RandomForestRegressor(**default_params)
    model.fit(X_train, y_train)
    return model


def predict_decision_tree_regressor(
    model,
    df: pd.DataFrame,
    feature_cols: List[str],
) -> pd.Series:
    """
    Generate predictions from a trained tree-based regressor.
    """
    if df.empty:
        raise ValueError("Prediction dataframe is empty.")

    missing_features = [col for col in feature_cols if col not in df.columns]
    if missing_features:
        raise ValueError(f"Missing feature columns: {missing_features}")

    prediction_df = df[feature_cols].dropna().copy()
    if prediction_df.empty:
        raise ValueError("No valid rows remain for prediction after dropping NaNs.")

    preds = model.predict(prediction_df)

    return pd.Series(preds, index=prediction_df.index, name="prediction")