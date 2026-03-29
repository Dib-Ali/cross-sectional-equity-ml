from __future__ import annotations

import pandas as pd

from src.models.train_linear import train_linear_regression, predict_linear_regression
from src.validation.evaluation_ml import compute_regression_metrics
from src.validation.splitters import chronological_train_validation_split


def main() -> None:
    data_path = "data/processed/model_dataset.csv"
    df = pd.read_csv(data_path)

    df["date"] = pd.to_datetime(df["date"], errors="coerce")
    df = df.sort_values(["date", "ticker"]).reset_index(drop=True)

    feature_cols = [
        "return_1d",
        "return_5d",
        "return_20d",
        "return_60d",
        "volatility_20d",
        "volume_avg_20d",
    ]
    target_col = "target_1d"

    required_cols = ["date", "ticker"] + feature_cols + [target_col]
    missing = [col for col in required_cols if col not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns in dataset: {missing}")

    model_df = df[required_cols].dropna().copy()

    train_df, val_df = chronological_train_validation_split(
        model_df,
        date_col="date",
        train_ratio=0.8,
    )

    print(f"Train shape: {train_df.shape}")
    print(f"Validation shape: {val_df.shape}")
    print(f"Train date range: {train_df['date'].min()} -> {train_df['date'].max()}")
    print(f"Validation date range: {val_df['date'].min()} -> {val_df['date'].max()}")

    model = train_linear_regression(train_df, feature_cols, target_col)

    val_df = val_df.copy()
    val_df["prediction"] = predict_linear_regression(model, val_df, feature_cols)

    metrics = compute_regression_metrics(val_df[target_col], val_df["prediction"])

    print("\nValidation metrics:")
    for metric_name, metric_value in metrics.items():
        print(f"{metric_name}: {metric_value:.6f}")


if __name__ == "__main__":
    main()