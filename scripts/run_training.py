from __future__ import annotations

from pathlib import Path
import pickle

import pandas as pd

from src.models.train_trees import train_decision_tree_regressor


def main() -> None:
    data_path = "data/processed/model_dataset.csv"
    model_output_path = "models_artifacts/decision_tree_target_1d.pkl"

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

    train_df = df[required_cols].dropna().copy()

    print(f"Training dataset shape: {train_df.shape}")
    print(f"Date range: {train_df['date'].min()} -> {train_df['date'].max()}")
    print(f"Number of tickers: {train_df['ticker'].nunique()}")

    model = train_decision_tree_regressor(
        train_df=train_df,
        feature_cols=feature_cols,
        target_col=target_col,
        model_params={
            "max_depth": 5,
            "min_samples_split": 20,
            "min_samples_leaf": 10,
            "random_state": 42,
        },
    )

    Path(model_output_path).parent.mkdir(parents=True, exist_ok=True)
    with open(model_output_path, "wb") as f:
        pickle.dump(model, f)

    print("\nDecision tree model trained successfully.")
    print(f"Model saved to: {model_output_path}")
    print(f"Tree depth: {model.get_depth()}")
    print(f"Number of leaves: {model.get_n_leaves()}")


if __name__ == "__main__":
    main()