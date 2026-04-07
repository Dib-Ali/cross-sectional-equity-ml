from __future__ import annotations

import json
import pickle
from pathlib import Path

import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

from src.models.train_trees import (
    predict_decision_tree_regressor,
    train_random_forest_regressor,
)


def main() -> None:
    data_path = "data/processed/model_dataset.csv"
    model_output_path = "models_artifacts/random_forest_target_1d.pkl"
    metrics_output_path = "results/random_forest_target_1d_metrics.json"

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

    print(f"Dataset shape used for training/evaluation: {model_df.shape}")
    print(f"Date range: {model_df['date'].min()} -> {model_df['date'].max()}")
    print(f"Number of tickers: {model_df['ticker'].nunique()}")

    model = train_random_forest_regressor(
        train_df=model_df,
        feature_cols=feature_cols,
        target_col=target_col,
        model_params={
            "n_estimators": 100,
            "max_depth": 8,
            "min_samples_split": 20,
            "min_samples_leaf": 10,
            "random_state": 42,
            "n_jobs": -1,
        },
    )

    preds = predict_decision_tree_regressor(model, model_df, feature_cols)
    eval_df = model_df.loc[preds.index].copy()
    eval_df["prediction"] = preds

    y_true = eval_df[target_col]
    y_pred = eval_df["prediction"]

    metrics = {
        "rows_evaluated": int(len(eval_df)),
        "mse": float(mean_squared_error(y_true, y_pred)),
        "rmse": float(mean_squared_error(y_true, y_pred) ** 0.5),
        "mae": float(mean_absolute_error(y_true, y_pred)),
        "r2": float(r2_score(y_true, y_pred)),
        "prediction_mean": float(y_pred.mean()),
        "prediction_std": float(y_pred.std()),
        "target_mean": float(y_true.mean()),
        "target_std": float(y_true.std()),
        "n_estimators": int(len(model.estimators_)),
        "max_depth": int(model.max_depth) if model.max_depth is not None else None,
    }

    Path(model_output_path).parent.mkdir(parents=True, exist_ok=True)
    with open(model_output_path, "wb") as f:
        pickle.dump(model, f)

    Path(metrics_output_path).parent.mkdir(parents=True, exist_ok=True)
    with open(metrics_output_path, "w") as f:
        json.dump(metrics, f, indent=4)

    print("\nRandom forest model trained and evaluated successfully.")
    print(f"Model saved to: {model_output_path}")
    print(f"Metrics saved to: {metrics_output_path}\n")

    print("Evaluation metrics:")
    for key, value in metrics.items():
        print(f"{key}: {value}")

    print("\nFirst 10 predictions vs actuals:")
    print(eval_df[["prediction", target_col]].head(10))


if __name__ == "__main__":
    main()