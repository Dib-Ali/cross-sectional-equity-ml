from __future__ import annotations

import os
import pandas as pd

from src.models.train_ensemble_final import (
    build_ensemble_prediction,
    evaluate_predictions,
    score_base_models,
    
    train_base_models,
)
from src.validation.splitters import chronological_train_val_test_split


DATA_PATH = "data/processed/model_dataset.csv"
DATE_COL = "date"
TICKER_COL = "ticker"
TARGET_COL = "target_5d"
FEATURE_COLS = [
    "return_1d",
    "return_5d",
    "return_20d",
    "return_60d",
    "volatility_20d",
    "volume_avg_20d",
    "momentum_ratio_1",
    "momentum_ratio_2",
    "risk_adjusted_return",
    "interaction_1",
    "interaction_2",
    "rank_return_20d",
    "volatility_regime",
    "trend_consistency",
]
OUTPUT_DIR = "reports/tables"


def main() -> None:
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    df = pd.read_csv(DATA_PATH)
    df[DATE_COL] = pd.to_datetime(df[DATE_COL], errors="coerce")
    df = df.dropna(subset=[DATE_COL]).sort_values([DATE_COL, TICKER_COL]).reset_index(drop=True)

    required_cols = [DATE_COL, TICKER_COL] + FEATURE_COLS + [TARGET_COL]
    df = df[required_cols].dropna().copy()

    train_df, val_df, test_df = chronological_train_val_test_split(
        df,
        date_col=DATE_COL,
        train_end="2022-12-31",
        val_end="2023-12-31",
    )

    print("\nTrain:", train_df.shape, train_df[DATE_COL].min(), train_df[DATE_COL].max())
    print("Val:  ", val_df.shape, val_df[DATE_COL].min(), val_df[DATE_COL].max())
    print("Test: ", test_df.shape, test_df[DATE_COL].min(), test_df[DATE_COL].max())

    # Step 1: train on train only
    base_models_train = train_base_models(
        train_df=train_df,
        feature_cols=FEATURE_COLS,
        target_col=TARGET_COL,
    )

    # Step 2: validation scoring and weight search
    scored_val = score_base_models(
        models=base_models_train,
        df=val_df,
        feature_cols=FEATURE_COLS,
    )
    scored_val = score_base_models(
    models=base_models_train,
    df=val_df,
    feature_cols=FEATURE_COLS,
)

    scored_val["prediction"] = build_ensemble_prediction(scored_val)

    val_metrics = evaluate_predictions(
        df=scored_val,
        date_col=DATE_COL,
        target_col=TARGET_COL,
        prediction_col="prediction",
        top_n=10,
        bottom_n=10,
        transaction_cost_bps=10.0,
        periods_per_year=52,
    )

    # Step 3: retrain on train + validation
    train_val_df = pd.concat([train_df, val_df], ignore_index=True)

    base_models_train_val = train_base_models(
        train_df=train_val_df,
        feature_cols=FEATURE_COLS,
        target_col=TARGET_COL,
    )

    scored_test = score_base_models(
        models=base_models_train_val,
        df=test_df,
        feature_cols=FEATURE_COLS,
    )
    scored_test["prediction"] = build_ensemble_prediction(scored_test)

    test_metrics = evaluate_predictions(
        df=scored_test,
        date_col=DATE_COL,
        target_col=TARGET_COL,
        prediction_col="prediction",
        top_n=10,
        bottom_n=10,
        transaction_cost_bps=10.0,
        periods_per_year=52,
    )


    summary_df = pd.DataFrame([
    {
        "stage": "validation",
        "ridge_weight": 0.4,
        "elasticnet_weight": 0.6,
        **val_metrics
    },
    {
        "stage": "test",
        "ridge_weight": 0.4,
        "elasticnet_weight": 0.6,
        **test_metrics
    }
])

    ensemble_search_path = os.path.join(OUTPUT_DIR, "ensemble_weight_search_validation.csv")
    scored_val_path = os.path.join(OUTPUT_DIR, "ensemble_scored_validation_target_5d.csv")
    scored_test_path = os.path.join(OUTPUT_DIR, "ensemble_scored_test_target_5d.csv")
    summary_path = os.path.join(OUTPUT_DIR, "ensemble_final_metrics.csv")


    scored_val.to_csv(scored_val_path, index=False)
    scored_test.to_csv(scored_test_path, index=False)
    summary_df.to_csv(summary_path, index=False)

    print("\nValidation metrics:")
    print(pd.DataFrame([val_metrics]).to_string(index=False))
    print("\nTest metrics:")
    print(pd.DataFrame([test_metrics]).to_string(index=False))
    print(f"\nSaved weight search to: {ensemble_search_path}")
    print(f"Saved validation scored ensemble to: {scored_val_path}")
    print(f"Saved test scored ensemble to: {scored_test_path}")
    print(f"Saved summary metrics to: {summary_path}")


if __name__ == "__main__":
    main()