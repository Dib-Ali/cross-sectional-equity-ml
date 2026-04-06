# Regularized Regression Module Guide

This document explains the implementation in `src/models/train_regularized.py` for:
- Ridge regression (L2)
- Lasso regression (L1)
- ElasticNet regression (L1 + L2)

It also explains how we evaluate both:
- machine-learning metrics (RMSE, MAE, R2)
- finance metrics from a long-short portfolio simulation

## Why Regularization Here

Financial return prediction is noisy and overfitting is common.
Regularization helps by shrinking model coefficients:

- Ridge (L2): shrinks all coefficients smoothly
- Lasso (L1): can force some coefficients to zero (sparse model)
- ElasticNet: mixes both behaviors

## Temporal Safety and No Shuffle

Your concern is correct: panel stock data should not be randomly shuffled.
The intended workflow is:

1. Create chronological train/validation splits by date.
2. Train on past dates only.
3. Evaluate on future dates.

The module assumes your train and validation data are already split chronologically.

Inside the model pipeline, feature scaling is fit only on the training subset used in `.fit(...)`.
This avoids leakage of validation feature statistics into training.

## Implemented API

Main functions in `src/models/train_regularized.py`:

- `train_regularized_regression(...)`
  - Generic trainer for `ridge`, `lasso`, or `elasticnet`
- `train_ridge_regression(...)`
- `train_lasso_regression(...)`
- `train_elasticnet_regression(...)`
- `predict_regularized_regression(...)`

Evaluation functions:

- `evaluate_regularized_predictions(...)`
  - Returns one dictionary containing both ML and financial metrics
- `compute_long_short_financial_metrics(...)`
  - Builds a long-short portfolio each date:
    - long top N predictions
    - short bottom N predictions
    - equal-weight each side
    - includes turnover-based transaction costs
- `run_regularized_validation(...)`
  - End-to-end helper:
    - train model
    - predict validation set
    - compute combined metrics
    - return model + scored validation dataframe + metrics

## Financial Evaluation Design

The financial metrics use predicted cross-sectional ranking and realized returns:

- Portfolio construction per date:
  - Long: top `top_n`
  - Short: bottom `bottom_n`
  - Equal weights on each side

- Costs:
  - Turnover is computed from weight changes between periods
  - Cost is deducted as:
    - `transaction_cost_bps / 10000 * turnover`

- Returned finance metrics:
  - `portfolio_mean_return`
  - `portfolio_volatility`
  - `portfolio_cumulative_return`
  - `portfolio_sharpe`
  - `portfolio_max_drawdown`
  - `portfolio_turnover_avg`
  - `portfolio_num_periods`

## Example Usage

```python
import pandas as pd

from src.validation.splitters import chronological_train_validation_split
from src.models.train_regularized import run_regularized_validation

feature_cols = [
    "return_1d",
    "return_5d",
    "return_20d",
    "return_60d",
    "volatility_20d",
    "volume_avg_20d",
]

target_col = "target_1d"

df = pd.read_csv("data/processed/model_dataset.csv")
df["date"] = pd.to_datetime(df["date"], errors="coerce")

df = df.dropna(subset=["date", "ticker", *feature_cols, target_col]).copy()
train_df, val_df = chronological_train_validation_split(df, date_col="date", train_ratio=0.8)

model, scored_val, metrics = run_regularized_validation(
    train_df=train_df,
    val_df=val_df,
    feature_cols=feature_cols,
    target_col=target_col,
    date_col="date",
    model_name="elasticnet",   # "ridge" | "lasso" | "elasticnet"
    alpha=0.5,
    l1_ratio=0.5,
    transaction_cost_bps=10.0,
)

print(metrics)
```

## Notes for Your 4-Day Deadline

- Start with Ridge as stable baseline.
- Compare Ridge vs Lasso vs ElasticNet on both `target_1d` and `target_5d`.
- Keep one consistent chronological split protocol for fair comparison.
- Report both ML and financial metrics side by side.

## Suggested Next Integration

To wire this into your existing script flow, update `scripts/run_validation.py` to:

1. Loop through model names (`ridge`, `lasso`, `elasticnet`)
2. Evaluate for both targets (`target_1d`, `target_5d`)
3. Save metrics table to `reports/tables/`
