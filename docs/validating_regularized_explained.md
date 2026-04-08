# Validating Regularized Regression: Beginner-Friendly Guide

This guide explains:
1. What regularized regression means
2. Why we use it in stock-return prediction
3. What the new script does end-to-end
4. How to read and present the output

The script is in `scripts/validating_regularized.py`.

## 1) What problem are we solving?

We want to predict future stock returns from historical features.
In this project, targets are:
- `target_1d` (forward 1-day return)
- `target_5d` (forward 5-day return)

Then we rank stocks by predicted return and evaluate whether this can produce useful portfolio performance.

## 2) What is regularized regression?

### Basic linear regression
Linear regression fits:

y_hat = b0 + b1*x1 + b2*x2 + ... + bk*xk

It learns coefficients b1, b2, ... to reduce prediction error.

### Why regularization?
Financial data is noisy. A model can overfit, meaning it learns noise instead of real signal.
Regularization adds a penalty on coefficient size so the model stays more stable.

### Ridge (L2)
Adds a penalty on squared coefficients.
- Tends to keep all features
- Shrinks coefficients smoothly
- Good default when many features are somewhat useful

### Lasso (L1)
Adds a penalty on absolute coefficient values.
- Can force some coefficients to exactly zero
- Performs feature selection automatically
- Useful when many features may be weak/useless

### ElasticNet (L1 + L2)
Combines both penalties.
- More flexible than only Ridge or only Lasso
- Controlled by `l1_ratio`

## 3) What key hyperparameters mean

- `alpha`: strength of regularization
  - larger alpha -> stronger shrinkage
  - smaller alpha -> weaker regularization

- `l1_ratio`: only for ElasticNet
  - 0.0 means mostly Ridge-like
  - 1.0 means mostly Lasso-like

## 4) Why we do NOT shuffle here

You are correct: this is time-dependent stock data.
We should not randomly shuffle rows when training/validating.

The script uses chronological split:
- Train on earlier dates
- Validate on later dates

This is closer to real trading conditions and avoids look-ahead bias from future observations leaking into training.

## 5) What `scripts/validating_regularized.py` does

### Step A: Load dataset
- Reads `data/processed/model_dataset.csv`
- Parses and sorts by date/ticker

### Step B: Run experiments for both targets
- `target_1d`
- `target_5d`

For each target:
1. Keep required columns (`date`, `ticker`, features, target)
2. Drop missing rows
3. Chronologically split into train/validation (80/20)

### Step C: Train and evaluate 3 models
- Ridge
- Lasso
- ElasticNet

For each model, it calls the regularized module and gets:
- predictions on validation data
- machine-learning metrics
- financial metrics from a long-short strategy

### Step D: Save outputs
- Per-model scored validation files:
  - `reports/tables/regularized_scored_target_1d_ridge.csv`
  - and similar for each target/model

- Summary metrics table:
  - `reports/tables/regularized_validation_metrics.csv`

## 6) Metrics you will see

### Machine-learning metrics
- RMSE: lower is better
- MAE: lower is better
- R2: higher is better (can be near zero or negative in finance)

### Financial metrics (from prediction-ranked long-short portfolio)
- `portfolio_mean_return`
- `portfolio_volatility`
- `portfolio_cumulative_return`
- `portfolio_sharpe`
- `portfolio_max_drawdown`
- `portfolio_turnover_avg`

Important: ML metrics and finance metrics can disagree.
A model with slightly worse RMSE can still produce better Sharpe if ranking quality is better.

## 7) How transaction cost is handled

The script uses 10 bps by default (`TRANSACTION_COST_BPS = 10.0`).
Cost is deducted using turnover each rebalancing period.

If you want to test sensitivity, try:
- 5 bps
- 10 bps
- 20 bps

Then compare how Sharpe and cumulative return change.

## 8) How to run it

From project root:

```bash
python -m scripts.validating_regularized
```

## 9) How to present this in your report

Suggested structure:
1. Explain why regularization is needed in noisy financial data
2. Show chronological split design (no random shuffling)
3. Compare Ridge/Lasso/ElasticNet on both horizons
4. Report both ML and financial metrics
5. Discuss trade-offs and stability under transaction costs

## 10) Practical interpretation tips

- If Ridge and ElasticNet perform similarly, Ridge is usually simpler to defend.
- If Lasso zeros many coefficients and performance holds, you can argue simpler signal structure.
- If 1-day is weak but 5-day is stronger, discuss horizon mismatch and market microstructure noise.
- If Sharpe collapses when cost increases, strategy may be too turnover-heavy.
