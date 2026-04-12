from __future__ import annotations

import os
from typing import Dict, List

import pandas as pd

from src.models.train_regularized import run_regularized_validation
from src.validation.splitters import chronological_train_validation_split

import numpy as np
from sklearn.model_selection import TimeSeriesSplit
from sklearn.metrics import mean_squared_error
from src.models.train_regularized import _build_regularized_pipeline

DATA_PATH = "data/processed/model_dataset.csv"
DATE_COL = "date"
TICKER_COL = "ticker"
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
TARGET_COLS = ["target_1d", "target_5d"]
TRAIN_RATIO = 0.8
TRANSACTION_COST_BPS = 10.0

OUTPUT_DIR = "reports/tables"

MODEL_NAMES = ["ridge", "lasso", "elasticnet"]

RIDGE_ALPHA_GRID = [0.001, 0.01, 0.1, 1.0, 10.0]
LASSO_ALPHA_GRID = [0.00005, 0.0001, 0.0005, 0.001, 0.005]
ELASTICNET_ALPHA_GRID = [0.00005, 0.0001, 0.0005, 0.001]
ELASTICNET_L1_RATIO_GRID = [0.2, 0.5, 0.8]
LASSO_ALPHA_GRID_1D = [0.000005, 0.00001, 0.00005, 0.0001, 0.0005]
ELASTICNET_ALPHA_GRID_1D = [0.000005, 0.00001, 0.00005, 0.0001]

def _validate_columns(df: pd.DataFrame, cols: List[str]) -> None:
    missing = [c for c in cols if c not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns: {missing}")


def _print_section(title: str) -> None:
    print("\n" + "=" * 80)
    print(title)
    print("=" * 80)

def find_best_regularized_params(
    train_df: pd.DataFrame,
    feature_cols: List[str],
    target_col: str,
    model_name: str,
    alpha_grid: List[float],
    l1_ratio_grid: List[float] | None = None,
    n_splits: int = 5,
) -> Dict[str, float]:
    """
    Tune regularization hyperparameters using TimeSeriesSplit on the train set only.
    """
    tscv = TimeSeriesSplit(n_splits=n_splits)

    clean_df = train_df[feature_cols + [target_col]].dropna().copy()
    if clean_df.empty:
        raise ValueError("No valid rows available for tuning.")

    X = clean_df[feature_cols]
    y = clean_df[target_col]

    best_params: Dict[str, float] = {}
    best_score = float("inf")

    ratios = l1_ratio_grid if l1_ratio_grid is not None else [0.5]

    for alpha in alpha_grid:
        for l1_ratio in ratios:
            fold_scores = []

            for train_idx, val_idx in tscv.split(X):
                X_train_fold = X.iloc[train_idx]
                y_train_fold = y.iloc[train_idx]
                X_val_fold = X.iloc[val_idx]
                y_val_fold = y.iloc[val_idx]

                pipe = _build_regularized_pipeline(
                    model_name=model_name,
                    alpha=alpha,
                    l1_ratio=l1_ratio,
                    random_state=42,
                    max_iter=10000,
                )
                pipe.fit(X_train_fold, y_train_fold)
                preds = pipe.predict(X_val_fold)

                mse = mean_squared_error(y_val_fold, preds)
                fold_scores.append(mse)

            mean_mse = float(np.mean(fold_scores))

            if mean_mse < best_score:
                best_score = mean_mse
                best_params = {"alpha": alpha, "l1_ratio": l1_ratio}

    return best_params

def _periods_per_year_for_target(target_col: str) -> int:
    if target_col == "target_1d":
        return 252
    if target_col == "target_5d":
        return 52
    return 252

def _run_one_target(df: pd.DataFrame, target_col: str) -> pd.DataFrame:
    required_cols = [DATE_COL, TICKER_COL] + FEATURE_COLS + [target_col]
    _validate_columns(df, required_cols)

    target_df = df[required_cols].dropna().copy()

    train_df, val_df = chronological_train_validation_split(
        target_df,
        date_col=DATE_COL,
        train_ratio=TRAIN_RATIO,
    )

    _print_section(f"Target: {target_col}")
    print(f"Train rows: {len(train_df):,} | Validation rows: {len(val_df):,}")
    print(f"Train dates: {train_df[DATE_COL].min()} -> {train_df[DATE_COL].max()}")
    print(f"Validation dates: {val_df[DATE_COL].min()} -> {val_df[DATE_COL].max()}")

    target_results: List[Dict[str, float | str]] = []
    periods_per_year = _periods_per_year_for_target(target_col)

    for model_name in MODEL_NAMES:
        if model_name == "ridge":
            best_params = find_best_regularized_params(
                train_df=train_df,
                feature_cols=FEATURE_COLS,
                target_col=target_col,
                model_name="ridge",
                alpha_grid=RIDGE_ALPHA_GRID,
            )
        elif model_name == "lasso":
            alpha_grid = LASSO_ALPHA_GRID_1D if target_col == "target_1d" else LASSO_ALPHA_GRID
            best_params = find_best_regularized_params(
                train_df=train_df,
                feature_cols=FEATURE_COLS,
                target_col=target_col,
                model_name="lasso",
                alpha_grid=alpha_grid,
            )
        elif model_name == "elasticnet":
            alpha_grid = ELASTICNET_ALPHA_GRID_1D if target_col == "target_1d" else ELASTICNET_ALPHA_GRID
            best_params = find_best_regularized_params(
                train_df=train_df,
                feature_cols=FEATURE_COLS,
                target_col=target_col,
                model_name="elasticnet",
                alpha_grid=alpha_grid,
                l1_ratio_grid=ELASTICNET_L1_RATIO_GRID,
            )
        else:
            raise ValueError(f"Unsupported model: {model_name}")

        alpha = float(best_params["alpha"])
        l1_ratio = float(best_params["l1_ratio"])

        print(
            f"Tuned params for {model_name} on {target_col}: "
            f"alpha={alpha}, l1_ratio={l1_ratio}"
        )

        model, scored_val, metrics = run_regularized_validation(
            train_df=train_df,
            val_df=val_df,
            feature_cols=FEATURE_COLS,
            target_col=target_col,
            date_col=DATE_COL,
            model_name=model_name,
            alpha=alpha,
            l1_ratio=l1_ratio,
            transaction_cost_bps=TRANSACTION_COST_BPS,
            periods_per_year=periods_per_year,
            random_state=42,
            max_iter=10000,
        )
        _ = model
        scored_val = scored_val.drop_duplicates(subset=["date", "ticker"], keep="last")
        # ── IC stability: 2023 vs 2024 ─────────────────────────────────────────────
        from src.validation.evaluation_ml import compute_ic_series
        import pandas as pd

        ic_series = compute_ic_series(scored_val, 'date', target_col, 'prediction')
        
        ic_series.index = pd.to_datetime(ic_series.index)

        ic_2023 = ic_series[ic_series.index.year == 2023].mean()
        ic_2024 = ic_series[ic_series.index.year == 2024].mean()

        print(f"\n--- IC Stability Check ({model_name} | {target_col}) ---")
        print(f"  IC 2023: {ic_2023:.4f}")
        print(f"  IC 2024: {ic_2024:.4f}")
        print(f"  Stable: {ic_2023 > 0 and ic_2024 > 0}")

        scored_path = os.path.join(
            OUTPUT_DIR,
            f"regularized_scored_{target_col}_{model_name}.csv",
        )
        scored_val.to_csv(scored_path, index=False)

        result_row: Dict[str, float | str] = {
            "target": target_col,
            "model": model_name,
            "alpha": alpha,
            "l1_ratio": l1_ratio,
            "train_rows": float(len(train_df)),
            "validation_rows": float(len(val_df)),
            "train_start": str(train_df[DATE_COL].min().date()),
            "train_end": str(train_df[DATE_COL].max().date()),
            "validation_start": str(val_df[DATE_COL].min().date()),
            "validation_end": str(val_df[DATE_COL].max().date()),
            **metrics,
        }
        target_results.append(result_row)

        print(
            f"[{model_name.upper()} | {target_col}] "
            f"rmse={metrics['rmse']:.6f}, mae={metrics['mae']:.6f}, r2={metrics['r2']:.6f}, "
            f"sharpe={metrics['portfolio_sharpe']:.4f}, "
            f"cumret={metrics['portfolio_cumulative_return']:.4f}, "
            f"maxdd={metrics['portfolio_max_drawdown']:.4f}"
        )

    return pd.DataFrame(target_results)


def main() -> None:
    _print_section("Loading processed dataset")
    df = pd.read_csv(DATA_PATH)
    df[DATE_COL] = pd.to_datetime(df[DATE_COL], errors="coerce")
    df = df.dropna(subset=[DATE_COL]).sort_values([DATE_COL, TICKER_COL]).reset_index(drop=True)

    os.makedirs(OUTPUT_DIR, exist_ok=True)

    all_results = []
    for target_col in TARGET_COLS:
        results_df = _run_one_target(df, target_col)
        all_results.append(results_df)

    final_results = pd.concat(all_results, ignore_index=True)
    final_results = final_results.sort_values(["target", "model"]).reset_index(drop=True)

    summary_path = os.path.join(OUTPUT_DIR, "regularized_validation_metrics.csv")
    final_results.to_csv(summary_path, index=False)

    _print_section("Summary table")
    print(final_results)
    print(f"\nSaved summary metrics to: {summary_path}")
    print(f"Saved scored validation files to: {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
