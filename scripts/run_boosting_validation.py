from __future__ import annotations

import os
from typing import Dict, List

import pandas as pd

from src.models.train_boosting import run_boosting_validation
from src.validation.splitters import chronological_train_validation_split


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

MODEL_PARAMS = {
    "n_estimators": 100,
    "learning_rate": 0.02,
    "max_depth": 2,
    "random_state": 42,
}

def _validate_columns(df: pd.DataFrame, cols: List[str]) -> None:
    missing = [c for c in cols if c not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns: {missing}")


def _print_section(title: str) -> None:
    print("\n" + "=" * 80)
    print(title)
    print("=" * 80)


def _periods_per_year_for_target(target_col: str) -> int:
    if target_col == "target_1d":
        return 252
    if target_col == "target_5d":
        return 52
    return 52


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

    periods_per_year = _periods_per_year_for_target(target_col)

    model, scored_val, metrics = run_boosting_validation(
        train_df=train_df,
        val_df=val_df,
        feature_cols=FEATURE_COLS,
        target_col=target_col,
        date_col=DATE_COL,
        transaction_cost_bps=TRANSACTION_COST_BPS,
        periods_per_year=periods_per_year,
        model_params=MODEL_PARAMS,
    )
    _ = model

    scored_path = os.path.join(
        OUTPUT_DIR,
        f"boosting_scored_{target_col}.csv",
    )
    scored_val.to_csv(scored_path, index=False)

    result_row: Dict[str, float | str] = {
        "target": target_col,
        "model": "gradient_boosting",
        "train_rows": float(len(train_df)),
        "validation_rows": float(len(val_df)),
        "train_start": str(train_df[DATE_COL].min().date()),
        "train_end": str(train_df[DATE_COL].max().date()),
        "validation_start": str(val_df[DATE_COL].min().date()),
        "validation_end": str(val_df[DATE_COL].max().date()),
        "n_estimators": float(MODEL_PARAMS["n_estimators"]),
        "learning_rate": float(MODEL_PARAMS["learning_rate"]),
        "max_depth": float(MODEL_PARAMS["max_depth"]),
        **metrics,
    }

    print(
        f"[GRADIENT BOOSTING | {target_col}] "
        f"rmse={metrics['rmse']:.6f}, mae={metrics['mae']:.6f}, r2={metrics['r2']:.6f}, "
        f"sharpe={metrics['portfolio_sharpe']:.4f}, "
        f"cumret={metrics['portfolio_cumulative_return']:.4f}, "
        f"maxdd={metrics['portfolio_max_drawdown']:.4f}"
    )

    return pd.DataFrame([result_row])


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

    summary_path = os.path.join(OUTPUT_DIR, "boosting_validation_metrics.csv")
    final_results.to_csv(summary_path, index=False)

    _print_section("Summary table")
    print(final_results)
    print(f"\nSaved summary metrics to: {summary_path}")
    print(f"Saved scored validation files to: {OUTPUT_DIR}")


if __name__ == "__main__":
    main()