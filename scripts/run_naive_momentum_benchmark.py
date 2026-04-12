from __future__ import annotations

import os
from typing import Dict, List

import numpy as np
import pandas as pd

from src.validation.splitters import chronological_train_val_test_split


DATA_PATH = "data/processed/model_dataset.csv"
DATE_COL = "date"
TICKER_COL = "ticker"
SIGNAL_COL = "return_5d"
TARGET_COL = "target_5d"
OUTPUT_DIR = "reports/tables"

TOP_N = 10
BOTTOM_N = 10
TRANSACTION_COST_BPS = 10.0
PERIODS_PER_YEAR = 52


def _max_drawdown(returns: pd.Series) -> float:
    equity_curve = (1 + returns).cumprod()
    running_max = equity_curve.cummax()
    drawdown = equity_curve / running_max - 1.0
    return float(drawdown.min())


def compute_long_short_financial_metrics(
    df: pd.DataFrame,
    date_col: str,
    prediction_col: str,
    realized_return_col: str,
    top_n: int = 10,
    bottom_n: int = 10,
    transaction_cost_bps: float = 10.0,
    periods_per_year: int = 52,
) -> Dict[str, float]:
    required = [date_col, TICKER_COL, prediction_col, realized_return_col]
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns for financial metrics: {missing}")

    work = df[required].dropna().copy()
    work[date_col] = pd.to_datetime(work[date_col], errors="coerce")
    work = work.dropna(subset=[date_col]).sort_values([date_col, TICKER_COL]).reset_index(drop=True)

    if work.empty:
        raise ValueError("No valid rows available to compute financial metrics.")

    # Non-overlapping 5-day evaluation
    unique_dates = sorted(work[date_col].dropna().unique())
    selected_dates = unique_dates[::5]
    work = work[work[date_col].isin(selected_dates)].copy()

    cost_rate = transaction_cost_bps / 10000.0

    period_returns: List[float] = []
    period_turnover: List[float] = []
    prev_weights: Dict[str, float] | None = None

    for _, group in work.groupby(date_col, sort=True):
        ranked = group.sort_values(prediction_col, ascending=False)
        if len(ranked) < top_n + bottom_n:
            continue

        long_leg = ranked.head(top_n)
        short_leg = ranked.tail(bottom_n)

        weights: Dict[str, float] = {}
        for ticker in long_leg[TICKER_COL]:
            weights[str(ticker)] = 1.0 / top_n
        for ticker in short_leg[TICKER_COL]:
            weights[str(ticker)] = weights.get(str(ticker), 0.0) - 1.0 / bottom_n

        gross_return = 0.0
        for _, row in ranked.iterrows():
            ticker = str(row[TICKER_COL])
            w = weights.get(ticker, 0.0)
            gross_return += w * float(row[realized_return_col])

        if prev_weights is None:
            turnover = float(sum(abs(w) for w in weights.values()))
        else:
            all_tickers = set(prev_weights).union(weights)
            turnover = float(
                sum(abs(weights.get(t, 0.0) - prev_weights.get(t, 0.0)) for t in all_tickers)
            )

        trading_cost = cost_rate * turnover
        net_return = gross_return - trading_cost

        period_returns.append(net_return)
        period_turnover.append(turnover)
        prev_weights = weights

    if not period_returns:
        raise ValueError("Unable to compute portfolio returns.")

    r = pd.Series(period_returns, dtype=float)

    mean_return = float(r.mean())
    volatility = float(r.std(ddof=1)) if len(r) > 1 else 0.0
    sharpe = 0.0 if volatility == 0.0 else float((mean_return / volatility) * np.sqrt(periods_per_year))

    metrics = {
        "portfolio_mean_return": mean_return,
        "portfolio_volatility": volatility,
        "portfolio_cumulative_return": float((1 + r).prod() - 1.0),
        "portfolio_sharpe": sharpe,
        "portfolio_max_drawdown": _max_drawdown(r),
        "portfolio_turnover_avg": float(np.mean(period_turnover)),
        "portfolio_num_periods": float(len(r)),
    }
    return metrics


def build_naive_scored_test(
    test_df: pd.DataFrame,
    signal_col: str,
) -> pd.DataFrame:
    required = [DATE_COL, TICKER_COL, signal_col, TARGET_COL]
    missing = [c for c in required if c not in test_df.columns]
    if missing:
        raise ValueError(f"Missing required columns in test_df: {missing}")

    scored = test_df[[DATE_COL, TICKER_COL, signal_col, TARGET_COL]].dropna().copy()
    scored = scored.sort_values([DATE_COL, TICKER_COL]).reset_index(drop=True)

    # Use return_5d directly as naive momentum score
    scored["prediction"] = scored[signal_col]
    scored = scored.drop_duplicates(subset=[DATE_COL, TICKER_COL], keep="last")
    return scored


def main() -> None:
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    df = pd.read_csv(DATA_PATH)
    df[DATE_COL] = pd.to_datetime(df[DATE_COL], errors="coerce")
    df = df.dropna(subset=[DATE_COL]).sort_values([DATE_COL, TICKER_COL]).reset_index(drop=True)

    required_cols = [DATE_COL, TICKER_COL, SIGNAL_COL, TARGET_COL]
    df = df[required_cols].dropna().copy()

    _, _, test_df = chronological_train_val_test_split(
        df,
        date_col=DATE_COL,
        train_end="2022-12-31",
        val_end="2023-12-31",
    )

    print("Test set shape:", test_df.shape)
    print("Test dates:", test_df[DATE_COL].min(), "->", test_df[DATE_COL].max())

    scored_test = build_naive_scored_test(
        test_df=test_df,
        signal_col=SIGNAL_COL,
    )

    metrics = compute_long_short_financial_metrics(
        df=scored_test,
        date_col=DATE_COL,
        prediction_col="prediction",
        realized_return_col=TARGET_COL,
        top_n=TOP_N,
        bottom_n=BOTTOM_N,
        transaction_cost_bps=TRANSACTION_COST_BPS,
        periods_per_year=PERIODS_PER_YEAR,
    )

    summary_df = pd.DataFrame(
        [
            {
                "model": "naive_5d_momentum",
                "signal": SIGNAL_COL,
                **metrics,
            }
        ]
    )

    scored_path = os.path.join(OUTPUT_DIR, "naive_momentum_scored_test_target_5d.csv")
    summary_path = os.path.join(OUTPUT_DIR, "naive_momentum_benchmark_metrics.csv")

    scored_test.to_csv(scored_path, index=False)
    summary_df.to_csv(summary_path, index=False)

    print("\nNaive momentum benchmark metrics:")
    print(summary_df.to_string(index=False))
    print(f"\nSaved scored test data to: {scored_path}")
    print(f"Saved summary metrics to: {summary_path}")


if __name__ == "__main__":
    main()