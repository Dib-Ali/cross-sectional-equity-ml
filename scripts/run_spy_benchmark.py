from __future__ import annotations

import os
from typing import Dict, List

import numpy as np
import pandas as pd

try:
    import yfinance as yf
except ImportError as exc:
    raise ImportError(
        "yfinance is required for the SPY benchmark. Install it with: pip install yfinance"
    ) from exc


DATE_COL = "date"
TICKER_COL = "ticker"
TARGET_COL = "target_5d"
PREDICTION_COL = "prediction"

INPUT_PATH = "reports/tables/ensemble_scored_test_target_5d.csv"
OUTPUT_DIR = "reports/tables"

TOP_N = 10
BOTTOM_N = 10
TRANSACTION_COST_BPS = 10.0
PERIODS_PER_YEAR = 52  # non-overlapping 5d evaluation


def _max_drawdown(returns: pd.Series) -> float:
    equity_curve = (1 + returns).cumprod()
    running_max = equity_curve.cummax()
    drawdown = equity_curve / running_max - 1.0
    return float(drawdown.min())


def compute_strategy_period_returns(
    df: pd.DataFrame,
    date_col: str,
    ticker_col: str,
    prediction_col: str,
    realized_return_col: str,
    top_n: int = 10,
    bottom_n: int = 10,
    transaction_cost_bps: float = 10.0,
) -> pd.DataFrame:
    required = [date_col, ticker_col, prediction_col, realized_return_col]
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns: {missing}")

    work = df[required].dropna().copy()
    work[date_col] = pd.to_datetime(work[date_col], errors="coerce")
    work = work.dropna(subset=[date_col]).sort_values([date_col, ticker_col]).reset_index(drop=True)

    if work.empty:
        raise ValueError("No valid rows in ensemble scored test file.")

    # Non-overlapping 5-day periods
    unique_dates = sorted(work[date_col].dropna().unique())
    selected_dates = unique_dates[::5]
    work = work[work[date_col].isin(selected_dates)].copy()

    cost_rate = transaction_cost_bps / 10000.0

    rows: List[Dict[str, float | str | pd.Timestamp]] = []
    prev_weights: Dict[str, float] | None = None

    for dt, group in work.groupby(date_col, sort=True):
        ranked = group.sort_values(prediction_col, ascending=False)
        if len(ranked) < top_n + bottom_n:
            continue

        long_leg = ranked.head(top_n)
        short_leg = ranked.tail(bottom_n)

        weights: Dict[str, float] = {}
        for ticker in long_leg[ticker_col]:
            weights[str(ticker)] = 1.0 / top_n
        for ticker in short_leg[ticker_col]:
            weights[str(ticker)] = weights.get(str(ticker), 0.0) - 1.0 / bottom_n

        gross_return = 0.0
        for _, row in ranked.iterrows():
            ticker = str(row[ticker_col])
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

        rows.append(
            {
                "date": dt,
                "strategy_return": net_return,
                "strategy_turnover": turnover,
            }
        )

        prev_weights = weights

    out = pd.DataFrame(rows).sort_values("date").reset_index(drop=True)
    if out.empty:
        raise ValueError("No strategy periods were generated.")
    return out


def download_spy_returns(start_date: pd.Timestamp, end_date: pd.Timestamp) -> pd.DataFrame:
    start = (start_date - pd.Timedelta(days=15)).strftime("%Y-%m-%d")
    end = (end_date + pd.Timedelta(days=15)).strftime("%Y-%m-%d")

    spy = yf.download("SPY", start=start, end=end, auto_adjust=True, progress=False)

    if spy.empty:
        raise ValueError("Downloaded SPY data is empty.")

    if isinstance(spy.columns, pd.MultiIndex):
        spy.columns = [
            "_".join([str(x) for x in col if str(x) != ""]).strip("_")
            for col in spy.columns.to_list()
        ]

    spy = spy.reset_index()

    possible_date_cols = ["Date", "Datetime", "index", "date"]
    date_col = next((c for c in possible_date_cols if c in spy.columns), None)
    if date_col is None:
        raise ValueError(f"SPY data columns are unexpected: {spy.columns.tolist()}")

    spy = spy.rename(columns={date_col: "date"})
    spy["date"] = pd.to_datetime(spy["date"], errors="coerce").dt.tz_localize(None).dt.normalize()
    spy = spy.dropna(subset=["date"]).sort_values("date").reset_index(drop=True)

    close_candidates = [c for c in spy.columns if "close" in str(c).lower()]
    if not close_candidates:
        raise ValueError(f"Could not find SPY close column. Columns: {spy.columns.tolist()}")

    close_name = close_candidates[0]

    # Compute 5-trading-day forward return on every SPY trading day
    spy["spy_return_5d"] = spy[close_name].shift(-5) / spy[close_name] - 1.0

    return spy[["date", "spy_return_5d"]].dropna().copy()


def compute_summary_metrics(returns: pd.Series, periods_per_year: int) -> Dict[str, float]:
    returns = returns.dropna().astype(float)
    if returns.empty:
        raise ValueError("No returns available for summary metrics.")

    mean_return = float(returns.mean())
    volatility = float(returns.std(ddof=1)) if len(returns) > 1 else 0.0
    sharpe = 0.0 if volatility == 0.0 else float((mean_return / volatility) * np.sqrt(periods_per_year))

    cumulative_return = float((1 + returns).prod() - 1.0)
    annualized_return = float((1 + cumulative_return) ** (periods_per_year / len(returns)) - 1.0)

    return {
        "mean_return": mean_return,
        "volatility": volatility,
        "annualized_return": annualized_return,
        "cumulative_return": cumulative_return,
        "sharpe": sharpe,
        "max_drawdown": _max_drawdown(returns),
        "num_periods": float(len(returns)),
    }


def main() -> None:
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    df = pd.read_csv(INPUT_PATH)
    df[DATE_COL] = pd.to_datetime(df[DATE_COL], errors="coerce")
    df = df.dropna(subset=[DATE_COL]).sort_values([DATE_COL, TICKER_COL]).reset_index(drop=True)

    strategy_df = compute_strategy_period_returns(
        df=df,
        date_col=DATE_COL,
        ticker_col=TICKER_COL,
        prediction_col=PREDICTION_COL,
        realized_return_col=TARGET_COL,
        top_n=TOP_N,
        bottom_n=BOTTOM_N,
        transaction_cost_bps=TRANSACTION_COST_BPS,
    )

    start_date = strategy_df["date"].min()
    end_date = strategy_df["date"].max()

    spy_df = download_spy_returns(start_date=start_date, end_date=end_date)

    strategy_df["date"] = pd.to_datetime(strategy_df["date"], errors="coerce").dt.tz_localize(None).dt.normalize()
    spy_df["date"] = pd.to_datetime(spy_df["date"], errors="coerce").dt.tz_localize(None).dt.normalize()

    merged = strategy_df.merge(spy_df, on="date", how="left")

    if merged["spy_return_5d"].isna().all():
        print("Strategy dates sample:", strategy_df["date"].head().tolist())
        print("SPY dates sample:", spy_df["date"].head().tolist())
        raise ValueError("SPY returns could not be aligned to strategy dates.")

    merged = merged.dropna(subset=["spy_return_5d"]).copy()

    strategy_metrics = compute_summary_metrics(merged["strategy_return"], PERIODS_PER_YEAR)
    spy_metrics = compute_summary_metrics(merged["spy_return_5d"], PERIODS_PER_YEAR)

    summary_df = pd.DataFrame(
        [
            {"series": "ensemble_strategy", **strategy_metrics},
            {"series": "spy_buy_hold_5d", **spy_metrics},
        ]
    )

    merged["strategy_equity"] = (1 + merged["strategy_return"]).cumprod()
    merged["spy_equity"] = (1 + merged["spy_return_5d"]).cumprod()

    returns_path = os.path.join(OUTPUT_DIR, "spy_benchmark_aligned_returns.csv")
    metrics_path = os.path.join(OUTPUT_DIR, "spy_benchmark_metrics.csv")

    merged.to_csv(returns_path, index=False)
    summary_df.to_csv(metrics_path, index=False)

    print("\nSPY benchmark metrics:")
    print(summary_df.to_string(index=False))
    print(f"\nSaved aligned returns to: {returns_path}")
    print(f"Saved benchmark metrics to: {metrics_path}")


if __name__ == "__main__":
    main()
