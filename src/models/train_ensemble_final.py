from __future__ import annotations

from itertools import product
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd

from src.models.train_regularized import (
    predict_regularized_regression,
    train_regularized_regression,
)
from src.models.train_boosting import (
    predict_gradient_boosting_regressor,
    train_gradient_boosting_regressor,
)
from src.validation.evaluation_ml import (
    compute_regression_metrics,
    compute_ic_series,
    compute_icir,
)


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
    required = [date_col, "ticker", prediction_col, realized_return_col]
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns for financial metrics: {missing}")

    work = df[required].dropna().copy()
    work[date_col] = pd.to_datetime(work[date_col], errors="coerce")
    work = work.dropna(subset=[date_col]).sort_values([date_col, "ticker"]).reset_index(drop=True)

    if work.empty:
        raise ValueError("No valid rows available to compute financial metrics.")

    if realized_return_col == "target_5d":
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
        for ticker in long_leg["ticker"]:
            weights[str(ticker)] = 1.0 / top_n
        for ticker in short_leg["ticker"]:
            weights[str(ticker)] = weights.get(str(ticker), 0.0) - 1.0 / bottom_n

        gross_return = 0.0
        for _, row in ranked.iterrows():
            ticker = str(row["ticker"])
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

    return {
        "portfolio_mean_return": mean_return,
        "portfolio_volatility": volatility,
        "portfolio_cumulative_return": float((1 + r).prod() - 1.0),
        "portfolio_sharpe": sharpe,
        "portfolio_max_drawdown": _max_drawdown(r),
        "portfolio_turnover_avg": float(np.mean(period_turnover)),
        "portfolio_num_periods": float(len(r)),
    }


def evaluate_predictions(
    df: pd.DataFrame,
    date_col: str,
    target_col: str,
    prediction_col: str = "prediction",
    top_n: int = 10,
    bottom_n: int = 10,
    transaction_cost_bps: float = 10.0,
    periods_per_year: int = 52,
) -> Dict[str, float]:
    ml_metrics = compute_regression_metrics(df[target_col], df[prediction_col])

    ic_series = compute_ic_series(
        df=df,
        date_col=date_col,
        target_col=target_col,
        prediction_col=prediction_col,
    )
    ml_metrics["ic_mean"] = float(ic_series.mean())
    ml_metrics["ic_std"] = float(ic_series.std())
    ml_metrics["icir"] = compute_icir(ic_series)
    ml_metrics["ic_pct_positive"] = float((ic_series > 0).mean())

    financial_metrics = compute_long_short_financial_metrics(
        df=df,
        date_col=date_col,
        prediction_col=prediction_col,
        realized_return_col=target_col,
        top_n=top_n,
        bottom_n=bottom_n,
        transaction_cost_bps=transaction_cost_bps,
        periods_per_year=periods_per_year,
    )

    return {
        **ml_metrics,
        **financial_metrics,
    }


def train_base_models(
    train_df: pd.DataFrame,
    feature_cols: List[str],
    target_col: str,
) -> Dict[str, object]:
    models: Dict[str, object] = {}

    models["ridge"] = train_regularized_regression(
        train_df=train_df,
        feature_cols=feature_cols,
        target_col=target_col,
        model_name="ridge",
        alpha=10.0,
        l1_ratio=0.5,
        random_state=42,
        max_iter=10000,
    )



    models["elasticnet"] = train_regularized_regression(
        train_df=train_df,
        feature_cols=feature_cols,
        target_col=target_col,
        model_name="elasticnet",
        alpha=0.0005,
        l1_ratio=0.5,
        random_state=42,
        max_iter=10000,
    )


    return models


def score_base_models(
    models: Dict[str, object],
    df: pd.DataFrame,
    feature_cols: List[str],
) -> pd.DataFrame:
    scored = df.copy()

    scored["pred_ridge"] = predict_regularized_regression(
        models["ridge"], scored, feature_cols
    )

    scored["pred_elasticnet"] = predict_regularized_regression(
        models["elasticnet"], scored, feature_cols
    )


    scored = scored.drop_duplicates(subset=["date", "ticker"], keep="last")
    return scored


def build_ensemble_prediction(df: pd.DataFrame) -> pd.Series:
    pred = (
        0.4 * df["pred_ridge"]
        + 0.6 * df["pred_elasticnet"]
    )
    return pd.Series(pred, index=df.index, name="prediction")





