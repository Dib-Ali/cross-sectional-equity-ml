from __future__ import annotations

from typing import Dict, List, Optional, Tuple

import pandas as pd
from sklearn.ensemble import RandomForestRegressor

from src.validation.evaluation_ml import (
    compute_regression_metrics,
    compute_ic_series,
    compute_icir,
)


def _validate_feature_target_columns(
    df: pd.DataFrame,
    feature_cols: List[str],
    target_col: str,
) -> None:
    missing_features = [col for col in feature_cols if col not in df.columns]
    if missing_features:
        raise ValueError(f"Missing feature columns: {missing_features}")

    if target_col not in df.columns:
        raise ValueError(f"Missing target column: {target_col}")


def train_random_forest_regression(
    train_df: pd.DataFrame,
    feature_cols: List[str],
    target_col: str,
    n_estimators: int = 100,
    max_depth: Optional[int] = None,
    min_samples_split: int = 2,
    min_samples_leaf: int = 1,
    random_state: int = 42,
) -> RandomForestRegressor:
    """
    Train a random forest regression model.

    Notes:
    - This function assumes train_df is already chronologically separated from
      validation/test data. Do not shuffle panel financial data.
    """
    if train_df.empty:
        raise ValueError("train_df is empty.")

    _validate_feature_target_columns(train_df, feature_cols, target_col)

    model = RandomForestRegressor(
        n_estimators=n_estimators,
        max_depth=max_depth,
        min_samples_split=min_samples_split,
        min_samples_leaf=min_samples_leaf,
        random_state=random_state,
        n_jobs=-1,
    )

    X_train = train_df[feature_cols]
    y_train = train_df[target_col]
    model.fit(X_train, y_train)
    return model


def predict_random_forest_regression(
    model: RandomForestRegressor,
    df: pd.DataFrame,
    feature_cols: List[str],
) -> pd.Series:
    """Generate predictions from a trained random forest model."""
    missing_features = [col for col in feature_cols if col not in df.columns]
    if missing_features:
        raise ValueError(f"Missing feature columns: {missing_features}")

    preds = model.predict(df[feature_cols])
    return pd.Series(preds, index=df.index, name="prediction")


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
    periods_per_year: int = 252,
) -> Dict[str, float]:
    """
    Compute long-short financial metrics from cross-sectional predictions.
    """
    required = [date_col, "ticker", prediction_col, realized_return_col]
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns for financial metrics: {missing}")

    if top_n <= 0 or bottom_n <= 0:
        raise ValueError("top_n and bottom_n must be positive integers.")
    if periods_per_year <= 0:
        raise ValueError("periods_per_year must be positive.")

    work = df[required].dropna().copy()
    work[date_col] = pd.to_datetime(work[date_col], errors="coerce")
    work = work.dropna(subset=[date_col]).sort_values([date_col, "ticker"]).reset_index(drop=True)

    if work.empty:
        raise ValueError("No valid rows available to compute financial metrics.")

    cost_rate = transaction_cost_bps / 10000.0

    period_returns: List[float] = []
    period_turnover: List[float] = []
    prev_weights: Optional[Dict[str, float]] = None

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
        raise ValueError("Unable to compute portfolio returns: not enough names per date.")

    r = pd.Series(period_returns, dtype=float)

    mean_return = float(r.mean())
    volatility = float(r.std(ddof=1)) if len(r) > 1 else 0.0
    sharpe = 0.0 if volatility == 0.0 else float((mean_return / volatility) * (periods_per_year ** 0.5))

    metrics = {
        "portfolio_mean_return": mean_return,
        "portfolio_volatility": volatility,
        "portfolio_cumulative_return": float((1 + r).prod() - 1.0),
        "portfolio_sharpe": sharpe,
        "portfolio_max_drawdown": _max_drawdown(r),
        "portfolio_turnover_avg": float(pd.Series(period_turnover).mean()),
        "portfolio_num_periods": float(len(r)),
    }
    return metrics


def evaluate_random_forest_predictions(
    df: pd.DataFrame,
    date_col: str,
    target_col: str,
    prediction_col: str = "prediction",
    top_n: int = 10,
    bottom_n: int = 10,
    transaction_cost_bps: float = 10.0,
    periods_per_year: int = 252,
) -> Dict[str, float]:
    """
    Return combined machine-learning and financial metrics in one dictionary.
    """
    if target_col not in df.columns:
        raise ValueError(f"Missing target column: {target_col}")
    if prediction_col not in df.columns:
        raise ValueError(f"Missing prediction column: {prediction_col}")

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


def run_random_forest_validation(
    train_df: pd.DataFrame,
    val_df: pd.DataFrame,
    feature_cols: List[str],
    target_col: str,
    date_col: str = "date",
    n_estimators: int = 100,
    max_depth: Optional[int] = None,
    min_samples_split: int = 2,
    min_samples_leaf: int = 1,
    transaction_cost_bps: float = 10.0,
    periods_per_year: int = 252,
    random_state: int = 42,
    top_n: int = 10,
    bottom_n: int = 10,
) -> Tuple[RandomForestRegressor, pd.DataFrame, Dict[str, float]]:
    """
    Train random forest model, predict on validation set, and evaluate.
    """
    _validate_feature_target_columns(train_df, feature_cols, target_col)
    _validate_feature_target_columns(val_df, feature_cols, target_col)

    model = train_random_forest_regression(
        train_df=train_df,
        feature_cols=feature_cols,
        target_col=target_col,
        n_estimators=n_estimators,
        max_depth=max_depth,
        min_samples_split=min_samples_split,
        min_samples_leaf=min_samples_leaf,
        random_state=random_state,
    )

    scored_val = val_df.copy()
    scored_val["prediction"] = predict_random_forest_regression(model, scored_val, feature_cols)

    metrics = evaluate_random_forest_predictions(
        df=scored_val,
        date_col=date_col,
        target_col=target_col,
        prediction_col="prediction",
        top_n=top_n,
        bottom_n=bottom_n,
        transaction_cost_bps=transaction_cost_bps,
        periods_per_year=periods_per_year,
    )

    return model, scored_val, metrics