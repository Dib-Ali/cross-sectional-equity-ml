from __future__ import annotations

from typing import Dict, List, Optional, Tuple

import pandas as pd
from sklearn.ensemble import GradientBoostingRegressor

from src.validation.evaluation_ml import compute_regression_metrics


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


def train_gradient_boosting_regressor(
    train_df: pd.DataFrame,
    feature_cols: List[str],
    target_col: str,
    model_params: Optional[Dict] = None,
) -> GradientBoostingRegressor:
    """
    Train a gradient boosting regressor.
    """
    if train_df.empty:
        raise ValueError("Training dataframe is empty.")

    _validate_feature_target_columns(train_df, feature_cols, target_col)

    model_df = train_df[feature_cols + [target_col]].dropna().copy()
    if model_df.empty:
        raise ValueError("No valid rows remain after dropping NaNs.")

    X_train = model_df[feature_cols]
    y_train = model_df[target_col]

    default_params = {
        "n_estimators": 100,
        "learning_rate": 0.05,
        "max_depth": 3,
        "random_state": 42,
    }

    if model_params is not None:
        default_params.update(model_params)

    model = GradientBoostingRegressor(**default_params)
    model.fit(X_train, y_train)
    return model


def predict_gradient_boosting_regressor(
    model: GradientBoostingRegressor,
    df: pd.DataFrame,
    feature_cols: List[str],
) -> pd.Series:
    """
    Generate predictions from a trained gradient boosting regressor.
    """
    if df.empty:
        raise ValueError("Prediction dataframe is empty.")

    missing_features = [col for col in feature_cols if col not in df.columns]
    if missing_features:
        raise ValueError(f"Missing feature columns: {missing_features}")

    prediction_df = df[feature_cols].dropna().copy()
    if prediction_df.empty:
        raise ValueError("No valid rows remain for prediction after dropping NaNs.")

    preds = model.predict(prediction_df)
    return pd.Series(preds, index=prediction_df.index, name="prediction")


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

    cost_rate = transaction_cost_bps / 10000.0

    period_returns = []
    period_turnover = []
    prev_weights = None

    for _, group in work.groupby(date_col, sort=True):
        ranked = group.sort_values(prediction_col, ascending=False)
        if len(ranked) < top_n + bottom_n:
            continue

        long_leg = ranked.head(top_n)
        short_leg = ranked.tail(bottom_n)

        weights = {}
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

    return {
        "portfolio_mean_return": mean_return,
        "portfolio_volatility": volatility,
        "portfolio_cumulative_return": float((1 + r).prod() - 1.0),
        "portfolio_sharpe": sharpe,
        "portfolio_max_drawdown": _max_drawdown(r),
        "portfolio_turnover_avg": float(sum(period_turnover) / len(period_turnover)),
        "portfolio_num_periods": float(len(r)),
    }


def evaluate_boosting_predictions(
    df: pd.DataFrame,
    date_col: str,
    target_col: str,
    prediction_col: str = "prediction",
    top_n: int = 10,
    bottom_n: int = 10,
    transaction_cost_bps: float = 10.0,
    periods_per_year: int = 52,
) -> Dict[str, float]:
    if target_col not in df.columns:
        raise ValueError(f"Missing target column: {target_col}")
    if prediction_col not in df.columns:
        raise ValueError(f"Missing prediction column: {prediction_col}")

    ml_metrics = compute_regression_metrics(df[target_col], df[prediction_col])
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


def run_boosting_validation(
    train_df: pd.DataFrame,
    val_df: pd.DataFrame,
    feature_cols: List[str],
    target_col: str,
    date_col: str = "date",
    transaction_cost_bps: float = 10.0,
    periods_per_year: int = 52,
    model_params: Optional[Dict] = None,
) -> Tuple[GradientBoostingRegressor, pd.DataFrame, Dict[str, float]]:
    _validate_feature_target_columns(train_df, feature_cols, target_col)
    _validate_feature_target_columns(val_df, feature_cols, target_col)

    model = train_gradient_boosting_regressor(
        train_df=train_df,
        feature_cols=feature_cols,
        target_col=target_col,
        model_params=model_params,
    )

    scored_val = val_df.copy()
    scored_val["prediction"] = predict_gradient_boosting_regressor(model, scored_val, feature_cols)

    metrics = evaluate_boosting_predictions(
        df=scored_val,
        date_col=date_col,
        target_col=target_col,
        prediction_col="prediction",
        transaction_cost_bps=transaction_cost_bps,
        periods_per_year=periods_per_year,
    )

    return model, scored_val, metrics