from __future__ import annotations

from typing import Any, Dict, List, Literal, Optional, Tuple

import numpy as np
import pandas as pd
from sklearn.linear_model import ElasticNet, Lasso, Ridge
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from src.validation.evaluation_ml import compute_regression_metrics

RegularizedModelName = Literal["ridge", "lasso", "elasticnet"]


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


def _build_regularized_pipeline(
	model_name: RegularizedModelName,
	alpha: float = 1.0,
	l1_ratio: float = 0.5,
	random_state: int = 42,
	max_iter: int = 10000,
) -> Pipeline:
	if alpha <= 0:
		raise ValueError("alpha must be > 0.")

	model_name = model_name.lower()

	if model_name == "ridge":
		estimator = Ridge(alpha=alpha, random_state=random_state)
	elif model_name == "lasso":
		estimator = Lasso(alpha=alpha, random_state=random_state, max_iter=max_iter)
	elif model_name == "elasticnet":
		if not 0 <= l1_ratio <= 1:
			raise ValueError("l1_ratio must be between 0 and 1 for ElasticNet.")
		estimator = ElasticNet(
			alpha=alpha,
			l1_ratio=l1_ratio,
			random_state=random_state,
			max_iter=max_iter,
		)
	else:
		raise ValueError("model_name must be one of: ridge, lasso, elasticnet.")

	# Scaling inside the pipeline avoids leakage from validation into train.
	return Pipeline(
		steps=[
			("scaler", StandardScaler()),
			("model", estimator),
		]
	)


def train_regularized_regression(
	train_df: pd.DataFrame,
	feature_cols: List[str],
	target_col: str,
	model_name: RegularizedModelName,
	alpha: float = 1.0,
	l1_ratio: float = 0.5,
	random_state: int = 42,
	max_iter: int = 10000,
) -> Pipeline:
	"""
	Train a regularized regression model (Ridge, Lasso, or ElasticNet).

	Notes:
	- This function assumes train_df is already chronologically separated from
	  validation/test data. Do not shuffle panel financial data.
	- Features are scaled inside the pipeline with train-only statistics.
	"""
	if train_df.empty:
		raise ValueError("train_df is empty.")

	_validate_feature_target_columns(train_df, feature_cols, target_col)

	model = _build_regularized_pipeline(
		model_name=model_name,
		alpha=alpha,
		l1_ratio=l1_ratio,
		random_state=random_state,
		max_iter=max_iter,
	)

	X_train = train_df[feature_cols]
	y_train = train_df[target_col]
	model.fit(X_train, y_train)
	return model


def train_ridge_regression(
	train_df: pd.DataFrame,
	feature_cols: List[str],
	target_col: str,
	alpha: float = 1.0,
	random_state: int = 42,
) -> Pipeline:
	"""Train a Ridge regression model."""
	return train_regularized_regression(
		train_df=train_df,
		feature_cols=feature_cols,
		target_col=target_col,
		model_name="ridge",
		alpha=alpha,
		random_state=random_state,
	)


def train_lasso_regression(
	train_df: pd.DataFrame,
	feature_cols: List[str],
	target_col: str,
	alpha: float = 1.0,
	random_state: int = 42,
	max_iter: int = 10000,
) -> Pipeline:
	"""Train a Lasso regression model."""
	return train_regularized_regression(
		train_df=train_df,
		feature_cols=feature_cols,
		target_col=target_col,
		model_name="lasso",
		alpha=alpha,
		random_state=random_state,
		max_iter=max_iter,
	)


def train_elasticnet_regression(
	train_df: pd.DataFrame,
	feature_cols: List[str],
	target_col: str,
	alpha: float = 1.0,
	l1_ratio: float = 0.5,
	random_state: int = 42,
	max_iter: int = 10000,
) -> Pipeline:
	"""Train an ElasticNet regression model."""
	return train_regularized_regression(
		train_df=train_df,
		feature_cols=feature_cols,
		target_col=target_col,
		model_name="elasticnet",
		alpha=alpha,
		l1_ratio=l1_ratio,
		random_state=random_state,
		max_iter=max_iter,
	)


def predict_regularized_regression(
	model: Pipeline,
	df: pd.DataFrame,
	feature_cols: List[str],
) -> pd.Series:
	"""Generate predictions from a trained regularized model pipeline."""
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
	periods_per_year: int = 52,
) -> Dict[str, float]:
	"""
	Compute long-short financial metrics from cross-sectional predictions.

	Strategy:
	- For each date, go long top_n by prediction and short bottom_n.
	- Equal-weight each side so gross exposure is 2.0 and net exposure is 0.0.
	- Deduct turnover-based transaction costs.

	transaction_cost_bps is interpreted as one-way bps cost per unit notional.
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

	for dt, group in work.groupby(date_col, sort=True):
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


def evaluate_regularized_predictions(
	df: pd.DataFrame,
	date_col: str,
	target_col: str,
	prediction_col: str = "prediction",
	top_n: int = 10,
	bottom_n: int = 10,
	transaction_cost_bps: float = 10.0,
	periods_per_year: int = 52,
) -> Dict[str, float]:
	"""
	Return combined machine-learning and financial metrics in one dictionary.
	"""
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


def run_regularized_validation(
	train_df: pd.DataFrame,
	val_df: pd.DataFrame,
	feature_cols: List[str],
	target_col: str,
	date_col: str = "date",
	model_name: RegularizedModelName = "ridge",
	alpha: float = 1.0,
	l1_ratio: float = 0.5,
	transaction_cost_bps: float = 10.0,
	periods_per_year: int = 52,
	random_state: int = 42,
	max_iter: int = 10000,
) -> Tuple[Pipeline, pd.DataFrame, Dict[str, float]]:
	"""
	Train selected regularized model, predict on validation set, and evaluate.

	Returns:
	- fitted model
	- validation dataframe with prediction column
	- merged metric dictionary (ML + financial)
	"""
	_validate_feature_target_columns(train_df, feature_cols, target_col)
	_validate_feature_target_columns(val_df, feature_cols, target_col)

	model = train_regularized_regression(
		train_df=train_df,
		feature_cols=feature_cols,
		target_col=target_col,
		model_name=model_name,
		alpha=alpha,
		l1_ratio=l1_ratio,
		random_state=random_state,
		max_iter=max_iter,
	)

	scored_val = val_df.copy()
	scored_val["prediction"] = predict_regularized_regression(model, scored_val, feature_cols)

	metrics = evaluate_regularized_predictions(
		df=scored_val,
		date_col=date_col,
		target_col=target_col,
		prediction_col="prediction",
		top_n=10,
		bottom_n=10,
		transaction_cost_bps=transaction_cost_bps,
		periods_per_year=periods_per_year,
	)

	return model, scored_val, metrics
