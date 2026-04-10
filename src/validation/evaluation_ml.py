from __future__ import annotations

from typing import Dict

import numpy as np
import pandas as pd
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

from scipy.stats import spearmanr
import numpy as np
import pandas as pd
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
def compute_regression_metrics(
    y_true: pd.Series,
    y_pred: pd.Series,
) -> Dict[str, float]:
    """
    Compute standard regression metrics.
    """
    if len(y_true) != len(y_pred):
        raise ValueError("y_true and y_pred must have the same length.")

    mse = mean_squared_error(y_true, y_pred)
    rmse = float(np.sqrt(mse))
    mae = mean_absolute_error(y_true, y_pred)
    r2 = r2_score(y_true, y_pred)

    return {
        "rmse": rmse,
        "mae": float(mae),
        "r2": float(r2),
    }
def compute_ic(y_true: pd.Series, y_pred: pd.Series) -> float:
    """
    Spearman rank Information Coefficient between predicted and realized returns.
    """
    mask = y_true.notna() & y_pred.notna()
    if mask.sum() < 2:
        return float("nan")

    ic, _ = spearmanr(y_true[mask], y_pred[mask])
    return float(ic)


def compute_ic_series(
    df: pd.DataFrame,
    date_col: str,
    target_col: str,
    prediction_col: str,
) -> pd.Series:
    """
    Compute per-date cross-sectional IC series.
    """
    def _ic(group: pd.DataFrame) -> float:
        return compute_ic(group[target_col], group[prediction_col])

    return df.groupby(date_col).apply(_ic)


def compute_icir(ic_series: pd.Series) -> float:
    """
    Information Coefficient Information Ratio.
    """
    std = ic_series.std(ddof=1)
    if std == 0 or np.isnan(std):
        return 0.0
    return float(ic_series.mean() / std)