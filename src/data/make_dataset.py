import pandas as pd
from pathlib import Path


def load_interim_data(path: str = "data/interim/prices_interim.csv") -> pd.DataFrame:
    return pd.read_csv(path, parse_dates=["date"])


def make_model_dataset(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df = df.sort_values(["ticker", "date"]).reset_index(drop=True)

    # Daily return
    df["return_1d"] = df.groupby("ticker")["adj_close"].pct_change(1)

    # Backward-looking features
    df["return_5d"] = df.groupby("ticker")["adj_close"].pct_change(5)
    df["return_20d"] = df.groupby("ticker")["adj_close"].pct_change(20)
    df["return_60d"] = df.groupby("ticker")["adj_close"].pct_change(60)

    # Rolling volatility from daily returns
    df["volatility_20d"] = (
        df.groupby("ticker")["return_1d"]
        .rolling(20)
        .std()
        .reset_index(level=0, drop=True)
    )

    # Simple liquidity feature
    df["volume_avg_20d"] = (
        df.groupby("ticker")["volume"]
        .rolling(20)
        .mean()
        .reset_index(level=0, drop=True)
    )

    # Forward targets
    df["target_1d"] = (
        df.groupby("ticker")["adj_close"].shift(-1) / df["adj_close"] - 1
    )

    df["target_5d"] = (
        df.groupby("ticker")["adj_close"].shift(-5) / df["adj_close"] - 1
    )

    # Keep rows where features and targets exist
    required_cols = [
        "return_1d",
        "return_5d",
        "return_20d",
        "return_60d",
        "volatility_20d",
        "volume_avg_20d",
        "target_1d",
        "target_5d",
    ]
    df = df.dropna(subset=required_cols).reset_index(drop=True)
    
    feature_cols = [
    "return_1d",
    "return_5d",
    "return_20d",
    "return_60d",
    "volatility_20d",
    "volume_avg_20d",
    ]

    # Cross-sectional z-score normalization per date
    df[feature_cols] = df.groupby("date")[feature_cols].transform(
        lambda x: (x - x.mean()) / (x.std() + 1e-8)
    )

    return df


def save_processed_data(df: pd.DataFrame, path: str = "data/processed/model_dataset.csv") -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, index=False)

