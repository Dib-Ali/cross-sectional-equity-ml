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
    #engineered features to introduce non-linearity
    eps = 1e-6

    df["momentum_ratio_1"] = df["return_5d"] / (df["return_20d"] + eps)
    df["momentum_ratio_2"] = df["return_20d"] / (df["return_60d"] + eps)
    df["risk_adjusted_return"] = df["return_20d"] / (df["volatility_20d"] + eps)
    df["interaction_1"] = df["return_20d"] * df["volume_avg_20d"]
    df["interaction_2"] = df["return_5d"] * df["volatility_20d"]
        # Cross-sectional rank feature
    df["rank_return_20d"] = df.groupby("date")["return_20d"].rank(pct=True)

    # Volatility regime feature: compare current vol to rolling 100-day average volatility
    df["volatility_20d_mean_100d"] = (
        df.groupby("ticker")["volatility_20d"]
        .rolling(100)
        .mean()
        .reset_index(level=0, drop=True)
    )
    df["volatility_regime"] = (
        df["volatility_20d"] > df["volatility_20d_mean_100d"]
    ).astype(int)

    # Trend consistency feature
    df["trend_consistency"] = (
        (df["return_5d"] > 0).astype(int) + (df["return_20d"] > 0).astype(int)
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
        "momentum_ratio_1",
        "momentum_ratio_2",
        "risk_adjusted_return",
        "interaction_1",
        "interaction_2",
        "rank_return_20d",
        "volatility_regime",
        "trend_consistency",
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
        "momentum_ratio_1",
        "momentum_ratio_2",
        "risk_adjusted_return",
        "interaction_1",
        "interaction_2",
        "rank_return_20d",
        "volatility_regime",
        "trend_consistency",
    ]

    # Cross-sectional z-score normalization per date
    df[feature_cols] = df.groupby("date")[feature_cols].transform(
        lambda x: (x - x.mean()) / (x.std() + 1e-8)
    )
    df = df.drop(columns=["volatility_20d_mean_100d"])
    return df


def save_processed_data(df: pd.DataFrame, path: str = "data/processed/model_dataset.csv") -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, index=False)

