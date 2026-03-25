import pandas as pd
from pathlib import Path


EXPECTED_COLUMNS = ["date", "ticker", "open", "high", "low", "close", "adj_close", "volume"]


def load_raw_data(path: str = "data/raw/prices_raw.csv") -> pd.DataFrame:
    return pd.read_csv(path)


def clean_price_data(df: pd.DataFrame) -> pd.DataFrame:
    # Ensure expected columns exist
    missing_cols = [col for col in EXPECTED_COLUMNS if col not in df.columns]
    if missing_cols:
        raise ValueError(f"Missing required columns: {missing_cols}")

    # Keep only needed columns
    df = df[EXPECTED_COLUMNS].copy()

    # Standardize ticker
    df["ticker"] = df["ticker"].astype(str).str.upper().str.strip()

    # Convert date
    df["date"] = pd.to_datetime(df["date"], errors="coerce", utc=True)
    df["date"] = df["date"].dt.tz_localize(None)

    # Convert numerics
    numeric_cols = ["open", "high", "low", "close", "adj_close", "volume"]
    for col in numeric_cols:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    # Drop rows with missing required values
    df = df.dropna(subset=EXPECTED_COLUMNS)

    # Remove impossible values
    price_cols = ["open", "high", "low", "close", "adj_close"]
    for col in price_cols:
        df = df[df[col] > 0]

    df = df[df["volume"] >= 0]

    # Drop duplicates
    df = df.drop_duplicates(subset=["date", "ticker"], keep="last")

    # Sort
    df = df.sort_values(["ticker", "date"]).reset_index(drop=True)

    return df


def save_interim_data(df: pd.DataFrame, path: str = "data/interim/prices_interim.csv") -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, index=False)