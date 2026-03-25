from pathlib import Path
from typing import List

import pandas as pd
import yfinance as yf


def download_price_data(
    tickers: List[str],
    start_date: str,
    end_date: str,
) -> pd.DataFrame:
    """
    Download OHLCV data from yfinance and return it in long format.
    """
    data = yf.download(
        tickers=tickers,
        start=start_date,
        end=end_date,
        group_by="ticker",
        auto_adjust=False,
        threads=True,
        progress=False,
    )

    frames = []

    for ticker in tickers:
        try:
            df_ticker = data[ticker].copy()
        except Exception:
            continue

        if df_ticker.empty:
            continue

        df_ticker = df_ticker.reset_index()
        df_ticker["ticker"] = ticker
        frames.append(df_ticker)

    if not frames:
        raise ValueError("No data downloaded from yfinance.")

    df = pd.concat(frames, ignore_index=True)
    df.columns = [str(c).strip().lower().replace(" ", "_") for c in df.columns]

    rename_map = {
        "adj_close": "adj_close",
        "date": "date",
        "open": "open",
        "high": "high",
        "low": "low",
        "close": "close",
        "volume": "volume",
        "ticker": "ticker",
    }
    df = df.rename(columns=rename_map)

    expected_cols = ["date", "ticker", "open", "high", "low", "close", "adj_close", "volume"]
    missing = [c for c in expected_cols if c not in df.columns]
    if missing:
        raise ValueError(f"Missing expected columns after download: {missing}")

    df = df[expected_cols].sort_values(["date", "ticker"]).reset_index(drop=True)
    return df


def save_raw_data(df: pd.DataFrame, output_path: str = "data/raw/prices_raw.csv") -> None:
    """
    Save raw downloaded data to CSV.
    """
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, index=False)