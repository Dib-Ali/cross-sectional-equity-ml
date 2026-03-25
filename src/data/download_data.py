from pathlib import Path
from typing import List
import time

import pandas as pd
import yfinance as yf


def download_price_data(
    tickers: List[str],
    start_date: str,
    end_date: str,
) -> pd.DataFrame:
    """
    Download OHLCV data from yfinance in long format.

    Downloads one ticker at a time for better reproducibility across machines.
    Raises an error only if all tickers fail.
    """
    frames = []
    failed = []

    for ticker in tickers:
        print(f"Downloading {ticker}...")

        try:
            stock = yf.Ticker(ticker)
            df_ticker = stock.history(
                start=start_date,
                end=end_date,
                auto_adjust=False,
            )

            if df_ticker is None or df_ticker.empty:
                print(f"No data returned for {ticker}.")
                failed.append(ticker)
                continue

            df_ticker = df_ticker.reset_index()
            df_ticker["ticker"] = ticker

            df_ticker.columns = [
                str(c).strip().lower().replace(" ", "_") for c in df_ticker.columns
            ]

            if "adj_close" not in df_ticker.columns:
                df_ticker["adj_close"] = df_ticker["close"]

            expected_cols = [
                "date", "ticker", "open", "high", "low",
                "close", "adj_close", "volume"
            ]
            missing = [c for c in expected_cols if c not in df_ticker.columns]
            if missing:
                print(f"Skipping {ticker}: missing columns {missing}")
                failed.append(ticker)
                continue

            df_ticker = df_ticker[expected_cols]
            frames.append(df_ticker)

            time.sleep(1)

        except Exception as e:
            print(f"Failed to download {ticker}: {e}")
            failed.append(ticker)

    if not frames:
        raise ValueError("No data downloaded from yfinance.")

    df = pd.concat(frames, ignore_index=True)
    df = df.sort_values(["date", "ticker"]).reset_index(drop=True)

    print(f"Successfully downloaded {len(frames)} ticker(s).")
    if failed:
        print("Failed tickers:", failed)

    return df


def save_raw_data(df: pd.DataFrame, output_path: str = "data/raw/prices_raw.csv") -> None:
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, index=False)