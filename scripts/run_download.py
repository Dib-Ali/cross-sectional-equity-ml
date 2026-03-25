from src.data.build_universe import get_universe
from src.data.download_data import download_price_data, save_raw_data


def main():
    tickers = get_universe()
    print(f"Downloading data for {len(tickers)} tickers...")

    df = download_price_data(tickers, "2015-01-01", "2024-12-31")

    save_raw_data(df, "data/raw/prices_raw.csv")

    print("Download complete.")
    print(df.head())
    print(df.shape)


if __name__ == "__main__":
    main()