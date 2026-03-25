from src.data.clean_data import load_raw_data, clean_price_data, save_interim_data


def main():
    df = load_raw_data()
    print("Raw shape:", df.shape)

    df_clean = clean_price_data(df)
    print("Clean shape:", df_clean.shape)

    save_interim_data(df_clean)
    print("Cleaned data saved.")


if __name__ == "__main__":
    main()