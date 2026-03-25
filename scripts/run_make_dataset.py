from src.data.make_dataset import (
    load_interim_data,
    make_model_dataset,
    save_processed_data,
)


def main():
    df = load_interim_data()
    print("Interim shape:", df.shape)

    df_model = make_model_dataset(df)
    print("Processed shape:", df_model.shape)
    print(df_model.head())

    save_processed_data(df_model)
    print("Processed dataset saved.")
    
    

if __name__ == "__main__":
    main()