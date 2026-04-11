import pandas as pd

from src.validation.splitters import chronological_train_val_test_split


def test_chronological_train_val_test_split():
    dates = pd.date_range("2022-12-20", periods=20, freq="M")
    df = pd.DataFrame(
        {
            "date": dates.tolist() * 2,
            "ticker": ["AAPL"] * 20 + ["MSFT"] * 20,
            "x": range(40),
        }
    )

    train_df, val_df, test_df = chronological_train_val_test_split(
        df,
        date_col="date",
        train_end="2023-06-30",
        val_end="2023-12-31",
    )

    assert not train_df.empty
    assert not val_df.empty
    assert not test_df.empty
    assert train_df["date"].max() <= pd.Timestamp("2023-06-30")
    assert val_df["date"].min() > pd.Timestamp("2023-06-30")
    assert val_df["date"].max() <= pd.Timestamp("2023-12-31")
    assert test_df["date"].min() > pd.Timestamp("2023-12-31")