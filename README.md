# Data Preprocessing

The pipeline follows:
raw → interim → processed

## Steps

### 1. Data Download (`data/raw/`)

- Source: `yfinance`
- Contains raw OHLCV data
- No transformations applied
- Output: `data/raw/prices_raw.csv`

---

### 2. Data Cleaning (`data/interim/`)

- Convert `date` to datetime
- Sort by `ticker` and `date`
- Remove duplicates
- Handle missing values
- Ensure valid prices (> 0) and volume (≥ 0)
- Output: `data/interim/prices_interim.csv`

---

### 3. Dataset Construction (`data/processed/`)

- Compute features:
  - Returns (1D, 5D, 20D, 60D)
  - Rolling volatility (20D)
  - Rolling average volume (20D)

- Compute targets:
  - Forward 1-day return
  - Forward 5-day return

- Apply cross-sectional normalization (per date)
- Output: `data/processed/model_dataset.csv`

---

## Scripts

Run from project root:

```bash
python -m scripts.run_download
python -m scripts.run_clean_data
python -m scripts.run_make_dataset
```

---

## Notes

- Data is in long format (`date`, `ticker`)
- Features are normalized per date
- No data leakage introduced
- Dataset is ready for modeling
