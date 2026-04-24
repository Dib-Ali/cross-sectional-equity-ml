# Cross-Sectional Equity Return Prediction

> End-to-end quantitative alpha research pipeline — ML Final Project  
> Ali Dib · Chris Wehbe · Anthony Charbel · Kevin Khoury

---

## What This Project Does

This project builds a **market-neutral long-short equity strategy** using machine learning. Rather than predicting whether the market goes up or down, it ranks S&P 100 stocks against each other each week and goes long the top 10 predicted performers and short the bottom 10. This is called **cross-sectional alpha prediction** — the approach used by systematic quantitative hedge funds.

**Test period results (2024, out-of-sample):**
- IC Mean: 0.024 | Sharpe: 0.98 | Cumulative Return: +21% | Max Drawdown: -15%

---

## Project Structure

```
cross-sectional-equity-ml/
├── app.py                          # Streamlit demo dashboard (main entry point)
├── requirements.txt
├── Dockerfile
├── docker-compose.yml
├── configs/                        # Model and pipeline configuration
├── data/
│   ├── interim/                    # Cleaned raw price data
│   └── processed/                  # Feature-engineered model dataset
├── reports/
│   ├── tables/                     # All model results CSVs (pre-computed)
│   │   ├── ensemble_scored_test_target_5d.csv
│   │   ├── ensemble_final_metrics.csv
│   │   └── spy_benchmark_aligned_returns.csv
│   └── plots/                      # All 6 diagnostic plots (PNG)
├── scripts/                        # Pipeline runner scripts
│   ├── run_download.py
│   ├── run_clean_data.py
│   ├── run_make_dataset.py
│   ├── run_regularized_validation.py
│   ├── run_boosting_validation.py
│   ├── run_decision_tree_validation.py
│   ├── run_random_forest_validation.py
│   ├── run_svr_validation.py
│   ├── run_ensemble_final.py       # Full pipeline from models → ensemble
│   ├── run_spy_benchmark.py
│   ├── run_naive_momentum_benchmark.py
│   └── run_plots.py
└── src/
    ├── data/                       # Data download, cleaning, feature engineering
    ├── models/                     # All model training modules
    └── validation/                 # Evaluation metrics (IC, ICIR, Sharpe)
```

---

## Quick Start — Option A: Docker 

The Docker image includes all pre-computed results and the dataset. No retraining required.

```bash

#  with docker-compose
docker-compose up
```

Then open **http://localhost:8501** in your browser.

---

## Quick Start — Option B: Local Installation

### 1. Clone the repository

```bash
git clone https://github.com/Dib-Ali/cross-sectional-equity-ml.git
cd cross-sectional-equity-ml
```

### 2. Create virtual environment and install dependencies

```bash
python -m venv .venv

# Windows
.venv\Scripts\activate

# macOS / Linux
source .venv/bin/activate

pip install -r requirements.txt
```

### 3. Run the demo dashboard (pre-computed results)

All model results are already saved as CSV files in `reports/tables/`. No retraining needed.

```bash
streamlit run app.py
```

Open **http://localhost:8501** in your browser.

---

## Option C: Rerun the Full Pipeline

If you want to reproduce all results from scratch:

```bash
# Step 1 — Download S&P 100 price data from yfinance
python -m scripts.run_download

# Step 2 — Clean the raw data
python -m scripts.run_clean_data

# Step 3 — Build features (14 cross-sectional factors)
python -m scripts.run_make_dataset

# Step 4 — Train and evaluate all individual models
python -m scripts.run_regularized_validation
python -m scripts.run_boosting_validation
python -m scripts.run_decision_tree_validation
python -m scripts.run_random_forest_validation
python -m scripts.run_svr_validation

# Step 5 — Build the IC-weighted ensemble and run final test evaluation
python -m scripts.run_ensemble_final

# Step 6 — Run SPY and naive benchmarks
python -m scripts.run_spy_benchmark
python -m scripts.run_naive_momentum_benchmark

# Step 7 — Generate all plots
python -m scripts.run_plots

# Step 8 — Launch the dashboard
streamlit run app.py
```

> **Note:** Steps 4–5 are computationally intensive. SVR on the full dataset is particularly slow. Estimated total runtime: 2–4 hours depending on hardware. All pre-computed results are already included in the repository so you can skip directly to Step 8.

---

## The Demo Dashboard

The Streamlit app (`app.py`) provides:

- **Metric cards** — IC Mean, Sharpe, Cumulative Return, Max Drawdown on the 2024 test set
- **Interactive date selector** — select any week in the test period to view the model's stock rankings
- **Rankings table** — top 10 long positions (green) and bottom 10 short positions (red) with predicted scores and realised returns
- **Equity curves** — Ensemble vs SPY buy-and-hold vs Naive Momentum baseline
- **IC over time** — signal stability across the validation period

---

## Models Implemented

| Model | IC (test) | Sharpe (val) | Status |
|---|---|---|---|
| ElasticNet | — | 1.43 | In ensemble |
| Lasso | — | 1.58 | In ensemble |
| Ridge | — | 1.22 | In ensemble |
| Gradient Boosting | — | 1.02 | Tested, excluded |
| Random Forest | — | neg | Excluded |
| Decision Tree | — | neg | Excluded |
| SVR | — | ~0.00 | Excluded |
| **Ensemble (Ridge 40% + ElasticNet 60%)** | **0.024** | **0.98** | **Final model** |

---

## Key Design Decisions

- **Temporal validation only** — no random splits. TimeSeriesSplit for hyperparameter tuning, strict chronological train/validation/test split (80/10/10).
- **Cross-sectional z-scoring** — all 14 features normalized across stocks per date, not globally.
- **Non-overlapping 5-day windows** — portfolio rebalanced every 5 trading days to avoid overlapping return windows for the 5-day target.
- **10 bps transaction costs** — applied per period based on actual portfolio turnover.
- **IC-weighted ensemble** — weights estimated on validation period (2023), applied to test period (2024).

---

## Requirements

See `requirements.txt` for the full list. Key dependencies:

```
pandas
numpy
scikit-learn
yfinance
scipy
streamlit
matplotlib
seaborn
```

Python 3.10 or higher recommended.

---

## Data Source

Stock price data is downloaded from **Yahoo Finance** via the `yfinance` library. The universe is the current S&P 100 constituents. Note: this introduces survivorship bias since historical delisted or removed stocks are not included. This is acknowledged as a limitation in the project report.

---

## GitHub Repository

[https://github.com/Dib-Ali/cross-sectional-equity-ml](https://github.com/Dib-Ali/cross-sectional-equity-ml)