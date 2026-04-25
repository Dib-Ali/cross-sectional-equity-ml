#!/usr/bin/env bash
set -euo pipefail

if [ -s "data/raw/prices_raw.csv" ]; then
	echo "Using bundled raw data at data/raw/prices_raw.csv (skipping download)."
else
	echo "Raw data not found in image; downloading from yfinance..."
	python -m scripts.run_download
fi

python -m scripts.run_clean_data
python -m scripts.run_make_dataset

echo "We used multiple models for testing and found two models that would work great in the ensemble"

python -m scripts.run_ensemble_final

exec streamlit run app.py --server.port=8501 --server.address=0.0.0.0
