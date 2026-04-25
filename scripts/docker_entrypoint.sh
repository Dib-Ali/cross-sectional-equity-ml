#!/usr/bin/env bash
set -euo pipefail

python -m scripts.run_download
python -m scripts.run_clean_data
python -m scripts.run_make_dataset

echo "We used multiple models for testing and found two models that would work great in the ensemble"

python -m scripts.run_ensemble_final

exec streamlit run app.py --server.port=8501 --server.address=0.0.0.0
