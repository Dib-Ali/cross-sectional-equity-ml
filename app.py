"""
Equity ML Strategy — Demo Dashboard
Streamlit app for the cross-sectional equity factor model project.

Run with:
    streamlit run app.py
"""

import os
from pathlib import Path

import numpy as np
import pandas as pd
import streamlit as st

# ---------------------------------------------------------------------------
# Page config — must be first Streamlit call
# ---------------------------------------------------------------------------
st.set_page_config(
    page_title="Equity ML Strategy — Demo Dashboard",
    layout="wide",
    initial_sidebar_state="collapsed",
)

# ---------------------------------------------------------------------------
# Custom CSS: white background, tighter metric cards, clean table
# ---------------------------------------------------------------------------
st.markdown(
    """
    <style>
        /* Page background */
        .stApp { background-color: #ffffff; }

        /* Top metric cards */
        [data-testid="metric-container"] {
            background: #f8f9fa;
            border: 1px solid #e9ecef;
            border-radius: 8px;
            padding: 16px 20px;
        }
        [data-testid="stMetricLabel"] { font-size: 0.82rem; color: #6c757d; }
        [data-testid="stMetricValue"] { font-size: 1.6rem; font-weight: 700; }

        /* Section dividers */
        hr { border: none; border-top: 1px solid #e9ecef; margin: 1.2rem 0; }

        /* Footer */
        .footer {
            text-align: center;
            color: #adb5bd;
            font-size: 0.78rem;
            margin-top: 3rem;
            padding-top: 1rem;
            border-top: 1px solid #e9ecef;
        }

        /* Table rank badges */
        .rank-long  { color: #1a9e78; font-weight: 700; }
        .rank-short { color: #e03434; font-weight: 700; }

        /* Remove default Streamlit padding on top */
        .block-container { padding-top: 1.5rem; }
    </style>
    """,
    unsafe_allow_html=True,
)

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
BASE = Path(__file__).parent
TABLES = BASE / "reports" / "tables"
PLOTS  = BASE / "plots"

# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------
@st.cache_data
def load_metrics() -> pd.DataFrame:
    return pd.read_csv(TABLES / "ensemble_final_metrics.csv")


@st.cache_data
def load_scored() -> pd.DataFrame:
    df = pd.read_csv(TABLES / "ensemble_scored_test_target_5d.csv")
    df["date"] = pd.to_datetime(df["date"]).dt.normalize()
    return df


@st.cache_data
def load_aligned() -> pd.DataFrame:
    df = pd.read_csv(TABLES / "spy_benchmark_aligned_returns.csv")
    df["date"] = pd.to_datetime(df["date"])
    return df


metrics_df = load_metrics()
scored_df  = load_scored()
aligned_df = load_aligned()

# Pull test-period row for metric cards
test_row = metrics_df[metrics_df["stage"] == "test"].iloc[0]

# ---------------------------------------------------------------------------
# Header
# ---------------------------------------------------------------------------
st.markdown("## Equity ML Strategy — Demo Dashboard")
st.markdown(
    "<span style='color:#6c757d;font-size:0.9rem;'>"
    "Cross-sectional factor model · Ensemble (Ridge 40% + ElasticNet 60%) · "
    "Test period: Jan – Dec 2024 · Universe: S&P 100"
    "</span>",
    unsafe_allow_html=True,
)

st.markdown("<hr/>", unsafe_allow_html=True)

# ---------------------------------------------------------------------------
# Section 1 — Metric Cards
# ---------------------------------------------------------------------------
st.markdown("#### Strategy Performance — Test Period (2024, out-of-sample)")

c1, c2, c3, c4 = st.columns(4)

ic_mean   = float(test_row["ic_mean"])
icir      = float(test_row["icir"])
cum_ret   = float(test_row["portfolio_cumulative_return"])
max_dd    = float(test_row["portfolio_max_drawdown"])
sharpe    = float(test_row["portfolio_sharpe"])

c1.metric(
    label="IC Mean",
    value=f"{ic_mean:.4f}",
    delta="vs 0 (signal)",
    delta_color="normal",
)
c2.metric(
    label="Sharpe Ratio (annualised)",
    value=f"{sharpe:.3f}",
    delta=f"ICIR: {icir:.3f}",
    delta_color="normal",
)
c3.metric(
    label="Cumulative Return",
    value=f"{cum_ret * 100:.1f}%",
    delta="long-short, net of costs",
    delta_color="off",
)
c4.metric(
    label="Max Drawdown",
    value=f"{max_dd * 100:.1f}%",
    delta="peak-to-trough",
    delta_color="inverse",
)

st.markdown("<hr/>", unsafe_allow_html=True)

