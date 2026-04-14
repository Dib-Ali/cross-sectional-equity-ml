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

# # ---------------------------------------------------------------------------
# # Section 2 — Date selector + rankings table
# # ---------------------------------------------------------------------------
# st.markdown("#### Weekly Stock Rankings — Predicted vs Realised Return")
# st.markdown(
#     "<span style='color:#6c757d;font-size:0.85rem;'>"
#     "Select a rebalancing date to view the model's cross-sectional ranking. "
#     "🟢 Top 10 = long positions · 🔴 Bottom 10 = short positions."
#     "</span>",
#     unsafe_allow_html=True,
# )

# # Use every-5th date to match backtest rebalancing periods
# all_dates   = sorted(scored_df["date"].unique())
# period_dates = all_dates[::5]

# # Format for display
# date_labels = {d: d.strftime("%Y-%m-%d") for d in period_dates}

# selected_date = st.selectbox(
#     "Select rebalancing date",
#     options=period_dates,
#     format_func=lambda d: date_labels[d],
#     index=len(period_dates) // 2,   # default to middle of test period
# )

# # Filter and rank
# day_df = (
#     scored_df[scored_df["date"] == selected_date]
#     .copy()
#     .sort_values("prediction", ascending=False)
#     .reset_index(drop=True)
# )
# day_df["Rank"] = day_df.index + 1
# day_df = day_df.rename(columns={
#     "ticker":     "Ticker",
#     "prediction": "Predicted Score",
#     "target_5d":  "Realised Return",
# })
# display_df = day_df[["Rank", "Ticker", "Predicted Score", "Realised Return"]].copy()

# n_stocks = len(display_df)

# # ------ Styling ------
# LONG_BG  = "#eaf7f2"   # light green
# SHORT_BG = "#fdecea"   # light red
# ROW_ALT  = "#fafafa"

# def _style_table(df: pd.DataFrame) -> "pd.io.formats.style.Styler":
#     n = len(df)
#     top10_idx    = list(range(min(10, n)))
#     bottom10_idx = list(range(max(0, n - 10), n))

#     def highlight(row):
#         i = row.name
#         if i in top10_idx:
#             return ["background-color: " + LONG_BG] * len(row)
#         elif i in bottom10_idx:
#             return ["background-color: " + SHORT_BG] * len(row)
#         elif i % 2 == 0:
#             return ["background-color: " + ROW_ALT] * len(row)
#         else:
#             return [""] * len(row)

#     styled = (
#         df.style
#         .apply(highlight, axis=1)
#         .format({
#             "Predicted Score":  "{:.5f}",
#             "Realised Return":  "{:+.4f}",
#         })
#         .set_properties(**{"font-size": "0.88rem"})
#         .set_table_styles([
#             {"selector": "thead th",
#              "props": [("background-color", "#f1f3f5"),
#                        ("font-weight", "600"),
#                        ("font-size", "0.82rem"),
#                        ("text-align", "left"),
#                        ("border-bottom", "2px solid #dee2e6")]},
#             {"selector": "tbody tr:hover",
#              "props": [("filter", "brightness(0.97)")]},
#         ])
#         .hide(axis="index")
#     )
#     return styled


# # Summary stats row above table
# top10_ret  = display_df.head(10)["Realised Return"].mean()
# bot10_ret  = display_df.tail(10)["Realised Return"].mean()
# ls_spread  = top10_ret - bot10_ret

# sm1, sm2, sm3, sm4 = st.columns(4)
# sm1.metric("Date",              selected_date.strftime("%d %b %Y"))
# sm2.metric("Top-10 Avg Return", f"{top10_ret:+.4f}",
#            delta="long leg", delta_color="off")
# sm3.metric("Bottom-10 Avg Return", f"{bot10_ret:+.4f}",
#            delta="short leg", delta_color="off")
# sm4.metric("L/S Spread",        f"{ls_spread:+.4f}",
#            delta="alpha proxy", delta_color="normal" if ls_spread > 0 else "inverse")

# st.dataframe(
#     _style_table(display_df),
#     use_container_width=True,
#     height=min(42 * n_stocks + 38, 680),
# )

# st.markdown("<hr/>", unsafe_allow_html=True)

# # ---------------------------------------------------------------------------
# # Section 3 — Charts (static PNGs side by side)
# # ---------------------------------------------------------------------------
# st.markdown("#### Strategy Visualisations")

# img_col1, img_col2 = st.columns(2)

# with img_col1:
#     equity_path = PLOTS / "plot3_equity_curves.png"
#     if equity_path.exists():
#         st.image(str(equity_path), use_container_width=True)
#         st.caption("Equity Curves — Ensemble vs SPY vs Naive Momentum (Test 2024)")
#     else:
#         st.warning(f"Plot not found: {equity_path}")

# with img_col2:
#     ic_path = PLOTS / "plot2_ic_over_time.png"
#     if ic_path.exists():
#         st.image(str(ic_path), use_container_width=True)
#         st.caption("IC Over Time — Validation Period (2023–2024), 10-period rolling mean")
#     else:
#         st.warning(f"Plot not found: {ic_path}")

# st.markdown("<hr/>", unsafe_allow_html=True)

# # ---------------------------------------------------------------------------
# # Footer
# # ---------------------------------------------------------------------------
# st.markdown(
#     "<div class='footer'>"
#     "Cross-Sectional Equity ML · Machine Learning Course Project · "
#     "Ensemble: Ridge 40% + ElasticNet 60% · Universe: S&P 100 · "
#     "Data: 2015–2024 via yfinance"
#     "</div>",
#     unsafe_allow_html=True,
# )

# # ---------------------------------------------------------------------------
# # Console output: best demo dates (printed once at startup)
# # ---------------------------------------------------------------------------
# @st.cache_data
# def _find_best_demo_dates() -> pd.DataFrame:
#     res = []
#     for dt in period_dates:
#         day = scored_df[scored_df["date"] == dt].sort_values("prediction", ascending=False)
#         if len(day) < 10:
#             continue
#         res.append({
#             "date":             dt.strftime("%Y-%m-%d"),
#             "top10_avg_return": day.head(10)["Realised Return"].mean()
#             if "Realised Return" in day.columns
#             else day.head(10)["target_5d"].mean(),
#         })
#     return pd.DataFrame(res).sort_values("top10_avg_return", ascending=False)


# _demo = _find_best_demo_dates()
# print("\n" + "=" * 60)
# print("BEST DEMO DATES (top-10 long leg highest avg realised return)")
# print("=" * 60)
# for i, row in _demo.head(3).iterrows():
#     print(f"  {row['date']}  →  top-10 avg realised return = {row['top10_avg_return']:+.4f}")
# print("=" * 60 + "\n")
