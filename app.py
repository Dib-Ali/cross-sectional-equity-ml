"""
Equity ML Strategy — Demo Dashboard
Streamlit app for the cross-sectional equity factor model project.

Run with:
    streamlit run app.py
"""

from pathlib import Path

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
        /* ── Force light background everywhere ─────────────────────── */
        .stApp,
        [data-testid="stAppViewContainer"],
        [data-testid="stHeader"],
        [data-testid="stToolbar"],
        section[data-testid="stSidebar"],
        .main .block-container          { background-color: #ffffff !important; color: #212529 !important; }

        /* ── Metric cards ───────────────────────────────────────────── */
        [data-testid="metric-container"] {
            background: #f8f9fa !important;
            border: 1px solid #e9ecef !important;
            border-radius: 10px;
            padding: 18px 22px;
            box-shadow: 0 1px 2px rgba(0,0,0,0.04);
        }
        [data-testid="stMetricLabel"]  { font-size: 0.82rem !important; color: #6c757d !important; }
        [data-testid="stMetricValue"]  { font-size: 1.7rem  !important; font-weight: 700 !important; color: #212529 !important; }
        [data-testid="stMetricDelta"]  { font-size: 0.78rem !important; }

        /* ── Headings ────────────────────────────────────────────────── */
        h1, h2, h3, h4, h5 { color: #212529 !important; }
        .subtitle { color: #6c757d !important; font-size: 0.9rem; }

        /* ── Selectbox / dropdown ────────────────────────────────────── */
        [data-testid="stSelectbox"] > div > div {
            background-color: #ffffff !important;
            color: #212529 !important;
            border: 1px solid #ced4da !important;
        }

        /* ── Dataframe wrapper ───────────────────────────────────────── */
        [data-testid="stDataFrame"],
        [data-testid="stDataFrame"] iframe,
        .stDataFrame                    { background-color: #ffffff !important; }

        /* ── Dividers & layout ───────────────────────────────────────── */
        hr { border: none; border-top: 1px solid #e9ecef; margin: 1.2rem 0; }
        .block-container { padding-top: 1.5rem; max-width: 1400px; }

        /* ── Footer ──────────────────────────────────────────────────── */
        .footer {
            text-align: center;
            color: #adb5bd !important;
            font-size: 0.78rem;
            margin-top: 3rem;
            padding-top: 1rem;
            border-top: 1px solid #e9ecef;
        }
    </style>
    """,
    unsafe_allow_html=True,
)

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
BASE = Path(__file__).parent
TABLES = BASE / "reports" / "tables"
PLOTS = BASE / "reports" / "plots"

# ---------------------------------------------------------------------------
# Data loading — cached at startup
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
    df["date"] = pd.to_datetime(df["date"]).dt.normalize()
    return df


metrics_df = load_metrics()
scored_df = load_scored()
aligned_df = load_aligned()

# Pull test-period row for metric cards
test_row = metrics_df[metrics_df["stage"] == "test"].iloc[0]

# Rebalance dates = 50 weekly rebalance points actually used in the backtest
rebalance_dates = sorted(aligned_df["date"].unique())

# ---------------------------------------------------------------------------
# Header
# ---------------------------------------------------------------------------
st.markdown("## Equity ML Strategy — Demo Dashboard")
st.markdown(
    "<div class='subtitle'>"
    "Cross-sectional factor model · Ensemble (Ridge 40% + ElasticNet 60%) · "
    "Test period: Jan – Dec 2024 · Universe: S&P 100"
    "</div>",
    unsafe_allow_html=True,
)

st.markdown("<hr/>", unsafe_allow_html=True)

# ---------------------------------------------------------------------------
# Section 1 — Metric Cards (always visible)
# ---------------------------------------------------------------------------
st.markdown("#### Strategy Performance — Test Period (2024, out-of-sample)")

ic_mean = float(test_row["ic_mean"])
icir = float(test_row["icir"])
sharpe = float(test_row["portfolio_sharpe"])
cum_ret = float(test_row["portfolio_cumulative_return"])
max_dd = float(test_row["portfolio_max_drawdown"])

c1, c2, c3, c4 = st.columns(4)

c1.metric(
    label="IC Mean",
    value=f"{ic_mean:.4f}",
    delta="cross-sectional signal",
    delta_color="off",
)
c2.metric(
    label="Sharpe Ratio (annualised)",
    value=f"{sharpe:.3f}",
    delta=f"ICIR: {icir:.3f}",
    delta_color="off",
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

# ---------------------------------------------------------------------------
# Section 2 — Date selector + rankings table
# ---------------------------------------------------------------------------
st.markdown("#### Weekly Stock Rankings — Predicted vs Realised Return")
st.markdown(
    "<div class='subtitle'>"
    "Select a rebalancing date to view the model's cross-sectional ranking. "
    "Top 10 = long positions (green) · Bottom 10 = short positions (red)."
    "</div>",
    unsafe_allow_html=True,
)
st.write("")

date_labels = {d: d.strftime("%Y-%m-%d  (%a)") for d in rebalance_dates}

selected_date = st.selectbox(
    "Rebalancing date",
    options=rebalance_dates,
    format_func=lambda d: date_labels[d],
    index=len(rebalance_dates) // 2,
)

# Filter and rank
day_df = (
    scored_df[scored_df["date"] == selected_date]
    .copy()
    .sort_values("prediction", ascending=False)
    .reset_index(drop=True)
)
day_df["Rank"] = day_df.index + 1
day_df = day_df.rename(columns={
    "ticker":     "Ticker",
    "prediction": "Predicted Score",
    "target_5d":  "Actual Realised Return",
})
display_df = day_df[["Rank", "Ticker", "Predicted Score", "Actual Realised Return"]].copy()
n_stocks = len(display_df)

# ------ Summary metrics for this week ------
top10_ret = display_df.head(10)["Actual Realised Return"].mean()
bot10_ret = display_df.tail(10)["Actual Realised Return"].mean()
ls_spread = top10_ret - bot10_ret

# Lookup the realised strategy return for this rebalance date (if present)
aligned_row = aligned_df[aligned_df["date"] == selected_date]
strat_ret = float(aligned_row["strategy_return"].iloc[0]) if len(aligned_row) else float("nan")
spy_ret   = float(aligned_row["spy_return_5d"].iloc[0])   if len(aligned_row) else float("nan")

sm1, sm2, sm3, sm4 = st.columns(4)
sm1.metric("Date",                selected_date.strftime("%d %b %Y"))
sm2.metric("Top-10 Avg Return",   f"{top10_ret:+.4f}",
           delta="long leg",  delta_color="off")
sm3.metric("Bottom-10 Avg Return", f"{bot10_ret:+.4f}",
           delta="short leg", delta_color="off")
sm4.metric("Long/Short Spread",   f"{ls_spread:+.4f}",
           delta=("alpha captured" if ls_spread > 0 else "alpha missed"),
           delta_color=("normal" if ls_spread > 0 else "inverse"))

st.write("")

# ------ Table styling ------
LONG_BG  = "#e6f7ef"   # light green
SHORT_BG = "#fdeceb"   # light red
ROW_EVEN = "#fafbfc"   # light grey stripe
ROW_ODD  = "#ffffff"   # pure white  ← explicit, never inherit dark-mode bg


def _style_table(df: pd.DataFrame):
    n = len(df)
    top10_idx    = set(range(min(10, n)))
    bottom10_idx = set(range(max(0, n - 10), n))

    def highlight(row):
        i = row.name
        if i in top10_idx:
            return [f"background-color: {LONG_BG}; color: #0b6b4a;"] * len(row)
        if i in bottom10_idx:
            return [f"background-color: {SHORT_BG}; color: #a11f1f;"] * len(row)
        # Every neutral row gets an explicit light bg — never inherit dark theme
        bg = ROW_EVEN if i % 2 == 0 else ROW_ODD
        return [f"background-color: {bg}; color: #212529;"] * len(row)

    styled = (
        df.style
        .apply(highlight, axis=1)
        .format({
            "Predicted Score":        "{:.5f}",
            "Actual Realised Return": "{:+.4f}",
        })
        .set_properties(**{"font-size": "0.88rem", "padding": "6px 10px"})
        .set_table_styles([
            {"selector": "thead th",
             "props": [("background-color", "#f1f3f5"),
                       ("font-weight", "600"),
                       ("font-size", "0.82rem"),
                       ("text-align", "left"),
                       ("padding", "8px 10px"),
                       ("border-bottom", "2px solid #dee2e6")]},
            {"selector": "tbody tr:hover",
             "props": [("filter", "brightness(0.97)")]},
        ])
        .hide(axis="index")
    )
    return styled


st.dataframe(
    _style_table(display_df),
    use_container_width=True,
    hide_index=True,
    height=min(40 * n_stocks + 42, 700),
)

st.markdown("<hr/>", unsafe_allow_html=True)

# ---------------------------------------------------------------------------
# Section 3 — Charts (static PNGs side by side)
# ---------------------------------------------------------------------------
st.markdown("#### Strategy Visualisations")

img_col1, img_col2 = st.columns(2)

with img_col1:
    equity_path = PLOTS / "plot3_equity_curves.png"
    if equity_path.exists():
        st.image(str(equity_path), use_container_width=True)
        st.caption("Equity Curves — Ensemble vs SPY vs Naive")
    else:
        st.warning(f"Plot not found: {equity_path}")

with img_col2:
    ic_path = PLOTS / "plot2_ic_over_time.png"
    if ic_path.exists():
        st.image(str(ic_path), use_container_width=True)
        st.caption("IC Over Time — Validation Period")
    else:
        st.warning(f"Plot not found: {ic_path}")

st.markdown("<hr/>", unsafe_allow_html=True)

# ---------------------------------------------------------------------------
# Footer
# ---------------------------------------------------------------------------
st.markdown(
    "<div class='footer'>"
    "Cross-Sectional Equity ML &nbsp;·&nbsp; Machine Learning Course Project &nbsp;·&nbsp; "
    "Ensemble: Ridge 40% + ElasticNet 60% &nbsp;·&nbsp; Universe: S&P 100 &nbsp;·&nbsp; "
    "Data: 2015–2024 via yfinance"
    "</div>",
    unsafe_allow_html=True,
)

# ---------------------------------------------------------------------------
# Console output: best demo dates (printed once at startup)
# ---------------------------------------------------------------------------
@st.cache_data
def _find_best_demo_dates() -> pd.DataFrame:
    rows = []
    for dt in rebalance_dates:
        day = scored_df[scored_df["date"] == dt].sort_values("prediction", ascending=False)
        if len(day) < 20:
            continue
        top10_avg = day.head(10)["target_5d"].mean()
        bot10_avg = day.tail(10)["target_5d"].mean()
        rows.append({
            "date":             dt.strftime("%Y-%m-%d"),
            "top10_avg_return": top10_avg,
            "bot10_avg_return": bot10_avg,
            "ls_spread":        top10_avg - bot10_avg,
        })
    return pd.DataFrame(rows).sort_values("top10_avg_return", ascending=False)


_demo = _find_best_demo_dates()
print("\n" + "=" * 72)
print(" BEST DEMO DATES — top-10 long leg with highest avg realised 5-day return")
print("=" * 72)
for _, row in _demo.head(3).iterrows():
    print(
        f"  {row['date']}   top10={row['top10_avg_return']:+.4f}"
        f"   bot10={row['bot10_avg_return']:+.4f}"
        f"   L/S spread={row['ls_spread']:+.4f}"
    )
print("=" * 72 + "\n")
