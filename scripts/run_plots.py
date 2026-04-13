"""
Publication-quality plot generation for cross-sectional equity ML project.
Produces exactly 6 plots saved to reports/plots/.
"""
 
from __future__ import annotations
 
import os
import sys
import warnings
warnings.filterwarnings("ignore")
 
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
from scipy.stats import spearmanr
from sklearn.ensemble import GradientBoostingRegressor
 
# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
TABLES   = os.path.join(BASE_DIR, "reports", "tables")
PLOTS = os.path.join(BASE_DIR, "reports", "plots")
os.makedirs(PLOTS, exist_ok=True)
 
# ---------------------------------------------------------------------------
# Global style
# ---------------------------------------------------------------------------
plt.rcParams.update({
    "figure.facecolor":    "white",
    "axes.facecolor":      "white",
    "axes.edgecolor":      "#CCCCCC",
    "axes.grid":           True,
    "grid.color":          "#E8E8E8",
    "grid.linewidth":      0.7,
    "axes.spines.top":     False,
    "axes.spines.right":   False,
    "axes.spines.left":    False,
    "font.family":         "DejaVu Sans",
    "font.size":           11,
    "axes.titlesize":      13,
    "axes.titleweight":    "bold",
    "axes.labelsize":      11,
    "xtick.labelsize":     10,
    "ytick.labelsize":     10,
    "legend.fontsize":     9,
    "legend.framealpha":   0.9,
    "legend.edgecolor":    "#CCCCCC",
    "figure.dpi":          150,
    "savefig.dpi":         150,
    "savefig.bbox":        "tight",
    "savefig.facecolor":   "white",
})
 
DPI = 150
 
# ---------------------------------------------------------------------------
# Colour palette
# ---------------------------------------------------------------------------
C_ENSEMBLE  = "#1A9E78"   # teal/green
C_SPY       = "#222222"   # near-black for dashed SPY
C_NAIVE     = "#9E9E9E"   # medium gray
 
# Blue family – linear models
C_RIDGE       = "#1F5FA6"
C_LASSO       = "#4A90D9"
C_ELASTICNET  = "#6FBAFF"
 
# Orange family – tree models
C_RF  = "#E07B00"
C_GB  = "#F4A833"
C_DT  = "#F7D080"
 
# Other
C_SVR = "#8E44AD"   # purple
 
 
# ---------------------------------------------------------------------------
# Helper: compute per-date Spearman IC
# ---------------------------------------------------------------------------
def compute_ic_series(df: pd.DataFrame,
                      date_col: str = "date",
                      pred_col: str = "prediction",
                      target_col: str = "target_5d") -> pd.Series:
    df = df.copy()
    df[date_col] = pd.to_datetime(df[date_col])
    records = []
    for dt, grp in df.groupby(date_col):
        sub = grp[[pred_col, target_col]].dropna()
        if len(sub) < 5:
            continue
        ic, _ = spearmanr(sub[pred_col], sub[target_col])
        records.append((dt, ic))
    s = pd.Series({r[0]: r[1] for r in records}, name="IC").sort_index()
    return s
 
 
# ---------------------------------------------------------------------------
# Helper: compute long-short portfolio equity from a scored dataframe
# ---------------------------------------------------------------------------
def compute_ls_equity(df: pd.DataFrame,
                      date_col: str = "date",
                      pred_col: str = "prediction",
                      ret_col: str = "target_5d",
                      top_n: int = 10,
                      bottom_n: int = 10,
                      step: int = 5,
                      cost_bps: float = 10.0) -> pd.Series:
    """Returns a cumulative equity curve indexed by date."""
    df = df.copy()
    df[date_col] = pd.to_datetime(df[date_col])
    all_dates = sorted(df[date_col].unique())
    # Sample every `step` dates to match 5-day rebalancing
    selected = all_dates[::step]
    work = df[df[date_col].isin(selected)].copy()
    cost_rate = cost_bps / 10_000.0
 
    period_returns = []
    period_dates   = []
    prev_weights: dict = {}
 
    for dt, grp in work.groupby(date_col, sort=True):
        sub = grp[[pred_col, ret_col, "ticker"]].dropna()
        if len(sub) < top_n + bottom_n:
            continue
        ranked = sub.sort_values(pred_col, ascending=False)
        long_tickers  = list(ranked.head(top_n)["ticker"])
        short_tickers = list(ranked.tail(bottom_n)["ticker"])
 
        weights: dict = {}
        for t in long_tickers:
            weights[str(t)] = 1.0 / top_n
        for t in short_tickers:
            weights[str(t)] = weights.get(str(t), 0.0) - 1.0 / bottom_n
 
        gross_ret = sum(
            weights.get(str(row.ticker), 0.0) * row[ret_col]
            for _, row in sub.iterrows()
        )
        turnover = (
            sum(abs(w) for w in weights.values())
            if not prev_weights
            else sum(abs(weights.get(t, 0.0) - prev_weights.get(t, 0.0))
                     for t in set(prev_weights) | set(weights))
        )
        period_returns.append(gross_ret - cost_rate * turnover)
        period_dates.append(dt)
        prev_weights = weights
 
    r = pd.Series(period_returns, index=period_dates)
    equity = (1 + r).cumprod()
    equity.iloc[0] = 1 + period_returns[0]   # ensure starts near 1
    return equity
 
 
# ---------------------------------------------------------------------------
# Helper: drawdown from equity curve
# ---------------------------------------------------------------------------
def drawdown_series(equity: pd.Series) -> pd.Series:
    peak = equity.cummax()
    return equity / peak - 1.0
 
 
# ===========================================================================
# PLOT 1 — Model Comparison Bar Chart
# ===========================================================================
def plot1_model_comparison():
    reg = pd.read_csv(os.path.join(TABLES, "regularized_validation_metrics.csv"))
    rf  = pd.read_csv(os.path.join(TABLES, "random_forest_validation_metrics.csv"))
    gb  = pd.read_csv(os.path.join(TABLES, "boosting_validation_metrics.csv"))
    dt  = pd.read_csv(os.path.join(TABLES, "decision_tree_validation_metrics.csv"))
    svr = pd.read_csv(os.path.join(TABLES, "svr_validation_metrics.csv"))
    ens = pd.read_csv(os.path.join(TABLES, "ensemble_final_metrics.csv"))
 
    # Best per model for target_5d (by IC mean)
    def best(df, model_name, target="target_5d"):
        sub = df[df["target"] == target].sort_values("ic_mean", ascending=False)
        row = sub.iloc[0] if not sub.empty else None
        return row
 
    ridge_row = reg[(reg["target"]=="target_5d") & (reg["model"]=="ridge")].sort_values("ic_mean", ascending=False).iloc[0]
    lasso_row = reg[(reg["target"]=="target_5d") & (reg["model"]=="lasso")].sort_values("ic_mean", ascending=False).iloc[0]
    enet_row  = reg[(reg["target"]=="target_5d") & (reg["model"]=="elasticnet")].sort_values("ic_mean", ascending=False).iloc[0]
    rf_row    = rf[rf["target"]=="target_5d"].sort_values("ic_mean", ascending=False).iloc[0]
    gb_row    = gb[gb["target"]=="target_5d"].sort_values("ic_mean", ascending=False).iloc[0]
    dt_row    = dt[dt["target"]=="target_5d"].sort_values("ic_mean", ascending=False).iloc[0]
    svr_row   = svr[svr["target"]=="target_5d"].sort_values("ic_mean", ascending=False).iloc[0]
    ens_row   = ens[ens["stage"]=="validation"].iloc[0]
 
    models = ["Ridge", "Lasso", "ElasticNet", "RF", "GB", "Dec. Tree", "SVR", "Ensemble"]
    ic_vals    = [ridge_row["ic_mean"], lasso_row["ic_mean"], enet_row["ic_mean"],
                  rf_row["ic_mean"],    gb_row["ic_mean"],    dt_row["ic_mean"],
                  svr_row["ic_mean"],   ens_row["ic_mean"]]
    icir_vals  = [ridge_row["icir"],    lasso_row["icir"],    enet_row["icir"],
                  rf_row["icir"],       gb_row["icir"],       dt_row["icir"],
                  svr_row["icir"],      ens_row["icir"]]
    sharpe_vals= [ridge_row["portfolio_sharpe"], lasso_row["portfolio_sharpe"],
                  enet_row["portfolio_sharpe"],  rf_row["portfolio_sharpe"],
                  gb_row["portfolio_sharpe"],    dt_row["portfolio_sharpe"],
                  svr_row["portfolio_sharpe"],   ens_row["portfolio_sharpe"]]
 
    colours = [C_RIDGE, C_LASSO, C_ELASTICNET, C_RF, C_GB, C_DT, C_SVR, C_ENSEMBLE]
 
    n_models  = len(models)
    n_metrics = 3
    x         = np.arange(n_metrics)
    total_w   = 0.75
    bar_w     = total_w / n_models
 
    fig, ax = plt.subplots(figsize=(13, 6))
 
    for i, (model, colour) in enumerate(zip(models, colours)):
        vals = [ic_vals[i], icir_vals[i], sharpe_vals[i]]
        offset = (i - n_models / 2 + 0.5) * bar_w
        bars = ax.bar(x + offset, vals, width=bar_w, color=colour,
                      label=model, edgecolor="white", linewidth=0.5)
 
    ax.set_xticks(x)
    ax.set_xticklabels(["IC Mean", "ICIR", "Sharpe Ratio"], fontsize=11)
    ax.set_ylabel("Metric Value")
    ax.set_title("Model Comparison — Validation Performance (target_5d)")
    ax.axhline(0, color="#999999", linewidth=0.8, linestyle="--")
    ax.legend(ncol=4, loc="upper right", fontsize=9)
    ax.yaxis.grid(True, color="#E8E8E8", linewidth=0.7)
    ax.set_axisbelow(True)
 
    out = os.path.join(PLOTS, "plot1_model_comparison.png")
    fig.savefig(out, dpi=DPI)
    plt.close(fig)
    print(f"Saved: {out}")
 
 
# ===========================================================================
# PLOT 2 — IC Over Time (Rolling)
# ===========================================================================
def plot2_ic_over_time():
    ridge_df = pd.read_csv(os.path.join(TABLES, "regularized_scored_target_5d_ridge.csv"))
    lasso_df = pd.read_csv(os.path.join(TABLES, "regularized_scored_target_5d_lasso.csv"))
    enet_df  = pd.read_csv(os.path.join(TABLES, "regularized_scored_target_5d_elasticnet.csv"))
    ens_df   = pd.read_csv(os.path.join(TABLES, "ensemble_scored_validation_target_5d.csv"))
 
    ic_ridge = compute_ic_series(ridge_df)
    ic_lasso = compute_ic_series(lasso_df)
    ic_enet  = compute_ic_series(enet_df)
    ic_ens   = compute_ic_series(ens_df)
 
    roll = 10
 
    fig, ax = plt.subplots(figsize=(13, 6))
 
    def _plot_model(ic, colour, label, zorder_base):
        ax.plot(ic.index, ic.values, color=colour, alpha=0.18,
                linewidth=0.7, zorder=zorder_base)
        ax.plot(ic.index, ic.rolling(roll, min_periods=3).mean(),
                color=colour, linewidth=1.8, label=label, zorder=zorder_base + 1)
 
    _plot_model(ic_ridge, C_RIDGE,      "Ridge",      2)
    _plot_model(ic_lasso, C_LASSO,      "Lasso",      4)
    _plot_model(ic_enet,  C_ELASTICNET, "ElasticNet", 6)
    _plot_model(ic_ens,   C_ENSEMBLE,   "Ensemble",   8)
 
    ax.axhline(0, color="#888888", linewidth=0.9, linestyle="--", zorder=1)
    ax.set_title("Information Coefficient Over Time — Validation Period (target_5d)")
    ax.set_xlabel("Date")
    ax.set_ylabel("Spearman IC")
    ax.legend(loc="upper right")
 
    # Shade 2024 to show out-of-sample portion
    ax.axvspan(pd.Timestamp("2024-01-01"), ic_ridge.index[-1],
               alpha=0.06, color="#888888", label=None)
    ax.text(pd.Timestamp("2024-01-15"), ax.get_ylim()[1] * 0.9,
            "2024 (signal decay)", fontsize=8, color="#666666")
 
    out = os.path.join(PLOTS, "plot2_ic_over_time.png")
    fig.savefig(out, dpi=DPI)
    plt.close(fig)
    print(f"Saved: {out}")
 
 
# ===========================================================================
# PLOT 3 — Equity Curves: Ensemble vs SPY vs Naive
# ===========================================================================
def plot3_equity_curves():
    aligned = pd.read_csv(os.path.join(TABLES, "spy_benchmark_aligned_returns.csv"))
    aligned["date"] = pd.to_datetime(aligned["date"])
    aligned = aligned.sort_values("date").reset_index(drop=True)
 
    # Ensemble and SPY from aligned file
    ens_equity = aligned.set_index("date")["strategy_equity"]
    spy_equity  = aligned.set_index("date")["spy_equity"]
 
    # Naive momentum equity curve (long-short from scored test data)
    naive_df = pd.read_csv(os.path.join(TABLES, "naive_momentum_scored_test_target_5d.csv"))
    naive_equity = compute_ls_equity(
        naive_df,
        date_col="date", pred_col="prediction", ret_col="target_5d",
        top_n=10, bottom_n=10, step=5, cost_bps=10.0
    )
 
    fig, ax = plt.subplots(figsize=(12, 6))
 
    ax.plot(ens_equity.index, ens_equity.values,
            color=C_ENSEMBLE, linewidth=2.2, label="Ensemble")
    ax.plot(spy_equity.index, spy_equity.values,
            color=C_SPY, linewidth=1.8, linestyle="--", label="SPY (buy & hold)")
    ax.plot(naive_equity.index, naive_equity.values,
            color=C_NAIVE, linewidth=1.6, linestyle="-", label="Naive Momentum (5d)")
 
    ax.axhline(1.0, color="#CCCCCC", linewidth=0.8, linestyle=":")
    ax.set_title("Equity Curves — Test Period 2024 (Ensemble vs SPY vs Naive)")
    ax.set_xlabel("Date")
    ax.set_ylabel("Cumulative Return (start = 1.0)")
    ax.legend(loc="upper left")
    ax.yaxis.set_major_formatter(mticker.FuncFormatter(lambda y, _: f"{y:.2f}"))
 
    out = os.path.join(PLOTS, "plot3_equity_curves.png")
    fig.savefig(out, dpi=DPI)
    plt.close(fig)
    print(f"Saved: {out}")
 
 
# ===========================================================================
# PLOT 4 — Validation vs Test Performance
# ===========================================================================
def plot4_val_vs_test():
    ens = pd.read_csv(os.path.join(TABLES, "ensemble_final_metrics.csv"))
    val_row  = ens[ens["stage"] == "validation"].iloc[0]
    test_row = ens[ens["stage"] == "test"].iloc[0]
 
    # Normalise each metric by its validation value so all fit on one axis
    metrics = ["IC Mean", "ICIR", "Sharpe"]
    raw_keys = ["ic_mean", "icir", "portfolio_sharpe"]
 
    val_vals  = [float(val_row[k])  for k in raw_keys]
    test_vals = [float(test_row[k]) for k in raw_keys]
 
    # Normalise: divide both by the validation value
    norm_val  = [1.0, 1.0, 1.0]
    norm_test = [
        test_vals[i] / val_vals[i] if abs(val_vals[i]) > 1e-9 else 0.0
        for i in range(3)
    ]
 
    x      = np.arange(len(metrics))
    bar_w  = 0.32
 
    fig, axes = plt.subplots(1, 2, figsize=(13, 6), gridspec_kw={"width_ratios": [1, 1.3]})
 
    # Left panel: raw values
    ax0 = axes[0]
    ax0.bar(x - bar_w / 2, val_vals,  width=bar_w, color=C_ENSEMBLE,   label="Validation",  edgecolor="white")
    ax0.bar(x + bar_w / 2, test_vals, width=bar_w, color="#B0B0B0",    label="Test (OOS)",  edgecolor="white")
    ax0.set_xticks(x)
    ax0.set_xticklabels(metrics)
    ax0.set_title("Raw Metric Values")
    ax0.set_ylabel("Metric Value")
    ax0.axhline(0, color="#AAAAAA", linewidth=0.8, linestyle="--")
    ax0.legend()
 
    # Annotate raw values
    for xi, (v, t) in enumerate(zip(val_vals, test_vals)):
        ax0.text(xi - bar_w / 2, v + 0.005 * abs(v) + 0.002, f"{v:.3f}",
                 ha="center", va="bottom", fontsize=8.5, color=C_ENSEMBLE)
        ax0.text(xi + bar_w / 2, t + 0.005 * abs(t) + 0.002, f"{t:.3f}",
                 ha="center", va="bottom", fontsize=8.5, color="#666666")
 
    # Right panel: normalised (validation = 1.0)
    ax1 = axes[1]
    ax1.bar(x - bar_w / 2, norm_val,  width=bar_w, color=C_ENSEMBLE, label="Validation (= 1.0)", edgecolor="white")
    ax1.bar(x + bar_w / 2, norm_test, width=bar_w, color="#B0B0B0",  label="Test (ratio to val)", edgecolor="white")
    ax1.axhline(1.0, color="#AAAAAA", linewidth=0.9, linestyle="--")
    ax1.axhline(0,   color="#AAAAAA", linewidth=0.8, linestyle=":")
    ax1.set_xticks(x)
    ax1.set_xticklabels(metrics)
    ax1.set_title("Test / Validation Ratio (degradation)")
    ax1.set_ylabel("Normalised Value (validation = 1.0)")
    ax1.legend()
 
    for xi, t in enumerate(norm_test):
        colour = "#CC3333" if t < 0.8 else "#226622"
        ax1.text(xi + bar_w / 2, t + 0.02, f"{t:.2f}×",
                 ha="center", va="bottom", fontsize=9, color=colour, fontweight="bold")
 
    fig.suptitle("Ensemble — Validation vs Test (Out-of-Sample) Performance", fontsize=13, fontweight="bold")
    plt.tight_layout()
 
    out = os.path.join(PLOTS, "plot4_val_vs_test.png")
    fig.savefig(out, dpi=DPI)
    plt.close(fig)
    print(f"Saved: {out}")
 
 
# ===========================================================================
# PLOT 5 — Feature Importance (retrain best GB on validation scored data)
# ===========================================================================
def plot5_feature_importance():
    FEATURE_COLS = [
        "return_1d", "return_5d", "return_20d", "return_60d",
        "volatility_20d", "volume_avg_20d",
        "momentum_ratio_1", "momentum_ratio_2",
        "risk_adjusted_return",
        "interaction_1", "interaction_2",
        "rank_return_20d", "volatility_regime", "trend_consistency",
    ]
    TARGET_COL = "target_5d"
 
    # Use the RF scored validation data (features + target) to retrain
    # best GB config: n_estimators=100, lr=0.05, max_depth=3
    # We use validation scored data as our fitting dataset
    rf_scored = pd.read_csv(
        os.path.join(TABLES, "random_forest_scored_target_5d_random_forest.csv")
    )
 
    sub = rf_scored[FEATURE_COLS + [TARGET_COL]].dropna()
    X   = sub[FEATURE_COLS]
    y   = sub[TARGET_COL]
 
    model = GradientBoostingRegressor(
        n_estimators=100, learning_rate=0.05, max_depth=3, random_state=42
    )
    model.fit(X, y)
    importances = pd.Series(model.feature_importances_, index=FEATURE_COLS)
    importances = importances.sort_values(ascending=True)  # ascending for horizontal bar
 
    # Human-readable labels
    label_map = {
        "return_1d":           "1-Day Return",
        "return_5d":           "5-Day Return",
        "return_20d":          "20-Day Return",
        "return_60d":          "60-Day Return",
        "volatility_20d":      "20-Day Volatility",
        "volume_avg_20d":      "20-Day Avg Volume",
        "momentum_ratio_1":    "Momentum Ratio 1",
        "momentum_ratio_2":    "Momentum Ratio 2",
        "risk_adjusted_return":"Risk-Adjusted Return",
        "interaction_1":       "Interaction Feature 1",
        "interaction_2":       "Interaction Feature 2",
        "rank_return_20d":     "Rank: 20-Day Return",
        "volatility_regime":   "Volatility Regime",
        "trend_consistency":   "Trend Consistency",
    }
    labels = [label_map.get(f, f) for f in importances.index]
 
    # Colour bars by family
    bar_colours = []
    for f in importances.index:
        if "return" in f and "risk" not in f and "rank" not in f:
            bar_colours.append(C_RF)
        elif "volatility" in f or "momentum" in f or "trend" in f or "risk" in f:
            bar_colours.append(C_GB)
        elif "volume" in f:
            bar_colours.append(C_DT)
        elif "interaction" in f or "rank" in f:
            bar_colours.append(C_ELASTICNET)
        else:
            bar_colours.append(C_RIDGE)
 
    fig, ax = plt.subplots(figsize=(10, 7))
    bars = ax.barh(labels, importances.values, color=bar_colours,
                   edgecolor="white", linewidth=0.5)
 
    ax.set_xlabel("Feature Importance (MDI)")
    ax.set_title("Feature Importances — Gradient Boosting (target_5d)")
    ax.xaxis.grid(True, color="#E8E8E8", linewidth=0.7)
    ax.yaxis.grid(False)
    ax.set_axisbelow(True)
 
    # Value labels
    for bar, val in zip(bars, importances.values):
        ax.text(val + 0.001, bar.get_y() + bar.get_height() / 2,
                f"{val:.4f}", va="center", fontsize=8.5)
 
    plt.tight_layout()
    out = os.path.join(PLOTS, "plot5_feature_importance.png")
    fig.savefig(out, dpi=DPI)
    plt.close(fig)
    print(f"Saved: {out}")
 
 
# ===========================================================================
# PLOT 6 — Drawdown Chart
# ===========================================================================
def plot6_drawdown():
    aligned = pd.read_csv(os.path.join(TABLES, "spy_benchmark_aligned_returns.csv"))
    aligned["date"] = pd.to_datetime(aligned["date"])
    aligned = aligned.sort_values("date").reset_index(drop=True)
 
    ens_equity = aligned.set_index("date")["strategy_equity"]
    spy_equity  = aligned.set_index("date")["spy_equity"]
 
    ens_dd = drawdown_series(ens_equity)
    spy_dd = drawdown_series(spy_equity)
 
    fig, ax = plt.subplots(figsize=(12, 5))
 
    ax.fill_between(ens_dd.index, ens_dd.values, 0,
                    color=C_ENSEMBLE, alpha=0.35, label="Ensemble")
    ax.fill_between(spy_dd.index, spy_dd.values, 0,
                    color=C_SPY, alpha=0.15, label="SPY")
 
    ax.plot(ens_dd.index, ens_dd.values, color=C_ENSEMBLE, linewidth=1.6)
    ax.plot(spy_dd.index, spy_dd.values, color=C_SPY, linewidth=1.6,
            linestyle="--")
 
    ax.axhline(0, color="#AAAAAA", linewidth=0.8)
    ax.set_title("Drawdown — Test Period 2024 (Ensemble vs SPY)")
    ax.set_xlabel("Date")
    ax.set_ylabel("Drawdown (%)")
    ax.yaxis.set_major_formatter(mticker.FuncFormatter(
        lambda y, _: f"{y*100:.0f}%"
    ))
    ax.legend()
 
    # Annotate max drawdowns
    ens_max = float(ens_dd.min())
    spy_max = float(spy_dd.min())
    ens_date = ens_dd.idxmin()
    spy_date = spy_dd.idxmin()
    ax.annotate(f"Max DD: {ens_max*100:.1f}%",
                xy=(ens_date, ens_max),
                xytext=(15, -20), textcoords="offset points",
                fontsize=9, color=C_ENSEMBLE,
                arrowprops=dict(arrowstyle="->", color=C_ENSEMBLE, lw=1.2))
    ax.annotate(f"Max DD: {spy_max*100:.1f}%",
                xy=(spy_date, spy_max),
                xytext=(15, 20), textcoords="offset points",
                fontsize=9, color="#444444",
                arrowprops=dict(arrowstyle="->", color="#444444", lw=1.2))
 
    plt.tight_layout()
    out = os.path.join(PLOTS, "plot6_drawdown.png")
    fig.savefig(out, dpi=DPI)
    plt.close(fig)
    print(f"Saved: {out}")
 
 
# ===========================================================================
# MAIN
# ===========================================================================
if __name__ == "__main__":
    print("Generating plots...")
    print()
 
    print("[1/6] Model Comparison Bar Chart")
    plot1_model_comparison()
 
    print("[2/6] IC Over Time")
    plot2_ic_over_time()
 
    print("[3/6] Equity Curves")
    plot3_equity_curves()
 
    print("[4/6] Validation vs Test Performance")
    plot4_val_vs_test()
 
    print("[5/6] Feature Importance")
    plot5_feature_importance()
 
    print("[6/6] Drawdown Chart")
    plot6_drawdown()
 
    print()
    print("All 6 plots saved successfully.")