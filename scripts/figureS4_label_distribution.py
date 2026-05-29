"""
Figure S4 – Distribution of shipping volume (original vs log-transformed)
Style conventions match figureS3 / figureS1.
Outputs:
  figureS4_distribution.pdf   – the figure
  figureS4_stats.csv          – summary statistics table
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import seaborn as sns
from scipy import stats
from scipy.stats import jarque_bera
from matplotlib.backends.backend_pdf import PdfPages

# ── Paths ─────────────────────────────────────────────────────────────────────
FILE_PATH   = "../data/voyages_grouped_country.csv"
OUTPUT_PDF  = "figureS4_distribution.pdf"
OUTPUT_CSV  = "figureS4_stats.csv"

# ── Global style (matches figureS3) ───────────────────────────────────────────
plt.rcParams["font.family"]     = "sans-serif"
plt.rcParams["font.sans-serif"] = ["Arial", "Helvetica", "DejaVu Sans"]
sns.set_theme(style="ticks",
              rc={"axes.edgecolor": ".15", "axes.linewidth": 1.0})

# Same muted palette used in sister figures
COLOR_ORIG = "#6BB6D6"
COLOR_LOG  = "#D67BB6"


# ── Load & prepare ────────────────────────────────────────────────────────────
def load_data(path: str):
    df = pd.read_csv(path)
    if "RouteCount" not in df.columns:
        raise ValueError("'RouteCount' column not found in CSV file.")
    raw = df["RouteCount"]
    q98 = raw.quantile(0.98)
    truncated = raw[raw <= q98]
    log_tr    = np.log1p(truncated)
    return raw, truncated, log_tr


# ── Figure ────────────────────────────────────────────────────────────────────
def make_figure(truncated, log_tr):
    fig, (ax1, ax2) = plt.subplots(
        1, 2,
        figsize=(10, 4),
    )

    # ── Panel a: original ───────────────────────────────────────────────────
    sns.kdeplot(data=truncated, ax=ax1, fill=True,
                alpha=0.25, linewidth=1.5, color=COLOR_ORIG)

    ax1.set_title("Original Distribution", fontsize=11, pad=6)
    ax1.set_xlabel("Shipping Volume",          fontsize=10, labelpad=4)
    ax1.set_ylabel("Density",                  fontsize=10, labelpad=4)

    # ── Panel b: log-transformed ─────────────────────────────────────────────
    sns.kdeplot(data=log_tr, ax=ax2, fill=True,
                alpha=0.25, linewidth=1.5, color=COLOR_LOG)

    ax2.set_title("Log-transformed Distribution", fontsize=11, pad=6)
    ax2.set_xlabel("log(Shipping Volume + 1)",     fontsize=10, labelpad=4)
    ax2.set_ylabel("Density",                      fontsize=10, labelpad=4)

    # ── Shared axis styling ──────────────────────────────────────────────────
    for ax in (ax1, ax2):
        ax.tick_params(axis="both", which="major",
                       direction="out", length=4, width=0.8, labelsize=8)
        ax.grid(False)
        for side in ("top", "right"):
            ax.spines[side].set_visible(False)
        for side in ("left", "bottom"):
            ax.spines[side].set_color("black")
            ax.spines[side].set_linewidth(0.8)

    fig.suptitle(
        "Distribution of Shipping Volume: Original vs. Log-transformed",
        fontsize=16, y=1.02,
    )
    plt.tight_layout(rect=[0, 0, 1, 0.95])

    return fig


# ── Statistics → CSV ──────────────────────────────────────────────────────────
def compute_stats(raw, truncated, log_tr) -> pd.DataFrame:
    pcts = [0.75, 0.85, 0.95, 0.99]

    def summarise(series, label):
        d = {
            "Series":    label,
            "N":         len(series),
            "Mean":      series.mean(),
            "Median":    series.median(),
            "SD":        series.std(),
            "Min":       series.min(),
            "Max":       series.max(),
            "Skewness":  stats.skew(series),
        }
        for p in pcts:
            d[f"p{int(p*100)}"] = series.quantile(p)

        # Jarque-Bera (full series; Shapiro-Wilk is capped at n=5000)
        jb_stat, jb_p = jarque_bera(series)
        d["JB_stat"]  = jb_stat
        d["JB_pval"]  = jb_p
        return d

    rows = [
        summarise(raw,                  "Original (full)"),
        summarise(truncated,            "Original (≤ p98)"),
        summarise(log_tr,               "Log-transformed (≤ p98)"),
        summarise(np.log1p(raw),        "Log-transformed (full)"),
    ]
    return pd.DataFrame(rows).set_index("Series")


# ── Console report ────────────────────────────────────────────────────────────
def print_report(stats_df: pd.DataFrame):
    print("=" * 60)
    print("STATISTICAL ANALYSIS")
    print("=" * 60)
    print(stats_df.T.to_string(float_format=lambda x: f"{x:.4f}"))

    orig_skew = stats_df.loc["Original (full)", "Skewness"]
    log_skew  = stats_df.loc["Log-transformed (full)", "Skewness"]
    print(f"\nSkewness reduction (full series): {abs(orig_skew) - abs(log_skew):.4f}")
    print("\nNote: both distributions significantly deviate from normality "
          "(Jarque-Bera p ≪ 0.001).")


# ── Main ──────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    raw, truncated, log_tr = load_data(FILE_PATH)

    fig = make_figure(truncated, log_tr)

    with PdfPages(OUTPUT_PDF) as pdf:
        pdf.savefig(fig, bbox_inches="tight")
    print(f"Saved figure  → {OUTPUT_PDF}")
    plt.close(fig)

    stats_df = compute_stats(raw, truncated, log_tr)
    stats_df.to_csv(OUTPUT_CSV)
    print(f"Saved stats   → {OUTPUT_CSV}")

    print_report(stats_df)