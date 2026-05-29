"""
Figure S5 – GDP distribution: historical baseline vs. OECD SSP projections
Style conventions match figureS3 / figureS4.
Outputs:
  figureS5_gdp_distribution.pdf  – the figure
  figureS5_stats.csv             – summary statistics table
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.lines as mlines
import seaborn as sns
from scipy import stats
from matplotlib.backends.backend_pdf import PdfPages

from figure_common import SSP_COLORS   # SSP1–SSP5 palette

# ── Paths ─────────────────────────────────────────────────────────────────────
TRADE_FILE  = "../data/voyages_grouped_country.csv"
GDP_FILE    = "../data/predict/GDP.csv"
OUTPUT_PDF  = "figureS5_gdp_distribution.pdf"
OUTPUT_CSV  = "figureS5_stats.csv"

PERCENTILE  = 98          # upper truncation percentile
SSP_LIST    = ["SSP1", "SSP2", "SSP3", "SSP4", "SSP5"]
YEAR_COLS   = [str(y) for y in range(2025, 2101, 5)]

# ── Global style (matches figureS3/S4) ────────────────────────────────────────
plt.rcParams["font.family"]     = "sans-serif"
plt.rcParams["font.sans-serif"] = ["Arial", "Helvetica", "DejaVu Sans"]
sns.set_theme(style="ticks",
              rc={"axes.edgecolor": ".15", "axes.linewidth": 1.0})


# ── Data loading ──────────────────────────────────────────────────────────────
def load_baseline_gdp(path: str) -> list:
    """Unique country GDP values from trade data (origin + destination)."""
    df = pd.read_csv(path)
    country_gdp = {}
    for col_c, col_g in [("OCountry", "OGDP"), ("DCountry", "DGDP")]:
        sub = df[[col_c, col_g]].dropna()
        for _, row in sub.iterrows():
            try:
                country_gdp[row[col_c]] = float(row[col_g]) / 1e9
            except (ValueError, TypeError):
                continue
    values = list(country_gdp.values())
    limit  = np.percentile(values, PERCENTILE)
    return [x for x in values if x <= limit]


def load_ssp_gdp(path: str) -> dict:
    """GDP values per SSP scenario from OECD ENV-Growth 2023 model."""
    df = pd.read_csv(path)
    df = df[df["Model"] == "OECD ENV-Growth 2023"]
    result = {}
    for ssp in SSP_LIST:
        sub = df[df["Scenario"] == ssp]
        vals = []
        for _, row in sub.iterrows():
            for yr in YEAR_COLS:
                if yr in sub.columns and pd.notna(row[yr]):
                    try:
                        vals.append(float(row[yr]))
                    except (ValueError, TypeError):
                        continue
        if vals:
            limit = np.percentile(vals, PERCENTILE)
            result[ssp] = [x for x in vals if x <= limit]
            print(f"{ssp}  p{PERCENTILE} upper limit: {limit:.1f} B USD")
    return result


# ── Figure ────────────────────────────────────────────────────────────────────
def make_figure(baseline: list, ssp_data: dict) -> plt.Figure:
    fig, ax = plt.subplots(figsize=(9, 5))

    # Baseline – dark, thicker line
    sns.kdeplot(data=baseline, ax=ax,
                fill=False, linewidth=2.0, color="#2F2F2F",
                label="Baseline (2002–2018)", clip=(0, None))

    # SSP scenarios – colours from figure_common
    for ssp in SSP_LIST:
        if ssp not in ssp_data or not ssp_data[ssp]:
            continue
        sns.kdeplot(data=ssp_data[ssp], ax=ax,
                    fill=False, linewidth=1.5,
                    color=SSP_COLORS[ssp],
                    label=ssp, clip=(0, None))

    # Training-data boundary line
    hist_max = max(baseline)
    ax.axvline(hist_max, color="red", linestyle="--",
               linewidth=1.2, alpha=0.7, zorder=2)

    # Log y-axis (must be set before using ax.get_ylim() in log space)
    ax.set_yscale("log")

    # ── Three annotation boxes (matching original figure) ────────────────────
    # Use axes-fraction coordinates (transform=ax.transAxes) so positions are
    # independent of data range; tweak x/y values to taste.

    # 1. "Training Data Boundary" – near top of the dashed line
    ax.annotate(
        "Training Data\nBoundary",
        xy=(hist_max, 1), xycoords=("data", "axes fraction"),
        xytext=(0, -12), textcoords="offset points",
        fontsize=8, va="top", ha="center",
        bbox=dict(boxstyle="round,pad=0.35",
                  facecolor="white", edgecolor="gray",
                  linewidth=0.6, alpha=0.9),
    )

    # 2. "Training Range" – lower-left, light-blue background
    ax.text(
        0.02, 0.08,                          # axes fraction (x, y)
        "Training Range\n(Historical Data)",
        transform=ax.transAxes,
        fontsize=8, va="bottom", ha="left",
        bbox=dict(boxstyle="round,pad=0.4",
                  facecolor="#AED6F1", edgecolor="none", alpha=0.75),
    )

    # 3. "Extrapolation Range" – lower-right, light-yellow background
    ax.text(
        0.30, 0.08,
        "Extrapolation Range\n(Future Projections)",
        transform=ax.transAxes,
        fontsize=8, va="bottom", ha="left",
        bbox=dict(boxstyle="round,pad=0.4",
                  facecolor="#FCF3CF", edgecolor="none", alpha=0.75),
    )

    # Labels
    ax.set_xlabel("GDP (billions USD)",     fontsize=10, labelpad=4)
    ax.set_ylabel("Density (log scale)",    fontsize=10, labelpad=4)
    ax.set_title(
        "GDP Distribution: Historical Baseline vs. OECD SSP Projections",
        fontsize=11, pad=8,
    )

    # Axis styling
    ax.tick_params(axis="both", which="major",
                   direction="out", length=4, width=0.8, labelsize=8)
    ax.tick_params(axis="both", which="minor",
                   direction="out", length=2, width=0.6)
    ax.grid(False)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color("black")
        ax.spines[side].set_linewidth(0.8)

    # Legend – matches figureS3 style (no frame, right of plot)
    baseline_handle = mlines.Line2D([0], [0], color="#2F2F2F",
                                    linewidth=2.0, label="Baseline (2002–2018)")
    ssp_handles = [
        mlines.Line2D([0], [0], color=SSP_COLORS[s], linewidth=1.5, label=s)
        for s in SSP_LIST if s in ssp_data
    ]
    boundary_handle = mlines.Line2D([0], [0], color="red", linewidth=1.2,
                                    linestyle="--", label="Training boundary")

    ax.legend(
        handles=[baseline_handle] + ssp_handles + [boundary_handle],
        fontsize=8,
        frameon=False,
        loc="upper right",
        labelspacing=0.6,
    )

    plt.tight_layout(rect=[0, 0, 1, 1])
    return fig


# ── Statistics → CSV ──────────────────────────────────────────────────────────
def compute_stats(baseline: list, ssp_data: dict) -> pd.DataFrame:
    pcts = [0.25, 0.50, 0.75, 0.90, 0.95]

    def row(label, vals):
        s = pd.Series(vals)
        d = {
            "Series":   label,
            "N":        len(vals),
            "Mean":     s.mean(),
            "Median":   s.median(),
            "SD":       s.std(),
            "Min":      s.min(),
            "Max":      s.max(),
            "Skewness": stats.skew(vals),
        }
        for p in pcts:
            d[f"p{int(p*100)}"] = s.quantile(p)
        return d

    rows = [row("Baseline (2002–2018)", baseline)]
    for ssp in SSP_LIST:
        if ssp in ssp_data and ssp_data[ssp]:
            rows.append(row(ssp, ssp_data[ssp]))

    return pd.DataFrame(rows).set_index("Series")


# ── Main ──────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    print("Loading baseline GDP ...")
    baseline = load_baseline_gdp(TRADE_FILE)
    print(f"  {len(baseline)} country records after p{PERCENTILE} truncation\n")

    print("Loading SSP GDP projections ...")
    ssp_data = load_ssp_gdp(GDP_FILE)

    fig = make_figure(baseline, ssp_data)

    with PdfPages(OUTPUT_PDF) as pdf:
        pdf.savefig(fig, bbox_inches="tight")
    print(f"\nSaved figure → {OUTPUT_PDF}")
    plt.close(fig)

    stats_df = compute_stats(baseline, ssp_data)
    stats_df.to_csv(OUTPUT_CSV)
    print(f"Saved stats  → {OUTPUT_CSV}")
    print("\n", stats_df.round(2).to_string())