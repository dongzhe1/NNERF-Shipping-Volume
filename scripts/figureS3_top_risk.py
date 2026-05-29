"""
Figure S3 - Risk change to Top 20 destination countries
Two independent figures (SSP1 and SSP3), each with its own title, y-label and legend.
Saved as a two-page PDF.

Fix (vs original):
  - Top-20 selection now uses TRAFFIC data (merged_predictions.csv), summing RF
    across all Years and SSPs per DCountry — exactly matching the R script logic.
    The original script derived the top-20 from the risk file (ISO codes), which
    produced a different country set.
  - Subplot panels are more compact (wspace / hspace tightened).
"""

import os
import math
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.lines as mlines
from matplotlib.gridspec import GridSpec
from matplotlib.backends.backend_pdf import PdfPages
import seaborn as sns

from figure_common import SSP_COLORS, scenario_to_policy

POLICY_COLORS = {
    "Policy 1": "#E07B54",
    "Policy 2": "#B5A642",
    "Policy 3": "#4DAF7C",
    "Policy 4": "#5B9BD5",
}

RISK_FILE    = "../data/ballast/p_country/ballast_country_aggregated.csv"
TRAFFIC_FILE = "../out/merged_predictions.csv"
COUNTRY_MAP  = "../data/predict/country.csv"


# -----------------------------------------------------------------------------
# 1.  Load country map: ISO <-> full name
# -----------------------------------------------------------------------------
def load_country_map(country_map_path: str):
    cmap = pd.read_csv(country_map_path)
    iso_to_name = dict(zip(cmap["COUNTRY_CODE"], cmap["REGION_NAME"]))
    name_to_iso = dict(zip(cmap["REGION_NAME"], cmap["COUNTRY_CODE"]))
    return iso_to_name, name_to_iso


# -----------------------------------------------------------------------------
# 2.  Identify top-20 destination countries from TRAFFIC data
#     (mirrors R / figureS1 logic exactly):
#       group by DCountry, sum RF across all Years & SSPs, take top 20 by name,
#       then convert full names → ISO codes for use with the risk file.
# -----------------------------------------------------------------------------
def get_top20_countries(traffic_path: str, country_map_path: str, top_n: int = 20):
    iso_to_name, name_to_iso = load_country_map(country_map_path)

    df_tr = pd.read_csv(traffic_path)
    df_tr = df_tr[df_tr["VesselType"] == "All"].copy()

    # Sum RF over all Year/SSP per DCountry  (same as R: group_by(DCountry) %>% summarize(total=sum(RF)))
    top_names = (
        df_tr.groupby("DCountry")["RF"]
        .sum()
        .nlargest(top_n)
        .index.tolist()
    )
    top_names_sorted = sorted(top_names)           # alphabetical, matching figureS1

    # Convert full names → ISO codes (needed to index into risk data)
    top_iso = []
    missing = []
    for name in top_names_sorted:
        iso = name_to_iso.get(name)
        if iso is None:
            missing.append(name)
            print(f"WARNING: no ISO code for traffic country '{name}' – skipped")
        else:
            top_iso.append(iso)

    print(f"Top-{top_n} destination countries (traffic-derived, alphabetical):")
    for name, iso in zip(top_names_sorted, top_iso):
        print(f"  {iso}  {name}")

    return top_iso, top_names_sorted, iso_to_name


# -----------------------------------------------------------------------------
# 3.  Load and prepare risk data
# -----------------------------------------------------------------------------
def load_and_prepare_risk(risk_path: str, top20_iso: list):
    print(f"Loading risk data from {risk_path} ...")
    df = pd.read_csv(risk_path)

    df = df[df["VesselType"] == "All"].copy()
    df["Policy"] = df["Scenario"].map(scenario_to_policy)
    df = df[df["Policy"].notna()].copy()

    df_agg = (
        df.groupby(["Year", "Policy", "SSP", "DestinationCountry"])[
            ["prob", "prob_lower", "prob_upper"]
        ]
        .sum()
        .reset_index()
    )

    baseline = (
        df_agg[
            (df_agg["Year"]   == 2018) &
            (df_agg["SSP"]    == "SSP1") &
            (df_agg["Policy"] == "Policy 1")
        ][["DestinationCountry", "prob"]]
        .rename(columns={"prob": "baseline"})
    )

    df_agg = df_agg.merge(baseline, on="DestinationCountry", how="left")
    df_agg["risk_ratio"]       = df_agg["prob"]       / df_agg["baseline"]
    df_agg["risk_ratio_lower"] = df_agg["prob_lower"] / df_agg["baseline"]
    df_agg["risk_ratio_upper"] = df_agg["prob_upper"] / df_agg["baseline"]

    valid_years = list(range(2020, 2105, 5))
    plot_df = df_agg[
        df_agg["DestinationCountry"].isin(top20_iso) &
        df_agg["Year"].isin(valid_years)
    ].copy()

    return plot_df


# -----------------------------------------------------------------------------
# 4.  Build one self-contained figure for a single SSP
#     Panel layout is tighter than the original (wspace/hspace reduced).
# -----------------------------------------------------------------------------
def make_ssp_figure(plot_df, top20_iso, top20_names, iso_to_name, ssp, n_cols=5):
    plt.rcParams["font.family"] = "sans-serif"
    plt.rcParams["font.sans-serif"] = ["Arial", "Helvetica", "DejaVu Sans"]
    sns.set_theme(style="ticks",
                  rc={"axes.edgecolor": ".15", "axes.linewidth": 1.0})

    n_rows = math.ceil(len(top20_iso) / n_cols)

    # Tighter panel dimensions vs original (2.2 → 1.9 each, spacing reduced)
    fig_w  = n_cols * 1.9 + 2.6
    fig_h  = n_rows * 1.9 + 1.1

    fig = plt.figure(figsize=(fig_w, fig_h))
    gs  = GridSpec(
        n_rows, n_cols, figure=fig,
        left=0.10, right=0.83,
        top=0.91, bottom=0.09,
        wspace=0.20,
        hspace=0.20,
    )

    policies = sorted(plot_df["Policy"].unique())
    x_ticks  = [2020, 2040, 2060, 2080, 2100]
    ssp_data = plot_df[plot_df["SSP"] == ssp]

    for i, country_iso in enumerate(top20_iso):
        row, col = divmod(i, n_cols)
        ax = fig.add_subplot(gs[row, col])

        c_data = ssp_data[ssp_data["DestinationCountry"] == country_iso]

        for policy in policies:
            d = c_data[c_data["Policy"] == policy].sort_values("Year")
            if d.empty:
                continue
            color = POLICY_COLORS.get(policy, "black")
            ax.plot(d["Year"], d["risk_ratio"],
                    label=policy, color=color,
                    linewidth=1.5, zorder=4)
            ax.fill_between(d["Year"],
                            d["risk_ratio_lower"], d["risk_ratio_upper"],
                            color=color, alpha=0.15,
                            zorder=3, edgecolor="none")

        ax.axhline(1, color="black", linewidth=0.8, linestyle="--", zorder=2)

        # Use the traffic-derived full country name as panel title
        # (fall back to iso_to_name from the country map, then bare ISO code)
        full_name = top20_names[i] if i < len(top20_names) else iso_to_name.get(country_iso, country_iso)
        ax.set_title(full_name, fontsize=7.5, pad=4)

        ax.set_xlim(2015, 2105)
        ax.set_xticks(x_ticks)
        ax.tick_params(axis="x", bottom=True, direction="out",
                       length=4, width=0.8, labelsize=6.5)
        ax.tick_params(axis="y", left=True, direction="out",
                       length=4, width=0.8, labelsize=6.5)
        ax.grid(False)
        for spine in ax.spines.values():
            spine.set_color("black")
            spine.set_linewidth(0.8)

        is_last_row_of_col = (i + n_cols >= len(top20_iso))
        if not is_last_row_of_col:
            plt.setp(ax.get_xticklabels(), visible=False)
        else:
            ax.tick_params(axis="x", rotation=45)
            ax.set_xlabel("Year", fontsize=7.5, labelpad=3)

        ax.set_ylabel("")

    # Shared y-axis label
    fig.text(
        0.06, 0.50,
        "Risk Change relative to 2018 (Ratio)",
        fontsize=10,
        va="center", ha="center",
        rotation=90,
    )

    # Figure title
    fig.suptitle(
        f"Risk Change to Selected Countries ({ssp})",
        fontsize=13, x=0.46, y=0.97,
    )

    # Legend
    policy_handles = [
        mlines.Line2D([0], [0], color=POLICY_COLORS.get(p, "black"),
                      linewidth=2.0, label=p)
        for p in policies
    ]
    baseline_handle = mlines.Line2D(
        [0], [0], color="black", linewidth=1.2,
        linestyle="--", label="2018 Baseline risk"
    )
    spacer = mlines.Line2D([], [], color="none", label=" ")

    fig.legend(
        handles=policy_handles + [spacer, baseline_handle],
        labels=policies + [" ", "2018 Baseline risk"],
        loc="center left",
        bbox_to_anchor=(0.84, 0.50),
        fontsize=10,
        frameon=False,
        labelspacing=0.8,
    )

    n_policies = len(policies)
    fig.text(0.84, 0.50 + (n_policies - 0.8) * 0.032,
             "Policy", fontsize=11, fontweight="bold", va="bottom")
    fig.text(0.84, 0.50 - 1.2 * 0.032,
             "Reference", fontsize=11, fontweight="bold", va="top")

    return fig


# -----------------------------------------------------------------------------
# 5.  Save both SSP figures into one multi-page PDF
# -----------------------------------------------------------------------------
def make_figure(plot_df, top20_iso, top20_names, iso_to_name,
                output_pdf="figureS3_risk_top20.pdf"):
    print("Generating Figure S3 ...")

    with PdfPages(output_pdf) as pdf:
        for ssp in ["SSP1", "SSP3"]:
            fig = make_ssp_figure(plot_df, top20_iso, top20_names, iso_to_name, ssp)
            pdf.savefig(fig, bbox_inches="tight")
            plt.close(fig)
            print(f"  saved {ssp} page")

    print(f"Saved -> {output_pdf}")


# -----------------------------------------------------------------------------
if __name__ == "__main__":
    for path in [RISK_FILE, TRAFFIC_FILE, COUNTRY_MAP]:
        if not os.path.exists(path):
            print(f"ERROR: '{path}' not found.")
            exit(1)

    print("Step 1: identifying top-20 destination countries from traffic data ...")
    top20_iso, top20_names, iso_to_name = get_top20_countries(
        TRAFFIC_FILE, COUNTRY_MAP, top_n=20
    )

    print("\nStep 2: loading and preparing risk data ...")
    plot_df = load_and_prepare_risk(RISK_FILE, top20_iso)
    plot_df.to_csv("figureS3_plot_data.csv", index=False)

    make_figure(plot_df, top20_iso, top20_names, iso_to_name,
                output_pdf="figureS3_risk_top20.pdf")