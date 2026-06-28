"""
Figure S1 – Top 20 destination countries by total shipping traffic
Vessel type: All
Y-axis: raw voyage counts, fixed 0–1,000,000
Matches the style conventions of figure1_predicted_traffic.py
"""

import os
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
from matplotlib.patches import Patch
import seaborn as sns
import math

from figure_common import SSP_COLORS, HandlerBandWithLine, PI_LABEL_PROPAGATED , MACRO_PI_RATE


def load_and_prepare_data(csv_file_path: str):
    print(f"Loading data from {csv_file_path} ...")
    df = pd.read_csv(csv_file_path)

    df = df[df["VesselType"] == "All"].copy()

    # Step 1: aggregate by Year, SSP, DCountry (all years, including 2018)
    df_agg = (
        df.groupby(["Year", "SSP", "DCountry"])["RF"]
        .sum()
        .reset_index()
    )

    # Step 2: PI bounds on raw counts
    df_agg["Lower"] = df_agg["RF"] * (1 - MACRO_PI_RATE)
    df_agg["Upper"] = df_agg["RF"] * (1 + MACRO_PI_RATE)


    all_countries = sorted(df_agg["DCountry"].unique().tolist())
    plot_df = df_agg.copy()

    # Filter to 5-year grid 2018, 2020–2100 for plotting only
    valid_years = [2018] + list(range(2020, 2105, 5))
    plot_df = plot_df[plot_df["Year"].isin(valid_years)].copy()

    # No unit scaling — keep raw voyage counts
    return plot_df, all_countries


def make_figure(plot_df, all_countries,
                output_pdf="figureS1_Dcountry_traffic.pdf"):
    print("Generating figure ...")

    plt.rcParams["font.family"] = "sans-serif"
    plt.rcParams["font.sans-serif"] = ["Arial", "Helvetica", "DejaVu Sans"]
    sns.set_theme(style="ticks",
                  rc={"axes.edgecolor": ".15", "axes.linewidth": 1.0})

    n_cols = 13
    n_rows = math.ceil(len(all_countries) / n_cols)

    fig = plt.figure(figsize=(45, n_rows * 3))
    gs  = GridSpec(n_rows, n_cols, figure=fig,
                   left=0.06, right=0.82,
                   wspace=0.30, hspace=0.45)

    ssps    = sorted(plot_df["SSP"].unique())
    x_ticks = [2020, 2040, 2060, 2080, 2100]

    for i, country in enumerate(all_countries):
        row, col = divmod(i, n_cols)
        ax = fig.add_subplot(gs[row, col])

        c_data = plot_df[plot_df["DCountry"] == country]

        for ssp in ssps:
            d = c_data[c_data["SSP"] == ssp].sort_values("Year")
            if d.empty:
                continue
            color = SSP_COLORS.get(ssp, "black")
            ax.plot(d["Year"], d["RF"],
                    label=ssp, color=color,
                    linewidth=1.5, marker="o",
                    markersize=3.5, zorder=4)
            ax.fill_between(d["Year"], d["Lower"], d["Upper"],
                            color=color, alpha=0.15,
                            zorder=3, edgecolor="none")

        ax.set_title(country, fontsize=11, pad=6)
        ax.set_xlim(2015, 2105)
        ax.set_xticks(x_ticks)

        ax.tick_params(axis="x", bottom=True, direction="out",
                       length=5, width=1.0, labelsize=9)
        ax.tick_params(axis="y", left=True, direction="out",
                       length=5, width=1.0, labelsize=9)
        ax.grid(False)
        for spine in ax.spines.values():
            spine.set_color("black")
            spine.set_linewidth(1.0)

        is_last_row_of_col = (i + n_cols >= len(all_countries))
        if not is_last_row_of_col:
            plt.setp(ax.get_xticklabels(), visible=False)
        else:
            ax.tick_params(axis="x", rotation=45)
            ax.set_xlabel("Year", fontsize=10, labelpad=6)

        if col == 0:
            ax.set_ylabel("Traffic", fontsize=13, labelpad=8)

    # ── Legend: figure1 style ─────────────────────────────────────────────────
    handles, labels = fig.get_axes()[0].get_legend_handles_labels()
    unique_dict = dict(zip(labels, handles))
    ssp_handles = list(unique_dict.values())
    ssp_labels  = list(unique_dict.keys())

    pi_handle = Patch(facecolor="gray", alpha=0.3, edgecolor="none")

    fig.legend(
        handles=ssp_handles + [pi_handle],
        labels=ssp_labels + [PI_LABEL_PROPAGATED],
        handler_map={pi_handle: HandlerBandWithLine()},
        title=None,
        loc="upper left",
        bbox_to_anchor=(0.82, 0.5),
        fontsize=18,
        title_fontsize=13,
        frameon=False,
        facecolor="white",
        edgecolor="black",
        fancybox=False,
        labelspacing=1.2
    )

    fig.suptitle(
        "Destination Countries' Predicted Traffic",
        fontsize=32, x=0.44, y=0.9
    )

    plt.savefig(output_pdf, bbox_inches="tight", dpi=150)
    print(f"Saved → {output_pdf}")
    return fig


if __name__ == "__main__":
    input_file = "../out/merged_predictions.csv"

    if not os.path.exists(input_file):
        print(f"ERROR: '{input_file}' not found.")
    else:
        plot_df, top_countries = load_and_prepare_data(input_file)
        plot_df.to_csv("figureS2_plot_data.csv", index=False)
        make_figure(plot_df, top_countries,
                    output_pdf="figureS2_all_Dcountry_traffic.pdf")