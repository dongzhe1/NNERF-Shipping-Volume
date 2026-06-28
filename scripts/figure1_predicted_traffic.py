import pandas as pd
from matplotlib.patches import Patch
from figure_common import SSP_COLORS, HandlerBandWithLine, PI_LABEL, MACRO_PI_RATE
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.gridspec import GridSpec
import os


def load_and_prepare_data(csv_file_path):
    print(f"Loading data from {csv_file_path}...")
    try:
        df = pd.read_csv(csv_file_path)
    except FileNotFoundError:
        print(f"Error: Could not find {csv_file_path}")
        return None

    df['Year'] = pd.to_numeric(df['Year'], errors='coerce')
    valid_years = [2018] + list(range(2020, 2105, 5))
    df = df[df['Year'].isin(valid_years)].copy()

    vessel_mapping = {'All': 'All vessel types', 'Oil': 'Tanker'}
    df['VesselType'] = df['VesselType'].replace(vessel_mapping)

    print("Aggregating country-pair predictions to global totals by VesselType...")

    plot_df = df.groupby(['Year', 'SSP', 'VesselType']).agg(RF=('RF', 'sum')).reset_index()

    plot_df['Lower'] = plot_df['RF'] * (1 - MACRO_PI_RATE)
    plot_df['Upper'] = plot_df['RF'] * (1 + MACRO_PI_RATE)

    plot_df['RF'] /= 1e6
    plot_df['Lower'] /= 1e6
    plot_df['Upper'] /= 1e6

    return plot_df

def reproduce_ggplot_layout(plot_df, output_pdf='figure1_Projected_global_shipping_traffic.pdf'):
    print("Generating plot with ggplot2 style layout...")

    plt.rcParams['font.family'] = 'sans-serif'
    plt.rcParams['font.sans-serif'] = ['Arial', 'Helvetica', 'DejaVu Sans']

    sns.set_theme(style="ticks", rc={"axes.edgecolor": ".15", "axes.linewidth": 1.0})

    fig = plt.figure(figsize=(11.5, 6.5))
    gs = GridSpec(2, 4, figure=fig, wspace=0.35, hspace=0.3)

    ax_main = fig.add_subplot(gs[:, 0:2])
    ax_bulk = fig.add_subplot(gs[0, 2])
    ax_cont = fig.add_subplot(gs[0, 3], sharey=ax_bulk)
    ax_gen  = fig.add_subplot(gs[1, 2], sharex=ax_bulk)
    ax_tank = fig.add_subplot(gs[1, 3], sharex=ax_cont, sharey=ax_gen)

    ssp_colors = SSP_COLORS

    ssps = sorted(plot_df['SSP'].unique())
    x_ticks = [2018, 2040, 2060, 2080, 2100]

    def plot_panel(ax, v_type, title, is_main=False):
        v_data = plot_df[plot_df['VesselType'] == v_type]
        if v_data.empty:
            print(f"Warning: No data found for '{v_type}'")
            return

        for ssp in ssps:
            d = v_data[v_data['SSP'] == ssp].sort_values('Year')
            x = d['Year']

            linewidth = 2.0 if is_main else 1.5
            ax.plot(x, d['RF'], label=ssp, color=ssp_colors.get(ssp, 'black'), linewidth=linewidth, zorder=4)
            ax.fill_between(x, d['Lower'], d['Upper'], color=ssp_colors.get(ssp, 'black'), alpha=0.15, zorder=3, edgecolor='none')

        ax.set_title(title, fontsize=16 if is_main else 15, pad=12)
        ax.set_xlim(2015, 2105)
        ax.set_xticks(x_ticks)

        ax.tick_params(axis='x', bottom=True, direction='out', length=6, width=1.0, labelsize=13)
        ax.tick_params(axis='y', left=True, direction='out', length=6, width=1.0, labelsize=13)

        ax.grid(False)
        for spine in ax.spines.values():
            spine.set_color('black')
            spine.set_linewidth(1.0)

    plot_panel(ax_main, 'All vessel types', 'All vessel types', is_main=True)

    ax_main.set_ylabel('Shipping Traffic ($10^6$)', fontsize=16, labelpad=12)
    ax_main.set_xlabel('Year', fontsize=16, labelpad=10)
    ax_main.tick_params(axis='x', rotation=45)

    handles, labels = ax_main.get_legend_handles_labels()
    unique_dict = dict(zip(labels, handles))
    ssp_handles = list(unique_dict.values())
    ssp_labels = list(unique_dict.keys())

    pi_handle = Patch(facecolor='gray', alpha=0.3, edgecolor='none')

    legend = ax_main.legend(
        handles=ssp_handles + [pi_handle],
        labels=ssp_labels + [PI_LABEL],
        handler_map={pi_handle: HandlerBandWithLine()},
        title=None,
        loc='upper left', bbox_to_anchor=(0.04, 0.96),
        fontsize=12, title_fontsize=14, frameon=True,
        facecolor='white', edgecolor='black', fancybox=False
    )
    legend.get_frame().set_linewidth(1.0)
    legend._legend_box.align = "left"

    plot_panel(ax_bulk, 'Bulk', 'Bulk')
    plot_panel(ax_cont, 'Container', 'Container')
    plot_panel(ax_gen, 'General', 'General')
    plot_panel(ax_tank, 'Tanker', 'Tanker')

    ax_bulk.set_ylim(0, 3)
    ax_gen.set_ylim(0, 3)
    y_ticks_dense = [0, 0.5, 1.0, 1.5, 2.0, 2.5, 3.0]
    ax_bulk.set_yticks(y_ticks_dense)
    ax_gen.set_yticks(y_ticks_dense)

    plt.setp(ax_bulk.get_xticklabels(), visible=False)
    plt.setp(ax_cont.get_xticklabels(), visible=False)
    plt.setp(ax_cont.get_yticklabels(), visible=False)
    plt.setp(ax_tank.get_yticklabels(), visible=False)

    ax_gen.tick_params(axis='x', rotation=45)
    ax_tank.tick_params(axis='x', rotation=45)

    ax_gen.set_xlabel('Year', fontsize=16, labelpad=8)
    ax_tank.set_xlabel('Year', fontsize=16, labelpad=8)

    plt.tight_layout()
    plt.subplots_adjust(bottom=0.15, wspace=0.25, hspace=0.3)

    plt.savefig(output_pdf, bbox_inches='tight')
    print(f"Success! Plot saved to {output_pdf}")
    plt.show()

if __name__ == "__main__":
    input_file = "../out/merged_predictions.csv"

    if os.path.exists(input_file):
        plot_data = load_and_prepare_data(input_file)
        if plot_data is not None:
            plot_data.to_csv('figure1_plot_data.csv', index=False)
            reproduce_ggplot_layout(plot_data)
    else:
        print(f"Please ensure '{input_file}' exists before running.")