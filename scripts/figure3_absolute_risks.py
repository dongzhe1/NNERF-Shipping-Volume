import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from figure_common import SSP_COLORS, HandlerBandWithLine, PI_LABEL_PROPAGATED, make_ssp_line_handles, scenario_to_policy
from matplotlib.patches import Patch
import matplotlib.patches as mpatches

TARGET_POLICY = 'Policy 2'

def reproduce_vessel_absolute_plot():
    data_path = "../data/ballast/p_country/ballast_country_aggregated.csv"
    print(f"Loading data from {data_path}...")

    try:
        df = pd.read_csv(data_path)
    except FileNotFoundError:
        print(f"Error: Could not find file {data_path}")
        return

    if 'VesselType' in df.columns:
        df_vessels = df[df['VesselType'] != 'All'].copy()
    else:
        df_vessels = df.copy()

    df_vessels['Policy'] = df_vessels['Scenario'].map(scenario_to_policy)
    df_mapped = df_vessels.dropna(subset=['Policy']).copy()

    tanker_types = ["Chemical", "Liquified-Gas", "Oil"]
    df_mapped['VesselType'] = df_mapped['VesselType'].apply(
        lambda x: 'Tanker' if x in tanker_types else x
    )

    print("Aggregating absolute risk sum by VesselType...")
    risk_sum_vessel = df_mapped.groupby(['Year', 'Policy', 'SSP', 'VesselType'])[['prob', 'prob_lower', 'prob_upper']].sum().reset_index()

    print(f"Generating plot for {TARGET_POLICY}...")

    plot_data = risk_sum_vessel[risk_sum_vessel['Policy'] == TARGET_POLICY].copy()

    if plot_data.empty:
        print(f"Error: No data found for {TARGET_POLICY}.")
        return

    plot_data.to_csv('figure3_plot_data.csv', index=False)

    plt.rcParams['font.family'] = 'sans-serif'
    plt.rcParams['font.sans-serif'] = ['Arial', 'Helvetica', 'DejaVu Sans']
    sns.set_theme(style="ticks", rc={"axes.edgecolor": ".15", "axes.linewidth": 1.0})

    ssp_colors = SSP_COLORS

    g = sns.FacetGrid(plot_data, col="VesselType", col_wrap=2, height=3.5, aspect=1.5, sharey=True)

    def plot_confidence_intervals(data, **kwargs):
        ax = plt.gca()
        data = data.sort_values('SSP')

        for ssp, ssp_data in data.groupby('SSP'):
            ssp_data = ssp_data.sort_values('Year')
            x = ssp_data['Year']

            y_mean = ssp_data['prob']
            lower_bound = ssp_data['prob_lower']
            upper_bound = ssp_data['prob_upper']

            color = ssp_colors.get(ssp, 'black')

            ax.fill_between(x, lower_bound, upper_bound,
                            color=color, alpha=0.15, zorder=4, edgecolor='none')
            ax.plot(x, y_mean, color=color, linewidth=1.8, zorder=5)

    g.map_dataframe(plot_confidence_intervals)

    g.set_titles("{col_name}", size=15, pad=12)

    y_label = g.fig.text(0.07, 0.5, f"Projected Aggregated Risk under {TARGET_POLICY}",
                         va='center', ha='center', rotation=90, fontsize=16)
    x_label = g.fig.text(0.5, 0.02, "Year",
                         va='center', ha='center', fontsize=16)

    for ax in g.axes.flatten():
        ax.set_ylabel('')
        ax.set_xlabel('')
        ax.set_xticks([2018, 2040, 2060, 2080, 2100])

        ax.tick_params(axis='x', bottom=True, direction='out', length=6, width=1.0, labelsize=13, rotation=45)
        ax.tick_params(axis='y', left=True, direction='out', length=6, width=1.0, labelsize=13)

        ax.grid(False)
        for spine in ax.spines.values():
            spine.set_visible(True)
            spine.set_color('black')
            spine.set_linewidth(1.0)

    ssp_handles = [
        Patch(color=ssp_colors[ssp], label=ssp, alpha=0.9, edgecolor='none')
        for ssp in sorted(plot_data['SSP'].unique())
    ]

    plt.tight_layout()
    plt.subplots_adjust(right=0.66, bottom=0.15, left=0.14, wspace=0.35,
                        hspace=0.3)

    pi_handle = mpatches.Patch(facecolor='gray', alpha=0.3, edgecolor='none')

    ssp_line_handles = make_ssp_line_handles(plot_data['SSP'].unique())

    first_legend = g.fig.legend(
        handles=ssp_line_handles + [pi_handle],
        labels=[h.get_label() for h in ssp_line_handles] + [
            PI_LABEL_PROPAGATED],
        handler_map={pi_handle: HandlerBandWithLine()},
        loc='center left', bbox_to_anchor=(0.68, 0.25),
        title=None, fontsize=12, title_fontsize=14, frameon=False,
        handlelength=1.5, handleheight=1.0
    )
    first_legend._legend_box.align = "left"
    g.fig.add_artist(first_legend)

    output_pdf = f"figure3_Projected_aggregated_risks_across_ship_types.pdf"

    plt.savefig(output_pdf, bbox_inches='tight')
    print(f"Success! High-Res Plot saved to: {output_pdf}")
    plt.show()

if __name__ == "__main__":
    reproduce_vessel_absolute_plot()