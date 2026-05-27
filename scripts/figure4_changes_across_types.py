import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from figure_common import SSP_COLORS, HandlerBandWithLine, PI_LABEL_PROPAGATED, make_ssp_line_handles
import matplotlib.lines as mlines
import matplotlib.patches as mpatches

TARGET_POLICY = 'Policy 2'

def reproduce_vessel_plot_with_bounds():
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

    scenario_to_policy = {
        0: 'Policy 1',
        3: 'Policy 2',
        4: 'Policy 3',
        5: 'Policy 4'
    }
    df_vessels['Policy'] = df_vessels['Scenario'].map(scenario_to_policy)
    df_mapped = df_vessels.dropna(subset=['Policy']).copy()

    tanker_types = ["Chemical", "Liquified-Gas", "Oil"]
    df_mapped['VesselType'] = df_mapped['VesselType'].apply(
        lambda x: 'Tanker' if x in tanker_types else x
    )

    print("Aggregating risk sum by VesselType...")
    risk_sum_vessel = df_mapped.groupby(['Year', 'Policy', 'SSP', 'VesselType'])[['prob', 'prob_lower', 'prob_upper']].sum().reset_index()

    baseline_df = risk_sum_vessel[
        (risk_sum_vessel['Year'] == 2018) &
        (risk_sum_vessel['Policy'] == 'Policy 1') &
        (risk_sum_vessel['SSP'] == 'SSP1')
        ][['VesselType', 'prob']].rename(columns={'prob': 'Baseline_2018'})

    risk_sum_vessel = risk_sum_vessel.merge(baseline_df, on='VesselType', how='left')

    risk_sum_vessel['risk_change'] = risk_sum_vessel['prob'] / risk_sum_vessel['Baseline_2018']
    risk_sum_vessel['lower'] = risk_sum_vessel['prob_lower'] / risk_sum_vessel['Baseline_2018']
    risk_sum_vessel['upper'] = risk_sum_vessel['prob_upper'] / risk_sum_vessel['Baseline_2018']

    print(f"Generating plot for {TARGET_POLICY}...")

    plot_data = risk_sum_vessel[risk_sum_vessel['Policy'] == TARGET_POLICY].copy()

    if plot_data.empty:
        print(f"Error: No data found for {TARGET_POLICY}.")
        return

    plt.rcParams['font.family'] = 'sans-serif'
    plt.rcParams['font.sans-serif'] = ['Arial', 'Helvetica', 'DejaVu Sans']

    sns.set_theme(style="ticks", rc={"axes.edgecolor": ".15", "axes.linewidth": 1.0})

    ssp_colors = SSP_COLORS

    g = sns.FacetGrid(plot_data, col="VesselType", col_wrap=2, height=3.5, aspect=1.6, sharey=True)

    def plot_confidence_intervals(data, **kwargs):
        ax = plt.gca()
        data = data.sort_values('SSP')

        ax.axhline(y=1, color="black", linestyle="--", linewidth=1.2, zorder=2)

        for ssp, ssp_data in data.groupby('SSP'):
            ssp_data = ssp_data.sort_values('Year')
            x = ssp_data['Year']
            y_mean = ssp_data['risk_change']
            color = ssp_colors.get(ssp, 'black')

            ax.fill_between(x, ssp_data['lower'], ssp_data['upper'],
                            color=color, alpha=0.15, zorder=4, edgecolor='none')
            ax.plot(x, y_mean, color=color, linewidth=1.8, zorder=5)

    g.map_dataframe(plot_confidence_intervals)

    g.set_titles("{col_name}", size=15, pad=12)

    y_label = g.fig.text(0.08, 0.5, f"Risk Change under {TARGET_POLICY} (Ratio)",
                         va='center', ha='center', rotation=90, fontsize=16)
    x_label = g.fig.text(0.5, 0.02, "Year",
                         va='center', ha='center', fontsize=16)

    y_ticks = [0, 1, 3, 6, 9]

    for ax in g.axes.flatten():
        ax.set_ylabel('')
        ax.set_xlabel('')

        ax.set_xticks([2020, 2040, 2060, 2080, 2100])

        ax.tick_params(axis='x', bottom=True, direction='out', length=6, width=1.0, labelsize=13, rotation=45)
        ax.set_yticks(y_ticks)
        ax.tick_params(axis='y', left=True, direction='out', length=6, width=1.0, labelsize=13)

        ax.grid(False)
        for spine in ax.spines.values():
            spine.set_visible(True)
            spine.set_color('black')
            spine.set_linewidth(1.0)

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
        loc='center left', bbox_to_anchor=(0.68, 0.72),
        title="SSP", fontsize=12, title_fontsize=14, frameon=False,
        handlelength=1.5, handleheight=1.0
    )
    first_legend._legend_box.align = "left"
    g.fig.add_artist(first_legend)

    ref_handles = [
        mlines.Line2D([], [], color='black', linestyle='--', linewidth=1.2,
                      label='2018 Baseline risk (=1)')
    ]
    second_legend = g.fig.legend(
        handles=ref_handles,
        loc='center left', bbox_to_anchor=(0.68, 0.25),
        title="Reference", fontsize=12, title_fontsize=14, frameon=False,
        handlelength=1.5
    )
    second_legend._legend_box.align = "left"

    output_pdf = f"figure4_Relative_risk_changes_from_the_2018_baseline_across_ship_types.pdf"

    plt.savefig(output_pdf, bbox_inches='tight')
    print(f"Success! High-Res Plot saved to: {output_pdf}")
    plt.show()

if __name__ == "__main__":
    reproduce_vessel_plot_with_bounds()