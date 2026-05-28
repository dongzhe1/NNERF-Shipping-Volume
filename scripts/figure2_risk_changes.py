import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from figure_common import SSP_COLORS, HandlerBandWithLine, PI_LABEL_PROPAGATED, make_ssp_line_handles, scenario_to_policy
import matplotlib.lines as mlines
from matplotlib.patches import Patch
import matplotlib.patches as mpatches


def reproduce_macro_policy_plot():
    data_path = "../data/ballast/p_country/ballast_country_aggregated.csv"
    print(f"Loading data from {data_path}...")

    try:
        df = pd.read_csv(data_path)
    except FileNotFoundError:
        print(f"Error: Could not find file {data_path}")
        return

    if 'VesselType' in df.columns:
        df_all = df[df['VesselType'] == 'All'].copy()
    else:
        df_all = df.copy()

    print("Aggregating global risk sum...")
    grouped_risk = df_all.groupby(['Year', 'Scenario', 'SSP'])[['prob', 'prob_lower', 'prob_upper']].sum().reset_index()

    grouped_risk['Policy'] = grouped_risk['Scenario'].map(scenario_to_policy)
    risk_mapped = grouped_risk.dropna(subset=['Policy']).copy()

    try:
        baseline_val = risk_mapped[
            (risk_mapped['Year'] == 2018) &
            (risk_mapped['Scenario'] == 0) &
            (risk_mapped['SSP'] == 'SSP1')
            ]['prob'].values[0]
        print(f"Extracted global baseline value: {baseline_val:.4f}")
    except IndexError:
        print("Warning: Global baseline value not found. Using mean.")
        baseline_val = risk_mapped[risk_mapped['Year'] == 2018]['prob'].mean()

    risk_mapped['risk_change'] = risk_mapped['prob'] / baseline_val
    risk_mapped['lower'] = risk_mapped['prob_lower'] / baseline_val
    risk_mapped['upper'] = risk_mapped['prob_upper'] / baseline_val

    risk_mapped.to_csv('figure2_plot_data.csv', index=False)

    print("Generating plot...")
    plt.rcParams['font.family'] = 'sans-serif'
    plt.rcParams['font.sans-serif'] = ['Arial', 'Helvetica', 'DejaVu Sans']

    sns.set_theme(style="ticks", rc={"axes.edgecolor": ".15", "axes.linewidth": 1.0})

    ssp_colors = SSP_COLORS

    g = sns.FacetGrid(risk_mapped, col="Policy", col_wrap=2, height=3.5, aspect=1.6, sharey=False)

    def plot_confidence_intervals(data, **kwargs):
        ax = plt.gca()
        data = data.sort_values('SSP')

        ax.axhline(y=1, color="black", linestyle="--", linewidth=1.2, zorder=2)

        mean_2018_data = data[data['Year'] == 2018]
        if not mean_2018_data.empty:
            y_val = mean_2018_data['risk_change'].values[0]
            ax.scatter(2018, y_val, color='black', marker='o', s=20, zorder=3)

            policy_name = data['Policy'].iloc[0]
            a_vals = {'Policy 1': '1', 'Policy 2': '0.12', 'Policy 3': '0.07', 'Policy 4': '0.01'}

            if policy_name in a_vals:
                ax.annotate(f"A: {a_vals[policy_name]}", xy=(2018, y_val),
                            xytext=(5, 6), textcoords='offset points',
                            fontsize=13, color='black', zorder=6)

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

    y_label = g.fig.text(0.08, 0.5, "Risk Change Relative to 2018 (Ratio)",
                         va='center', ha='center', rotation=90, fontsize=16)
    x_label = g.fig.text(0.5, 0.02, "Year",
                         va='center', ha='center', fontsize=16)

    for ax in g.axes.flatten():
        ax.set_ylabel('')
        ax.set_xlabel('')
        ax.set_xticks([2020, 2040, 2060, 2080, 2100])

        ax.tick_params(axis='x', bottom=True, direction='out', length=6, width=1.0, labelsize=13, rotation=45)
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

    ssp_line_handles = make_ssp_line_handles(risk_mapped['SSP'].unique())

    first_legend = g.fig.legend(
        handles=ssp_line_handles + [pi_handle],
        labels=[h.get_label() for h in ssp_line_handles] + [
            PI_LABEL_PROPAGATED],
        handler_map={pi_handle: HandlerBandWithLine()},
        loc='center left', bbox_to_anchor=(0.68, 0.72),
        title=None, fontsize=11, title_fontsize=13, frameon=False,
        handlelength=1.5
    )
    first_legend._legend_box.align = "left"
    g.fig.add_artist(first_legend)

    ref_handles = [
        mlines.Line2D([], [], color='black', marker='o', linestyle='None',
                      markersize=5, label='2018 regulated risk (A)'),
        mlines.Line2D([], [], color='black', linestyle='--', linewidth=1.2,
                      label='2018 baseline risk (=1)')
    ]
    second_legend = g.fig.legend(
        handles=ref_handles,
        loc='center left', bbox_to_anchor=(0.68, 0.25),
        fontsize=11, frameon=False,
        title="Reference",
        handlelength=1.5, labelspacing=1.5
    )
    second_legend._legend_box.align = "left"

    output_pdf = "figure2_Projected_global_NIS_risk_changes.pdf"

    plt.savefig(output_pdf, bbox_inches='tight')
    plt.show()

if __name__ == "__main__":
    reproduce_macro_policy_plot()