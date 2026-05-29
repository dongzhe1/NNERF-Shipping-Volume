import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib.lines as mlines
from matplotlib.legend_handler import HandlerBase

SSP_COLORS = {
    'SSP1': '#F8766D', 'SSP2': '#A3A500',
    'SSP3': '#00BF7D', 'SSP4': '#00B0F6', 'SSP5': '#E76BF3'
}

scenario_to_policy = {
    0: 'Policy 1',
    3: 'Policy 2',
    4: 'Policy 3',
    5: 'Policy 4'
}

MACRO_PI_RATE = 0.105

class HandlerBandWithLine(HandlerBase):
    """Legend key: wider shaded band with a centre line."""
    def create_artists(self, legend, orig_handle,
                       xdescent, ydescent, width, height, fontsize, trans):
        band = mpatches.Rectangle(
            [xdescent, ydescent + height * 0.15],
            width, height * 0.7,
            facecolor='gray', alpha=0.3, edgecolor='none',
            transform=trans
        )
        line = plt.Line2D(
            [xdescent, xdescent + width],
            [ydescent + height / 2, ydescent + height / 2],
            color='gray', linewidth=1.5, transform=trans
        )
        return [band, line]

PI_LABEL = 'Mean ± 10.5% prediction interval'
PI_LABEL_PROPAGATED = 'Mean ± 10.5% PI'

def make_ssp_line_handles(ssps, linewidth=1.5):
    """Line2D handles for SSP legend, consistent with Figure 1 style."""
    return [
        mlines.Line2D([], [], color=SSP_COLORS[ssp], linewidth=linewidth, label=ssp)
        for ssp in sorted(ssps)
    ]