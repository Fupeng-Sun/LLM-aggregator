"""Figure 4, fig:market_evolution_without_aggregator: market segmentation without an active aggregator.

Illustrates Proposition 1 (prop:market_segmentation_without_aggregator) when Delta_B > 0, at three falling prices.
Each panel plots u_A(mu) and u_B(mu), shades the area under the chosen utility, marks the cutoffs (a short tick at a
participation cutoff on u = 0, a dashed line at the competitive cutoff mu_AB) and writes the segment letters:
    (a) fig:seg_noagg_OA   O-A,   case (b), split at mu_AO = 0.75
                                                        -> figure/fig_pre/segmentation_2-way_O-A.pdf
    (b) fig:seg_noagg_BOA  B-O-A, case (c), split at mu_BO = 0.2333, mu_AO = 0.55
                                                        -> figure/fig_pre/segmentation_2-way_B-O-A.pdf
    (c) fig:seg_noagg_BA   B-A,   case (d), split at mu_AB = 4/9
                                                        -> figure/fig_pre/segmentation_2-way.pdf
Each panel is included at width 0.28 textwidth = 131.5 pt (textwidth 469.755 pt); the height, 122.8 pt, keeps the
aspect ratio of the previous (Mathematica) files, 600 x 560.

Parameters (earlier Mathematica notebook, not in this repository: three Modules without an Export line; the
values agree with the tick-label positions and cutoff marks measured in the previous PDFs):
    a_A = 1.4, b_A = 0.8, a_B = 0.9, b_B = 1.2   qualities; b_B = 1.2 here, while Figures 3(a) and 5 use b_B = 1.25
                                                 (the paper does not state either value)
    p = 1.25, 1.13, 0.5                          prices of panels (a), (b), (c); the price ranges of Proposition 1
                                                 are (1.2, 1.4], (1.0667, 1.2] and p <= 1.0667
    kappa                                        not drawn: Proposition 1 assumes kappa > kappa_bar = 2/9, and the
                                                 segmentation does not depend on kappa (the notebook's 0.15 < 2/9 is
                                                 unused in these panels)
    u in [-0.55, 0.5], [-0.55, 0.5], [-0.15, 0.95]   frame ranges (notebook PlotRange; the lower end of (c) is
                                                 lowered from -0.1 to make room for the segment letters)
Callout positions are chosen for illustration.
"""

from dataclasses import replace
from pathlib import Path

import matplotlib.pyplot as plt

from figures.segmentation_panels import (
    TEXT_WIDTH_PT,
    PanelLayout,
    SegmentationPanel,
    check_segmentation_panel,
    draw_segmentation_panel,
)
from llmagg.cli import run_figure
from llmagg.io import save_pdf
from llmagg.plotting import FONT_SIZE
from llmagg.segmentation import Qualities

QUALITIES = Qualities(a_A=1.4, b_A=0.8, a_B=0.9, b_B=1.2)
PARTIAL = (-0.55, 0.5)  # frame range of the partially covered panels
FULL = (-0.15, 0.95)  # frame range of the fully covered panel

_PANEL = SegmentationPanel(Path(), QUALITIES, None, 0.0, "", (), PARTIAL, {})
PANELS = {
    "(a)": replace(
        _PANEL,
        file=Path("figure/fig_pre/segmentation_2-way_O-A.pdf"),
        price=1.25,
        case="b",
        order=("O", "A"),
        callouts={"A": (0.88, (0.72, 0.30)), "B": (0.27, (0.13, -0.24))},
    ),
    "(b)": replace(
        _PANEL,
        file=Path("figure/fig_pre/segmentation_2-way_B-O-A.pdf"),
        price=1.13,
        case="c",
        order=("B", "O", "A"),
        callouts={"A": (0.74, (0.66, 0.30)), "B": (0.06, (0.24, 0.15))},
    ),
    "(c)": replace(
        _PANEL,
        file=Path("figure/fig_pre/segmentation_2-way.pdf"),
        price=0.5,
        case="d",
        order=("B", "A"),
        u_range=FULL,
        callouts={"A": (0.82, (0.62, 0.88)), "B": (0.12, (0.24, 0.86))},
    ),
}

LAYOUT = PanelLayout(
    width_pt=0.28 * TEXT_WIDTH_PT,  # \begin{subfigure}{0.28\textwidth}, \includegraphics[width=\linewidth]
    height_pt=0.28 * TEXT_WIDTH_PT * 560 / 600,  # aspect ratio of the previous files
    margins_pt=(22.0, 22.0, 21.0, 3.0),
    font_size=FONT_SIZE - 2,
    mu_ticks=(0.5, 0.1),  # the panels are too narrow for labels every 0.2
)


def make(outdir: Path) -> list[Path]:
    """Write the three panels of Figure 4 under ``outdir``."""
    paths = []
    for panel in PANELS.values():
        fig = draw_segmentation_panel(panel, LAYOUT)
        paths.append(save_pdf(fig, Path(outdir) / panel.file))
        plt.close(fig)
    return paths


def check() -> list[str]:
    """Check every panel against Proposition 1 and the exact segmentation."""
    lines = []
    for name, panel in PANELS.items():
        lines += check_segmentation_panel(panel, f"Figure 4{name}")
    return lines


if __name__ == "__main__":
    run_figure(make, check)
