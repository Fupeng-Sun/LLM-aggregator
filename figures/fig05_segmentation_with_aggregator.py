"""Figure 5, fig:market_evolution_with_aggregator: market segmentation with an active aggregator.

Illustrates cases (c)-(f) of Theorem 1 (thm:market_segmentation_with_aggregator) when Delta_B > 0. Each panel plots
u_A(mu), u_B(mu) and u_G(mu), shades the area under the chosen utility, marks the cutoffs (a short tick at a
participation cutoff on u = 0, a dashed line at a competitive cutoff) and writes the segment letters:
    (a) fig:seg_agg_OGA      O-G-A,   case (c), split at mu_GO = 0.4118, mu_AG = 0.6944
                                                        -> figure/fig_pre/segmentation_3-way_O-G-A.pdf
    (b) fig:seg_agg_BOA_fig  B-O-A,   case (d), split at mu_BO = 0.1143, mu_AO = 0.6833
                                                        -> figure/fig_pre/segmentation_3-way_B-O-A.pdf
    (c) fig:seg_agg_BOGA     B-O-G-A, case (e), split at mu_BO = 0.2286, mu_GO = 0.4667, mu_AG = 0.6667
                                                        -> figure/fig_pre/segmentation_3-way_B-O-G-A.pdf
    (d) fig:seg_agg_BGA      B-G-A,   case (f), split at mu_GB = 0.3, mu_AG = 0.6667
                                                        -> figure/fig_pre/segmentation_3-way.pdf
Each panel is included at width 0.243 textwidth = 114.2 pt (textwidth 469.755 pt, includegraphics at width=linewidth),
so that neighbouring panels do not overlap; the height, 105.2 pt, is the aspect ratio of the previous (Mathematica)
files, 316 x 260, stretched by 12% to fit the frame labels at 7 pt.

Parameters (earlier Mathematica notebook, not in this repository: four Modules, each followed by its
Export line; the values agree with the tick-label positions and cutoff marks measured in the previous PDFs):
    panels (b)-(d): a_A = 1.4, b_A = 0.8, a_B = 0.9, b_B = 1.25, kappa = 0.15   as in Figure 3(a); kappa_bar = 0.2368
    panel (a):      a_A = 1.5, b_A = 0.8, a_B = 0.9, b_B = 1.16, kappa = 0.11   a different market: with the
                    qualities of (b)-(d), p_R = 1.2 < b_B = 1.25, so case (c) O-G-A cannot occur (Theorem 1(c));
                    here p_R = 1.2861 > b_B = 1.16. The caption does not say so; the symbolic ticks hide it
    p = 1.19, 1.21, 1.17, 0.5   prices of panels (a)-(d); the price ranges of Theorem 1 are (1.16, 1.2861],
                                (1.2, 1.25], (1.145, 1.2] and p <= 1.145
    u in [-0.55, 0.5] for (a)-(c), [-0.15, 1.05] for (d)   frame ranges (notebook PlotRange; the range of (d) is
                                widened from [-0.1, 0.95] to make room for the segment letters and the callouts)
Callout positions are chosen for illustration.
"""

from dataclasses import replace
from pathlib import Path

import matplotlib.pyplot as plt

from figures.segmentation_panels import (
    KAPPA,
    QUALITIES,
    TEXT_WIDTH_PT,
    PanelLayout,
    SegmentationPanel,
    check_segmentation_panel,
    draw_segmentation_panel,
)
from llmagg.cli import run_figure
from llmagg.io import save_pdf
from llmagg.plotting import FONT_SIZE
from llmagg.segmentation import Qualities, price_thresholds

QUALITIES_OGA = Qualities(a_A=1.5, b_A=0.8, a_B=0.9, b_B=1.16)  # panel (a) only
KAPPA_OGA = 0.11  # panel (a) only
PARTIAL = (-0.55, 0.5)  # frame range of the partially covered panels
FULL = (-0.15, 1.05)  # frame range of the fully covered panel

_PANEL = SegmentationPanel(Path(), QUALITIES, KAPPA, 0.0, "", (), PARTIAL, {})
PANELS = {
    "(a)": replace(
        _PANEL,
        file=Path("figure/fig_pre/segmentation_3-way_O-G-A.pdf"),
        qualities=QUALITIES_OGA,
        kappa=KAPPA_OGA,
        price=1.19,
        case="c",
        order=("O", "G", "A"),
        callouts={"A": (0.80, (0.74, 0.34)), "B": (0.06, (0.13, 0.16)), "G": (0.51, (0.42, 0.16))},
    ),
    "(b)": replace(
        _PANEL,
        file=Path("figure/fig_pre/segmentation_3-way_B-O-A.pdf"),
        price=1.21,
        case="d",
        order=("B", "O", "A"),
        callouts={"A": (0.85, (0.74, 0.34)), "B": (0.05, (0.13, 0.20)), "G": (0.60, (0.40, 0.15))},
    ),
    "(c)": replace(
        _PANEL,
        file=Path("figure/fig_pre/segmentation_3-way_B-O-G-A.pdf"),
        price=1.17,
        case="e",
        order=("B", "O", "G", "A"),
        callouts={"A": (0.83, (0.68, 0.32)), "B": (0.05, (0.13, 0.24)), "G": (0.56, (0.40, 0.15))},
    ),
    "(d)": replace(
        _PANEL,
        file=Path("figure/fig_pre/segmentation_3-way.pdf"),
        price=0.5,
        case="f",
        order=("B", "G", "A"),
        u_range=FULL,
        callouts={"A": (0.90, (0.78, 0.99)), "B": (0.10, (0.16, 0.94)), "G": (0.50, (0.45, 0.88))},
    ),
}

LAYOUT = PanelLayout(
    width_pt=0.243 * TEXT_WIDTH_PT,  # \begin{subfigure}{0.243\textwidth}, \includegraphics[width=\linewidth]
    height_pt=0.243 * TEXT_WIDTH_PT * 260 / 316 * 1.12,  # aspect ratio of the previous files, slightly taller
    margins_pt=(27.0, 27.0, 19.0, 3.0),
    font_size=FONT_SIZE - 3,
    mu_ticks=(0.5, 0.1),  # the panels are too narrow for labels every 0.2
)


def make(outdir: Path) -> list[Path]:
    """Write the four panels of Figure 5 under ``outdir``."""
    paths = []
    for panel in PANELS.values():
        fig = draw_segmentation_panel(panel, LAYOUT)
        paths.append(save_pdf(fig, Path(outdir) / panel.file))
        plt.close(fig)
    return paths


def check() -> list[str]:
    """Check every panel against Theorem 1 and the exact segmentation, and why panel (a) needs another market."""
    lines = []
    for name, panel in PANELS.items():
        lines += check_segmentation_panel(panel, f"Figure 5{name}")
    shared = price_thresholds(QUALITIES, KAPPA)
    own = price_thresholds(QUALITIES_OGA, KAPPA_OGA)
    assert QUALITIES.Delta_B > 0 and shared["p_R"] < QUALITIES.b_B, "case (c) would be possible in the shared market"
    assert own["p_R"] > QUALITIES_OGA.b_B
    lines.append(
        f"Figure 5(a) needs another market: with the shared qualities p_R = {shared['p_R']:.4g} < b_B = "
        f"{QUALITIES.b_B}, so case (c) does not occur; in panel (a) p_R = {own['p_R']:.4g} > b_B = {QUALITIES_OGA.b_B}"
    )
    return lines


if __name__ == "__main__":
    run_figure(make, check)
