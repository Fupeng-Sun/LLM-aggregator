"""Figure 3(a), fig:illustrative_revenue_profiles: illustrative utility profiles.

Panel (a) of Figure 3 (fig:revenue_and_segmentation; panel (b) is drawn in TikZ in main.tex). It plots the utilities
u_A(mu), u_B(mu) and u_G(mu) of provider A, provider B and the aggregator G against the user type mu. The frame ticks
mark the endpoints u_X(0) and u_X(1) (b_A-p, b_B-p, b_B-kappa-p on the left; a_A-p, a_B-p, a_A-kappa-p on the
right). The crossings u_G = u_B at mu_GB = 0.3 and u_A = u_G at mu_AG = 2/3 split the market into B-G-A
(Theorem 1, case (f)); unlike Figures 4 and 5, the panel does not mark the segmentation.

Produces figure/fig_pre/utility_functions.pdf, included at width 0.4 textwidth = 187.9 pt (textwidth 469.755 pt);
the height, 140.8 pt, keeps the aspect ratio of the previous (Mathematica) file, 630 x 472.

Parameters (earlier Mathematica notebook, not in this repository: a Module without an Export line; the values agree
with the tick-label positions measured in the previous PDF):
    a_A = 1.4, b_A = 0.8, a_B = 0.9, b_B = 1.25   qualities, shared with Figure 5(b)-(d) (Figure 4 uses b_B = 1.2)
    kappa = 0.15                                  aggregator friction, shared with Figure 5(b)-(d); kappa_bar = 0.2368
    p = 0.5                                       price, from the notebook; the previous PDF had no 0 tick, so it fixed
                                                  only differences of utilities, which do not depend on p
    u in [0.1, 1.0]                               frame range of the notebook (PlotRange)

The drawing and checking code shared with Figures 4 and 5 is in ``figures.segmentation_panels``.
"""

from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.figure import Figure

from figures.segmentation_panels import (
    KAPPA,
    QUALITIES,
    TEXT_WIDTH_PT,
    TOL,
    Callouts,
    PanelLayout,
    check_frame,
    draw_utility_panel,
    new_panel_figure,
)
from llmagg.cli import run_figure
from llmagg.io import save_pdf
from llmagg.plotting import FONT_SIZE, use_style
from llmagg.segmentation import (
    cutoffs,
    regime_with_aggregator,
    segments,
    utility_lines,
)

PRICE = 0.5
U_RANGE = (0.1, 1.0)
FILE = Path("figure/fig_pre/utility_functions.pdf")

# ---------------------------------------------------------------------------------------------------------------------
# Figure 3(a)
# ---------------------------------------------------------------------------------------------------------------------
LAYOUT = PanelLayout(
    width_pt=0.4 * TEXT_WIDTH_PT,  # \begin{subfigure}{0.4\textwidth}, \includegraphics[width=\linewidth]
    height_pt=0.4 * TEXT_WIDTH_PT * 472 / 630,  # aspect ratio of the previous file
    margins_pt=(34.0, 36.0, 24.0, 4.0),
    font_size=FONT_SIZE - 1,
    mu_ticks=(0.2, 0.05),
)
CALLOUTS: Callouts = {"A": (0.80, (0.80, 0.90)), "B": (0.12, (0.27, 0.76)), "G": (0.70, (0.70, 0.60))}


def draw() -> Figure:
    """Figure 3(a) as a matplotlib figure."""
    use_style()
    fig, ax = new_panel_figure(LAYOUT)
    draw_utility_panel(ax, QUALITIES, PRICE, KAPPA, U_RANGE, CALLOUTS, LAYOUT)
    return fig


def make(outdir: Path) -> list[Path]:
    """Write figure/fig_pre/utility_functions.pdf under ``outdir``."""
    fig = draw()
    path = save_pdf(fig, Path(outdir) / FILE)
    plt.close(fig)
    return [path]


def check() -> list[str]:
    """Check the panel against Theorem 1 and Eq. (pairwise_cutoffs)."""
    q, p, kappa = QUALITIES, PRICE, KAPPA
    regime = regime_with_aggregator(q, p, kappa)
    assert (regime.case, regime.order) == ("f", ("B", "G", "A"))
    segs = segments(q, p, kappa)
    assert tuple(s.option for s in segs) == regime.order
    mu = cutoffs(q, p, kappa)
    for seg, split in zip(segs[:-1], regime.split_points, strict=True):
        assert abs(seg.right - mu[split]) < TOL
    lines = utility_lines(q, p, kappa)
    for x, y, name in (("G", "B", "mu_GB"), ("A", "G", "mu_AG"), ("A", "B", "mu_AB")):
        crossing = (lines[y][0] - lines[x][0]) / (lines[x][1] - lines[y][1])
        assert abs(crossing - mu[name]) < TOL
    check_frame(q, p, kappa, U_RANGE, CALLOUTS)
    return [
        f"Theorem 1 case (f) at p = {p}: B-G-A, split at mu_GB = {mu['mu_GB']:.4g} and mu_AG = {mu['mu_AG']:.4g}; "
        "the exact segmentation agrees",
        f"the lines cross at mu_GB, mu_AG and mu_AB = {mu['mu_AB']:.4g} (Eq. (pairwise_cutoffs))",
        "frame ticks at b_A-p, b_B-p, b_B-kappa-p (left) and a_A-p, a_B-p, a_A-kappa-p (right), inside the frame",
    ]


if __name__ == "__main__":
    run_figure(make, check)
