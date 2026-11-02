"""Figure 8 (``fig:impact_low_price``): aggregator impact in Theorem ``thm:impact_aggregator``(iv), p < p*.

Panels
- (a) ``fig:impact_low_price_segmentation``, file ``figure/fig_pre/impact_G_2-way.pdf``: the utilities u_A, u_B, u_G
  at one price below p*. The market is fully covered with and without the aggregator, which turns B-A into B-G-A;
  braces mark the gaps delta_a, delta_b, the API usage gain [mu_GB, mu_AG] and the subscription losses of B
  [mu_AB, mu_GB] and of A [mu_AG, mu_AB].
- (b) ``fig:impact_low_price_A``, files
  ``figure/aggregator_impact/pstar_less_lower_A/{w_line_data,phi_curve_data}.dat``: w_A = p / rho_0 and
  w_A = Psi_A p / rho_0 on [0, p*], for delta_a <= delta_b.
- (c) ``fig:impact_low_price_B``, files ``figure/aggregator_impact/pstar_less_lower_B/...``: w_B = p / rho_0 and
  w_B = Psi_B p / rho_0 on [0, p*], for delta_a >= delta_b.

Panels (b) and (c) are drawn by pgfplots in main.tex from the .dat files (columns ``p w`` and ``p phi``), which
hard-codes their p* tick and the domain [0, p*]: p* = 22/5 = 4.4 in (b) and 13/3 in (c).

Parameters, all chosen for illustration; (a_A, b_A, a_B, b_B, kappa):
- (a): (1.4, 0.8, 0.9, 1.25, 0.1), p = 0.5 (earlier Mathematica notebook, "Impact of the aggregator"). Panel
  (a) only illustrates the segmentation; its gaps (delta_a = 0.5 > delta_b = 0.45) are not those of panel (b).
- (b): (6, 2, 4, 5, 0.8), rho_0 = 2, 400 prices on [0, p*] (the earlier notebook aggregator_plot.ipynb, cell 9).
- (c): (7, 3, 3, 5, 1), rho_0 = 2, 400 prices on [0, p*] (same notebook, cell 10).
"""

from pathlib import Path

from figures.impact_panels import (
    Brace,
    Callout,
    SegmentationPanel,
    ThresholdPanel,
    check_panel,
    check_segmentation,
    save_segmentation,
    write_panel_data,
)
from llmagg import impact
from llmagg.cli import run_figure
from llmagg.impact import Params
from llmagg.plotting import use_style

SEGMENTATION_PARAMS = Params(a_A=1.4, b_A=0.8, a_B=0.9, b_B=1.25, kappa=0.1)
SEGMENTATION_PRICE = 0.5


def _segmentation_panel() -> SegmentationPanel:
    m, p = SEGMENTATION_PARAMS, SEGMENTATION_PRICE
    mu_AB, mu_GB, mu_AG = impact.mu_AB(m), impact.mu_GB(m), impact.mu_AG(m)
    return SegmentationPanel(
        name="impact_G_2-way",
        label="fig:impact_low_price_segmentation",
        params=m,
        p=p,
        region="iv",
        segmentation="BGA",
        split_points=(mu_GB, mu_AG),
        without="BA",
        without_split_points=(mu_AB,),
        # size of the original panel at the printed width (690 x 698 bp after the 30 bp main.tex trims off its
        # bottom), frame as high as the original's (89.8-600 x 30.5-617.3 bp) but 4.6 pt higher up, to make room
        # for the tick labels
        height=152.1,
        margins=(19.6, 19.6, 2.0, 22.2),
        ylim=(-0.2, 0.95),
        segment_labels={"B": (mu_GB / 2, -0.05), "G": (0.30, 0.30), "A": ((mu_AG + 1) / 2, -0.05)},
        dashed=((mu_AG, "G"), (mu_GB, "G"), (mu_AB, "A")),
        marks=(),
        callouts=(
            Callout("A", (0.793, 0.864), (0.805, 0.828), (0.805, 0.81), 0.80),
            Callout("G", (0.492, 0.798), (0.565, 0.80), (0.60, 0.80), 0.69),
            Callout("B", (0.257, 0.746), (0.18, 0.746), (0.16, 0.746), 0.12),
        ),
        braces=(
            Brace((0, m.b_B - p), (0, m.b_A - p), 0.035, r"$\delta_b$", 0.025),
            Brace((1, m.a_B - p), (1, m.a_A - p), 0.035, r"$\delta_a$", 0.025),
            Brace((mu_GB, 0), (mu_AG, 0), 0.01, "API usage gain", 0.03),
            # the two loss labels are wider than their braces: A's sits higher, above B's
            Brace((mu_AG, 0), (mu_AB, 0), 0.01, "subscription\nloss (A)", 0.26),
            Brace((mu_AB, 0), (mu_GB, 0), 0.01, "subscription\nloss (B)", 0.07),
        ),
    )


SEGMENTATION = _segmentation_panel()

PANELS = (
    ThresholdPanel(
        case="pstar_less_lower_A",
        label="fig:impact_low_price_A",
        params=Params(6.0, 2.0, 4.0, 5.0, 0.8),
        rho_0=2.0,
        provider="A",
        threshold="Psi",
        grid=(0.0, "p_star", 400),
        ticks=("p_star",),
        lemma="(i): delta_a <= delta_b => Psi_A <= 1 <= Psi_B",
        lemma_holds=lambda m: m.delta_a <= m.delta_b and impact.Psi_A(m) <= 1 <= impact.Psi_B(m),
    ),
    ThresholdPanel(
        case="pstar_less_lower_B",
        label="fig:impact_low_price_B",
        params=Params(7.0, 3.0, 3.0, 5.0, 1.0),
        rho_0=2.0,
        provider="B",
        threshold="Psi",
        grid=(0.0, "p_star", 400),
        ticks=("p_star",),
        lemma="(i): delta_a >= delta_b => Psi_B <= 1 <= Psi_A",
        lemma_holds=lambda m: m.delta_a >= m.delta_b and impact.Psi_B(m) <= 1 <= impact.Psi_A(m),
    ),
)


def check() -> list[str]:
    """Check Figure 8 against Theorem thm:impact_aggregator(iv), Lemma lem:aggregator_impact_tech(i) and the exact
    market model."""
    lines = check_segmentation(SEGMENTATION)
    for panel in PANELS:
        lines += check_panel(panel)
    return lines


def make(outdir: Path) -> list[Path]:
    """Write impact_G_2-way.pdf and the .dat files of panels (b) and (c)."""
    use_style()
    files = [save_segmentation(SEGMENTATION, outdir)]
    for panel in PANELS:
        files += write_panel_data(panel, outdir)
    return files


if __name__ == "__main__":
    run_figure(make, check)
