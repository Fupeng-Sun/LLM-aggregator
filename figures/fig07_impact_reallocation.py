"""Figure 7 (``fig:impact_regionII``): aggregator impact in Theorem ``thm:impact_aggregator``(iii), p* <= p < p_L.

Panels
- (a) ``fig:impact_regionII_segmentation``, file ``figure/fig_pre/impact_G_3-way-1.pdf``: the utilities u_A, u_B, u_G
  at one price in [p*, p_L). The aggregator turns B-O-A into B-G-A and covers the market; braces mark the gaps
  delta_a, delta_b, the API usage gain [mu_GB, mu_AG], the additional coverage [mu_BO, mu_AO] and the subscription
  losses of B [mu_BO, mu_GB] and of A [mu_AG, mu_AO].
- (b), (c) ``fig:impact_regionII_B_case1``, ``fig:impact_regionII_B_case2``, files
  ``figure/aggregator_impact/{pstar_p_L_B, pstar_less_B}/{w_line_data,phi_curve_data}.dat``: the lines
  w_B = p / rho_0 and w_B = Omega_B(p) / rho_0, for p* < underline-p_B <= p_L and for 0 <= underline-p_B < p*.
- (d), (e), (f) ``fig:impact_regionII_A_case1`` to ``_case3``, files
  ``figure/aggregator_impact/{pstar_less, pstar_p_L, p_L_greater}/...``: w_A = p / rho_0 and w_A = Omega_A(p) / rho_0,
  for underline-p_A <= p*, p* < underline-p_A < p_L and p_L <= underline-p_A < p_R.

Panels (b)-(f) are drawn by pgfplots in main.tex from the .dat files (columns ``p w`` and ``p phi``), which hard-codes
their x ticks (the model values in ``ThresholdPanel.ticks``). Omega_X is the threshold only on [p*, p_L), the region
main.tex shades; the data run over the grids below, which in (c)-(f) extend beyond that region, to show where Omega_X
crosses p (the switching price) and its zero p_R.

Parameters, all chosen for illustration; (a_A, b_A, a_B, b_B, kappa):
- (a): (1.4, 0.8, 0.9, 1.25, 0.08), p = 1.15 (earlier Mathematica notebook, "Impact of the aggregator").
- (b): (8, 1, 4, 6, 1), rho_0 = 1.1, 1000 prices on [p*, p_L]; Lemma ``lem:aggregator_impact_tech``(iii).
- (c): (9, 0.5, 2, 7, 0.1), rho_0 = 1.1, 1000 prices on [underline-p_B, p_L]; Lemma (iii).
- (d): (5, 2, 3.5, 4, 0.6), rho_0 = 1.5, 1200 prices on [underline-p_A, p_R]; Lemma (ii)(a).
- (e): (4, 1.5, 1, 3, 0.3), rho_0 = 1.5, 1000 prices on [p*, p_R]; Lemma (ii)(b), kappa <= kappa_A^dagger.
- (f): (5, 2.5, 1, 4, 0.8), rho_0 = 1.5, 1000 prices on [p*, p_R]; Lemma (ii)(b), kappa >= kappa_A^dagger.
(b)-(f) are cells 7, 8, 6, 3 and 5 of the earlier notebook aggregator_plot.ipynb.
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

SEGMENTATION_PARAMS = Params(a_A=1.4, b_A=0.8, a_B=0.9, b_B=1.25, kappa=0.08)
SEGMENTATION_PRICE = 1.15


def _lemma_ii_b(m: Params) -> bool:
    """Common conditions of Lemma lem:aggregator_impact_tech(ii)(b)."""
    return m.delta_b / m.delta_a < m.Delta_A / (m.Delta_A + m.Delta_B) and m.a_A + m.b_A <= 2 * m.b_B


def _segmentation_panel() -> SegmentationPanel:
    m, p = SEGMENTATION_PARAMS, SEGMENTATION_PRICE
    mu_BO, mu_AO, mu_GB, mu_AG = impact.mu_BO(m, p), impact.mu_AO(m, p), impact.mu_GB(m), impact.mu_AG(m)
    return SegmentationPanel(
        name="impact_G_3-way-1",
        label="fig:impact_regionII_segmentation",
        params=m,
        p=p,
        region="iii",
        segmentation="BGA",
        split_points=(mu_GB, mu_AG),
        without="BOA",
        without_split_points=(mu_BO, mu_AO),
        # layout of the original panel (690 x 677 bp, frame 89.8-600 x 30.5-566 bp) at the printed width
        height=147.5,
        margins=(19.6, 19.6, 6.6, 24.2),
        ylim=(-0.5, 0.55),
        segment_labels={"B": (mu_GB / 2, -0.05), "G": ((mu_GB + mu_AG) / 2, 0.05), "A": ((mu_AG + 1) / 2, -0.05)},
        dashed=((mu_AG, "G"), (mu_GB, "G")),
        marks=(mu_BO, mu_AO),
        callouts=(
            Callout("A", (0.840, 0.370), (0.849, 0.327), (0.849, 0.309), 0.905),
            Callout("G", (0.398, 0.313), (0.40, 0.271), (0.40, 0.257), 0.545),
            Callout("B", (0.135, 0.266), (0.105, 0.230), (0.072, 0.214), 0.035),
        ),
        braces=(
            Brace((0, m.b_B - p), (0, m.b_A - p), 0.06, r"$\delta_b$", 0.025),
            Brace((1, m.a_B - p), (1, m.a_A - p), 0.06, r"$\delta_a$", 0.025),
            Brace((mu_GB, 0), (mu_AG, 0), 0.12, "API usage gain", 0.03),
            Brace((mu_AG, 0), (mu_AO, 0), 0.01, "subscription\nloss (A)", 0.07),
            Brace((mu_BO, 0), (mu_GB, 0), 0.01, "subscription\nloss (B)", 0.07),
            Brace((mu_BO, 0), (mu_AO, 0), 0.01, "additional coverage", 0.03),
        ),
    )


SEGMENTATION = _segmentation_panel()

PANELS = (
    ThresholdPanel(
        case="pstar_p_L_B",
        label="fig:impact_regionII_B_case1",
        params=Params(8.0, 1.0, 4.0, 6.0, 1.0),
        rho_0=1.1,
        provider="B",
        threshold="Omega",
        grid=("p_star", "p_L", 1000),
        ticks=("p_star", "p_B_switch", "p_L"),
        lemma="(iii): delta_a <= delta_b => p* <= underline-p_B <= p_L",
        lemma_holds=lambda m: m.delta_a <= m.delta_b,
    ),
    ThresholdPanel(
        case="pstar_less_B",
        label="fig:impact_regionII_B_case2",
        params=Params(9.0, 0.5, 2.0, 7.0, 0.1),
        rho_0=1.1,
        provider="B",
        threshold="Omega",
        grid=("p_B_switch", "p_L", 1000),
        ticks=("p_B_switch", "p_star", "p_L"),
        lemma="(iii): delta_a >= delta_b => underline-p_B <= p*",
        lemma_holds=lambda m: m.delta_a >= m.delta_b,
    ),
    ThresholdPanel(
        case="pstar_less",
        label="fig:impact_regionII_A_case1",
        params=Params(5.0, 2.0, 3.5, 4.0, 0.6),
        rho_0=1.5,
        provider="A",
        threshold="Omega",
        grid=("p_A_switch", "p_R", 1200),
        ticks=("p_A_switch", "p_star", "p_L", "p_R"),
        lemma="(ii)(a): delta_a <= delta_b => underline-p_A <= p*",
        lemma_holds=lambda m: m.delta_a <= m.delta_b,
    ),
    ThresholdPanel(
        case="pstar_p_L",
        label="fig:impact_regionII_A_case2",
        params=Params(4.0, 1.5, 1.0, 3.0, 0.3),
        rho_0=1.5,
        provider="A",
        threshold="Omega",
        grid=("p_star", "p_R", 1000),
        ticks=("p_star", "p_A_switch", "p_L", "p_R"),
        lemma="(ii)(b): delta_b/delta_a < Delta_A/(Delta_A + Delta_B), a_A + b_A <= 2 b_B, kappa <= kappa_A^dagger "
        "=> p* <= underline-p_A <= p_L",
        lemma_holds=lambda m: _lemma_ii_b(m) and m.kappa <= impact.kappa_A_dagger(m),
    ),
    ThresholdPanel(
        case="p_L_greater",
        label="fig:impact_regionII_A_case3",
        params=Params(5.0, 2.5, 1.0, 4.0, 0.8),
        rho_0=1.5,
        provider="A",
        threshold="Omega",
        grid=("p_star", "p_R", 1000),
        ticks=("p_star", "p_L", "p_A_switch", "p_R"),
        lemma="(ii)(b): delta_b/delta_a < Delta_A/(Delta_A + Delta_B), a_A + b_A <= 2 b_B, kappa_A^dagger <= kappa "
        "<= kappa_bar => underline-p_A >= p_L",
        lemma_holds=lambda m: _lemma_ii_b(m) and impact.kappa_A_dagger(m) <= m.kappa <= impact.kappa_bar(m),
    ),
)


def check() -> list[str]:
    """Check Figure 7 against Theorem thm:impact_aggregator(iii), Lemma lem:aggregator_impact_tech(ii)-(iii) and
    the exact market model."""
    lines = check_segmentation(SEGMENTATION)
    for panel in PANELS:
        lines += check_panel(panel)
    return lines


def make(outdir: Path) -> list[Path]:
    """Write impact_G_3-way-1.pdf and the .dat files of panels (b)-(f)."""
    use_style()
    files = [save_segmentation(SEGMENTATION, outdir)]
    for panel in PANELS:
        files += write_panel_data(panel, outdir)
    return files


if __name__ == "__main__":
    run_figure(make, check)
