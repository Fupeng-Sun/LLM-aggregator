"""Figure 6 (``fig:impact_regionI``): aggregator impact in Theorem ``thm:impact_aggregator``(ii), p_L <= p < p_R.

Panels
- (a) ``fig:impact_regionI_segmentation``, file ``figure/fig_pre/impact_G_3-way-2.pdf``: the utilities u_A, u_B, u_G
  at one price in [p_L, p_R). The aggregator turns the segmentation B-O-A into B-O-G-A; braces mark the gaps delta_a,
  delta_b, the API usage gain [mu_GO, mu_AG], the additional coverage [mu_GO, mu_AO] and A's subscription loss
  [mu_AG, mu_AO].
- (b) ``fig:impact_regionI_case1``, files ``figure/aggregator_impact/p_L_b_B/{w_line_data,phi_curve_data}.dat``: the
  lines w_A = p / rho_0 and w_A = Phi_A^L(p) / rho_0 on [p_L, p_R], for a case with p_L <= underline-p_A^L <= b_B.
- (c) ``fig:impact_regionI_case2``, files ``figure/aggregator_impact/b_B_p_R/...``: the same, b_B <= underline-p_A^L
  <= p_R.

Panels (b) and (c) are drawn by pgfplots in main.tex from the .dat files (columns ``p w`` and ``p phi``); main.tex
hard-codes their x ticks, which are the model values listed in ``ThresholdPanel.ticks``.

Parameters, all chosen for illustration; (a_A, b_A, a_B, b_B, kappa):
- (a): (1.35, 0.7, 0.8, 1.2, 0.1), p = 1.15 (the earlier Mathematica notebook, not in this repository,
  section "Impact of the aggregator").
- (b): (4.330640648432909, 0.8601310965575639, 2.4345298541715636, 2.610262960064678, 0.8511680197810718),
  rho_0 = 1.4, 3000 prices (the earlier notebook aggregator_plot.ipynb, cell 0); a case of Lemma
  ``lem:aggregator_impact_tech``(iv)(d), middle branch.
- (c): (12.586905537485833, 0.6913497310480877, 1.5730224509019064, 5.303092884086805, 1.2731709208970366),
  rho_0 = 1.4, 2000 prices (same notebook, cell 1); a case of Lemma ``lem:aggregator_impact_tech``(iv)(b).

The panel specifications, the segmentation drawing, the .dat output and the checks shared with Figures 7 and 8 are in
``figures.impact_panels``.
"""

from pathlib import Path

import numpy as np

from figures.impact_panels import (
    Brace,
    Callout,
    SegmentationPanel,
    ThresholdPanel,
    check_panel,
    check_segmentation,
    prices,
    save_segmentation,
    threshold_curve,
    write_panel_data,
)
from llmagg import impact
from llmagg.cli import run_figure
from llmagg.impact import Params
from llmagg.plotting import use_style

# ---------------------------------------------------------------------------------------------------------------
# Figure 6
# ---------------------------------------------------------------------------------------------------------------

SEGMENTATION_PARAMS = Params(a_A=1.35, b_A=0.7, a_B=0.8, b_B=1.2, kappa=0.1)
SEGMENTATION_PRICE = 1.15


def _segmentation_panel() -> SegmentationPanel:
    m, p = SEGMENTATION_PARAMS, SEGMENTATION_PRICE
    mu_BO, mu_GO, mu_AO, mu_AG = impact.mu_BO(m, p), impact.mu_GO(m, p), impact.mu_AO(m, p), impact.mu_AG(m)
    return SegmentationPanel(
        name="impact_G_3-way-2",
        label="fig:impact_regionI_segmentation",
        params=m,
        p=p,
        region="ii",
        segmentation="BOGA",
        split_points=(mu_BO, mu_GO, mu_AG),
        without="BOA",
        without_split_points=(mu_BO, mu_AO),
        # layout of the original panel (630 x 617 bp, frame 59.8-569.8 x 0.5-536 bp) at the printed width, with
        # 2 pt wider side margins for the brace labels
        height=147.2,
        margins=(16.5, 16.5, 0.1, 19.3),
        ylim=(-0.55, 0.5),
        segment_labels={
            "B": (mu_BO / 2, 0.05),
            "O": ((mu_BO + mu_GO) / 2, 0.05),
            "G": ((mu_GO + mu_AG) / 2, 0.05),
            "A": ((mu_AG + 1) / 2, 0.05),
        },
        dashed=((mu_AG, "G"),),
        marks=(mu_BO, mu_GO, mu_AO),
        callouts=(
            Callout("A", (0.843, 0.369), (0.852, 0.317), (0.852, 0.303), 0.92),
            Callout("G", (0.349, 0.270), (0.351, 0.223), (0.351, 0.205), 0.445),
            Callout("B", (0.135, 0.266), (0.105, 0.232), (0.07, 0.214), 0.032),
        ),
        braces=(
            Brace((0, m.b_B - p), (0, m.b_A - p), 0.03, r"$\delta_b$", 0.02),
            Brace((1, m.a_B - p), (1, m.a_A - p), 0.03, r"$\delta_a$", 0.02),
            Brace((mu_GO, 0), (mu_AG, 0), 0.12, "API usage gain", 0.03),
            Brace((mu_AG, 0), (mu_AO, 0), 0.01, "subscription\nloss (A)", 0.07),
            Brace((mu_GO, 0), (mu_AO, 0), 0.01, "additional coverage", 0.03),
        ),
    )


SEGMENTATION = _segmentation_panel()

PANELS = (
    ThresholdPanel(
        case="p_L_b_B",
        label="fig:impact_regionI_case1",
        params=Params(4.330640648432909, 0.8601310965575639, 2.4345298541715636, 2.610262960064678, 0.8511680197810718),
        rho_0=1.4,
        provider="A",
        threshold="Phi_A_L",
        grid=("p_L", "p_R", 3000),
        ticks=("p_L", "p_A_L_switch", "b_B", "p_R"),
        lemma="(iv)(d): delta_a > delta_b, a_A + b_A <= 2 b_B, delta_b/delta_a < Delta_A/(Delta_A + Delta_B), "
        "kappa_A^dagger <= kappa <= kappa_A^ddagger => p_L <= underline-p_A^L <= min(p_R, b_B)",
        lemma_holds=lambda m: (
            m.delta_a > m.delta_b
            and m.a_A + m.b_A <= 2 * m.b_B
            and m.delta_b / m.delta_a < m.Delta_A / (m.Delta_A + m.Delta_B)
            and impact.kappa_A_dagger(m) <= m.kappa <= impact.kappa_A_ddagger(m)
        ),
    ),
    ThresholdPanel(
        case="b_B_p_R",
        label="fig:impact_regionI_case2",
        params=Params(
            12.586905537485833, 0.6913497310480877, 1.5730224509019064, 5.303092884086805, 1.2731709208970366
        ),
        rho_0=1.4,
        provider="A",
        threshold="Phi_A_L",
        grid=("p_L", "p_R", 2000),
        ticks=("p_L", "b_B", "p_A_L_switch", "p_R"),
        lemma="(iv)(b): delta_a > delta_b, a_A + b_A >= 2 b_B, kappa <= kappa_A^ddagger "
        "=> b_B <= underline-p_A^L <= p_R",
        lemma_holds=lambda m: (
            m.delta_a > m.delta_b and m.a_A + m.b_A >= 2 * m.b_B and m.kappa <= impact.kappa_A_ddagger(m)
        ),
    ),
)


def check() -> list[str]:
    """Check Figure 6 against Theorem thm:impact_aggregator(ii), Lemma lem:aggregator_impact_tech(iv) and the
    exact market model."""
    lines = check_segmentation(SEGMENTATION)
    for panel in PANELS:
        lines += check_panel(panel)
        if panel.threshold == "Phi_A_L":  # Prop. property_phi_AL(ii): monotone, sign of p_R + 2 (kappa - b_B)
            m = panel.params
            slope = impact.p_R(m) + 2 * (m.kappa - m.b_B)
            curve = threshold_curve(panel, prices(panel))
            assert np.all(np.sign(np.diff(curve)) == np.sign(slope))
            lines.append(f"{panel.label}: Phi_A^L is {'in' if slope > 0 else 'de'}creasing (Prop. property_phi_AL)")
    return lines


def make(outdir: Path) -> list[Path]:
    """Write impact_G_3-way-2.pdf and the .dat files of panels (b) and (c)."""
    use_style()
    files = [save_segmentation(SEGMENTATION, outdir)]
    for panel in PANELS:
        files += write_panel_data(panel, outdir)
    return files


if __name__ == "__main__":
    run_figure(make, check)
