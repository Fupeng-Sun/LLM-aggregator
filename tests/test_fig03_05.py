"""Figures 3(a), 4 and 5 (utility profiles and market segmentation) and the formulas of llmagg.segmentation."""

import filecmp

import matplotlib.pyplot as plt
import numpy as np
import pytest

from figures import (
    fig03_utility_profiles,
    fig04_segmentation_without_aggregator,
    fig05_segmentation_with_aggregator,
    segmentation_panels,
)
from llmagg.segmentation import (
    Qualities,
    cutoffs,
    price_thresholds,
    regime_with_aggregator,
    regime_without_aggregator,
    segments,
)

MODULES = [fig03_utility_profiles, fig04_segmentation_without_aggregator, fig05_segmentation_with_aggregator]
EXPECTED_FILES = {
    fig03_utility_profiles: ["utility_functions.pdf"],
    fig04_segmentation_without_aggregator: [
        "segmentation_2-way_O-A.pdf",
        "segmentation_2-way_B-O-A.pdf",
        "segmentation_2-way.pdf",
    ],
    fig05_segmentation_with_aggregator: [
        "segmentation_3-way_O-G-A.pdf",
        "segmentation_3-way_B-O-A.pdf",
        "segmentation_3-way_B-O-G-A.pdf",
        "segmentation_3-way.pdf",
    ],
}

# markets for the comparison of Proposition 1 and Theorem 1 with the exact segmentation
MARKETS = {
    "Figure 5(b)-(d), Delta_B > 0, p_R < b_B": (Qualities(1.4, 0.8, 0.9, 1.25), 0.15),
    "Figure 5(a), Delta_B > 0, p_R > b_B": (Qualities(1.5, 0.8, 0.9, 1.16), 0.11),
    "Delta_B < 0": (Qualities(1.4, 0.8, 1.3, 1.2), 0.05),
}
PRICES = 0.0107 + 0.004 * np.arange(401)  # a grid that avoids the price thresholds of the markets above


@pytest.mark.parametrize("module", MODULES, ids=lambda m: m.__name__.split(".")[-1])
def test_check(module):
    lines = module.check()
    assert lines and all(isinstance(line, str) and line for line in lines)


@pytest.mark.parametrize("module", MODULES, ids=lambda m: m.__name__.split(".")[-1])
def test_make_is_deterministic(module, tmp_path):
    first = module.make(tmp_path / "first")
    second = module.make(tmp_path / "second")
    assert [p.name for p in first] == EXPECTED_FILES[module]
    for a, b in zip(first, second, strict=True):
        assert a.relative_to(tmp_path / "first").parts[:2] == ("figure", "fig_pre")
        assert filecmp.cmp(a, b, shallow=False), f"{a.name} differs between two runs"


def _figures():
    yield "Figure 3(a)", fig03_utility_profiles.draw()
    for module, number in ((fig04_segmentation_without_aggregator, 4), (fig05_segmentation_with_aggregator, 5)):
        for name, panel in module.PANELS.items():
            yield f"Figure {number}{name}", segmentation_panels.draw_segmentation_panel(panel, module.LAYOUT)


def test_everything_inside_page():
    for name, fig in _figures():
        drawn = fig.get_tightbbox(fig.canvas.get_renderer())  # inches; includes every visible label and leader
        width, height = fig.get_size_inches()
        tolerance = 0.5 / 72
        assert drawn.x0 >= -tolerance and drawn.x1 <= width + tolerance, f"{name}: content wider than the page"
        assert drawn.y0 >= -tolerance and drawn.y1 <= height + tolerance, f"{name}: content taller than the page"
        plt.close(fig)


@pytest.mark.parametrize("market", MARKETS, ids=list(MARKETS))
def test_theorem_1_matches_exact_segmentation(market):
    q, kappa = MARKETS[market]
    assert kappa <= price_thresholds(q, kappa)["kappa_bar"]
    seen = set()
    for p in PRICES:
        regime = regime_with_aggregator(q, p, kappa)
        segs = segments(q, p, kappa)
        assert tuple(s.option for s in segs) == regime.order, (p, regime.case)
        mu = cutoffs(q, p, kappa)
        for seg, split in zip(segs[:-1], regime.split_points, strict=True):
            assert seg.right == pytest.approx(mu[split], abs=1e-12)
        seen.add(regime.case)
    expected = {"a", "b", "e", "f"} | ({"d"} if q.Delta_B > 0 and price_thresholds(q, kappa)["p_R"] < q.b_B else {"c"})
    assert seen == expected


@pytest.mark.parametrize("market", MARKETS, ids=list(MARKETS))
def test_proposition_1_matches_exact_segmentation(market):
    q, _ = MARKETS[market]
    kappa_above = 1.5 * price_thresholds(q, 0.0)["kappa_bar"]  # Proposition 1 assumes kappa > kappa_bar
    seen = set()
    for p in PRICES:
        regime = regime_without_aggregator(q, p)
        segs = segments(q, p)
        assert tuple(s.option for s in segs) == regime.order, (p, regime.case)
        assert segments(q, p, kappa_above) == segs
        mu = cutoffs(q, p)
        for seg, split in zip(segs[:-1], regime.split_points, strict=True):
            assert seg.right == pytest.approx(mu[split], abs=1e-12)
        seen.add(regime.case)
    assert seen == {"a", "b", "c", "d"}


def test_price_thresholds_closed_forms():
    q, kappa = MARKETS["Figure 5(b)-(d), Delta_B > 0, p_R < b_B"]
    t = price_thresholds(q, kappa)
    assert t["p_star"] == pytest.approx((q.delta_a * q.b_A + q.delta_b * q.a_A) / (q.delta_a + q.delta_b))
    assert t["p_L"] == pytest.approx(q.b_B - q.Delta_B * kappa / q.delta_a)
    assert t["p_R"] == pytest.approx(q.a_A - q.Delta_A * kappa / q.delta_b)
    assert (t["p_B_low"], t["p_B_high"]) == (t["p_star"], q.b_B)
    assert (t["p_G_low"], t["p_GLR"], t["p_G_high"]) == (t["p_L"], min(q.b_B, t["p_R"]), q.b_B)
