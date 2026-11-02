"""Tests of Figures 6-8 (aggregator impact) and of the thresholds in llmagg.impact.

The main.tex tests run only if the environment variable LLMAGG_PAPER_DIR points to the paper's source directory.
"""

import filecmp
import os
import re
from pathlib import Path

import numpy as np
import pytest

from figures import fig06_impact_market_expansion as fig06
from figures import fig07_impact_reallocation as fig07
from figures import fig08_impact_saturated_market as fig08
from figures import impact_panels
from llmagg import impact

MODULES = (fig06, fig07, fig08)
PANELS = [panel for module in MODULES for panel in module.PANELS]
SEGMENTATIONS = [module.SEGMENTATION for module in MODULES]
PRICE_FRACTIONS = (0.01, 0.2, 0.5, 0.8, 0.99)


def _region_prices(m: impact.Params, region: str) -> np.ndarray:
    lo, hi = {
        "ii": (impact.p_L(m), impact.p_R(m)),
        "iii": (impact.p_star(m), impact.p_L(m)),
        "iv": (0.0, impact.p_star(m)),
    }[region]
    return lo + (hi - lo) * np.array(PRICE_FRACTIONS)


def _files(outdir: Path) -> list[Path]:
    return sorted(p for p in outdir.rglob("*") if p.is_file())


# ---------------------------------------------------------------------------------------------------------------
# figure scripts
# ---------------------------------------------------------------------------------------------------------------


@pytest.mark.parametrize("module", MODULES, ids=lambda m: m.__name__.split(".")[-1])
def test_check_and_make(module, tmp_path):
    lines = module.check()
    assert lines and all(isinstance(line, str) and line for line in lines)
    files = module.make(tmp_path)
    expected = {Path("figure/fig_pre") / f"{module.SEGMENTATION.name}.pdf"}
    for panel in module.PANELS:
        expected |= {
            Path("figure/aggregator_impact") / panel.case / f for f in ("w_line_data.dat", "phi_curve_data.dat")
        }
    assert {f.relative_to(tmp_path) for f in files} == expected
    assert all(f.stat().st_size > 0 for f in files)


@pytest.mark.parametrize("module", MODULES, ids=lambda m: m.__name__.split(".")[-1])
def test_make_is_deterministic(module, tmp_path):
    module.make(tmp_path / "first")
    module.make(tmp_path / "second")
    first, second = _files(tmp_path / "first"), _files(tmp_path / "second")
    assert [f.relative_to(tmp_path / "first") for f in first] == [f.relative_to(tmp_path / "second") for f in second]
    assert all(filecmp.cmp(a, b, shallow=False) for a, b in zip(first, second, strict=True))


@pytest.mark.parametrize("panel", PANELS, ids=lambda p: p.case)
def test_dat_files(panel, tmp_path):
    w_file, phi_file = impact_panels.write_panel_data(panel, tmp_path)
    assert w_file.read_text().splitlines()[0] == "p w"
    assert phi_file.read_text().splitlines()[0] == "p phi"
    w, phi = np.loadtxt(w_file, skiprows=1), np.loadtxt(phi_file, skiprows=1)
    assert len(w) == len(phi) == panel.grid[2]
    np.testing.assert_array_equal(w[:, 0], phi[:, 0])
    np.testing.assert_allclose(w[:, 1], w[:, 0] / panel.rho_0, atol=1e-12)


def test_fig08_curves_run_to_p_star(tmp_path):
    for panel, p_star in zip(fig08.PANELS, (4.4, 13 / 3), strict=True):
        assert impact.p_star(panel.params) == pytest.approx(p_star, rel=1e-15)
        p = impact_panels.prices(panel)
        assert p[0] == 0.0 and p[-1] == pytest.approx(p_star, rel=1e-15)


# ---------------------------------------------------------------------------------------------------------------
# formulas: closed forms, definitions and the exact market model
# ---------------------------------------------------------------------------------------------------------------


@pytest.mark.parametrize("panel", PANELS, ids=lambda p: p.case)
def test_closed_forms_equal_cutoff_ratio_definitions(panel):
    m = panel.params
    assert impact.Psi_A(m) == pytest.approx(impact.Psi_A_from_cutoffs(m), rel=1e-12)
    assert impact.Psi_B(m) == pytest.approx(impact.Psi_B_from_cutoffs(m), rel=1e-12)
    p = _region_prices(m, "iii")
    np.testing.assert_allclose(impact.Omega_A(m, p), impact.Omega_A_from_cutoffs(m, p), rtol=1e-12)
    np.testing.assert_allclose(impact.Omega_B(m, p), impact.Omega_B_from_cutoffs(m, p), rtol=1e-12)
    p = _region_prices(m, "ii")  # the definition of Phi_A^L is 0/0 at p_R, which these prices avoid
    np.testing.assert_allclose(impact.Phi_A_L(m, p), impact.Phi_A_L_from_cutoffs(m, p), rtol=1e-9)


@pytest.mark.parametrize("panel", PANELS, ids=lambda p: p.case)
def test_thresholds_are_roots_of_revenue_change(panel):
    """Theorem thm:impact_aggregator: Delta R_X(w) = 0 at w = threshold / rho_0, in every region with a threshold."""
    m, rho_0 = panel.params, panel.rho_0
    for region, providers in (("ii", "A"), ("iii", "AB"), ("iv", "AB")):
        for p in _region_prices(m, region):
            for X in providers:
                root = impact.breakeven_wholesale_price(m, p, rho_0, X)
                assert root * rho_0 == pytest.approx(impact.wholesale_threshold(m, p, X), rel=1e-10, abs=1e-12)


@pytest.mark.parametrize("panel", PANELS, ids=lambda p: p.case)
def test_switching_prices(panel):
    """Props. property_phi_AL and property_Omega_AB: each threshold equals p at its switching price."""
    m = panel.params
    p_A, p_B, p_AL = impact.p_A_switch(m), impact.p_B_switch(m), impact.p_A_L_switch(m)
    assert impact.Omega_A(m, p_A) == pytest.approx(p_A, rel=1e-12)
    assert impact.Omega_B(m, p_B) == pytest.approx(p_B, rel=1e-12)
    assert impact.Phi_A_L(m, p_AL) == pytest.approx(p_AL, rel=1e-12)


@pytest.mark.parametrize("panel", PANELS, ids=lambda p: p.case)
def test_price_thresholds(panel):
    """p*, p_L, p_R are the values of v_A, v_B at the competitive cutoffs (Eqs. (pstar_rewrite), (pR_rewrite))."""
    m = panel.params
    assert impact.p_star(m) == pytest.approx(impact.utility(m, 0.0, "A", impact.mu_AB(m)), rel=1e-12)
    assert impact.p_star(m) == pytest.approx(impact.utility(m, 0.0, "B", impact.mu_AB(m)), rel=1e-12)
    assert impact.p_L(m) == pytest.approx(impact.utility(m, 0.0, "B", impact.mu_GB(m)), rel=1e-12)
    assert impact.p_R(m) == pytest.approx(impact.utility(m, 0.0, "A", impact.mu_AG(m)), rel=1e-12)


# ---------------------------------------------------------------------------------------------------------------
# main.tex (only with LLMAGG_PAPER_DIR)
# ---------------------------------------------------------------------------------------------------------------


@pytest.fixture(scope="module")
def paper_dir() -> Path:
    path = os.environ.get("LLMAGG_PAPER_DIR")
    if not path:
        pytest.skip("LLMAGG_PAPER_DIR not set")
    return Path(path)


@pytest.fixture(scope="module")
def main_tex(paper_dir) -> str:
    return (paper_dir / "main.tex").read_text()


def _axis_block(tex: str, case: str) -> str:
    """The pgfplots axis that reads figure/aggregator_impact/<case>/."""
    at = tex.index(f"figure/aggregator_impact/{case}/")
    return tex[tex.rindex(r"\begin{axis}", 0, at) : tex.index(r"\end{axis}", at)]


def _braced(text: str, key: str) -> str:
    """Content of ``key={...}`` (balanced braces)."""
    start = text.index(key + "={") + len(key) + 2
    depth = 1
    for i in range(start, len(text)):
        depth += {"{": 1, "}": -1}.get(text[i], 0)
        if depth == 0:
            return text[start:i]
    raise ValueError(key)


def _tick_labels(text: str) -> list[str]:
    """Math content of each tick label, e.g. ``{\\strut\\makebox[0pt][r]{$p_L$}}`` -> ``p_L``."""
    return [label.strip() for label in re.findall(r"\$([^$]*)\$", _braced(text, "xticklabels"))]


NUMBER = r"(-?\d+(?:\.\d+)?)"


@pytest.mark.parametrize("panel", PANELS, ids=lambda p: p.case)
def test_main_tex_ticks_and_domains(panel, main_tex):
    """Every x tick, fill/plot domain bound and vertical-line position in the panel's pgfplots code is a model value."""
    block = _axis_block(main_tex, panel.case)
    ticks = impact_panels.tick_values(panel)
    tex_ticks = [float(x) for x in _braced(block, "xtick").replace(",", " ").split()]
    assert _tick_labels(block) == [impact_panels.TICK_TEX[name] for name in panel.ticks]
    np.testing.assert_allclose(tex_ticks, list(ticks.values()), rtol=1e-14)

    allowed = np.array([0.0, *ticks.values()])
    positions = [float(x) for pair in re.findall(rf"domain={NUMBER}:{NUMBER}", block) for x in pair]
    positions += [float(x) for x in re.findall(rf"axis cs:{NUMBER},", block)]
    for coordinates in re.findall(r"coordinates\s*\{([^}]*)\}", block):
        positions += [float(x) for x in re.findall(rf"\(\s*{NUMBER}\s*,", coordinates)]
    assert positions
    for x in positions:
        assert np.min(np.abs(allowed - x)) <= 1e-14 * max(1.0, abs(x)), f"{x} is not a tick value {allowed}"


@pytest.mark.parametrize("panel", PANELS, ids=lambda p: p.case)
def test_dat_files_match_paper(panel, paper_dir, tmp_path):
    """The regenerated tables equal the paper's up to formatting (line ends) and the last printed digit."""
    for path in impact_panels.write_panel_data(panel, tmp_path):
        paper = paper_dir / path.relative_to(tmp_path)
        assert paper.read_text().splitlines()[0] == path.read_text().splitlines()[0]
        np.testing.assert_allclose(np.loadtxt(path, skiprows=1), np.loadtxt(paper, skiprows=1), rtol=0, atol=5e-12)


@pytest.mark.parametrize("panel", SEGMENTATIONS, ids=lambda s: s.name)
def test_main_tex_includes_segmentation_at_printed_size(panel, main_tex):
    """The segmentation PDF is drawn at its printed size: main.tex includes it at full subfigure width, untrimmed."""
    options = re.search(rf"\\includegraphics\[([^\]]*)\]\{{figure/fig_pre/{panel.name}\}}", main_tex).group(1)
    assert options == r"width=1\linewidth"
    subfigure = main_tex[: main_tex.index(f"figure/fig_pre/{panel.name}")].rsplit(r"\begin{subfigure}", 1)[1]
    assert subfigure.splitlines()[0].endswith(r"{0.32\textwidth}")
