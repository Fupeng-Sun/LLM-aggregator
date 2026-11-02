"""Figure 10 (competition patterns): checks, output, and agreement with the Excel workbook it replaces.

The reference values below are the cached cell values of test_game.xlsx (not in this repository) (sheets "0", "3", "2",
"2 (2)", "3 (2)", "1"), read once at full precision and converted to the paper's labels (the workbook's columns a and
b are the paper's b and a): the state in the last row, x(T), and the feature each provider invested in, periods
0..T-1. Together they pin down the whole path.
"""

import filecmp

import numpy as np
import pytest

from figures import fig10_competition_patterns as fig10
from llmagg.market import myopic_lambda
from llmagg.patterns import invests_in_a, myopic_path

# name: (x(T) = (a_A, b_A, a_B, b_B), features of A, features of B), from the workbook
WORKBOOK = {
    "pattern_0": (
        (4.256780148158449, 4.3145585328025895, 4.3145585328025895, 4.256780148158449),
        "babababababababababababababababab",
        "ababababababababababababababababa",
    ),
    "pattern_3": (
        (4.9011505433031175, 3.3382168104660352, 4.933776575893878, 3.2884076513952643),
        "baaabababababababaaaaaaaaaaaaa",
        "abbbababababababaaaaaaaaaaaaaa",
    ),
    "pattern_2": (
        (2.8614223832416275, 4.885526126470387, 2.934044823150041, 4.844309491815944),
        "bbbababababbbbbbbbbbbbbbbb",
        "aaabababababbbbbbbbbbbbbbb",
    ),
    "pattern_2-2": (
        (4.889551300857739, 1.9330573248407643, 4.864409258707208, 2.003),
        "bbaaaaaaaaaaaaaaaaa",
        "aaaaaaaaaaaaaaaaaaa",
    ),
    "pattern_3-2": (
        (4.9058002938509055, 1.9000000000000001, 4.923570230923056, 1.8487896718665948),
        "baaaaaaaaaaaaaaaaaaaaaaa",
        "abbaaaaaaaaaaaaaaaaaaaaa",
    ),
    "pattern_1": (
        (4.835935155183583, 3.235555923163245, 4.885526126470387, 3.1571897244475355),
        "bababababababababaaaaaaaaaaaaa",
        "ababababababababaaaaaaaaaaaaaa",
    ),
}
# periods stated in the subcaptions of Figure 10 (None: not stated)
SUBCAPTIONS = {
    "pattern_0": {"T_lo": 1, "T_hi": 2, "T_s": None},
    "pattern_3": {"T_lo": 1, "T_hi": 4},
    "pattern_2": {"T_lo": 3, "T_hi": 4},
    "pattern_2-2": {"T_lo": None, "T_hi": None, "T_s": 2},
    "pattern_3-2": {"T_lo": 1, "T_hi": None, "T_s": 3},
    "pattern_1": {"T_lo": 1, "T_hi": 2},
}
PANELS = {panel.name: panel for panel in fig10.PANELS}


def features(shares: np.ndarray) -> str:
    return "".join("a" if s == 1.0 else "b" for s in shares)


def test_check_passes():
    lines = fig10.check()
    assert len(lines) == len(fig10.PANELS)


def test_make_is_deterministic(tmp_path):
    first = fig10.make(tmp_path / "1")
    second = fig10.make(tmp_path / "2")
    names = sorted(f.relative_to(tmp_path / "1").as_posix() for f in first)
    assert names == sorted([f"figure/fig_pre/{n}.pdf" for n in WORKBOOK] + [f"data/fig10/{n}.csv" for n in WORKBOOK])
    for a, b in zip(first, second, strict=True):
        assert filecmp.cmp(a, b, shallow=False), a.name


def test_pdf_has_the_printed_size(tmp_path):
    pdf = tmp_path / "figure" / "fig_pre" / "pattern_0.pdf"
    fig10.make(tmp_path)
    box = [float(v) for v in pdf.read_bytes().split(b"/MediaBox [")[1].split(b"]")[0].split()]
    assert box[2] - box[0] == pytest.approx(0.32 * 469.755, abs=0.01)


@pytest.mark.parametrize("name", list(WORKBOOK))
def test_paths_equal_the_workbook(name):
    panel = PANELS[name]
    path, lam = myopic_path(panel.x0, fig10.C, panel.delta_T, panel.T)
    final, features_A, features_B = WORKBOOK[name]
    assert len(features_A) == len(features_B) == panel.T
    assert features(lam[:, 0]) == features_A
    assert features(lam[:, 1]) == features_B
    np.testing.assert_allclose(path[-1], final, rtol=0, atol=1e-12)


@pytest.mark.parametrize("name", list(WORKBOOK))
def test_csv_holds_the_path(tmp_path, name):
    fig10.make(tmp_path)
    rows = (tmp_path / "data" / "fig10" / f"{name}.csv").read_text().splitlines()
    final, features_A, features_B = WORKBOOK[name]
    assert rows[0] == "period,a_A,b_A,a_B,b_B,delta_a,delta_b,invest_A,invest_B"
    assert len(rows) == PANELS[name].T + 2
    last = rows[-1].split(",")
    np.testing.assert_allclose([float(v) for v in last[1:5]], final, rtol=0, atol=1e-12)
    assert "".join(r.split(",")[7] for r in rows[1:-1]) == features_A
    assert "".join(r.split(",")[8] for r in rows[1:-1]) == features_B


@pytest.mark.parametrize("name", list(WORKBOOK))
def test_periods_match_the_subcaptions(name):
    found = fig10.pattern(PANELS[name])["periods"]
    for key, value in SUBCAPTIONS[name].items():
        assert found[key] == value, key


@pytest.mark.parametrize("name", list(WORKBOOK))
def test_closed_form_equals_myopic_lambda(name):
    """Proposition 4 and the heuristic of Eq. (phiA_myopic_bangbang) on the exact market model agree every period."""
    panel = PANELS[name]
    path, lam = myopic_path(panel.x0, fig10.C, panel.delta_T, panel.T)
    market = (fig10.P, fig10.KAPPA, fig10.W_A, fig10.W_B, fig10.RHO_0, False)
    for t in range(panel.T):
        assert myopic_lambda(*path[t], *market, 0) == lam[t, 0], t
        assert myopic_lambda(*path[t], *market, 1) == lam[t, 1], t


def test_rule_is_invariant_to_relabeling_the_providers():
    """Proposition 4 with both gaps negative: swapping A and B leaves each provider's choice unchanged."""
    x = np.array([1.2, 1.425, 1.425, 1.2])  # pattern_0 at t = 1: delta_a, delta_b < 0
    a_A, b_A, a_B, b_B = x
    assert invests_in_a(a_A, b_A, a_A - a_B, b_B - b_A) == invests_in_a(a_A, b_A, a_B - a_A, b_A - b_B)
    with pytest.raises(ValueError):
        invests_in_a(1.0, 1.0, 0.1, -0.1)
