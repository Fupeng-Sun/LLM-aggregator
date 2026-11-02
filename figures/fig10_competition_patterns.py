"""Figure 10 (``fig:competition_patterns``): representative competition patterns without the aggregator.

Six panels, each the path of the two providers' quality profiles under the myopic marginal-revenue heuristic
(Proposition 4, ``llmagg.patterns``) without an aggregator, in the full-coverage regime. Provider X is drawn at
(b_X, a_X): feature b on the horizontal axis and feature a on the vertical axis, as in Figure 9; one marker per period
t = 0, ..., T, the dashed line is a = b. Special markers show the periods of Theorem 4 (with the reversals of
Definition 3): triangle = first reversal T_lo (the paper's underline T), square = second reversal T_hi (overline T),
diamond = onset T_s of the single-feature race.

| panel | file        | x(0) = (a_A, b_A, a_B, b_B) | c Delta_T | T  | subcaption                              |
|-------|-------------|-----------------------------|-----------|----|-----------------------------------------|
| (a)   | pattern_0   | (1.2, 0.8, 0.8, 1.2)        | 0.5       | 33 | perpetual switching; T_lo=1, T_hi=2     |
| (b)   | pattern_3   | (1.103, 0.3, 0.3, 1.1)      | 0.5       | 30 | catch-up phase; T_lo=1, T_hi=4 < T_s    |
| (c)   | pattern_2   | (2, 1, 1, 2.003)            | 0.5       | 26 | catch-up phase; T_lo=3, T_hi=4 < T_s    |
| (d)   | pattern_2-2 | (2, 1, 1, 2.003)            | 0.57      | 19 | race without a switch; T_s=2            |
| (e)   | pattern_3-2 | (1.103, 0.3, 0.3, 1.1)      | 0.48      | 24 | race with one switch; T_lo=1, T_s=3     |
| (f)   | pattern_1   | (1.3, 0.8, 1, 1.2)          | 0.5       | 30 | race, > 2 switches; T_lo=1, T_hi=2 < T_s |

T is the horizon: the panel shows x(0), ..., x(T). The checks find T_s = 17 in (b), 11 in (c) and 17 in (f).

Sources. The initial states, per-period budgets c Delta_T and horizons T are those of the Excel workbook
(test_game.xlsx, not in this repository) that produced the paper's original panels, read in the paper's labels (the
workbook's columns a and b are the paper's b and a). Only the product c Delta_T affects a path; the script uses
c = 1 (normalization) and Delta_T = the value in the table. The paths equal the workbook's values (tests/test_fig10.py).
Facts about these states: panel (a) starts from an exactly symmetric state (a_A = b_B = 1.2, b_A = a_B = 0.8, so
delta_a = delta_b = 0.4), and panels (c) and (d) have a_A < b_B by 0.003 (a_A = 2, b_B = 2.003), whereas the main
text otherwise assumes a_A > b_B. All six states have delta_a(0), delta_b(0) > 0 and full coverage (checked).

Market parameters (chosen for illustration; they matter only for the full-coverage check and for evaluating
``llmagg.market.myopic_lambda``, since without an aggregator and with full coverage the heuristic does not depend on
them): p = 0.25 (below p^* at every initial state, so the market is fully covered in every period, as qualities
only grow), w_A = w_B = 0.1, rho_0 = 1 (Assumption 1: p > w rho_0).

Produces figure/fig_pre/pattern_{0,3,2,2-2,3-2,1}.pdf and the six paths as data/fig10/pattern_*.csv (paper labels).
"""

from dataclasses import dataclass, field
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.lines import Line2D

from llmagg.cli import run_figure
from llmagg.io import save_pdf
from llmagg.market import myopic_lambda, revenues
from llmagg.patterns import classify, myopic_path
from llmagg.plotting import COLOR_A, COLOR_B, COLOR_OUTSIDE, use_style


@dataclass(frozen=True)
class Panel:
    """One panel of Figure 10: initial state x(0) = (a_A, b_A, a_B, b_B), period length, horizon, subcaption."""

    letter: str
    name: str
    x0: tuple[float, float, float, float]
    delta_T: float
    T: int
    subcaption: str
    caption: dict[str, int] = field(default_factory=dict)  # values of T_lo, T_hi, T_s stated in the subcaption
    race: str | None = None  # feature of the single-feature race (Theorem 4(b)); None = perpetual switching (a)


C = 1.0  # budget rate c (normalization: only c Delta_T matters)
PANELS = (
    Panel(
        letter="a",
        name="pattern_0",
        x0=(1.2, 0.8, 0.8, 1.2),
        delta_T=0.5,
        T=33,
        subcaption="Perpetual switching",
        caption={"T_lo": 1, "T_hi": 2},
    ),
    Panel(
        letter="b",
        name="pattern_3",
        x0=(1.103, 0.3, 0.3, 1.1),
        delta_T=0.5,
        T=30,
        race="a",
        subcaption="Catch-up phase between first two switches",
        caption={"T_lo": 1, "T_hi": 4},
    ),
    Panel(
        letter="c",
        name="pattern_2",
        x0=(2.0, 1.0, 1.0, 2.003),
        delta_T=0.5,
        T=26,
        race="b",
        subcaption="Catch-up phase before first switch",
        caption={"T_lo": 3, "T_hi": 4},
    ),
    Panel(
        letter="d",
        name="pattern_2-2",
        x0=(2.0, 1.0, 1.0, 2.003),
        delta_T=0.57,
        T=19,
        race="a",
        subcaption="Single-feature race without a switch",
        caption={"T_s": 2},
    ),
    Panel(
        letter="e",
        name="pattern_3-2",
        x0=(1.103, 0.3, 0.3, 1.1),
        delta_T=0.48,
        T=24,
        race="a",
        subcaption="Single-feature race with one switch",
        caption={"T_lo": 1, "T_s": 3},
    ),
    Panel(
        letter="f",
        name="pattern_1",
        x0=(1.3, 0.8, 1.0, 1.2),
        delta_T=0.5,
        T=30,
        race="a",
        subcaption="Single-feature race with more than two switches",
        caption={"T_lo": 1, "T_hi": 2},
    ),
)
EXTRA_PERIODS = 2000  # periods simulated beyond T to check that a race or perpetual switching persists

P, KAPPA, W_A, W_B, RHO_0 = 0.25, 0.0, 0.1, 0.1, 1.0  # market (chosen for illustration, see docstring)

# ---------------------------------------------------------------------------------------------------------------
# style and layout
# ---------------------------------------------------------------------------------------------------------------
TEXTWIDTH_PT = 469.755  # \textwidth of the paper
PANEL_WIDTH_IN = 0.32 * TEXTWIDTH_PT / 72  # each panel is a subfigure of width 0.32\textwidth
FIGSIZE = (PANEL_WIDTH_IN, PANEL_WIDTH_IN)
AXES_RECT = (0.155, 0.135, 0.815, 0.815)  # left, bottom, width, height in figure fractions (square plot area)
LIMIT = 5.0
TICKS = np.arange(6)
COLOR_DIAGONAL = COLOR_OUTSIDE  # the a = b line, neutral grey (green would suggest the aggregator)
COLOR_GRID = "#D9D9D9"
COLOR_TEXT = "#404040"
LINE_WIDTH = 0.8
MARKER_SIZE = 2.0  # circles at ordinary periods
SPECIAL_SIZE = 3.6  # markers of T_lo, T_hi, T_s
SPECIAL_MARKERS = {"T_lo": "^", "T_hi": "s", "T_s": "D"}
TEX = {"T_lo": r"$\underline{T}$", "T_hi": r"$\overline{T}$", "T_s": r"$T_s$"}
TICK_SIZE = 8
LABEL_SIZE = 9
# position of each period label: (provider whose marker is labeled, offset in points, horizontal alignment)
LABEL_POSITIONS = {
    "pattern_0": {"T_lo": ("B", (-2, 2), "right"), "T_hi": ("B", (3, -2), "left")},
    "pattern_3": {"T_lo": ("B", (-2, 2), "right"), "T_hi": ("B", (3, -2), "left"), "T_s": ("A", (3, -2), "left")},
    "pattern_2": {"T_lo": ("B", (-2, 2), "right"), "T_hi": ("B", (3, -2), "left"), "T_s": ("B", (-1, 4), "center")},
    "pattern_2-2": {"T_s": ("A", (-2, 2), "right")},
    "pattern_3-2": {"T_lo": ("B", (-2, 2), "right"), "T_s": ("A", (3, -2), "left")},
    "pattern_1": {"T_lo": ("B", (-2, 2), "right"), "T_hi": ("B", (3, -2), "left"), "T_s": ("A", (3, -2), "left")},
}


def panel_path(panel: Panel, extra: int = 0) -> tuple[np.ndarray, np.ndarray]:
    """Path x(0..T+extra) and budget shares in feature a in periods 0..T+extra-1 under the heuristic."""
    return myopic_path(panel.x0, C, panel.delta_T, panel.T + extra)


def pattern(panel: Panel) -> dict:
    """Definition 3 / Theorem 4 on the path continued EXTRA_PERIODS beyond T (``llmagg.patterns.classify``), with
    T_lo, T_hi (first two reversal periods) and T_s (onset of the single-feature race; None if there is none)."""
    path, lam = panel_path(panel, EXTRA_PERIODS)
    info = classify(path, lam)
    rev = info["reversals"]
    info["periods"] = {
        "T_lo": rev[0] if rev else None,
        "T_hi": rev[1] if len(rev) > 1 else None,
        "T_s": info["race_start"],
    }
    info["path"], info["lam"] = path, lam
    return info


def check() -> list[str]:
    """Check each path against Proposition 4, Theorem 4 and the subcaptions of Figure 10."""
    lines = []
    for panel in PANELS:
        tag = f"({panel.letter}) {panel.name}"
        x0 = np.asarray(panel.x0)
        assert x0[0] > x0[2] and x0[3] > x0[1], f"{tag}: Theorem 4 assumes delta_a(0), delta_b(0) > 0"
        path, lam = panel_path(panel)
        for t in range(panel.T):
            shares = revenues(*path[t], P, KAPPA, W_A, W_B, RHO_0, False)[3:5]
            assert abs(sum(shares) - 1.0) < 1e-12, f"{tag}: market not fully covered in period {t}"
            heuristic = [myopic_lambda(*path[t], P, KAPPA, W_A, W_B, RHO_0, False, who) for who in (0, 1)]
            assert np.array_equal(heuristic, lam[t]), f"{tag}: Proposition 4 and myopic_lambda differ in period {t}"

        info = pattern(panel)
        found = info["periods"]
        T_lo, T_hi, T_s = found["T_lo"], found["T_hi"], found["T_s"]
        rev, sw = info["reversals"], info["switches"]
        n = len(info["path"]) - 1
        assert info["dominance"] is None, f"{tag}: dominance in period {info['dominance']}"
        assert rev == sw, f"{tag}: a reversal is not a switch"
        for key, value in panel.caption.items():
            assert found[key] == value, f"{tag}: {key} = {found[key]}, subcaption {value}"
        if panel.race is None:
            assert info["label"] == "perpetual switching", f"{tag}: {info['label']}"
            end = n
        else:
            assert info["label"] == f"race on {panel.race} from t={T_s}", f"{tag}: {info['label']}"
            assert T_s <= panel.T, f"{tag}: T_s = {T_s} lies beyond the horizon"
            end = T_s
        # pattern (a) before T_s: weak-feature investment, reversal periods {T_lo} U {T_hi, ..., T_s},
        # and T_lo = 1 or T_hi = T_lo + 1
        da, db = info["da"], info["db"]
        for t in range(min(end, n)):
            weak = (0.0 if da[t] > 0 else 1.0, 1.0 if db[t] > 0 else 0.0)
            assert tuple(info["lam"][t]) == weak, f"{tag}: no weak-feature investment in period {t}"
        if T_hi is not None:
            assert rev == [T_lo, *range(T_hi, end + 1)], f"{tag}: reversal periods {rev[:8]}"
            assert T_lo == 1 or T_hi == T_lo + 1, f"{tag}: timing condition of Theorem 4(a)"
            if T_s is not None:
                assert T_hi < T_s, f"{tag}: subcaption states T_hi < T_s"
        # the subcaption's description
        if panel.letter == "b":
            assert T_lo == 1 and T_hi > T_lo + 1, f"{tag}: catch-up phase between the first two switches"
        if panel.letter == "c":
            assert T_lo >= 2 and T_hi == T_lo + 1, f"{tag}: catch-up phase before the first switch"
        if panel.letter == "d":
            assert not sw, f"{tag}: race without a switch"
        if panel.letter == "e":
            assert len(sw) == 1, f"{tag}: race after one switch"
        if panel.letter == "f":
            assert len(sw) > 2, f"{tag}: race after more than two switches"
        stated = ", ".join(f"{k} = {v}" for k, v in found.items() if v is not None)
        lines.append(
            f"{tag}: {panel.subcaption.lower()} ({stated}; {len(sw)} switch(es) and no dominance in periods "
            f"1..{n}); full coverage, and Proposition 4 = myopic_lambda in periods 0..{panel.T - 1}"
        )
    return lines


def draw(panel: Panel) -> plt.Figure:
    """One panel: both providers' paths at (b_X, a_X), period markers and labels."""
    path, _ = panel_path(panel)
    marks = {k: v for k, v in pattern(panel)["periods"].items() if v is not None}
    points = {"A": (path[:, 1], path[:, 0]), "B": (path[:, 3], path[:, 2])}
    fig = plt.figure(figsize=FIGSIZE)
    ax = fig.add_axes(AXES_RECT)
    ax.plot([0, LIMIT], [0, LIMIT], ls=(0, (3, 2)), color=COLOR_DIAGONAL, lw=LINE_WIDTH, zorder=1)
    for who, color in (("A", COLOR_A), ("B", COLOR_B)):
        x, y = points[who]
        ordinary = [t for t in range(panel.T + 1) if t not in marks.values()]
        ax.plot(x, y, color=color, lw=LINE_WIDTH, zorder=2)
        ax.plot(x[ordinary], y[ordinary], ls="none", marker="o", ms=MARKER_SIZE, color=color, mew=0, zorder=3)
        for key, t in marks.items():
            ax.plot(x[t], y[t], ls="none", marker=SPECIAL_MARKERS[key], ms=SPECIAL_SIZE, color=color, mew=0, zorder=4)
    for key, (who, offset, ha) in LABEL_POSITIONS[panel.name].items():
        x, y = points[who]
        t = marks[key]
        va = "bottom" if offset[1] > 0 else "top"
        ax.annotate(
            TEX[key],
            (x[t], y[t]),
            xytext=offset,
            textcoords="offset points",
            ha=ha,
            va=va,
            fontsize=LABEL_SIZE,
            color=COLOR_TEXT,
            zorder=5,
        )
    ax.set_xlim(0, LIMIT)
    ax.set_ylim(0, LIMIT)
    ax.set_xticks(TICKS)
    ax.set_yticks(TICKS)
    ax.grid(True, color=COLOR_GRID, lw=0.5, zorder=0)
    ax.set_axisbelow(True)
    for spine in ax.spines.values():
        spine.set_visible(False)
    ax.tick_params(length=0, labelsize=TICK_SIZE, pad=2)
    ax.set_xlabel(r"$b_X$", fontsize=LABEL_SIZE, labelpad=1)
    ax.set_ylabel(r"$a_X$", fontsize=LABEL_SIZE, labelpad=2, rotation=0, va="center")
    handles = [
        Line2D([], [], color=c, lw=LINE_WIDTH, marker="o", ms=MARKER_SIZE, mew=0, label=who)
        for who, c in (("A", COLOR_A), ("B", COLOR_B))
    ]
    ax.legend(
        handles=handles,
        loc="lower right",
        ncol=2,
        frameon=False,
        fontsize=TICK_SIZE,
        handlelength=1.5,
        handletextpad=0.3,
        columnspacing=0.8,
        borderaxespad=0.2,
    )
    return fig


def write_csv(panel: Panel, path_csv: Path) -> Path:
    """The path in the paper's labels: one row per period; invest_X = feature chosen in that period (empty at T)."""
    path, lam = panel_path(panel)
    path_csv.parent.mkdir(parents=True, exist_ok=True)
    rows = ["period,a_A,b_A,a_B,b_B,delta_a,delta_b,invest_A,invest_B"]
    for t, (a_A, b_A, a_B, b_B) in enumerate(path):
        invest = ["a" if s == 1.0 else "b" for s in lam[t]] if t < panel.T else ["", ""]
        values = ",".join(repr(float(v)) for v in (a_A, b_A, a_B, b_B, a_A - a_B, b_B - b_A))
        rows.append(f"{t},{values},{invest[0]},{invest[1]}")
    path_csv.write_text("\n".join(rows) + "\n")
    return path_csv


def make(outdir: Path) -> list[Path]:
    """Write the six panels (figure/fig_pre/) and their paths (data/fig10/)."""
    use_style()
    outdir = Path(outdir)
    files = []
    for panel in PANELS:
        fig = draw(panel)
        files.append(save_pdf(fig, outdir / "figure" / "fig_pre" / f"{panel.name}.pdf"))
        plt.close(fig)
    for panel in PANELS:
        files.append(write_csv(panel, outdir / "data" / "fig10" / f"{panel.name}.csv"))
    return files


if __name__ == "__main__":
    run_figure(make, check)
