"""Panel specifications, drawing and checks shared by the aggregator-impact figures (Figures 6, 7 and 8)."""

from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib import patheffects
from matplotlib.patches import PathPatch
from matplotlib.path import Path as GeomPath

from llmagg import impact
from llmagg.impact import Params
from llmagg.io import save_pdf, write_dat
from llmagg.plotting import COLOR_INK, PROVIDER_COLORS

# ---------------------------------------------------------------------------------------------------------------
# style and size (shared by Figures 6-8)
# ---------------------------------------------------------------------------------------------------------------

TEXTWIDTH_PT = 469.755  # \textwidth of the paper
PANEL_WIDTH_PT = 0.32 * TEXTWIDTH_PT  # subfigure width; main.tex includes the PDF at width=1\linewidth
TEXT_SIZE = 8  # pt: tick labels and line labels
NOTE_SIZE = 7  # pt: brace annotations
HALO = [patheffects.withStroke(linewidth=2.2, foreground="white")]  # keeps text legible where it crosses lines
CURVE_WIDTH = 1.0  # utility lines
GUIDE_WIDTH = 0.5  # braces, dashed cutoff lines, leader lines
FILL_ALPHA = 0.2  # shading under the upper envelope
LEADER_COLOR = "#AAAAAA"
ZERO_LINE_COLOR = "#999999"
BRACE_TIP = 0.02  # size of the rounded corners and the tip of a brace, in data units
TICK_LENGTH = 3.0  # pt (major x ticks; minor ticks and y ticks are half as long)
OPTIONS = ("A", "B", "G")

# x-tick label that main.tex uses for each model quantity (for the main.tex tests)
TICK_TEX = {
    "p_star": r"p^\ast",
    "p_L": r"p_L",
    "p_R": r"p_R",
    "b_B": r"b_B",
    "p_A_switch": r"\underline p_A",
    "p_B_switch": r"\underline p_B",
    "p_A_L_switch": r"\underline p_A^L",
}

# segmentations of Theorem thm:impact_aggregator per price region: (without G, with G), left to right in mu
THEOREM_SEGMENTATION = {"ii": ("BOA", "BOGA"), "iii": ("BOA", "BGA"), "iv": ("BA", "BGA")}

# the price region of each threshold function and the switching price at which it crosses p
REGION_OF = {"Phi_A_L": "ii", "Omega": "iii", "Psi": "iv"}
SWITCH_OF = {("Phi_A_L", "A"): "p_A_L_switch", ("Omega", "A"): "p_A_switch", ("Omega", "B"): "p_B_switch"}
SAMPLE_FRACTIONS = (0.03, 0.25, 0.5, 0.75, 0.97)  # where in its price region each threshold is checked
TOL = 1e-10  # relative tolerance of the closed forms against the market model


# ---------------------------------------------------------------------------------------------------------------
# threshold panels (pgfplots data)
# ---------------------------------------------------------------------------------------------------------------


@dataclass(frozen=True)
class ThresholdPanel:
    """A pgfplots panel: the lines w_X = p / rho_0 and w_X = threshold(p) / rho_0 on a price grid."""

    case: str  # data directory figure/aggregator_impact/<case>/
    label: str  # LaTeX label of the subfigure
    params: Params
    rho_0: float
    provider: str  # "A" or "B"
    threshold: str  # "Phi_A_L" (region ii), "Omega" (region iii) or "Psi" (region iv)
    grid: tuple[str | float, str | float, int]  # first price, last price (model quantity or number), points
    ticks: tuple[str, ...]  # model quantities at the x ticks, left to right: the order the subcaption states
    lemma: str  # the statement of Lemma lem:aggregator_impact_tech that the panel illustrates
    lemma_holds: Callable[[Params], bool]


def _value(m: Params, x: str | float) -> float:
    return impact.grid_value(m, x) if isinstance(x, str) else float(x)


def prices(panel: ThresholdPanel) -> np.ndarray:
    """The price grid of the panel's .dat files."""
    start, end, n = panel.grid
    return np.linspace(_value(panel.params, start), _value(panel.params, end), n)


def threshold_curve(panel: ThresholdPanel, p):
    """Wholesale-price threshold threshold(p) / rho_0 drawn in the panel (closed forms of llmagg.impact)."""
    m, X = panel.params, panel.provider
    if panel.threshold == "Phi_A_L":
        value = impact.Phi_A_L(m, p)
    elif panel.threshold == "Omega":
        value = impact.Omega_A(m, p) if X == "A" else impact.Omega_B(m, p)
    else:
        value = (impact.Psi_A(m) if X == "A" else impact.Psi_B(m)) * p
    return value / panel.rho_0


def tick_values(panel: ThresholdPanel) -> dict[str, float]:
    """Model value of each x tick of the panel."""
    return {name: impact.grid_value(panel.params, name) for name in panel.ticks}


def write_panel_data(panel: ThresholdPanel, outdir: Path) -> list[Path]:
    """Write the panel's two pgfplots tables, as read by main.tex."""
    p = prices(panel)
    folder = Path(outdir) / "figure" / "aggregator_impact" / panel.case
    return [
        write_dat(folder / "w_line_data.dat", {"p": p, "w": p / panel.rho_0}),
        write_dat(folder / "phi_curve_data.dat", {"p": p, "phi": threshold_curve(panel, p)}),
    ]


def _close(x: float, y: float, tol: float = TOL) -> bool:
    return abs(x - y) <= tol * max(1.0, abs(x), abs(y))


def check_assumptions(m: Params) -> None:
    """Assumptions of Theorem thm:impact_aggregator and the baseline profile."""
    assert m.a_A > m.b_B and m.Delta_B > 0 and 0 <= m.kappa < impact.kappa_bar(m)
    assert m.delta_a > 0 and m.delta_b > 0 and m.Delta_A > 0


def check_panel(panel: ThresholdPanel) -> list[str]:
    """Check a threshold panel against Theorem thm:impact_aggregator, its lemma and the exact market model."""
    m, X, rho_0 = panel.params, panel.provider, panel.rho_0
    Y = "B" if X == "A" else "A"
    name = f"{panel.label} ({panel.case})"
    check_assumptions(m)
    lines = [f"{name}: a_A > b_B, Delta_B > 0, 0 <= kappa < kappa_bar (Theorem thm:impact_aggregator)"]

    ticks = list(tick_values(panel).values())
    assert all(a < b for a, b in zip(ticks, ticks[1:], strict=False)), "ticks not in the subcaption's order"
    assert panel.lemma_holds(m), panel.lemma
    lines.append(f"{name}: tick order {' < '.join(panel.ticks)}; Lemma lem:aggregator_impact_tech{panel.lemma}")

    switch = SWITCH_OF.get((panel.threshold, X))
    if switch is not None:
        p_sw = impact.grid_value(m, switch)
        assert _close(threshold_curve(panel, p_sw) * rho_0, p_sw)
        lines.append(f"{name}: threshold(p) = p at the switching price {switch} = {p_sw:.6g}")

    region = REGION_OF[panel.threshold]
    lo, hi = {"ii": (impact.p_L(m), impact.p_R(m)), "iii": (impact.p_star(m), impact.p_L(m))}.get(
        region, (0.0, impact.p_star(m))
    )
    without, with_g = THEOREM_SEGMENTATION[region]
    for q in lo + (hi - lo) * np.array(SAMPLE_FRACTIONS):
        assert impact.impact_region(m, q) == region
        root = impact.breakeven_wholesale_price(m, q, rho_0, X)
        assert _close(root, threshold_curve(panel, q)), (q, root, threshold_curve(panel, q))
        assert _close(root * rho_0, impact.wholesale_threshold(m, q, X))
        assert ("O" in with_g) == (impact.shares(m, q, True)["O"] > 1e-12)
        assert ("O" in without) == (impact.shares(m, q, False)["O"] > 1e-12)
        if region == "ii":  # provider B is always better off
            assert all(impact.revenue_change(m, q, f * q / rho_0, rho_0, "B") > 0 for f in (0.01, 0.99))
        if region == "iv":  # the provider with the larger gap is worse off for every w_Y < p / rho_0
            assert impact.revenue_change(m, q, 0.999 * q / rho_0, rho_0, Y) < 0
    lines.append(
        f"{name}: on region ({region}) the root of Delta R_{X}(w) = 0 (exact market model) equals the curve at "
        f"{len(SAMPLE_FRACTIONS)} prices; segmentation {without} -> {with_g}"
    )
    if region == "ii":
        lines.append(f"{name}: Delta R_B > 0 for every wholesale price (Theorem thm:impact_aggregator(ii))")
    if region == "iv":
        psi_Y = impact.Psi_A(m) if Y == "A" else impact.Psi_B(m)
        assert psi_Y >= 1
        lines.append(f"{name}: Psi_{Y} = {psi_Y:.4g} >= 1, provider {Y} worse off (Theorem thm:impact_aggregator(iv))")
    return lines


# ---------------------------------------------------------------------------------------------------------------
# segmentation panels (utilities against the user type mu)
# ---------------------------------------------------------------------------------------------------------------


@dataclass(frozen=True)
class Brace:
    """Curly brace from ``start`` to ``end`` (data coordinates), bulging to the right of the direction start -> end."""

    start: tuple[float, float]
    end: tuple[float, float]
    arm: float  # length of the straight arms at both ends
    text: str
    gap: float  # distance from the tip to the text centre


@dataclass(frozen=True)
class Callout:
    """Label of a utility line with a leader line from ``line_start`` through ``knee`` to the line at ``mu``."""

    option: str
    text_xy: tuple[float, float]
    line_start: tuple[float, float]
    knee: tuple[float, float]
    mu: float


@dataclass(frozen=True)
class SegmentationPanel:
    """Panel (a): u_A, u_B, u_G against mu at one price, the upper envelope shaded by segment."""

    name: str  # file figure/fig_pre/<name>.pdf
    label: str  # LaTeX label of the subfigure
    params: Params
    p: float
    region: str  # price region of Theorem thm:impact_aggregator that p lies in
    segmentation: str  # with the aggregator, left to right in mu, e.g. "BOGA"
    split_points: tuple[float, ...]  # cutoffs between its segments
    without: str  # segmentation without the aggregator
    without_split_points: tuple[float, ...]
    height: float  # pt
    margins: tuple[float, float, float, float]  # left, right, top, bottom of the axes box, pt
    ylim: tuple[float, float]
    segment_labels: dict[str, tuple[float, float]]  # position of each segment's script letter
    dashed: tuple[tuple[float, str], ...]  # vertical dashed lines from 0 to the utility of an option at mu
    marks: tuple[float, ...]  # short tick marks on the zero line
    callouts: tuple[Callout, ...]
    braces: tuple[Brace, ...]


def segments(letters: str, split_points: tuple[float, ...]) -> list[tuple[str, float, float]]:
    """Segments (option, left end, right end) of a segmentation such as "BOGA" with its split points."""
    ends = (0.0, *split_points, 1.0)
    return [(x, ends[i], ends[i + 1]) for i, x in enumerate(letters)]


def check_segmentation(panel: SegmentationPanel) -> list[str]:
    """The panel's price lies in its region and the drawn segmentations are those of the exact market model."""
    m, p = panel.params, panel.p
    check_assumptions(m)
    assert impact.impact_region(m, p) == panel.region
    assert (panel.without, panel.segmentation) == THEOREM_SEGMENTATION[panel.region]
    for letters, points, with_g in (
        (panel.segmentation, panel.split_points, True),
        (panel.without, panel.without_split_points, False),
    ):
        drawn = dict.fromkeys("ABGO", 0.0)
        for x, lo, hi in segments(letters, points):
            assert lo < hi
            drawn[x] += hi - lo
        model = impact.shares(m, p, with_g)
        assert all(abs(drawn[x] - model[x]) < 1e-12 for x in "ABGO"), (letters, drawn, model)
    return [
        f"{panel.label}: p = {panel.p} in region ({panel.region}); segmentation {panel.without} -> "
        f"{panel.segmentation} with the cutoffs drawn equals the exact market model"
    ]


def draw_brace(ax, brace: Brace) -> None:
    """Draw a curly brace with rounded corners and a centred tip, and its label beyond the tip."""
    p1, p2 = np.asarray(brace.start, float), np.asarray(brace.end, float)
    length = np.linalg.norm(p2 - p1)
    d = (p2 - p1) / length
    n = np.array([d[1], -d[0]])
    w, r, h = brace.arm, BRACE_TIP, length / 2

    def at(s, t):
        return p1 + s * d + t * n

    verts = [
        at(0, 0), at(0, w), at(0, w + r), at(r, w + r), at(h - r, w + r), at(h, w + r), at(h, w + 2 * r),
        at(h, w + r), at(h + r, w + r), at(length - r, w + r), at(length, w + r), at(length, w), at(length, 0),
    ]  # fmt: skip
    M, L, C = GeomPath.MOVETO, GeomPath.LINETO, GeomPath.CURVE3
    codes = [M, L, C, C, L, C, C, C, C, L, C, C, L]
    ax.add_patch(
        PathPatch(GeomPath(verts, codes), fill=False, lw=GUIDE_WIDTH, color=COLOR_INK, zorder=5, clip_on=False)
    )
    ax.text(
        *at(h, w + 2 * r + brace.gap),
        brace.text,
        ha="center",
        va="center",
        fontsize=NOTE_SIZE,
        zorder=6,
        path_effects=HALO,
        linespacing=1.0,
    )


def draw_segmentation(panel: SegmentationPanel) -> plt.Figure:
    """Draw a segmentation panel at its printed size."""
    m, p = panel.params, panel.p
    height = panel.height
    left, right, top, bottom = panel.margins
    fig = plt.figure(figsize=(PANEL_WIDTH_PT / 72, height / 72))
    ax = fig.add_axes(
        (left / PANEL_WIDTH_PT, bottom / height, 1 - (left + right) / PANEL_WIDTH_PT, 1 - (top + bottom) / height)
    )
    ax.set_xlim(0, 1)
    ax.set_ylim(*panel.ylim)

    mu = np.linspace(0, 1, 201)
    for x in OPTIONS:
        ax.plot(mu, impact.utility(m, p, x, mu), color=PROVIDER_COLORS[x], lw=CURVE_WIDTH, zorder=3)
    for x, lo, hi in segments(panel.segmentation, panel.split_points):
        if x != "O":
            piece = np.linspace(lo, hi, 2)
            ax.fill_between(piece, impact.utility(m, p, x, piece), 0, color=PROVIDER_COLORS[x], alpha=FILL_ALPHA, lw=0)
    for x, xy in panel.segment_labels.items():
        ax.text(
            *xy, rf"$\mathcal{{{x}}}$", ha="center", va="center", fontsize=TEXT_SIZE + 1, zorder=6, path_effects=HALO
        )
    ax.axhline(0, color=ZERO_LINE_COLOR, lw=GUIDE_WIDTH, zorder=2)
    for x_mu, option in panel.dashed:
        ax.plot([x_mu, x_mu], [0, impact.utility(m, p, option, x_mu)], color=COLOR_INK, lw=GUIDE_WIDTH, ls="--")
    for x_mu in panel.marks:
        ax.plot([x_mu, x_mu], [-0.01, 0.01], color=COLOR_INK, lw=GUIDE_WIDTH, zorder=4)
    for c in panel.callouts:
        xs = [c.line_start[0], c.knee[0], c.mu]
        ys = [c.line_start[1], c.knee[1], impact.utility(m, p, c.option, c.mu)]
        ax.plot(xs, ys, color=LEADER_COLOR, lw=GUIDE_WIDTH, zorder=4)
        ax.text(
            *c.text_xy,
            rf"$u_{c.option}(\mu)$",
            ha="center",
            va="center",
            fontsize=TEXT_SIZE,
            zorder=6,
            path_effects=HALO,
        )
    for b in panel.braces:
        draw_brace(ax, b)

    # frame: x ticks every 0.2 (minor 0.05); y ticks at 0 and at the utilities at mu = 0 (left) and mu = 1 (right)
    ax.set_xticks(np.linspace(0, 1, 6), [f"{v:.1f}" for v in np.linspace(0, 1, 6)])
    ax.set_xticks(np.linspace(0, 1, 21), minor=True)
    ax.tick_params(axis="x", which="both", direction="in", top=False, labelsize=TEXT_SIZE, pad=3)
    ax.tick_params(axis="x", which="major", length=TICK_LENGTH)
    ax.tick_params(axis="x", which="minor", length=TICK_LENGTH / 2)
    ax.set_yticks([0, m.b_A - p, m.b_B - p], ["0", "", ""])
    ax.tick_params(axis="y", direction="in", length=TICK_LENGTH / 2, labelsize=TEXT_SIZE, pad=2)
    right_axis = ax.twinx()
    right_axis.set_ylim(*panel.ylim)
    right_axis.set_yticks([0, m.a_A - p, m.a_B - p], ["0", "", ""])
    right_axis.tick_params(axis="y", direction="in", length=TICK_LENGTH / 2, labelsize=TEXT_SIZE, pad=2)
    ax.set_xlabel(r"$\mu$", fontsize=TEXT_SIZE + 1, labelpad=1)
    return fig


def save_segmentation(panel: SegmentationPanel, outdir: Path) -> Path:
    """Draw the segmentation panel and save it as figure/fig_pre/<name>.pdf."""
    fig = draw_segmentation(panel)
    path = save_pdf(fig, Path(outdir) / "figure" / "fig_pre" / f"{panel.name}.pdf")
    plt.close(fig)
    return path
