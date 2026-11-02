"""Drawing and checking code shared by the utility and segmentation panels of Figures 3(a), 4 and 5.

Also the market of Figures 3(a) and 5(b)-(d): qualities (a_A, b_A, a_B, b_B) = (1.4, 0.8, 0.9, 1.25) and aggregator
friction kappa = 0.15 (the earlier Mathematica notebook (not in this repository)).
"""

from dataclasses import dataclass
from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.axes import Axes
from matplotlib.figure import Figure
from matplotlib.ticker import FormatStrFormatter, MultipleLocator

from llmagg.market import revenues
from llmagg.plotting import COLOR_INK, PROVIDER_COLORS, use_style
from llmagg.segmentation import (
    Qualities,
    Segment,
    cutoffs,
    price_thresholds,
    regime_with_aggregator,
    regime_without_aggregator,
    segments,
    utility_lines,
)

QUALITIES = Qualities(a_A=1.4, b_A=0.8, a_B=0.9, b_B=1.25)
KAPPA = 0.15

TEXT_WIDTH_PT = 469.755  # \textwidth of the paper (informs3.cls with mnsc)
TOL = 1e-12
KAPPA_MARGIN = 1e-9  # relative margin above kappa_bar at which Figure 4 is checked with an aggregator

# ---------------------------------------------------------------------------------------------------------------------
# the utility panel (Figures 3(a), 4 and 5)
# ---------------------------------------------------------------------------------------------------------------------
TICK_LABELS = {  # option -> (label of u_X(0) on the left frame, label of u_X(1) on the right frame)
    "A": (r"$b_A{-}p$", r"$a_A{-}p$"),
    "B": (r"$b_B{-}p$", r"$a_B{-}p$"),
    "G": (r"$b_B{-}\kappa{-}p$", r"$a_A{-}\kappa{-}p$"),
}
SEGMENT_LETTERS = {x: rf"$\mathcal{{{x}}}$" for x in "ABGO"}
FILL_ALPHA = 0.2  # opacity of the area under the chosen utility
LEADER_COLOR = "0.6"  # callout leaders
LEADER_WIDTH = 0.5
CUTOFF_WIDTH = 0.6  # dashed competitive cutoffs and participation ticks
CUTOFF_DASHES = (2.0, 1.5)
PARTICIPATION_TICK_PT = 4.0  # length of the tick at a participation cutoff
TICK_LENGTH_PT = 2.5
LABEL_GAP = 0.95  # minimum distance between two frame labels, in units of the font size
LETTER_OFFSET_PT = 0.5  # gap between the line u = 0 and the segment letters below it

Callouts = dict[str, tuple[float, tuple[float, float]]]  # option -> (mu where the leader meets u_X, label centre)


@dataclass(frozen=True)
class PanelLayout:
    """Printed size and frame margins of a panel in points, and the font size of all its text."""

    width_pt: float
    height_pt: float
    margins_pt: tuple[float, float, float, float]  # left, right, bottom, top
    font_size: float
    mu_ticks: tuple[float, float]  # spacing of the labelled and of the minor ticks on the mu axis

    @property
    def frame_height_pt(self) -> float:
        """Height of the frame in points."""
        return self.height_pt - self.margins_pt[2] - self.margins_pt[3]


@dataclass(frozen=True)
class SegmentationPanel:
    """One panel of Figure 4 or 5: the market, the price, and the case and segmentation named in its subcaption."""

    file: Path
    qualities: Qualities
    kappa: float | None  # None: no aggregator (Figure 4, Proposition 1)
    price: float
    case: str
    order: tuple[str, ...]
    u_range: tuple[float, float]  # frame range (layout)
    callouts: Callouts  # layout


def new_panel_figure(layout: PanelLayout) -> tuple[Figure, Axes]:
    """A figure of the printed size with one frame placed by the margins (no automatic layout)."""
    left, right, bottom, top = layout.margins_pt
    fig = plt.figure(figsize=(layout.width_pt / 72, layout.height_pt / 72))
    ax = fig.add_axes(
        (
            left / layout.width_pt,
            bottom / layout.height_pt,
            1 - (left + right) / layout.width_pt,
            1 - (bottom + top) / layout.height_pt,
        )
    )
    return fig, ax


def frame_ticks(q: Qualities, p: float, kappa: float | None) -> tuple[list[tuple[float, str]], list[tuple[float, str]]]:
    """Left and right frame ticks (value, label): the endpoints u_X(0) and u_X(1) of the drawn utilities."""
    lines = utility_lines(q, p, kappa)
    left = [(c0, TICK_LABELS[x][0]) for x, (c0, _) in lines.items()]
    right = [(c0 + c1, TICK_LABELS[x][1]) for x, (c0, c1) in lines.items()]
    return left, right


def spread(values: list[float], gap: float) -> list[float]:
    """Positions for labels at sorted ``values``, moved apart to at least ``gap``; each cluster stays centred."""
    clusters: list[list[float]] = []
    for v in values:
        clusters.append([v])
        while len(clusters) > 1:
            lower, upper = clusters[-2], clusters[-1]
            top_of_lower = sum(lower) / len(lower) + (len(lower) - 1) * gap / 2
            bottom_of_upper = sum(upper) / len(upper) - (len(upper) - 1) * gap / 2
            if bottom_of_upper - top_of_lower >= gap:
                break
            clusters[-2:] = [lower + upper]
    out = []
    for c in clusters:
        centre = sum(c) / len(c)
        out += [centre + (i - (len(c) - 1) / 2) * gap for i in range(len(c))]
    return out


def _label_frame(ax: Axes, ticks: list[tuple[float, str]], side: str, layout: PanelLayout) -> None:
    """Labels outside the frame at the tick values, moved apart where they would overlap."""
    y0, y1 = ax.get_ylim()
    pt_per_unit = layout.frame_height_pt / (y1 - y0)
    ticks = sorted(ticks)
    positions = spread([v * pt_per_unit for v, _ in ticks], LABEL_GAP * layout.font_size)
    x, ha, sign = (0.0, "right", -1) if side == "left" else (1.0, "left", 1)
    for (value, label), pos in zip(ticks, positions, strict=True):
        ax.annotate(
            label,
            xy=(x, value),
            xycoords=("axes fraction", "data"),
            xytext=(sign * TICK_LENGTH_PT, pos - value * pt_per_unit),
            textcoords="offset points",
            ha=ha,
            va="center",
            fontsize=layout.font_size,
        )


def _mark_segmentation(ax: Axes, lines: dict, segmentation: list[Segment], font_size: float) -> None:
    """Shade the area under the chosen utility, mark the cutoffs and write the segment letters below u = 0."""
    ax.axhline(0, color=COLOR_INK, lw=plt.rcParams["axes.linewidth"], zorder=2)
    for seg in segmentation:
        if seg.option != "O":
            c0, c1 = lines[seg.option]
            ends = [seg.left, seg.right]
            heights = [c0 + c1 * mu for mu in ends]
            ax.fill_between(ends, 0, heights, color=PROVIDER_COLORS[seg.option], alpha=FILL_ALPHA, lw=0, zorder=1)
        ax.annotate(
            SEGMENT_LETTERS[seg.option],
            xy=(0.5 * (seg.left + seg.right), 0),
            xytext=(0, -LETTER_OFFSET_PT),
            textcoords="offset points",
            ha="center",
            va="top",
            fontsize=font_size,
            zorder=5,
        )
    for before, after in zip(segmentation[:-1], segmentation[1:], strict=True):
        mu = before.right
        if "O" in (before.option, after.option):  # participation cutoff: short tick on u = 0
            ax.plot(
                [mu],
                [0],
                marker="|",
                markersize=PARTICIPATION_TICK_PT,
                markeredgewidth=CUTOFF_WIDTH,
                color=COLOR_INK,
                zorder=4,
            )
        else:  # competitive cutoff: dashed line from u = 0 up to the crossing
            c0, c1 = lines[before.option]
            ax.plot([mu, mu], [0, c0 + c1 * mu], color=COLOR_INK, lw=CUTOFF_WIDTH, dashes=CUTOFF_DASHES, zorder=4)


def draw_utility_panel(
    ax: Axes,
    q: Qualities,
    p: float,
    kappa: float | None,
    u_range: tuple[float, float],
    callouts: Callouts,
    layout: PanelLayout,
    segmentation: list[Segment] | None = None,
) -> None:
    """Draw u_A, u_B and (unless kappa is None) u_G on mu in [0, 1] with symbolic frame ticks and callouts.

    With a ``segmentation``, also shade the area under the chosen utility on each segment, mark each boundary
    (dashed line up to the crossing of two subscribed options, short tick at a participation cutoff) and write the
    segment letters below the line u = 0.
    """
    lines = utility_lines(q, p, kappa)
    ax.set_xlim(0, 1)
    ax.set_ylim(*u_range)
    for x, (c0, c1) in lines.items():
        ax.plot([0, 1], [c0, c0 + c1], color=PROVIDER_COLORS[x], solid_capstyle="butt", zorder=3)
    if segmentation is not None:
        _mark_segmentation(ax, lines, segmentation, layout.font_size)
    for x, (mu, label) in callouts.items():
        c0, c1 = lines[x]
        ax.annotate(
            rf"$u_{x}(\mu)$",
            xy=(mu, c0 + c1 * mu),
            xytext=label,
            ha="center",
            va="center",
            fontsize=layout.font_size,
            zorder=6,
            arrowprops={"arrowstyle": "-", "color": LEADER_COLOR, "lw": LEADER_WIDTH, "shrinkA": 1, "shrinkB": 0},
        )
    left, right = frame_ticks(q, p, kappa)
    if u_range[0] <= 0 <= u_range[1]:
        left.append((0.0, "0"))
        right.append((0.0, "0"))
    ax.set_yticks(sorted(v for v, _ in left))
    twin = ax.twinx()
    twin.set_ylim(*u_range)
    twin.set_yticks(sorted(v for v, _ in right))
    for a in (ax, twin):
        a.tick_params(axis="y", direction="in", length=TICK_LENGTH_PT, labelleft=False, labelright=False)
    _label_frame(ax, left, "left", layout)
    _label_frame(ax, right, "right", layout)
    ax.xaxis.set_major_locator(MultipleLocator(layout.mu_ticks[0]))
    ax.xaxis.set_minor_locator(MultipleLocator(layout.mu_ticks[1]))
    ax.xaxis.set_major_formatter(FormatStrFormatter("%.1f"))
    ax.tick_params(axis="x", which="both", direction="in", top=False, labelsize=layout.font_size, pad=2)
    ax.tick_params(axis="x", which="major", length=TICK_LENGTH_PT)
    ax.tick_params(axis="x", which="minor", length=TICK_LENGTH_PT / 2)
    ax.set_xlabel(r"$\mu$", fontsize=layout.font_size, labelpad=1)


def draw_segmentation_panel(panel: SegmentationPanel, layout: PanelLayout) -> Figure:
    """One panel of Figure 4 or 5 as a matplotlib figure."""
    use_style()
    fig, ax = new_panel_figure(layout)
    segs = segments(panel.qualities, panel.price, panel.kappa)
    draw_utility_panel(ax, panel.qualities, panel.price, panel.kappa, panel.u_range, panel.callouts, layout, segs)
    return fig


def check_frame(q: Qualities, p: float, kappa: float | None, u_range: tuple[float, float], callouts: Callouts) -> None:
    """The frame ticks are the paper's expressions b_A-p, ..., a_A-kappa-p; ticks and callouts lie inside the frame."""
    left, right = frame_ticks(q, p, kappa)
    expected_left = [q.b_A - p, q.b_B - p] + ([] if kappa is None else [q.b_B - kappa - p])
    expected_right = [q.a_A - p, q.a_B - p] + ([] if kappa is None else [q.a_A - kappa - p])
    assert all(abs(v - e) < TOL for (v, _), e in zip(left, expected_left, strict=True))
    assert all(abs(v - e) < TOL for (v, _), e in zip(right, expected_right, strict=True))
    assert all(u_range[0] <= v <= u_range[1] for v, _ in left + right), "a frame tick lies outside the frame"
    lines = utility_lines(q, p, kappa)
    for x, (mu, (label_mu, label_u)) in callouts.items():
        c0, c1 = lines[x]
        assert u_range[0] <= c0 + c1 * mu <= u_range[1], f"the leader of u_{x} starts outside the frame"
        assert 0 <= label_mu <= 1 and u_range[0] <= label_u <= u_range[1], f"the label of u_{x} lies outside the frame"


def check_segmentation_panel(panel: SegmentationPanel, name: str) -> list[str]:
    """Check one panel of Figure 4 or 5 against Proposition 1 or Theorem 1 and the exact model."""
    q, p, kappa = panel.qualities, panel.price, panel.kappa
    if kappa is None:
        result = "Proposition 1"
        regime = regime_without_aggregator(q, p)
    else:
        result = "Theorem 1"
        regime = regime_with_aggregator(q, p, kappa)
    order = "-".join(panel.order)
    assert (regime.case, regime.order) == (panel.case, panel.order), f"{name}: {result} gives case ({regime.case})"
    segs = segments(q, p, kappa)
    assert tuple(s.option for s in segs) == regime.order, f"{name}: the exact segmentation differs from {result}"
    mu = cutoffs(q, p, kappa)
    for seg, split in zip(segs[:-1], regime.split_points, strict=True):
        assert abs(seg.right - mu[split]) < TOL, f"{name}: boundary {seg.right} is not {split} = {mu[split]}"
    share = {x: sum(s.right - s.left for s in segs if s.option == x) for x in "ABG"}
    with_aggregator = kappa is not None
    _, _, _, share_A, share_B, share_G = revenues(
        q.a_A, q.b_A, q.a_B, q.b_B, p, kappa if with_aggregator else 0.0, 0.0, 0.0, 1.0, with_aggregator
    )
    assert max(abs(share["A"] - share_A), abs(share["B"] - share_B), abs(share["G"] - share_G)) < 1e-9
    lo, hi = regime.price_range
    lines = [
        f"{name}: {result} case ({regime.case}) for p = {p} in ({lo:.4g}, {hi:.4g}]: {order}, as in the subcaption",
        f"{name}: exact segmentation {order}, split at "
        + ", ".join(f"{s} = {mu[s]:.4g}" for s in regime.split_points)
        + "; shares match llmagg.market.revenues",
    ]
    if kappa is None:  # Proposition 1 assumes kappa > kappa_bar, where G attracts nobody (G only shrinks in kappa)
        kappa_bar = price_thresholds(q, 0.0)["kappa_bar"]
        assert segments(q, p, kappa_bar * (1 + KAPPA_MARGIN)) == segs
        lines.append(
            f"{name}: the segmentation is the same with an aggregator at any kappa > kappa_bar = {kappa_bar:.4g}"
        )
    check_frame(q, p, kappa, panel.u_range, panel.callouts)
    return lines
