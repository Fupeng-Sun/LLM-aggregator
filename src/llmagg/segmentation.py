"""Market segmentation of the static model: Section 3.1 of the paper (Proposition 1, Theorem 1, Appendix B.1).

Notation follows the paper. Providers A and B have feature qualities (a_A, b_A) and (a_B, b_B); in the baseline A is
strong in feature a and B in feature b (a_A > a_B, b_B > b_A), so the aggregator G offers the composite profile
(a_A, b_B). Cross-provider gaps delta_a = a_A - a_B and delta_b = b_B - b_A; within-provider differences
Delta_A = a_A - b_A and Delta_B = b_B - a_B. A user of type mu in [0, 1] has utilities
    u_A(mu) = mu a_A + (1 - mu) b_A - p,   u_B(mu) = mu a_B + (1 - mu) b_B - p,
    u_G(mu) = mu a_A + (1 - mu) b_B - p - kappa                                   (Eq. (u_agg)),
and the outside option O gives utility 0. Options are labelled "A", "B", "G" and "O".

The pairwise cutoffs mu_AB, mu_GB, mu_AG, the thresholds p*, p_L, p_R and kappa_bar are taken from
``llmagg.market.thresholds``; this module adds the participation cutoffs, the remaining price thresholds of
Appendix B.1, the case structure of Proposition 1 and Theorem 1, and the exact segmentation of [0, 1].
"""

import math
from dataclasses import dataclass

from llmagg.market import thresholds


@dataclass(frozen=True)
class Qualities:
    """Feature qualities (a_A, b_A, a_B, b_B) of providers A and B."""

    a_A: float
    b_A: float
    a_B: float
    b_B: float

    @property
    def delta_a(self) -> float:
        """Cross-provider quality gap delta_a = a_A - a_B."""
        return self.a_A - self.a_B

    @property
    def delta_b(self) -> float:
        """Cross-provider quality gap delta_b = b_B - b_A."""
        return self.b_B - self.b_A

    @property
    def Delta_A(self) -> float:
        """Within-provider difference Delta_A = a_A - b_A."""
        return self.a_A - self.b_A

    @property
    def Delta_B(self) -> float:
        """Within-provider difference Delta_B = b_B - a_B."""
        return self.b_B - self.a_B


@dataclass(frozen=True)
class Regime:
    """One case of Proposition 1 or Theorem 1.

    ``order`` lists the segments from left to right in mu (e.g. ("B", "O", "A") for B-O-A), ``split_points`` names
    the cutoffs between consecutive segments (e.g. ("mu_BO", "mu_AO")), and the case holds for p in
    ``price_range`` = (lo, hi].
    """

    case: str
    order: tuple[str, ...]
    split_points: tuple[str, ...]
    price_range: tuple[float, float]


@dataclass(frozen=True)
class Segment:
    """The users mu in [left, right] choose ``option``."""

    option: str
    left: float
    right: float


def utility_lines(q: Qualities, p: float, kappa: float | None = None) -> dict[str, tuple[float, float]]:
    """Intercept and slope in mu of u_A, u_B and, unless kappa is None, u_G (Eq. (u_agg)).

    u_X(mu) = intercept + slope * mu. Requires the baseline a_A > a_B and b_B > b_A, under which the aggregator's
    composite quality profile is (a_A, b_B).
    """
    if not (q.delta_a > 0 and q.delta_b > 0):
        raise ValueError("baseline a_A > a_B and b_B > b_A required")
    lines = {"A": (q.b_A - p, q.a_A - q.b_A), "B": (q.b_B - p, q.a_B - q.b_B)}
    if kappa is not None:
        lines["G"] = (q.b_B - p - kappa, q.a_A - q.b_B)
    return lines


def cutoffs(q: Qualities, p: float, kappa: float | None = None) -> dict[str, float]:
    """Competitive and participation cutoffs.

    mu_AB, mu_GB, mu_AG from Eq. (pairwise_cutoffs) (via ``llmagg.market.thresholds``) and
    mu_AO = (p - b_A)/Delta_A, mu_BO = (b_B - p)/Delta_B, mu_GO = (p + kappa - b_B)/(a_A - b_B) from
    Eq. (participation_cutoffs). A cutoff with a zero denominator is nan. Without an aggregator (kappa None) only
    mu_AB, mu_AO and mu_BO are returned.
    """
    th = thresholds(q.a_A, q.b_A, q.a_B, q.b_B, 0.0 if kappa is None else kappa)
    out = {
        "mu_AB": th["muAB"],
        "mu_AO": _ratio(p - q.b_A, q.Delta_A),
        "mu_BO": _ratio(q.b_B - p, q.Delta_B),
    }
    if kappa is not None:
        out["mu_GB"] = th["muGB"]
        out["mu_AG"] = th["muAG"]
        out["mu_GO"] = _ratio(p + kappa - q.b_B, q.a_A - q.b_B)
    return out


def price_thresholds(q: Qualities, kappa: float) -> dict[str, float]:
    """Price thresholds of Appendix B.1 and the viability threshold kappa_bar (Eq. (kappa_bar)).

    Keys: p_star (Eq. (pstar_rewrite)), p_L, p_R (Eq. (pR_rewrite)), p_B_low and p_B_high (underline/overline p_B^*,
    Eq. (pB_high)), p_G_low and p_GLR (underline p_G and p_GLR, Eq. (pG_low)), p_G_high (overline p_G, Eq. (pG_high)),
    kappa_bar.
    """
    th = thresholds(q.a_A, q.b_A, q.a_B, q.b_B, kappa)
    p_star, p_L, p_R = th["pstar"], th["pL"], th["pR"]
    if q.Delta_B > 0:
        p_B_low, p_B_high = p_star, q.b_B
        p_G_low, p_GLR, p_G_high = p_L, min(q.b_B, p_R), q.b_B
    else:
        p_B_low, p_B_high = q.b_B, p_star
        p_G_low, p_GLR, p_G_high = q.b_B, p_L, p_L
    return {
        "p_star": p_star,
        "p_L": p_L,
        "p_R": p_R,
        "p_B_low": p_B_low,
        "p_B_high": p_B_high,
        "p_G_low": p_G_low,
        "p_GLR": p_GLR,
        "p_G_high": p_G_high,
        "kappa_bar": th["kbar"],
    }


def regime_without_aggregator(q: Qualities, p: float) -> Regime:
    """The case of Proposition 1 (market segmentation without an aggregator) that applies at price p.

    The proposition assumes Delta_A > 0, a_A > b_B and kappa > kappa_bar; its segmentation does not depend on kappa.
    The knife-edge Delta_B = 0 (Remark knife_edge_DeltaB) is not covered.
    """
    _check_standing_assumptions(q)
    t = price_thresholds(q, 0.0)
    middle = (("B", "O", "A"), ("mu_BO", "mu_AO")) if q.Delta_B > 0 else (("O", "B", "A"), ("mu_BO", "mu_AB"))
    cases = [
        Regime("a", ("O",), (), (q.a_A, math.inf)),
        Regime("b", ("O", "A"), ("mu_AO",), (t["p_B_high"], q.a_A)),
        Regime("c", *middle, (t["p_B_low"], t["p_B_high"])),
        Regime("d", ("B", "A"), ("mu_AB",), (-math.inf, t["p_B_low"])),
    ]
    return _select(cases, p)


def regime_with_aggregator(q: Qualities, p: float, kappa: float) -> Regime:
    """The case of Theorem 1 (market segmentation with an aggregator) that applies at price p.

    The theorem assumes Delta_A > 0, a_A > b_B and 0 <= kappa <= kappa_bar. Case (c) O-G-A has an empty price range
    if Delta_B > 0 and p_R < b_B; case (d) B-O-A has a nonempty one only in that situation. The knife-edge
    Delta_B = 0 (Remark knife_edge_DeltaB) is not covered.
    """
    _check_standing_assumptions(q)
    t = price_thresholds(q, kappa)
    if not 0 <= kappa <= t["kappa_bar"]:
        raise ValueError("Theorem 1 requires 0 <= kappa <= kappa_bar")
    upper_c = max(q.b_B, t["p_R"])
    if q.Delta_B > 0:
        case_e = (("B", "O", "G", "A"), ("mu_BO", "mu_GO", "mu_AG"))
    else:
        case_e = (("O", "B", "G", "A"), ("mu_BO", "mu_GB", "mu_AG"))
    cases = [
        Regime("a", ("O",), (), (q.a_A, math.inf)),
        Regime("b", ("O", "A"), ("mu_AO",), (upper_c, q.a_A)),
        Regime("c", ("O", "G", "A"), ("mu_GO", "mu_AG"), (t["p_G_high"], upper_c)),
        Regime("d", ("B", "O", "A"), ("mu_BO", "mu_AO"), (t["p_GLR"], t["p_G_high"])),
        Regime("e", *case_e, (t["p_G_low"], t["p_GLR"])),
        Regime("f", ("B", "G", "A"), ("mu_GB", "mu_AG"), (-math.inf, t["p_G_low"])),
    ]
    return _select(cases, p)


def segments(q: Qualities, p: float, kappa: float | None = None) -> list[Segment]:
    """Exact market segmentation of [0, 1], computed from the model rather than from the results.

    Each user type picks the option with the highest utility if that utility is nonnegative and the outside option
    otherwise (without an aggregator if kappa is None). The utilities are affine in mu, so the choice is constant
    between consecutive pairwise crossings; segments are listed from left to right, adjacent equal choices merged.
    """
    lines = utility_lines(q, p, kappa) | {"O": (0.0, 0.0)}
    points = {0.0, 1.0}
    names = list(lines)
    for i, x in enumerate(names):
        for y in names[i + 1 :]:
            slope_gap = lines[x][1] - lines[y][1]
            if slope_gap != 0:
                mu = (lines[y][0] - lines[x][0]) / slope_gap
                if 0 < mu < 1:
                    points.add(mu)
    grid = sorted(points)
    out: list[Segment] = []
    for left, right in zip(grid[:-1], grid[1:], strict=True):
        mid = 0.5 * (left + right)
        utility = {x: c0 + c1 * mid for x, (c0, c1) in lines.items() if x != "O"}
        best = max(utility, key=utility.__getitem__)
        choice = best if utility[best] >= 0 else "O"
        if out and out[-1].option == choice:
            out[-1] = Segment(choice, out[-1].left, right)
        else:
            out.append(Segment(choice, left, right))
    return out


def _ratio(num: float, den: float) -> float:
    return num / den if den != 0 else math.nan


def _check_standing_assumptions(q: Qualities) -> None:
    if not (q.Delta_A > 0 and q.a_A > q.b_B):
        raise ValueError("Proposition 1 and Theorem 1 assume Delta_A > 0 and a_A > b_B")
    if q.Delta_B == 0:
        raise ValueError("the knife-edge Delta_B = 0 is not covered")


def _select(cases: list[Regime], p: float) -> Regime:
    hits = [c for c in cases if c.price_range[0] < p <= c.price_range[1]]
    if len(hits) != 1:
        raise AssertionError(f"price p = {p} lies in {len(hits)} cases; the price ranges should partition the line")
    return hits[0]
