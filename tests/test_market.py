"""The exact market model (`llmagg.market`) against the paper's closed forms.

Proposition 1 and Theorem 1 (segmentation, shares and revenues), Proposition 4 (heuristic without the aggregator) and
Theorem 5 (heuristic with the aggregator), on random quality profiles that satisfy the main text's assumptions.
"""

import numpy as np
import pytest

from llmagg.market import delta0, kappa_star, myopic_lambda, revenues, thresholds

P, WA, WB, RHO0 = 1.0, 0.5, 0.5, 1.0
N = 1500


def close(a: float, b: float, tol: float = 1e-9) -> bool:
    return abs(a - b) <= tol * max(1.0, abs(a), abs(b))


def random_states(seed: int, n: int = N):
    rng = np.random.default_rng(seed)
    while n:
        a_A, b_B = rng.uniform(2.5, 5, 2)
        b_A, a_B = rng.uniform(1.2, 2.4, 2)
        if a_A > a_B and b_B > b_A and a_A > b_A and a_A > b_B and b_B > a_B:
            n -= 1
            yield rng, (a_A, b_A, a_B, b_B)


def test_saturated_market_with_and_without_aggregator():
    """Theorem 1(f) (B-G-A for p <= p_L) and Proposition 1(d) (B-A for p <= p*): shares and revenues."""
    for rng, (a_A, b_A, a_B, b_B) in random_states(0):
        kappa = rng.uniform(0.02, 0.98) * thresholds(a_A, b_A, a_B, b_B, 0.0)["kbar"]
        th = thresholds(a_A, b_A, a_B, b_B, kappa)
        p = rng.uniform(0.3, 1.0) * th["pL"]
        if p <= max(WA, WB) * RHO0:
            continue
        R_A, R_B, _, s_A, s_B, s_G = revenues(a_A, b_A, a_B, b_B, p, kappa, WA, WB, RHO0, True)
        mu_AB, mu_GB, mu_AG = th["muAB"], th["muGB"], th["muAG"]
        assert close(s_B, mu_GB) and close(s_G, mu_AG - mu_GB) and close(s_A, 1 - mu_AG)
        assert close(R_A, p * (1 - mu_AG) + WA * RHO0 * (mu_AG**2 - mu_GB**2) / 2)
        assert close(R_B, p * mu_GB + WB * RHO0 * ((mu_AG - mu_GB) - (mu_AG**2 - mu_GB**2) / 2))
        if p <= th["pstar"]:
            da, db = a_A - a_B, b_B - b_A
            R_A0, R_B0, _, s_A0, _, _ = revenues(a_A, b_A, a_B, b_B, p, kappa, WA, WB, RHO0, False)
            assert close(s_A0, 1 - mu_AB) and close(R_A0, p * da / (da + db)) and close(R_B0, p * db / (da + db))


def test_partially_covered_markets():
    """Theorem 1 (B-O-G-A, O-G-A) and Proposition 1 (B-O-A): cutoffs and shares."""
    for rng, (a_A, b_A, a_B, b_B) in random_states(1):
        kappa = rng.uniform(0.02, 0.98) * thresholds(a_A, b_A, a_B, b_B, 0.0)["kbar"]
        th = thresholds(a_A, b_A, a_B, b_B, kappa)
        lo, hi = th["pL"], min(b_B, th["pR"])
        if hi > lo:
            p = rng.uniform(lo, hi)
            _, _, _, s_A, s_B, s_G = revenues(a_A, b_A, a_B, b_B, p, kappa, WA, WB, RHO0, True)
            mu_BO, mu_GO = (b_B - p) / (b_B - a_B), (p + kappa - b_B) / (a_A - b_B)
            assert close(s_B, mu_BO) and close(s_G, th["muAG"] - mu_GO) and close(s_A, 1 - th["muAG"])
            _, _, _, s_A0, s_B0, s_G0 = revenues(a_A, b_A, a_B, b_B, p, kappa, WA, WB, RHO0, False)
            assert close(s_B0, mu_BO) and close(s_A0, 1 - (p - b_A) / (a_A - b_A)) and close(s_G0, 0.0)
        if th["pR"] > b_B:
            p = rng.uniform(b_B, th["pR"])
            _, _, _, s_A, s_B, s_G = revenues(a_A, b_A, a_B, b_B, p, kappa, WA, WB, RHO0, True)
            mu_GO = (p + kappa - b_B) / (a_A - b_B)
            assert close(s_B, 0.0) and close(s_G, th["muAG"] - mu_GO) and close(s_A, 1 - th["muAG"])


def test_inactive_aggregator():
    """The aggregator attracts nobody for p >= p_R or kappa > kappa_bar."""
    for rng, (a_A, b_A, a_B, b_B) in random_states(2, 500):
        th0 = thresholds(a_A, b_A, a_B, b_B, 0.0)
        kappa = rng.uniform(0.02, 0.98) * th0["kbar"]
        p = thresholds(a_A, b_A, a_B, b_B, kappa)["pR"] + rng.uniform(0, 0.3)
        assert close(revenues(a_A, b_A, a_B, b_B, p, kappa, WA, WB, RHO0, True)[5], 0.0)
        s_G = revenues(a_A, b_A, a_B, b_B, 0.5 * th0["pstar"], 1.01 * th0["kbar"], WA, WB, RHO0, True)[5]
        assert close(s_G, 0.0)


def test_heuristic_without_aggregator():
    """Proposition 4: invest in a iff delta_b/delta_a >= a_X/b_X (full coverage, ties to a)."""
    for rng, (a_A, b_A, a_B, b_B) in random_states(3):
        p = rng.uniform(0.6, 1.0) * thresholds(a_A, b_A, a_B, b_B, 0.0)["pstar"]
        ratio = (b_B - b_A) / (a_A - a_B)
        expected = (float(ratio >= a_A / b_A), float(ratio >= a_B / b_B))
        got = tuple(myopic_lambda(a_A, b_A, a_B, b_B, p, 0.0, WA, WB, RHO0, False, who) for who in (0, 1))
        assert got == expected


@pytest.mark.parametrize("seed", [4, 5])
def test_heuristic_with_aggregator(seed):
    """Theorem 5 (thm:myopic_full_coverage_agg_selector_rewrite) in the saturated market with an active aggregator."""
    for rng, (a_A, b_A, a_B, b_B) in random_states(seed):
        kappa = rng.uniform(0.02, 0.98) * thresholds(a_A, b_A, a_B, b_B, 0.0)["kbar"]
        th = thresholds(a_A, b_A, a_B, b_B, kappa)
        p = rng.uniform(0.5, 0.98) * min(th["pL"], th["pstar"])
        if p <= max(WA, WB) * RHO0:
            continue
        da, db = a_A - a_B, b_B - b_A
        q = db / da
        k_A, k_B = kappa_star(a_A, b_A, da, db, WA, p, RHO0), kappa_star(b_B, a_B, db, da, WB, p, RHO0)
        if q < 1 / delta0(b_B, a_B, p, WB, RHO0):
            expected = (0.0, 1.0 if kappa <= k_B else 0.0)
        elif q < delta0(a_A, b_A, p, WA, RHO0):
            expected = (0.0, 1.0)
        else:
            expected = (0.0 if kappa < k_A else 1.0, 1.0)
        got = tuple(myopic_lambda(a_A, b_A, a_B, b_B, p, kappa, WA, WB, RHO0, True, who) for who in (0, 1))
        assert got == expected
