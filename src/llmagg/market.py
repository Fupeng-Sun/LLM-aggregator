"""Static market model of Sun, Liu and Wiesemann, "LLM Competition and the Aggregator Economy".

State x = (a_A, b_A, a_B, b_B).  Users mu ~ U[0,1] pick the option with the highest nonnegative utility among
    u_A(mu) = mu a_A + (1-mu) b_A - p,   u_B(mu) = mu a_B + (1-mu) b_B - p,
    u_G(mu) = mu a* + (1-mu) b* - p - kappa,   a* = max(a_A, a_B), b* = max(b_A, b_B),
and the outside option (utility 0).  Revenues (Eq. (provider_revenues) and (RG) of the paper):
    R_A = p|A| + w_A rho_0 int_G mu dmu          (if A leads in dimension a; otherwise B receives this term)
    R_B = p|B| + w_B rho_0 int_G (1-mu) dmu      (if B leads in dimension b; otherwise A receives this term)
    R_G = p|G| - (API payments).
The generic implementation integrates the exact upper envelope of the affine utilities, so it is valid for every
quality profile: switched comparative advantages, dominance, and an aggregator that attracts nobody.
"""

import numpy as np
from numba import njit

# ---------------------------------------------------------------------------------------------------------------
# exact envelope
# ---------------------------------------------------------------------------------------------------------------


@njit(cache=True)
def _segments(aA, bA, aB, bB, p, kappa, agg):
    """Breakpoints of the envelope on [0,1].  Lines: 0 = A, 1 = B, 2 = G (if agg), 3 = outside (zero)."""
    nl = 3 if agg else 2
    c0 = np.empty(4)
    c1 = np.empty(4)  # u_k(mu) = c0[k] + c1[k] mu
    c0[0] = bA - p
    c1[0] = aA - bA
    c0[1] = bB - p
    c1[1] = aB - bB
    astar = aA if aA >= aB else aB
    bstar = bA if bA >= bB else bB
    c0[2] = bstar - p - kappa
    c1[2] = astar - bstar
    c0[3] = 0.0
    c1[3] = 0.0
    pts = np.empty(16)
    n = 0
    pts[n] = 0.0
    n += 1
    pts[n] = 1.0
    n += 1
    for i in range(nl + 1):
        ii = i if i < nl else 3
        for j in range(i + 1, nl + 1):
            jj = j if j < nl else 3
            d = c1[ii] - c1[jj]
            if d != 0.0:
                m = (c0[jj] - c0[ii]) / d
                if m > 0.0 and m < 1.0:
                    pts[n] = m
                    n += 1
    pts = np.sort(pts[:n])
    return pts, n, c0, c1, nl


@njit(cache=True)
def revenues(aA, bA, aB, bB, p, kappa, wA, wB, rho0, agg):
    """Returns (R_A, R_B, R_G, |A|, |B|, |G|)."""
    pts, n, c0, c1, nl = _segments(aA, bA, aB, bB, p, kappa, agg)
    shA = 0.0
    shB = 0.0
    shG = 0.0
    intmu = 0.0
    int1mu = 0.0  # integrals over the aggregator segment
    for s in range(n - 1):
        l = pts[s]
        r = pts[s + 1]
        if r - l <= 0.0:
            continue
        m = 0.5 * (l + r)
        bv = -1.0
        for k in range(nl):
            v = c0[k] + c1[k] * m
            if v >= 0.0 and v > bv:
                bv = v
        if bv < 0.0:
            continue  # outside option
        # options tied at the maximum (identical utility lines) share the segment equally
        ntie = 0
        for k in range(nl):
            if c0[k] + c1[k] * m >= bv - 1e-12 * (1.0 + abs(bv)):
                ntie += 1
        f = 1.0 / ntie
        for k in range(nl):
            if c0[k] + c1[k] * m >= bv - 1e-12 * (1.0 + abs(bv)):
                if k == 0:
                    shA += f * (r - l)
                elif k == 1:
                    shB += f * (r - l)
                else:
                    shG += f * (r - l)
                    intmu += f * 0.5 * (r * r - l * l)
                    int1mu += f * ((r - l) - 0.5 * (r * r - l * l))
    payA = 0.0
    payB = 0.0  # API payments received by each provider (split at equal quality)
    if aA > aB:
        payA += wA * rho0 * intmu
    elif aA < aB:
        payB += wB * rho0 * intmu
    else:
        payA += 0.5 * wA * rho0 * intmu
        payB += 0.5 * wB * rho0 * intmu
    if bB > bA:
        payB += wB * rho0 * int1mu
    elif bB < bA:
        payA += wA * rho0 * int1mu
    else:
        payA += 0.5 * wA * rho0 * int1mu
        payB += 0.5 * wB * rho0 * int1mu
    RA = p * shA + payA
    RB = p * shB + payB
    RG = p * shG - payA - payB
    return RA, RB, RG, shA, shB, shG


@njit(cache=True)
def rev_pair(aA, bA, aB, bB, p, kappa, wA, wB, rho0, agg):
    """Revenues (R_A, R_B) at a quality profile (see `revenues`)."""
    RA, RB, RG, sA, sB, sG = revenues(aA, bA, aB, bB, p, kappa, wA, wB, rho0, agg)
    return RA, RB


@njit(cache=True)
def rev_arrays(X, p, kappa, wA, wB, rho0, agg):
    """Vectorised revenues for an (N,4) array of states.  Returns (N,2) array [R_A, R_B]."""
    N = X.shape[0]
    out = np.empty((N, 2))
    for i in range(N):
        RA, RB = rev_pair(X[i, 0], X[i, 1], X[i, 2], X[i, 3], p, kappa, wA, wB, rho0, agg)
        out[i, 0] = RA
        out[i, 1] = RB
    return out


# ---------------------------------------------------------------------------------------------------------------
# myopic marginal-revenue heuristic, Eq. (phiA_myopic_bangbang): right derivatives, ties to feature a
# ---------------------------------------------------------------------------------------------------------------

H_FD = 1e-7


@njit(cache=True)
def own_derivs(aA, bA, aB, bB, p, kappa, wA, wB, rho0, agg, who):
    """Right partial derivatives of R_X w.r.t. (a_X, b_X) by forward differences; who = 0 (A) or 1 (B)."""
    h = H_FD
    if who == 0:
        R0, _ = rev_pair(aA, bA, aB, bB, p, kappa, wA, wB, rho0, agg)
        Ra, _ = rev_pair(aA + h, bA, aB, bB, p, kappa, wA, wB, rho0, agg)
        Rb, _ = rev_pair(aA, bA + h, aB, bB, p, kappa, wA, wB, rho0, agg)
    else:
        _, R0 = rev_pair(aA, bA, aB, bB, p, kappa, wA, wB, rho0, agg)
        _, Ra = rev_pair(aA, bA, aB + h, bB, p, kappa, wA, wB, rho0, agg)
        _, Rb = rev_pair(aA, bA, aB, bB + h, p, kappa, wA, wB, rho0, agg)
    return (Ra - R0) / h, (Rb - R0) / h


@njit(cache=True)
def myopic_lambda(aA, bA, aB, bB, p, kappa, wA, wB, rho0, agg, who):
    """Budget share invested in feature a under the myopic heuristic (1.0 = feature a, 0.0 = feature b)."""
    da, db = own_derivs(aA, bA, aB, bB, p, kappa, wA, wB, rho0, agg, who)
    if who == 0:
        qa, qb = aA, bA
    else:
        qa, qb = aB, bB
    # ties (within rounding of the finite difference) go to feature a, as in the paper
    if da / qa >= db / qb - 1e-9:
        return 1.0
    return 0.0


# ---------------------------------------------------------------------------------------------------------------
# closed forms from the paper (used only for validation)
# ---------------------------------------------------------------------------------------------------------------


def thresholds(aA, bA, aB, bB, kappa):
    """Closed-form cutoffs mu_AB, mu_GB, mu_AG and thresholds p*, p_L, p_R, kappa_bar (Theorem 1, Appendix B.1)."""
    da, db = aA - aB, bB - bA
    DA, DB = aA - bA, bB - aB
    muAB = db / (da + db)
    muGB = kappa / da
    muAG = 1 - kappa / db
    pstar = (da * bA + db * aA) / (da + db)
    pL = bB - DB * kappa / da
    pR = aA - DA * kappa / db
    kbar = da * db / (da + db)
    return dict(muAB=muAB, muGB=muGB, muAG=muAG, pstar=pstar, pL=pL, pR=pR, kbar=kbar)


def kappa_star(a, b, da, db, w, p, rho0):
    """Eq. (kappa_myopic_fullcov_agg)."""
    if da <= 0 or db <= 0 or w <= 0:
        return np.inf
    den = w * rho0 * (b * db**3 - a * da**3)
    if den <= 0:
        return np.inf
    return a * da**3 * db * (p - w * rho0) / den


def delta0(a, b, p, w, rho0):
    """Largest real root of f*(x) = x^3 - (a/b)(p/(w rho0) - 1)(x+1) - a/b, Eq. (f)."""
    k = a / b
    coeffs = [1.0, 0.0, -k * (p / (w * rho0) - 1), -k * (p / (w * rho0) - 1) - k]
    roots = np.roots(coeffs)
    real = roots[np.abs(roots.imag) < 1e-9].real
    return real.max()
