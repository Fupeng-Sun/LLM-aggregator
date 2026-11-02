"""Impact of the aggregator on the providers' revenues (Section 3.3, Theorem ``thm:impact_aggregator``).

Every price, friction and wholesale-revenue threshold that the theorem and its auxiliary results use, in the paper's
notation:

- competitive and participation cutoffs, Eqs. (pairwise_cutoffs) and (participation_cutoffs);
- the viability threshold kappa_bar, Eq. (kappa_bar), and the price thresholds p*, p_L, p_R of Appendix B.1,
  Eqs. (pstar_rewrite) and (pR_rewrite);
- the switching prices underline-p_A, underline-p_B, underline-p_A^L, Eqs. (pBbar_impact) and (pALbar_impact), and the
  friction thresholds kappa_A^dagger, kappa_A^ddagger, kappa_A^0, Eq. (kappaA_ddagger_impact), all in Appendix
  ``subsubsec:aux_notation_impact_aggregator``;
- the wholesale-revenue threshold functions Psi_A, Psi_B, Omega_A(p), Omega_B(p), Phi_A^L(p), Eqs. (PsiA_impact),
  (PsiB_impact) and (PhiAL_impact). Each is implemented twice: by its defining ratio of cutoffs (``*_from_cutoffs``)
  and by the simplified closed form of Eqs. (PsiA_impact), (PsiB_impact) and Propositions ``prop:property_Omega_AB``(i)
  and ``prop:property_phi_AL``(i).

Theorem ``thm:impact_aggregator`` states Delta R_X >= 0 iff w_X rho_0 >= threshold, so threshold / rho_0 is the root
of Delta R_X(w_X) = R_X^G - R_X^0. ``breakeven_wholesale_price`` computes that root independently, from the exact
market model ``llmagg.market.revenues``.

The functions of a price ``p`` accept floats or numpy arrays.
"""

from dataclasses import dataclass

from llmagg.market import revenues


@dataclass(frozen=True)
class Params:
    """Feature qualities (a_A, b_A, a_B, b_B) and aggregator friction kappa."""

    a_A: float
    b_A: float
    a_B: float
    b_B: float
    kappa: float

    @property
    def delta_a(self) -> float:
        """Cross-provider quality gap delta_a := a_A - a_B."""
        return self.a_A - self.a_B

    @property
    def delta_b(self) -> float:
        """Cross-provider quality gap delta_b := b_B - b_A."""
        return self.b_B - self.b_A

    @property
    def Delta_A(self) -> float:
        """Within-provider difference Delta_A := a_A - b_A."""
        return self.a_A - self.b_A

    @property
    def Delta_B(self) -> float:
        """Within-provider difference Delta_B := b_B - a_B."""
        return self.b_B - self.a_B


# ---------------------------------------------------------------------------------------------------------------
# utilities, cutoffs and price thresholds (Sections 3.1-3.2 and Appendix B.1)
# ---------------------------------------------------------------------------------------------------------------


def utility(m: Params, p: float, option: str, mu):
    """Utility of user type mu: u_A = mu a_A + (1-mu) b_A - p, u_B = mu a_B + (1-mu) b_B - p, Eq. (u_provider);
    u_G = mu a_A + (1-mu) b_B - p - kappa, Eq. (u_agg) (baseline profile a_A > a_B, b_B > b_A)."""
    a, b, friction = {"A": (m.a_A, m.b_A, 0.0), "B": (m.a_B, m.b_B, 0.0), "G": (m.a_A, m.b_B, m.kappa)}[option]
    return mu * a + (1 - mu) * b - p - friction


def mu_AB(m: Params) -> float:
    """Competitive cutoff mu_AB = delta_b / (delta_a + delta_b), Eq. (pairwise_cutoffs)."""
    return m.delta_b / (m.delta_a + m.delta_b)


def mu_GB(m: Params) -> float:
    """Competitive cutoff mu_GB = kappa / delta_a, Eq. (pairwise_cutoffs)."""
    return m.kappa / m.delta_a


def mu_AG(m: Params) -> float:
    """Competitive cutoff mu_AG = 1 - kappa / delta_b, Eq. (pairwise_cutoffs)."""
    return 1 - m.kappa / m.delta_b


def mu_AO(m: Params, p):
    """Participation cutoff mu_AO = (p - b_A) / Delta_A, Eq. (participation_cutoffs)."""
    return (p - m.b_A) / m.Delta_A


def mu_BO(m: Params, p):
    """Participation cutoff mu_BO = (b_B - p) / Delta_B, Eq. (participation_cutoffs)."""
    return (m.b_B - p) / m.Delta_B


def mu_GO(m: Params, p):
    """Participation cutoff mu_GO = (p + kappa - b_B) / (a_A - b_B), Eq. (participation_cutoffs)."""
    return (p + m.kappa - m.b_B) / (m.a_A - m.b_B)


def kappa_bar(m: Params) -> float:
    """Viability threshold kappa_bar := delta_a delta_b / (delta_a + delta_b), Eq. (kappa_bar)."""
    return m.delta_a * m.delta_b / (m.delta_a + m.delta_b)


def p_star(m: Params) -> float:
    """Price threshold p* := (delta_a b_A + delta_b a_A) / (delta_a + delta_b), Eq. (pstar_rewrite)."""
    return (m.delta_a * m.b_A + m.delta_b * m.a_A) / (m.delta_a + m.delta_b)


def p_L(m: Params) -> float:
    """Price threshold p_L := v_B(mu_GB) = b_B - Delta_B kappa / delta_a, Eq. (pR_rewrite)."""
    return m.b_B - m.Delta_B / m.delta_a * m.kappa


def p_R(m: Params) -> float:
    """Price threshold p_R := v_A(mu_AG) = a_A - Delta_A kappa / delta_b, Eq. (pR_rewrite)."""
    return m.a_A - m.Delta_A / m.delta_b * m.kappa


def impact_region(m: Params, p: float) -> str:
    """Price region of Theorem thm:impact_aggregator, Eq. (impact_price_regions): "i" (p >= p_R), "ii"
    (p_L <= p < p_R), "iii" (p* <= p < p_L) or "iv" (p < p*)."""
    if p >= p_R(m):
        return "i"
    if p >= p_L(m):
        return "ii"
    if p >= p_star(m):
        return "iii"
    return "iv"


# ---------------------------------------------------------------------------------------------------------------
# switching prices and friction thresholds (Appendix subsubsec:aux_notation_impact_aggregator)
# ---------------------------------------------------------------------------------------------------------------


def _gap_ratio(m: Params) -> float:
    """(delta_a^2 - delta_b^2) / (delta_a^2 delta_b^2), the common factor of Eq. (pBbar_impact)."""
    return (m.delta_a**2 - m.delta_b**2) / (m.delta_a**2 * m.delta_b**2)


def p_A_switch(m: Params) -> float:
    """Switching price underline-p_A := b_A + Delta_A/2 - (Delta_A/2) kappa^2 (delta_a^2 - delta_b^2) /
    (delta_a^2 delta_b^2), Eq. (pBbar_impact); Omega_A(p) <= p iff p >= underline-p_A (Prop. property_Omega_AB)."""
    return m.b_A + m.Delta_A / 2 - m.Delta_A / 2 * m.kappa**2 * _gap_ratio(m)


def p_B_switch(m: Params) -> float:
    """Switching price underline-p_B := b_B - Delta_B/2 + (Delta_B/2) kappa^2 (delta_a^2 - delta_b^2) /
    (delta_a^2 delta_b^2), Eq. (pBbar_impact); Omega_B(p) <= p iff p >= underline-p_B (Prop. property_Omega_AB)."""
    return m.b_B - m.Delta_B / 2 + m.Delta_B / 2 * m.kappa**2 * _gap_ratio(m)


def p_A_L_switch(m: Params) -> float:
    """Switching price underline-p_A^L := 2 (a_A - b_B)^2 / Delta_A - 2 (kappa - b_B) - a_A + Delta_A kappa / delta_b,
    Eq. (pALbar_impact); Phi_A^L(p) <= p iff p >= underline-p_A^L (Prop. property_phi_AL)."""
    return 2 * (m.a_A - m.b_B) ** 2 / m.Delta_A - 2 * (m.kappa - m.b_B) - m.a_A + m.Delta_A / m.delta_b * m.kappa


def kappa_A_dagger(m: Params) -> float:
    """Friction threshold kappa_A^dagger := (2 b_B - a_A - b_A) / Delta_A * delta_a delta_b / (delta_a - delta_b),
    Eq. (kappaA_ddagger_impact); defined for delta_a != delta_b."""
    if m.delta_a == m.delta_b:
        raise ValueError("kappa_A^dagger is defined only for delta_a != delta_b")
    return (2 * m.b_B - m.a_A - m.b_A) / m.Delta_A * m.delta_a * m.delta_b / (m.delta_a - m.delta_b)


def kappa_A_ddagger(m: Params) -> float:
    """Friction threshold kappa_A^ddagger := delta_b^2 / Delta_A, Eq. (kappaA_ddagger_impact)."""
    return m.delta_b**2 / m.Delta_A


def kappa_A_0(m: Params) -> float:
    """Friction threshold kappa_A^0 := delta_b (a_A - b_B) / Delta_A, Eq. (kappaA_ddagger_impact)."""
    return m.delta_b / m.Delta_A * (m.a_A - m.b_B)


# ---------------------------------------------------------------------------------------------------------------
# wholesale-revenue threshold functions (Eqs. (PsiA_impact)-(PhiAL_impact))
# ---------------------------------------------------------------------------------------------------------------


def Psi_A(m: Params) -> float:
    """Psi_A = 2 delta_a / (delta_a + delta_b) * delta_a delta_b / (delta_a delta_b + kappa (delta_b - delta_a)),
    the simplified right-hand side of Eq. (PsiA_impact). Threshold of provider A for p < p*: w_A rho_0 >= Psi_A p."""
    da, db, k = m.delta_a, m.delta_b, m.kappa
    return 2 * da / (da + db) * da * db / (da * db + k * (db - da))


def Psi_B(m: Params) -> float:
    """Psi_B = 2 delta_b / (delta_a + delta_b) * delta_a delta_b / (delta_a delta_b + kappa (delta_a - delta_b)),
    the simplified right-hand side of Eq. (PsiB_impact). Threshold of provider B for p < p*: w_B rho_0 >= Psi_B p."""
    da, db, k = m.delta_a, m.delta_b, m.kappa
    return 2 * db / (da + db) * da * db / (da * db + k * (da - db))


def Omega_A(m: Params, p):
    """Omega_A(p) = 2 delta_a^2 delta_b^2 (p_R - p) p / (Delta_A ((delta_a delta_b - kappa delta_a)^2 - kappa^2
    delta_b^2)), Prop. property_Omega_AB (i). Threshold of provider A for p* <= p < p_L."""
    da, db, k = m.delta_a, m.delta_b, m.kappa
    return 2 * da**2 * db**2 * (p_R(m) - p) * p / (m.Delta_A * ((da * db - k * da) ** 2 - k**2 * db**2))


def Omega_B(m: Params, p):
    """Omega_B(p) = 2 delta_a^2 delta_b^2 (p_L - p) p / (Delta_B ((delta_a delta_b - kappa delta_b)^2 - kappa^2
    delta_a^2)), Prop. property_Omega_AB (i). Threshold of provider B for p* <= p < p_L."""
    da, db, k = m.delta_a, m.delta_b, m.kappa
    return 2 * da**2 * db**2 * (p_L(m) - p) * p / (m.Delta_B * ((da * db - k * db) ** 2 - k**2 * da**2))


def Phi_A_L(m: Params, p):
    """Phi_A^L(p) = 2 (a_A - b_B)^2 p / (Delta_A (p + p_R + 2 (kappa - b_B))), Prop. property_phi_AL (i).
    Threshold of provider A for p_L <= p < p_R; also the continuous extension of Eq. (PhiAL_impact) at p = p_R."""
    return 2 * (m.a_A - m.b_B) ** 2 * p / (m.Delta_A * (p + p_R(m) + 2 * (m.kappa - m.b_B)))


def Psi_A_from_cutoffs(m: Params) -> float:
    """Psi_A := 2 (mu_AG - mu_AB) / (mu_AG^2 - mu_GB^2), the definition in Eq. (PsiA_impact)."""
    return 2 * (mu_AG(m) - mu_AB(m)) / (mu_AG(m) ** 2 - mu_GB(m) ** 2)


def Psi_B_from_cutoffs(m: Params) -> float:
    """Psi_B := 2 (mu_AB - mu_GB) / ((mu_AG - mu_GB)(2 - mu_AG - mu_GB)), the definition in Eq. (PsiB_impact)."""
    return 2 * (mu_AB(m) - mu_GB(m)) / ((mu_AG(m) - mu_GB(m)) * (2 - mu_AG(m) - mu_GB(m)))


def Omega_A_from_cutoffs(m: Params, p):
    """Omega_A(p) := 2 (mu_AG - mu_AO) / (mu_AG^2 - mu_GB^2) p, the definition in Eq. (PhiAL_impact)."""
    return 2 * (mu_AG(m) - mu_AO(m, p)) / (mu_AG(m) ** 2 - mu_GB(m) ** 2) * p


def Omega_B_from_cutoffs(m: Params, p):
    """Omega_B(p) := 2 (mu_BO - mu_GB) / ((mu_AG - mu_GB)(2 - mu_AG - mu_GB)) p, the definition in
    Eq. (PhiAL_impact)."""
    return 2 * (mu_BO(m, p) - mu_GB(m)) / ((mu_AG(m) - mu_GB(m)) * (2 - mu_AG(m) - mu_GB(m))) * p


def Phi_A_L_from_cutoffs(m: Params, p):
    """Phi_A^L(p) := 2 (mu_AG - mu_AO) / (mu_AG^2 - mu_GO^2) p, the definition in Eq. (PhiAL_impact) (0/0 at
    p = p_R)."""
    return 2 * (mu_AG(m) - mu_AO(m, p)) / (mu_AG(m) ** 2 - mu_GO(m, p) ** 2) * p


def wholesale_threshold(m: Params, p, provider: str):
    """Threshold of Theorem thm:impact_aggregator: Delta R_X >= 0 iff w_X rho_0 >= wholesale_threshold(p, X).

    Phi_A^L(p) in region (ii) (provider A only; B is always better off there), Omega_X(p) in region (iii), Psi_X p
    in region (iv). ``p`` is a scalar; raises ValueError where the theorem states no threshold.
    """
    region = impact_region(m, p)
    if region == "ii" and provider == "A":
        return Phi_A_L(m, p)
    if region == "iii":
        return Omega_A(m, p) if provider == "A" else Omega_B(m, p)
    if region == "iv":
        return (Psi_A(m) if provider == "A" else Psi_B(m)) * p
    raise ValueError(f"no wholesale threshold for provider {provider} in region ({region})")


# ---------------------------------------------------------------------------------------------------------------
# the same thresholds from the exact market model
# ---------------------------------------------------------------------------------------------------------------


def _market(m: Params, p: float, w: float, rho_0: float, aggregator: bool) -> tuple:
    """``llmagg.market.revenues`` with w_A = w_B = w (R_X depends only on its own wholesale price)."""
    return revenues(m.a_A, m.b_A, m.a_B, m.b_B, p, m.kappa, w, w, rho_0, aggregator)


def revenue_change(m: Params, p: float, w: float, rho_0: float, provider: str) -> float:
    """Delta R_X := R_X^G - R_X^0 at wholesale price w_X = w, from the exact market model."""
    i = "AB".index(provider)
    return _market(m, p, w, rho_0, True)[i] - _market(m, p, w, rho_0, False)[i]


def breakeven_wholesale_price(m: Params, p: float, rho_0: float, provider: str) -> float:
    """Root w of Delta R_X(w) = 0, from the exact market model; Delta R_X is affine in w_X, so the root is exact."""
    d0 = revenue_change(m, p, 0.0, rho_0, provider)
    d1 = revenue_change(m, p, 1.0, rho_0, provider)
    return -d0 / (d1 - d0)


def shares(m: Params, p: float, aggregator: bool) -> dict[str, float]:
    """Segment measures |A|, |B|, |G|, |O| from the exact market model (|G| = 0 without the aggregator)."""
    s_A, s_B, s_G = _market(m, p, 0.0, 1.0, aggregator)[3:]
    return {"A": s_A, "B": s_B, "G": s_G, "O": 1 - s_A - s_B - s_G}


def grid_value(m: Params, name: str) -> float:
    """Value of a named model quantity used as a tick or grid end in the figures (e.g., "p_L", "b_B", "p_A_switch")."""
    if name in ("a_A", "b_A", "a_B", "b_B", "kappa"):
        return getattr(m, name)
    return {
        "p_star": p_star,
        "p_L": p_L,
        "p_R": p_R,
        "p_A_switch": p_A_switch,
        "p_B_switch": p_B_switch,
        "p_A_L_switch": p_A_L_switch,
    }[name](m)
