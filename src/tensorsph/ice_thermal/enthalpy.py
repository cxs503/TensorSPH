"""Enthalpy method for phase change: fresh ice (Stefan) and sea ice (mushy).

Two models, both **analytic in enthalpy** so the only numerical step is the
inversion ``T = T(e)`` (vectorised bisection, exact to the float64 bracket):

``fresh``
    Pure substance with a numerical mushy band ``mushy_dT`` (K) that spreads the
    latent heat ``L`` over that band.  Reference ``e(T_melt) = 0``::

        e(T) = c_i (T - Tm) - L * clamp((Tm - T)/mushy_dT, 0, 1)   (T < Tm)
        e(T) = c_w (T - Tm)                                        (T >= Tm)

    The band exists only so that an enthalpy-carrying SPH particle can hold a
    *partially* frozen state; ``mushy_dT -> 0`` recovers the classical Stefan
    formulation.  Its numerical cost is quantified by the resolution ladder in
    ``benchmarks/ice_thermal_multiphysics``.

``sea``
    Bulk sea ice of practical salinity ``S`` (psu).  The liquid fraction of the
    ice is ``w_l = S / S_b(T)`` with ``S_b`` the brine salinity of
    :mod:`tensorsph.ice_thermal.properties`; with the linear liquidus
    ``S_b = -T/mu`` this is ``w_l = mu S / (-T)`` (all in psu / K).  Integrating
    ``de = (c_i + (c_w - c_i) w_l) dT + L dw_l`` from ``Tf(S)`` gives, in closed
    form (``mu = LIQUIDUS_SLOPE``, ``s = S/1000`` in kg salt per kg ice)::

        e(T) = c_i (T - Tf) + (c_w - c_i) Tf' ln(Tf'/theta)
               + L Tf' (1/theta - 1/Tf')     theta = 273.15 - T, Tf' = -Tf
        e(T) = c_w (T - Tf)                               (T >= Tf)

    which is continuous at ``Tf`` and releases exactly ``L`` of latent heat as
    ``T`` falls from ``Tf`` to absolute zero -- i.e. the Maykut & Untersteiner
    (1971) sea-ice enthalpy written as an enthalpy rather than as a heat
    capacity.  Differentiating gives the standard apparent heat capacity
    ``c_eff = c_i + (c_w - c_i) w_l + L mu s / T^2``, whose latent term is what
    makes a sea-ice column hold ``O(10)`` times more heat per kelvin than fresh
    ice near its freezing point.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import torch

from . import properties as props

__all__ = [
    "EnthalpyModel",
    "FRESH",
    "SEA_ICE",
    "neumann_lambda",
    "stefan_front",
    "stefan_number_fresh",
]


def neumann_lambda(stefan: float) -> float:
    """Similarity root of the one-phase Stefan problem.

    Solve ``sqrt(pi) * lam * exp(lam^2) * erf(lam) = Ste`` by bisection
    (monotone in ``lam``, bracket ``[0, 6]``; residual < 1e-14 at float64).
    """
    ste = max(float(stefan), 1e-12)
    lo, hi = 0.0, 6.0
    for _ in range(200):
        mid = 0.5 * (lo + hi)
        f = math.sqrt(math.pi) * mid * math.exp(mid * mid) * math.erf(mid) - ste
        if f < 0.0:
            lo = mid
        else:
            hi = mid
    return 0.5 * (lo + hi)


def stefan_front(time: float, lam: float, alpha: float) -> float:
    """Front position ``X(t) = 2 lam sqrt(alpha t)`` of the Neumann solution."""
    if time <= 0.0:
        return 0.0
    return 2.0 * lam * math.sqrt(alpha * float(time))


def stefan_number_fresh(t_melt: float, t_cold: float, latent: float = props.L_FUSION,
                        c: float = props.C_ICE) -> float:
    """``Ste = c (T_melt - T_cold)/L`` (no sensible heat in the liquid phase)."""
    return c * (float(t_melt) - float(t_cold)) / latent


@dataclass(frozen=True)
class EnthalpyModel:
    """Analytic enthalpy curve with a vectorised inverse.

    ``kind`` is ``"fresh"`` (pure substance + mushy band) or ``"sea"`` (mushy
    sea ice with salinity-dependent freezing point).
    """

    kind: str = "fresh"
    t_melt: float = props.T_MELT
    mushy_dt: float = 0.05
    latent: float = props.L_FUSION
    c_solid: float = props.C_ICE
    c_liquid: float = props.C_FRESH

    # ------------------------------------------------------------- derived
    def __post_init__(self) -> None:
        if self.kind not in ("fresh", "sea"):
            raise ValueError(f"unknown enthalpy kind {self.kind!r}")

    def freezing_point(self, salinity):
        if self.kind == "fresh":
            return torch.full_like(torch.as_tensor(salinity, dtype=torch.float64),
                                   float(self.t_melt))
        s = torch.as_tensor(salinity, dtype=torch.float64)
        drop = (-0.0575 * s + 1.710523e-3 * s.clamp_min(0.0) ** 1.5
                - 2.154996e-4 * s.clamp_min(0.0) ** 2)
        return self.t_melt + drop

    def enthalpy(self, temperature, salinity=0.0):
        """``e(T, S)`` in J/kg (reference: liquid at the freezing point = 0)."""
        t = torch.as_tensor(temperature, dtype=torch.float64)
        s = torch.as_tensor(salinity, dtype=torch.float64)
        s, t = torch.broadcast_tensors(s, t)
        tf = self.freezing_point(s)
        if self.kind == "fresh":
            frac = ((self.t_melt - t) / self.mushy_dt).clamp(0.0, 1.0)
            solid = self.c_solid * (t - self.t_melt) - self.latent * frac
            liquid = self.c_liquid * (t - self.t_melt)
            return torch.where(t >= self.t_melt, liquid, solid)
        # theta = 273.15 - T (>0) is the freezing-point depression variable:
        # with the linear liquidus S_b = theta/mu the liquid fraction is
        # w_l = mu S / theta, and the latent term integrates to
        # L mu S (1/theta_f - 1/theta).  Using T instead of theta here is a
        # 273.15x error -- the trap that this module documents.
        theta = (props.T_MELT - t).clamp_min(1e-6)
        theta_f = (props.T_MELT - tf).clamp_min(1e-6)
        solid = (self.c_solid * (t - tf)
                 + (self.c_liquid - self.c_solid) * theta_f * torch.log(theta_f / theta)
                 + self.latent * theta_f * (1.0 / theta - 1.0 / theta_f))
        liquid = self.c_liquid * (t - tf)
        return torch.where(t >= tf, liquid, solid)

    def heat_capacity(self, temperature, salinity=0.0):
        """Apparent heat capacity ``de/dT`` (J/(kg K)); diverges at ``Tf``."""
        t = torch.as_tensor(temperature, dtype=torch.float64)
        s = torch.as_tensor(salinity, dtype=torch.float64)
        s, t = torch.broadcast_tensors(s, t)
        if self.kind == "fresh":
            in_band = (t < self.t_melt) & (t > self.t_melt - self.mushy_dt)
            band_cp = self.c_solid + self.latent / self.mushy_dt
            cp = torch.where(in_band, torch.full_like(t, band_cp),
                             torch.where(t >= self.t_melt,
                                         torch.full_like(t, self.c_liquid),
                                         torch.full_like(t, self.c_solid)))
            return cp
        tf = self.freezing_point(s)
        w_l = _liquid_fraction_t(t, s)
        theta = (props.T_MELT - t).clamp_min(1e-6)
        theta_f = (props.T_MELT - tf).clamp_min(1e-6)
        latent = self.latent * theta_f / theta ** 2
        solid = self.c_solid + (self.c_liquid - self.c_solid) * w_l + latent
        return torch.where(t >= tf, torch.full_like(t, self.c_liquid), solid)

    def temperature(self, enthalpy, salinity=0.0, iterations: int = 20):
        """``T = e^{-1}(H, S)`` by vectorised bisection (monotone ``e(T)``)."""
        h = torch.as_tensor(enthalpy, dtype=torch.float64)
        s = torch.as_tensor(salinity, dtype=torch.float64)
        s, h = torch.broadcast_tensors(s, h)
        if self.kind == "fresh":
            # exact piecewise inverse of the fresh-ice curve (no iteration)
            x_band = -h / (self.c_solid + self.latent / self.mushy_dt)
            cold = self.t_melt + (h + self.latent) / self.c_solid
            return torch.where(h >= 0.0, self.t_melt + h / self.c_liquid,
                               torch.where(x_band <= self.mushy_dt,
                                           self.t_melt - x_band, cold))
        tf = self.freezing_point(s)
        ci, cl, lat = self.c_solid, self.c_liquid, self.latent
        theta_f = (props.T_MELT - tf).clamp_min(1e-6)
        coef = lat * theta_f
        sens = (cl - ci) * theta_f
        lo = (tf - 400.0).clone()          # deep-cold bracket, e << H
        hi = torch.full_like(lo, self.t_melt + 200.0)  # warm bracket, e >> H
        for _ in range(iterations):
            mid = 0.5 * (lo + hi)
            theta = (props.T_MELT - mid).clamp_min(1e-6)
            solid = (ci * (mid - tf) + sens * torch.log((theta_f / theta).clamp_min(1e-12))
                     + coef * (1.0 / theta - 1.0 / theta_f))
            e_mid = torch.where(mid >= tf, cl * (mid - tf), solid)
            below = e_mid < h
            lo = torch.where(below, mid, lo)
            hi = torch.where(below, hi, mid)
        return 0.5 * (lo + hi)

    def liquid_fraction(self, temperature, salinity=0.0):
        """Mass fraction of liquid; 1 for T >= Tf, else ``w_l = mu s / (-T)``."""
        t = torch.as_tensor(temperature, dtype=torch.float64)
        s = torch.as_tensor(salinity, dtype=torch.float64)
        s, t = torch.broadcast_tensors(s, t)
        return _liquid_fraction_t(t, s)


def _liquid_fraction_t(t: torch.Tensor, s: torch.Tensor) -> torch.Tensor:
    """``w_l = mu*s/(-T)`` clipped to [0, 1] (piecewise, torch-only)."""
    tf = (props.T_MELT - 0.0575 * s.clamp_min(0.0) + 1.710523e-3 * s.clamp_min(0.0) ** 1.5
          - 2.154996e-4 * s.clamp_min(0.0) ** 2)
    theta = (props.T_MELT - t).clamp_min(1e-6)
    theta_f = (props.T_MELT - tf).clamp_min(1e-6)
    w = (theta_f / theta).clamp(0.0, 1.0)
    return torch.where(t >= tf, torch.ones_like(t), w)


FRESH = EnthalpyModel(kind="fresh")
SEA_ICE = EnthalpyModel(kind="sea", c_liquid=props.C_SEAWATER)
