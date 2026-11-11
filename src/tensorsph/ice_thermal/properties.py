"""Thermophysical properties of sea ice, brine and seawater (SI, T in kelvin).

Every relation here is a *documented* one, not a tuned fit of this module:

* freezing-point depression -- UNESCO (1983) polynomial for the seawater
  liquidus, ``Tf(S) = -0.0575 S + 1.710523e-3 S^1.5 - 2.154996e-4 S^2``
  (S in psu, i.e. g salt per kg seawater);
* brine salinity -- the *inverse* of that same liquidus (the brine is by
  definition the liquid in equilibrium with the ice), so no independent
  empirical fit is introduced and the liquidus is exactly self-consistent;
* brine volume fraction -- Cox & Weeks (1983),
  ``v_b = rho_i S / (rho_b S_b)`` (salt mass conservation between bulk ice and
  brine);
* brine-filled pore permeability -- Golden, Ackley & Lytle (1998),
  ``k = 3e-8 (v_b/0.05)^3.1`` m^2 for ``v_b >= 0.05`` and ``k = 0`` below the
  percolation threshold (brine is not connected, so gravity drainage stops).

The *only* free parameter in the whole package is the drainage efficiency
``theta_drain`` in :mod:`tensorsph.ice_thermal.brine`; it is declared and
calibrated against a published desalination *band*, never silently tuned.

Thermodynamics of sea ice: the apparent heat capacity following from these
properties is derived (not fitted) in
:mod:`tensorsph.ice_thermal.enthalpy` -- liquid fraction
``w_l = S / S_b(T)``, latent term ``L * dw_l/dT = L mu S / T^2``.
"""

from __future__ import annotations

import math

import torch

# ---------------------------------------------------------------- constants
RHO_ICE = 917.0            # kg/m^3, pure bubble-free ice
RHO_FRESH = 1000.0         # kg/m^3, fresh water at 0 C
C_ICE = 2100.0             # J/(kg K), specific heat of ice
C_FRESH = 4218.0           # J/(kg K), specific heat of fresh water
C_SEAWATER = 3985.0        # J/(kg K), seawater at 0 C / 32 psu
L_FUSION = 3.34e5          # J/kg, latent heat of fusion of pure ice
T_MELT = 273.15            # K, melting point of fresh ice at 1 atm
K_ICE = 2.2                # W/(m K), thermal conductivity of ice (bubbly -> ~2.0)
K_FRESH = 0.56             # W/(m K), fresh water at 0 C
K_SEAWATER = 0.57          # W/(m K), seawater at 0 C / 32 psu
DIFF_FRESH = K_FRESH / (RHO_FRESH * C_FRESH)     # m^2/s, ~1.33e-7
DIFF_ICE = K_ICE / (RHO_ICE * C_ICE)             # m^2/s, ~1.14e-6

# linear liquidus slope used *inside* the derived apparent heat capacity
# (0.054 K/psu == the leading term of the UNESCO polynomial)
LIQUIDUS_SLOPE = 0.054     # K/psu

# mechanical properties of columnar first-year sea ice
E_ICE = 9.0e9              # Pa, Young's modulus (9-11 GPa)
NU_ICE = 0.33              # Poisson ratio
ALPHA_TH = 5.0e-5          # 1/K, linear thermal expansion coefficient
SIGMA_TENSILE = 0.7e6      # Pa, tensile strength (range 0.3-1.5 MPa)
K_IC = 0.1e6               # Pa m^0.5, fracture toughness (~0.05-0.15 MPa m^0.5)

#: strength of sea ice as a function of brine volume (Timco & Frederking 1990):
#: ``sigma(v_b) = SIGMA_F_MAX (1 - sqrt(v_b / V_B_CLOSURE))``.  ``v_b = 0.27``
#: is the pore-closure volume fraction at which the skeleton carries no load;
#: ``SIGMA_F_MAX = 1.76 MPa`` is the measured columnar first-year value.
SIGMA_F_MAX = 1.76e6       # Pa, strength at v_b -> 0 (Timco & Frederking 1990)
V_B_CLOSURE = 0.27         # -, brine volume fraction at zero strength

# brine / seawater transport
MU_BRINE = 2.5e-3          # Pa s, dynamic viscosity of cold brine
#: permeability at the percolation threshold.  Golden's measured envelope is
#: ``3e-8 (v_b/0.05)^3.1``; published sea-ice values at ``v_b ~ 0.1`` span
#: 1e-11...1e-7 m^2, so the *absolute scale* is the single largest uncertainty
#: of any drainage model.  The default is one decade below the Golden envelope
#: (columnar first-year ice) and the benchmark runs a k_ref x {0.1, 1, 10}
#: sensitivity scan instead of hiding the choice.
K_REF = 3.0e-9
K_GOLDEN_ENVELOPE = 3.0e-8
GRAVITY = 9.81             # m/s^2
BETA_SALINITY = 0.78       # kg/m^3 per psu, haline contraction coefficient
RHO_SEAWATER_REF = 1028.0  # kg/m^3 at S = 32 psu, T = 0 C


def freezing_point(salinity):
    """Seawater liquidus temperature (K) for practical salinity ``salinity`` (psu).

    UNESCO (1983) polynomial, valid for 0 <= S <= 40 psu; for S = 0 it returns
    exactly :data:`T_MELT`.
    """
    s = max(float(salinity), 0.0)
    drop = -0.0575 * s + 1.710523e-3 * s ** 1.5 - 2.154996e-4 * s ** 2
    return T_MELT + drop


def freezing_point_vec(salinity):
    """Vectorised :func:`freezing_point` (torch, kelvin)."""
    s = torch.as_tensor(salinity, dtype=torch.float64).clamp_min(0.0)
    return T_MELT - 0.0575 * s + 1.710523e-3 * s ** 1.5 - 2.154996e-4 * s ** 2


def brine_salinity_vec(temperature, salinity_max=250.0, iterations=14):
    """Vectorised :func:`brine_salinity` (torch bisection on the liquidus)."""
    t = torch.as_tensor(temperature, dtype=torch.float64)
    lo = torch.zeros_like(t)
    hi = torch.full_like(t, float(salinity_max))
    above = t >= T_MELT
    for _ in range(iterations):
        mid = 0.5 * (lo + hi)
        lo = torch.where(freezing_point_vec(mid) > t, mid, lo)
        hi = torch.where(freezing_point_vec(mid) > t, hi, mid)
    out = 0.5 * (lo + hi)
    return torch.where(above, torch.zeros_like(out), out)


def permeability_vec(v_b, k_max=K_REF, threshold=0.05, exponent=3.1):
    """Vectorised :func:`permeability` (torch, m^2)."""
    v = torch.as_tensor(v_b, dtype=torch.float64).clamp_min(0.0)
    k = k_max * (v / threshold) ** exponent
    return torch.where(v < threshold, torch.zeros_like(k), k)


def brine_salinity(temperature, salinity_max=250.0):
    """Brine salinity (psu) in equilibrium with ice at ``temperature`` (K).

    Obtained by inverting :func:`freezing_point` (bisection, 200 halvings, so
    the result satisfies ``freezing_point(S_b) == T`` to machine precision).
    Below the eutectic (T < 251.2 K, S_b -> 250 psu) the inverse is clamped --
    the eutectic of the NaCl-H2O system is 252 K / 233 psu and below it the
    liquidus no longer applies, which is stated as a model limit.
    """
    t = float(temperature)
    if t >= T_MELT:
        return 0.0
    lo, hi = 0.0, salinity_max
    for _ in range(200):
        mid = 0.5 * (lo + hi)
        if freezing_point(mid) > t:
            lo = mid
        else:
            hi = mid
    return 0.5 * (lo + hi)


def brine_density(brine_salinity_psu):
    """Brine density (kg/m^3); linear in salinity about 1000 kg/m^3."""
    return 1000.0 + 0.8 * max(float(brine_salinity_psu), 0.0)


def brine_volume_fraction(temperature, salinity):
    """Brine volume fraction ``v_b`` (m^3 brine / m^3 bulk ice).

    Cox & Weeks (1983): salt mass conservation ``rho_i S = rho_b S_b v_b``.
    Returns 1.0 for ``T >= Tf(S)`` (the ice is fully molten) and 0 for fresh
    ice (no salt -> no brine).
    """
    s = max(float(salinity), 0.0)
    t = float(temperature)
    if s <= 0.0:
        return 0.0
    tf = freezing_point(s)
    if t >= tf:
        return 1.0
    sb = brine_salinity(t)
    if sb <= 0.0:
        return 0.0
    return min(1.0, RHO_ICE * s / (brine_density(sb) * sb))


def liquid_fraction(temperature, salinity):
    """Mass fraction of liquid (brine) in bulk sea ice, ``w_l = S / S_b``."""
    s = max(float(salinity), 0.0)
    t = float(temperature)
    if s <= 0.0:
        return 0.0
    tf = freezing_point(s)
    if t >= tf:
        return 1.0
    sb = brine_salinity(t)
    return min(1.0, s / sb)


def permeability(v_b, k_max=K_REF, threshold=0.05, exponent=3.1):
    """Brine permeability (m^2) vs brine volume fraction.

    Golden, Ackley & Lytle (1998): ``k = k_max (v_b / 0.05)^3.1`` with their
    measured ``k_max = 3e-8 m^2``; this package uses :data:`K_REF` by default
    (see the note there).  Below the
    percolation threshold ``v_b = 0.05`` the brine inclusions are isolated and
    the permeability is set to zero -- this threshold is what makes gravity
    drainage *switch on* at a temperature determined by the bulk salinity, and
    it is the mechanism behind the two-regime desalination curve.
    """
    v = max(float(v_b), 0.0)
    if v < threshold:
        return 0.0
    return k_max * (v / threshold) ** exponent


def density_anomaly(salinity, temperature_c=0.0):
    """Seawater density (kg/m^3) minus the 32 psu reference -- linear EOS.

    ``d_rho = rho0 * (beta_S (S - 32) - alpha_T (T - 0))`` with
    ``alpha_T = 5.2e-5 1/K`` (near-freezing seawater is close to its density
    maximum, so the thermal expansion is deliberately small).
    """
    d_s = float(salinity) - 32.0
    d_t = float(temperature_c)
    return RHO_SEAWATER_REF * (BETA_SALINITY / RHO_SEAWATER_REF * d_s - 5.2e-5 * d_t)


def flexural_strength(temperature, salinity, sigma_max=SIGMA_F_MAX,
                      v_closure=V_B_CLOSURE):
    """Flexural strength of sea ice (Pa) from its thermal-saline state.

    Timco & Frederking (1990) measured the strength of columnar sea ice against
    its brine volume fraction and found a single collapse curve

    ``sigma(v_b) = sigma_max * (1 - sqrt(v_b / v_closure))``,

    where ``v_b`` is :func:`brine_volume_fraction` (a function of ``T`` and the
    bulk salinity ``S``), ``v_closure = 0.27`` is the pore-closure volume
    fraction at which the ice skeleton loses all load capacity, and
    ``sigma_max = 1.76 MPa`` at ``v_b -> 0``.  The same brine-volume collapse
    holds for the flexural and the compressive strength; the absolute level
    differs (flexural ~1.5x lower for the same ``v_b``), which is the dominant
    uncertainty of the relation and is why the value is returned with
    :func:`strength_band` instead of being treated as exact.

    Inversion check: the platform default flexural strength of 0.7 MPa
    corresponds to ``v_b = 0.098``, i.e. first-year ice near -5 C with a bulk
    salinity of ~10 psu.

    Parameters
    ----------
    temperature : float
        Bulk ice temperature [K].
    salinity : float
        Bulk ice salinity [psu].
    """
    v_b = brine_volume_fraction(temperature, salinity)
    if v_b >= v_closure:
        return 0.0
    return float(sigma_max) * (1.0 - math.sqrt(v_b / float(v_closure)))


def strength_band(temperature, salinity, factor=(0.6, 1.0, 1.4)):
    """Flexural strength (Pa) with the observed flexural/compressive spread.

    Returns ``(sigma, low, high)`` where ``low``/``high`` bracket the value
    using the multiplicative band of the flexural-to-brine-volume relation
    (Timco & Frederking 1990 report a factor-of-~2 total scatter for a given
    ``v_b``; the band used here is deliberately narrower and symmetric).
    """
    sigma = flexural_strength(temperature, salinity)
    return sigma, sigma * factor[0], sigma * factor[2]


def thermal_diffusivity(phase="ice"):
    """Thermal diffusivity (m^2/s) for ``phase`` in {"ice", "fresh", "seawater"}."""
    if phase == "ice":
        return DIFF_ICE
    if phase == "fresh":
        return DIFF_FRESH
    if phase == "seawater":
        return K_SEAWATER / (RHO_SEAWATER_REF * C_SEAWATER)
    raise ValueError(f"unknown phase {phase!r}")


def stefan_number(t_hot, t_cold, latent=L_FUSION, c=C_ICE):
    """Stefan number ``c (T_melt - T_cold) / L`` for a freezing front."""
    return c * (float(t_hot) - float(t_cold)) / latent


def format_property_table() -> str:
    """Small human-readable table used in the report (single source of truth)."""
    rows = [
        ("rho_ice", RHO_ICE, "kg/m^3"),
        ("c_ice", C_ICE, "J/(kg K)"),
        ("c_fresh", C_FRESH, "J/(kg K)"),
        ("L_fusion", L_FUSION, "J/kg"),
        ("k_ice", K_ICE, "W/(m K)"),
        ("k_fresh", K_FRESH, "W/(m K)"),
        ("diff_ice", DIFF_ICE, "m^2/s"),
        ("diff_fresh", DIFF_FRESH, "m^2/s"),
        ("E_ice", E_ICE, "Pa"),
        ("nu_ice", NU_ICE, "-"),
        ("alpha_th", ALPHA_TH, "1/K"),
        ("mu_brine", MU_BRINE, "Pa s"),
    ]
    width = max(len(r[0]) for r in rows)
    out = [f"{name:<{width}}  {value:>12.6g}  {unit}" for name, value, unit in rows]
    out.append(f"Tf(S=32 psu) = {freezing_point(32.0) - T_MELT:.3f} C")
    out.append(f"sqrt(pi)*liquidus slope = {math.sqrt(math.pi) * LIQUIDUS_SLOPE:.4f}")
    return "\n".join(out)
