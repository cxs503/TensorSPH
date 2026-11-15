"""Analytic modal reference for the FSI2 beam (torch-free).

Anchor: ``XVAL-SPH-FSI2-BENCHMARK-AND-FORCE-FROM-FLUID``.

Why this module exists
----------------------
The FSI2 benchmark verdict
(``benchmarks/xval_fsi2/verdict_body_force_transient.py``) screens an
identified oscillation period against a *fluid* band, ``D/(St U_mean)`` with
``St`` in :data:`tensorsph.fsi2_beam_modes.FSI2_STROUHAL_BAND`.  That band is a
plausibility window borrowed from the vortex street; nothing in the repository
ever said what period the **structure** would prefer, so the screening had only
one of its two independent yardsticks.  A partitioned FSI answer can only be
believed when the shedding period and the coupled structural period agree (that
agreement *is* the lock-in the benchmark is built on), and when they disagree
the case cannot be reproducing the benchmark at all.

This module supplies the structural yardstick from Euler-Bernoulli beam theory,
in closed form, with no dependency on torch or numpy, so that the verdict
script (which is deliberately numpy-only) can carry it:

* :func:`cantilever_frequency_hz` -- ``fn = betan^2/(2 pi) sqrt(EI / (rho A L^4))``
  for a clamped-free beam, ``betan`` = 1.87510, 4.69409, 7.85476, 10.99554.
* :func:`added_mass_per_length` -- 2-D flat-plate added mass ``rho_f pi b^2``
  with ``b`` the half-thickness; it enters the coupled frequency as
  ``f_c = fn / sqrt(1 + m_a / (rho_s t))``.
* :func:`lockin_period_band_s` -- the period window ``T_c (1 -/+)``
  :data:`LOCKIN_HALF_WIDTH` in which a *coupled* response is dynamically
  admissible.
* :func:`structural_period_from_case` -- the same numbers derived from the
  ``case`` dict a run artefact carries, so a verdict can be recomputed from the
  artefact alone.

Nothing here is fitted: every constant is either a textbook eigenvalue or the
benchmark's own geometry.

Caveat (honest): the added-mass term is the unbounded-fluid flat-plate value.
In the FSI2 channel the blockage of the beam (``t / H = 0.02 / 0.41``) raises it
by ``1/(1 - (t/H)^2) ~ 1.002``, i.e. far below the discretisation error this
module is used to judge; the correction is reported, not silently applied.
"""
from __future__ import annotations

import math
from typing import Optional

__all__ = [
    "EULER_BERNOULLI_BETAS",
    "FSI2_STROUHAL_BAND",
    "LOCKIN_HALF_WIDTH",
    "cantilever_frequency_hz",
    "added_mass_per_length",
    "added_mass_channel_correction",
    "coupled_frequency_hz",
    "lockin_period_band_s",
    "structural_period_from_case",
]

#: First four eigenvalues of the clamped-free Euler-Bernoulli beam, ``cos b
#: cosh b = -1`` (Blevins, *Formulas for Natural Frequency and Mode Shape*,
#: table 8-1; also the standard values in Turek-Hron-adjacent FSI literature).
EULER_BERNOULLI_BETAS = (
    1.875104068711961,
    4.694091132974174,
    7.854757438237613,
    10.995540734875467,
)

#: Strouhal window of a circular cylinder's vortex street at ``Re ~ 100``.
#: The FSI2 verdict uses exactly this band; it is repeated here so that the
#: *fluid* and the *structural* predictors can be compared in one place.
FSI2_STROUHAL_BAND = (0.13, 0.30)

#: Half-width of the lock-in window around the coupled structural period.
#: A response within +-25 % of the structure's own frequency is inside the
#: lock-in range of a lightly damped beam; wider than that and the structure is
#: being driven, not resonating.  Fixed here, before any run is compared.
LOCKIN_HALF_WIDTH = 0.25


def _case_value(case, key: str, default: float) -> float:
    """Read a numeric field from a case dict or dataclass, with a default."""
    value = None
    if isinstance(case, dict):
        value = case.get(key)
    else:
        value = getattr(case, key, None)
    if value is None:
        return float(default)
    return float(value)


def cantilever_frequency_hz(beam_E: float, beam_rho: float, beam_thickness: float,
                            beam_length: float, mode: int = 1) -> float:
    """Clamped-free Euler-Bernoulli bending frequency [Hz] (per unit width).

    ``EI = E t^3 / 12`` (plane stress, unit depth) and ``rho A = rho_s t``.
    """
    if mode < 1 or mode > len(EULER_BERNOULLI_BETAS):
        raise ValueError(f"mode must be 1..{len(EULER_BERNOULLI_BETAS)}")
    if beam_E <= 0 or beam_rho <= 0 or beam_thickness <= 0 or beam_length <= 0:
        raise ValueError("beam parameters must be positive")
    beta = EULER_BERNOULLI_BETAS[mode - 1]
    ei = beam_E * beam_thickness ** 3 / 12.0
    rho_a = beam_rho * beam_thickness
    return beta ** 2 / (2.0 * math.pi) * math.sqrt(ei / (rho_a * beam_length ** 4))


def added_mass_per_length(rho_f: float, beam_thickness: float) -> float:
    """2-D added mass per unit length of a flat plate [kg/m].

    ``m_a = rho_f pi b^2`` with ``b`` the half-thickness (a flat plate of width
    ``2b`` accelerating normal to itself in an unbounded inviscid fluid; the
    added mass of a plate is identical to that of the ellipse of the same width
    in the limit of vanishing thickness, Lamb 1932, art. 73).
    """
    b = 0.5 * float(beam_thickness)
    return float(rho_f) * math.pi * b * b


def added_mass_channel_correction(beam_thickness: float,
                                  channel_height: float) -> float:
    """Factor by which a wall-bounded channel raises a plate's added mass.

    ``1 / (1 - (t/H)^2)`` -- the image-system first term for a plate centred
    between two walls.  Reported for completeness; it is ~1.002 for FSI2.
    """
    if channel_height <= 0:
        return 1.0
    r = float(beam_thickness) / float(channel_height)
    return 1.0 / max(1.0 - r * r, 1e-12)


def coupled_frequency_hz(beam_E: float, beam_rho: float, beam_thickness: float,
                         beam_length: float, rho_f: float, mode: int = 1,
                         channel_height: Optional[float] = None) -> float:
    """Added-mass-reduced (wetted) bending frequency [Hz]."""
    f_dry = cantilever_frequency_hz(beam_E, beam_rho, beam_thickness,
                                    beam_length, mode=mode)
    m_a = added_mass_per_length(rho_f, beam_thickness)
    if channel_height:
        m_a *= added_mass_channel_correction(beam_thickness, channel_height)
    m_struct = float(beam_rho) * float(beam_thickness)
    return f_dry / math.sqrt(1.0 + m_a / m_struct)


def lockin_period_band_s(beam_E: float, beam_rho: float, beam_thickness: float,
                         beam_length: float, rho_f: float, mode: int = 1,
                         channel_height: Optional[float] = None,
                         half_width: float = LOCKIN_HALF_WIDTH) -> tuple:
    """Period window [s] in which a coupled response is dynamically admissible."""
    f_c = coupled_frequency_hz(beam_E, beam_rho, beam_thickness, beam_length,
                               rho_f, mode=mode, channel_height=channel_height)
    t_c = 1.0 / f_c
    return (t_c * (1.0 - half_width), t_c * (1.0 + half_width))


def structural_period_from_case(case, mode: int = 1,
                                channel_height: Optional[float] = None) -> dict:
    """Every structural period a run artefact supports, from its ``case`` dict.

    Defaults are the shipped :class:`tensorsph.fsi2_turek_hron.FSI2Case`
    values (``E = 4 MPa``, ``t = 0.02 m``, ``L = 0.35 m``, ``rho_s = 1000``),
    so a run recorded before a field existed is still graded with the physics
    it was run with rather than with today's defaults silently substituted.
    """
    e = _case_value(case, "beam_E", 4.0e6)
    rho_s = _case_value(case, "beam_rho", 1000.0)
    t = _case_value(case, "beam_thickness", 0.02)
    length = _case_value(case, "beam_length", 0.35)
    rho_f = _case_value(case, "rho0", 1000.0)
    f_dry = cantilever_frequency_hz(e, rho_s, t, length, mode=mode)
    f_coupled = coupled_frequency_hz(e, rho_s, t, length, rho_f, mode=mode,
                                     channel_height=channel_height)
    band = lockin_period_band_s(e, rho_s, t, length, rho_f, mode=mode,
                                channel_height=channel_height)
    return {
        "mode": int(mode),
        "beam_E": e,
        "beam_rho": rho_s,
        "beam_thickness": t,
        "beam_length": length,
        "rho_f": rho_f,
        "frequency_dry_hz": f_dry,
        "frequency_coupled_hz": f_coupled,
        "period_dry_s": 1.0 / f_dry,
        "period_coupled_s": 1.0 / f_coupled,
        "added_mass_per_length": added_mass_per_length(rho_f, t),
        "added_mass_over_structural_mass":
            added_mass_per_length(rho_f, t) / (rho_s * t),
        "lockin_band_s": list(band),
        "lockin_half_width": LOCKIN_HALF_WIDTH,
    }


def period_strouhal_band_s(u_mean: float, diameter: float) -> Optional[tuple]:
    """Fluid-side predictor ``D/(St U_mean)`` [s]; ``None`` if inputs unusable."""
    if not u_mean or u_mean <= 0 or not diameter or diameter <= 0:
        return None
    lo, hi = FSI2_STROUHAL_BAND
    return (float(diameter) / (hi * u_mean), float(diameter) / (lo * u_mean))