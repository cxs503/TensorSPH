"""Thermal stress in a sea-ice sheet and the cracking criterion.

Two textbook thermo-elastic states, both in plane stress and equibiaxial (an
ice sheet is wide in both horizontal directions, so the biaxial modulus
``E/(1-nu)`` is the right coefficient):

``restrained``
    The sheet cannot contract in plane (shore-fast ice, or the interior of a
    large sheet whose edges are held)::

        sigma(z) = -E alpha (T(z) - T_ref) / (1 - nu)

    With a linear through-thickness profile ``T_top = T_ref + dT/2``,
    ``T_bot = T_ref - dT/2`` the surface stress is the classical
    ``|sigma| = E alpha dT / (2 (1 - nu))``, and ``T_ref`` is the restraint
    temperature (the value at which the sheet is stress free).

``free``
    Free edges: in-plane strain and curvature relax until the net force and the
    net moment vanish, ::

        sigma(z) = E/(1-nu) * (alpha*T_bar + kappa*z - alpha*T(z))
        alpha*T_bar = (1/h) int alpha T dz ,  kappa = (12/h^3) int alpha T z dz

    A **linear** profile then gives ``sigma == 0`` identically (a free plate
    with a linear gradient is stress free -- it just bends); only the
    *non-linear* part of the profile is self-equilibrated.  Both statements are
    machine-checked in ``tests/test_ice_thermal_stress.py``.

Cracking
--------
Two criteria, both reported:

* strength (Rankine): ``sigma_max >= sigma_t`` (0.7 MPa, range 0.3-1.5 MPa);
* fracture: an edge crack of depth ``a`` has ``K_I = 1.12 sigma sqrt(pi a)``
  (1.12 = edge-crack geometry factor), so the critical depth at a given stress
  is ``a_c = (K_IC / (1.12 sigma))^2 / pi`` and the critical temperature drop
  for a flaw ``a`` is ``dT_c = (1 - nu) K_IC / (1.12 E alpha sqrt(pi a))``.

For ``E = 9 GPa``, ``nu = 0.33``, ``alpha = 5e-5 1/K``, ``K_IC = 0.1 MPa m^0.5``
and ``a = 1 cm`` this gives ``dT_c ~ 0.8 K``: sea ice cracks on a one-kelvin
cooling event once a centimetre-scale flaw exists, which is why a 1 K surface
temperature drop is an engineering-relevant threshold and not a curiosity.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import torch

from . import properties as props

__all__ = [
    "biaxial_modulus",
    "restrained_stress",
    "free_plate_stress",
    "max_tensile_stress",
    "critical_temperature_drop",
    "stress_intensity",
    "critical_crack_depth",
    "crack_assessment",
    "surface_temperature_drop",
]

DTYPE = torch.float64


def biaxial_modulus(e=props.E_ICE, nu=props.NU_ICE) -> float:
    """Plane-stress biaxial modulus ``E/(1-nu)`` (Pa)."""
    return e / (1.0 - nu)


def restrained_stress(temperature, t_ref=None, temperature_ref_axis=None):
    """Fully restrained in-plane thermal stress (Pa): ``-E alpha (T - T_ref)/(1-nu)``.

    ``t_ref`` may be a scalar or an array; when omitted the mid-thickness
    temperature of ``temperature`` is used (the usual restraint assumption).
    """
    t = torch.as_tensor(temperature, dtype=DTYPE)
    ref = t.mean() if t_ref is None else torch.as_tensor(t_ref, dtype=DTYPE)
    return -biaxial_modulus() * props.ALPHA_TH * (t - ref)


def free_plate_stress(temperature, z):
    """Self-equilibrated stress (Pa) of a plate with free edges and free bending.

    Returns the stress field only; :func:`free_plate_moments` gives the
    equilibrium residuals that the tests assert to be zero.
    """
    t, zz = _as_arrays(temperature, z)
    zc = zz - 0.5 * (zz[0] + zz[-1])                         # mid-plane coordinate
    t_bar = float(t.mean())
    # Moment balance about the mid-plane: kappa = alpha * int(T z) / int(z^2), with
    # the *same* discrete measure in numerator and denominator.  That is what makes
    # a linear profile exactly stress free on a discrete grid; using 12/h^3 with a
    # continuum h leaves an O(dz^2/h^2) residual (66 Pa in the old version).
    kappa = props.ALPHA_TH * float((t * zc).sum()) / float((zc * zc).sum())
    return biaxial_modulus() * (props.ALPHA_TH * t_bar + kappa * zc - props.ALPHA_TH * t)


def free_plate_moments(stress, z):
    """Net force and net moment of a stress distribution (both must vanish)."""
    s, zz = _as_arrays(stress, z)
    return float(s.mean()), float((s * (zz - zz.mean())).mean())


def _as_arrays(a, b):
    x = torch.as_tensor(a, dtype=DTYPE).reshape(-1)
    y = torch.as_tensor(b, dtype=DTYPE).reshape(-1)
    if x.shape != y.shape:
        raise ValueError("temperature and z must have the same shape")
    return x, y


def max_tensile_stress(stress):
    """Largest tensile (positive) stress of a distribution (Pa)."""
    s = torch.as_tensor(stress, dtype=DTYPE)
    return float(s.max()) if s.numel() else 0.0


def surface_temperature_drop(t_top, t_ref) -> float:
    """``dT`` driving the classical restrained surface stress (K)."""
    return abs(float(t_top) - float(t_ref))


def critical_temperature_drop(sigma_t=props.SIGMA_TENSILE, e=props.E_ICE,
                              nu=props.NU_ICE, alpha=props.ALPHA_TH) -> float:
    """Temperature drop at which the restrained surface stress reaches ``sigma_t``."""
    return sigma_t * (1.0 - nu) / (e * alpha)


def stress_intensity(stress, crack_depth, geometry=1.12):
    """Mode-I stress intensity ``K_I = geometry * sigma * sqrt(pi a)`` (Pa m^0.5)."""
    return geometry * float(stress) * math.sqrt(math.pi * float(crack_depth))


def critical_crack_depth(stress, k_ic=props.K_IC, geometry=1.12) -> float:
    """Depth of edge crack that becomes critical at ``stress`` (m)."""
    s = max(float(stress), 1e-9)
    return (k_ic / (geometry * s)) ** 2 / math.pi


@dataclass
class CrackAssessment:
    """Outcome of the two cracking criteria for one stress state."""

    max_tensile_pa: float
    strength_pa: float
    strength_ratio: float
    strength_exceeded: bool
    k_i_pa_sqrt_m: float
    k_ic_pa_sqrt_m: float
    fracture_ratio: float
    fracture_exceeded: bool
    critical_depth_m: float
    flaw_depth_m: float

    @property
    def cracks(self) -> bool:
        return self.strength_exceeded or self.fracture_exceeded


def crack_assessment(stress, flaw_depth=0.01, sigma_t=props.SIGMA_TENSILE,
                     k_ic=props.K_IC, geometry=1.12) -> CrackAssessment:
    """Evaluate the Rankine and the fracture criterion for a stress field."""
    s_max = max_tensile_stress(stress)
    k_i = stress_intensity(s_max, flaw_depth, geometry) if s_max > 0 else 0.0
    return CrackAssessment(
        max_tensile_pa=s_max,
        strength_pa=sigma_t,
        strength_ratio=s_max / sigma_t,
        strength_exceeded=s_max >= sigma_t,
        k_i_pa_sqrt_m=k_i,
        k_ic_pa_sqrt_m=k_ic,
        fracture_ratio=k_i / k_ic,
        fracture_exceeded=k_i >= k_ic,
        critical_depth_m=critical_crack_depth(s_max, k_ic, geometry) if s_max > 0 else math.inf,
        flaw_depth_m=float(flaw_depth),
    )
