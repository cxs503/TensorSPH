"""Independent audit of the Wagner pressure reference used by CASE-SPH-WEDGE-ENTRY-2D.

Why this file exists
--------------------
The case grades measured wall pressures (``cp = p / q``, ``q = rho V^2 / 2``)
against two analytic numbers taken from ``tensorsph.wedge_entry``:

=====================================  ===========  =========================
quantity                               reference    used for
=====================================  ===========  =========================
``wagner_peak_over_q(beta)``           ``C^2 + 1``  ``cp_band_peak`` / ``cp_max``
``refs_derived()["wagner"]["cp_centre"]``  ``C``    ``cp_centre`` (stagnation)
=====================================  ===========  =========================

Both claims are testable *inside the repository*, without new runs and without
the closed-access target paper:

A1. **Profile shape.**  ``wedge_entry.wagner_p_over_q`` states the wall
    pressure profile ``p/q = 2 C s - s^2 + 1`` with ``s = 1/sqrt(1 - tau^2)``.
    Is that the Eulerian Bernoulli pressure of the Wagner flat-plate potential?
    This file answers it by *re-deriving the profile numerically*: the exact
    complex potential ``W(z) = -i V (z - sqrt(z^2 - c^2))`` is evaluated on a
    grid, ``dphi/dt``, ``u = dphi/dx`` and ``v = dphi/dy`` are obtained by
    finite differences of that potential only (no use of the closed form being
    audited), and the pressure follows from Eulerian Bernoulli
    ``p = -rho (dphi/dt + (u^2 + v^2)/2)``.

A2. **Normalisation of the stagnation reference.**  ``cp_centre`` is measured
    as ``p(0)/q`` but graded against ``C`` (= ``p(0)/rho V^2``), i.e. half the
    correct ``q``-unit value ``2C``.  The module's own profile gives ``2C`` at
    ``tau = 0`` and ``wagner_p_centre`` gives ``rho V^2 C = q * 2C``: the
    graded reference contradicts two functions in the same module.

Neither finding changes a solver result: they are reference-side defects, so the
committed measurements stand and only the reported errors move.  Nothing here is
graded against Zhao & Faltinsen (1993); that reference remains WITHHELD.

Usage
-----
    python benchmarks/case_sph_wedge_entry_2d/audit_wagner_reference.py            # table
    python benchmarks/case_sph_wedge_entry_2d/audit_wagner_reference.py --json     # machine readable
"""

from __future__ import annotations

import argparse
import json
import math
from dataclasses import dataclass

import numpy as np

from tensorsph.wedge_entry import (
    wagner_c_coeff,
    wagner_p_centre,
    wagner_p_over_q,
    wagner_peak_over_q,
    wagner_peak_tau,
)

# --------------------------------------------------------------------------- #
# 1. The Wagner flat-plate potential, handled only through its complex form
# --------------------------------------------------------------------------- #

#: Distance used for the finite differences in ``x`` and ``y``, in units of c.
#: ``1e-4`` keeps the central-difference truncation (``O(h^2)``) at the 1e-8
#: level without hitting the cancellation floor of the ``sqrt`` branch.
_FD_H = 1.0e-4


def _w(z, c: float, V: float):
    """``W(z) = -i V (z - sqrt(z^2 - c^2))`` in a cancellation-free form.

    ``z - sqrt(z^2-c^2) = c^2 / (z + sqrt(z^2-c^2))``; the plain difference
    loses all significant digits for ``|z| >> c`` (far-field check) and for
    ``|x| -> c``, which is exactly where the profile is steep.
    """
    z = np.asarray(z, dtype=complex)
    rt = np.sqrt(z * z - c * c)
    # Pin the physical branch: Im(rt) > 0 in the fluid half-plane y > 0.  Plain
    # np.sqrt follows signed zeros, and ``x < 0`` yields ``z*z`` with -0.0
    # imaginary part, which silently returns the conjugate branch and flips the
    # sign of phi for x < 0.
    rt = np.where(np.imag(rt) < 0.0, -rt, rt)
    return -1.0j * V * (c * c / (z + rt))


def potential_phi(x, c: float, V: float):
    """``phi(x, 0)`` from ``W(z) = -i V (z - sqrt(z^2 - c^2))``, ``z = x + 0i``.

    Branch: the principal square root, which on ``|x| < c`` equals
    ``+i sqrt(c^2 - x^2)``, so ``phi = -V sqrt(c^2 - x^2)`` there and ``phi = 0``
    outside the footprint.  The sign is fixed by physics, not by the formula
    being audited: the stagnation region of an impacting body must be
    over-pressured, which for ``p = -rho(dphi/dt + |grad phi|^2/2)`` requires
    ``dphi/dt < 0``, i.e. ``phi < 0`` at the apex.
    """
    x = np.asarray(x, dtype=float)
    return np.real(_w(x + 0.0j, c, V))


def potential_psi(x, c: float, V: float):
    """``psi(x, 0)``, the imaginary part of the same potential.

    Used for the plate-normal velocity, because differencing ``phi`` across
    ``y = 0`` is meaningless: the principal branch flips there, so a central
    difference in ``y`` measures the branch jump, not the derivative.  By the
    Cauchy-Riemann relation ``dphi/dy = -dpsi/dx`` the normal velocity is a
    smooth one-dimensional difference in ``x``.
    """
    x = np.asarray(x, dtype=float)
    return np.imag(_w(x + 0.0j, c, V))


def potential_phi_offplate(x, y, c: float, V: float):
    """Same potential evaluated above the plate (``y > 0``), diagnostics only."""
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    return np.real(_w(x + 1j * y, c, V))


def _velocities(x, c: float, h: float):
    """``(u, v)`` on the plate from finite differences of ``phi`` and ``psi``."""
    u = (potential_phi(x + h, c, 1.0) - potential_phi(x - h, c, 1.0)) / (2.0 * h)
    v = -(potential_psi(x + h, c, 1.0) - potential_psi(x - h, c, 1.0)) / (2.0 * h)
    return u, v


def _dc_dt(c: float, V: float, beta: float) -> float:
    """``c = C V t``  =>  ``dc/dt = C V``."""
    return wagner_c_coeff(beta) * V


def eulerian_pressure_over_q(tau, beta: float, c: float | None = None):
    """``p/q`` from Eulerian Bernoulli, every derivative taken numerically.

    Deterministic and free of the audited closed form: only
    :func:`potential_phi` / :func:`potential_phi_offplate` (the complex
    potential) and finite differences are used.
    """
    V = 1.0
    rho = 1.0
    q = 0.5 * rho * V * V
    C = wagner_c_coeff(beta)
    if c is None:
        c = 1.0
    tau = np.atleast_1d(np.asarray(tau, dtype=float))
    x = tau * c
    h = _FD_H * c

    # dphi/dt at fixed x: the footprint edge moves, c = c(t).
    dc = _FD_H * c
    dt = dc / _dc_dt(c, V, beta)
    dphi_dt = (potential_phi(x, c + dc, V) - potential_phi(x, c - dc, V)) / (2.0 * dt)

    u, v = _velocities(x, c, h)

    p = -rho * (dphi_dt + 0.5 * (u * u + v * v))
    return p / q


def relative_kinetic_pressure_over_q(tau, beta: float, c: float | None = None):
    """``p/q`` with only the *tangential* kinetic energy (``v^2`` dropped).

    This is the variant that reproduces the in-repo profile bit for bit; it is
    computed here to *identify* the origin of the offset, not to endorse it.
    """
    V = 1.0
    rho = 1.0
    q = 0.5 * rho * V * V
    C = wagner_c_coeff(beta)
    if c is None:
        c = 1.0
    tau = np.atleast_1d(np.asarray(tau, dtype=float))
    x = tau * c
    h = _FD_H * c
    dc = _FD_H * c
    dt = dc / _dc_dt(c, V, beta)
    dphi_dt = (potential_phi(x, c + dc, V) - potential_phi(x, c - dc, V)) / (2.0 * dt)
    u, _v = _velocities(x, c, h)
    p = -rho * (dphi_dt + 0.5 * u * u)
    return p / q


# --------------------------------------------------------------------------- #
# 2. Independent root-level BVP checks on the potential
# --------------------------------------------------------------------------- #

def bvp_residuals(beta: float, c: float = 1.0, n: int = 201) -> dict:
    """Numeric residuals of the three defining conditions of the potential.

    (i) ``phi = 0`` on the dry surface ``|x| > c``;
    (ii) ``dphi/dy = -V`` (body displacement rate) on ``|x| < c``;
    (iii) ``|W| ~ O(c^2/(2|z|))`` far field, i.e. no mass source at infinity.
    """
    V = 1.0
    h = _FD_H * c
    x_wet = np.linspace(-0.98 * c, 0.98 * c, n)
    x_dry = np.concatenate([np.linspace(-20.0 * c, -1.05 * c, n),
                            np.linspace(1.05 * c, 20.0 * c, n)])
    _u, v_wet = _velocities(x_wet, c, h)
    phi_dry = potential_phi(x_dry, c, V)
    far = np.array([10.0, 100.0, 1000.0]) * c
    decay = np.abs(potential_phi(far, c, V)) * far / (c * c)
    return {
        "max_abs_dphi_dy_minus_V": float(np.max(np.abs(np.abs(v_wet) - V))),
        "max_phi_dry": float(np.max(np.abs(phi_dry))),
        "far_field_c2_scaled": [float(v) for v in decay],
    }


# --------------------------------------------------------------------------- #
# 3. Reference numbers, in q units, side by side
# --------------------------------------------------------------------------- #

@dataclass
class ApexAudit:
    beta_deg: float
    C: float
    cp_apex_repo_profile: float
    cp_apex_eulerian: float
    cp_apex_graded_reference: float
    cp_peak_repo: float
    cp_peak_eulerian: float
    tau_peak: float

    def as_dict(self) -> dict:
        return dict(self.__dict__)


def apex_audit(beta_deg: float) -> ApexAudit:
    beta = math.radians(beta_deg)
    C = wagner_c_coeff(beta)
    return ApexAudit(
        beta_deg=beta_deg,
        C=C,
        cp_apex_repo_profile=float(wagner_p_over_q(0.0, beta)),
        cp_apex_eulerian=float(eulerian_pressure_over_q(0.0, beta)[0]),
        cp_apex_graded_reference=C,          # refs_derived()["wagner"]["cp_centre"]
        cp_peak_repo=float(wagner_peak_over_q(beta)),
        cp_peak_eulerian=float(np.max(eulerian_pressure_over_q(
            np.linspace(0.0, 0.999, 20001), beta))),
        tau_peak=float(wagner_peak_tau(beta)),
    )


def profile_offset(beta_deg: float, n: int = 401, tau_max: float = 0.95) -> dict:
    """``repo - eulerian`` and ``repo - relative-kinetic`` over the footprint.

    ``tau_max`` defaults to ``0.95`` on purpose: the in-repo profile is clamped
    to zero beyond ``wagner_validity_tau`` (0.984 at 30 deg) while the numerical
    variants are not, so including the clamped tail would mix a convention
    difference into a test of the profile *value*.
    """
    beta = math.radians(beta_deg)
    tau = np.linspace(0.0, tau_max, n)
    repo = np.array([float(wagner_p_over_q(t, beta)) for t in tau])
    eul = eulerian_pressure_over_q(tau, beta)
    rel = relative_kinetic_pressure_over_q(tau, beta)
    return {
        "max_abs_repo_minus_relative": float(np.max(np.abs(repo - rel))),
        "offset_repo_minus_eulerian_mean": float(np.mean(repo - eul)),
        "offset_repo_minus_eulerian_std": float(np.std(repo - eul)),
    }


def apex_unit_check(beta_deg: float, rho: float = 1000.0, V: float = 1.0) -> dict:
    """Is the graded stagnation reference in the same units as the measurement?"""
    beta = math.radians(beta_deg)
    q = 0.5 * rho * V * V
    C = wagner_c_coeff(beta)
    return {
        "measured_normalisation": "cp = p / q, q = rho V^2 / 2",
        "graded_reference": C,
        "graded_reference_in_q_units": C,                     # taken at face value
        "module_profile_at_tau0": float(wagner_p_over_q(0.0, beta)),
        "module_p_centre_in_q_units": wagner_p_centre(V, beta, rho) / q,
        "ratio_graded_over_profile": C / float(wagner_p_over_q(0.0, beta)),
        "wrong_by_factor": 2.0 if abs(C / float(wagner_p_over_q(0.0, beta)) - 0.5) < 1e-12 else None,
    }


# --------------------------------------------------------------------------- #
# 4. Effect on the case's reported errors (no re-run: reference-side shift only)
# --------------------------------------------------------------------------- #

def shift_errors(measured_cp: dict, beta_deg: float) -> dict:
    """Re-express quoted measurements against the corrected references.

    ``measured_cp`` maps a label to a value already in ``q`` units, e.g.
    ``{"cp_centre_mean": 4.873, "cp_band_peak": 28.05}``.  The numbers quoted in
    the case document are used only to show the *direction and size* of the
    shift; they are not re-graded here because the raw artefacts are not
    tracked in git.
    """
    beta = math.radians(beta_deg)
    C = wagner_c_coeff(beta)
    refs = {
        "cp_centre_mean": {"as_graded": C, "q_unit_corrected": 2.0 * C,
                           "eulerian_profile": float(eulerian_pressure_over_q(0.0, beta)[0])},
        "cp_band_peak": {"as_graded": float(wagner_peak_over_q(beta)),
                         "q_unit_corrected": float(wagner_peak_over_q(beta)),
                         "eulerian_profile": float(np.max(eulerian_pressure_over_q(
                             np.linspace(0.0, 0.99, 20001), beta)))},
        "cp_at_peak_tau": {"as_graded": float(wagner_peak_over_q(beta)),
                           "q_unit_corrected": float(wagner_peak_over_q(beta)),
                           "eulerian_profile": float(np.max(eulerian_pressure_over_q(
                               np.linspace(0.0, 0.999, 20001), beta)))},
    }
    out = {}
    for label, value in measured_cp.items():
        if label not in refs:
            continue
        r = refs[label]
        out[label] = {
            "measured": value,
            **r,
            "err_as_graded": (value - r["as_graded"]) / r["as_graded"],
            "err_q_unit_corrected": (value - r["q_unit_corrected"]) / r["q_unit_corrected"],
            "err_eulerian": (value - r["eulerian_profile"]) / r["eulerian_profile"],
        }
    return out


def _closed_form_dev(C: float, tau_max: float, n: int = 2001) -> float:
    """``max |numeric Eulerian - (2 C s - s^2)|`` on ``0 <= tau <= tau_max``."""
    tau = np.linspace(0.0, tau_max, n)
    s = 1.0 / np.sqrt(1.0 - tau * tau)
    closed = 2.0 * C * s - s * s
    return float(np.max(np.abs(eulerian_pressure_over_q(tau, math.radians(30.0)) - closed)))


def report(beta_deg: float = 30.0) -> dict:
    a = apex_audit(beta_deg)
    return {
        "beta_deg": beta_deg,
        "apex": a.as_dict(),
        "profile_offset": profile_offset(beta_deg),
        "apex_unit_check": apex_unit_check(beta_deg),
        "bvp_residuals": bvp_residuals(math.radians(beta_deg)),
        "eulerian_closed_form_check": {
            "note": ("Eulerian profile should equal 2 C s - s^2 with s = 1/sqrt(1-tau^2); "
                     "evaluated on 0 <= tau <= 0.95 (the flat-plate potential is singular at tau -> 1)"),
            "max_abs_dev": _closed_form_dev(a.C, 0.95),
        },
    }


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--beta", type=float, default=30.0)
    ap.add_argument("--json", action="store_true")
    args = ap.parse_args()
    rep = report(args.beta)
    if args.json:
        print(json.dumps(rep, indent=2, default=float))
        return 0
    a = rep["apex"]
    print(f"beta = {a['beta_deg']} deg, C = {a['C']:.6f}, tau_peak = {a['tau_peak']:.6f}")
    print(f"  apex  Cp (q units): repo profile {a['cp_apex_repo_profile']:.4f} | "
          f"eulerian {a['cp_apex_eulerian']:.4f} | graded reference {a['cp_apex_graded_reference']:.4f}")
    print(f"  peak  Cp (q units): repo profile {a['cp_peak_repo']:.4f} | "
          f"eulerian {a['cp_peak_eulerian']:.4f}")
    o = rep["profile_offset"]
    print(f"  max|repo - relative-kinetic| = {o['max_abs_repo_minus_relative']:.3e}")
    print(f"  repo - eulerian = {o['offset_repo_minus_eulerian_mean']:.6f} "
          f"+- {o['offset_repo_minus_eulerian_std']:.2e}  (q units)")
    print(f"  BVP residuals: {json.dumps(rep['bvp_residuals'])[:160]}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())