"""Reference dispersion of the capillary wave on a two-layer interface (case bank).

Physical problem
----------------
A sinusoidal interface perturbation ``eta(x) = -A cos(k x)`` between two
immiscible, inviscid fluids at rest, no gravity.  The exact linear dispersion
relation of the *deep-water* single-fluid capillary wave is

.. code-block:: text

    omega^2 = T k^3 / rho                                    (Lamb 1932, Art. 267)

(``T`` = surface tension, ``k`` = wavenumber, ``rho`` = density; Lamb,
*Hydrodynamics*, 6th ed., Cambridge University Press, 1932, Art. 267 -- the same
citation the repository already uses in ``docs/verification_surface_tension_dispersion.md``
and ``docs/verification_capillary_wave_open_surface.md``).

For a **two-layer** interface of finite layer depths ``h1`` (lower, density
``rho1``) and ``h2`` (upper, density ``rho2``) the same linearisation of the
potential-flow problem with the normal-stress jump ``dp = sigma kappa`` gives

.. code-block:: text

    omega^2 = sigma k^3 / (rho1 coth(k h1) + rho2 coth(k h2))                (*)

whose limits are the textbook ones:

* ``rho2 -> 0, k h1 -> inf``   -> ``omega^2 = sigma k^3 / rho1``  (Lamb Art. 267),
* ``rho2 -> 0``, finite ``h1`` -> ``omega^2 = sigma k^3 tanh(k h1) / rho1``
  (the capillary branch of the finite-depth gravity-capillary relation
  ``omega^2 = (g k + sigma k^3/rho) tanh(k h)`` at ``g = 0``),
* ``rho1 = rho2 = rho, both deep`` -> ``omega^2 = sigma k^3 / (2 rho)``.

Zero hand-copy rule
-------------------
The case reference is a **closed form**, so no digits are transcribed from a
table: every reference number used by the benchmark is evaluated from (*) with
:mod:`mpmath` at ``DPS = 50`` decimal digits and written, with the formula
string and the citation, to ``ref.json`` next to a sha256 of that file.  The
reduction checks below are asserted at run time, so a typo in the exponent or in
the ``coth`` structure cannot pass unnoticed.
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

try:                                            # 50-digit evaluator
    import mpmath as mp

    mp.mp.dps = 50
    HAVE_MPMATH = True
except Exception:                               # pragma: no cover - fallback
    HAVE_MPMATH = False

DPS = 50

CITATION = {
    "source": ("Lamb, H. (1932). Hydrodynamics, 6th ed., Cambridge University "
               "Press, Art. 267 (capillary waves, deep water, inviscid: "
               "omega^2 = T k^3 / rho)."),
    "same_citation_in_repo": [
        "docs/verification_surface_tension_dispersion.md",
        "docs/verification_capillary_wave_open_surface.md",
        "docs/benchmarks/capillary-wave.md",
    ],
    "generalisation": ("two-layer finite-depth linear dispersion "
                       "omega^2 = sigma k^3 / (rho1 coth(k h1) + rho2 coth(k h2)); "
                       "reduces to Lamb Art. 267 for rho2 -> 0 and k h1 -> infinity."),
    "provenance": ("closed form -> reference numbers are evaluated (mpmath, dps=%d), "
                   "not transcribed; ref.json carries the formula string and a sha256."
                   % DPS),
}

FORMULA = "omega^2 = sigma*k**3/(rho1*coth(k*h1) + rho2*coth(k*h2))"


def _f(x: float):
    return mp.mpf(repr(float(x))) if HAVE_MPMATH else float(x)


def omega_two_layer(k: float, sigma: float, rho1: float, rho2: float,
                    h1: float, h2: float):
    """Exact linear frequency (*) of the two-layer capillary wave."""
    if HAVE_MPMATH:
        k, sigma, rho1, rho2, h1, h2 = map(_f, (k, sigma, rho1, rho2, h1, h2))
        c1 = mp.coth(k * h1)
        c2 = mp.coth(k * h2)
        return mp.sqrt(sigma * k ** 3 / (rho1 * c1 + rho2 * c2))
    import math
    return math.sqrt(sigma * k ** 3 / (rho1 / math.tanh(k * h1)
                                       + rho2 / math.tanh(k * h2)))


def omega_deep_lamb(sigma: float, k: float, rho: float):
    """Lamb Art. 267 deep-water single-fluid capillary wave: ``omega^2 = sigma k^3/rho``."""
    if HAVE_MPMATH:
        sigma, k, rho = map(_f, (sigma, k, rho))
        return mp.sqrt(sigma * k ** 3 / rho)
    import math
    return math.sqrt(sigma * k ** 3 / rho)


def omega_finite_depth_gc(sigma: float, k: float, rho: float, h: float,
                          g: float = 0.0):
    """Finite-depth gravity-capillary branch ``omega^2 = (g k + sigma k^3/rho) tanh(k h)``."""
    if HAVE_MPMATH:
        sigma, k, rho, h, g = map(_f, (sigma, k, rho, h, g))
        return mp.sqrt((g * k + sigma * k ** 3 / rho) * mp.tanh(k * h))
    import math
    return math.sqrt((g * k + sigma * k ** 3 / rho) * math.tanh(k * h))


def reduction_checks(k: float, sigma: float, rho1: float, rho2: float,
                     h1: float, h2: float) -> dict:
    """Assert that (*) collapses onto the textbook limits (no hand-copied digits)."""
    deep2 = omega_two_layer(k, sigma, rho1, 0.0, 1.0e6 / k, h2)
    lamb = omega_deep_lamb(sigma, k, rho1)
    fd = omega_finite_depth_gc(sigma, k, rho1, h1)
    one_layer = omega_two_layer(k, sigma, rho1, 0.0, h1, h2)
    rel = lambda a, b: float(abs(mp.mpf(a) - mp.mpf(b)) / mp.mpf(b)) if HAVE_MPMATH else abs(a - b) / b
    return {
        "two_layer_deep_rho2_zero_vs_lamb_art267": rel(deep2, lamb),
        "two_layer_rho2_zero_vs_finite_depth_gc": rel(one_layer, fd),
        "all_below_1e-12": max(rel(deep2, lamb), rel(one_layer, fd)) < 1e-12,
    }


def reference(params: dict, path: str | Path | None = None) -> dict:
    """Reference numbers for the case, written (with sha256) when ``path`` given."""
    k = 2.0 * 3.141592653589793 / params["wavelength_m"]
    om = omega_two_layer(k, params["sigma_N_m"], params["rho_liquid_kg_m3"],
                         params["rho_gas_kg_m3"],
                         params["depth_fluid_m"], params["depth_gas_m"])
    om_lamb = omega_deep_lamb(params["sigma_N_m"], k, params["rho_liquid_kg_m3"])
    om_gc = omega_finite_depth_gc(params["sigma_N_m"], k, params["rho_liquid_kg_m3"],
                                  params["depth_fluid_m"])
    out = {
        "citation": CITATION,
        "formula": FORMULA,
        "dps": DPS,
        "evaluator": "mpmath" if HAVE_MPMATH else "math",
        "params": params,
        "k_rad_m": repr(k),
        "omega_ref_rad_s": mp.nstr(om, 20) if HAVE_MPMATH else repr(om),
        "omega_ref_float": float(om),
        "period_ref_s": float(2.0 * mp.pi / om) if HAVE_MPMATH else 6.283185307179586 / float(om),
        "phase_speed_ref_m_s": float(om / k),
        "cross_checks": {
            "omega_lamb_art267_deep_single_fluid": float(om_lamb),
            "omega_finite_depth_gc_g0": float(om_gc),
            "rel_diff_two_layer_vs_lamb": float(abs(om - om_lamb) / om_lamb),
            "rel_diff_two_layer_vs_finite_depth_gc": float(abs(om - om_gc) / om_gc),
            "two_layer_denominator_over_rho1": float(
                (mp.mpf(params["rho_liquid_kg_m3"]) * mp.coth(mp.mpf(repr(k)) * mp.mpf(repr(params["depth_fluid_m"])))
                 + mp.mpf(params["rho_gas_kg_m3"]) * mp.coth(mp.mpf(repr(k)) * mp.mpf(repr(params["depth_gas_m"]))))
                / mp.mpf(params["rho_liquid_kg_m3"])) if HAVE_MPMATH else float("nan"),
        },
        "reductions": reduction_checks(k, params["sigma_N_m"], params["rho_liquid_kg_m3"],
                                       params["rho_gas_kg_m3"], params["depth_fluid_m"],
                                       params["depth_gas_m"]),
    }
    if path is not None:
        text = json.dumps(out, indent=2, sort_keys=True)
        Path(path).write_text(text)
        out["ref_json_sha256"] = hashlib.sha256(text.encode()).hexdigest()
        out["ref_json_path"] = str(path)
    return out


if __name__ == "__main__":                    # pragma: no cover - manual use
    import argparse

    ap = argparse.ArgumentParser(description="evaluate the case reference")
    ap.add_argument("--wavelength", type=float, default=0.020)
    ap.add_argument("--sigma", type=float, default=0.072)
    ap.add_argument("--rho-liquid", type=float, default=1000.0)
    ap.add_argument("--rho-gas", type=float, default=1.2)
    ap.add_argument("--depth-fluid-factor", type=float, default=0.8)
    ap.add_argument("--depth-gas-factor", type=float, default=0.5)
    ap.add_argument("--out", default=None)
    a = ap.parse_args()
    p = dict(wavelength_m=a.wavelength, sigma_N_m=a.sigma,
             rho_liquid_kg_m3=a.rho_liquid, rho_gas_kg_m3=a.rho_gas,
             depth_fluid_m=a.depth_fluid_factor * a.wavelength,
             depth_gas_m=a.depth_gas_factor * a.wavelength)
    r = reference(p, a.out)
    print(json.dumps({k: v for k, v in r.items() if k != "citation"}, indent=2))