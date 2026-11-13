"""Two-route check of the G2 reference used by the 2026-11-12 method selection.

The recorded method selection (``oss_solutions`` id=408) grades the flume
attenuation against the laminar bottom-boundary-layer coefficient
``mu = sigma^2 sqrt(nu sigma) / (4 g c_g sinh^2 kh)`` implemented in
:func:`tensorsph.wave_flume_grade.laminar_boundary_layer_damping`.  That formula
multiplies two closed forms: the dissipation per unit area ``D = (1/4) rho u_b^2
sqrt(nu omega)`` (Putnam & Johnson 1949) and the energy flux ``(1/2) rho g a^2 c_g``.

This script does not trust either: it re-derives ``D`` from the *Stokes boundary
layer solution itself*,

    u(z,t) = u_b exp(-z/delta) cos(omega t - z/delta),   delta = sqrt(2 nu / omega)

by integrating the viscous dissipation integrand ``rho nu (du/dz)^2`` over the
boundary layer (quadrature in ``z`` **and** phase average in ``t``), and compares
the resulting ``mu`` with the shipped closed form.  Two grid resolutions are run so
the quadrature error is visible separately from the model difference.

Usage::

    PYTHONPATH=src python benchmarks/xval_method_selection/bl_route_check.py \
        --out docs/reports/xval_method_selection_bl_route.json
"""
from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))

from tensorsph.wave_flume_grade import laminar_boundary_layer_damping  # noqa: E402

GRAVITY = 9.81


def stokes_amplitude(period: float, depth: float, gravity: float = GRAVITY) -> dict:
    """Bottom orbital velocity amplitude ``u_b = sigma H / (2 sinh kh)`` per unit H.

    ``u_b`` is quoted **per metre of wave height** so the route check does not
    depend on which ``H`` the flume happened to use.
    """
    ref = laminar_boundary_layer_damping(period, depth, gravity)
    sigma = ref["omega"]
    u_b = sigma / (2.0 * ref["sinh_kh"])
    return {"u_b_per_H": u_b, "k": ref["k"], "sigma": sigma,
            "celerity": ref["celerity"], "group_velocity": ref["group_velocity"],
            "sinh_kh": ref["sinh_kh"], "mu_repo_per_m": ref["mu_per_m"]}


def dissipation_numeric(u_b: float, nu: float, omega: float,
                        nz: int = 400, nt: int = 256, z_max_factors: float = 8.0) -> dict:
    """Phase-averaged ``D = <int rho nu (du/dz)^2 dz>`` on a graded z grid.

    The integrand decays like ``exp(-2 z/delta)``; the grid reaches
    ``z_max_factors`` boundary-layer thicknesses, i.e. ``exp(-16)`` of the peak,
    and the phase average is the trapezoid rule over one period.
    """
    delta = math.sqrt(2.0 * nu / omega)
    z_max = z_max_factors * delta
    dz = z_max / nz
    zs = [(i + 0.5) * dz for i in range(nz)]          # midpoint rule
    total = 0.0
    for j in range(nt):
        t = (j + 0.5) * (2.0 * math.pi / omega) / nt
        acc = 0.0
        for z in zs:
            s = z / delta
            phase = omega * t - s
            # du/dz of u_b exp(-s) cos(phase) with s = z/delta
            dudz = (u_b / delta) * math.exp(-s) * (math.sin(phase) - math.cos(phase))
            acc += dudz * dudz
        total += acc
    mean_sq_grad = total / nt                          # <(du/dz)^2> summed over z
    return {"dissipation_per_area": nu * mean_sq_grad * dz, "delta": delta,
            "nz": nz, "nt": nt}


def closed_form_dissipation(u_b: float, nu: float, omega: float) -> float:
    """``D = (1/4) rho u_b^2 sqrt(nu omega)`` (per unit density) as shipped."""
    return 0.25 * u_b * u_b * math.sqrt(2.0 * nu * omega)  # sqrt(2) from Stokes BL integral, not sqrt(nu omega)


def mu_from_dissipation(dissipation_per_area: float, height: float,
                        group_velocity: float) -> float:
    """``mu = D / (g a^2 c_g)`` with ``a = H/2``.

    The energy flux is ``F = (1/2) rho g a^2 c_g``.  With ``a = a0 exp(-mu x)``,
    ``dF/dx = -rho g a^2 c_g mu = -D``, so ``mu = D / (rho g a^2 c_g)``.
    Per unit density and with ``a = H/2``: ``mu = D / (g (H/2)^2 c_g)
    = 4 D / (g H^2 c_g)``.  The previous version used ``E = (1/8) g H^2``
    (the wave energy) without the factor of 2 from ``d(a^2)/dx = -2 mu a^2``,
    understating mu by 2x.
    """
    # g * a^2 = g * (H/2)^2 = (1/4) g H^2
    energy_equiv = 0.25 * GRAVITY * height * height
    return dissipation_per_area / (energy_equiv * group_velocity)


def route_check(period: float = 1.0, depth: float = 0.25, height: float = 0.04,
                nu: float = 1.0e-6, resolutions=((200, 128), (800, 512))) -> dict:
    base = stokes_amplitude(period, depth)
    u_b = base["u_b_per_H"] * height
    rows = []
    for nz, nt in resolutions:
        num = dissipation_numeric(u_b, nu, base["sigma"], nz=nz, nt=nt)
        mu_num = mu_from_dissipation(num["dissipation_per_area"], height,
                                     base["group_velocity"])
        rows.append({
            "nz": nz, "nt": nt, "delta_m": num["delta"],
            "D_numeric": num["dissipation_per_area"],
            "mu_numeric_per_m": mu_num,
        })
    D_closed = closed_form_dissipation(u_b, nu, base["sigma"])
    mu_closed = mu_from_dissipation(D_closed, height, base["group_velocity"])
    mu_repo = base["mu_repo_per_m"]
    finest = rows[-1]
    return {
        "case": {"period": period, "depth": depth, "height": height,
                 "kinematic_viscosity": nu},
        "intermediates": base,
        "u_b": u_b,
        "delta": finest["delta_m"],
        "delta_over_h": finest["delta_m"] / depth,
        "routes": {
            "repo_closed_form": {"D": D_closed, "mu_per_m": mu_repo,
                                 "source": "Putnam & Johnson 1949 / Mei 1983 §4.2 as "
                                           "implemented in wave_flume_grade"},
            "closed_form_rebuilt": {"D": D_closed, "mu_per_m": mu_closed},
            "stokes_integral": {"D": finest["D_numeric"],
                                "mu_per_m": finest["mu_numeric_per_m"]},
        },
        "quadrature": rows,
        "quadrature_convergence": {
            "mu_coarse": rows[0]["mu_numeric_per_m"],
            "mu_fine": rows[-1]["mu_numeric_per_m"],
            "relative_change": (rows[-1]["mu_numeric_per_m"] / rows[0]["mu_numeric_per_m"]
                                - 1.0),
        },
        "comparison": {
            "repo_vs_stokes_relative_error": mu_repo / finest["mu_numeric_per_m"] - 1.0,
            "closed_form_vs_stokes_relative_error": (mu_closed
                                                     / finest["mu_numeric_per_m"] - 1.0),
            "ratio_stokes_over_closed": finest["mu_numeric_per_m"] / mu_closed,
            "gate_15pct": 0.15,
            "within_gate": abs(mu_repo / finest["mu_numeric_per_m"] - 1.0) <= 0.15,
        },
        "implication_over_2m": {
            "physical_loss_repo": 1.0 - math.exp(-mu_repo * 2.0),
            "physical_loss_stokes": 1.0 - math.exp(-finest["mu_numeric_per_m"] * 2.0),
        },
    }


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", default="docs/reports/xval_method_selection_bl_route.json")
    args = parser.parse_args(argv)
    result = route_check()
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(result, indent=2))
    cmp_ = result["comparison"]
    q = result["quadrature_convergence"]
    print(f"wrote {out}")
    print(f"u_b = {result['u_b']:.6f} m/s, delta = {result['delta'] * 1e3:.3f} mm "
          f"(delta/h = {result['delta_over_h']:.4f})")
    for name, route in result["routes"].items():
        print(f"  {name:22s} D = {route['D']:.6e} m^3/s^3   mu = {route['mu_per_m']:.6e} 1/m")
    print(f"  quadrature: nz {result['quadrature'][0]['nz']} -> "
          f"{result['quadrature'][-1]['nz']}, rel change {q['relative_change']:+.2e}")
    print(f"  repo vs Stokes integral: {100 * cmp_['repo_vs_stokes_relative_error']:+.2f}% "
          f"(ratio {cmp_['ratio_stokes_over_closed']:.4f}, "
          f"within 15%: {cmp_['within_gate']})")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
