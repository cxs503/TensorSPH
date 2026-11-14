"""CASE-SPH-OSC-DROP-FREQ: 3-D drop oscillation frequency vs Rayleigh (1879).

Reference (Rayleigh 1879, *Proc. R. Soc. Lond.* 29:71; Lamb 1932 Art. 275):
``omega_l^2 = l (l-1) (l+2) sigma / (rho R^3)`` -- ``l = 2`` gives
``8 sigma / (rho R^3)``.  Gate: ``< 6 %`` on ``omega``.

Two measurements, both recorded whether they pass or not:

``axisym`` (primary)
    linear-response modal stiffness of the *discretised* CSF operator on the
    meridional (axisymmetric) drop: ``omega_meas/omega_exact =
    sqrt(dp_meas/dp_exact)``.  Discrete-dispersion measurement, not a
    trajectory.  Resolution ladder ``R/dx = 8..20`` x modes ``l = 2, 3`` x
    amplitudes ``eps = 0.02 .. 0.1``, plus ``eps -> 0`` rows that measure the
    mode-``l`` noise floor of the estimator.
``lattice3d`` (audit)
    the shipped 3-D CSF operator (:mod:`tensorsph.threed.csf3`, "mls" curvature)
    on an FCC sphere: delta measure, mean curvature ``kappa R`` and the modal
    traction ratio.  Recorded because it is the operator a full-3-D anchor would
    have to use, and its numbers are what rule it out.

Usage::

    PYTHONPATH=src python benchmarks/oscillating_drop_freq/run_freq.py
    PYTHONPATH=src python benchmarks/oscillating_drop_freq/run_freq.py --quick
"""
from __future__ import annotations

import argparse
import json
import math
import platform
import sys
import time
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))

from tensorsph.oscillating_drop3d import (  # noqa: E402
    axisym_drop_geometry,
    axisym_modal_stress,
    csf3_audit,
    fcc_sphere,
    flat_surface,
    rayleigh_frequency_2d,
    rayleigh_frequency_3d,
)

RADIUS = 0.02                 # m
SIGMA = 7.2                   # N/m
RHO = 1000.0                  # kg/m^3
GATE_PCT = 6.0
OUT = Path(__file__).resolve().parent


def analytic_block() -> dict:
    out = {"radius": RADIUS, "sigma": SIGMA, "rho": RHO, "gate_pct": GATE_PCT,
           "reference": "Rayleigh 1879, Proc. R. Soc. Lond. 29:71; Lamb 1932 Art. 275",
           "branches": {}}
    for mode in (2, 3, 4):
        w3 = rayleigh_frequency_3d(SIGMA, RHO, RADIUS, mode)
        w2 = rayleigh_frequency_2d(SIGMA, RHO, RADIUS, mode)
        out["branches"][str(mode)] = {
            "omega_3d_rad_s": w3,
            "omega_2d_rad_s": w2,
            "period_3d_s": 2.0 * math.pi / w3,
            "formula_3d": f"{mode}*{mode - 1}*{mode + 2}*sigma/(rho R^3)",
            "formula_2d": f"{mode}*({mode}^2-1)*sigma/(rho R^3)",
            "frequency_ratio_3d_over_2d": w3 / w2,
            "frequency_gap_pct": (w3 / w2 - 1.0) * 100.0,
        }
    return out


def axisym_block(res_list, modes, eps_list, noise_eps) -> dict:
    rows = []
    for mode in modes:
        for eps in list(eps_list) + [noise_eps]:
            for rdx in res_list:
                geom = axisym_drop_geometry(RADIUS, RADIUS / rdx, mode=mode,
                                            eps=eps, rho0=RHO)
                t0 = time.time()
                row = axisym_modal_stress(geom, SIGMA)
                row["wall_seconds"] = time.time() - t0
                row["is_noise_floor"] = eps <= noise_eps
                row["omega_analytic"] = rayleigh_frequency_3d(SIGMA, RHO, RADIUS, mode)
                row["omega_measured"] = row["omega_analytic"] * row["omega_ratio"]
                rows.append(row)
                print(f"  axisym l={mode} eps={eps:g} R/dx={rdx:g} "
                      f"werr={row['omega_error_pct']:+7.2f}% "
                      f"kappaR={row['kappa_mean_R']:.3f} "
                      f"measure={row['surface_measure']:.3f}", flush=True)
    noise = {}
    for mode in modes:
        for rdx in res_list:
            sub = [r for r in rows if r["mode"] == mode
                   and abs(r["radius_over_dx"] - rdx) < 1e-9
                   and r["is_noise_floor"]]
            if sub:
                # spurious mode amplitude: the noise-floor traction converted to
                # the equivalent P_l radial amplitude eps_noise
                noise[f"l{mode}_rdx{rdx:g}"] = abs(
                    sub[0]["dp_measured"]) / (SIGMA * (mode - 1) * (mode + 2) / RADIUS)
    return {"rows": rows, "noise_equivalent_eps": noise}


def lattice3d_block(res_list) -> dict:
    plane = []
    for hf in (1.3, 2.0):
        geom = flat_surface(0.05, half_height=0.4, h_factor=hf)
        t0 = time.time()
        res = csf3_audit(geom, geom["h"], rho0=1.0, area=geom["area"],
                         stencil=2.1)
        res["wall_seconds"] = time.time() - t0
        res["h_factor"] = hf
        plane.append(res)
        print(f"  plane h={hf}: delta measure {res['delta_measure_ratio']:.3f}",
              flush=True)
    sphere = []
    for rdx in res_list:
        for eps in (0.0, 0.03):
            geom = fcc_sphere(1.0, 1.0 / rdx, mode=2, eps=eps)
            t0 = time.time()
            res = csf3_audit(geom, geom["h"], rho0=1.0, mode=2, eps=max(eps, 1e-9),
                             radius=1.0, stencil=2.1)
            res["wall_seconds"] = time.time() - t0
            res["radius_over_dx"] = rdx
            res["eps"] = eps
            sphere.append(res)
            print(f"  sphere3d R/dx={rdx} eps={eps}: kappaR="
                  f"{res['kappa_mean_R']:.3f} measure={res['delta_measure_ratio']:.3f}",
                  flush=True)
    return {"plane": plane, "sphere": sphere}


def verdict(payload) -> dict:
    """Score the primary (axisym) ladder against the 6 % gate."""
    rows = [r for r in payload["axisym"]["rows"] if not r["is_noise_floor"]]
    best = min(rows, key=lambda r: abs(r["omega_error_pct"]))
    finest = sorted(rows, key=lambda r: -r["radius_over_dx"])[0]
    passing = [r for r in rows if abs(r["omega_error_pct"]) < GATE_PCT]
    return {
        "gate_pct": GATE_PCT,
        "n_rows": len(rows),
        "n_passing": len(passing),
        "best": {k: best[k] for k in ("mode", "eps", "radius_over_dx",
                                      "omega_analytic", "omega_measured",
                                      "omega_error_pct")},
        "finest_resolution_row": {k: finest[k] for k in
                                  ("mode", "eps", "radius_over_dx",
                                   "omega_error_pct")},
        "gate_pass": abs(finest["omega_error_pct"]) < GATE_PCT,
        "gate_pass_statement": (
            "finest-resolution row within 6 %" if abs(finest["omega_error_pct"]) < GATE_PCT
            else "no resolution/mode/amplitude combination reproduces the Rayleigh "
                 "frequency within 6 %; the estimator's mode-2 noise floor is of "
                 "the same order as the seeded perturbation (see "
                 "noise_equivalent_eps)"),
    }


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--quick", action="store_true")
    args = ap.parse_args()
    res_list = [8.0, 12.0] if args.quick else [8.0, 12.0, 16.0, 20.0]
    modes = [2] if args.quick else [2, 3]
    eps_list = [0.02, 0.05] if args.quick else [0.02, 0.05, 0.1]
    noise_eps = 1.0e-6
    payload = {
        "case": "SPH-OSC-DROP-FREQ",
        "reference": "Rayleigh 1879",
        "gate_pct": GATE_PCT,
        "generated": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "environment": {"python": platform.python_version(),
                        "torch": torch.__version__,
                        "device": "cpu", "threads": torch.get_num_threads()},
        "analytic": analytic_block(),
    }
    print("analytic:", json.dumps(payload["analytic"]["branches"]["2"],
                                  indent=None)[:160], flush=True)
    print("axisym ladder:", flush=True)
    payload["axisym"] = axisym_block(res_list, modes, eps_list, noise_eps)
    print("3-D lattice operator audit:", flush=True)
    payload["lattice3d"] = lattice3d_block([8.0, 12.0] if args.quick else [8.0, 12.0, 16.0])
    payload["verdict"] = verdict(payload)
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "report.json").write_text(json.dumps(payload, indent=2))
    print("verdict:", json.dumps(payload["verdict"], indent=2), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())