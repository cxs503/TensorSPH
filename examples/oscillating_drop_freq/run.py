"""CASE-SPH-OSC-DROP-FREQ: 2-D oscillating drop frequency vs Rayleigh (1879).

Reference
---------
Rayleigh, Lord (1879). "On the Capillary Phenomena of Jets".
*Proc. R. Soc. Lond.* **29**: 71–97.

For a 2-D drop (liquid cylinder) the exact oscillation frequency is::

    omega_m^2 = sigma * m * (m^2 - 1) / (rho * R^3)

where ``m >= 2`` is the mode number, ``sigma`` the surface tension,
``rho`` the density and ``R`` the equilibrium radius.

Gate
----
``|omega_meas - omega_exact| / omega_exact < 6 %`` at ``R/dx = 24``.

Usage
-----
::

    PYTHONPATH=src python examples/oscillating_drop_freq/run.py
    PYTHONPATH=src python examples/oscillating_drop_freq/run.py --quick

The ``--quick`` flag runs a single coarse point (R/dx = 8, 0.4 periods)
for a smoke test; the default runs the full verification ladder
(R/dx = 12, 16, 24, 3 periods each).
"""
from __future__ import annotations

import argparse
import json
import math
import sys
import time
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))

from tensorsph.capillary_oscillation import (
    MODE_DEFAULT,
    RADIUS_DEFAULT,
    RHO_WATER,
    SIGMA_WATER_AIR,
    mode_amplitude,
    period_from_crossings,
    rayleigh_2d,
    run_point,
)

# --------------------------------------------------------------------------- #
# Default parameters (water drop in air, SI units)
# --------------------------------------------------------------------------- #
SIGMA = SIGMA_WATER_AIR       # 0.072 N/m
RHO = RHO_WATER               # 1000 kg/m^3
RADIUS = RADIUS_DEFAULT       # 0.05 m
MODE = MODE_DEFAULT            # 2 (quadrupole)
GATE_PCT = 6.0


def analytic_reference() -> dict:
    """Return the exact Rayleigh frequency and period."""
    omega = rayleigh_2d(SIGMA, RHO, RADIUS, MODE)
    period = 2.0 * math.pi / omega
    return {
        "sigma": SIGMA,
        "rho": RHO,
        "radius": RADIUS,
        "mode": MODE,
        "omega_exact_rad_s": omega,
        "period_exact_s": period,
        "formula": f"omega^2 = sigma * {MODE} * ({MODE}^2 - 1) / (rho * R^3)",
        "reference": "Rayleigh 1879, Proc. R. Soc. Lond. 29:71",
    }


def run_ladder(resolutions: list[int], periods: float = 3.0,
               eps: float = 0.05) -> list[dict]:
    """Run the verification ladder at multiple resolutions.

    Each point integrates the drop for ``periods`` exact periods and
    extracts the measured period from zero-crossings of the mode amplitude.
    """
    omega_exact = rayleigh_2d(SIGMA, RHO, RADIUS, MODE)
    t_exact = 2.0 * math.pi / omega_exact

    rows = []
    for rdx in resolutions:
        t0 = time.time()
        result = run_point(
            radius_over_dx=rdx,
            sigma=SIGMA,
            periods=periods,
            device="cpu",
            eps=eps,
            radius=RADIUS,
            mode=MODE,
            rho0=RHO,
            sample_every=5,
        )
        wall_s = time.time() - t0

        period_meas = result["period_measured_s"]
        omega_meas = 2.0 * math.pi / period_meas if math.isfinite(period_meas) else float("nan")
        omega_err_pct = 100.0 * (omega_meas - omega_exact) / omega_exact if math.isfinite(omega_meas) else float("nan")

        row = {
            "rdx": rdx,
            "n_particles": result["particles"],
            "n_steps": result["steps"],
            "wall_seconds": round(wall_s, 2),
            "period_exact_s": t_exact,
            "period_measured_s": round(period_meas, 6) if math.isfinite(period_meas) else None,
            "omega_exact_rad_s": omega_exact,
            "omega_measured_rad_s": round(omega_meas, 6) if math.isfinite(omega_meas) else None,
            "omega_error_pct": round(omega_err_pct, 4) if math.isfinite(omega_err_pct) else None,
            "gate_pass": abs(omega_err_pct) < GATE_PCT if math.isfinite(omega_err_pct) else False,
            "n_crossings": result["zero_crossings"],
            "q_initial": round(result["q_initial"], 6),
            "q_final": round(result["q_final"], 6),
            "max_r_over_R": round(result["containment_max_r_over_R"], 4),
            "escaped_particles": result["escaped_particles"],
        }
        rows.append(row)
        status = "PASS" if row["gate_pass"] else "FAIL"
        print(f"  R/dx={rdx:2d}: omega_err={omega_err_pct:+7.3f}%  "
              f"T_meas={period_meas:.4f}s  n_cross={result['zero_crossings']}  "
              f"particles={result['particles']}  [{status}]")

    return rows


def verdict(rows: list[dict]) -> dict:
    """Determine whether the gate is met at the finest resolution."""
    if not rows:
        return {"status": "no_data", "statement": "No data points."}

    finest = max(rows, key=lambda r: r["rdx"])
    all_pass = all(r["gate_pass"] for r in rows)
    finest_pass = finest["gate_pass"]

    if finest_pass:
        status = "PASS"
        statement = (
            f"Gate met at R/dx={finest['rdx']}: "
            f"omega error = {finest['omega_error_pct']:+.3f}% "
            f"(threshold {GATE_PCT}%)."
        )
    else:
        status = "FAIL"
        statement = (
            f"Gate NOT met at R/dx={finest['rdx']}: "
            f"omega error = {finest['omega_error_pct']:+.3f}% "
            f"(threshold {GATE_PCT}%)."
        )

    return {
        "status": status,
        "finest_resolution": finest["rdx"],
        "finest_omega_error_pct": finest["omega_error_pct"],
        "all_rows_pass": all_pass,
        "n_rows": len(rows),
        "n_passing": sum(1 for r in rows if r["gate_pass"]),
        "statement": statement,
    }


def main() -> int:
    ap = argparse.ArgumentParser(
        description="CASE-SPH-OSC-DROP-FREQ: 2-D oscillating drop vs Rayleigh (1879)")
    ap.add_argument("--quick", action="store_true",
                    help="Run a single coarse point (R/dx=8, 0.4 periods) for smoke test")
    ap.add_argument("--eps", type=float, default=0.05,
                    help="Initial perturbation amplitude (default 0.05)")
    ap.add_argument("--periods", type=float, default=None,
                    help="Number of exact periods to integrate (default: 3 for full, 0.4 for quick)")
    ap.add_argument("--output", type=str, default=None,
                    help="Output JSON path (default: examples/oscillating_drop_freq/out/result.json)")
    args = ap.parse_args()

    quick = args.quick
    eps = args.eps
    periods = args.periods if args.periods is not None else (0.4 if quick else 3.0)
    resolutions = [8] if quick else [12, 16, 24]

    print("=" * 70)
    print("CASE-SPH-OSC-DROP-FREQ: 2-D Oscillating Drop Frequency")
    print("Reference: Rayleigh 1879, Proc. R. Soc. Lond. 29:71")
    print("=" * 70)

    ref = analytic_reference()
    print(f"\nAnalytic reference (mode m={ref['mode']}):")
    print(f"  omega_exact = {ref['omega_exact_rad_s']:.6f} rad/s")
    print(f"  T_exact     = {ref['period_exact_s']:.6f} s")
    print(f"  sigma       = {ref['sigma']} N/m")
    print(f"  rho         = {ref['rho']} kg/m^3")
    print(f"  R           = {ref['radius']} m")
    print(f"  Gate        = < {GATE_PCT}%")

    print(f"\nVerification ladder (eps={eps}, {periods} periods):")
    rows = run_ladder(resolutions, periods=periods, eps=eps)

    v = verdict(rows)
    print(f"\nVerdict: {v['status']}")
    print(f"  {v['statement']}")

    # Build output payload
    payload = {
        "case": "SPH-OSC-DROP-FREQ",
        "dimension": "2D (liquid cylinder)",
        "reference": ref["reference"],
        "formula": ref["formula"],
        "parameters": {
            "sigma": ref["sigma"],
            "rho": ref["rho"],
            "radius": ref["radius"],
            "mode": ref["mode"],
            "eps": eps,
        },
        "analytic": {
            "omega_exact_rad_s": ref["omega_exact_rad_s"],
            "period_exact_s": ref["period_exact_s"],
        },
        "ladder": rows,
        "verdict": v,
        "gate_pct": GATE_PCT,
    }

    out_path = Path(args.output) if args.output else (
        Path(__file__).resolve().parent / "out" / "result.json")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(payload, indent=2))
    print(f"\nResults written to: {out_path}")

    return 0 if v["status"] == "PASS" else 1


if __name__ == "__main__":
    raise SystemExit(main())
