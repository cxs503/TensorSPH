"""Round 6 of ``CASE-SPH-OSC-DROP-FREQ``: the repaired time-advancing arm.

Round 5 built the axisymmetric dynamic arm and reported an honest
FAIL-to-instrument: the free drop diverged in every configuration, *including the
surface-tension free control*, so the defect was localised to the
density/pressure evolution rather than to the CSF operator.  ``round6_diagnose.py``
turns that localisation into two concrete defects and both are fixed in
``src/tensorsph/oscillating_drop_dynamic.py``:

1. **Per-particle mass calibration inflated the free surface.**  The calibration
   ``mass_i <- mass_i rho0 / rho_summation,i`` was applied to every particle,
   surface rows included; there the support is truncated by 10-25 %, so those
   masses were scaled *up* and the first interior shell reached ``1.04 rho0``.
   With ``c0 = 20`` that is ``p = 18`` where the physical pressure is
   ``2 sigma/R = 2`` -- an acoustic blast of ``|a| = 110`` on a cloud at rest.
   Round 5 measured exactly this as the runaway.  Fix: the calibration is gated
   on the Shepard volume completeness (:func:`tensorsph.oscillating_drop_dynamic.
   completeness`), so truncated rows keep the nominal lattice mass and their
   deficit correctly reads as ``p = 0``.
2. **The Tait inversion was missing its ``gamma``.**  The initial state imposed
   ``rho0 (1 + p/(rho0 c0^2))^(1/gamma)`` where the inverse of
   ``p = (rho0 c0^2/gamma)(s^gamma - 1)`` is ``(1 + gamma p/(rho0 c0^2))^(1/gamma)``.
   The interior therefore started at ``p = p_lap/gamma = 0.286`` instead of
   ``2.0`` -- a factor-``gamma`` force imbalance against a *correct* CSF traction.

Both are now regression-tested (``tests/test_oscillating_drop_freq_round6.py``):
the tension-free drop must stay at rest to machine precision
(``|v| = 0``, measured) and the initial interior pressure must equal ``2 sigma/R``
to EOS round-off (``p_max = 2.0000``, measured).

This driver is the payoff: it marches the repaired arm and reports the measured
mode-2 frequency against Rayleigh (1879), ``omega^2 = l(l-1)(l+2) sigma/(rho R^3)``
= 8 for ``l = 2`` in the normalisation ``R = rho0 = sigma = 1``.

``c0`` is a real design knob here and round 5's choice was the wrong side of the
trade-off: the Laplace compression is ``gamma p_lap/(rho0 c0^2)``, so ``c0 = 20``
resolves the capillary pressure with a 0.49 % density signal -- the same order as
the summation-density noise of this lattice -- while ``c0 = 5`` gives 6.6 %.
The measured capillary velocity is ``omega eps R = 0.14``, so ``c0 = 5`` keeps the
acoustic Mach number at 0.028.

Usage::

    PYTHONPATH=src python benchmarks/oscillating_drop_freq/round6_dynamic.py --quick
    PYTHONPATH=src python benchmarks/oscillating_drop_freq/round6_dynamic.py   # ladder
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from tensorsph.oscillating_drop_dynamic import (DynamicConfig, run_dynamic_drop,
                                                dynamic_frequency)

OUT = Path(__file__).with_name("out")


def run_grid(points, periods, mode=2, eps=0.05):
    rows = []
    for p in points:
        cfg = DynamicConfig(radius=1.0, rho0=1.0, sigma=1.0, dx=1.0 / p["rdx"],
                            mode=mode, eps=eps, c0=p["c0"],
                            delta_sph=p.get("delta_sph", 0.1),
                            periods=periods, skip_periods=0.25,
                            alpha_av=0.02, sample_every=4)
        row = run_dynamic_drop(cfg)
        row["point"] = p
        rows.append(row)
        print(f"rdx={p['rdx']:>5} c0={p['c0']:>4} ds={p.get('delta_sph', 0.1):>3} "
              f"n={row['n_particles']:>4} steps={row['n_steps']:>6} "
              f"omega={row['omega_measured']:8.4f} err={row['omega_error_pct']:+7.2f}% "
              f"vmax={row['max_speed']:8.3f} rho_drift={row['density_drift_max']:.4f}")
    return rows


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--quick", action="store_true",
                    help="single coarse point, ~2 periods (smoke)")
    ap.add_argument("--periods", type=float, default=3.0)
    ap.add_argument("--out", default="round6_dynamic.json")
    args = ap.parse_args()
    if args.quick:
        points = [{"rdx": 9.0, "c0": 5.0}]
        periods = 1.0
    else:
        # resolution ladder at the chosen c0, plus the c0 sensitivity it replaces
        points = [
            {"rdx": 9.0, "c0": 5.0},
            {"rdx": 12.0, "c0": 5.0},
            {"rdx": 16.0, "c0": 5.0},
            {"rdx": 12.0, "c0": 10.0},
            {"rdx": 12.0, "c0": 20.0},
        ]
        periods = args.periods
    rows = run_grid(points, periods)
    OUT.mkdir(parents=True, exist_ok=True)
    payload = {"mode": 2, "eps": 0.05, "sigma": 1.0, "periods": periods,
               "omega_analytic": 8.0 ** 0.5, "rows": rows}
    (OUT / args.out).write_text(json.dumps(payload, indent=1, default=float))
    print(f"wrote {OUT / args.out}")


if __name__ == "__main__":
    main()