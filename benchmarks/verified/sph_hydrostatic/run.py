"""M2 benchmark: hydrostatic pressure profile, ``p(z) = rho g (H - z)``.

    python benchmarks/verified/sph_hydrostatic/run.py --dx 0.02 --tank-width 1.0
    python benchmarks/verified/sph_hydrostatic/run.py --figure

The case is sized so that the campaign's ``N >= 1000`` is met *and measured*
(``particles`` is read from the run, not computed from the lattice): with
``dx = 0.02 m`` the tank is 1.0 m wide by 0.4 m deep, i.e. 50 x 20 = 1000
particles.  The gate is the interior-band error of the time-averaged profile; the
free-surface band is excluded (summation density is truncated there) and reported
separately.

Exit status is non-zero when the gate or the ``rho0 = 0`` control fails.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[3] / "src"))

from tensorsph.verification import hydrostatic as m2, report
from tensorsph.verification.evidence import EvidenceStore

HERE = Path(__file__).resolve().parent
STORE = HERE / "evidence.sqlite"


def _verdict(passed: bool) -> str:
    return "PASS" if passed else "FAIL"


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--dx", type=float, default=0.02)
    ap.add_argument("--tank-width", type=float, default=1.0)
    ap.add_argument("--depth", type=float, default=0.4)
    ap.add_argument("--seconds", type=float, default=2.0)
    ap.add_argument("--sound-speed", type=float, default=20.0)
    ap.add_argument("--min-particles", type=int, default=1000)
    ap.add_argument("--limit-pct", type=float, default=2.0)
    ap.add_argument("--figure", action="store_true")
    ap.add_argument("--threads", type=int, default=8)
    args = ap.parse_args(argv)
    torch.set_num_threads(args.threads)

    store = EvidenceStore(STORE)
    if args.figure:
        rows = store.query("SELECT run_id FROM runs WHERE case_name="
                           "'hydrostatic_pressure_profile' ORDER BY run_id DESC LIMIT 1")
        if not rows:
            raise SystemExit("no hydrostatic run recorded")
        png = report.figure_hydrostatic(store, int(rows[0]["run_id"]),
                                        HERE / "figures" / "hydrostatic_profile.png")
        print("figure:", png.relative_to(HERE))
        store.close()
        return 0

    case = m2.HydrostaticCase(dx=args.dx, tank_width=args.tank_width,
                              depth=args.depth, seconds=args.seconds,
                              sound_speed=args.sound_speed,
                              min_particles=args.min_particles)
    result = m2.run_hydrostatic_case(case)
    gate = m2.gate(result, limit_pct=args.limit_pct)
    control = m2.zero_density_control()

    run_id = store.add_run("hydrostatic_pressure_profile",
                           {"dx": args.dx, "tank_width": args.tank_width,
                            "depth": args.depth, "seconds": args.seconds,
                            "sound_speed": args.sound_speed,
                            "min_particles": args.min_particles},
                           notes="still-water tank, interior bands, time mean")
    measured = result["measured"]
    store.add_metrics(run_id, {
        "max_band_error_pct": dict(value=gate["measured_pct"], unit="%",
                                   gate_limit=gate["limit_pct"],
                                   verdict=_verdict(gate["pass"]),
                                   provenance="max |p_SPH - rho g (H-z)| / reference "
                                              "over interior bands, time mean"),
        "mean_band_error_pct": dict(value=measured["mean_band_error_pct"], unit="%"),
        "n_particles": dict(value=float(measured["n_particles"]), unit="1",
                            gate_limit=float(args.min_particles),
                            verdict=_verdict(measured["n_particles"]
                                             >= args.min_particles)),
        "surface_band_error_pct": dict(value=measured["surface_band_error_pct"],
                                       unit="%"),
        "max_band_error_pct_instantaneous": dict(
            value=measured["max_band_error_pct_instantaneous"], unit="%"),
        "steps": dict(value=float(result["config"]["steps"]), unit="1"),
        "gate_pass": dict(value=float(gate["pass"]), unit="bool", gate_limit=1.0,
                          verdict=_verdict(gate["pass"])),
    })
    store.add_series(run_id, "bands", measured["bands"])

    ctrl_id = store.add_run("hydrostatic_zero_density_control",
                            {"control": "rho0 = 0"}, notes=str(control))
    store.add_metrics(ctrl_id, {
        "eos_at_zero_density": dict(value=control["eos_at_zero_density"], unit="Pa",
                                    gate_limit=0.0,
                                    verdict=_verdict(control["eos_pointwise_zero"])),
        "gate_pass": dict(value=float(control["pass"]), unit="bool", gate_limit=1.0,
                          verdict=_verdict(control["pass"])),
    })
    out = {"config": result["config"], "measured": measured, "gate": gate,
           "control": control, "run_id": run_id, "control_run_id": ctrl_id}
    (HERE / "results.json").write_text(json.dumps(out, indent=1))
    print(f"N={measured['n_particles']} max band error "
          f"{gate['measured_pct']:.3f} % (mean "
          f"{measured['mean_band_error_pct']:.3f} %, surface band "
          f"{measured['surface_band_error_pct']:.1f} %), gate="
          f"{'PASS' if gate['pass'] else 'FAIL'}, rho0=0 control "
          f"{'PASS' if control['pass'] else 'FAIL'}", flush=True)
    store.close()
    return 0 if (gate["pass"] and control["pass"]) else 1


if __name__ == "__main__":
    raise SystemExit(main())
