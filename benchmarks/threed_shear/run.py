#!/usr/bin/env python3
"""3-D shear-wave decay against the exact unsteady-Stokes solution.

Case
----
Triply periodic cube, one wavelength per edge.  Initial velocity

    u_x(y, z, 0) = A sin(k y) sin(k z),   k = 2 pi / L,

which is divergence free and has a vanishing convective term (``u_x`` is
independent of ``x``), so the incompressible Navier-Stokes equations reduce to
``du/dt = nu laplacian u`` with the exact solution

    u_x(y, z, t) = A sin(k y) sin(k z) exp(-2 nu k^2 t).

The measured quantity is the exponential decay exponent of the mode amplitude
``a(t) = 4 <u_x sin(ky) sin(kz)>``: reference ``2 nu k^2``, measured by least
squares on ``ln a``.  Nothing is calibrated against the run itself: ``A``, ``nu``,
``k`` and the sound speed enter as inputs.

What it tests
-------------
* the 3-D cubic-spline kernel and its gradient/Laplacian on an evolving cloud,
* the triply periodic minimum-image neighbour search,
* the summation density (calibrated mass keeps ``rho = rho0`` to round-off),
* the viscous Laplacian: raw versus second-moment renormalised, which is the
  difference between +17 % and a couple of percent at the same resolution,
* conservation: no wall, no gravity, central pair forces and a periodic box, so
  total linear momentum must sit at round-off -- that is a gate, not a comment.

Usage::

    PYTHONPATH=src python benchmarks/threed_shear/run.py --only ladder-n16 --out /tmp/c.json
    PYTHONPATH=src python benchmarks/threed_shear/run.py --merge --figure
"""
from __future__ import annotations

import argparse
import glob
import json
import math
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))

from tensorsph.threed import run_shear_wave  # noqa: E402

REPO = Path(__file__).resolve().parents[2]
OUT = REPO / "benchmarks" / "threed_shear"
BOX = 1.0
NU = 0.05
AMPLITUDE = 0.1
SOUND_SPEED = 10.0

#: ``n`` points per wavelength along each axis at the reference ``h/dx``.
LADDER = (12, 16, 20, 24)
#: the h/dx sweep at fixed resolution: the resolution/consistency trade-off.
H_SWEEP = (1.0, 1.2, 1.3, 1.5, 1.8)


def case_params(name: str) -> dict:
    if name.startswith("ladder-n"):
        return {"n": int(name.split("-n")[1]), "h_factor": 1.3, "renormalise": True}
    if name.startswith("raw-n"):
        return {"n": int(name.split("-n")[1]), "h_factor": 1.3, "renormalise": False}
    if name.startswith("h"):
        return {"n": 20, "h_factor": float(name[1:]), "renormalise": True}
    raise KeyError(name)


CASES = ([f"ladder-n{n}" for n in LADDER]
         + [f"raw-n{LADDER[2]}"]
         + [f"h{h:g}" for h in H_SWEEP])


def run_case(name: str, *, steps: int | None = None, sample_every: int = 10) -> dict:
    params = case_params(name)
    result = run_shear_wave(n=params["n"], box=BOX, h_factor=params["h_factor"],
                            nu=NU, amplitude=AMPLITUDE,
                            sound_speed=SOUND_SPEED, steps=steps,
                            sample_every=sample_every,
                            renormalise=params["renormalise"])
    payload = result.as_dict()
    payload["case"] = name
    payload["analytic_formula"] = "2 nu k^2 on u = A sin(ky) sin(kz) e^{-2 nu k^2 t}"
    payload["kinetic_energy_analytic_ratio"] = {
        "expected_decay_exponent": 2.0 * result.analytic_rate,
        "measured_ratio_final_over_initial": (
            payload["conservation"]["kinetic_energy_final"]
            / payload["conservation"]["kinetic_energy_initial"]),
        "expected_ratio_final_over_initial": math.exp(
            -2.0 * result.analytic_rate * result.times[-1]),
    }
    return payload


def merge(results: list[dict]) -> dict:
    by_name = {r["case"]: r for r in results}
    ladder = [by_name[f"ladder-n{n}"] for n in LADDER if f"ladder-n{n}" in by_name]
    sweep = [by_name[f"h{h:g}"] for h in H_SWEEP if f"h{h:g}" in by_name]
    raw = by_name.get(f"raw-n{LADDER[2]}")
    renorm = by_name.get(f"ladder-n{LADDER[2]}")
    finest = min(ladder, key=lambda r: r["kh"]) if ladder else None
    momentum_ok = all(
        (r["conservation"]["momentum_final_over_scale"] or 0.0) <= 1e-12
        for r in ladder)
    criteria = {
        "decay_rate_error_below_3pct_at_finest": bool(
            finest and abs(finest["decay"]["rel_error"]) < 0.03),
        "decay_rate_error_below_15pct_all_cases": bool(
            ladder and all(abs(r["decay"]["rel_error"]) < 0.15 for r in ladder)),
        "momentum_conserved_to_1e-12_of_sum_m_abs_v": bool(momentum_ok),
        "mass_exactly_conserved": bool(
            ladder and all(abs(r["conservation"]["mass_relative_drift"]) == 0.0
                           for r in ladder)),
        "renormalisation_improves_decay_rate": bool(
            raw and renorm and abs(renorm["decay"]["rel_error"])
            < abs(raw["decay"]["rel_error"])),
        "renormalisation_removes_most_of_the_bias": bool(
            raw and renorm and abs(renorm["decay"]["rel_error"])
            < 0.5 * abs(raw["decay"]["rel_error"])),
    }
    missing = sorted(set(CASES) - set(by_name))
    return {
        "benchmark": "threed_shear",
        "case": "3-D double-shear (unsteady Stokes) on a triply periodic cube",
        "reference": "u = A sin(ky) sin(kz) exp(-2 nu k^2 t); decay rate 2 nu k^2",
        "params": {"box": BOX, "nu": NU, "amplitude": AMPLITUDE,
                   "sound_speed": SOUND_SPEED, "periods": 1,
                   "h_over_dx_default": 1.3},
        "resolution_ladder": ladder,
        "h_over_dx_sweep": sweep,
        "raw_vs_renormalised": {
            "raw": _thin(raw), "renormalised": _thin(renorm),
        } if raw and renorm else None,
        "criteria": criteria,
        "missing_cases": missing,
        "verdict": "pass" if all(criteria.values()) and not missing else "fail",
    }


def _thin(row: dict | None) -> dict | None:
    if row is None:
        return None
    return {
        "case": row["case"],
        "n": row["n"],
        "kh": row["kh"],
        "decay": row["decay"],
        "momentum_final_over_scale": row["conservation"]["momentum_final_over_scale"],
        "momentum_scale": row["conservation"]["momentum_scale_sum_m_abs_v"],
    }


def figure(report: dict, path: Path) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(1, 3, figsize=(13.5, 3.9))
    ladder = report["resolution_ladder"]
    if ladder:
        series = ladder[0]["series"]
        times = series["times"]
        amp = series["amplitudes"]
        rate = ladder[0]["decay"]["analytic_rate"]
        axes[0].semilogy(times, amp, "o", ms=3, label=f"SPH n={ladder[0]['n']}")
        axes[0].semilogy(times, [amp[0] * math.exp(-rate * t) for t in times], "-",
                         label="exact $e^{-2\\nu k^2 t}$")
        axes[0].set_xlabel("t (s)")
        axes[0].set_ylabel("mode amplitude $a$")
        axes[0].set_title("shear-wave decay (coarsest ladder point)")
        axes[0].legend(fontsize=8)
        axes[0].grid(alpha=0.3, which="both")

        axes[1].axhline(0.0, color="k", lw=0.8)
        axes[1].axhline(0.03, color="k", ls=":", lw=1, label="3 %")
        axes[1].axhline(-0.03, color="k", ls=":", lw=1)
        axes[1].axhline(0.15, color="r", ls="--", lw=1, label="15 % platform limit")
        axes[1].plot([r["kh"] for r in ladder],
                      [r["decay"]["rel_error"] for r in ladder], "o-",
                      label="renormalised")
        raw = report.get("raw_vs_renormalised")
        if raw and raw["raw"]:
            axes[1].plot([raw["raw"]["kh"]], [raw["raw"]["decay"]["rel_error"]], "s",
                         label="raw Laplacian")
        axes[1].set_xlabel("$k h$")
        axes[1].set_ylabel("decay-rate relative error")
        axes[1].set_title("decay rate vs $2\\nu k^2$")
        axes[1].legend(fontsize=7)
        axes[1].grid(alpha=0.3)

    sweep = report["h_over_dx_sweep"]
    if sweep:
        axes[2].plot([r["h_over_dx"] for r in sweep],
                     [r["decay"]["rel_error"] for r in sweep], "o-")
        axes[2].axhline(0.0, color="k", lw=0.8)
        axes[2].axhline(0.03, color="k", ls=":", lw=1)
        axes[2].axhline(-0.03, color="k", ls=":", lw=1)
        axes[2].set_xlabel("$h/\\Delta x$")
        axes[2].set_ylabel("decay-rate relative error")
        axes[2].set_title(f"parameter sweep at n = {sweep[0]['n']}")
        axes[2].grid(alpha=0.3)

    fig.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=130)
    plt.close(fig)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--only", action="append", default=None)
    parser.add_argument("--out", default=None)
    parser.add_argument("--steps", type=int, default=None)
    parser.add_argument("--merge", nargs="*", default=None)
    parser.add_argument("--figure", action="store_true")
    args = parser.parse_args()

    if args.merge is not None:
        pattern = args.merge or [str(OUT / "cases" / "*.json")]
        files = [Path(p) for pat in pattern for p in glob.glob(pat)]
        results = [json.loads(f.read_text()) for f in files]
        report = merge(results)
        OUT.mkdir(parents=True, exist_ok=True)
        (OUT / "report.json").write_text(json.dumps(report, indent=2) + "\n")
        if args.figure:
            figure(report, OUT / "figures" / "threed_shear.png")
        print(json.dumps({"report": str(OUT / "report.json"),
                          "verdict": report["verdict"],
                          "criteria": report["criteria"],
                          "missing": report["missing_cases"]}, indent=2))
        return 0 if report["verdict"] == "pass" else 1

    torch.set_default_dtype(torch.float64)
    names = args.only or CASES
    out_path = Path(args.out) if args.out else None
    if out_path and len(names) != 1:
        raise SystemExit("--out needs exactly one --only case")
    for name in names:
        payload = run_case(name, steps=args.steps)
        if out_path:
            out_path.parent.mkdir(parents=True, exist_ok=True)
            out_path.write_text(json.dumps(payload, indent=2) + "\n")
        print(json.dumps({k: payload[k] for k in
                          ("case", "n", "kh", "decay", "pairs_last")}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
