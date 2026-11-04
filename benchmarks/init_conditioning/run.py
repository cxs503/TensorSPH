"""Measure the lattice initial-density offset and the spurious pressure it implies.

The question this benchmark answers (``backlog_update_m2971.md`` item C2 / P1):

    Why does every summation-density case start with a spurious pressure, and
    where does it come from - the resolution, the sound speed, or the lattice?

Method: build a static disc on a hexagonal and a square lattice, compute the
summation density of the *fully supported* particles (kernel neighbourhood
saturated, so no free-surface deficit is mixed in), and read the offset
``b = rho_init / rho0 - 1``. Then turn ``b`` into a pressure with the solver's
own Tait law and compare against the acceptance criterion
``max|p| / (rho0 c0^2) < 1e-4`` with and without mass calibration.

Everything here is a t = 0 evaluation: no time stepping, no time integration
scheme, so the numbers are properties of the initial condition alone.
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import math
import os
import sys
from pathlib import Path

import torch

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[1]
sys.path.insert(0, str(REPO / "src"))

from tensorsph.droplet_impact import density_from_neighbor_pairs, tait_pressure  # noqa: E402
from tensorsph.initial_conditions import (  # noqa: E402
    TAIT_ACCEPTANCE_LIMIT,
    calibrate_lattice_mass,
    measure_density_bias,
    spurious_pressure,
)
from tensorsph.neighbor_search import cell_linked_pairs  # noqa: E402

RHO0 = 1000.0
#: Geometry of the Young-Laplace verification droplet (benchmarks/laplace_pressure_floor):
#: sigma = 7.2 N/m (10 x water, to shorten the capillary time), R = 0.02 m, so
#: sigma / R = 360 Pa is the signal the spurious pressure is measured against.
RADIUS = 0.02
SIGMA = 7.2
GAMMA = 7.0
H_FACTOR = 1.3
CENTER = 0.5

LAPLACE_SIGNAL_PA = SIGMA / RADIUS


def droplet_lattice(radius: float, dx: float, lattice: str = "hex",
                    center: float = CENTER, rho0: float = RHO0):
    """Particles inside a disc plus the mass per particle (``rho0 * V0``)."""
    if lattice == "hex":
        dy = dx * math.sqrt(3.0) / 2.0
        rows = torch.arange(center - radius - 2 * dx, center + radius + 2 * dx,
                            dy, dtype=torch.float64)
        pts = []
        for k, y in enumerate(rows):
            xs = torch.arange(center - radius - 2 * dx, center + radius + 2 * dx,
                              dx, dtype=torch.float64)
            xs = xs + (0.5 * dx if k % 2 else 0.0)
            pts.extend((float(x), float(y)) for x in xs)
        pos = torch.tensor(pts, dtype=torch.float64)
        area = dx * dy
    elif lattice == "square":
        xs = torch.arange(center - radius - 2 * dx, center + radius + 2 * dx,
                          dx, dtype=torch.float64)
        xx, yy = torch.meshgrid(xs, xs, indexing="ij")
        pos = torch.stack([xx.flatten(), yy.flatten()], dim=1)
        area = dx * dx
    else:
        raise ValueError(f"unknown lattice {lattice!r}")
    keep = (pos[:, 0] - center).square() + (pos[:, 1] - center).square() <= radius ** 2
    pos = pos[keep].clone()
    mass = torch.full((pos.shape[0],), rho0 * area, dtype=torch.float64)
    return pos, mass, area


def probe(radius: float, dx: float, lattice: str, h_factor: float, c0: float,
          rho0: float = RHO0, gamma: float = GAMMA) -> dict:
    """Bias + spurious pressure before and after mass calibration, at t = 0."""
    pos, mass0, _ = droplet_lattice(radius, dx, lattice, rho0=rho0)
    h = h_factor * dx
    pairs = cell_linked_pairs(pos, 2.0 * h)

    before = spurious_pressure(pos, mass0, h, rho0, c0, gamma, pairs=pairs)
    bias = measure_density_bias(pos, mass0, h, rho0, pairs=pairs)
    mass_cal, info = calibrate_lattice_mass(pos, mass0, h, rho0, pairs=pairs)
    after = spurious_pressure(pos, mass_cal, h, rho0, c0, gamma, pairs=pairs)

    return {
        "radius": radius,
        "dx": dx,
        "lattice": lattice,
        "h_factor": h_factor,
        "sound_speed": c0,
        "n_particles": int(pos.shape[0]),
        "n_supported": bias["n_samples"],
        "bias": bias["bias"],
        "bias_percent": bias["bias_percent"],
        "calibrated_bias": info["bias_after"],
        "mass_change": info["mass_change"],
        "p_abs_max_before": before["p_abs_max"],
        "p_norm_before": before["p_max_over_rho0c0sq"],
        "p_abs_max_after": after["p_abs_max"],
        "p_norm_after": after["p_max_over_rho0c0sq"],
        "signal_ratio_before": before["p_abs_max"] / LAPLACE_SIGNAL_PA,
        "signal_ratio_after": after["p_abs_max"] / LAPLACE_SIGNAL_PA,
        "acceptance_limit": TAIT_ACCEPTANCE_LIMIT,
        "passes_before": before["passes_acceptance"],
        "passes_after": after["passes_acceptance"],
    }


def laplace_impact(dx: float) -> dict:
    """Effect of the calibration on the merged CSF jump measurement.

    Runs the Laplace verification instrument (``benchmarks/laplace_droplet``)
    with the calibrated masses substituted into the same geometry, so the delta
    is attributable to C2 alone: same lattice, same force model, same bins.
    """
    spec = importlib.util.spec_from_file_location(
        "laplace_droplet_run", REPO / "benchmarks" / "laplace_droplet" / "run.py")
    harness = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(harness)

    original = harness.droplet_lattice

    def make(radius, dx_, lattice="hex", center=(harness.CENTER, harness.CENTER),
             rho0=harness.RHO0, _cal=True):
        pos, mass, area = original(radius, dx_, lattice, center, rho0)
        if _cal:
            pos_c, mass_c, _ = original(radius, dx_, lattice, center, rho0)
            h = H_FACTOR * dx_
            mass, _ = calibrate_lattice_mass(pos_c, mass_c, h, rho0)
            pos = pos_c
        return pos, mass, area

    rows = {}
    for calibrated in (False, True):
        harness.droplet_lattice = (
            lambda *a, _c=calibrated, **k: make(*a, _cal=_c, **k))
        rows["calibrated" if calibrated else "uncalibrated"] = harness.run_case(
            radius=RADIUS, dx=dx, sigma=0.072, lattice="hex")
    harness.droplet_lattice = original

    uncal = rows["uncalibrated"]
    cal = rows["calibrated"]
    return {
        "dx": dx,
        "dp_ratio_uncalibrated": uncal["dp_ratio"],
        "dp_ratio_calibrated": cal["dp_ratio"],
        "dp_ratio_delta_points": 100.0 * (cal["dp_ratio"] - uncal["dp_ratio"]),
        "n_delta_uncalibrated": uncal["N_delta"],
        "n_delta_calibrated": cal["N_delta"],
        "kappa_delta_percent": 100.0 * (
            cal.get("kappa_mean", float("nan")) / uncal.get("kappa_mean", float("nan"))
            - 1.0) if "kappa_mean" in uncal else None,
    }


def radial_profile(dx: float, lattice: str = "hex", h_factor: float = H_FACTOR,
                   n_bins: int = 20) -> dict:
    """rho / rho0 against r / R before and after the mass calibration.

    The interior plateau is the lattice offset (what the calibration removes);
    the drop below r/R ~ 0.8 is the free-surface kernel deficit, which is a
    different problem (the free-surface pressure closure, P0) and is *not*
    touched here.
    """
    pos, mass0, _ = droplet_lattice(RADIUS, dx, lattice)
    h = h_factor * dx
    pairs = cell_linked_pairs(pos, 2.0 * h)
    mass_cal, info = calibrate_lattice_mass(pos, mass0, h, RHO0, pairs=pairs)
    r = torch.linalg.vector_norm(
        pos - torch.tensor([CENTER, CENTER], dtype=torch.float64), dim=1) / RADIUS
    edges = torch.linspace(0.0, 1.2, n_bins + 1, dtype=torch.float64)
    out = {"dx": dx, "lattice": lattice, "h_factor": h_factor, "bins": [],
           "bias": info["bias"], "mass_change": info["mass_change"]}
    ratios = {
        "uncalibrated": density_from_neighbor_pairs(pos, mass0, h, pairs) / RHO0,
        "calibrated": density_from_neighbor_pairs(pos, mass_cal, h, pairs) / RHO0,
    }
    for k in range(n_bins):
        lo, hi = float(edges[k]), float(edges[k + 1])
        sel = (r >= lo) & (r < hi)
        if int(sel.sum()) == 0:
            continue
        row = {"r_over_R": 0.5 * (lo + hi), "n": int(sel.sum())}
        for name, ratio in ratios.items():
            row[name] = float(ratio[sel].mean())
        out["bins"].append(row)
    return out


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--out", default=str(HERE / "report.json"))
    ap.add_argument("--quick", action="store_true", help="skip the finest spacing")
    args = ap.parse_args(argv)

    spacings = [0.004, 0.002, 0.001]
    if args.quick:
        spacings = spacings[:3]

    sweep = []
    for lattice in ("hex", "square"):
        for h_factor in (1.2, 1.3, 1.4, 1.5):
            for dx in spacings:
                if dx == spacings[-1] and h_factor != H_FACTOR:
                    continue  # the finest spacing only at the reference h/dx
                sweep.append(probe(RADIUS, dx, lattice, h_factor, c0=20.0))

    # Sound-speed ladder at the default hex geometry: the pressure offset goes
    # as c0^2 while the bias stays fixed, which identifies it as p = B (rho/rho0)^gamma - B.
    c0_ladder = [probe(RADIUS, 0.002, "hex", H_FACTOR, c0=c0)
                 for c0 in (5.0, 10.0, 20.0, 40.0)]

    # Resolution ladder at fixed lattice: pins that the bias does *not* converge.
    resolution = [probe(RADIUS, dx, "hex", H_FACTOR, c0=20.0) for dx in spacings]

    impacts = [laplace_impact(0.01)]

    hex_default = [r for r in sweep
                   if r["lattice"] == "hex" and r["h_factor"] == H_FACTOR]
    profiles = [radial_profile(dx) for dx in (0.002,)]
    import numpy as _np
    profiles[0]["bins"] = [{k: (None if isinstance(v, _np.floating) else v)
                            for k, v in b.items()} for b in profiles[0]["bins"]]

    report = {
        "generated_by": "benchmarks/init_conditioning/run.py",
        "radial_profiles": profiles,
        "settling": {
            "rho0": RHO0,
            "radius": RADIUS,
            "gamma": GAMMA,
            "sigma": SIGMA,
            "laplace_signal_Pa": LAPLACE_SIGNAL_PA,
            "acceptance_limit": TAIT_ACCEPTANCE_LIMIT,
            "note": "t = 0 evaluation only; no time integration",
        },
        "sweep": sweep,
        "resolution_ladder_hex_h13": resolution,
        "sound_speed_ladder_hex_h13": c0_ladder,
        "laplace_impact": impacts,
        "summary": {
            "hex_bias_percent": sorted({round(r["bias_percent"], 6)
                                        for r in hex_default}),
            "hex_bias_dx_independent": len({round(r["bias"], 8)
                                            for r in hex_default}) == 1,
            "hex_p_norm_before": sorted({round(r["p_norm_before"], 8)
                                         for r in hex_default}),
            "hex_p_norm_after_max": max(r["p_norm_after"] for r in hex_default),
            "hex_signal_ratio_before": hex_default[0]["signal_ratio_before"],
            "hex_mass_change_percent": 100.0 * hex_default[0]["mass_change"],
            "worst_p_norm_after": max(r["p_norm_after"] for r in sweep),
            "all_pass_after": all(r["passes_after"] for r in sweep),
            "worst_bias_percent": max(sweep, key=lambda r: abs(r["bias"]))["bias_percent"],
        },
    }
    Path(args.out).write_text(json.dumps(report, indent=2))
    print(json.dumps(report["summary"], indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())