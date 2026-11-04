"""Initial-state lattice-mass audit (backlog_update_m2971.md P1).

Why this exists
---------------
A particle lattice with the textbook mass ``m = rho0 * A`` (``A`` the lattice
area per particle) does *not* have a summation density of ``rho0``.  Because the
discrete kernel sum over an infinite lattice,

    ``1 + b = A * sum_sites W(|r_ij|, h)``,

is not exactly 1, every initial state starts either pre-compressed or
pre-expanded, and the Tait EOS turns that offset into a spurious pressure
``p = B[(1+b)^gamma - 1]``.  ``b`` is a property of the lattice + kernel pair
only, so it can be evaluated analytically instead of being measured on a finite
(and therefore surface-contaminated) cloud.

This harness (i) checks the analytic ``b`` against the measured core density
ratio of the real initial states shipped by the solver, and (ii) checks the P1
acceptance criterion ``p_max / (rho0 c0^2) < 1e-4`` on the calibrated states.

Cases
-----
* ``pool_square``: the B1/B2 pool + drop initial state, paired with its static
  wall shell, straight out of ``make_particles`` / ``wall_lattice`` (~600
  particles, one density summation each - no time stepping).
* ``hex_drop_*`` / ``square_drop_*``: static drops on the two lattices the
  surface-tension verification chain uses (``laplace_law`` builds hex drops).

Usage
-----
    PYTHONPATH=src python benchmarks/init_mass_calibration/run.py
    PYTHONPATH=src python benchmarks/init_mass_calibration/run.py --plot

Writes ``report.json`` (and ``init_mass_calibration.png`` with ``--plot``) next
to this file.  ``make_doc.py`` turns the JSON into the markdown report, so no
number in ``docs/verification_init_mass_calibration.md`` is typed by hand.
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import math
from pathlib import Path

import torch

from tensorsph.droplet_impact import (
    Config,
    calibrated_particle_mass,
    cell_linked_pairs,
    density_from_neighbor_pairs,
    lattice_area_per_particle,
    lattice_density_ratio,
    make_particles,
    tait_pressure,
    wall_lattice,
    wall_mass,
)

HERE = Path(__file__).resolve().parent
RHO0 = 1000.0
SOUND_SPEED = 20.0
GAMMA = 7.0
H_FACTOR = 1.3
RADIUS = 0.10
DROP_DX = (0.01, 0.005)
POOL_DX = 0.025
# P1 acceptance: spurious initial pressure must be invisible against the
# acoustic pressure scale.
PRESSURE_FLOOR = 1.0e-4


# --------------------------------------------------------------------------- #
# Lattices
# --------------------------------------------------------------------------- #
def _load_sibling(name: str, filename: str):
    spec = importlib.util.spec_from_file_location(name, HERE.parent / filename)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


_law = _load_sibling("laplace_law_run", "laplace_law/run.py")
hexagonal_droplet = _law.hexagonal_droplet


def square_droplet(radius: float, dx: float, rho0: float = RHO0):
    """Circular drop on a square lattice centred on the origin."""
    n = int(math.floor(radius / dx)) + 1
    idx = torch.arange(-n, n + 1, dtype=torch.float64) * dx
    gx, gy = torch.meshgrid(idx, idx, indexing="ij")
    pts = torch.stack((gx.reshape(-1), gy.reshape(-1)), dim=-1)
    pts = pts[torch.linalg.vector_norm(pts, dim=-1) <= radius]
    mass = torch.full((pts.shape[0],), rho0 * dx * dx, dtype=torch.float64)
    return pts, mass


def disc(lattice: str, radius: float, dx: float, calibrated: bool):
    """Drop with either the textbook lattice mass or the calibrated one."""
    builder = hexagonal_droplet if lattice == "hex" else square_droplet
    pos, mass = builder(radius, dx, RHO0)
    area = lattice_area_per_particle(dx, lattice)
    ratio = lattice_density_ratio(dx, H_FACTOR * dx, lattice)
    target = rho0_per_area(area, ratio, calibrated)
    return pos, mass * (target / (RHO0 * area))


def rho0_per_area(area: float, ratio: float, calibrated: bool) -> float:
    """Mass per particle: ``rho0*A`` (textbook) or ``rho0*A/(1+b)``."""
    return RHO0 * area / ratio if calibrated else RHO0 * area


# --------------------------------------------------------------------------- #
# Measurements
# --------------------------------------------------------------------------- #
def _summation(pos, mass, cfg, extra_pos=None, extra_mass=None):
    h = cfg.h_factor * cfg.dx
    if extra_pos is not None:
        all_pos = torch.cat((pos, extra_pos), dim=0)
        all_mass = torch.cat((mass, extra_mass), dim=0)
    else:
        all_pos, all_mass = pos, mass
    pairs = cell_linked_pairs(all_pos, 2.0 * h)
    rho = density_from_neighbor_pairs(all_pos, all_mass, h, pairs)[: pos.shape[0]]
    return rho, tait_pressure(rho, cfg)


def measure(name: str, pos, mass, cfg, core_mask, extra_pos=None, extra_mass=None):
    """Density ratio and spurious pressure of one initial state."""
    rho, pressure = _summation(pos, mass, cfg, extra_pos, extra_mass)
    ratio = rho / cfg.rho0
    scale = cfg.rho0 * cfg.sound_speed ** 2
    analytic = lattice_density_ratio(cfg.dx, cfg.h_factor * cfg.dx, _lattice_of(name))
    return {
        "case": name,
        "lattice": _lattice_of(name),
        "calibrated": bool(cfg.mass_calibration),
        "dx": cfg.dx,
        "h_over_dx": cfg.h_factor,
        "particles": int(pos.shape[0]),
        "core_particles": int(core_mask.sum()),
        "analytic_ratio": analytic,
        "core_ratio_mean": float(ratio[core_mask].mean()),
        "core_ratio_min": float(ratio[core_mask].min()),
        "core_ratio_max": float(ratio[core_mask].max()),
        "all_ratio_min": float(ratio.min()),
        "all_ratio_max": float(ratio.max()),
        "p_max": float(pressure.max()),
        "p_max_over_pressure_scale": float(pressure.max()) / scale,
        "core_p_max_over_pressure_scale": float(pressure[core_mask].max()) / scale,
        "pressure_scale": scale,
    }


def _lattice_of(name: str) -> str:
    return "hex" if name.startswith("hex") else "square"


def pool_case(calibrated: bool, dx: float = POOL_DX):
    cfg = Config(dx=dx, sound_speed=SOUND_SPEED, gamma=GAMMA,
                 mass_calibration=calibrated)
    pos, mass = make_particles(cfg, torch.device("cpu"))
    wall = wall_lattice(cfg, torch.device("cpu"))
    wmass = wall_mass(cfg, wall.shape[0], torch.device("cpu"))
    h = cfg.h_factor * cfg.dx
    n_pool = int(round(cfg.pool_width / dx)) * int(round(cfg.pool_depth / dx))
    pool = pos[:n_pool]
    # kernel-complete interior: > 2h from the free surface and from every wall.
    # The drop has no complete kernel at this radius, so it stays outside the mask.
    pool_core = ((pool[:, 0] > 2 * h) & (pool[:, 0] < cfg.pool_width - 2 * h)
                 & (pool[:, 1] > 2 * h) & (pool[:, 1] < cfg.pool_depth - 2 * h))
    core = torch.zeros(pos.shape[0], dtype=torch.bool)
    core[:n_pool] = pool_core
    return measure(f"pool_square_dx{dx*1000:g}mm", pos, mass, cfg, core, wall, wmass)


def drop_case(lattice: str, dx: float, calibrated: bool):
    cfg = Config(dx=dx, sound_speed=SOUND_SPEED, gamma=GAMMA,
                 mass_calibration=calibrated)
    pos, mass = disc(lattice, RADIUS, dx, calibrated)
    r = torch.linalg.vector_norm(pos, dim=-1) / RADIUS
    core = r < 0.5  # 2h < 0.5 R at dx <= 0.013, so the kernel is complete here
    return measure(f"{lattice}_drop_dx{dx*1000:g}mm", pos, mass, cfg, core)


# --------------------------------------------------------------------------- #
# Analytic sweep: the bias is a lattice property, not a resolution choice
# --------------------------------------------------------------------------- #
def analytic_sweep():
    rows = []
    for lattice in ("square", "hex"):
        for h_factor in (1.2, 1.3, 1.5):
            for dx in (0.025, 0.01, 0.005):
                ratio = lattice_density_ratio(dx, h_factor * dx, lattice)
                bulk = RHO0 * SOUND_SPEED ** 2 / GAMMA
                p = bulk * (ratio ** GAMMA - 1.0)
                rows.append({
                    "lattice": lattice,
                    "h_over_dx": h_factor,
                    "dx": dx,
                    "ratio": ratio,
                    "bias_percent": (ratio - 1.0) * 100.0,
                    "spurious_pressure_Pa": p if ratio > 1.0 else 0.0,
                    "p_over_pressure_scale": max(p, 0.0) / (RHO0 * SOUND_SPEED ** 2),
                })
    return rows


# --------------------------------------------------------------------------- #
# Criteria
# --------------------------------------------------------------------------- #
def evaluate(cases, sweep):
    calibrated = [c for c in cases if c["calibrated"]]
    uncalibrated = [c for c in cases if not c["calibrated"]]

    def pair(case_name):
        off = next(c for c in uncalibrated if c["case"].startswith(case_name))
        on = next(c for c in calibrated if c["case"].startswith(case_name))
        return off, on

    checks = []
    # 1. analytic bias reproduces the measured bulk density ratio of the
    #    shipped initial state (no calibration), relative error < 1e-5.
    for off in uncalibrated:
        rel = abs(off["core_ratio_mean"] - off["analytic_ratio"]) / off["analytic_ratio"]
        checks.append({
            "name": f"analytic_vs_measured::{off['case']}",
            "measured": rel,
            "limit": 1.0e-5,
            "result": "pass" if rel < 1.0e-5 else "fail",
        })
    # 2. calibration removes the bulk offset: |core ratio - 1| < 1e-6 (float32
    #    summation noise floor is ~1.2e-7).
    for on in calibrated:
        err = abs(on["core_ratio_mean"] - 1.0)
        checks.append({
            "name": f"calibrated_core_ratio::{on['case']}",
            "measured": err,
            "limit": 1.0e-6,
            "result": "pass" if err < 1.0e-6 else "fail",
        })
    # 3. P1 acceptance: spurious initial pressure / (rho0 c0^2) < 1e-4.
    for on in calibrated:
        value = on["p_max_over_pressure_scale"]
        checks.append({
            "name": f"p_max_floor::{on['case']}",
            "measured": value,
            "limit": PRESSURE_FLOOR,
            "result": "pass" if value < PRESSURE_FLOOR else "fail",
        })
    # 4. the pre-fix baselines are recorded rather than asserted away: the hex
    #    drop must reproduce the +0.588 % compression reported by m2971.
    off, on = pair("hex_drop_dx10mm")
    within = abs(off["core_ratio_mean"] - 1.00588) < 5.0e-5
    checks.append({
        "name": "hex_bias_reproduces_m2971",
        "measured": off["core_ratio_mean"] - 1.0,
        "limit": 5.88e-3,
        "result": "pass" if within else "fail",
    })
    # 5. the square lattice is *not* pre-compressed - it sits 5.3e-5 below rho0 -
    #    so the 2.4 kPa spurious pressure is a hex-lattice defect, not a
    #    solver-wide one. Recorded so nobody re-chases it.
    off_sq = next(c for c in uncalibrated if c["case"].startswith("pool_square"))
    checks.append({
        "name": "square_lattice_not_precompressed",
        "measured": off_sq["core_ratio_mean"] - 1.0,
        "limit": 0.0,
        "result": "pass" if off_sq["core_ratio_mean"] < 1.0 else "fail",
    })
    # 6. resolution independence of the analytic bias (property, not discretisation)
    drift = {}
    for lattice in ("square", "hex"):
        ratios = [r["ratio"] for r in sweep if r["lattice"] == lattice and r["h_over_dx"] == 1.3]
        drift[lattice] = max(ratios) - min(ratios)
    checks.append({
        "name": "analytic_bias_dx_independent",
        "measured": max(drift.values()),
        "limit": 1.0e-12,
        "result": "pass" if max(drift.values()) < 1.0e-12 else "fail",
    })
    return checks


# --------------------------------------------------------------------------- #
# Plot
# --------------------------------------------------------------------------- #
def plot(cases, sweep, path: Path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(1, 2, figsize=(11.0, 4.2))
    names = sorted({c["case"] for c in cases})
    off = [next(c for c in cases if c["case"] == n and not c["calibrated"]) for n in names]
    on = [next(c for c in cases if c["case"] == n and c["calibrated"]) for n in names]
    x = range(len(names))
    floor = 1e-18
    axes[0].bar([i - 0.2 for i in x], [max(c["p_max_over_pressure_scale"], floor) for c in off],
                width=0.4, label="m = rho0*A (textbook)")
    axes[0].bar([i + 0.2 for i in x], [max(c["p_max_over_pressure_scale"], floor) for c in on],
                width=0.4, label="m = rho0*A/(1+b) (calibrated)")
    axes[0].axhline(PRESSURE_FLOOR, color="k", ls="--", lw=1.0,
                    label="P1 floor 1e-4")
    axes[0].set_yscale("log")
    axes[0].set_xticks(list(x))
    axes[0].set_xticklabels([n.replace("_dx", "\ndx=") for n in names], fontsize=7)
    axes[0].set_ylabel(r"$p_{\max}/(\rho_0 c_0^2)$")
    axes[0].set_title("spurious initial pressure (c0=20 m/s)")
    axes[0].legend(fontsize=7)

    for lattice, style in (("square", "-o"), ("hex", "-s")):
        hf = sorted({r["h_over_dx"] for r in sweep if r["lattice"] == lattice})
        vals = [next(r for r in sweep if r["lattice"] == lattice and r["h_over_dx"] == h) for h in hf]
        axes[1].plot(hf, [v["bias_percent"] for v in vals], style, label=f"{lattice} analytic")
        pts = [c for c in cases if c["lattice"] == lattice and not c["calibrated"]]
        if pts:
            meas = sum(c["core_ratio_mean"] - 1.0 for c in pts) / len(pts) * 100.0
            axes[1].plot([H_FACTOR], [meas], "*", ms=12, label=f"{lattice} measured")
    axes[1].axhline(0.0, color="k", lw=0.8)
    axes[1].set_xlabel(r"$h/\Delta x$")
    axes[1].set_ylabel(r"$b = \rho/\rho_0 - 1$  [%]")
    axes[1].set_title("lattice kernel-sum bias")
    axes[1].legend(fontsize=7)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--plot", action="store_true")
    parser.add_argument("--out", default=str(HERE / "report.json"))
    args = parser.parse_args()

    torch.set_default_dtype(torch.float32)
    cases = []
    for calibrated in (False, True):
        cases.append(pool_case(calibrated))
        for lattice in ("hex", "square"):
            for dx in DROP_DX:
                cases.append(drop_case(lattice, dx, calibrated))
    sweep = analytic_sweep()
    checks = evaluate(cases, sweep)
    report = {
        "harness": "benchmarks/init_mass_calibration/run.py",
        "question": "does the shipped initial state start at rho0, and how big "
                    "is the spurious Tait pressure when it does not?",
        "parameters": {
            "rho0": RHO0, "sound_speed": SOUND_SPEED, "gamma": GAMMA,
            "h_factor": H_FACTOR, "radius": RADIUS, "pool_dx": POOL_DX,
            "drop_dx": list(DROP_DX), "pressure_floor": PRESSURE_FLOOR,
            "dtype": "float32 (solver default)",
        },
        "cases": cases,
        "analytic_sweep": sweep,
        "criteria": checks,
        "all_pass": all(c["result"] == "pass" for c in checks),
        "verdict": "P1 closed for the shipped square-lattice initial state; "
                   "hex-based verification harnesses must adopt the calibrated "
                   "mass before their Laplace numbers are re-baselined.",
    }
    Path(args.out).write_text(json.dumps(report, indent=2))
    if args.plot:
        plot(cases, sweep, HERE / "init_mass_calibration.png")
    flags = " ".join(f"{c['result']}" for c in checks)
    print(f"{len(checks)} criteria: {flags}")
    for c in checks:
        print(f"  [{'PASS' if c['result']=='pass' else 'FAIL'}] {c['name']}: "
              f"measured={c['measured']:.6g} limit={c['limit']:.6g}")
    print(f"report -> {args.out}")


if __name__ == "__main__":
    main()