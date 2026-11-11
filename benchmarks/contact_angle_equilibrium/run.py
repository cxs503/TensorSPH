#!/usr/bin/env python3
"""Contact angle: an *equilibrium* test, and the initial-condition defect that
made the shipped relaxation test insensitive to the set angle.

Why this benchmark exists
-------------------------
``benchmarks/contact_angle/diagnose_bc.py`` (``docs/verification_contact_angle_bc.md``)
relaxes a drop on a wall for ``1.5 t_sigma`` and reads 151-154 deg for
``theta_set = 60 / 90 / 120`` deg at ``R/dx = 13.3``.  That table was read as
"the BC does not deliver the angle".  It cannot support that reading, because
the three runs are **the same run**: ``benchmarks/contact_angle/run.py::make_drop_on_wall``
puts the drop centre at ``y = RADIUS`` and keeps every lattice particle within
``RADIUS`` of it, i.e. a **full disc tangent to the wall** (``theta = 180 deg`` by
construction) -- the argument ``theta`` only reaches ``Config.contact_angle`` and
never the geometry.  All three runs therefore start from the non-wetting limit
and differ only through the weak contact-line drive that follows; a reading
within 3 deg of itself is then expected whether the BC works or not.

This benchmark replaces the test with one that is sensitive to the set angle:

* **Geometry (analytic, area-preserving).** A circular cap of contact angle
  ``theta`` has radius ``a`` and centre height ``y_c = -a cos(theta)``; its area
  above the wall is ``A = a^2 (theta - sin(theta) cos(theta))``.  Holding the
  drop area at ``pi R^2`` fixes ``a = R sqrt(pi / (theta - sin(theta) cos(theta)))``,
  so the family spans ``theta = 180 deg`` (``a = R``, ``y_c = R``, i.e. the
  shipped initial condition, reproduced exactly) down to ``theta -> 0``.
* **Instrument (lattice cap).** The shipped whole-drop envelope instrument is
  run on the *undisturbed* cap lattice: it must return the angle that built the
  cap.  This is the device's discretisation floor at production ``R/dx``, and
  it is the number a physics reading has to beat.
* **Equilibrium (physics).** A cap built at ``theta_set`` is already in
  equilibrium, so with a correct BC its angle must *persist* over
  ``1.5 t_sigma``.  Run the same cap with the BC switched off (``contact_angle
  = None``) and the interface is free to find its own angle -- the pair is a
  positive control: it proves the BC changes the answer, and in which direction.

Acceptance (machine-checkable, identical to the shipped ``ANGLE_GATE_DEG``):
``|theta_measured - theta_set| <= 10 deg`` AND the run is admissible
(wall penetration < 0.5 dx, spurious current ``v_max / sqrt(sigma / (rho R))``
< 0.1, Adami, Ihmsen & Adams 2012).  ``theta = 180 deg`` is the degenerate end
of the family and is graded as a control, not as a wetting test.

Usage
-----
    PYTHONPATH=src python benchmarks/contact_angle_equilibrium/run.py \
        --dx 0.0015 --theta 60,120 --bc on,off --out benchmarks/contact_angle_equilibrium
"""
from __future__ import annotations

import argparse
import importlib.util
import json
import math
from pathlib import Path

import torch

from tensorsph.droplet_impact import cell_centres, rhs, stable_dt, wall_lattice
from tensorsph.neighbor_search import cell_linked_pairs

torch.set_num_threads(2)

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]


def _harness():
    """Import ``benchmarks/contact_angle/run.py`` as a module (it owns the
    instrument, the fluid properties and the gates)."""
    path = ROOT / "benchmarks" / "contact_angle" / "run.py"
    spec = importlib.util.spec_from_file_location("contact_angle_run", path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


CA = _harness()


# --------------------------------------------------------------------------- #
# geometry: an area-preserving circular cap at a prescribed contact angle
# --------------------------------------------------------------------------- #

def cap_radius(theta: float) -> float:
    """Cap radius ``a`` that holds the drop area at ``pi R^2`` for angle ``theta``."""
    return CA.RADIUS * math.sqrt(math.pi / (theta - math.sin(theta) * math.cos(theta)))


def make_cap_on_wall(theta: float, dx: float, device: torch.device):
    """Lattice drop on ``y = 0`` whose free surface meets the wall at ``theta``.

    ``theta = pi`` reproduces the shipped initial condition (a full disc of
    radius ``R`` centred at ``y = R``) particle for particle; that identity is
    what makes the old and the new physics layer comparable.
    """
    cfg = CA.make_drop_on_wall(theta, device)[0]
    cfg.dx = dx
    h = cfg.h_factor * cfg.dx
    x0 = CA.POOL_WIDTH / 2.0
    a = cap_radius(theta)
    y_c = -a * math.cos(theta)
    height = y_c + a + 2.0 * dx          # top of the cap, plus one cell of slack
    xs = cell_centres(CA.POOL_WIDTH, dx, device)
    ys = cell_centres(max(height, 2.0 * dx), dx, device)
    gx, gy = torch.meshgrid(xs, ys, indexing="xy")
    cand = torch.stack((gx.reshape(-1), gy.reshape(-1)), dim=-1)
    centre = torch.tensor([x0, y_c], device=device)
    keep = cand[:, 1] >= 0.0
    pos = cand[keep]
    pos = pos[(pos - centre).norm(dim=1) <= a]
    mass = torch.full((pos.shape[0],), CA.RHO0 * dx * dx, device=device)
    wall = _wall_lattice(cfg, dx, device)
    return cfg, h, pos, mass, wall, x0


def _wall_lattice(cfg, dx: float, device: torch.device):
    """Wall lattice at the requested ``dx`` (the harness builds at its own DX)."""
    saved = cfg.dx
    cfg.dx = dx
    wall = wall_lattice(cfg, device)
    cfg.dx = saved
    return wall


# --------------------------------------------------------------------------- #
# one relaxation
# --------------------------------------------------------------------------- #

def run_case(theta_deg: float, bc: str, dx: float,
             t_end_over_t_sigma: float = 1.5) -> dict:
    """Build the cap at ``theta_deg``, relax it, read the angle.

    ``bc = "on"`` keeps ``Config.contact_angle``; ``bc = "off"`` sets it to
    ``None`` (no wall BC: the interface finds its own angle).
    """
    theta = math.radians(theta_deg)
    cfg, h, pos, mass, wall, x0 = make_cap_on_wall(theta, dx, torch.device("cpu"))
    if bc == "off":
        cfg.contact_angle = None
    dt = float(stable_dt(cfg))
    t_end = t_end_over_t_sigma * CA.T_SIGMA
    steps = int(math.ceil(t_end / dt))
    dt = t_end / steps

    ic = CA.measure_contact_angle(pos[:, 0], pos[:, 1], x0, dx)
    vel = torch.zeros_like(pos)
    trace = {"t_over_t_sigma": [], "theta_deg": [], "vmax": []}
    for step in range(steps):
        pairs = cell_linked_pairs(pos, 2.0 * h)
        _, _, acc = rhs(pos, vel, mass, cfg, pairs=pairs, wall=wall, dt=dt)
        vel = vel + dt * acc
        pos = pos + dt * vel
        if step % 25 == 0 or step == steps - 1:
            m = CA.measure_contact_angle(pos[:, 0], pos[:, 1], x0, dx)
            trace["t_over_t_sigma"].append((step + 1) * dt / CA.T_SIGMA)
            trace["theta_deg"].append(m["theta_deg"])
            trace["vmax"].append(float(vel.norm(dim=1).max()))

    tail = [v for v in trace["theta_deg"][-max(1, int(0.4 * len(trace["theta_deg"]))):]
            if v == v]
    theta_final = sum(tail) / len(tail) if tail else float("nan")
    v_ref = math.sqrt(CA.SIGMA / (CA.RHO0 * CA.RADIUS))
    penetration_dx = max(0.0, -float(pos[:, 1].min())) / dx
    spurious = trace["vmax"][-1] / v_ref
    err = theta_final - theta_deg
    return {
        "theta_set_deg": theta_deg,
        "bc": bc,
        "dx_m": dx,
        "R_over_dx": CA.RADIUS / dx,
        "cap_radius_over_R": cap_radius(theta) / CA.RADIUS,
        "theta_ic_deg": ic["theta_deg"],
        "ic_model": ic.get("model"),
        "theta_final_deg": theta_final,
        "error_deg": err,
        "within_gate": bool(err == err and abs(err) <= CA.ANGLE_GATE_DEG),
        "wall_penetration_dx": penetration_dx,
        "spurious_current_ratio": spurious,
        "admissible": bool(penetration_dx < CA.WALL_PENETRATION_GATE_DX
                           and spurious < CA.SPURIOUS_CURRENT_GATE),
        "n_particles": int(pos.shape[0]),
        "steps": steps,
        "dt_s": dt,
        "t_end_over_t_sigma": t_end_over_t_sigma,
        "trace": trace,
    }


# --------------------------------------------------------------------------- #
# the shipped initial condition, for the record
# --------------------------------------------------------------------------- #

def shipped_ic_audit(dx: float) -> dict:
    """Particle count and measured angle of ``make_drop_on_wall`` vs theta.

    ``theta`` is an argument of ``make_drop_on_wall``; if the geometry were
    angle-dependent the count (or the reading) would move with it.
    """
    CA.DX = dx                     # build the lattice at the instrument spacing
    rows = []
    for theta_deg in (60.0, 90.0, 120.0):
        cfg, h, pos, mass, wall, x0 = CA.make_drop_on_wall(
            math.radians(theta_deg), torch.device("cpu"))
        m = CA.measure_contact_angle(pos[:, 0], pos[:, 1], x0, dx)
        rows.append({
            "theta_set_deg": theta_deg,
            "n_particles": int(pos.shape[0]),
            "theta_ic_deg": m["theta_deg"],
            "top_y_over_R": float(pos[:, 1].max()) / CA.RADIUS,
            "centre_y_over_R": (float(pos[:, 1].min() + pos[:, 1].max())
                                / 2.0 / CA.RADIUS),
        })
    counts = {r["n_particles"] for r in rows}
    angles = [r["theta_set_deg"] for r in rows]
    readings = [r["theta_ic_deg"] for r in rows]
    return {
        "dx_m": dx,
        "rows": rows,
        "n_particles_identical": len(counts) == 1,
        "reading_independent_of_theta": bool(
            max(readings) - min(readings) < 1.0),
        "theta_spread_deg": max(angles) - min(angles),
        "reading_spread_deg": max(readings) - min(readings),
    }


# --------------------------------------------------------------------------- #
# main
# --------------------------------------------------------------------------- #

def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dx", type=float, default=0.0015)
    ap.add_argument("--theta", default="60,120")
    ap.add_argument("--bc", default="on,off")
    ap.add_argument("--t-end", type=float, default=1.5)
    ap.add_argument("--out", default=str(HERE))
    args = ap.parse_args()

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    thetas = [float(s) for s in args.theta.split(",") if s]
    bcs = [s.strip() for s in args.bc.split(",") if s]

    audit = shipped_ic_audit(args.dx)
    print("shipped IC:", json.dumps({k: v for k, v in audit.items()
                                     if k != "rows"}))
    rows = []
    for theta_deg in thetas:
        for bc in bcs:
            row = run_case(theta_deg, bc, args.dx, args.t_end)
            rows.append(row)
            print(f"theta_set={theta_deg:6.1f} bc={bc:3s} "
                  f"ic={row['theta_ic_deg']:7.2f} final={row['theta_final_deg']:7.2f} "
                  f"err={row['error_deg']:+7.2f} n={row['n_particles']:4d} "
                  f"gate={row['within_gate']} adm={row['admissible']}")

    result = {
        "case": "contact_angle_equilibrium",
        "gate_deg": CA.ANGLE_GATE_DEG,
        "admissibility": {"penetration_gate_dx": CA.WALL_PENETRATION_GATE_DX,
                          "spurious_gate": CA.SPURIOUS_CURRENT_GATE},
        "t_sigma_s": CA.T_SIGMA,
        "t_end_over_t_sigma": args.t_end,
        "shipped_ic_audit": audit,
        "rows": rows,
    }
    (out / "results.json").write_text(json.dumps(result, indent=1))
    print("wrote", out / "results.json")


if __name__ == "__main__":
    main()