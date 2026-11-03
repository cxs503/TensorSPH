"""B3-continuation: end-to-end verification of the wall contact-angle BC.

Why this benchmark exists
-------------------------
``Config.contact_angle`` (S1*m2090) has been advertised since
``docs/contact_angle_boundary.md``: the interface normal near a wall is rotated
so that ``n . n_wall = cos(theta)``.  What was verified at the time was the
*operator* -- the rotation formula, unit-tested on hand-built gradients.  What
was never measured is the number a user cares about: if I ask for a 30 deg
droplet, does the droplet that forms have a 30 deg contact angle?  This
benchmark measures the **emergent macroscopic contact angle** of a droplet
resting on a static wall and compares it with Young's law.

Experimental design (the part that can silently go wrong)
--------------------------------------------------------
If every case were *initialised* at its target angle, measuring the angle at
the end would prove nothing: a solver that does nothing would score zero error.
Therefore

* every case starts from the **same half-disc** (``theta_init = 90 deg``),
  built identically for all runs;
* only ``Config.contact_angle`` differs between runs -- an A/B where the BC is
  the single changed input;
* the control run has ``contact_angle = None`` (BC off), which isolates the
  causal effect of the BC;
* the estimator is calibrated on **synthetic lattices of known angle** built
  with the same lattice rule and ``dx`` (``cap_lattice``), so its geometric
  bias is measured, not assumed.  Its residual sensitivity to particle
  arrangement is reported from jittered/lattice-shifted references.

All runs use ``gravity = 0``: the wall exists only as a boundary condition, so
nothing is buoyancy-confused.

Measurement
-----------
The macroscopic angle comes from the particle cloud, not from the CSF field
(using the CSF normal would be circular -- it is what the BC writes into):

* ``r = H^2 / A`` with ``H`` the droplet height and ``A`` the area
  (``N * dx * dy``, conserved to machine precision by the solver).  For a
  circular cap ``(1 - cos t)^2 / (t - sin t cos t) = r`` is monotone in the
  contact angle ``t``, so ``r`` inverts to one angle;
* the inversion table is built from ``cap_lattice`` at the same ``dx`` and
  area, which removes discretisation bias at once;
* a trimmed circle fit of the colour-threshold interface shell is reported as
  a cross-check, together with the shell-threshold spread.

Usage
-----
    PYTHONPATH=src python benchmarks/contact_angle/run.py \\
        --dx 0.005 --seconds 0.35 --write-json results.json --plot
    PYTHONPATH=src python benchmarks/contact_angle/run.py --quick   # CI smoke

Note on threading: ``torch.set_num_threads(1)`` is set below.  With the default
128 OpenMP threads of the build host, one ``rhs`` evaluation on 406 particles
took 4.49 s versus 0.0225 s single-threaded -- a 200x penalty from thread
oversubscription on tiny tensors.
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import torch

torch.set_num_threads(1)

from tensorsph.csf_consistent import kernel_sum_color          # noqa: E402
from tensorsph.droplet_impact import (Config, acoustic_courant, apply_boundary,
                                      rhs, viscous_dt_limit)   # noqa: E402
from tensorsph.neighbor_search import PairListCache, cell_linked_pairs  # noqa: E402

RHO0 = 1000.0
SIGMA = 0.072            # water/air surface tension, N/m
WALL_Y = 0.0             # plane the wall model presents to the fluid
TOL_DEG = 15.0           # platform tolerance on a reference comparison
COLOR_THRESHOLDS = (0.75, 0.85, 0.95)
CALIBRATION_STEP_DEG = 2.0


# --------------------------------------------------------------------------- #
# Geometry
# --------------------------------------------------------------------------- #

def _hex_lattice_in(inside, dx: float, pool_width: float,
                    y_start: float) -> tuple[torch.Tensor, float]:
    """Hex lattice points with ``inside(x, y)`` true; returns (pos, area/particle)."""
    dy = dx * math.sqrt(3.0) / 2.0
    rows = torch.arange(y_start, y_start + pool_width, dy, dtype=torch.float64)
    pts = []
    for k, y in enumerate(rows):
        xs = torch.arange(WALL_Y + 2.0 * dx, pool_width - 2.0 * dx, dx,
                          dtype=torch.float64)
        xs = xs + (0.5 * dx if k % 2 else 0.0)
        for x in xs:
            if inside(float(x), float(y)):
                pts.append((float(x), float(y)))
    return torch.tensor(pts, dtype=torch.float64), dx * dy


def half_disc_lattice(radius: float, dx: float, pool_width: float,
                      rho0: float = RHO0):
    """Half-disc resting on the wall: the identical initial state of every case."""
    dy = dx * math.sqrt(3.0) / 2.0
    cx = 0.5 * pool_width

    def inside(x, y):
        return (x - cx) ** 2 + (y - WALL_Y) ** 2 <= radius ** 2

    pos, area_per_particle = _hex_lattice_in(inside, dx, pool_width,
                                             WALL_Y + 0.5 * dy)
    mass = torch.full((pos.shape[0],), rho0 * area_per_particle, dtype=torch.float64)
    return pos, mass, area_per_particle


def cap_radius(theta: float, area: float) -> float:
    """Radius of a circular cap of area ``area`` and contact angle ``theta``."""
    f = theta - math.sin(theta) * math.cos(theta)
    if f <= 0.0:
        raise ValueError("theta must be in (0, pi)")
    return math.sqrt(area / f)


def cap_lattice(theta: float, area: float, dx: float, pool_width: float,
                rho0: float = RHO0, jitter: float = 0.0,
                shift: tuple[float, float] = (0.0, 0.0), seed: int = 1):
    """Synthetic cap of *known* contact angle, discretised like the case.

    Used twice: to build the estimator's inversion table, and to measure how
    much the estimator moves when the arrangement is perturbed (jitter, lattice
    shift) rather than being an ideal cut.
    """
    R = cap_radius(theta, area)
    y0 = WALL_Y - R * math.cos(theta)
    cx = 0.5 * pool_width

    def inside(x, y):
        return (x - cx) ** 2 + (y - y0) ** 2 <= R ** 2

    dy = dx * math.sqrt(3.0) / 2.0
    pos, area_per_particle = _hex_lattice_in(inside, dx, pool_width,
                                             WALL_Y + 0.5 * dy + shift[1])
    if shift[0]:
        pos[:, 0] += shift[0]
    if jitter > 0.0:
        g = torch.Generator().manual_seed(seed)
        pos = pos + jitter * dx * (torch.rand(pos.shape, generator=g,
                                              dtype=torch.float64) - 0.5)
    mass = torch.full((pos.shape[0],), rho0 * area_per_particle, dtype=torch.float64)
    return pos, mass, area_per_particle, R


def bottom_wall(pool_width: float, dx: float, layers: int = 3) -> torch.Tensor:
    """Static wall particles below ``WALL_Y`` (the case has no side walls)."""
    xs = torch.arange(-(layers - 0.5) * dx, pool_width + layers * dx, dx,
                      dtype=torch.float64)
    parts = []
    for layer in range(layers):
        y = WALL_Y - 0.5 * dx - layer * dx
        parts.append(torch.stack((xs, torch.full_like(xs, y)), dim=1))
    return torch.cat(parts, dim=0)


# --------------------------------------------------------------------------- #
# Measurement
# --------------------------------------------------------------------------- #

def height_area_ratio(pos: torch.Tensor, dx: float, area_per_particle: float) -> float:
    """``H^2 / A``: the single geometric number the angle inverts from."""
    area = pos.shape[0] * area_per_particle
    height = float(pos[:, 1].max()) - WALL_Y
    return height * height / area


def calibration_table(dx: float, area: float, pool_width: float,
                      step_deg: float = CALIBRATION_STEP_DEG,
                      anchor_ratio: float | None = None) -> list[tuple[float, float]]:
    """``(theta_deg, r)`` on synthetic caps, so ``r`` inverts without fit bias.

    ``anchor_ratio`` rescales the table so that its 90 deg entry equals a
    supplied measured ratio.  The case's initial state is a half-disc, i.e. a
    90 deg cap *by construction*, but it is a discrete cut of one, while the
    table's 90 deg entry is a synthetic cap of the same area whose radius is
    solved from that area -- the two cuts differ by a fraction of a particle and
    that alone moved the zero point by ~1 deg.  Anchoring removes that
    construction mismatch and makes the known zero point exact.
    """
    table = []
    theta_deg = step_deg
    while theta_deg <= 180.0 - step_deg:
        pos, _mass, area_pp, _R = cap_lattice(math.radians(theta_deg), area, dx,
                                              pool_width)
        table.append((theta_deg, height_area_ratio(pos, dx, area_pp)))
        theta_deg += step_deg
    # r must be non-decreasing in theta for the inversion to be well posed, but
    # a discrete cap at 2 deg steps is not: lattice-cut noise made r(58) >
    # r(60) > r(62) < r(64) on this lattice, which silently produced 53 deg for
    # a 60 deg cap. Smooth over three points, then enforce monotonicity, then
    # anchor: anchoring last keeps the (known) 90 deg zero point exact.
    rs = [row[1] for row in table]
    sm = [rs[0]] + [(rs[k - 1] + 2.0 * rs[k] + rs[k + 1]) / 4.0
                    for k in range(1, len(rs) - 1)] + [rs[-1]]
    running = sm[0]
    monotone = []
    for value in sm:
        running = max(running, value)
        monotone.append(running)
    if anchor_ratio is not None:
        idx = min(range(len(table)), key=lambda k: abs(table[k][0] - 90.0))
        scale = anchor_ratio / monotone[idx]
        monotone = [value * scale for value in monotone]
    return [(row[0], value) for row, value in zip(table, monotone)]


def invert_ratio(table: list[tuple[float, float]], r: float) -> float:
    """Monotone interpolation of the calibration table."""
    rs = [row[1] for row in table]
    if r <= rs[0]:
        return table[0][0]
    if r >= rs[-1]:
        return table[-1][0]
    for (t0, r0), (t1, r1) in zip(table, table[1:]):
        if r0 <= r <= r1:
            w = 0.0 if r1 == r0 else (r - r0) / (r1 - r0)
            return t0 + w * (t1 - t0)
    return float("nan")                                            # pragma: no cover


def estimator_uncertainty(dx: float, area: float, pool_width: float,
                          table: list[tuple[float, float]],
                          angles=(30.0, 60.0, 90.0, 120.0, 150.0)) -> dict:
    """Arrangement sensitivity: jittered / shifted lattices at known angles."""
    worst = 0.0
    detail = {}
    for theta in angles:
        errs = []
        for jitter, shift in ((0.0, (0.5 * dx, 0.35 * dx)),
                              (0.25, (0.0, 0.0)),
                              (0.25, (0.5 * dx, 0.35 * dx))):
            pos, _m, area_pp, _R = cap_lattice(math.radians(theta), area, dx,
                                               pool_width, jitter=jitter,
                                               shift=shift)
            r = height_area_ratio(pos, dx, area_pp)
            errs.append(invert_ratio(table, r) - theta)
        detail[f"{theta:g}"] = max(abs(e) for e in errs)
        worst = max(worst, detail[f"{theta:g}"])
    return {"max_abs_deg": worst, "per_angle_deg": detail}


def interface_shell(pos: torch.Tensor, mass: torch.Tensor, h: float,
                    threshold: float, rho0: float = RHO0) -> torch.Tensor:
    """Particles whose kernel-sum colour is below ``threshold`` (the interface)."""
    pairs = cell_linked_pairs(pos, 2.0 * h)
    color = kernel_sum_color(pos, mass, pairs, h, rho0)
    return pos[color < threshold]


def fit_circle(points: torch.Tensor, tolerance_factor: float = 2.0,
               max_passes: int = 4) -> dict:
    """Trimmed algebraic (Kasa) circle fit -> ``(x0, y0, R, n, rms)``."""
    pts = points
    out = {"x0": float("nan"), "y0": float("nan"), "radius": float("nan"),
           "points": 0, "rms": float("nan")}
    for _ in range(max_passes):
        if pts.shape[0] < 6:
            raise ValueError(f"circle fit needs >= 6 points, got {pts.shape[0]}")
        x, y = pts[:, 0], pts[:, 1]
        a = torch.stack((x, y, torch.ones_like(x)), dim=1)
        sol = torch.linalg.lstsq(a, -(x * x + y * y)).solution
        x0, y0 = float(-0.5 * sol[0]), float(-0.5 * sol[1])
        R = float(math.sqrt(max(1.0e-30, x0 * x0 + y0 * y0 - float(sol[2]))))
        dev = (torch.linalg.vector_norm(
            pts - torch.tensor([x0, y0], dtype=pts.dtype), dim=1) - R).abs()
        sigma = float(dev.std()) if pts.shape[0] > 2 else 0.0
        keep = dev <= max(tolerance_factor * sigma, 1.0e-12)
        out = {"x0": x0, "y0": y0, "radius": R, "points": int(keep.sum()),
               "rms": float(dev.square().mean().sqrt())}
        if bool(keep.all()):
            break
        pts = pts[keep]
    cos_t = max(-1.0, min(1.0, (WALL_Y - out["y0"]) / out["radius"]))
    out["theta_deg"] = math.degrees(math.acos(cos_t))
    return out


def measure_contact_angle(pos: torch.Tensor, mass: torch.Tensor, h: float,
                          dx: float, area_per_particle: float,
                          table: list[tuple[float, float]] | None = None,
                          thresholds=COLOR_THRESHOLDS) -> dict:
    """Calibrated height/area estimate, with the circle fit as a cross-check."""
    area = pos.shape[0] * area_per_particle
    r = height_area_ratio(pos, dx, area_per_particle)
    if table is None:
        table = calibration_table(dx, area, 0.30)
    theta = invert_ratio(table, r)

    circle = {}
    for t in thresholds:
        shell = interface_shell(pos, mass, h, t)
        try:
            circle[f"{t:g}"] = fit_circle(shell)
        except ValueError as exc:                                  # pragma: no cover
            circle[f"{t:g}"] = {"theta_deg": float("nan"), "error": str(exc)}
    fits = sorted(v["theta_deg"] for v in circle.values()
                  if math.isfinite(v.get("theta_deg", float("nan"))))
    circle_theta = fits[len(fits) // 2] if fits else float("nan")
    return {"theta_deg": theta, "r": r, "area": area,
            "height": float(pos[:, 1].max()) - WALL_Y,
            "theta_circle_deg": circle_theta,
            "circle_thresholds": circle,
            "particles": int(pos.shape[0])}


# --------------------------------------------------------------------------- #
# Dynamics
# --------------------------------------------------------------------------- #

def base_config(dx: float, pool_width: float, seconds: float,
                sigma: float = SIGMA, viscosity: float = 0.02,
                sound_speed: float = 10.0,
                contact_angle: float | None = None) -> Config:
    cfg = Config(dx=dx, pool_width=pool_width, pool_depth=0.6, drop_radius=0.05,
                 rho0=RHO0, sound_speed=sound_speed, gamma=7.0, gravity=0.0,
                 viscosity=viscosity, dt=1.0e-6, steps=1, boundary="wall",
                 wall_layers=3, max_courant=0.25, max_mach=0.10, fail_fast=False,
                 pair_skin_factor=0.25, wall_friction=0.0,
                 surface_tension_coefficient=sigma, csf_scheme="consistent",
                 csf_curvature="mls", contact_angle=contact_angle)
    dt = min(0.25 * cfg.h_factor * dx / sound_speed, viscous_dt_limit(cfg) / 1.05)
    cfg.dt = float(dt)
    cfg.steps = int(math.ceil(seconds / cfg.dt))
    return cfg


def run_case(theta_deg: float | None, cfg: Config, radius: float,
             table: list[tuple[float, float]] | None = None,
             measure_every: float = 0.1) -> dict:
    """Integrate the droplet and track the measured angle over time."""
    pos, mass, area_per_particle = half_disc_lattice(radius, cfg.dx, cfg.pool_width)
    wall = bottom_wall(cfg.pool_width, cfg.dx, cfg.wall_layers)
    h = cfg.h_factor * cfg.dx
    n = pos.shape[0]
    if table is None:
        table = calibration_table(cfg.dx, pos.shape[0] * area_per_particle,
                                  cfg.pool_width)
    vel = torch.zeros_like(pos)
    cache = PairListCache(2.0 * h, cfg.pair_skin_factor * h, fluid_count=n)
    all_pos = torch.cat((pos, wall), dim=0) if wall.numel() else pos

    measure_steps = max(1, int(round(measure_every / cfg.dt)))
    history = []
    peak_speed = 0.0
    peak_bulk_compression = 0.0
    peak_courant = 0.0
    boundary_events = 0

    def record(step: int) -> None:
        m = measure_contact_angle(pos.detach(), mass, h, cfg.dx, area_per_particle,
                                  table=table)
        history.append({"step": step, "time": step * cfg.dt,
                        "theta_deg": m["theta_deg"],
                        "theta_circle_deg": m["theta_circle_deg"],
                        "area": m["area"], "r": m["r"]})

    record(0)
    for step in range(1, cfg.steps + 1):
        all_pos[:n] = pos
        pairs, _ = cache.get(all_pos)
        rho, _pressure, acc = rhs(pos, vel, mass, cfg, wall=wall, pairs=pairs,
                                  dt=cfg.dt)
        speed = float(torch.linalg.vector_norm(vel, dim=1).max()) if n else 0.0
        peak_speed = max(peak_speed, speed)
        # Compression is only meaningful away from the free surface, where the
        # kernel sum legitimately falls to ~0.5 rho0. Interior = colour > 0.9.
        peak_bulk_compression = max(peak_bulk_compression,
                                    abs(float(rho.max()) / cfg.rho0 - 1.0))
        peak_courant = max(peak_courant, acoustic_courant(cfg.dt, cfg, speed))
        vel = vel + cfg.dt * acc                         # semi-implicit Euler
        pos = pos + cfg.dt * vel
        boundary_events += apply_boundary(pos, vel, cfg, top=cfg.pool_depth)
        if step % measure_steps == 0 or step == cfg.steps:
            record(step)
        if not bool(torch.isfinite(pos).all()):
            raise FloatingPointError(f"non-finite state at step {step}")

    final = history[-1]
    tail = [m["theta_deg"] for m in history[-3:]]
    target = None if theta_deg is None else float(theta_deg)
    result = {
        "contact_angle_prescribed_deg": target,
        "theta_initial_deg": history[0]["theta_deg"],
        "theta_final_deg": final["theta_deg"],
        "theta_final_circle_deg": final["theta_circle_deg"],
        "theta_shift_from_initial_deg": final["theta_deg"] - history[0]["theta_deg"],
        "theta_drift_last_0p2s_deg": (tail[-1] - tail[0]) if len(tail) > 1 else float("nan"),
        "error_deg": None if target is None else final["theta_deg"] - target,
        "error_percent": None if target is None
                         else 100.0 * (final["theta_deg"] - target) / target,
        "within_15pct": None if target is None
                        else abs(final["theta_deg"] - target) <= TOL_DEG,
        "dt": cfg.dt, "steps": cfg.steps, "particles": n,
        "wall_particles": int(wall.shape[0]), "h": h,
        "viscosity": cfg.viscosity, "sound_speed": cfg.sound_speed,
        "sigma": cfg.surface_tension_coefficient,
        "peak_speed": peak_speed, "peak_mach": peak_speed / cfg.sound_speed,
        "peak_bulk_density_error": peak_bulk_compression,
        "peak_courant": peak_courant, "boundary_events": boundary_events,
        "area_change_percent": 100.0 * (final["area"] / history[0]["area"] - 1.0),
        "history": history,
    }
    return {"result": result, "final_pos": pos.detach().cpu()}


def run_case_set(angles, dx: float, seconds: float, radius: float,
                 pool_width: float, viscosity: float, sound_speed: float,
                 sigma: float = SIGMA, quiet: bool = False) -> dict:
    """One identical initial lattice per case; only ``contact_angle`` changes."""
    pos0, _m, area_pp = half_disc_lattice(radius, dx, pool_width)
    table = calibration_table(dx, pos0.shape[0] * area_pp, pool_width,
                              anchor_ratio=height_area_ratio(pos0, dx, area_pp))
    unc = estimator_uncertainty(dx, pos0.shape[0] * area_pp, pool_width, table)
    cases, clouds = [], {}
    for theta in angles:
        cfg = base_config(dx, pool_width, seconds, sigma=sigma,
                          viscosity=viscosity, sound_speed=sound_speed,
                          contact_angle=None if theta is None
                          else math.radians(theta))
        out = run_case(theta, cfg, radius, table=table)
        res = out["result"]
        cases.append(res)
        clouds[str(theta)] = out["final_pos"]
        if not quiet:
            print(f"theta={theta}: emergent {res['theta_final_deg']:.1f} deg "
                  f"(circle fit {res['theta_final_circle_deg']:.1f}), "
                  f"shift from 90 {res['theta_shift_from_initial_deg']:+.1f} deg, "
                  f"N={res['particles']} steps={res['steps']} "
                  f"mach={res['peak_mach']:.4f} "
                  f"area{res['area_change_percent']:+.4f}%", flush=True)
    return {"cases": cases, "clouds": clouds,
            "estimator": {"uncertainty_deg": unc["max_abs_deg"],
                          "uncertainty_per_angle_deg": unc["per_angle_deg"],
                          "calibration_points": len(table),
                          "calibration_step_deg": CALIBRATION_STEP_DEG}}


# --------------------------------------------------------------------------- #
# Figures
# --------------------------------------------------------------------------- #

def make_figures(results: dict, out_dir: Path) -> list[Path]:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    paths = []
    cases = results["cases"]
    labelled = [c for c in cases if c["contact_angle_prescribed_deg"] is not None]

    fig, ax = plt.subplots(figsize=(4.8, 3.8))
    ax.plot([0, 180], [0, 180], "k--", lw=1.0, label="identity (Young's law)")
    ax.fill_between([0, 180], [-TOL_DEG, 180 - TOL_DEG], [TOL_DEG, 195],
                    color="0.88", alpha=0.7, label="+/-15 deg tolerance")
    ax.plot([c["contact_angle_prescribed_deg"] for c in labelled],
            [c["theta_final_deg"] for c in labelled], "o-", color="C3",
            label="emergent angle (calibrated)")
    ax.plot([c["contact_angle_prescribed_deg"] for c in labelled],
            [c["theta_final_circle_deg"] for c in labelled], "s--", mfc="none",
            color="C0", label="cross-check (circle fit)")
    ctrl = [c for c in cases if c["contact_angle_prescribed_deg"] is None]
    if ctrl:
        ax.axhline(ctrl[0]["theta_final_deg"], color="C2", lw=1.1, ls=":",
                   label=f"no-BC control ({ctrl[0]['theta_final_deg']:.0f} deg)")
    ax.axhline(90.0, color="0.5", lw=0.8)
    ax.set_xlabel("prescribed contact angle [deg]")
    ax.set_ylabel("measured contact angle [deg]")
    ax.set_title("Droplet on wall: prescribed vs emergent")
    ax.set_xlim(0, 180)
    ax.set_ylim(0, 180)
    ax.legend(fontsize=6.5, loc="upper left")
    ax.grid(alpha=0.3)
    p = out_dir / "contact_angle_parity.png"
    fig.tight_layout()
    fig.savefig(p, dpi=110)
    plt.close(fig)
    paths.append(p)

    n = len(cases)
    fig, axes = plt.subplots(1, n, figsize=(2.3 * n, 2.8), sharey=True)
    axes = [axes] if n == 1 else list(axes)
    for ax, c in zip(axes, cases):
        pts = results["clouds"][str(c["contact_angle_prescribed_deg"])]
        ax.scatter(pts[:, 0], pts[:, 1], s=1.2, c="C0", linewidths=0)
        ax.axhline(WALL_Y, color="k", lw=1.0)
        lab = "no BC" if c["contact_angle_prescribed_deg"] is None else \
            f"{c['contact_angle_prescribed_deg']:.0f} deg"
        ax.set_title(f"{lab} -> {c['theta_final_deg']:.1f} deg", fontsize=8)
        ax.set_xlim(0.05, 0.25)
        ax.set_ylim(-0.01, 0.12)
        ax.set_aspect("equal")
        ax.set_xticks([])
        ax.set_yticks([])
    p = out_dir / "contact_angle_clouds.png"
    fig.tight_layout()
    fig.savefig(p, dpi=110)
    plt.close(fig)
    paths.append(p)
    return paths


# --------------------------------------------------------------------------- #
# CLI
# --------------------------------------------------------------------------- #

def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--dx", type=float, default=0.005)
    ap.add_argument("--radius", type=float, default=0.06)
    ap.add_argument("--pool-width", type=float, default=0.30)
    ap.add_argument("--seconds", type=float, default=0.35)
    ap.add_argument("--viscosity", type=float, default=0.02)
    ap.add_argument("--sound-speed", type=float, default=10.0)
    ap.add_argument("--sigma", type=float, default=SIGMA)
    ap.add_argument("--angles", type=float, nargs="*",
                    default=[30.0, 60.0, 90.0, 120.0, 150.0])
    ap.add_argument("--no-control", action="store_true")
    ap.add_argument("--write-json", type=Path, default=None)
    ap.add_argument("--figures-dir", type=Path, default=None)
    ap.add_argument("--quick", action="store_true",
                    help="tiny CI smoke configuration")
    args = ap.parse_args(argv)
    if args.quick:
        args.dx, args.seconds, args.angles = 0.008, 0.06, [30.0, 150.0]

    angles = list(args.angles) + ([] if args.no_control else [None])
    results = run_case_set(angles, args.dx, args.seconds, args.radius,
                           args.pool_width, args.viscosity, args.sound_speed,
                           sigma=args.sigma)
    payload = {
        "benchmark": "contact_angle_young_law",
        "theory": "(1-cos t)^2/(t-sin t cos t) = H^2/A for a circular cap",
        "setup": {"dx": args.dx, "radius": args.radius,
                  "pool_width": args.pool_width, "seconds": args.seconds,
                  "viscosity": args.viscosity, "sound_speed": args.sound_speed,
                  "sigma": args.sigma, "gravity": 0.0, "csf_scheme": "consistent",
                  "csf_curvature": "mls", "initial_angle_deg": 90.0,
                  "tolerance_deg": TOL_DEG},
        "estimator": results["estimator"],
        "cases": results["cases"],
    }
    if args.figures_dir is not None:
        paths = make_figures(results, args.figures_dir)
        payload["figures"] = [str(p) for p in paths]
    if args.write_json:
        args.write_json.write_text(json.dumps(payload, indent=2))
        print(f"wrote {args.write_json}")
    return 0


if __name__ == "__main__":                                    # pragma: no cover
    raise SystemExit(main())