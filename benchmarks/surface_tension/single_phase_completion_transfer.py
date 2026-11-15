"""Scope of the single-phase interface-completion closure: transfer and calibration.

Anchor: ``XVAL-SPH-MIXTURE-MATERIAL-AND-SURFACE-TENSION``.

What round 2 established, and what it left open
-----------------------------------------------
``benchmarks/surface_tension/single_phase_completion.py`` closed the round-2
open item on the **single-phase (vacuum-reduced)** CSS model: the one-sided
stress flux reads the Young-Laplace jump ``Delta p = sigma/R`` 30 % short, and a
specular-mirror completion of the missing half support, weighted by one closure
coefficient ``alpha``, removes the deficit.  ``alpha`` was fitted at a single
configuration (square lattice, ``R/dx = 32``, ``h/dx = 1.3``) and applied
unchanged to other resolutions and smoothing widths -- a *within-configuration*
transfer test.

That leaves the question this benchmark answers, because it decides how the
closure may be described in the capability ledger:

    **Is ``alpha* = 0.4263`` a physical closure constant, or a discretisation
    calibration of one lattice?**

The distinction matters.  A physical constant must reproduce the same physics on
any reasonable particle arrangement; a calibration may not be carried to another
lattice (or another functional) and still be called a fix.  Nothing below is
fitted after the fact: the closure weight is the round-2 value, taken as-is.

Predeclared acceptance (fixed before the run)
---------------------------------------------
``TRANSFER``  ``|err(alpha*)| < 15 %`` (the ledger gate) in **every**
              configuration that the round-2 fit never saw: a hexagonal lattice
              (different neighbour geometry) and square lattices offset by a
              fraction of ``dx`` (interface not centred on a lattice site).
``CALIBRATION``  the scan's zero-crossing weight ``alpha_null`` is *not*
              transportable: ``|alpha_null(hex) - alpha_null(square)| > 0.10 *
              alpha_null(square)``.  If it holds, ``alpha*`` is a discretisation
              calibration and must be reported as such.
``MOMENTUM``   the completed flux stays momentum conservative (residual
              ``< 1e-12`` relative) on every lattice -- a structural property
              that must not depend on the arrangement.
Final verdict is reported whatever the outcome (honest-FAIL if a threshold is
missed); no threshold is re-tuned after inspecting the numbers.

Two further readings, both explicitly *diagnostic* rather than gated:

* ``mode_stiffness`` -- the projection of the same force field on the ``m = 2``
  Rayleigh mode of a perturbed drop, ``A_ddot/A`` vs ``-omega_2^2``.  This is a
  different *functional* of the same operator (it weights the interface shell by
  ``cos 2theta`` instead of the radial flux integral), evaluated on one
  configuration without the pressure feedback, so it bounds the
  functional-dependence of ``alpha``; it is not a dynamical validation.
* ``dynamic_probe`` -- a short (0.35 Rayleigh period) time integration of the
  perturbed drop with the CSS single-phase force in the loop of
  :func:`tensorsph.droplet_impact.rhs`, next to the production CSF force as a
  control.  It reports containment and the particle-scale force noise, i.e.
  whether the *static* repair is usable in a time loop at these resolutions.  A
  short window is used on purpose: containment is the reading, so letting the
  run diverge for many periods would measure nothing.

Everything below is written to ``out_single_phase_completion_transfer/gate.json``
and read back by ``docs/xval_sphinxys_mixture_laplace/single_phase_completion_transfer.md``
-- no hand-copied number.

Usage::

    PYTHONPATH=src python benchmarks/surface_tension/single_phase_completion_transfer.py
"""
from __future__ import annotations

import argparse
import json
import math
import sys
import time
from pathlib import Path

import torch

import tensorsph.sphinxys_css as sc
from tensorsph.neighbor_search import cell_linked_pairs

HERE = Path(__file__).resolve().parent
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))

import single_phase_completion as SPC  # noqa: E402  (round-2 closure benchmark)

SIGMA = SPC.SIGMA                 # 0.072 N/m
RADIUS = SPC.RADIUS               # 0.1 m
RHO_WATER = SPC.RHO_WATER
H_OVER_DX = SPC.H_OVER_DX
GATE = SPC.GATE                   # 0.15
ALPHA_STAR = 0.4263334171275849   # round-2 closure coefficient, used as-is
LATTICES = ("square", "hex")
ALPHA_SCAN = (0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6)
OFFSETS_DX = ((0.0, 0.0), (0.5, 0.0), (0.3, 0.37))
MOMENTUM_TOL = 1.0e-12
OUT = HERE / "out_single_phase_completion_transfer"


# --------------------------------------------------------------------------- #
# particle arrangements
# --------------------------------------------------------------------------- #
def lattice_points(kind: str, dx: float, radius: float, offset_dx=(0.0, 0.0)
                   ) -> tuple[torch.Tensor, float]:
    """Lattice sites of the body ``r <= radius``; returns points and the cell area.

    ``kind="square"``  sites on a ``dx`` grid; cell area ``dx**2``.
    ``kind="hex"``     triangular lattice, rows ``dx*sqrt(3)/2`` apart with every
                       second row shifted by ``dx/2``; cell area ``dx*dx*sqrt(3)/2``.

    ``offset_dx`` shifts the *lattice* in units of ``dx`` while the body stays
    the disc of radius ``radius`` about the origin, so the interface no longer
    passes through lattice sites in the same way -- the round-2 configuration is
    ``offset_dx = (0, 0)``.
    """
    if kind == "hex":
        dy = dx * math.sqrt(3.0) / 2.0
        cell = dx * dy
    else:
        dy = dx
        cell = dx * dx
    span = radius + 2.0 * dx
    xs = torch.arange(-span, span + 0.5 * dx, dx, dtype=torch.float64)
    ys = torch.arange(-span, span + 0.5 * dy, dy, dtype=torch.float64)
    gx, gy = torch.meshgrid(xs, ys, indexing="xy")
    gx = gx.clone()
    if kind == "hex":
        gx[1::2, :] += 0.5 * dx
    pts = torch.stack((gx.reshape(-1), gy.reshape(-1)), dim=1)
    pts = pts + torch.tensor([offset_dx[0] * dx, offset_dx[1] * dy],
                             dtype=torch.float64)
    keep = torch.linalg.norm(pts, dim=1) <= radius + 1.0e-12
    return pts[keep].clone(), cell


def body(kind: str, radius: float, dx: float, offset_dx=(0.0, 0.0)):
    """Positions, radii, volumes and masses of a lattice disc of fluid."""
    pos, cell = lattice_points(kind, dx, radius, offset_dx)
    r = torch.linalg.norm(pos, dim=1)
    vol = torch.full((pos.shape[0],), cell, dtype=torch.float64)
    rho0 = torch.full((pos.shape[0],), RHO_WATER, dtype=torch.float64)
    return pos, r, rho0, vol, rho0 * vol


# --------------------------------------------------------------------------- #
# readings
# --------------------------------------------------------------------------- #
def laplace(radius: float, dx: float, alpha: float, kind: str = "square",
            offset_dx=(0.0, 0.0), h_over_dx: float = H_OVER_DX) -> dict:
    """Round-2 reading: inward force density integrated over the body -> sigma/R."""
    h = h_over_dx * dx
    pos, r, rho0, vol, mass = body(kind, radius, dx, offset_dx)
    pairs = cell_linked_pairs(pos, 2.0 * h)
    acc, fields = sc.surface_tension_acceleration(
        pos, mass, rho0, vol, pairs, None, SIGMA, h, frac=torch.ones(pos.shape[0]),
        mode="single_phase", delta="sphinxys", hourglass=False,
        assembly="consistent", completion="mirror", completion_weight=alpha)
    n_out = pos / r.clamp_min(1e-300)[:, None]
    f_inward = -rho0 * (acc * n_out).sum(dim=1) * vol
    area = 2.0 * math.pi * radius
    dp = float(f_inward.sum()) / area
    scale = float((rho0 * vol * torch.linalg.vector_norm(acc, dim=1)).sum())
    mom = (rho0 * vol)[:, None].mul(acc).sum(dim=0)
    theory = SIGMA / radius
    return {
        "lattice": kind, "offset_dx": list(offset_dx),
        "radius_over_dx": radius / dx, "dx_m": dx, "particles": int(pos.shape[0]),
        "cell_area_m2": float(vol[0]),
        "completion_weight": alpha,
        "dp_pa": dp, "dp_theory_pa": theory,
        "err_pct": (dp - theory) / theory * 100.0,
        "momentum_residual": float(torch.linalg.vector_norm(mom)) / max(scale, 1e-300),
        "a_peak_over_physical": float(torch.linalg.vector_norm(acc, dim=1).max())
        / (SIGMA / (RHO_WATER * radius)),
        "momentum_net_over_sum": float(torch.linalg.vector_norm(
            ((rho0 * vol)[:, None] * acc).sum(dim=0)))
        / max(float(torch.linalg.vector_norm((rho0 * vol)[:, None] * acc, dim=1).sum()),
              1e-300),
    }


def null_weight(scan: list[dict]) -> dict:
    """Zero-crossing weight of the Laplace error, by linear interpolation.

    The error is monotone in ``alpha`` on both lattices (it is a linear closure
    weight multiplying one flux), so the crossing is well defined; the bracket is
    reported with the value so the interpolation error is visible.
    """
    pts = [(rec["completion_weight"], rec["err_pct"]) for rec in scan]
    for (a0, e0), (a1, e1) in zip(pts, pts[1:]):
        if e0 == 0.0:
            return {"alpha_null": a0, "bracket": [a0, a0], "method": "exact node"}
        if e0 * e1 < 0.0:
            return {"alpha_null": a0 + (a1 - a0) * abs(e0) / (abs(e0) + abs(e1)),
                    "bracket": [a0, a1], "method": "linear interpolation"}
    return {"alpha_null": float("nan"), "bracket": [pts[0][0], pts[-1][0]],
            "method": "no crossing in scan"}


def mode_stiffness(radius: float, dx: float, alpha: float, kind: str = "square",
                   eps: float = 0.05, mode: int = 2) -> dict:
    """Indicator: projection of the same force on the m-mode radial acceleration.

    A perturbed drop ``r = R (1 + eps cos m theta)`` at rest has, for an inviscid
    Rayleigh mode, ``a_r = -omega_m^2 A r cos m theta`` with
    ``A = sum m r cos(m theta) / sum m r``; so the normalised projection
    ``A_ddot / A`` should equal ``-omega_m^2``.  Only the surface-tension
    acceleration is projected, so the pressure feedback of the real mode is
    absent -- this is a *functional* indicator of the operator, not a dynamical
    validation (see the module docstring).
    """
    from tensorsph.capillary_oscillation import rayleigh_2d
    h = H_OVER_DX * dx
    span = radius + 2.0 * dx
    if kind == "hex":
        dy = dx * math.sqrt(3.0) / 2.0
        cell = dx * dy
    else:
        dy = dx
        cell = dx * dx
    xs = torch.arange(-span, span + 0.5 * dx, dx, dtype=torch.float64)
    ys = torch.arange(-span, span + 0.5 * dy, dy, dtype=torch.float64)
    gx, gy = torch.meshgrid(xs, ys, indexing="xy")
    gx = gx.clone()
    if kind == "hex":
        gx[1::2, :] += 0.5 * dx
    pts = torch.stack((gx.reshape(-1), gy.reshape(-1)), dim=1)
    r0 = torch.linalg.norm(pts, dim=1)
    th0 = torch.atan2(pts[:, 1], pts[:, 0])
    keep = r0 <= radius * (1.0 + eps * torch.cos(mode * th0))
    pos = pts[keep].clone()
    vol = torch.full((pos.shape[0],), cell, dtype=torch.float64)
    rho0 = torch.full((pos.shape[0],), RHO_WATER, dtype=torch.float64)
    mass = rho0 * vol
    pairs = cell_linked_pairs(pos, 2.0 * h)
    acc, _ = sc.surface_tension_acceleration(
        pos, mass, rho0, vol, pairs, None, SIGMA, h, frac=torch.ones(pos.shape[0]),
        mode="single_phase", delta="sphinxys", hourglass=False,
        assembly="consistent", completion="mirror", completion_weight=alpha)
    r = torch.linalg.norm(pos, dim=1)
    th = torch.atan2(pos[:, 1], pos[:, 0])
    c = torch.cos(mode * th)
    a_r = (pos * acc).sum(dim=1) / r.clamp_min(1e-300)
    A = float((mass * r * c).sum() / (mass * r).sum())
    A_ddot = float((mass * a_r * c).sum() / (mass * r).sum())
    omega2 = rayleigh_2d(SIGMA, RHO_WATER, radius, mode) ** 2
    return {"lattice": kind, "radius_over_dx": radius / dx, "particles": int(pos.shape[0]),
            "mode": mode, "eps": eps, "completion_weight": alpha,
            "A": A, "A_ddot": A_ddot, "omega2_measured": -A_ddot / A, "omega2_exact": omega2,
            "omega_err_pct": (math.sqrt(max(-A_ddot / A, 1e-300)) / math.sqrt(omega2) - 1.0) * 100.0}


def dynamic_probe(radius: float, r_over_dx: int, alpha: float, model: str = "css",
                  periods: float = 0.35, sample: int = 20) -> dict:
    """Short time integration: does the repaired force survive in a time loop?

    ``model="css"``  WCSPH hydrodynamics (:func:`tensorsph.droplet_impact.rhs`)
                     with ``sigma = 0``, plus the CSS single-phase acceleration.
    ``model="csf"``  the production CSF force, i.e. the control: identical
                     harness, identical readings, different surface-tension model.
    """
    from tensorsph.capillary_oscillation import (interior_pressure, make_droplet,
                                                mode_amplitude, rayleigh_2d)
    from tensorsph.droplet_impact import Config, rhs, suggested_dt
    from tensorsph.neighbor_search import PairListCache

    dx = radius / r_over_dx
    cfg = Config(dx=dx, h_factor=H_OVER_DX, rho0=RHO_WATER, sound_speed=0.5, gamma=7.0,
                 gravity=0.0, viscosity=1.0e-6, pool_width=2.4 * radius,
                 pool_depth=2.4 * radius,
                 surface_tension_coefficient=(SIGMA if model == "csf" else 0.0),
                 device="cpu", boundary="clamp", adaptive_dt=True, max_courant=0.25,
                 density_calibration=True)
    cfg.dt = suggested_dt(cfg)
    pos, mass = make_droplet(dx, 0.05, torch.device("cpu"), radius=radius, mode=2,
                             rho0=RHO_WATER)
    vel = torch.zeros_like(pos)
    h = cfg.h_factor * dx
    cache = PairListCache(2.0 * h, cfg.pair_skin_factor * h, fluid_count=pos.shape[0])
    t_exact = 2.0 * math.pi / rayleigh_2d(SIGMA, RHO_WATER, radius, 2)
    t, step, max_r = 0.0, 0, 0.0
    ts, qs = [], []
    while t < periods * t_exact:
        pairs, _ = cache.get(pos)
        rho, p, acc = rhs(pos, vel, mass, cfg, pairs=pairs, wall=None, dt=cfg.dt)
        if model == "css":
            a_css, _ = sc.surface_tension_acceleration(
                pos, mass, rho, mass / rho, pairs, None, SIGMA, h,
                frac=torch.ones(pos.shape[0]), mode="single_phase", delta="sphinxys",
                hourglass=False, assembly="consistent", completion="mirror",
                completion_weight=alpha)
            acc = acc + a_css
        vel = vel + cfg.dt * acc
        pos = pos + cfg.dt * vel
        t += cfg.dt
        step += 1
        cfg.dt = suggested_dt(cfg)
        max_r = max(max_r, float(torch.linalg.norm(pos, dim=1).max()) / radius)
        if step % sample == 0:
            ts.append(t)
            qs.append(mode_amplitude(pos, mass, 2))
    return {"model": model, "completion_weight": alpha if model == "css" else None,
            "radius_over_dx": r_over_dx, "particles": int(pos.shape[0]), "steps": step,
            "t_end_over_period": periods,
            "containment_max_r_over_R": max_r,
            "contained": bool(max_r <= 1.6),
            "q_start": qs[0], "q_end": qs[-1],
            "q_half_range": 0.5 * (max(qs) - min(qs))}


# --------------------------------------------------------------------------- #
# driver
# --------------------------------------------------------------------------- #
def run(out: Path, rungs=(16, 24, 32), probe_rung: int = 16,
        threads: int = 2) -> dict:
    # the scenes are small (a few thousand particles): more threads than this only
    # adds synchronisation overhead, and the readings must not depend on the
    # thread count, so it is pinned here instead of being inherited.
    torch.set_default_dtype(torch.float64)
    torch.set_num_threads(threads)
    out.mkdir(parents=True, exist_ok=True)
    t0 = time.time()
    payload: dict = {
        "case": {"radius_m": RADIUS, "sigma_nm": SIGMA, "rho_water": RHO_WATER,
                 "h_over_dx": H_OVER_DX, "kernel": "wendland_c2", "dimension": 2,
                 "reference": "Delta p = sigma / R (exact)"},
        "gate_pct": GATE * 100.0,
        "closure_weight_round2": ALPHA_STAR,
        "predeclared": {
            "TRANSFER": f"|err(alpha*)| < {GATE * 100:.0f} % on every configuration "
                        "not used by the round-2 fit (hex lattice, offset lattices)",
            "CALIBRATION": "|alpha_null(hex) - alpha_null(square)| > 0.10 * "
                           "alpha_null(square) => the weight is a discretisation calibration",
            "MOMENTUM": f"relative momentum residual < {MOMENTUM_TOL:g} on every lattice",
        },
    }

    # S1: the round-2 configuration and its un-fitted relatives, alpha* untouched
    configs = []
    for kind in LATTICES:
        offsets = OFFSETS_DX if kind == "square" else ((0.0, 0.0),)
        for off in offsets:
            for r in rungs:
                for alpha in (0.0, ALPHA_STAR):
                    configs.append(laplace(RADIUS, RADIUS / r, alpha, kind, off))
    payload["configurations"] = configs

    # S2: closure scan per lattice -> the weight that nulls the reading
    scan = {}
    for kind in LATTICES:
        scan[kind] = [laplace(RADIUS, RADIUS / 32.0, a, kind) for a in ALPHA_SCAN]
    payload["alpha_scan"] = scan
    payload["alpha_null"] = {k: null_weight(v) for k, v in scan.items()}

    # S3: functional indicator (mode-2 projection of the same force)
    payload["mode_stiffness"] = [
        dict(mode_stiffness(RADIUS, RADIUS / r, a, kind), approx=True)
        for kind in LATTICES for r in (16,) for a in (0.0, ALPHA_STAR)]

    # S4: short time-integration probe (CSS vs the CSF control)
    payload["dynamic_probe"] = [
        dynamic_probe(RADIUS, probe_rung, ALPHA_STAR, model="css"),
        dynamic_probe(RADIUS, probe_rung, 0.0, model="css"),
        dynamic_probe(RADIUS, probe_rung, 0.0, model="csf"),
    ]

    # S5: verdict, computed from the numbers above and the predeclared rules
    a_sq = payload["alpha_null"]["square"]["alpha_null"]
    a_hex = payload["alpha_null"]["hex"]["alpha_null"]
    untried = [c for c in configs if c["completion_weight"] == ALPHA_STAR
               and (c["lattice"] == "hex" or c["offset_dx"] != [0.0, 0.0])]
    worst = max(untried, key=lambda c: abs(c["err_pct"]))
    payload["verdict"] = {
        "transfer_ok": bool(all(abs(c["err_pct"]) < GATE * 100.0 for c in untried)),
        "transfer_worst": {"lattice": worst["lattice"], "offset_dx": worst["offset_dx"],
                           "radius_over_dx": worst["radius_over_dx"],
                           "err_pct": worst["err_pct"]},
        "calibration_dependent": bool(abs(a_hex - a_sq) > 0.10 * a_sq),
        "alpha_null": {"square": a_sq, "hex": a_hex,
                       "relative_shift_pct": (a_hex - a_sq) / a_sq * 100.0},
        "momentum_ok": bool(all(c["momentum_residual"] < MOMENTUM_TOL for c in configs)),
        "dynamic_probe_contained": {p["model"]: p["contained"]
                                    for p in payload["dynamic_probe"]},
    }
    payload["runtime_s"] = time.time() - t0
    (out / "gate.json").write_text(json.dumps(payload, indent=2))
    plot(payload, out / "transfer.png")
    print(f"alpha* = {ALPHA_STAR:.4f}   alpha_null square {a_sq:.4f}   hex {a_hex:.4f}")
    for c in configs:
        if c["completion_weight"] == ALPHA_STAR:
            print(f"  {c['lattice']:6s} off={c['offset_dx']} R/dx={c['radius_over_dx']:3.0f}"
                  f"  err={c['err_pct']:+7.3f} %")
    print("  verdict:", json.dumps(payload["verdict"], indent=1)[:1200])
    print(f"  runtime {payload['runtime_s']:.1f} s -> {out}")
    return payload


def plot(payload: dict, path: Path) -> None:
    """Figure: error vs lattice configuration, closure scan, functional indicator."""
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except Exception:  # pragma: no cover - plotting is optional
        return
    fig, axes = plt.subplots(1, 3, figsize=(13.5, 4.0))

    ax = axes[0]
    star = [c for c in payload["configurations"] if c["completion_weight"] == ALPHA_STAR]
    base = [c for c in payload["configurations"] if c["completion_weight"] == 0.0]
    labels = [f"{c['lattice'][0].upper()}{'' if c['offset_dx'] == [0.0, 0.0] else '*'}"
              f"\n{c['radius_over_dx']:.0f}" for c in star]
    x = range(len(star))
    ax.bar([i - 0.2 for i in x], [c["err_pct"] for c in base], 0.4,
           label=r"$\alpha=0$ (round-2 baseline)")
    ax.bar([i + 0.2 for i in x], [c["err_pct"] for c in star], 0.4,
           label=rf"$\alpha=\alpha^*={ALPHA_STAR:.3f}$")
    ax.axhspan(-GATE * 100, GATE * 100, color="0.85", zorder=0)
    ax.axhline(0.0, color="k", lw=0.8)
    ax.set_xticks(list(x))
    ax.set_xticklabels(labels, fontsize=6)
    ax.set_ylabel(r"$\Delta p$ error vs $\sigma/R$  [%]")
    ax.set_title("(a) transfer to unseen lattices\n(S=square, H=hex, * = offset)")
    ax.legend(fontsize=8)
    ax.grid(alpha=0.3, axis="y")

    ax = axes[1]
    for kind, marker in (("square", "o-"), ("hex", "s--")):
        recs = payload["alpha_scan"][kind]
        ax.plot([r["completion_weight"] for r in recs], [r["err_pct"] for r in recs],
                marker, label=kind, ms=4)
        a0 = payload["alpha_null"][kind]["alpha_null"]
        ax.axvline(a0, color="0.4", lw=0.8, ls=":")
        ax.annotate(rf"$\alpha_{{null}}$={a0:.3f}", xy=(a0, 10), fontsize=8)
    ax.axhline(0.0, color="k", lw=0.8)
    ax.axhspan(-GATE * 100, GATE * 100, color="0.85", zorder=0)
    ax.set_xlabel(r"closure weight $\alpha$")
    ax.set_ylabel("error [%]")
    ax.set_title("(b) the weight that nulls the reading\ndepends on the lattice")
    ax.legend(fontsize=8)
    ax.grid(alpha=0.3)

    ax = axes[2]
    recs = payload["mode_stiffness"]
    alphas = sorted({r["completion_weight"] for r in recs})
    width = 0.8 / len(LATTICES)
    for k, kind in enumerate(LATTICES):
        vals = [next(r["omega_err_pct"] for r in recs
                     if r["completion_weight"] == a and r["lattice"] == kind
                     and r["radius_over_dx"] == 16) for a in alphas]
        ax.bar([i + k * width for i in range(len(alphas))], vals, width,
               label=f"{kind} (R/dx=16)")
    ax.axhspan(-GATE * 100, GATE * 100, color="0.85", zorder=0)
    ax.axhline(0.0, color="k", lw=0.8)
    ax.set_xticks(range(len(alphas)))
    ax.set_xticklabels([f"{a:.2f}" for a in alphas])
    ax.set_xlabel(r"closure weight $\alpha$")
    ax.set_ylabel(r"$\omega_2$ indicator error [%]")
    ax.set_title("(c) different functional, different weight\n(mode-2 projection, no pressure feedback)")
    ax.legend(fontsize=7)
    ax.grid(alpha=0.3, axis="y")

    fig.suptitle("Scope of the single-phase interface-completion closure: "
                 "transfer across lattices vs the lattice dependence of its weight")
    fig.tight_layout()
    fig.savefig(path, dpi=130)
    plt.close(fig)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=str(OUT))
    ap.add_argument("--rungs", default="16,24,32")
    ap.add_argument("--probe-rung", type=int, default=16)
    ap.add_argument("--threads", type=int, default=2)
    args = ap.parse_args()
    run(Path(args.out), tuple(int(s) for s in args.rungs.split(",")),
        args.probe_rung, args.threads)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())