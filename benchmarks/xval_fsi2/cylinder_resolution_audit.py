"""Cylinder load arm at the *coupled fsi2* resolution: ``h/R`` up to 1.3.

Anchor ``XVAL-SPH-FSI2-BENCHMARK-AND-FORCE-FROM-FLUID``.

Why this audit exists
---------------------
Every static audit of the cylinder load in this repository (``CYLINDER_BODY_FORCE.md``,
``cylinder_load_audit.py``) used the SPHinXsys ``test_2d_fsi2`` geometry, cylinder
radius ``R = 0.5 m`` with ``dx <= 0.04 m``, i.e. ``h/R <= 0.10``.  The *coupled*
Turek-Hron runs use the benchmark's own scale, ``D = 0.1 m`` -> ``R = 0.05 m``,
with ``dx = 0.02 m`` -- ``h/R = 0.52`` and a kernel support ``2h = 0.052 m`` that
is **larger than the body radius**.  In that regime a surface particle's neighbour
set is no longer a half space: it wraps around the body, so a kernel average can
mix the stagnation side with the base side.  No published audit covers it, and it
is the one configuration in which the coupled driver reports ``Cd_mean = 0.052``
against a benchmark value ``3.2`` (``docs/xval_fsi2/VIV_PERIODIC_VERDICT.md``).

What is measured (analytic reference, no solver, no calibration)
---------------------------------------------------------------
A closed circular surface of radius ``R`` in the linear pressure field
``p = p0 + c.x`` has the exact load

    F = -oint p n dl = -c * A,      A = pi R^2

(divergence theorem). Two arms of :mod:`tensorsph.fsi2_turek_hron` are scored on
the same geometry:

* ``kernel``  -- ``boundary_pressure_from_fluid`` (kernel-weighted mean),
* ``fit``     -- ``boundary_pressure_from_fluid_linear_exact`` (weighted plane fit),

both converted to traction with ``boundary_force_from_pressure`` and the exact
tributary arc ``2 pi R / n_circ``.

Three sweeps:

``regime``   ``R = 0.05`` (coupled fsi2 body) with the driver's own station rule
             ``n_circ = max(ceil(2 pi R / dx), 8)``; ``dx`` from 0.04 to 0.005.
``control``  ``R = 0.5`` (SPHinXsys scale) at the same ``dx`` ladder: the
             previously audited regime, kept as a control.
``stations`` ``R = 0.05`` and ``dx = 0.02`` fixed, ``n_circ`` refined 8 -> 128:
             separates *quadrature* error (fixable without touching the fluid)
             from *support* error (not fixable without changing ``h``).

Both components are reported: ``x`` is the drag direction (the component the
coupled run fails on) and ``y`` is the gradient direction (the component the
earlier audits scored).
"""
from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))

from tensorsph.fsi2_turek_hron import (  # noqa: E402
    boundary_force_from_pressure,
    boundary_pressure_from_fluid,
    boundary_pressure_from_fluid_linear_exact,
)
from tensorsph.neighbor_search import cell_linked_pairs  # noqa: E402

GRAD = torch.tensor([1.0, -1.0], dtype=torch.float64)  # p = 1 + GRAD . x
P0 = 1.0
H_FACTOR = 1.3
SUPPORT_FACTOR = 2.0


def circle_surface(radius: float, n_circ: int, centre=(0.0, 0.0)):
    """Stations on the circle, outward normals, exact tributary arc."""
    th = torch.arange(n_circ, dtype=torch.float64) * (2.0 * math.pi / n_circ)
    nrm = torch.stack((torch.cos(th), torch.sin(th)), dim=1)
    pos = torch.stack((centre[0] + radius * nrm[:, 0],
                       centre[1] + radius * nrm[:, 1]), dim=1)
    return pos, nrm, torch.full((n_circ,), 2.0 * math.pi * radius / n_circ,
                                dtype=torch.float64)


def fluid_lattice(radius: float, dx: float, centre=(0.0, 0.0), pad_cells: int = 5):
    """Lattice fluid with the *driver's own* clearance rule.

    ``build_fluid_particles`` removes every lattice point with
    ``|x - c| < R + 0.5 dx``; the same rule is applied here so the neighbour sets
    are the ones the coupled run actually has.
    """
    ext = radius + pad_cells * H_FACTOR * dx
    xs = torch.arange(-ext, ext + 0.5 * dx, dx, dtype=torch.float64)
    gx, gy = torch.meshgrid(xs, xs, indexing="xy")
    pos = torch.stack((gx.reshape(-1) + centre[0], gy.reshape(-1) + centre[1]),
                      dim=1)
    d = torch.linalg.vector_norm(pos - torch.tensor(centre, dtype=torch.float64),
                                 dim=1)
    return pos[d > radius + 0.5 * dx]


def _pressures(pos_fluid, pos_solid, h):
    p_fluid = P0 + pos_fluid @ GRAD
    wall = pos_solid
    allp = torch.cat((pos_fluid, wall), dim=0)
    pairs = cell_linked_pairs(allp, SUPPORT_FACTOR * h)
    p_kernel, w_sum = boundary_pressure_from_fluid(
        pos_fluid, p_fluid, wall, 0, pairs, h, wall_stop_idx=wall.shape[0])
    p_fit, _ = boundary_pressure_from_fluid_linear_exact(
        pos_fluid, p_fluid, wall, 0, wall.shape[0], pairs, h)
    return p_kernel, p_fit, pairs, allp, p_fluid


def score(radius: float, dx: float, n_circ: int) -> dict:
    h = H_FACTOR * dx
    solid, normal, trib = circle_surface(radius, n_circ)
    fluid = fluid_lattice(radius, dx)
    p_k, p_f, pairs, allp, p_fluid = _pressures(fluid, solid, h)
    f_k = boundary_force_from_pressure(p_k, normal, trib).sum(dim=0)
    f_f = boundary_force_from_pressure(p_f, normal, trib).sum(dim=0)
    area = math.pi * radius ** 2
    ref = -GRAD * area
    # gap statistics: distance from each station to its nearest fluid particle
    chunks = []
    for k in range(0, n_circ, 64):
        chunks.append(torch.cdist(solid[k:k + 64], fluid).min(dim=1).values)
    gap = torch.cat(chunks)
    covered = int(torch.isfinite(gap).sum())
    out = {"radius": radius, "dx": dx, "h": h, "h_over_R": h / radius,
           "support_over_R": SUPPORT_FACTOR * h / radius, "n_circ": n_circ,
           "trib": 2.0 * math.pi * radius / n_circ,
           "n_fluid": int(fluid.shape[0]), "covered_stations": covered,
           "coverage": covered / n_circ,
           "gap_min_over_dx": float(gap[torch.isfinite(gap)].min()) / dx,
           "gap_max_over_dx": float(gap[torch.isfinite(gap)].max()) / dx,
           "ref_x": float(ref[0]), "ref_y": float(ref[1])}
    for tag, f in (("kernel", f_k), ("fit", f_f)):
        out[f"{tag}_x"] = float(f[0])
        out[f"{tag}_y"] = float(f[1])
        out[f"{tag}_err_x"] = float(f[0] / ref[0] - 1.0)
        out[f"{tag}_err_y"] = float(f[1] / ref[1] - 1.0)
        out[f"{tag}_mag_ratio"] = float(torch.linalg.vector_norm(f)
                                       / torch.linalg.vector_norm(ref))
    return out


def regime_sweep():
    rows = []
    for radius in (0.05, 0.5):
        for dx in (0.04, 0.02, 0.01, 0.005):
            n_circ = max(int(math.ceil(2.0 * math.pi * radius / dx)), 8)
            rows.append(score(radius, dx, n_circ))
    return rows


def station_sweep(radius=0.05, dx=0.02):
    return [score(radius, dx, n) for n in (8, 16, 24, 32, 48, 64, 96, 128)]


def _fmt(rows, keys, hdr):
    lines = ["| " + " | ".join(hdr) + " |", "|" + "---|" * len(hdr)]
    for r in rows:
        cells = []
        for k in keys:
            v = r[k]
            if isinstance(v, float):
                cells.append(f"{v:.3e}" if abs(v) < 1e-3 and v != 0 else f"{v:.4g}")
            else:
                cells.append(str(v))
        lines.append("| " + " | ".join(cells) + " |")
    return "\n".join(lines)


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="artifacts/xval_fsi2/cylinder_resolution.json")
    ap.add_argument("--figure", default="docs/xval_fsi2/assets/cylinder_resolution.png")
    args = ap.parse_args(argv)

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    payload = {"gradient": GRAD.tolist(), "p0": P0, "h_factor": H_FACTOR,
               "support_factor": SUPPORT_FACTOR,
               "regime": regime_sweep(), "stations": station_sweep()}
    out.write_text(json.dumps(payload, indent=1))

    print("### regime sweep (driver station rule n_circ = max(ceil(2 pi R/dx), 8))")
    print(_fmt(payload["regime"],
               ("radius", "dx", "h_over_R", "support_over_R", "n_circ",
                "coverage", "gap_min_over_dx", "kernel_err_x", "kernel_err_y",
                "fit_err_x", "fit_err_y", "kernel_mag_ratio", "fit_mag_ratio"),
               ("R", "dx", "h/R", "2h/R", "n_circ", "cover", "gap_min/dx",
                "kernel err_x", "kernel err_y", "fit err_x", "fit err_y",
                "kernel |F|/|ref|", "fit |F|/|ref|")))
    print()
    print("### station sweep (R = 0.05, dx = 0.02, i.e. the coupled configuration)")
    print(_fmt(payload["stations"],
               ("n_circ", "trib", "coverage", "gap_min_over_dx", "kernel_err_x",
                "kernel_err_y", "fit_err_x", "fit_err_y"),
               ("n_circ", "trib", "cover", "gap_min/dx", "kernel err_x",
                "kernel err_y", "fit err_x", "fit err_y")))

    try:
        _figure(payload, Path(args.figure))
        print(f"\nfigure -> {args.figure}")
    except Exception as exc:  # pragma: no cover - plotting is optional
        print(f"figure skipped: {exc}")
    print(f"artifacts -> {args.out}")
    return 0


def _figure(payload, path: Path) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    path.parent.mkdir(parents=True, exist_ok=True)
    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    ax = axes[0]
    for arm, style in (("kernel", "o-"), ("fit", "s-")):
        for radius, colour in ((0.05, "C0"), (0.5, "C1")):
            rows = [r for r in payload["regime"] if r["radius"] == radius]
            rows.sort(key=lambda r: -r["h_over_R"])
            ax.plot([r["h_over_R"] for r in rows],
                    [100.0 * r[f"{arm}_err_x"] for r in rows], style, color=colour,
                    label=f"{arm} x, R={radius}")
    ax.axhline(15.0, color="0.5", ls="--", lw=1)
    ax.axhline(-15.0, color="0.5", ls="--", lw=1)
    ax.set_xscale("log")
    ax.set_xlabel("h / R")
    ax.set_ylabel("drag-direction error [%]")
    ax.set_title("(a) drag arm error vs h/R (dashed: 15 % gate)")
    ax.legend(fontsize=7)
    ax = axes[1]
    rows = sorted(payload["stations"], key=lambda r: r["n_circ"])
    for arm, style in (("kernel", "o-"), ("fit", "s-")):
        ax.plot([r["n_circ"] for r in rows], [100.0 * r[f"{arm}_err_x"] for r in rows],
                style, label=f"{arm} err_x")
        ax.plot([r["n_circ"] for r in rows], [100.0 * r[f"{arm}_err_y"] for r in rows],
                style, ls=":", label=f"{arm} err_y")
    ax.axhline(15.0, color="0.5", ls="--", lw=1)
    ax.axhline(-15.0, color="0.5", ls="--", lw=1)
    ax.set_xscale("log")
    ax.set_xlabel("n_circ [stations]")
    ax.set_ylabel("error [%]")
    ax.set_title("(b) station refinement at R=0.05, dx=0.02")
    ax.legend(fontsize=7)
    fig.tight_layout()
    fig.savefig(path, dpi=140)


if __name__ == "__main__":
    raise SystemExit(main())