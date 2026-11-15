"""Far-field (control-volume) force-from-fluid audit for the ``fsi2`` anchor.

Anchor ``XVAL-SPH-FSI2-BENCHMARK-AND-FORCE-FROM-FLUID`` (Xiangyu-Hu/SPHinXsys
``test_2d_fsi2`` + ``FSI::ForceFromFluid``).  Module under audit:
:mod:`tensorsph.farfield_force`.

Four legs, every one with an answer that is known *before* the run:

G1  closed-surface identities  a) uniform flow + constant pressure -> force = 0;
                              b) linear pressure, no flow -> ``F = -A grad p``.
G2  Kutta-Joukowski / d'Alembert  steady potential flow past a cylinder,
                              circular control volume -> ``F_y = -rho U Gamma``,
                              ``F_x = 0``.  Swept in ``R/a`` and ``dx/a``.
G3  wake momentum deficit     synthetic Gaussian wake, plane downstream ->
                              closed-form ``rho U^2 eps b sqrt(pi) (1 - eps/sqrt 2)``.
G4  benchmark scale           the same potential flow at the coupled ``fsi2``
                              geometry (``D = 0.1 m``, ``dx = 0.02..0.005``,
                              ``h/R`` up to 1.04) -- far-field arm against the
                              *near-field* kernel and plane-fit arms evaluated
                              on the same analytic field.

Everything is deterministic; ``--out`` holds every number in the document.
"""
from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path

import torch

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
sys.path.insert(0, str(ROOT / "src"))

from tensorsph.farfield_force import (  # noqa: E402
    control_volume_force,
    potential_flow_cylinder,
    wake_plane_drag,
)
from tensorsph.fsi2_turek_hron import (  # noqa: E402
    FSI2Case,
    boundary_force_from_pressure,
    boundary_pressure_from_fluid,
    boundary_pressure_from_fluid_linear_exact,
    cylinder_surface_geometry,
)
from tensorsph.neighbor_search import cell_linked_pairs  # noqa: E402

DT = torch.float64


def lattice(x0, x1, y0, y1, dx):
    xs = torch.arange(x0, x1 + 0.5 * dx, dx, dtype=DT)
    ys = torch.arange(y0, y1 + 0.5 * dx, dx, dtype=DT)
    gx, gy = torch.meshgrid(xs, ys, indexing="xy")
    return torch.stack((gx.reshape(-1), gy.reshape(-1)), dim=-1)


def annulus_lattice(centre, r_in, r_out, dx):
    ext = r_out + dx
    pts = lattice(centre[0] - ext, centre[0] + ext,
                  centre[1] - ext, centre[1] + ext, dx)
    r = torch.linalg.norm(pts - centre, dim=1)
    return pts[(r > r_in) & (r < r_out)]


# --------------------------------------------------------------------------- #
# G1: closed-surface exactness                                            #
# --------------------------------------------------------------------------- #
def gate_g1() -> dict:
    """Two closed-surface identities with different answers.

    G1a null       uniform flow + **constant** pressure -> force = 0.  Only a
                   quadrature that closes the surface (``oint n dS = 0``) passes.
    G1b buoyancy   no flow + **linear** pressure ``p = grad_p . x`` ->
                   ``F = -oint p n dS = -A grad_p`` (``A = pi R^2``).  This leg
                   exists because the first version of this audit assumed the
                   linear field was a null case; it is not, and the measured
                   ``-A grad_p`` is what exposed the mistake.
    """
    centre = torch.zeros(2, dtype=DT)
    grad_p = torch.tensor([0.7, -1.1], dtype=DT)
    p0, rho = 1.0, 1.0
    u_uniform = torch.tensor([0.3, -0.2], dtype=DT)
    rows = []
    for radius in (0.4, 0.8):
        for dx in (0.02, 0.01):
            band = 1.0 * dx
            pos = annulus_lattice(centre, radius - band, radius + band, dx)
            u = u_uniform.expand_as(pos)
            # G1a: constant pressure
            out_a = control_volume_force(pos, u, torch.full((pos.shape[0],), p0,
                                                            dtype=DT),
                                         rho, centre, radius, dx, p_ref=p0)
            scale_a = rho * float((u_uniform ** 2).sum()) * radius
            # G1b: linear pressure, no flow
            pres = p0 + pos @ grad_p
            out_b = control_volume_force(pos, torch.zeros_like(pos), pres, rho,
                                         centre, radius, dx, p_ref=p0)
            area = math.pi * radius ** 2
            exact_b = -(grad_p * area)
            rows.append({
                "radius": radius,
                "dx": dx,
                "n_bins": out_a["n_bins"],
                "n_particles": out_a["n_surface_particles"],
                "coverage": out_a["coverage"],
                "null_fx": float(out_a["force"][0]),
                "null_fy": float(out_a["force"][1]),
                "null_abs": float(torch.linalg.norm(out_a["force"])),
                "null_rel_to_scale": float(torch.linalg.norm(out_a["force"])) / scale_a,
                "buoy_exact": [float(exact_b[0]), float(exact_b[1])],
                "buoy_fx": float(out_b["force"][0]),
                "buoy_fy": float(out_b["force"][1]),
                "buoy_rel_err": float(torch.linalg.norm(out_b["force"] - exact_b))
                                / float(torch.linalg.norm(exact_b)),
                "buoy_mean_reconstruction_rel_err": float(
                    torch.linalg.norm(out_b["force_mean_reconstruction"] - exact_b))
                    / float(torch.linalg.norm(exact_b)),
            })
    return {
        "rows": rows,
        "worst_null_rel": max(r["null_rel_to_scale"] for r in rows),
        "worst_buoy_rel": max(r["buoy_rel_err"] for r in rows),
        "worst_buoy_mean_rel": max(r["buoy_mean_reconstruction_rel_err"] for r in rows),
    }


# --------------------------------------------------------------------------- #
# G2: Kutta-Joukowski lift and d'Alembert drag                            #
# --------------------------------------------------------------------------- #
def gate_g2(a=0.1, u_inf=1.0, circulation=None, rho=1.0) -> list:
    if circulation is None:
        circulation = 2.0 * math.pi * a * u_inf * 0.5
    centre = torch.zeros(2, dtype=DT)
    exact_lift = -rho * u_inf * circulation
    rows = []
    for r_over_a in (3.0, 5.0, 10.0):
        for dx_over_a in (0.08, 0.04, 0.02):
            radius = r_over_a * a
            dx = dx_over_a * a
            band = 0.75 * dx
            pos = annulus_lattice(centre, radius - band, radius + band, dx)
            vel, pres = potential_flow_cylinder(pos, a, u_inf, circulation, rho)
            out = control_volume_force(pos, vel, pres, rho, centre, radius, dx,
                                       p_ref=0.0)
            fy = float(out["force"][1])
            fx = float(out["force"][0])
            rows.append({
                "r_over_a": r_over_a,
                "dx_over_a": dx_over_a,
                "dx": dx,
                "radius": radius,
                "n_bins": out["n_bins"],
                "coverage": out["coverage"],
                "exact_lift": exact_lift,
                "lift": fy,
                "lift_rel_err": (fy - exact_lift) / abs(exact_lift),
                "drag": fx,
                "drag_rel_to_lift": fx / abs(exact_lift),
                "lift_momentum_flux": float(out["force_momentum_flux"][1]),
                "lift_pressure": float(out["force_pressure"][1]),
            })
    return rows


# --------------------------------------------------------------------------- #
# G3: wake-plane momentum deficit against a closed-form Gaussian wake      #
# --------------------------------------------------------------------------- #
def gate_g3(u_inf=1.0, eps=0.25, b=0.2, rho=1.0) -> dict:
    y_range = 6.0 * b
    rows = []
    for dx in (0.02, 0.01, 0.005):
        x_plane = 0.5
        band = 0.75 * dx
        pos = lattice(x_plane - band, x_plane + band, -y_range, y_range, dx)
        vel = torch.zeros_like(pos)
        vel[:, 0] = u_inf * (1.0 - eps * torch.exp(-(pos[:, 1] / b) ** 2))
        pres = torch.zeros(pos.shape[0], dtype=DT)
        out = wake_plane_drag(pos, vel, pres, rho, u_inf, 0.0, x_plane,
                              y_range, dx)
        exact = rho * u_inf ** 2 * eps * b * math.sqrt(math.pi) * (1.0 - eps / math.sqrt(2.0))
        rows.append({
            "dx": dx,
            "n_bins": out["n_bins"],
            "coverage": out["coverage"],
            "exact_drag": exact,
            "drag": float(out["drag"]),
            "rel_err": (float(out["drag"]) - exact) / exact,
        })
    return {"u_inf": u_inf, "eps": eps, "b": b, "rows": rows}


# --------------------------------------------------------------------------- #
# G4: the coupled fsi2 scale -- far-field vs near-field on the same field  #
# --------------------------------------------------------------------------- #
def nearfield_arms(case, fluid_pos, pressure, wall_pos, normals, trib, n_wall,
                   h):
    """Kernel arm and plane-fit arm on the analytic field (the driver defaults)."""
    pairs = cell_linked_pairs(torch.cat((fluid_pos, wall_pos), dim=0), 2.0 * h)
    p_avg, w_avg = boundary_pressure_from_fluid(
        fluid_pos, pressure, wall_pos, 0, pairs, h, wall_stop_idx=n_wall)
    p_fit, _ = boundary_pressure_from_fluid_linear_exact(
        fluid_pos, pressure, wall_pos, 0, n_wall, pairs, h)
    f_kernel = boundary_force_from_pressure(p_avg, normals, trib).sum(dim=0)
    f_fit = boundary_force_from_pressure(p_fit, normals, trib).sum(dim=0)
    return {
        "force_kernel": f_kernel,
        "force_fit": f_fit,
        "coverage": float((w_avg > 1e-30).to(DT).mean()),
    }


def gate_g4(dx_list=(0.02, 0.01, 0.005), circulation_factor=0.8) -> list:
    """Three load channels on one analytic field, at the coupled fsi2 spacing.

    Control volumes are swept over two radii: ``10 R`` (free-space far field)
    and ``3 R`` (the largest circle that still fits **inside** the 0.41 m
    channel around the cylinder at ``y = 0.2 m`` -- a 10 R circle would cross
    the channel walls, so the free-space value is an operator check, not a
    configuration the coupled run could actually use).  The near-field arms are
    evaluated on two neighbour sets (annulus thickness ``3 h`` and ``5 h``)
    because a surface operator is a function of the neighbour set it is handed.
    """
    rows = []
    for dx in dx_list:
        case = FSI2Case()
        case.dx = dx
        a = 0.5 * case.cylinder_d                      # R = 0.05 m
        # Turek-Hron: parabolic inlet u_max = 3/2 U_mean -> U_mean = 0.2 m/s
        u_inf = case.inlet_u_max * 2.0 / 3.0
        rho = case.rho0
        circulation = 2.0 * math.pi * a * u_inf * circulation_factor
        exact_lift = -rho * u_inf * circulation
        centre = torch.tensor([case.cylinder_x, case.cylinder_y], dtype=DT)
        band = 1.0 * dx

        far = {}
        for r_over_R in (3.0, 10.0):
            radius = r_over_R * a
            cv_pos = annulus_lattice(centre, radius - band, radius + band, dx)
            vel, pres = potential_flow_cylinder(cv_pos, a, u_inf, circulation,
                                                rho, centre=centre)
            out = control_volume_force(cv_pos, vel, pres, rho, centre, radius,
                                       dx, p_ref=0.0)
            far[r_over_R] = {
                "lift": float(out["force"][1]),
                "drag": float(out["force"][0]),
                "rel_err": (float(out["force"][1]) - exact_lift) / abs(exact_lift),
                "n_bins": out["n_bins"],
                "coverage": out["coverage"],
                "in_channel": bool(radius <= 0.15),
            }

        # near-field arms on the same analytic field, at the coupled resolution
        pos_w, nrm, trib = cylinder_surface_geometry(case)
        n_wall = int(pos_w.shape[0])
        h = case.h()
        near = {}
        for thick_over_h in (3.0, 5.0):
            fluid_pos = annulus_lattice(centre, a + 0.5 * dx, a + thick_over_h * h, dx)
            _, pres_f = potential_flow_cylinder(fluid_pos, a, u_inf, circulation,
                                                rho, centre=centre)
            nf = nearfield_arms(case, fluid_pos, pres_f, pos_w, nrm, float(trib),
                                n_wall, h)
            near[thick_over_h] = {
                "n_fluid": int(fluid_pos.shape[0]),
                "force_kernel": [float(nf["force_kernel"][0]),
                                 float(nf["force_kernel"][1])],
                "force_fit": [float(nf["force_fit"][0]),
                              float(nf["force_fit"][1])],
                "kernel_rel_err": (float(nf["force_kernel"][1]) - exact_lift)
                                  / abs(exact_lift),
                "fit_rel_err": (float(nf["force_fit"][1]) - exact_lift)
                               / abs(exact_lift),
                "coverage": nf["coverage"],
            }
        rows.append({
            "dx": dx,
            "h_over_R": h / a,
            "support_over_R": 2.0 * h / a,
            "n_stations": n_wall,
            "exact_lift": exact_lift,
            "farfield": {str(k): v for k, v in far.items()},
            "nearfield": {str(k): v for k, v in near.items()},
            "farfield_lift_rel_err": far[10.0]["rel_err"],
            "farfield_lift_rel_err_cv3R": far[3.0]["rel_err"],
            "farfield_drag_rel_to_lift": far[10.0]["drag"] / abs(exact_lift),
            "farfield_drag_rel_to_lift_cv3R": far[3.0]["drag"] / abs(exact_lift),
            "cv_coverage": far[10.0]["coverage"],
            "cv3R_coverage": far[3.0]["coverage"],
            "kernel_lift_rel_err": near[3.0]["kernel_rel_err"],
            "fit_lift_rel_err": near[3.0]["fit_rel_err"],
            "kernel_lift_rel_err_5h": near[5.0]["kernel_rel_err"],
            "fit_lift_rel_err_5h": near[5.0]["fit_rel_err"],
            "nearfield_coverage": near[3.0]["coverage"],
            "n_fluid_nearfield": near[3.0]["n_fluid"],
        })
    return rows


# --------------------------------------------------------------------------- #
def make_figure(g2, g4, path: Path) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(1, 2, figsize=(11, 4.2))

    # (a) far-field lift error vs control-volume radius, per spacing
    dxs = sorted({r["dx_over_a"] for r in g2})
    for dxa in dxs:
        sel = [r for r in g2 if r["dx_over_a"] == dxa]
        ax[0].plot([r["r_over_a"] for r in sel],
                   [100 * r["lift_rel_err"] for r in sel], "o-",
                   label=f"dx/a = {dxa}")
    ax[0].axhline(0, color="k", lw=0.8)
    ax[0].axhspan(-3, 3, color="tab:green", alpha=0.15, label="3 % line")
    ax[0].set_xscale("log")
    ax[0].set_xlabel("control-volume radius  R/a")
    ax[0].set_ylabel("Kutta-Joukowski lift error [%]")
    ax[0].set_title("G2 far-field lift vs CV radius")
    ax[0].legend(fontsize=8)
    ax[0].grid(alpha=0.3)

    # (b) fsi2 scale: near-field arms vs far-field, on the same analytic field
    x = [r["dx"] for r in g4]
    w = 0.25
    idx = range(len(x))
    ax[1].bar([i - w for i in idx], [100 * r["kernel_lift_rel_err"] for r in g4],
              width=w, label="near-field kernel arm")
    ax[1].bar(list(idx), [100 * r["fit_lift_rel_err"] for r in g4], width=w,
              label="near-field plane-fit arm")
    ax[1].bar([i + w for i in idx], [100 * r["farfield_lift_rel_err"] for r in g4],
              width=w, label="far-field control volume (10R)")
    ax[1].axhspan(-3, 3, color="tab:green", alpha=0.15, label="3 % line")
    ax[1].axhline(0, color="k", lw=0.8)
    ax[1].set_xticks(list(idx))
    ax[1].set_xticklabels([f"dx={v:g}\nh/R={r['h_over_R']:.2f}" for v, r in zip(x, g4)],
                          fontsize=8)
    ax[1].plot([i + w for i in idx], [100 * r["farfield_lift_rel_err_cv3R"] for r in g4],
               "k*", ms=9, label="far-field CV that fits in the channel (3R)")
    ax[1].set_ylabel("lift error vs analytic [%]")
    ax[1].set_title("G4 fsi2 coupled scale: three load channels")
    ax[1].legend(fontsize=8)
    ax[1].grid(alpha=0.3, axis="y")

    fig.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=140)
    plt.close(fig)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=str(ROOT / "artifacts/xval_fsi2/farfield_force.json"))
    ap.add_argument("--fig", default=str(ROOT / "docs/xval_fsi2/assets/farfield_force.png"))
    args = ap.parse_args()

    torch.manual_seed(0)
    g1 = gate_g1()
    g2 = gate_g2()
    g3 = gate_g3()
    g4 = gate_g4()

    payload = {
        "anchor": "XVAL-SPH-FSI2-BENCHMARK-AND-FORCE-FROM-FLUID",
        "module": "src/tensorsph/farfield_force.py",
        "g1_closed_surface": g1,
        "g2_kutta_joukowski": g2,
        "g3_wake_deficit": g3,
        "g4_fsi2_scale": g4,
        "g1_rows": g1["rows"],
        "g1_worst_null_rel": g1["worst_null_rel"],
        "g1_worst_buoy_rel": g1["worst_buoy_rel"],
        "g1_worst_buoy_mean_rel": g1["worst_buoy_mean_rel"],
        "g2_worst_lift_rel_err": max(abs(r["lift_rel_err"]) for r in g2),
        "g3_worst_rel_err": max(abs(r["rel_err"]) for r in g3["rows"]),
        "g4_farfield_worst_rel_err": max(abs(r["farfield_lift_rel_err"]) for r in g4),
        "g4_farfield_cv3R_worst_rel_err": max(abs(r["farfield_lift_rel_err_cv3R"])
                                              for r in g4),
        "g4_kernel_worst_rel_err": max(abs(r["kernel_lift_rel_err"]) for r in g4),
        "g4_fit_worst_rel_err": max(abs(r["fit_lift_rel_err"]) for r in g4),
    }
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(payload, indent=2))
    make_figure(g2, g4, Path(args.fig))

    print(json.dumps({k: v for k, v in payload.items()
                      if k.startswith("g") and "worst" in k}, indent=2))
    print("g4 rows:")
    for r in g4:
        print(f"  dx={r['dx']:.4f} h/R={r['h_over_R']:.2f} "
              f"far-field(10R) {100*r['farfield_lift_rel_err']:+.3f} % "
              f"far-field(3R) {100*r['farfield_lift_rel_err_cv3R']:+.3f} % "
              f"kernel {100*r['kernel_lift_rel_err']:+.2f} % "
              f"fit {100*r['fit_lift_rel_err']:+.2f} % "
              f"kernel(5h) {100*r['kernel_lift_rel_err_5h']:+.2f} % "
              f"fit(5h) {100*r['fit_lift_rel_err_5h']:+.2f} %")
    print(f"wrote {out} and {args.fig}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())