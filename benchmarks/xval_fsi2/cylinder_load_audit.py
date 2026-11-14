"""Analytic audit of the force-from-fluid path on a *curved* rigid body.

Anchor ``XVAL-SPH-FSI2-BENCHMARK-AND-FORCE-FROM-FLUID``.  Four gates, all with
an exact answer, all cheap (no time integration):

A  bookkeeping     ``sum(-p n trib)`` with the analytic field imposed on the
                   cylinder stations -> ``-grad p * A``.  Exact identity for a
                   regular station set with radial normals (it checks the
                   normal sign and the tributary convention).
B  kernel average  the OSS-style operator (``sum w p / sum w``) on a fluid
                   annulus whose pressure is the analytic field.  The
                   one-sided (half-space) support biases the face pressure by
                   ``grad p . delta``, ``delta = O(h)`` outward, which on a
                   convex body is a radial overload of relative size O(h/r).
C  plane fit       ``boundary_pressure_from_fluid_linear_exact``: weighted
                   least-squares plane through the same neighbours, evaluated
                   at the surface station.  Exact for a linear field.
D  viscous wiring  a flat wall under a linear shear ``u = gamma (y - y0)``:
                   ``F_x`` per unit length must be ``mu gamma``.

Writes ``artifacts/xval_fsi2/cylinder_load_audit.json`` (programmatic, nothing
hand-typed) and the figure ``docs/xval_fsi2/assets/cylinder_load_convergence.png``.
"""
from __future__ import annotations

import json
import math
import sys
from pathlib import Path

import torch

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
sys.path.insert(0, str(ROOT / "src"))

from tensorsph.fsi2_turek_hron import (  # noqa: E402
    FSI2Case, boundary_force_from_pressure, boundary_pressure_from_fluid,
    boundary_pressure_from_fluid_linear_exact, cylinder_surface_geometry,
    wall_shear_from_velocity_gradient,
)
from tensorsph.neighbor_search import cell_linked_pairs  # noqa: E402

DX_LADDER = (0.04, 0.02, 0.01, 0.005)
GRAD_P = torch.tensor([0.0, -1.0], dtype=torch.float64)  # p = p0 + grad_p . x
P0 = 1.0


def _field(x):
    return P0 + x @ GRAD_P


def _annulus(case, dx, pad=3.0):
    """Fluid lattice around the cylinder, outside the surface, within 3 h."""
    r0 = 0.5 * case.cylinder_d
    h = case.h()
    ext = r0 + pad * h
    xs = torch.arange(-ext, ext + dx, dx, dtype=torch.float64)
    gx, gy = torch.meshgrid(xs, xs, indexing="xy")
    d = torch.stack((gx.reshape(-1), gy.reshape(-1)), dim=-1)
    keep = d.norm(dim=1) > (r0 + 0.5 * dx)
    centre = torch.tensor([case.cylinder_x, case.cylinder_y], dtype=torch.float64)
    return d[keep] + centre


def run_gate(dx: float) -> dict:
    case = FSI2Case()
    case.dx = dx
    pos, nrm, trib = cylinder_surface_geometry(case)
    n = pos.shape[0]
    trib_t = torch.full((n,), trib, dtype=torch.float64)
    area = math.pi * (0.5 * case.cylinder_d) ** 2
    exact = -(GRAD_P * area)  # -grad p * A

    # Gate A: analytic pressure imposed directly on the stations
    f_a = boundary_force_from_pressure(_field(pos), nrm, trib_t).sum(dim=0)

    # Gate B / C: the two operators on the same neighbour set
    fpos = _annulus(case, dx)
    pf = _field(fpos)
    pairs = cell_linked_pairs(torch.cat((fpos, pos), dim=0), 2.0 * case.h())
    p_avg, w_avg = boundary_pressure_from_fluid(
        fpos, pf, pos, 0, pairs, case.h(), wall_stop_idx=n)
    p_fit, _ = boundary_pressure_from_fluid_linear_exact(
        fpos, pf, pos, 0, n, pairs, case.h())
    f_b = boundary_force_from_pressure(p_avg, nrm, trib_t).sum(dim=0)
    f_c = boundary_force_from_pressure(p_fit, nrm, trib_t).sum(dim=0)

    def rel(f):
        return float((f[1] - exact[1]) / exact[1].abs())

    return {
        "dx": dx, "n_stations": n, "n_fluid": int(fpos.shape[0]),
        "tributary": float(trib),
        "exact_fy": float(exact[1]),
        "gate_a_rel_err": rel(f_a),
        "gate_b_rel_err": rel(f_b),
        "gate_b_drag_component_rel_err": float(f_b[0] / exact[1]),
        "gate_c_rel_err": rel(f_c),
        "gate_c_drag_component_rel_err": float(f_c[0] / exact[1]),
        "coverage": float((w_avg > 1.0e-30).to(torch.float64).mean()),
    }


def run_gate_d(dx: float = 0.01, gamma: float = 2.0, mu: float = 1.0) -> dict:
    """Flat wall under linear shear: F_x per particle must be ``mu gamma trib``."""
    n_wall = 20
    y0 = 0.0
    xs = (torch.arange(n_wall, dtype=torch.float64) + 0.5) * dx
    wall = torch.stack((xs, torch.full_like(xs, y0)), dim=-1)
    normals = torch.zeros_like(wall)
    normals[:, 1] = 1.0
    trib = torch.full((n_wall,), dx, dtype=torch.float64)
    h = 1.3 * dx
    ys = (torch.arange(1, 5, dtype=torch.float64)) * dx
    gx, gy = torch.meshgrid(xs, ys, indexing="xy")
    fpos = torch.stack((gx.reshape(-1), gy.reshape(-1)), dim=-1)
    vel = torch.zeros_like(fpos)
    vel[:, 0] = gamma * (fpos[:, 1] - y0)
    vol = torch.full((fpos.shape[0],), dx * dx, dtype=torch.float64)
    muf = torch.full((fpos.shape[0],), mu, dtype=torch.float64)
    f = wall_shear_from_velocity_gradient(wall, normals, trib, fpos, vel, vol, muf, h)
    analytic = mu * gamma * dx  # per station, per unit depth
    mid = f[n_wall // 2]
    return {
        "dx": dx, "gamma": gamma, "mu": mu,
        "analytic_fx_per_station": analytic,
        "measured_fx_per_station": float(mid[0]),
        "rel_err_mid_station": float((mid[0] - analytic) / analytic),
        "rel_err_mean": float((f[:, 0].mean() - analytic) / analytic),
    }


def make_figure(rows, gate_d, path: Path) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    dx = [r["dx"] for r in rows]
    fig, ax = plt.subplots(1, 2, figsize=(10, 4))
    ax[0].loglog(dx, [abs(r["gate_b_rel_err"]) * 100 for r in rows], "o-",
                 label="kernel average (OSS arm)")
    ax[0].loglog(dx, [abs(r["gate_c_rel_err"]) * 100 for r in rows], "s-",
                 label="plane fit (linear-exact)")
    ax[0].loglog(dx, [1.0 * d / (0.5 * 0.1) for d in dx], "k--",
                 label=r"$O(h/r)$ guide")
    ax[0].set_xlabel("dx [m]"); ax[0].set_ylabel("|F_y error| vs $-\\nabla p A$ [%]")
    ax[0].invert_xaxis(); ax[0].grid(True, which="both", alpha=0.3)
    ax[0].axhline(15.0, color="r", ls=":", label="15 % acceptance")
    ax[0].set_title("Convex body, imposed linear pressure field")
    ax[0].legend(fontsize=8)
    ax[1].axis("off")
    txt = ["gate D (viscous wiring, Couette)"]
    for k, v in gate_d.items():
        txt.append(f"  {k} = {v:.6g}" if isinstance(v, float) else f"  {k} = {v}")
    txt.append("")
    txt.append("gate A (bookkeeping): exact to machine precision")
    ax[1].text(0.0, 0.9, "\n".join(txt), va="top", family="monospace", fontsize=9)
    fig.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=160)


def main() -> dict:
    rows = [run_gate(dx) for dx in DX_LADDER]
    gate_d = run_gate_d()
    out = {"dx_ladder": list(DX_LADDER), "gates": rows, "gate_d": gate_d}
    art = ROOT / "artifacts" / "xval_fsi2" / "cylinder_load_audit.json"
    art.parent.mkdir(parents=True, exist_ok=True)
    art.write_text(json.dumps(out, indent=2))
    make_figure(rows, gate_d, ROOT / "docs" / "xval_fsi2" / "assets" / "cylinder_load_convergence.png")
    for r in rows:
        print(f"dx={r['dx']:.3f}  A={r['gate_a_rel_err']:+.2e}  "
              f"B(avg)={r['gate_b_rel_err']*100:+.2f}%  C(fit)={r['gate_c_rel_err']*100:+.2e}%  "
              f"drag-comp B={r['gate_b_drag_component_rel_err']*100:+.2f}% C={r['gate_c_drag_component_rel_err']*100:+.2e}%")
    print("gate D:", gate_d)
    return out


if __name__ == "__main__":
    main()
