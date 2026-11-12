#!/usr/bin/env python3
"""Viscous-operator cross-implementation consistency audit (TensorSPH).

Why this exists
---------------
``docs/verification_poiseuille_convergence.md`` (section 6.3, "long-term /
architectural") records an open gap in one sentence:

    The repository has 4 implementations of the WCSPH viscous operator
    (droplet_impact, wcsp_h/solver, threed, sod_shock).  Consolidating to a
    single, well-verified operator would eliminate cross-implementation
    discrepancies.

That sentence *asserts* a discrepancy and never measures it.  Before anything
is consolidated, the size of the discrepancy has to be known: two operators that
agree to 0.1 % do not need consolidating, and an operator that silently applies
``0.5 nu`` poisons every case that uses it.  This benchmark measures the
discrepancy on a common analytic field, so the claim becomes a number.

What is measured
----------------
Every implementation is applied to the *same* analytic velocity field on the
*same* lattice and spacing, and its effective kinematic viscosity is read back:

    acc = nu_eff * lap(u)   =>   nu_eff / nu = sum(acc . lap u) / (nu sum |lap u|^2)

``nu_eff/nu`` is the ratio that enters the physics (it is what a case's Reynolds
number actually is), and it is normalisation-free: an operator that is a
half-strength Laplacian returns 0.5 whatever field it is applied to.  The
residual ``||acc - nu_eff lap u|| / ||nu_eff lap u||`` is reported next to it,
because a ratio of 1.0 with a 10 % residual means the *shape* is wrong even
though the *level* is right.

Field and reference
-------------------
2-D:  the Taylor-Green field of :mod:`tensorsph.vv.tgv`,
      ``u = U e^{i k x} cos k y, v = -U cos k x sin k y`` with ``lap u = -2 k^2 u``
      (exact, and the shape is an exact invariant of the Navier-Stokes equations,
      so it is a reference and not a fit).
3-D:  ``u = (sin kx sin ky sin kz, 0, 0)`` with ``lap u = -3 k^2 u`` (component
      zero, exact).  A genuinely 3-D field is used deliberately: applying a 3-D
      operator to a z-independent field convolves the 3-D kernel with a 2-D one,
      which would confound the operator under test with a kernel-projection
      effect.

Implementations under test
--------------------------
============== ==========================================================
key            code path
============== ==========================================================
``tgv_morris`` :func:`tensorsph.vv.tgv.viscous_acceleration` -- Morris et al.
               (1997) Laplacian, library ``VISCOUS_LAPLACIAN_FACTOR``, pairs
               from ``tensorsph.vv.tgv.periodic_pairs``.
``droplet_prod`` :func:`tensorsph.droplet_impact.rhs` (production, cell-linked
               pair path), isolated by ``acc(nu) - acc(0)``: the RHS is linear
               in ``nu``, so the difference is the viscous term alone and no
               private re-implementation is needed.
``droplet_dense`` :func:`tensorsph.droplet_impact.rhs_all_pairs` (dense O(N^2)
               reference path), isolated the same way.
``isph_morris`` :func:`tensorsph.isph_channel.viscous_laplacian` -- Morris
               Laplacian with a hard-coded factor 2 and an ``eta`` regulariser.
``threed_renorm2`` / ``threed_renorm4`` / ``threed_raw``
               :func:`tensorsph.threed.operators3.viscous_acceleration` with
               moment renormalisation (order 2 / order 4) and without.
============== ==========================================================

The ``droplet_*`` paths are non-periodic (the production neighbour search has no
periodic image), so their number is read on the interior band where every
particle sees its full 2h support.  The periodic paths are measured on the whole
lattice and additionally on the same interior band, so the two families are
comparable.

Output
------
``results.json`` next to this file: one record per (implementation,
resolution), plus the derived findings.  ``make_doc.py`` renders the table, the
figure and ``docs/verification_viscous_operator_consistency.md`` from it -- no
number in the document is typed by hand.

Usage
-----
    PYTHONPATH=src python benchmarks/viscous_operator_consistency/run.py
"""

from __future__ import annotations

import argparse
import json
import math
import sys
import time
from pathlib import Path

import torch

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "src"))

from tensorsph import isph_channel                                    # noqa: E402
from tensorsph.droplet_impact import (                                # noqa: E402
    VISCOUS_LAPLACIAN_FACTOR, Config, pair_kernels, rhs,
    rhs_all_pairs,
)
from tensorsph.threed import operators3 as ops3                       # noqa: E402
from tensorsph.vv import tgv                                          # noqa: E402

OUT = REPO / "benchmarks" / "viscous_operator_consistency" / "results.json"

#: Common parameters.  ``nu`` is irrelevant to ``nu_eff/nu`` (every operator is
#: linear in ``nu``) but is set to the TGV case's value so the field amplitude
#: and the viscous time scale are physically sensible.
NU = 1.0 / (10.0 * 2.0 * math.pi)      # vv.tgv default: Re = 10
U0 = 1.0
H_FACTOR = 1.3
RHO0 = 1000.0
SOUND_SPEED = 20.0
GAMMA = 7.0
BOX = 1.0
WAVES = 1
K = 2.0 * math.pi * WAVES / BOX
DTYPE = torch.float64
ETA_FACTOR = 0.01                      # Morris regulariser eta^2 = 0.01 h^2


# --------------------------------------------------------------------------- #
# Analytic fields and the error metric
# --------------------------------------------------------------------------- #

def tgv_field(pos: torch.Tensor, t: float = 0.0) -> torch.Tensor:
    """2-D Taylor-Green velocity field (exact shape, amplitude 1 at t=0)."""
    a = U0 * math.exp(-2.0 * NU * K * K * t)
    u = a * torch.sin(K * pos[:, 0]) * torch.cos(K * pos[:, 1])
    v = -a * torch.cos(K * pos[:, 0]) * torch.sin(K * pos[:, 1])
    return torch.stack((u, v), dim=-1)


def tgv_laplacian(pos: torch.Tensor, t: float = 0.0) -> torch.Tensor:
    """``lap u = -2 k^2 u`` -- exact for the Taylor-Green field."""
    return -2.0 * K * K * tgv_field(pos, t)


def box3d_field(pos: torch.Tensor) -> torch.Tensor:
    """3-D field ``u = (sin kx sin ky sin kz, 0, 0)``; only component 0 is alive."""
    f = (torch.sin(K * pos[:, 0]) * torch.sin(K * pos[:, 1])
         * torch.sin(K * pos[:, 2]))
    out = torch.zeros_like(pos)
    out[:, 0] = f
    return out


def box3d_laplacian(pos: torch.Tensor) -> torch.Tensor:
    """``lap u = -3 k^2 u`` -- exact for :func:`box3d_field`."""
    return -3.0 * K * K * box3d_field(pos)


def operator_readout(acc: torch.Tensor, lap: torch.Tensor, nu: float,
                     mask: torch.Tensor | None = None) -> dict:
    """Effective viscosity ratio and residual of ``acc`` against ``nu lap``.

    ``nu_eff/nu`` is the least-squares projection of the computed acceleration on
    the exact Laplacian; the residual is what is left after that projection, so
    a level error (wrong constant) and a shape error (wrong stencil) are
    reported separately.
    """
    if mask is not None:
        acc, lap = acc[mask], lap[mask]
    denom = (lap * lap).sum()
    if float(denom) == 0.0:
        raise ValueError("reference Laplacian vanishes on the selected particles")
    nu_eff = (acc * lap).sum() / denom
    resid = (acc - nu_eff * lap).square().sum().sqrt()
    scale = (nu_eff * lap).square().sum().sqrt()
    return {
        "nu_eff_over_nu": float(nu_eff / nu),
        "residual_pct": float(resid / scale * 100.0) if float(scale) > 0 else float("nan"),
        "n_particles": int(acc.shape[0]),
    }


# --------------------------------------------------------------------------- #
# Lattices
# --------------------------------------------------------------------------- #

def lattice2d(dx: float):
    """Periodic ``n x n`` lattice of cell centres with calibrated-like masses."""
    n = int(round(BOX / dx))
    if abs(n * dx - BOX) > 1e-9 * BOX:
        raise ValueError(f"dx={dx:g} does not divide the box {BOX:g}")
    line = (torch.arange(n, dtype=DTYPE) + 0.5) * dx
    gx, gy = torch.meshgrid(line, line, indexing="xy")
    pos = torch.stack((gx.reshape(-1), gy.reshape(-1)), dim=-1)
    mass = torch.full((pos.shape[0],), RHO0 * dx * dx, dtype=DTYPE)
    return pos, mass, n


def lattice3d(dx: float):
    """Periodic ``n x n x n`` lattice of cell centres."""
    n = int(round(BOX / dx))
    if abs(n * dx - BOX) > 1e-9 * BOX:
        raise ValueError(f"dx={dx:g} does not divide the box {BOX:g}")
    line = (torch.arange(n, dtype=DTYPE) + 0.5) * dx
    g = torch.meshgrid(line, line, line, indexing="ij")
    pos = torch.stack((g[0].reshape(-1), g[1].reshape(-1), g[2].reshape(-1)), dim=-1)
    mass = torch.full((pos.shape[0],), RHO0 * dx ** 3, dtype=DTYPE)
    return pos, mass, n


def interior_mask_2d(n: int, cells: int) -> torch.Tensor:
    """Particles at least ``cells`` lattice spacings from every edge."""
    idx = torch.arange(n)
    keep = (idx >= cells) & (idx <= n - 1 - cells)
    m = keep[:, None] & keep[None, :]
    return m.reshape(-1)


def interior_mask_3d(n: int, cells: int) -> torch.Tensor:
    idx = torch.arange(n)
    keep = (idx >= cells) & (idx <= n - 1 - cells)
    m = (keep[:, None, None] & keep[None, :, None] & keep[None, None, :])
    return m.reshape(-1)


def pair_list_periodic(pos: torch.Tensor, cutoff: float, box: float,
                       chunk: int = 2048):
    """Unordered ``i < j`` pair list under the minimum-image convention (D-D)."""
    n = pos.shape[0]
    rows_i, rows_j = [], []
    for start in range(0, n, chunk):
        block = pos[start:start + chunk]
        delta = block[:, None, :] - pos[None, :, :]
        delta = delta - box * torch.round(delta / box)
        r = torch.linalg.vector_norm(delta, dim=-1)
        idx = torch.nonzero(r < cutoff, as_tuple=True)
        i = idx[0] + start
        j = idx[1]
        keep = i < j
        rows_i.append(i[keep])
        rows_j.append(j[keep])
    return torch.cat(rows_i), torch.cat(rows_j)


# --------------------------------------------------------------------------- #
# The implementations under test
# --------------------------------------------------------------------------- #

def op_tgv_morris(pos, mass, dx):
    """``vv.tgv.viscous_acceleration``: Morris Laplacian, library factor."""
    cfg = tgv.TGVConfig(dx=dx, h_factor=H_FACTOR, rho0=RHO0, U0=U0,
                        viscosity=NU, waves=WAVES, dtype="float64")
    vel = tgv_field(pos)
    return tgv.viscous_acceleration(pos, vel, mass, cfg)


def _droplet_cfg(dx: float, viscosity: float) -> Config:
    return Config(dx=dx, h_factor=H_FACTOR, rho0=RHO0, viscosity=viscosity,
                  sound_speed=SOUND_SPEED, gamma=GAMMA, gravity=0.0,
                  wall_friction=0.0, density_diffusion=0.0,
                  surface_tension_coefficient=0.0, surface_pressure_closure=False)


def op_droplet(rhs_fn, pos, mass, dx, vel, nu=NU):
    """Isolate the viscous term of a full RHS by differencing against ``nu = 0``.

    ``rhs`` is linear in ``cfg.viscosity`` (the pressure/density terms do not
    reference it), so ``acc(nu) - acc(0)`` is exactly the viscous acceleration --
    measured through the production code path rather than re-implemented here.
    """
    cfg_nu = _droplet_cfg(dx, nu)
    cfg_0 = _droplet_cfg(dx, 0.0)
    _, _, a_nu = rhs_fn(pos, vel, mass, cfg_nu)
    _, _, a_0 = rhs_fn(pos, vel, mass, cfg_0)
    return a_nu - a_0


def op_isph_morris(pos, mass, dx, vel):
    """``isph_channel.viscous_laplacian`` on a periodic pair geometry.

    ``isph_channel`` builds its own geometry dict; the operator only needs
    ``ii/jj/g/delta/r``, so the same periodic pair machinery is supplied here and
    the *operator itself* (hard-coded factor 2, ``eta`` regularisation) is what
    gets measured.
    """
    h = H_FACTOR * dx
    cutoff = 2.4 * h
    i, j = pair_list_periodic(pos, cutoff, BOX)
    delta = pos[i] - pos[j]
    delta = delta - BOX * torch.round(delta / BOX)
    _w, grad = pair_kernels(delta, h)
    r = torch.linalg.vector_norm(delta, dim=-1)
    geo = {"ii": i, "jj": j, "g": grad, "delta": delta, "r": r}
    rho = density_periodic(pos, mass, dx, h)
    volume = mass / rho
    # ``viscous_laplacian`` returns the *bare* Laplacian L[v] (the ISPH solver
    # multiplies it by nu itself, and reuses it for the pressure Poisson
    # operator), so the kinematic viscosity is applied here to put the result on
    # the same footing as the other implementations: acc = nu * L[v].
    return NU * isph_channel.viscous_laplacian(geo, vel, volume, ETA_FACTOR * h)


def density_periodic(pos, mass, dx, h):
    """Summation density on the periodic lattice (self term included)."""
    i, j = pair_list_periodic(pos, 2.0 * h, BOX)
    delta = pos[i] - pos[j]
    delta = delta - BOX * torch.round(delta / BOX)
    w, _ = pair_kernels(delta, h)
    zero = torch.zeros(1, dtype=DTYPE)
    w0 = float(pair_kernels(zero, h)[0])
    rho = mass * w0
    rho = rho.index_add(0, i, mass[j] * w)
    rho = rho.index_add(0, j, mass[i] * w)
    return rho


def op_threed_adapted(pos, mass, dx, vel, renormalise: bool, order: int):
    """Adapter for the actual ``operators3.viscous_acceleration`` signature."""
    h = H_FACTOR * dx
    pairs_i, pairs_j = pair_list_periodic(pos, 2.4 * h, BOX)
    pairs = torch.stack((pairs_i, pairs_j), dim=1)
    # Density from the 3-D summation operator itself, so the operator under test
    # is fed a rho produced by the matching kernel (not by the 2-D droplet one).
    rho = ops3.summation_density(pos, mass, h, pairs, box=BOX)
    return ops3.viscous_acceleration(pos, vel, mass, rho, NU, h, pairs,
                                     box=BOX, renormalise=renormalise,
                                     order=order)


# --------------------------------------------------------------------------- #
# Driver
# --------------------------------------------------------------------------- #

DX_2D = (0.05, 0.025, 0.0125)
DX_2D_DENSE = (0.05, 0.025)
DX_3D = (0.1, 0.05, 0.04)
INTERIOR_CELLS = 3          # 2h/dx = 2.6 spacings -> 3 cells guarantees full support


def sweep_2d() -> list[dict]:
    rows = []
    for dx in DX_2D:
        pos, mass, n = lattice2d(dx)
        vel = tgv_field(pos)
        lap = tgv_laplacian(pos)
        mask = interior_mask_2d(n, INTERIOR_CELLS)

        acc = op_tgv_morris(pos, mass, dx)
        rows.append({"impl": "tgv_morris", "dim": 2, "dx": dx, "n": n,
                     "family": "morris", **operator_readout(acc, lap, NU),
                     "interior": operator_readout(acc, lap, NU, mask)})

        acc = op_droplet(rhs, pos, mass, dx, vel)
        rows.append({"impl": "droplet_prod", "dim": 2, "dx": dx, "n": n,
                     "family": "morris",
                     "interior": operator_readout(acc, lap, NU, mask)})

        acc = op_isph_morris(pos, mass, dx, vel)
        rows.append({"impl": "isph_morris", "dim": 2, "dx": dx, "n": n,
                     "family": "morris", **operator_readout(acc, lap, NU),
                     "interior": operator_readout(acc, lap, NU, mask)})

        if dx in DX_2D_DENSE:
            acc = op_droplet(rhs_all_pairs, pos, mass, dx, vel)
            rows.append({"impl": "droplet_dense", "dim": 2, "dx": dx, "n": n,
                         "family": "morris",
                         "interior": operator_readout(acc, lap, NU, mask)})
    return rows


def sweep_3d() -> list[dict]:
    rows = []
    for dx in DX_3D:
        pos, mass, n = lattice3d(dx)
        vel = box3d_field(pos)
        lap = box3d_laplacian(pos)
        mask = interior_mask_3d(n, INTERIOR_CELLS)
        for key, renorm, order in (("threed_renorm2", True, 2),
                                   ("threed_renorm4", True, 4),
                                   ("threed_raw", False, 2)):
            acc = op_threed_adapted(pos, mass, dx, vel, renorm, order)
            rows.append({"impl": key, "dim": 3, "dx": dx, "n": n,
                         "family": "brookshaw",
                         **operator_readout(acc, lap, NU),
                         "interior": operator_readout(acc, lap, NU, mask)})
    return rows


def summarise(rows: list[dict]) -> dict:
    """Derived findings: spread inside each dimension family, per resolution."""
    out = {"morris_spread_pct": {}, "threed_spread_pct": {}, "headline": []}
    for dim, family, key in ((2, "morris", "morris_spread_pct"),
                             (3, "brookshaw", "threed_spread_pct")):
        for dx in sorted({r["dx"] for r in rows if r["dim"] == dim}):
            vals = [(r["impl"], r["interior"]["nu_eff_over_nu"])
                    for r in rows
                    if r["dim"] == dim and abs(r["dx"] - dx) < 1e-12]
            if not vals:
                continue
            lo = min(v for _, v in vals)
            hi = max(v for _, v in vals)
            # Annotate the rows themselves so the document table and the JSON
            # carry the same spread without recomputing it downstream.
            for r in rows:
                if r["dim"] == dim and abs(r["dx"] - dx) < 1e-12:
                    r["family_spread_pct"] = (hi - lo) / max(abs(lo), 1e-30) * 100.0
            out[key][f"{dx:.6g}"] = {
                "min": lo, "max": hi,
                "spread_pct": (hi - lo) / max(abs(lo), 1e-30) * 100.0,
                "values": {name: v for name, v in vals},
            }
    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", type=Path, default=OUT)
    ap.add_argument("--threads", type=int, default=1)
    args = ap.parse_args()
    torch.set_num_threads(args.threads)

    t0 = time.time()
    rows = sweep_2d() + sweep_3d()
    summary = summarise(rows)
    payload = {
        "benchmark": "viscous_operator_consistency",
        "criterion": ("nu_eff/nu measured per implementation on a shared analytic "
                      "field; spread inside a dimension family is the "
                      "cross-implementation discrepancy"),
        "parameters": {
            "nu": NU, "U0": U0, "h_factor": H_FACTOR, "rho0": RHO0,
            "sound_speed": SOUND_SPEED, "gamma": GAMMA, "box": BOX,
            "waves": WAVES, "k": K, "eta_factor": ETA_FACTOR,
            "library_VISCOUS_LAPLACIAN_FACTOR": VISCOUS_LAPLACIAN_FACTOR,
            "interior_cells": INTERIOR_CELLS,
            "dx_2d": list(DX_2D), "dx_2d_dense": list(DX_2D_DENSE),
            "dx_3d": list(DX_3D),
        },
        "rows": rows,
        "summary": summary,
        "wall_clock_s": time.time() - t0,
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(payload, indent=2) + "\n")
    print(f"wrote {args.out}  ({len(rows)} rows, {payload['wall_clock_s']:.1f} s)")
    for r in rows:
        inner = r["interior"]
        full = r.get("nu_eff_over_nu")
        print(f"  {r['impl']:<14} {r['dim']}D dx={r['dx']:<9.6g} n={r['n']:<4d} "
              f"nu_eff/nu(interior)={inner['nu_eff_over_nu']:.4f} "
              f"resid={inner['residual_pct']:.2f}%"
              + (f"  full={full:.4f}" if full is not None else ""))
    print(json.dumps(summary["morris_spread_pct"], indent=2))
    print(json.dumps(summary["threed_spread_pct"], indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())