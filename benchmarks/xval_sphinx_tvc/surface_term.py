"""Analytic anchor for the *free-surface* branch of the TVC operator.

Why this file exists
--------------------
The transport-velocity-correction operator ported from SPHinXsys
(``docs/xval_sphinx_tvc/oss_source/transport_velocity_correction.hpp``) builds

    g_i = -sum_j (C_i + C_j) V_j grad_i W_ij          (upstream ``inconsistency``)
    dr_i = alpha h_ref^2 * g_i                        (upstream ``update``)

Every verification leg so far (``docs/verification_transport_velocity.md``,
``docs/xval_sphinx_tvc/DISTRIBUTION_LADDER.md``) tested the *bulk* behaviour:
a perfect lattice is a fixed point, so ``g_i = 0`` there and the sum has no
non-zero reference to be graded against.  The free-surface branch -- the part
that actually moves particles and bounds stability -- was only ever asserted
by an *invariant* ("no normal component after projection"), never against a
number.  This script supplies the number.

Analytic anchor
---------------
The continuum limit of the same discrete sum is exact by the gradient theorem.
For a particle at depth ``d`` below a plane free surface with outward normal
``n`` (fluid below the plane)::

    int_fluid  grad_i W(x_i - x') dV' = -n * I(h, d),
    I(h, d) = int_{-inf}^{inf} W_2D(s, d) ds          (line integral, 1/length)

so with ``C = 1`` (``NoKernelCorrection``) and ``V_j = dx^2`` the lattice sum
must converge to

    g_i -> +2 I(h, d) n .

``I`` has a closed form for the 2-D cubic spline; it is evaluated here by
quadrature over the explicit profile (and the profile normalisation
``int W dA = 1`` is itself re-checked by quadrature, so the anchor is not taken
on faith).

Consequence that the anchor makes quantitative: ``I ~ 1/h`` in 2-D, hence

    |dr| / h = 2 alpha h |g| = 2 alpha h I(h, 0) = 2 alpha * const

i.e. the *per-application* surface shift is **resolution independent in units
of h** and therefore grows linearly in ``dx`` -- the mechanism behind the
fixed-``alpha`` degradation measured in ``RESOLUTION_LADDER.md`` section 3,
here derived from the operator's own continuum limit instead of fitted.

Usage
-----
    python benchmarks/xval_sphinx_tvc/surface_term.py --out docs/xval_sphinx_tvc/data
"""

from __future__ import annotations

import argparse
import json
import math
import platform
import subprocess
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))

from tensorsph.neighbor_search import cell_linked_pairs  # noqa: E402
from tensorsph.transport_velocity_correction import (  # noqa: E402
    kernel_gradient_integral,
)

HBAR = 1.3  # h / dx of every production run in this repo


# --------------------------------------------------------------------------- #
# kernel profile and its analytic line integral
# --------------------------------------------------------------------------- #
def kernel_2d(r: torch.Tensor, h: float) -> torch.Tensor:
    """2-D cubic spline (M4) kernel, ``int W dA = 1``."""
    q = r / h
    sigma = 10.0 / (7.0 * math.pi * h * h)
    profile = torch.where(
        q < 1.0,
        1.0 - 1.5 * q.square() + 0.75 * q.pow(3),
        torch.where(q < 2.0, 0.25 * (2.0 - q).pow(3), torch.zeros_like(q)),
    )
    return sigma * profile


def line_integral(h: float, depth: float, n: int = 40001) -> float:
    """``I(h, d) = int W_2D(s, d) ds`` over the full line, by quadrature."""
    span = 2.0 * h * math.sqrt(max(0.0, 1.0 - (depth / (2.0 * h)) ** 2))
    if span <= 0.0:
        return 0.0
    s = torch.linspace(-span, span, n, dtype=torch.float64)
    w = kernel_2d(torch.sqrt(s.square() + depth * depth), h)
    # trapezoid, exact to double precision for this smooth integrand
    return float((w[1:] + w[:-1]).sum() * 0.5 * float(s[1] - s[0]))


def check_normalisation(h: float = 1.0, n: int = 4001) -> float:
    """``int_0^{2h} W(r) 2 pi r dr`` -- must be 1."""
    r = torch.linspace(0.0, 2.0 * h, n, dtype=torch.float64)
    w = kernel_2d(r, h) * 2.0 * math.pi * r
    return float((w[1:] + w[:-1]).sum() * 0.5 * float(r[1] - r[0]))


def closed_form_I0(h: float) -> float:
    """``I(h, 0) = 15 / (7 pi h)`` (closed form, used as a second check)."""
    return 15.0 / (7.0 * math.pi * h)


# --------------------------------------------------------------------------- #
# lattice configuration: half-space of fluid under a plane free surface
# --------------------------------------------------------------------------- #
def truncated_lattice(h: float, h_over_dx: float, depth_steps: int):
    """Square lattice of fluid particles on ``x2 <= 0``; target at ``(0, -m dx)``.

    The plane ``x2 = 0`` is the free surface and the fluid is the half-space
    below it, so particles *above* the target (between the plane and it) are
    part of the fluid and stay in the sum.  The target depth is quoted in
    lattice steps ``m = depth / dx`` so it never lands arbitrarily close to a
    neighbour (which would make the kernel gradient dominated by one pair).
    The target is particle 0 and is excluded from the lattice list.
    """
    dx = h / h_over_dx
    reach = 2.0 * h + 2.0 * dx
    depth = depth_steps * dx
    if depth > 2.0 * h + 1.0e-12:
        raise ValueError("depth beyond kernel support")
    # the box must contain the *full support* of the target: from the plane
    # (x2 = 0) down to depth + 2h.  A shorter box would truncate the sum at the
    # bottom and fake a surface term where the fluid is homogeneous.
    k_max = int(math.ceil((depth + reach) / dx)) + 1
    m_max = int(math.ceil(reach / dx)) + 1
    coords = []
    for k in range(0, k_max + 1):  # rows x2 = -k dx, k = 0 is the surface row
        for m in range(-m_max, m_max + 1):
            x1, x2 = m * dx, -k * dx
            if min(x1, -x2) > reach:
                continue
            coords.append((x1, x2))
    target = torch.tensor([[0.0, -depth]], dtype=torch.float64)
    lattice = torch.tensor(coords, dtype=torch.float64)
    keep = torch.linalg.vector_norm(lattice - target, dim=1) > 1.0e-9
    positions = torch.cat([target, lattice[keep]])
    volume = torch.full((positions.shape[0],), dx * dx, dtype=torch.float64)
    return positions, volume, dx, depth


def target_gradient_dense(positions: torch.Tensor, volume: torch.Tensor,
                          h: float) -> torch.Tensor:
    """Independent brute-force evaluation of ``g_0`` (no linked cells)."""
    delta = positions[0] - positions[1:]
    r = torch.linalg.vector_norm(delta, dim=1)
    inside = (r < 2.0 * h) & (r > 1.0e-12)
    q = r[inside] / h
    sigma = 10.0 / (7.0 * math.pi * h * h)
    dshape = torch.where(
        q < 1.0,
        -3.0 * q + 2.25 * q.square(),
        -0.75 * (2.0 - q).square(),
    )
    grad_w = delta[inside] * (sigma * dshape / (h * q * h))[:, None]
    # upstream: g = -sum_j (C_i + C_j) V_j grad_i W_ij, C = 1
    return -2.0 * (volume[1:][inside, None] * grad_w).sum(dim=0)



def wall_image_case(h: float, h_over_dx: float, wall_factor: float = 2.0):
    """Wall-contact branch against an **exact** anchor (mirror ghost images).

    A fluid lattice sits on ``x2 >= 0`` in rows ``x2 = (k + 1/2) dx`` and the
    wall is its exact mirror image through ``x2 = 0`` (rows ``-(k + 1/2) dx``),
    i.e. the ghost-fluid picture of Adami et al. (2012).  No fluid node
    coincides with a wall node, so no pair is singular.

    Upstream weights a fluid pair by ``C_i + C_j`` and a wall pair by
    ``wall_factor * C_i``; with ``NoKernelCorrection`` (all ``C = 1``) both
    weights are 2 only when ``wall_factor = 2``.  Then fluid + wall is the
    *full-space* lattice sum about the target, which vanishes exactly.  So

        |g| = 0   to round-off,  if and only if  wall_factor = 2,

    which is an exact anchor -- no quadrature, no tolerance judgement.  The
    wall is deliberately under-resolved in nothing: it is the same lattice.
    """
    dx = h / h_over_dx
    reach = 2.0 * h + 2.0 * dx
    n_side = int(math.ceil(reach / dx)) + 1
    rows = int(math.ceil(reach / dx)) + 1
    fluid, wall = [], []
    for k in range(0, rows + 1):
        x2f = (k + 0.5) * dx
        for m in range(-n_side, n_side + 1):
            x1 = m * dx
            if math.hypot(x1, x2f) <= reach:
                fluid.append((x1, x2f))
            if math.hypot(x1, x2f) <= reach:
                wall.append((x1, -x2f))
    fluid = torch.tensor(fluid, dtype=torch.float64)
    wall = torch.tensor(wall, dtype=torch.float64)
    target = torch.tensor([[0.0, 0.5 * dx]], dtype=torch.float64)
    keep = torch.linalg.vector_norm(fluid - target, dim=1) > 1.0e-9
    fluid = torch.cat([target, fluid[keep]])
    positions = torch.cat([fluid, wall])
    volume = torch.full((positions.shape[0],), dx * dx, dtype=torch.float64)
    pairs = cell_linked_pairs(positions, 2.0 * h)
    g = kernel_gradient_integral(positions, volume, h, pairs=pairs,
                                 wall_count=wall.shape[0],
                                 wall_factor=wall_factor)[0]
    return {
        "h": h,
        "h_over_dx": h_over_dx,
        "dx": dx,
        "wall_factor": wall_factor,
        "n_fluid": int(fluid.shape[0]),
        "n_wall": int(wall.shape[0]),
        "g_mag": float(torch.linalg.vector_norm(g)),
        "g_x2": float(g[1]),
        "surface_scale": 2.0 * line_integral(h, 0.0),
    }


def measure(h: float, h_over_dx: float, depth_steps: int) -> dict:
    positions, volume, dx, depth = truncated_lattice(h, h_over_dx, depth_steps)
    pairs = cell_linked_pairs(positions, 2.0 * h)
    g = kernel_gradient_integral(positions, volume, h, pairs=pairs)[0]
    g_dense = target_gradient_dense(positions, volume, h)
    anchor = torch.tensor([0.0, 2.0 * line_integral(h, depth)], dtype=torch.float64)
    mag = float(torch.linalg.vector_norm(g))
    anchor_mag = float(torch.linalg.vector_norm(anchor))
    return {
        "h": h,
        "h_over_dx": h_over_dx,
        "dx": dx,
        "depth_steps": depth_steps,
        "depth_over_h": depth / h,
        "n_particles": int(positions.shape[0]),
        "n_neighbours_2h": int((torch.linalg.vector_norm(
            positions[0] - positions, dim=1) < 2.0 * h).sum()) - 1,
        "g_x1": float(g[0]),
        "g_x2": float(g[1]),
        "g_mag": mag,
        "anchor_g_x2": float(anchor[1]),
        "rel_err": (abs(mag - anchor_mag) / anchor_mag
                    if anchor_mag > 1.0e-12 * max(mag, 1.0e-300) else 0.0),
        "anchor_mag": anchor_mag,
        "dense_vs_linked_rel": float(torch.linalg.vector_norm(g - g_dense))
        / max(mag, 1.0e-300),
        "shift_over_h_alpha_0p1": 0.1 * h * mag,  # |dr| = alpha h^2 |g| -> /h
    }


# --------------------------------------------------------------------------- #
# driver
# --------------------------------------------------------------------------- #
def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--h", type=float, default=0.026, help="smoothing length [m]")
    ap.add_argument("--out", type=Path,
                    default=Path("docs/xval_sphinx_tvc/data"))
    ap.add_argument("--figure", type=Path,
                    default=Path("docs/xval_sphinx_tvc/figure_surface_anchor.png"))
    args = ap.parse_args()

    errors = [check_normalisation(args.h)]
    i0 = line_integral(args.h, 0.0)
    closed = closed_form_I0(args.h)
    print(f"kernel normalisation        : {errors[0]:.12f}  (target 1)")
    print(f"I(h,0) quadrature           : {i0:.10e}")
    print(f"I(h,0) closed form 15/(7pi h): {closed:.10e}  rel diff {abs(i0-closed)/closed:.3e}")

    # (a) lattice convergence at fixed h (dx -> 0): does the sum converge to
    #     the continuum value, and at what order?
    ladder = []
    for n in (HBAR, 2.0, 4.0, 8.0, 16.0, 32.0):
        row = measure(args.h, n, 0)
        row["label"] = f"h/dx={n:g}"
        ladder.append(row)
        print(f"surface, h/dx={n:5g}: g2={row['g_x2']:.6e} "
              f"anchor={row['anchor_g_x2']:.6e} rel_err={100*row['rel_err']:6.3f}% "
              f"g1={row['g_x1']:+.2e} dense_diff={row['dense_vs_linked_rel']:.1e}")

    # (b) depth decay: the operator's spatial structure against the analytic I.
    #     Depth is an exact multiple of dx, so the coarse lattice (h/dx = 1.3)
    #     can only sample m = 0, 1 (2h = 2.6 dx) -- that quantisation is itself
    #     a reported limit, not something to smooth over.
    depth_rows = []
    for n, steps in ((HBAR, (0, 1, 2)), (8.0, (0, 1, 2, 4, 8, 12, 16))):
        for m in steps:
            row = measure(args.h, n, m)
            depth_rows.append(row)
            print(f"depth={row['depth_over_h']:5.2f}h (m={m:2d}) h/dx={n:4g}: "
                  f"|g|={row['g_mag']:.5e} anchor={row['anchor_mag']:.5e} "
                  f"rel_err={100*row['rel_err']:7.3f}%")

    # (c) wall branch: exact mirror-image anchor (|g| must vanish iff factor=2)
    wall_rows = []
    for n in (HBAR, 4.0, 8.0):
        for wf in (2.0, 1.0):
            row = wall_image_case(args.h, n, wf)
            wall_rows.append(row)
            print(f"wall  h/dx={n:4g} wall_factor={wf:3.1f}: |g|={row['g_mag']:.3e} "
                  f"({row['g_mag']/row['surface_scale']:.2e} x surface scale)")

    production = next(r for r in ladder if r["h_over_dx"] == HBAR)
    rate = 2.0 * production["shift_over_h_alpha_0p1"]
    payload = {
        "case": "plane free surface, square lattice, single TVC application",
        "operator": "tensorsph.transport_velocity_correction."
                    "kernel_gradient_integral (upstream TransportVelocityCorrection)",
        "anchor": "gradient theorem: g -> +2 * n * I(h,d), "
                  "I = int W_2D(s,d) ds; I(h,0) = 15/(7 pi h)",
        "h": args.h,
        "production_h_over_dx": HBAR,
        "kernel_normalisation": errors[0],
        "I0_quadrature": i0,
        "I0_closed_form": closed,
        "I0_rel_diff": abs(i0 - closed) / closed,
        "ladder_fixed_h": ladder,
        "depth_ladder": depth_rows,
        "wall_ladder": wall_rows,
        "summary": {
            "production_rel_err": production["rel_err"],
            "finest_ladder_rel_err": ladder[-1]["rel_err"],
            "loglog_slope": math.log(ladder[0]["rel_err"] / ladder[-1]["rel_err"])
            / math.log(ladder[-1]["h_over_dx"] / ladder[0]["h_over_dx"]),
            "shift_over_h_alpha_0p1_measured": production["shift_over_h_alpha_0p1"],
            "shift_over_h_alpha_0p1_analytic": 0.1 * 2.0 * closed * args.h,
            "rung_over_h_alpha_0p1": rate,
            "wall_factor_2_max_abs": max(r["g_mag"] for r in wall_rows
                                        if r["wall_factor"] == 2.0),
            "wall_factor_1_min_abs": min(r["g_mag"] for r in wall_rows
                                        if r["wall_factor"] == 1.0),
            "wall_exact_anchor_ratio": max(r["g_mag"] for r in wall_rows
                                           if r["wall_factor"] == 2.0)
            / max(r["surface_scale"] for r in wall_rows),
        },
        "environment": {
            "python": platform.python_version(),
            "torch": torch.__version__,
            "git_head": subprocess.run(["git", "rev-parse", "HEAD"], text=True,
                                       capture_output=True).stdout.strip(),
        },
    }
    args.out.mkdir(parents=True, exist_ok=True)
    path = args.out / "surface_anchor.json"
    path.write_text(json.dumps(payload, indent=2))
    print(f"\nwrote {path}")

    s = payload["summary"]
    print(f"production h/dx=1.3 rel err : {100*s['production_rel_err']:.3f} %")
    print(f"ladder slope (h fixed)      : {s['loglog_slope']:.3f}")
    print(f"shift/h at alpha=0.1        : measured {s['shift_over_h_alpha_0p1_measured']:.4f}"
          f"  analytic {s['shift_over_h_alpha_0p1_analytic']:.4f}")

    make_figure(payload, args.figure)
    return 0


def make_figure(payload: dict, path: Path) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    ladder = payload["ladder_fixed_h"]
    n = [r["h_over_dx"] for r in ladder]
    err = [100.0 * r["rel_err"] for r in ladder]
    dep_fine = [r for r in payload["depth_ladder"] if r["h_over_dx"] == 8.0]
    dep_coarse = [r for r in payload["depth_ladder"] if r["h_over_dx"] == HBAR]

    fig, ax = plt.subplots(1, 3, figsize=(15.0, 4.2))
    ax[0].loglog(n, err, "o-", label="discrete lattice sum vs continuum anchor")
    ax[0].axhline(15.0, color="tab:red", ls="--", lw=1, label="owner gate 15 %")
    ax[0].axhline(3.0, color="tab:orange", ls=":", lw=1, label="strict 3 %")
    ax[0].axvline(HBAR, color="0.4", ls="-.", lw=1, label=f"production h/dx={HBAR}")
    ax[0].set_xlabel("h / dx")
    ax[0].set_ylabel("relative error of free-surface term [%]")
    ax[0].set_title("(a) plane free surface: lattice sum vs gradient theorem")
    ax[0].legend(fontsize=8, loc="lower left")
    ax[0].grid(alpha=0.3, which="both")

    grid = [i / 200.0 for i in range(0, 401)]
    ax[1].plot(grid, [line_integral(payload["h"], g * payload["h"]) / line_integral(payload["h"], 0.0)
                      for g in grid], "k--", lw=1.2, label="analytic I(h,d)/I(h,0)")
    for rows, style, lab in ((dep_coarse, "s-", "discrete h/dx=1.3 (production)"),
                             (dep_fine, "o-", "discrete h/dx=8")):
        ax[1].plot([r["depth_over_h"] for r in rows],
                   [r["g_mag"] / payload["ladder_fixed_h"][0]["g_mag"] for r in rows],
                   style, ms=4, label=lab)
    ax[1].set_xlabel("depth below free surface, d / h")
    ax[1].set_ylabel(r"$|g(d)| \,/\, |g(0)|$")
    ax[1].set_title("(b) depth decay of the free-surface term")
    ax[1].legend(fontsize=8)
    ax[1].grid(alpha=0.3)
    wall = payload["wall_ladder"]
    levels = sorted({r["h_over_dx"] for r in wall})
    for k, (wf, lab) in enumerate([(2.0, "wall_factor = 2 (upstream, ghost-fluid)"),
                                   (1.0, "wall_factor = 1 (control: wrong weighting)")]):
        vals = [next(r["g_mag"] for r in wall
                     if r["wall_factor"] == wf and r["h_over_dx"] == n)
                for n in levels]
        vals = [max(v / next(r["surface_scale"] for r in wall), 1e-18) for v in vals]
        ax[2].bar([i + (k - 0.5) * 0.36 for i in range(len(levels))], vals,
                  width=0.34, label=lab)
    ax[2].set_yscale("log")
    ax[2].set_xticks(range(len(levels)))
    ax[2].set_xticklabels([f"h/dx={n:g}" for n in levels])
    ax[2].axhline(1e-12, color="tab:red", ls="--", lw=1, label="round-off floor 1e-12")
    ax[2].set_ylabel("|g| / (2 I(h,0))   [log]")
    ax[2].set_title("(c) wall branch, exact mirror-ghost anchor")
    ax[2].legend(fontsize=7, loc="lower right")
    fig.suptitle("TVC free-surface branch vs analytic anchor "
                 "(upstream TransportVelocityCorrection, C=NoKernelCorrection)", fontsize=10)
    fig.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=150)
    print(f"wrote {path}")


if __name__ == "__main__":
    raise SystemExit(main())