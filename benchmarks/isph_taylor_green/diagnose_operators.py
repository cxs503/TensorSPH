#!/usr/bin/env python3
"""Decompose the 23.6 % Taylor-Green error into operator-level defects.

``docs/xval_isph_projection_taylor_green.md`` (m20500) left two open suspects for
the honest FAIL of the periodic projection ISPH run on the Taylor-Green vortex:

  (i)  the pressure Poisson operator ``L`` (pair / graph Laplacian) and the
       composition ``D G`` actually applied by the velocity correction are two
       *different* stencils, so a projection step only removes ~20 % of the
       discrete divergence (measured ``div_after/div_before`` = 0.79);
  (ii) the kinetic energy is over-dissipated (``E/E_exact`` = 0.755 at t = 0.5).

This driver measures each suspect in isolation, on the same lattice the
benchmark uses, so that a fix can be aimed at the dominant term instead of at
whichever one is easier to edit:

  A  viscous rate      -- effective decay rate of the TG Fourier mode under one
                          ``viscous_laplacian`` application vs the exact k^2.
  B  projection quality -- divergence annihilation of a *known* compressive field
                          with the shipped operator pair (``L`` + ``D``/``G``).
  C  projection quality -- the same with a Poisson operator built from the very
                          composition the correction applies (``A = D G``).
  D  energy cost        -- kinetic-energy removal per projection step for both.

Everything is deterministic and lands in ``results.json``; no number in the
accompanying report is typed by hand.

    PYTHONPATH=src python benchmarks/isph_taylor_green/diagnose_operators.py \
        --out outputs/isph_taylor_green
"""
from __future__ import annotations

import argparse
import json
import math
import platform
import sys
import time
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "src"))

import torch  # noqa: E402

from tensorsph.isph_periodic import (divergence, gradient, kernel_moments,  # noqa: E402
                                     laplacian, periodic_pairs, poisson,
                                     project, viscous_laplacian)

L_BOX = 1.0
U0 = 1.0
RE = 10.0
RHO = 1000.0


def lattice(L: float, dx: float) -> torch.Tensor:
    n = int(round(L / dx))
    line = (torch.arange(n, dtype=torch.float64) + 0.5) * dx
    gx, gy = torch.meshgrid(line, line, indexing="xy")
    return torch.stack((gx.reshape(-1), gy.reshape(-1)), dim=-1)


def exact_field(pos, t, *, L=L_BOX, U0=U0, Re=RE):
    k = 2.0 * math.pi / L
    nu = U0 / (Re * k)
    amp = U0 * math.exp(-2.0 * nu * k * k * t)
    return torch.stack((amp * torch.sin(k * pos[:, 0]) * torch.cos(k * pos[:, 1]),
                        -amp * torch.cos(k * pos[:, 0]) * torch.sin(k * pos[:, 1])), dim=-1), k, nu


def composed_poisson(pos, source, vol, pairs, h, box, *, rel_tol=1e-12, max_iter=2000,
                     moments=None):
    """CG on ``A p = source`` with ``A = D G`` -- the operator the correction uses.

    ``A`` is symmetric positive semi-definite (``D`` is minus the ``V``-weighted
    adjoint of ``G``) with *two* null directions on a collocated lattice: the
    constant mode and the checkerboard mode (``p_i = (-1)^{i+j}`` makes every
    symmetric neighbour contribution cancel).  Both are deflated by projecting
    the iterate and the residual onto their orthogonal complement; the
    checkerboard projector is built from the Hamiltonian path ``i -> i+1`` of the
    lattice index because the driver's particles are lattice-ordered.
    """
    if moments is None:
        moments = kernel_moments(pos, vol, pairs, h, box)

    def scal(q):
        return divergence(pos, gradient(pos, q, vol, pairs, h, box, moments),
                          vol, pairs, h, box, moments)

    n = pos.shape[0]

    def deflate(q):
        q = q - q.mean()
        # checkerboard vector, Gram-Schmidt orthonormalised against constants
        n_side = int(round(math.sqrt(n)))
        cb = torch.ones(n, dtype=q.dtype, device=q.device)
        if n_side * n_side == n:
            ix = torch.arange(n_side, device=q.device)
            sign = ((ix[:, None] + ix[None, :]) % 2).reshape(-1).to(q.dtype) * 2 - 1
            cb = sign
        cb = cb - cb.mean()
        cb = cb / torch.linalg.vector_norm(cb).clamp_min(1e-300)
        return q - (q * cb).sum() * cb

    b = deflate(source)
    p = torch.zeros_like(b)
    r = b.clone()
    d = r.clone()
    rr = (r * r).sum()
    norm0 = torch.linalg.vector_norm(b).clamp_min(1e-300)
    it = 0
    for it in range(1, max_iter + 1):
        ad = deflate(scal(d))
        den = (d * ad).sum()
        if float(den.abs()) <= 1e-300:
            break
        alpha = rr / den
        p = p + alpha * d
        r = r - alpha * ad
        if float(torch.linalg.vector_norm(r) / norm0) < rel_tol:
            break
        rr_new = (r * r).sum()
        d = r + (rr_new / rr) * d
        rr = rr_new
    return deflate(p), it


def case(dx: float, courant: float, t_end: float):
    k = 2.0 * math.pi / L_BOX
    nu = U0 / (RE * k)
    h = 1.3 * dx
    dt = courant * h / U0
    pos = lattice(L_BOX, dx)
    vol = torch.full((pos.shape[0],), (L_BOX * L_BOX) / pos.shape[0], dtype=torch.float64)
    pairs = periodic_pairs(pos, 2.0 * h, L_BOX)
    moments = kernel_moments(pos, vol, pairs, h, L_BOX)

    # ---- A: effective viscous decay rate -------------------------------------
    u_exact, _k, _nu = exact_field(pos, 0.0)
    du = viscous_laplacian(pos, u_exact, vol, pairs, h, L_BOX, nu)
    # du = -nu * k2_eff * u  ->  k2_eff = -(du . u) / (nu (u . u))
    k2_eff = -float((du * u_exact).sum()) / (nu * float((u_exact * u_exact).sum()))
    # the Taylor-Green velocity is sin(kx)cos(ky): laplacian(u) = -2 k^2 u, so the
    # exact eigenvalue of ``-laplacian`` on this field is 2 k^2 (not k^2) -- the
    # scalar check in tests/test_isph_periodic.py asserts exactly this.
    k2_exact = 2.0 * k * k
    rate_error_pct = (k2_eff / k2_exact - 1.0) * 100.0

    # ---- B/C: projection quality on a controlled compressive field ----------
    phi = torch.stack((torch.cos(k * pos[:, 0]) * torch.cos(k * pos[:, 1]),
                       torch.zeros_like(pos[:, 0])), dim=-1)
    u_star = u_exact + 0.05 * gradient(pos, phi[:, 0], vol, pairs, h, L_BOX, moments)
    div_star = divergence(pos, u_star, vol, pairs, h, L_BOX, moments)
    ke_star = float((u_star * u_star).sum())

    u_shipped, p_ship, info_ship = project(pos, u_star, vol, pairs, h, L_BOX, RHO, dt)
    ke_ship = float((u_shipped * u_shipped).sum())

    p_comp, its_comp = composed_poisson(pos, (RHO / dt) * div_star, vol, pairs, h, L_BOX,
                                        moments=moments)
    u_comp = u_star - (dt / RHO) * gradient(pos, p_comp, vol, pairs, h, L_BOX, moments)
    div_comp = divergence(pos, u_comp, vol, pairs, h, L_BOX, moments)
    ke_comp = float((u_comp * u_comp).sum())
    scale = float(torch.linalg.vector_norm(div_star))

    # ---- D: what the projection does to a *purely solenoidal* field ---------
    # u_exact is divergence-free (discrete divergence ~ machine zero), so a
    # consistent projection must be the identity here; any KE loss is spurious
    # dissipation manufactured by the operator mismatch.
    u_pure, _p, info_pure = project(pos, u_exact, vol, pairs, h, L_BOX, RHO, dt)
    ke_exact = float((u_exact * u_exact).sum())
    pure_kept = float((u_pure * u_pure).sum()) / ke_exact

    # and the same for the composed-operator projection (correct sign this time:
    # A = D G ~ +laplacian, so the source is +(rho/dt) div u*)
    p_pure, its_pure = composed_poisson(pos, (RHO / dt) * divergence(
        pos, u_exact, vol, pairs, h, L_BOX, moments), vol, pairs, h, L_BOX, moments=moments)
    u_pure_c = u_exact - (dt / RHO) * gradient(pos, p_pure, vol, pairs, h, L_BOX, moments)
    pure_kept_c = float((u_pure_c * u_pure_c).sum()) / ke_exact

    return {
        "dx": dx, "h": h, "courant": courant, "dt": dt, "particles": int(pos.shape[0]),
        "reynolds": RE, "nu": nu, "t_end": t_end,
        "viscous_rate": {
            "k2_exact": k2_exact, "k2_effective": k2_eff,
            "rate_error_pct": rate_error_pct,
            "extra_amplitude_decay_over_t_end_pct":
                100.0 * (1.0 - math.exp(-(k2_eff - k2_exact) * nu * t_end)),
        },
        "solenoidal_field": {
            "div_exact_rms": float(torch.sqrt((divergence(
                pos, u_exact, vol, pairs, h, L_BOX, moments) ** 2).mean())),
            "shipped_energy_kept": pure_kept,
            "shipped_div_after_over_before": info_pure["div_after_over_div_before"],
            "composed_energy_kept": pure_kept_c,
            "composed_cg_iterations": int(its_pure),
        },
        "projection": {
            "div_star_rms": float(torch.sqrt((div_star * div_star).mean())),
            "shipped": {
                "div_after_over_div_before":
                    float(torch.linalg.vector_norm(divergence(pos, u_shipped, vol, pairs,
                                                              h, L_BOX, moments)) / scale),
                "cg_iterations": info_ship["cg_iterations"],
                "energy_kept_fraction": ke_ship / ke_star,
                },
            "composed_operator": {
                "div_after_over_div_before": float(torch.linalg.vector_norm(div_comp) / scale),
                "cg_iterations": int(its_comp),
                "energy_kept_fraction": ke_comp / ke_star,
            },
        },
    }


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--out", default=str(REPO / "outputs" / "isph_taylor_green"))
    ap.add_argument("--configs", default="0.05:0.1")
    ap.add_argument("--threads", type=int, default=4)
    args = ap.parse_args(argv)
    torch.set_num_threads(args.threads)
    torch.set_default_dtype(torch.float64)

    t0 = time.time()
    rows = []
    for spec in args.configs.split(","):
        dx_s, co_s = spec.split(":")
        rows.append(case(float(dx_s), float(co_s), 0.5))
    doc = {
        "case": "CASE-SPH-PROJECTION-TAYLOR-GREEN-OPERATOR-DIAGNOSTICS",
        "anchor": "XVAL-SPH-METHOD-SELECTION-2026-11-12",
        "rows": rows,
        "wall_seconds": time.time() - t0,
        "environment": {"python": sys.version.split()[0], "torch": torch.__version__,
                        "platform": platform.platform()},
    }
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    target = out / "operator_diagnostics.json"
    target.write_text(json.dumps(doc, indent=2))
    for r in rows:
        v, pj = r["viscous_rate"], r["projection"]
        print(f"dx={r['dx']:g} Co={r['courant']:g}  k2_eff/k2-1={v['rate_error_pct']:.2f}%  "
              f"div_ratio shipped={pj['shipped']['div_after_over_div_before']:.3f} "
              f"composed={pj['composed_operator']['div_after_over_div_before']:.3f}  "
              f"KE kept shipped={pj['shipped']['energy_kept_fraction']:.4f} "
              f"composed={pj['composed_operator']['energy_kept_fraction']:.4f}")
        sf = r["solenoidal_field"]
        print(f"   solenoidal field (exact div rms {sf['div_exact_rms']:.2e}): "
              f"KE kept shipped={sf['shipped_energy_kept']:.6f} "
              f"composed={sf['composed_energy_kept']:.6f}")
    print(f"wrote {target}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())