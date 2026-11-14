"""Probe: what does the shipped 3-D CSF curvature estimator return on a lattice sphere?

Diagnostic only.  Measures, for a static sphere of radius ``R`` on a lattice of
spacing ``dx``:

    kappa_mls   the shipped ``csf_fields3(curvature="mls")`` (3-D ``div n``)
    int_delta   sum_i delta_s,i V_i / (4 pi R^2)        (must be 1)
    normal_err  mean angle between n_i and the exact radial direction

so the ``kappa R = 0.77`` figure quoted in ``oscillating_drop3d`` can be traced
to either the curvature operator or the delta function.
"""
from __future__ import annotations

import math
import sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))

from tensorsph.threed.csf3 import csf_fields3            # noqa: E402
from tensorsph.threed.kernel3 import cubic_spline_kernel  # noqa: E402
from tensorsph.threed.neighbors3 import box_pairs        # noqa: E402

torch.set_num_threads(4)


def sphere_block(radius: float, dx: float, outer: float, rho0: float = 1000.0):
    """Simple-cubic lattice over a cube, with the sphere marked as liquid.

    ``V_i = dx^3`` is the exact cell volume of the lattice, so the rest volume
    used by the CSF fields equals the geometric one.
    """
    half = radius + outer
    n = int(round(2.0 * half / dx))
    axis = (torch.arange(n, dtype=torch.float64) + 0.5) * dx - half
    g = torch.stack(torch.meshgrid(axis, axis, axis, indexing="ij"), dim=-1)
    pos = g.reshape(-1, 3).contiguous()
    liquid = torch.linalg.vector_norm(pos, dim=1) < radius
    mass = torch.full((pos.shape[0],), rho0 * dx ** 3, dtype=torch.float64)
    return pos, mass, liquid


def case(radius=0.02, dx=0.0025, outer=0.012, h_factor=1.3, rho0=1000.0):
    pos, mass, liquid = sphere_block(radius, dx, outer, rho0)
    h = h_factor * dx
    pairs = box_pairs(pos, 2.0 * h)
    st = csf_fields3(pos, mass, h, pairs, rho0=rho0)
    vol = mass / rho0
    dsv = st.delta_s * vol
    r = torch.linalg.vector_norm(pos, dim=1)
    band = st.surface_mask(0.05)
    rhat = pos / r.clamp_min(1e-30).unsqueeze(-1)
    cosang = (st.normal * rhat).sum(dim=1).clamp(-1.0, 1.0)
    surface = 4.0 * math.pi * radius ** 2
    kappa_w = float((st.kappa * dsv).sum() / dsv.sum())
    return {
        "dx_m": dx, "R_over_dx": radius / dx, "n_particles": int(pos.shape[0]),
        "n_liquid": int(liquid.sum()), "n_pairs": int(pairs.shape[0]),
        "n_band": int(band.sum()),
        "w0_V": float(vol[0] * cubic_spline_kernel(torch.zeros((), dtype=torch.float64), h)),
        "int_delta_over_4piR2": float(dsv.sum() / surface),
        "kappa_analytic_2_over_R": 2.0 / radius,
        "kappa_delta_weighted": kappa_w,
        "kappaR_delta_weighted": kappa_w * radius,
        "kappaR_band_mean": float(st.kappa[band].mean() * radius),
        "normal_angle_mean_deg": float(torch.rad2deg(torch.acos(cosang[band])).mean()),
        "delta_max": float(st.delta_s.max()),
        "delta_over_1_over_2h_median_band": float(
            (st.delta_s[band] * 2.0 * h).median()),
    }


if __name__ == "__main__":
    for dx in (0.0025, 0.0016):
        out = case(dx=dx)
        print({k: (round(v, 5) if isinstance(v, float) else v) for k, v in out.items()})