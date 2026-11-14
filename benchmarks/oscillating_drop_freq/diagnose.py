"""Root-cause diagnostics for the axisymmetric drop-frequency estimator.

Round 1 of ``CASE-SPH-OSC-DROP-FREQ`` measured a modal stiffness that is (a) of
the wrong *sign* and (b) 2-6x too large, with a delta-weighted mean curvature
``<kappa> R = 1.2`` instead of the exact ``2``.  This script isolates the three
candidates one at a time on the *undeformed* (``eps = 0``) circle and on the
``l = 2`` drop, so the defect can be attributed to a single normalisation:

``volume``      ``sum_i dV_i / (4/3 pi R^3)`` should be 1 (cylindrical weights).
``coarea2d``    ``sum_i delta_s,i dA_i / (2 pi R)`` -- the plane-curve coarea
                identity that the 2-D CSF operator is calibrated for.
``coarea3d``    ``sum_i delta_s,i dV_i / (4 pi R^2)`` -- the same identity after
                the axisymmetric weight.  For an integrand that is 1 per particle
                the full-circle map double-covers the meridian once, so the
                surface integral of revolution is ``2 * sum delta_s dV``; the
                script reports both the raw ratio and the ``2x`` corrected one.
``kappa``       ``kappa_meridional`` and the azimuthal term ``n_rho/rho`` summed
                separately, delta-weighted, on the exact circle (both should give
                ``1/R`` so their sum is the sphere ``2/R``).

Usage::

    PYTHONPATH=src python benchmarks/oscillating_drop_freq/diagnose.py
"""
from __future__ import annotations

import json
import math
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))

from tensorsph.oscillating_drop3d import (  # noqa: E402
    axisym_curvature,
    axisym_drop_geometry,
    legendre_p,
)
from tensorsph.surface_tension_dispersion import csf_state  # noqa: E402

R = 0.02
SIGMA = 7.2
RHO = 1000.0
OUT = Path(__file__).resolve().parent / "diagnose.json"


def measure(rdx: float, mode: int, eps: float, delta_normalization: str = "legacy",
            curvature_scheme: str = "fit_quadratic") -> dict:
    dx = R / rdx
    geom = axisym_drop_geometry(R, dx, mode=mode, eps=eps, rho0=RHO)
    state, h, _ = csf_state(geom, SIGMA, periodic_x=False,
                            delta_normalization=delta_normalization,
                            curvature_scheme=curvature_scheme)
    dvol = geom["dvol"]
    dA = geom["mass"] / RHO                       # plane area per particle
    ds = state.delta_s
    share = legendre_p(mode, geom["cos_theta"])
    kappa_m = state.kappa                         # meridional plane-curve curvature
    sign = torch.where(geom["pos"][:, 0] >= 0.0, torch.ones_like(geom["rho"]),
                       -torch.ones_like(geom["rho"]))
    kappa_az = (state.normal[:, 0] * sign) / geom["rho_safe"]
    kappa = axisym_curvature(state, geom)
    w = ds * dvol
    w2 = ds * dA
    out = {
        "radius_over_dx": rdx, "mode": mode, "eps": eps, "n": int(geom["pos"].shape[0]),
        "h_over_dx": h / dx,
        "delta_normalization": delta_normalization,
        "curvature_scheme": curvature_scheme,
        "volume_ratio": float(dvol.sum() / (4.0 / 3.0 * math.pi * R ** 3)),
        "coarea2d_ratio": float(w2.sum() / (2.0 * math.pi * R)),
        "coarea3d_ratio_raw": float(w.sum() / (4.0 * math.pi * R ** 2)),
        "coarea3d_ratio_x2": float(2.0 * w.sum() / (4.0 * math.pi * R ** 2)),
        "kappaR_meridional": float((kappa_m * w).sum() / w.sum()) * R,
        "kappaR_azimuthal": float((kappa_az * w).sum() / w.sum()) * R,
        "kappaR_total": float((kappa * w).sum() / w.sum()) * R,
        "kappaR_total_unweighted": float(kappa.mean()) * R,
        "delta_weight_share_interior": float(
            (ds > 0.02 * float(ds.max())).float().mean()),
    }
    return out


def main() -> int:
    rows = []
    for scheme in ("fit_quadratic", "mls"):
        for rdx in (8.0, 12.0, 16.0, 20.0):
            for mode, eps in ((2, 0.0), (2, 0.05)):
                row = measure(rdx, mode, eps, curvature_scheme=scheme)
                rows.append(row)
                print(f"{scheme:14s} R/dx={rdx:4.1f} l={mode} eps={eps:g}: "
                      f"Vol={row['volume_ratio']:.4f} "
                      f"coarea2d={row['coarea2d_ratio']:.4f} "
                      f"coarea3d(x2)={row['coarea3d_ratio_x2']:.4f} "
                      f"kR_m={row['kappaR_meridional']:+.4f} "
                      f"kR_az={row['kappaR_azimuthal']:+.4f} "
                      f"kR={row['kappaR_total']:+.4f}", flush=True)
    OUT.write_text(json.dumps({"rows": rows, "R": R, "sigma": SIGMA, "rho": RHO},
                              indent=2))
    print(f"wrote {OUT}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())