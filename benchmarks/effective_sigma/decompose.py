"""Decomposition of the effective-surface-tension deficit (diagnostic probe).

Every quantity is a static, closed-form-checkable integral of the CSF operator
on a circular droplet:

``int delta_s dV``              -> interface area ``2 pi R``  (unit depth)
``int kappa delta_s dV``        -> ``kappa A_s = 2 pi``       (if kappa = 1/R)
``|int kappa delta_s n dV|``    -> ``2 sigma``-equivalent    (force balance)

Run: ``python benchmarks/effective_sigma/decompose.py``
"""
from __future__ import annotations

import importlib.util
import math
import sys
from pathlib import Path

import torch

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
sys.path.insert(0, str(ROOT / "src"))

from tensorsph import Config                      # noqa: E402
from tensorsph.csf_consistent import csf_fields    # noqa: E402
from tensorsph.neighbor_search import cell_linked_pairs  # noqa: E402

_spec = importlib.util.spec_from_file_location(
    "laplace_benchmark", HERE.parent / "laplace_pressure" / "run.py")
laplace = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(laplace)

SIGMA = 0.072
RHO0 = 1000.0


def decompose(dx: float, curv: str, delta_norm: str = "legacy",
              radius: float = 0.04) -> dict:
    cfg = Config(dx=dx, rho0=RHO0, sound_speed=20.0, gravity=0.0,
                 surface_tension_coefficient=SIGMA, csf_curvature=curv)
    h = cfg.h_factor * cfg.dx
    pos, mass, density = laplace.circular_droplet(radius, dx)
    pairs = cell_linked_pairs(pos, 2.0 * cfg.h_factor * dx)
    stencil = 4.0
    dpairs = (cell_linked_pairs(pos, stencil * h) if curv != "mls" else pairs)
    st = csf_fields(pos, mass, dpairs, h, RHO0, curvature_scheme=curv,
                    curvature_stencil=stencil, delta_normalization=delta_norm)
    vol = mass / RHO0
    ds = st.delta_s.clone()
    mag = torch.linalg.vector_norm(st.grad_c, dim=1)
    ds[mag <= 1.0e-3 * mag.max()] = 0.0
    area = 2.0 * math.pi * radius
    s_int = float((vol * ds).sum())
    k_int = float((vol * ds * st.kappa).sum())
    fvec = (vol * ds * st.kappa)[:, None] * st.normal
    right = pos[:, 0] > 0.0
    f_right = float(torch.linalg.vector_norm(fvec[right].sum(dim=0)))
    iface = ds > 1.0e-3 * ds.max()
    return {
        "dx_mm": dx * 1e3,
        "curvature": curv,
        "delta_norm": delta_norm,
        "n_particles": int(pos.shape[0]),
        "c_bulk_max": float(st.color.max()),
        "c_iface_mean": float(st.color[iface].mean()),
        "int_delta_s_over_area": s_int / area,
        "int_kappa_delta_s_over_2pi": k_int / (2.0 * math.pi),
        "vector_over_2pi": float(torch.linalg.vector_norm(fvec.sum(dim=0))) / (2.0 * math.pi),
        "kappa_mean_iface": float(st.kappa[iface].mean()),
        "one_over_R": 1.0 / radius,
        "delta_s_mean": float(ds[iface].mean()),
        "one_over_h": 1.0 / h,
        "force_ratio_right_half": SIGMA * f_right / (2.0 * SIGMA),
    }


def main() -> None:
    for norm in ("legacy", "local_range"):
        for curv in ("fit_quadratic", "mls"):
            for dx in (0.005, 0.0025):
                r = decompose(dx, curv, norm)
                print(" ".join(f"{k}={v:.4f}" if isinstance(v, float) else f"{k}={v}"
                               for k, v in r.items()))


if __name__ == "__main__":
    main()
