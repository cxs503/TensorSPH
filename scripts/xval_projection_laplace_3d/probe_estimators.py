"""Probe: score the 3-D curvature estimators on a static lattice sphere.

Compares the shipped 3-D ``div n`` (``curvature_method="mls"``) against the new
3-D local quadratic surface fit (``fit3``) for several fit slabs / stencils, and
reports the discrete surface measure ``int delta_s dV / (4 pi R^2)``.
"""
from __future__ import annotations

import math
import sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))

from tensorsph.laplace_isph3d import csf_state3_two_phase, sphere_block  # noqa

torch.set_num_threads(4)


def score(radius=0.02, dx=0.0025, outer=0.012, h_factor=1.3, rho0=1000.0,
          bands=(0.4, 0.6, 1.0), stencils=(2.0, 2.5)):
    h = h_factor * dx
    pos, mass, liquid = sphere_block(radius, dx, outer, rho0)
    out = {"dx_m": dx, "R_over_dx": radius / dx, "n_particles": int(pos.shape[0]),
           "variants": {}}
    base = csf_state3_two_phase(pos, mass, liquid, h, rho0, 0.072,
                               curvature_method="mls")
    vol, dsv = base["vol"], base["delta_s"] * base["vol"]
    r = torch.linalg.vector_norm(pos, dim=1)
    rhat = pos / r.clamp_min(1e-30).unsqueeze(-1)
    surface = 4.0 * math.pi * radius ** 2
    cosang = (base["normal"] * rhat).sum(dim=1).clamp(-1, 1)
    band = base["active"]
    out["color_max"] = float(base["color"].max())
    out["color_min"] = float(base["color"].min())
    out["int_delta_over_4piR2"] = float(dsv.sum() / surface)
    out["normal_angle_mean_deg"] = float(torch.rad2deg(
        torch.acos(cosang[band])).mean())
    out["variants"]["mls"] = float((base["kappa"] * dsv).sum() / dsv.sum() * radius)
    for st in stencils:
        for b in bands:
            st3 = csf_state3_two_phase(pos, mass, liquid, h, rho0, 0.072,
                                       curvature_method="fit3",
                                       curvature_stencil=st, fit_band=b)
            w = st3["delta_s"] * vol
            out["variants"][f"fit3[band={b},stencil={st}]"] = float(
                (st3["kappa"] * w).sum() / w.sum() * radius)
    return out


if __name__ == "__main__":
    import json
    for dx in (0.0025, 0.0016):
        rep = score(dx=dx)
        print(json.dumps(rep, indent=1))