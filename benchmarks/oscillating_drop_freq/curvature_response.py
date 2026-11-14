"""Does the shipped curvature estimator respond to an interface *deformation*?

The repository validates its CSF curvature operators on a **static** circle
(``docs/verification_surface_tension_consistent.md``, ``docs/verification_csf_curvature_bias.md``):
``fit_quadratic`` is documented as "+5 % at R/dx = 10" and that number holds
(``diagnose.py`` re-measures ``<kappa> R = 2.05 .. 2.14``).  A capillary *solver*
needs something stronger: the curvature of a surface that is being deformed,
because the restoring force of a mode is the *perturbation* of the curvature,
``delta kappa = (l-1)(l+2) eps P_l / R`` -- at ``eps = 0.02`` that is 4 % of
``2/R``.

This script measures that response directly.  On the axisymmetric drop
``r = R[1 + eps P_l(cos theta)]`` it projects the ``delta_s``-weighted meridional
curvature onto ``P_l``,

    <kappa_m>_l = (2l+1)/A * sum_i kappa_i delta_s,i dV_i P_l(cos theta_i),

and compares the increment ``<kappa_m>_l(eps) - <kappa_m>_l(0)`` with the exact
``(l-1)(l+1) eps / R``.  The **response ratio** is 1 for a usable estimator, 0 for
one that ignores the deformation and negative for one that pushes the wrong way.
Knobs swept: ``curvature_scheme`` x ``curvature_stencil`` x
``curvature_color_smoothing``, i.e. everything the estimator exposes.

Usage::

    PYTHONPATH=src python benchmarks/oscillating_drop_freq/curvature_response.py
"""
from __future__ import annotations

import argparse
import json
import math
import platform
import sys
import time
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))

from tensorsph.csf_consistent import csf_fields  # noqa: E402
from tensorsph.oscillating_drop3d import axisym_drop_geometry, legendre_p  # noqa: E402
from tensorsph.surface_tension_dispersion import cell_linked_pairs  # noqa: E402

OUT = Path(__file__).resolve().parent
R_DEFAULT, SIGMA_DEFAULT, RHO_DEFAULT = 0.02, 7.2, 1000.0


def project_meridional(geom, *, scheme, stencil, color_smoothing, normal_smoothing,
                       mode) -> dict:
    """``delta_s``-weighted ``P_mode`` projector of the meridional curvature."""
    h = geom["h"]
    pairs = cell_linked_pairs(geom["pos"], stencil * h)
    state = csf_fields(geom["pos"], geom["mass"], pairs, h, RHO_DEFAULT,
                       normal_smoothing=normal_smoothing,
                       curvature_scheme=scheme, curvature_stencil=stencil,
                       curvature_color_smoothing=color_smoothing)
    w = state.delta_s * geom["dvol"]
    area = 4.0 * math.pi * geom["radius"] ** 2
    share = legendre_p(mode, geom["cos_theta"])
    norm = float((2.0 * mode + 1.0) / area * w.sum())
    proj = float((2.0 * mode + 1.0) / area * (state.kappa * w * share).sum())
    return {"proj": proj / norm if norm else float("nan"),
            "kappaR_mean": float((state.kappa * w).sum() / w.sum())
            * geom["radius"] if float(w.sum()) else float("nan"),
            "coarea2d_ratio": float(float((2.0 * w.sum() / area) / norm))
            if norm else float("nan")}


def response(radius, *, rdx, scheme, stencil, color_smoothing, normal_smoothing,
             mode=2, eps_ref=0.05, eps_base=0.0) -> dict:
    geom_b = axisym_drop_geometry(radius, radius / rdx, mode=mode, eps=eps_base,
                                  rho0=RHO_DEFAULT)
    geom_r = axisym_drop_geometry(radius, radius / rdx, mode=mode, eps=eps_ref,
                                  rho0=RHO_DEFAULT)
    kw = dict(scheme=scheme, stencil=stencil, color_smoothing=color_smoothing,
              normal_smoothing=normal_smoothing, mode=mode)
    base = project_meridional(geom_b, **kw)
    ref = project_meridional(geom_r, **kw)
    exact = (mode - 1.0) * (mode + 1.0) * eps_ref / radius
    ratio = (ref["proj"] - base["proj"]) / exact
    return {"radius_over_dx": rdx, "scheme": scheme, "stencil": stencil,
            "color_smoothing": color_smoothing, "normal_smoothing": normal_smoothing,
            "eps_base": eps_base, "eps_ref": eps_ref,
            "proj_base": base["proj"], "proj_ref": ref["proj"],
            "exact_increment": exact, "response_ratio": ratio,
            "kappaR_mean_base": base["kappaR_mean"],
            "kappaR_mean_ref": ref["kappaR_mean"]}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--quick", action="store_true")
    args = ap.parse_args()
    schemes = ("fit_quadratic", "fit_circle") if not args.quick else ("fit_quadratic",)
    stencils = (2.0, 3.0, 4.0) if not args.quick else (4.0,)
    smooths = (0, 1, 3) if not args.quick else (0,)
    ladder = (12.0, 16.0, 20.0) if not args.quick else (12.0,)
    t0 = time.time()
    rows = []
    for scheme in schemes:
        for stencil in stencils:
            for cs in smooths:
                for rdx in ladder:
                    r = response(R_DEFAULT, rdx=rdx, scheme=scheme, stencil=stencil,
                                 color_smoothing=cs, normal_smoothing=2)
                    rows.append(r)
                    print(f"{scheme:13s} stencil={stencil:3.1f} cs={cs} R/dx={rdx:4.0f}: "
                          f"response={r['response_ratio']:+.3f} "
                          f"(static kappaR={r['kappaR_mean_ref']:.3f})", flush=True)
    # mode-independence check
    modes = [{"mode": l,
              "response_ratio": response(R_DEFAULT, rdx=16.0, scheme="fit_quadratic",
                                         stencil=4.0, color_smoothing=0,
                                         normal_smoothing=2, mode=l)["response_ratio"]}
             for l in (2, 3)]
    payload = {
        "case": "SPH-OSC-DROP-FREQ", "diagnostic": "curvature deformation response",
        "question": "does the shipped CSF curvature estimator resolve the O(eps) "
                    "perturbation of a drop interface?",
        "reference": "exact meridional increment (l-1)(l+1) eps / R",
        "generated": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "environment": {"python": platform.python_version(), "torch": torch.__version__},
        "parameters": {"radius_m": R_DEFAULT, "rho_kg_m3": RHO_DEFAULT,
                       "wall_seconds": time.time() - t0},
        "rows": rows, "mode_independence": modes,
        "summary": {
            "n_rows": len(rows),
            "response_min": min(r["response_ratio"] for r in rows),
            "response_max": max(r["response_ratio"] for r in rows),
            "n_within_20pct": sum(1 for r in rows
                                  if abs(r["response_ratio"] - 1.0) < 0.20),
            "static_kappaR_range": [min(r["kappaR_mean_ref"] for r in rows),
                                    max(r["kappaR_mean_ref"] for r in rows)],
        },
    }
    (OUT / "curvature_response.json").write_text(json.dumps(payload, indent=2))
    print(json.dumps(payload["summary"], indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())