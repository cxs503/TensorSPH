"""End-to-end Young-Laplace pressure A/B: ``delta_s`` normalisation ``fixed`` vs ``jump``.

Static diagnostics (``run.py``) show that the measured-jump normalisation closes
the delta-function integral and the geometric cut force.  This harness asks the
other question: what does it do to the *pressure*, which is the only end-to-end
Young-Laplace criterion the repository has?

Relax one capillary time at two resolutions with everything else fixed, and
report the pressure jump ``p_core - p_shell`` against ``sigma / R = 360 Pa``,
together with the spurious-current ratio ``max|v| / sqrt(sigma / (rho R))``
(Adami et al. 2012 threshold 0.1).  See
``docs/verification_delta_s_normalization.md`` §3.5.

Usage::

    PYTHONPATH=src python benchmarks/csf_delta_s/run_relax.py
"""
from __future__ import annotations

import importlib.util
import json
from pathlib import Path

import torch

torch.set_num_threads(2)

HERE = Path(__file__).resolve().parent
CASES = ((0.004, 380), (0.002, 600))


def _load(name: str, rel: str):
    spec = importlib.util.spec_from_file_location(name, HERE.parent / rel)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def main():
    law = _load("laplace_law_run_relax", "laplace_law/run.py")
    rows = []
    for dx, steps in CASES:
        for norm in ("fixed", "jump"):
            rep = law.relax(radius=0.02, dx=dx, sigma=7.2, steps=steps,
                            curvature="fit_quadratic", delta_s=norm)
            lap = rep["laplace"]
            sp = rep.get("spurious_current", {})
            rows.append({
                "dx_m": dx,
                "steps": steps,
                "normalization": norm,
                "curvature_operator": "fit_quadratic",
                "delta_p_theory_Pa": lap["delta_p_theory_Pa"],
                "delta_p_measured_Pa": lap["delta_p_measured_Pa"],
                "rel_error_pct": lap["rel_error_pct"],
                "within_15pct": bool(lap["within_15pct"]),
                "spurious_ratio": float(sp.get("ratio", float("nan"))),
                "spurious_passes_adami": bool(sp.get("passes_adami_0p1", False)),
            })
            print(f"dx={dx} steps={steps} {norm:5s} "
                  f"dp={lap['delta_p_measured_Pa']:+.1f} Pa "
                  f"({lap['rel_error_pct']:+.1f}%) vmax_ratio={sp.get('ratio', float('nan')):.3f}",
                  flush=True)
    payload = {"setup": {"radius_m": 0.02, "sigma_N_per_m": 7.2, "rho0": 1000.0,
                         "capillary_time_s": 0.03333333333333334,
                         "curvature_operator": "fit_quadratic"},
               "rows": rows}
    (HERE / "results_relax.json").write_text(json.dumps(payload, indent=2))
    return payload


if __name__ == "__main__":
    main()