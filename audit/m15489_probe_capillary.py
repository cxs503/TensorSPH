"""Independent probe for the m15371 CASE-CAPILLARY-WAVE audit (board-audit m15489).

Two checks that the case's own summary table does not expose:

1. **Per-lattice-phase stiffness ratios.**  ``case_point`` averages over
   ``shifts**2`` lattice phases and reports only the mean +- std.  This probe
   prints every phase, so the reader can see whether the ensemble mean is an
   accurate result or a cancellation of two oppositely biased results.
2. **Independent reference recomputation.**  ``ref.json`` is re-derived here
   from the closed form with mpmath at dps = 50, so the reference numbers are
   not taken on trust from the case artifacts.

Nothing in the repository is modified; this script only imports the shipped
case module and prints/records numbers.

Run::

    OMP_NUM_THREADS=4 PYTHONPATH=src python audit/m15489_probe_capillary.py \
        --out audit/capillary_probe_m15489.json
"""
from __future__ import annotations

import argparse
import importlib.util
import json
import math
from pathlib import Path

CASE_DIR = Path(__file__).resolve().parent.parent / "benchmarks" / "case_capillary_wave_dispersion"


def _load(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def per_phase(run, *, ppw: int, amplitude_fraction: float, sigma: float,
              shifts: int = 2) -> list[dict]:
    params = dict(run.PARAMS)
    lam = params["wavelength_m"]
    k = 2.0 * math.pi / lam
    amp = amplitude_fraction * lam
    rows = []
    for a in range(shifts):
        for b in range(shifts):
            geom = run.build_two_layer_wave(k, amp, ppw, params, a / shifts, b / shifts)
            m = run.measure(geom, sigma, curvature_scheme="mls")
            rows.append({"x_shift": a / shifts, "y_shift": b / shifts,
                         "stiffness_ratio": m["stiffness_ratio"],
                         "omega_error_pct": m["omega_error_pct"],
                         "delta_s_integral": m["delta_s_integral"]})
    return rows


def independent_reference(params: dict) -> dict:
    import mpmath as mp
    mp.mp.dps = 50
    k = 2 * mp.pi / mp.mpf(repr(params["wavelength_m"]))
    sig = mp.mpf(repr(params["sigma_N_m"]))
    r1 = mp.mpf(repr(params["rho_liquid_kg_m3"]))
    r2 = mp.mpf(repr(params["rho_gas_kg_m3"]))
    h1 = mp.mpf(repr(params["depth_fluid_m"]))
    h2 = mp.mpf(repr(params["depth_gas_m"]))
    om = mp.sqrt(sig * k ** 3 / (r1 * mp.coth(k * h1) + r2 * mp.coth(k * h2)))
    lamb = mp.sqrt(sig * k ** 3 / r1)
    return {"omega_ref_mpmath50": mp.nstr(om, 20),
            "omega_ref_float": float(om),
            "omega_lamb_deep": float(lamb),
            "rel_diff_vs_lamb": float(abs(om - lamb) / lamb),
            "k_rad_m": float(k)}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="audit/capillary_probe_m15489.json")
    args = ap.parse_args()

    run = _load("case_cw_run", CASE_DIR / "run.py")
    ref = _load("case_cw_ref", CASE_DIR / "ref.py")

    ref_ind = independent_reference(run.PARAMS)
    ref_case = ref.reference(dict(run.PARAMS))
    ref_delta = abs(ref_ind["omega_ref_float"] - ref_case["omega_ref_float"])

    phases = {}
    for ppw in (16, 24, 32, 48):
        for amp in (0.01, 0.02):
            rows = per_phase(run, ppw=ppw, amplitude_fraction=amp,
                             sigma=run.PARAMS["sigma_N_m"])
            ratios = [r["stiffness_ratio"] for r in rows]
            mean = sum(ratios) / len(ratios)
            phases[f"ppw{ppw}_amp{amp}"] = {
                "rows": rows,
                "mean_stiffness_ratio": mean,
                "mean_omega_error_pct": (math.copysign(math.sqrt(abs(mean)), mean) - 1.0) * 100.0,
                "worst_abs_omega_error_pct": max(abs(r["omega_error_pct"]) for r in rows),
                "n_null_phases": sum(1 for r in rows if abs(r["stiffness_ratio"]) < 1e-12),
            }

    # sigma invariance is exact by construction: sigma multiplies both the
    # measured and the theoretical stress, so the ratio cannot move.
    sigma_rows = [run.case_point(ppw=32, curvature_scheme="mls", amplitude_fraction=0.02,
                                 sigma=f * run.PARAMS["sigma_N_m"], shifts=2)
                  for f in (0.5, 1.0, 2.0)]
    out = {
        "case": "CASE-CAPILLARY-WAVE",
        "audit_mission": 15489,
        "audited_mission": 15371,
        "reference_independent": ref_ind,
        "reference_case_artifact": {"omega_ref_float": ref_case["omega_ref_float"],
                                    "ref_json_sha256": None},
        "reference_abs_diff_float": ref_delta,
        "reductions": ref_case["reductions"],
        "per_phase": phases,
        "sigma_ladder": [{"sigma_over_ref": r["sigma_over_ref"],
                          "stiffness_ratio": r["stiffness_ratio"],
                          "stiffness_ratio_std": r["stiffness_ratio_std"]}
                         for r in sigma_rows],
        "sigma_identical": len({r["stiffness_ratio"] for r in sigma_rows}) == 1,
    }
    Path(args.out).write_text(json.dumps(out, indent=2, sort_keys=True))
    print(json.dumps({k: v for k, v in out.items()
                      if k not in ("per_phase", "sigma_ladder")}, indent=2))
    for key, block in out["per_phase"].items():
        print(f"{key:>14}  mean_err={block['mean_omega_error_pct']:+7.3f}%  "
              f"worst_phase={block['worst_abs_omega_error_pct']:6.2f}%  "
              f"nulls={block['n_null_phases']}  ratios="
              + ", ".join(f"{r['stiffness_ratio']:+.4f}" for r in block["rows"]))
    print("sigma rows identical:", out["sigma_identical"], out["sigma_ladder"])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())