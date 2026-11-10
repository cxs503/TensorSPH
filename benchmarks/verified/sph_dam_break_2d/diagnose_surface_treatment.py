"""A/B diagnosis of the M1 front error: which physics switch moves it.

The M1 gate fails against the measured front, and a failing gate is only useful
if it says *why*.  This script integrates the same case twice at a coarse
spacing -- once with the shipped defaults, once with the optional free-surface
density renormalisation (``Config.density_renormalization``, documented in
``docs/benchmarks/free-surface-treatment.md``) -- and records both in the
evidence store under ``dam_break_2d_surface_treatment_ab``.

The two runs are a *diagnosis*, not a fix: the renormalised variant is not the
shipped configuration, it has not been verified on the other benchmarks, and it
is not gated here.  What it shows is the direction and size of the effect, which
is what the report quotes.

    python benchmarks/verified/sph_dam_break_2d/diagnose_surface_treatment.py --dx 0.04
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[3] / "src"))

from tensorsph.verification import dam_break_2d as m1
from tensorsph.verification.evidence import EvidenceStore

HERE = Path(__file__).resolve().parent
STORE = HERE / "evidence.sqlite"
CASE = "dam_break_2d_surface_treatment_ab"


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--dx", type=float, default=0.04)
    ap.add_argument("--seconds", type=float, default=0.35)
    ap.add_argument("--threads", type=int, default=8)
    args = ap.parse_args(argv)
    torch.set_num_threads(args.threads)

    table = m1.load_experimental()
    store = EvidenceStore(STORE)
    out = {}
    for label, kwargs in (("shipped_default", {}),
                          ("density_renormalization",
                           {"density_renormalization": True})):
        cfg = m1.DamBreak2DConfig(dx=args.dx, seconds=args.seconds,
                                  sample_interval=0.02, **kwargs)
        result = m1.run(cfg, label=label)
        front = m1.front_errors(result, table, min_tau=0.5)
        height = m1.height_anchor_errors(result)
        ritter = m1.ritter_errors(result)
        run_id = store.add_run(
            CASE, {**cfg.as_dict(), "variant": label},
            notes="A/B diagnosis of the dam-break front error; not a gated "
                  "verification run")
        store.add_metrics(run_id, {
            "front_max_rel_error": dict(value=front["max_rel_error"], unit="1"),
            "front_rms_rel_error": dict(value=front["rms_rel_error"], unit="1"),
            "ritter_ratio_mean": dict(value=ritter["ratio_mean"], unit="1"),
            "height_residual_max_rel_error": dict(
                value=height["residual_max_rel_error"], unit="1"),
            "height_ballistic_max_rel_error": dict(
                value=height["ballistic_max_rel_error"], unit="1"),
            "variant_density_renormalization": dict(
                value=float(cfg.density_renormalization), unit="bool"),
            "variant_surface_pressure_closure": dict(
                value=float(cfg.surface_pressure_closure), unit="bool"),
        })
        store.add_series(run_id, "front_curve", {
            "dx": args.dx, "particles": int(result["rows"][0]["particles"]),
            "tau": [r["tau"] for r in result["rows"]],
            "Z_over_a": [(r["front_coherent"] - cfg.column_width) / cfg.column_width
                         for r in result["rows"]]})
        out[label] = {"run_id": run_id,
                      "front_max_rel_error": front["max_rel_error"],
                      "front_rms_rel_error": front["rms_rel_error"],
                      "ritter_ratio_mean": ritter["ratio_mean"]}
        print(f"{label}: front max rel {front['max_rel_error']:.3f} "
              f"(rms {front['rms_rel_error']:.3f}), Ritter ratio "
              f"{ritter['ratio_mean']:.3f}", flush=True)
    (HERE / "results_ab.json").write_text(json.dumps(out, indent=1))
    store.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
