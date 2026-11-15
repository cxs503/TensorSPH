"""XVAL leg: SPHinXsys' ``TruncatedLinear`` limiter, ported and measured.

Anchor: ``XVAL-SPH-TRANSPORT-VELOCITY-CORRECTION``.

Upstream ships two limiter objects in ``src/shared/common/common_functors.h``
(sha256 ``c6ccd39c...``, verbatim copy in
``docs/xval_sphinx_tvc/oss_source/common_functors.h``)::

    NoLimiter        -> limiter(x) = 1
    TruncatedLinear  -> limiter(x) = SMIN(slope * x, 1),  slope = 100 by default

with ``x = h_ref**2 |inconsistency|**2`` (``update()`` in
``transport_velocity_correction.hpp``).  ``NoLimiter`` is the limiter of
``TransportVelocityCorrectionComplex`` -- the alias most upstream examples
instantiate -- but the three ``TransportVelocityLimitedCorrection*Complex``
aliases (corrected lid-driven cavity, non-Newtonian lid-driven cavity, throat)
instantiate ``TruncatedLinear``.  The port shipped only ``NoLimiter``; this leg
grades the newly transcribed limiter and answers the question that matters for
the other legs:

    does the un-ported limiter change any conclusion drawn with NoLimiter?

Two measurements, both predeclared before any run (``HYPOTHESES`` below):

* **operator level** -- where the limiter acts at all.  The threshold is
  ``|g| h_ref = sqrt(1/slope) = 0.1``; the plane free surface sits at
  ``|g| h_ref = 2 I(h,0) ~ 1.364`` (``SURFACE_ANCHOR.md``, measured 1.171 on
  the production lattice), a nearly regular bulk at ``~1e-2``.
* **physics level** -- the Martin & Moyce dam break, the case all the other
  legs are scored on, with the two upstream coefficients (0.2 dual-criteria
  default, 0.05 single acoustic step) x both limiters.

Usage
-----
    python benchmarks/xval_sphinx_tvc/limiter_leg.py            # measure
    python benchmarks/xval_sphinx_tvc/limiter_leg.py --doc      # render md+png
"""
from __future__ import annotations

import argparse
import datetime as _dt
import json
import math
import platform
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "benchmarks" / "xval_sphinx_tvc"))

import torch  # noqa: E402

import run as leg  # noqa: E402  (the dam-break harness of the other legs)
from tensorsph.dam_break import DamBreakConfig, run_case  # noqa: E402
from tensorsph.transport_velocity_correction import (  # noqa: E402
    TRUNCATED_LINEAR_DEFAULT_SLOPE, kernel_gradient_integral, no_limiter,
    transport_velocity_correction, truncated_linear)

OUT_JSON = ROOT / "docs" / "xval_sphinx_tvc" / "data" / "limiter_leg.json"
OUT_MD = ROOT / "docs" / "xval_sphinx_tvc" / "LIMITER_LEG.md"
OUT_FIG = ROOT / "docs" / "xval_sphinx_tvc" / "figure_limiter_leg.png"

#: Fixed before the runs.  A hypothesis that fails is reported as failed.
HYPOTHESES = {
    "H1_operator": (
        "TruncatedLinear(100) only acts where |g| h_ref < 0.1, so it damps "
        "nearly-regular bulk corrections (|g| h_ref ~ 1e-2) but leaves the "
        "free-surface displacement (|g| h_ref ~ 1.17) bit-identical to "
        "NoLimiter."),
    "H2_physics": (
        "On the free-surface dominated dam break the limiter changes the "
        "Martin & Moyce MAE by less than 0.02 in Z/a (13 % of the 0.15 gate) "
        "at both resolutions and flips no gate verdict, i.e. the missing "
        "limiter does not explain the residual error of the NoLimiter runs."),
    "H3_scope": (
        "If H2 holds it is because the correction is surface-driven: the "
        "share of particles with limiter < 1 in a dam-break state stays below "
        "50 % while their contribution to the maximum displacement is ~0."),
}
#: Absolute MAE tolerance of H2, in Z/a units.
H2_TOL_ABS = 0.02


def half_plane(nx: int = 60, ny: int = 24, dx: float = 0.02,
               dtype=torch.float64):
    xs = torch.arange(nx, dtype=dtype) * dx
    ys = torch.arange(ny, dtype=dtype) * dx
    gx, gy = torch.meshgrid(xs, ys, indexing="ij")
    pos = torch.stack((gx.reshape(-1), gy.reshape(-1)), dim=-1)
    return pos, torch.full((pos.shape[0],), dx * dx, dtype=dtype), nx, ny


def square(n: int = 25, dx: float = 0.02, jitter: float = 0.02,
           seed: int = 0, dtype=torch.float64):
    xs = torch.arange(n, dtype=dtype) * dx
    gx, gy = torch.meshgrid(xs, xs, indexing="ij")
    pos = torch.stack((gx.reshape(-1), gy.reshape(-1)), dim=-1)
    gen = torch.Generator().manual_seed(seed)
    pos = pos + (torch.rand(pos.shape, generator=gen, dtype=dtype) - 0.5) \
        * 2.0 * jitter * dx
    return pos, torch.full((pos.shape[0],), dx * dx, dtype=dtype), n


def survey(pos, vol, h, alpha=0.05):
    """|g|h and the two limiters on one particle set, elementwise."""
    g = kernel_gradient_integral(pos, vol, h)
    mag_h = torch.linalg.vector_norm(g, dim=1) * h
    x = mag_h * mag_h
    tl = truncated_linear()(x)
    none = no_limiter(x)
    tl_res = transport_velocity_correction(pos, vol, h, alpha=alpha,
                                           limiter=truncated_linear())
    nl_res = transport_velocity_correction(pos, vol, h, alpha=alpha,
                                           limiter=no_limiter)
    return {
        "grad_abs_h": mag_h,
        "limiter_tl": tl,
        "limiter_none": none,
        "shift_tl": torch.linalg.vector_norm(tl_res.shift, dim=1),
        "shift_none": torch.linalg.vector_norm(nl_res.shift, dim=1),
    }


def q(t: torch.Tensor, ps=(0.5, 0.9, 0.99)):
    if t.numel() == 0:
        return {}
    vals = torch.quantile(t.to(torch.float64), torch.tensor(ps,
                                                            dtype=torch.float64))
    return {f"p{int(p * 100)}": float(v) for p, v in zip(ps, vals)}


def operator_level(dx: float = 0.02, h_factor: float = 1.3) -> dict:
    """Where the limiter acts: plane free surface vs nearly regular bulk."""
    h = h_factor * dx
    out = {"dx": dx, "h": h, "h_over_dx": h_factor,
           "slope": TRUNCATED_LINEAR_DEFAULT_SLOPE,
           "threshold_abs_g_h": math.sqrt(1.0 / TRUNCATED_LINEAR_DEFAULT_SLOPE)}

    # (a) plane free surface: top row of a half plane, mid-column
    pos, vol, nx, ny = half_plane(dx=dx)
    s = survey(pos, vol, h)
    surface = (pos[:, 1] >= (ny - 1) * dx - 1e-12) \
        & (pos[:, 0] > 3 * h) & (pos[:, 0] < (nx - 1) * dx - 3 * h)
    out["free_surface"] = {
        "particles": int(surface.sum()),
        "grad_abs_h_mean": float(s["grad_abs_h"][surface].mean()),
        "limiters_all_one": bool((s["limiter_tl"][surface] == 1.0).all()),
        "shift_identical_to_NoLimiter": bool(
            torch.equal(s["shift_tl"][surface], s["shift_none"][surface])),
        "continuum_anchor_2I0": 30.0 / (7.0 * math.pi),
    }
    # (b) perfectly regular bulk: interior of a full square lattice (all four
    # free surfaces excluded), the state a well-behaved run approaches
    pos0, vol0, n0 = square(dx=dx, jitter=0.0)
    s0 = survey(pos0, vol0, h)
    interior0 = ((pos0[:, 0] > 2.2 * h) & (pos0[:, 0] < (n0 - 1) * dx - 2.2 * h)
                 & (pos0[:, 1] > 2.2 * h) & (pos0[:, 1] < (n0 - 1) * dx - 2.2 * h))
    out["bulk_regular"] = {
        "particles": int(interior0.sum()),
        "grad_abs_h_mean": float(s0["grad_abs_h"][interior0].mean()),
        "limiter_mean": float(s0["limiter_tl"][interior0].mean()),
        "limiter_max": float(s0["limiter_tl"][interior0].max()),
        "shift_ratio_mean": float(
            (s0["shift_tl"][interior0].mean() / s0["shift_none"][interior0].mean())),
    }

    # (c) jittered square: the "disturbed bulk" regime -- lattice + 2 % dx noise
    pos2, vol2, n = square(dx=dx, jitter=0.02)
    s2 = survey(pos2, vol2, h)
    interior = ((pos2[:, 0] > 2.2 * h) & (pos2[:, 0] < (n - 1) * dx - 2.2 * h)
                & (pos2[:, 1] > 2.2 * h) & (pos2[:, 1] < (n - 1) * dx - 2.2 * h))
    ratio = s2["shift_tl"][interior] / s2["shift_none"][interior]
    out["jittered_bulk"] = {
        "particles": int(interior.sum()),
        "jitter_amplitude_over_dx": 0.02,
        "grad_abs_h_mean": float(s2["grad_abs_h"][interior].mean()),
        "limiter_quantiles": q(s2["limiter_tl"][interior]),
        "shift_ratio_quantiles": q(ratio),
        "shift_ratio_equals_limiter": bool(torch.allclose(
            ratio, s2["limiter_tl"][interior], atol=1e-12, rtol=1e-12)),
    }
    detail = {
        "free_surface_above_threshold":
            out["free_surface"]["grad_abs_h_mean"] > out["threshold_abs_g_h"],
        "free_surface_limiter_is_one": out["free_surface"]["limiters_all_one"],
        "free_surface_shift_bit_identical":
            out["free_surface"]["shift_identical_to_NoLimiter"],
        "jittered_bulk_below_threshold":
            out["jittered_bulk"]["grad_abs_h_mean"] < out["threshold_abs_g_h"],
        "jittered_bulk_limiter_p50_below_half":
            out["jittered_bulk"]["limiter_quantiles"]["p50"] < 0.5,
    }
    out["H1_detail"] = detail
    out["H1_supported"] = bool(all(detail.values()))
    return out


def limiter_diagnostic(pos: torch.Tensor, dx: float, h: float,
                       alpha: float) -> dict:
    """Post-hoc limiter census on a snapshot (one lattice volume each)."""
    vol = torch.full((pos.shape[0],), dx ** pos.shape[1], dtype=pos.dtype)
    s = survey(pos, vol, h, alpha=alpha)
    active = s["limiter_tl"] < 1.0
    shift = s["shift_none"]
    top = shift == shift.max()
    return {
        "particles": int(pos.shape[0]),
        "limiter_below_one_fraction": float(active.to(torch.float64).mean()),
        "grad_abs_h_quantiles": q(s["grad_abs_h"]),
        "shift_share_of_limiters_only": float(
            shift[active].sum() / shift.sum()) if active.any() else 0.0,
        "max_shift_particle_limited": bool(active[top].any()),
        "mean_shift_ratio_tl_over_none": float(
            s["shift_tl"].mean() / s["shift_none"].mean()),
    }


def run_one(name: str, variant: dict, dx: float, seconds: float,
            output_samples: int = 60, gravity: float = 9.81) -> dict:
    """One dam-break run, scored exactly like the other legs (``run.score``)."""
    dt = DamBreakConfig(dx=dx).dt_from_courant()
    steps = int(round(seconds / dt))
    cfg = DamBreakConfig(dx=dx, dt=dt, steps=steps,
                         output_every=max(1, steps // output_samples),
                         device="cpu", **variant)
    t0 = time.time()
    run = run_case(cfg)
    wall = time.time() - t0
    tau, za = leg.nondim_front(run.times, run.front, cfg.column_width, gravity)
    exp = leg.load_experiment()
    h = cfg.h_factor * dx
    alpha = variant.get("tvc_alpha")
    return {
        "variant": name,
        "params": variant,
        "dx": dx,
        "dt": dt,
        "steps": run.steps_taken,
        "simulated_time": run.simulated_time,
        "wall_clock_s": wall,
        "tau": tau,
        "Z_over_a": za,
        "max_tvc_shift_over_h": run.max_tvc_shift / h,
        "max_density_compression": run.max_density_compression,
        "mass_drift": run.mass_drift,
        "escaped": run.escaped,
        "score": leg.score(tau, za, exp),
        "clumping": leg.clumping_metrics(run.final_positions, dx, h),
        "limiter_census": (limiter_diagnostic(run.final_positions, dx, h,
                                             0.05 if alpha is None else alpha)
                           if variant.get("tvc_every") else None),
    }


def physics_level(resolutions, seconds: float,
                  output_samples: int = 60) -> dict:
    variants = {
        "baseline": dict(),
        "tvc_a005_all": dict(tvc_every=1, tvc_alpha=0.05, tvc_scope="all"),
        "tvc_a005_all_tl": dict(tvc_every=1, tvc_alpha=0.05, tvc_scope="all",
                                tvc_limiter="truncated_linear"),
        "tvc_a020_all": dict(tvc_every=1, tvc_alpha=0.20, tvc_scope="all"),
        "tvc_a020_all_tl": dict(tvc_every=1, tvc_alpha=0.20, tvc_scope="all",
                                tvc_limiter="truncated_linear"),
    }
    records = [run_one(name, variant, dx, seconds, output_samples)
               for dx in resolutions for name, variant in variants.items()]
    return {"seconds": seconds, "resolutions": list(resolutions),
            "variants": list(variants), "records": records}


def verdicts(payload: dict) -> dict:
    op = payload["operator_level"]
    phys = payload["physics_level"]
    pairs = [("tvc_a005_all", "tvc_a005_all_tl"),
             ("tvc_a020_all", "tvc_a020_all_tl")]
    rows = []
    worst = 0.0
    verdict_flip = False
    for dx in phys["resolutions"]:
        by = {r["variant"]: r for r in phys["records"] if r["dx"] == dx}
        for a, b in pairs:
            if a not in by or b not in by:
                continue
            d_mae = by[b]["score"]["mae"] - by[a]["score"]["mae"]
            worst = max(worst, abs(d_mae))
            va = by[a]["score"]["gate_abs_pass"] and by[a]["score"]["gate_rel_pass"]
            vb = by[b]["score"]["gate_abs_pass"] and by[b]["score"]["gate_rel_pass"]
            verdict_flip |= bool(va != vb)
            rows.append({
                "dx": dx, "pair": f"{a} vs {b}",
                "mae_no_limiter": by[a]["score"]["mae"],
                "mae_truncated_linear": by[b]["score"]["mae"],
                "delta_mae": d_mae,
                "rel_mean_no_limiter": by[a]["score"]["mean_rel_error"],
                "rel_mean_truncated_linear": by[b]["score"]["mean_rel_error"],
                "gate_verdict_changed": bool(va != vb),
                "max_shift_over_h_no_limiter": by[a]["max_tvc_shift_over_h"],
                "max_shift_over_h_truncated_linear": by[b]["max_tvc_shift_over_h"],
            })
    census = [r["limiter_census"] for r in phys["records"]
              if r["variant"].endswith("_tl") and r["limiter_census"]]
    max_limited_fraction = max(
        (c["limiter_below_one_fraction"] for c in census), default=float("nan"))
    by_alpha = {}
    for c in rows:
        alpha = c["pair"].split("alpha")[0]
        key = ("a005" if "a005" in c["pair"].split(" vs ")[0]
               else "a020" if "a020" in c["pair"].split(" vs ")[0] else "other")
        by_alpha.setdefault(key, []).append(abs(c["delta_mae"]))
    return {
        "comparisons": rows,
        "worst_abs_delta_mae_by_alpha": {k: max(v) for k, v in by_alpha.items()},
        "worst_abs_delta_mae": worst,
        "H2_tolerance_abs": H2_TOL_ABS,
        "H2_supported": bool(worst < H2_TOL_ABS and not verdict_flip),
        "any_gate_verdict_flip": bool(verdict_flip),
        "max_limiter_active_fraction": max_limited_fraction,
        "H3_supported": bool(max_limited_fraction < 0.5),
        "H1_supported": bool(op["H1_supported"]),
    }


def measure(resolutions, seconds: float, out: Path = OUT_JSON) -> dict:
    t0 = time.time()
    payload = {
        "generated_utc": _dt.datetime.now(_dt.timezone.utc).isoformat(
            timespec="seconds"),
        "anchor": "XVAL-SPH-TRANSPORT-VELOCITY-CORRECTION",
        "leg": "limiter (TruncatedLinear) transcription + effect measurement",
        "hypotheses_predeclared": HYPOTHESES,
        "env": {"python": platform.python_version(),
                "torch": torch.__version__,
                "cpu_count": torch.get_num_threads(),
                "machine": platform.machine()},
        "upstream": {
            "file": "src/shared/common/common_functors.h",
            "sha256": "c6ccd39c037ea86769438b8fd8ed1d32e6609b78e942f6b2c3aee1ab3c8759e6",
            "definition": "SMIN(slope_ * dimensionless_measure, Real(1))",
            "default_slope": TRUNCATED_LINEAR_DEFAULT_SLOPE,
            "aliases_using_it": [
                "TransportVelocityLimitedCorrectionCorrectedComplex",
                "TransportVelocityLimitedCorrectionComplex",
                "TransportVelocityLimitedCorrectionCorrectedForOpenBoundaryFlowComplex"],
            "upstream_examples_using_it": [
                "test_2d_lid_driven_cavity_corrected",
                "test_2d_lid_driven_cavity_non_newtonian",
                "test_2d_throat"],
        },
        "operator_level": operator_level(),
        "physics_level": physics_level(resolutions, seconds),
    }
    payload["verdicts"] = verdicts(payload)
    payload["wall_clock_s"] = time.time() - t0
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(payload, indent=1, default=float))
    return payload


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--resolutions", type=float, nargs="+", default=[0.03, 0.02])
    ap.add_argument("--seconds", type=float, default=0.45)
    ap.add_argument("--out", type=Path, default=OUT_JSON)  # noqa: PLW0603
    ap.add_argument("--doc", action="store_true", help="render the doc instead")
    ap.add_argument("--recompute-verdicts", action="store_true",
                    help="re-derive the verdicts from an existing payload")
    args = ap.parse_args(argv)
    if args.doc:
        import make_limiter_doc
        return make_limiter_doc.render(args.out, OUT_MD, OUT_FIG)
    if args.recompute_verdicts:
        payload = json.loads(args.out.read_text())
        op = payload["operator_level"]
        thr = op["threshold_abs_g_h"]
        op.setdefault("H1_detail", {
            "free_surface_above_threshold":
                op["free_surface"]["grad_abs_h_mean"] > thr,
            "free_surface_limiter_is_one":
                op["free_surface"]["limiters_all_one"],
            "free_surface_shift_bit_identical":
                op["free_surface"]["shift_identical_to_NoLimiter"],
            "jittered_bulk_below_threshold":
                op["jittered_bulk"]["grad_abs_h_mean"] < thr,
            "jittered_bulk_limiter_p50_below_half":
                op["jittered_bulk"]["limiter_quantiles"]["p50"] < 0.5,
        })
        op["H1_supported"] = bool(all(op["H1_detail"].values()))
        payload["verdicts"] = verdicts(payload)
        args.out.write_text(json.dumps(payload, indent=1, default=float))
        print(json.dumps(payload["verdicts"], indent=1))
        return 0
    payload = measure(args.resolutions, args.seconds, args.out)
    print(json.dumps(payload["verdicts"], indent=1))
    print(f"wrote {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())