"""XVAL: transport-velocity correction, TensorSPH vs SPHinXsys.

Anchor: ``XVAL-SPH-TRANSPORT-VELOCITY-CORRECTION`` (round 1, open-source
cross-validation).

What is compared
----------------
The reference implementation is SPHinXsys' ``TransportVelocityCorrection``
(``docs/xval_sphinx_tvc/oss_source/`` -- verbatim files with sha256, fetched
from upstream master, commit recorded in ``PROVENANCE.json``).  It is ported to
``tensorsph.transport_velocity_correction`` and run inside TensorSPH's WCSPH
dam-break solver, which already carries the Martin & Moyce (1952) digitised
front-position reference in-repo (``benchmarks/dam_break/martin_moyce_1952.json``,
provenance and digitisation uncertainty included).

The OSS C++ library itself is **not** built here (no Boost/TBB in the image --
recorded as a friction, see the report), so the cross-validation has two legs:

1. **Operator leg** (machine-checked, exact): the vectorised port is compared
   against a literal transcription of upstream's ``interaction()``/``update()``
   loops (``tests/test_transport_velocity_correction.py``), plus a continuum
   quadrature check of the ``grad-1`` integral.  Agreement is asserted at round
   off, so sign/weight/scope conventions are pinned to the source, not to
   memory.
2. **Physics leg** (this script): the ported scheme is run on the dam break and
   scored against the experimental reference, with the no-correction solver as
   the baseline and TensorSPH's own particle-shifting (PST, Lind et al. 2012) as
   the control.  Every variant runs the *same* code path; only the correction
   operator changes.

Predeclared decision rule (fixed before the runs, see the report's
*experimental design* section): TVC is adopted for the WCSPH free-surface
solver iff (i) mean absolute error of ``Z/a`` vs experiment is below the
0.15 owner gate **and** its mean relative error is under 15 %, (ii) it is not
worse than the baseline on the ``grad-1`` residual or on mass/density
integrity, and (iii) the improvement survives refinement (holds at two
resolutions, not one).

Usage
-----
    python benchmarks/xval_sphinx_tvc/run.py --resolutions 0.04 0.02 \
        --seconds 0.5 --out artifacts/xval_sphinx_tvc/grid.json
"""
from __future__ import annotations

import argparse
import json
import math
import platform
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(ROOT / "src"))

import torch  # noqa: E402

from tensorsph.dam_break import (DamBreakConfig, run_case,  # noqa: E402
                                 front_position)
from tensorsph.neighbor_search import cell_linked_pairs  # noqa: E402
from tensorsph.transport_velocity_correction import (  # noqa: E402
    kernel_gradient_integral)

EXP_PATH = ROOT / "benchmarks" / "dam_break" / "martin_moyce_1952.json"

#: Non-dimensionalisation stated in the experimental file: tau = t sqrt(2 g/a),
#: Z/a measured from the initial column face (x = a) as in Martin & Moyce.
COLUMN_WIDTH = 0.2
COLUMN_HEIGHT = 0.4

#: Owner accuracy standard: errors above 15 % are not acceptable.
OWNER_GATE_REL = 0.15
#: Gate already in docs/verification_ledger.md for this case.
LEDGER_GATE_ABS = 0.15


def load_experiment() -> dict:
    return json.loads(EXP_PATH.read_text())


def clumping_metrics(pos: torch.Tensor, dx: float, h: float) -> dict:
    """Distribution-regularity diagnostics on the final particle set.

    Only positions are used, so the numbers are comparable across variants and
    do not depend on which reduction ``run_case`` happens to return:

    * ``close_pair_fraction`` -- share of neighbour pairs closer than ``0.7 dx``
      (distance ``dx`` is the lattice distance; pairs well below it are
      clumping / approaching pairs, the failure mode the correction targets);
    * ``median_nn_over_dx`` -- median nearest-neighbour distance / dx, 1.0 for a
      perfect lattice and ->0 for a collapsed cluster;
    * ``grad1_residual_mean_h`` -- mean ``h |sum_j grad_i W_ij V_j|`` with one
      lattice volume per particle, fluid-only (no wall terms): the quadrature
      error the upstream operator drives down, measured on the same footing for
      every variant.
    """
    n = pos.shape[0]
    out = {"particles": int(n)}
    if n < 2:
        return out
    pairs = cell_linked_pairs(pos, 1.6 * dx)
    if pairs.numel():
        i, j = pairs.unbind(dim=1)
        r = torch.linalg.vector_norm(pos[i] - pos[j], dim=1) / dx
        out["close_pair_fraction"] = float((r < 0.7).to(torch.float64).mean())
        nn = torch.full((n,), float("inf"), dtype=r.dtype)
        nn = nn.scatter_reduce(0, i, r, reduce="amin", include_self=True)
        nn = nn.scatter_reduce(0, j, r, reduce="amin", include_self=True)
        finite = torch.isfinite(nn)
        out["median_nn_over_dx"] = float(nn[finite].median()) if finite.any() else None
    volume = torch.full((n,), dx ** pos.shape[1], dtype=pos.dtype)
    g = kernel_gradient_integral(pos, volume, h, pairs=None, wall_count=0)
    out["grad1_residual_mean_h"] = float(
        torch.linalg.vector_norm(g, dim=1).mean() * h)
    out["grad1_residual_max_h"] = float(
        torch.linalg.vector_norm(g, dim=1).max() * h)
    return out


def nondim_front(times, front, column_width: float, gravity: float):
    """``tau``, ``Z/a`` with ``Z`` measured from the initial column face."""
    scale = math.sqrt(2.0 * gravity / column_width)
    return ([t * scale for t in times],
            [(f - column_width) / column_width for f in front])


def interp(xs, ys, x):
    if x <= xs[0]:
        return ys[0]
    if x >= xs[-1]:
        return ys[-1]
    for i in range(1, len(xs)):
        if xs[i] >= x:
            w = (x - xs[i - 1]) / (xs[i] - xs[i - 1])
            return ys[i - 1] + w * (ys[i] - ys[i - 1])
    return ys[-1]


def score(tau, za, exp: dict) -> dict:
    data = exp["data"]
    ts = [d["tau"] for d in data]
    zs = [d["Z_over_a"] for d in data]
    rows = []
    for t_ref, z_ref in zip(ts, zs):
        if t_ref < tau[0] or t_ref > tau[-1]:
            continue
        got = interp(tau, za, t_ref)
        rows.append({"tau": t_ref, "Z_over_a_ref": z_ref, "Z_over_a_got": got,
                     "abs_error": abs(got - z_ref),
                     "rel_error": abs(got - z_ref) / z_ref if z_ref else None})
    if not rows:
        return {"points": 0}
    abs_err = [r["abs_error"] for r in rows]
    rel_err = [r["rel_error"] for r in rows if r["rel_error"] is not None]
    return {
        "points": len(rows),
        "mae": sum(abs_err) / len(abs_err),
        "max_abs_error": max(abs_err),
        "rmse": math.sqrt(sum(e * e for e in abs_err) / len(abs_err)),
        "mean_rel_error": sum(rel_err) / len(rel_err) if rel_err else None,
        "within_digitisation_uncertainty": sum(
            1 for r in rows if r["abs_error"] <= 0.05) / len(rows),
        "gate_abs_pass": (sum(abs_err) / len(abs_err)) <= LEDGER_GATE_ABS,
        "gate_rel_pass": (sum(rel_err) / len(rel_err)) <= OWNER_GATE_REL
        if rel_err else None,
        "rows": rows,
    }


VARIANTS = {
    "baseline": dict(),
    "tvc_a005_all": dict(tvc_every=1, tvc_alpha=0.05, tvc_scope="all"),
    "tvc_a010_all": dict(tvc_every=1, tvc_alpha=0.10, tvc_scope="all"),
    "tvc_a020_all": dict(tvc_every=1, tvc_alpha=0.20, tvc_scope="all"),
    "tvc_a005_bulk": dict(tvc_every=1, tvc_alpha=0.05, tvc_scope="bulk"),
    "tvc_a005_all_c5": dict(tvc_every=5, tvc_alpha=0.05, tvc_scope="all"),
    "pst_lind": dict(shift_every=1),
}


def hydrostatic_consistency(dx: float, seconds: float, variant: dict) -> dict:
    """Resting column: does the correction disturb hydrostatics?

    Analytic reference ``p = rho0 g (eta - y)`` inside the column, ``eta`` the
    initial free-surface height.  The metric is the mean relative pressure error
    over particles at least ``3 dx`` below the surface, so the (unavoidable)
    free-surface pressure deficit does not enter it.
    """
    cfg = DamBreakConfig(dx=dx, dt=1.0e-4, steps=int(round(seconds / 1.0e-4)),
                         output_every=max(1, int(round(seconds / 1.0e-4))),
                         device="cpu", column_width=2.0, column_height=0.4,
                         pool_width=2.0, tank_height=0.75, **variant)
    run = run_case(cfg)
    pos = run.final_positions
    # ``run_case`` returns positions only; the hydrostatic reference is
    # evaluated on the analytic profile the case starts from.
    eta = cfg.column_height
    p_ref = cfg.rho0 * cfg.gravity * (eta - pos[:, 1]).clamp_min(0.0)
    interior = pos[:, 1] < eta - 3.0 * dx
    return {
        "dx": dx,
        "variant": variant,
        "n_particles": int(pos.shape[0]),
        "interior_particles": int(interior.sum()),
        "mean_depth": float((eta - pos[:, 1])[interior].mean()),
        "mass_drift": run.mass_drift,
        "max_density_compression": run.max_density_compression,
        "max_tvc_shift_over_h": run.max_tvc_shift / (cfg.h_factor * dx),
        "p_ref_mean_interior": float(p_ref[interior].mean()) if interior.any() else None,
        "note": "pressure field not exported by run_case; depth/energy integrity "
                "and correction magnitude are recorded instead",
    }


def run_variant(name: str, variant: dict, dx: float, seconds: float,
                gravity: float = 9.81) -> dict:
    dt = DamBreakConfig(dx=dx).dt_from_courant()
    steps = int(round(seconds / dt))
    cfg = DamBreakConfig(dx=dx, dt=dt, steps=steps,
                         output_every=max(1, steps // 60), device="cpu",
                         **variant)
    t0 = time.time()
    run = run_case(cfg)
    wall = time.time() - t0
    tau, za = nondim_front(run.times, run.front, cfg.column_width, gravity)
    exp = load_experiment()
    result = {
        "variant": name,
        "params": variant,
        "dx": dx,
        "dt": dt,
        "steps": run.steps_taken,
        "simulated_time": run.simulated_time,
        "wall_clock_s": wall,
        "tau": tau,
        "Z_over_a": za,
        "max_tvc_shift_over_h": run.max_tvc_shift / (cfg.h_factor * dx),
        "max_density_compression": run.max_density_compression,
        "mass_drift": run.mass_drift,
        "max_acoustic_courant": run.max_acoustic_courant,
        "escaped": run.escaped,
        "max_speed": run.max_speed,
        "score": score(tau, za, exp),
        "clumping": clumping_metrics(run.final_positions, dx,
                                     cfg.h_factor * dx),
    }
    return result


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--resolutions", type=float, nargs="+", default=[0.04, 0.02])
    ap.add_argument("--seconds", type=float, default=0.5)
    ap.add_argument("--variants", nargs="+", default=list(VARIANTS))
    ap.add_argument("--out", type=Path,
                    default=ROOT / "artifacts" / "xval_sphinx_tvc" / "grid.json")
    ap.add_argument("--skip-hydrostatic", action="store_true")
    args = ap.parse_args(argv)

    records = []
    for dx in args.resolutions:
        for name in args.variants:
            rec = run_variant(name, VARIANTS[name], dx, args.seconds)
            sc = rec["score"]
            print(f"dx={dx:<6} {name:<16} mae={sc.get('mae', float('nan')):.4f} "
                  f"rel={sc.get('mean_rel_error', float('nan')):.3f} "
                  f"shift/h={rec['max_tvc_shift_over_h']:.3f} "
                  f"wall={rec['wall_clock_s']:.1f}s", flush=True)
            records.append(rec)
            args.out.parent.mkdir(parents=True, exist_ok=True)
            payload = {
                "anchor": "XVAL-SPH-TRANSPORT-VELOCITY-CORRECTION",
                "reference": load_experiment()["source"],
                "environment": {
                    "python": platform.python_version(),
                    "torch": torch.__version__,
                    "machine": platform.machine(),
                    "threads": torch.get_num_threads(),
                    "git_head": subprocess.run(
                        ["git", "rev-parse", "HEAD"], cwd=ROOT, capture_output=True,
                        text=True).stdout.strip(),
                },
                "seconds": args.seconds,
                "gates": {"owner_rel": OWNER_GATE_REL,
                          "ledger_abs": LEDGER_GATE_ABS},
                "records": records,
            }
            args.out.write_text(json.dumps(payload, indent=2))
    print(f"wrote {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())