"""XVAL leg: is the TVC coefficient's *unit* (per call, not per second) right?

Anchor: ``XVAL-SPH-TRANSPORT-VELOCITY-CORRECTION``.

The gap this leg closes
-----------------------
Upstream's ``transport_velocity_correction.hpp`` ``update()`` is

    pos_[i] += correction_scaling_ * limiter_(squared_h_ref_ * |g_i|^2) * g_i

with ``correction_scaling_ = coefficient * h_ref**2`` -- **there is no ``dt``
in the shift**.  The operator therefore applies a fixed-length displacement per
*call*, and the physical correction *rate* is

    rate_i = coefficient * h_ref**2 * |g_i|  x  (calls per unit time).

The upstream header then prescribes one coefficient per cadence (``0.2`` for the
dual-criteria scheme, i.e. one call per *convection* step; "decreased to about
1/4" -- ``0.05`` -- for a single *acoustic* step).  That prescription is only
meaningful if

    N calls with coefficient a/N  ==  1 call with coefficient a            (INV)

i.e. if the coefficient is a *rate coefficient* and not a per-call constant.
``INV`` is neither in the vendored sources nor in any earlier leg of this
anchor: rounds 1-3 swept ``alpha`` at ``tvc_every = 1`` only, so "alpha = 0.05
is the single-step value and it fails the gate" and "alpha = 0.10 passes" were
both statements about a cadence that was never cross-checked.

Design (variants and gates fixed **before** the runs)
-----------------------------------------------------
Two probes, both purely internal to the port, so no experimental reference is
needed and no digitisation noise enters:

1. **Operator probe** (no time stepping).  On a fixed dam-break configuration
   take one call with ``alpha`` and ``N`` successive calls with ``alpha/N``
   (re-evaluating ``g`` on the shifted positions after every call).  Measures
   the composition error ``|r_1 - r_N|`` in units of the one-call shift.
   Under ``INV`` this goes to zero; the residual is the second-order term of
   the shift map.  Run with ``NoLimiter`` (upstream's default) and with
   ``TruncatedLinear`` (upstream's other shipped limiter).

2. **Physics probe** (dam break, Martin & Moyce 1952).  Variants
   ``tvc_every = N, alpha = 0.20`` for ``N = 1..6`` and ``alpha = 0.05,
   tvc_every = 1``.  ``INV`` predicts ``d(s005_e1, d020_e4) = 0``; the
   measured curve distance is reported for every ``N`` and the minimiser
   ``k* = argmin_N d`` is the empirically implied equivalence ratio (a direct
   measurement of "how many acoustic calls one dual-criteria call is worth").

Predeclared gates
-----------------
* ``INV`` **holds** iff ``d(s005_e1, d020_e4) <= 0.03`` mean relative
  difference of ``Z/a`` (self-comparison of one solver: the 15 % owner gate is
  far too loose to call two configurations equivalent) **and** the same
  ``d <= 0.05`` absolute, the reference's own digitisation uncertainty.
* ``k*`` is reported with the curve-distance profile, not asserted.

Usage
-----
    python benchmarks/xval_sphinx_tvc/cadence_invariance.py \
        --resolutions 0.03 0.02 --seconds 0.5 \
        --out docs/xval_sphinx_tvc/data/cadence_invariance.json
"""
from __future__ import annotations

import argparse
import json
import math
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parent))

import torch  # noqa: E402

from tensorsph.dam_break import (DamBreakConfig, make_column,  # noqa: E402
                                wall_lattice)
from tensorsph.droplet_impact import lattice_mass_unit  # noqa: E402
from tensorsph.transport_velocity_correction import (  # noqa: E402
    kernel_gradient_integral,
    no_limiter,
    truncated_linear,
)
from run import interp, load_experiment, nondim_front, run_variant, score  # noqa: E402

#: Single-step coefficient implied by the upstream header (0.2 / 4).
ALPHA_SINGLE = 0.05
#: Dual-criteria coefficient, the literal constructor default.
ALPHA_DUAL = 0.2
#: Equivalence ratio predicted by the header's "about 1/4" rule.
N_PRED = 4
#: Cadence sweep, fixed before any run.
N_SWEEP = (1, 2, 3, 4, 5, 6)
#: Gate: mean relative difference of Z/a for two cadences to be the same scheme.
INV_REL_GATE = 0.03
#: Gate: absolute curve distance inside the reference digitisation band.
INV_ABS_GATE = 0.05

PHYSICS_VARIANTS = {
    "baseline": dict(),
    "s005_e1": dict(tvc_every=1, tvc_alpha=ALPHA_SINGLE, tvc_scope="all"),
    **{f"d020_e{n}": dict(tvc_every=n, tvc_alpha=ALPHA_DUAL, tvc_scope="all")
       for n in N_SWEEP},
}


# --------------------------------------------------------------------------- #
# probe 1: operator composition (INV at the level of the shift map)
# --------------------------------------------------------------------------- #
def operator_composition(dx: float, alpha: float, n_sub: int,
                         limiter: str = "none", max_sub_rounds: int = 64) -> dict:
    """One ``alpha`` call vs ``n_sub`` calls of ``alpha / n_sub``.

    The configuration is TensorSPH's own dam-break initial state (a free
    surface, so ``g != 0``); ``g`` is re-evaluated on the shifted positions
    after every sub-call, exactly as a time loop would.
    """
    cfg = DamBreakConfig(dx=dx, column_width=0.2, column_height=0.4,
                         pool_width=2.0, tank_height=0.75)
    device = torch.device("cpu")
    fluid, mass = make_column(cfg, device)
    wall = wall_lattice(cfg, device, top=cfg.tank_top())
    pos0 = torch.cat([fluid, wall], dim=0)
    n_fluid = fluid.shape[0]
    # volumes of the solver at the first step it corrects: m/rho with the
    # calibrated lattice mass unit, wall volume dx**D (dam_break.run_case).
    volume = torch.cat((
        mass / lattice_mass_unit(cfg) * dx ** fluid.shape[1],
        torch.full((wall.shape[0],), dx ** fluid.shape[1], dtype=fluid.dtype)))
    h = cfg.h_factor * dx
    lim = no_limiter if limiter == "none" else truncated_linear()

    n_wall = int(wall.shape[0])
    big = kernel_gradient_integral(pos0, volume, h, wall_count=n_wall)
    pos_big = pos0.clone()
    pos_big[:n_fluid] += alpha * h * h * lim(h * h * big.square().sum(1))[:, None] * big
    shift_big = pos_big[:n_fluid] - pos0[:n_fluid]

    pos_small = pos0.clone()
    inv_h2 = 1.0 / (h * h)
    for _ in range(n_sub):
        g = kernel_gradient_integral(pos_small, volume, h, wall_count=n_wall)
        pos_small[:n_fluid] += (alpha / n_sub) * h * h \
            * lim(h * h * g.square().sum(1))[:, None] * g
    shift_small = pos_small[:n_fluid] - pos0[:n_fluid]

    diff = (shift_big - shift_small).norm(dim=-1)
    ref = shift_big.norm(dim=-1)
    denom = float(ref.mean())
    return {
        "dx": dx,
        "alpha": alpha,
        "n_sub": n_sub,
        "limiter": limiter,
        "n_fluid": n_fluid,
        "h_over_dx": cfg.h_factor,
        "mean_shift_over_h": denom / h,
        "max_shift_over_h": float(ref.max()) / h,
        "composition_error_mean_over_shift": float(diff.mean()) / denom,
        "composition_error_max_over_h": float(diff.max()) / h,
        "inv_h2_unused": inv_h2,
    }


def operator_sweep(dx: float) -> dict:
    rows = []
    for limiter in ("none", "truncated_linear"):
        for alpha in (0.05, 0.10, 0.20, 0.40):
            for n_sub in (1, 2, 4, 8):
                r = operator_composition(dx, alpha, n_sub, limiter)
                r.pop("inv_h2_unused", None)
                rows.append(r)
    return {"dx": dx, "rows": rows}


# --------------------------------------------------------------------------- #
# probe 2: curve distance between physics runs
# --------------------------------------------------------------------------- #
def curve_distance(a: dict, b: dict) -> dict:
    """Mean |dZ/a| between two runs on their common ``tau`` window."""
    lo = max(a["tau"][0], b["tau"][0])
    hi = min(a["tau"][-1], b["tau"][-1])
    n = 60
    grid = [lo + (hi - lo) * k / (n - 1) for k in range(n)]
    za = [interp(a["tau"], a["Z_over_a"], t) for t in grid]
    zb = [interp(b["tau"], b["Z_over_a"], t) for t in grid]
    dz = [abs(x - y) for x, y in zip(za, zb)]
    scale = sum(abs(y) for y in zb) / len(zb)
    mae = sum(dz) / len(dz)
    return {
        "tau_window": [lo, hi],
        "mean_abs_dz_over_a": mae,
        "max_abs_dz_over_a": max(dz),
        "mean_rel_difference": mae / scale if scale else None,
        "inv_rel_pass": (mae / scale) <= INV_REL_GATE if scale else None,
        "inv_abs_pass": mae <= INV_ABS_GATE,
    }


def physics_sweep(resolutions, seconds: float) -> dict:
    rows = []
    for dx in resolutions:
        only = None if dx == resolutions[0] else ("baseline", "s005_e1",
                                                 f"d020_e{N_PRED}")
        for name, variant in PHYSICS_VARIANTS.items():
            if only is not None and name not in only:
                continue
            t0 = time.time()
            r = run_variant(name, variant, dx, seconds)
            r["wall_clock_s"] = time.time() - t0
            r["variant"] = name
            rows.append(r)
            print(f"  dx={dx} {name:10s} rel="
                  f"{r['score'].get('mean_rel_error', float('nan')):.4f}", flush=True)
    return {"rows": rows}


def analyse(physics: dict) -> dict:
    rows = physics["rows"]
    by_dx: dict[float, dict[str, dict]] = {}
    for r in rows:
        by_dx.setdefault(r["dx"], {})[r["variant"]] = r
    out = {}
    for dx, runs in by_dx.items():
        if "s005_e1" not in runs:
            continue
        pairs = {}
        for name, run in runs.items():
            if name.startswith("d020_e"):
                pairs[name] = curve_distance(runs["s005_e1"], run)
        if not pairs:
            continue
        k_star = min(pairs, key=lambda k: pairs[k]["mean_abs_dz_over_a"])
        matched = pairs.get(f"d020_e{N_PRED}")
        out[str(dx)] = {
            "pairs_vs_s005_e1": pairs,
            "k_star_variant": k_star,
            "k_star_n": int(k_star.split("_e")[1]),
            "n_predicted": N_PRED,
            "rate_matched_pair": matched,
            "inv_holds": None if matched is None else bool(
                matched["inv_rel_pass"] and matched["inv_abs_pass"]),
        }
    return out


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--resolutions", type=float, nargs="+", default=[0.03, 0.02])
    ap.add_argument("--seconds", type=float, default=0.5)
    ap.add_argument("--out", type=Path,
                    default=ROOT / "docs/xval_sphinx_tvc/data/cadence_invariance.json")
    args = ap.parse_args(argv)

    payload = {
        "leg": "cadence-invariance",
        "anchor": "XVAL-SPH-TRANSPORT-VELOCITY-CORRECTION",
        "claim_under_test": "N calls with coefficient a/N == 1 call with "
                            "coefficient a (upstream header's 'about 1/4' rule)",
        "gates": {"inv_rel": INV_REL_GATE, "inv_abs": INV_ABS_GATE,
                  "owner_rel": 0.15, "ledger_abs": 0.15},
        "alpha_single": ALPHA_SINGLE,
        "alpha_dual": ALPHA_DUAL,
        "n_sweep": list(N_SWEEP),
        "operator": operator_sweep(args.resolutions[0]),
        "physics": physics_sweep(args.resolutions, args.seconds),
        "reference": load_experiment()["source"],
    }
    payload["analysis"] = analyse(payload["physics"])
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(payload, indent=2))
    print(f"wrote {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())