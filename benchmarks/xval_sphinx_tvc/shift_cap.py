"""Is ``tvc_max_shift`` a real safety valve, and what does it mean in ``alpha``?

Anchor: ``XVAL-SPH-TRANSPORT-VELOCITY-CORRECTION``.

Why this leg exists
-------------------
``RESOLUTION_LADDER.md`` ends with a practical rule (section 6) that tells a
user of the port to add "a ``tvc_max_shift`` cap" when refining, and
``make_limiter_doc.py`` repeats the advice.  **Neither ever ran the cap.**  The
solver flag exists (``DamBreakConfig.tvc_max_shift``, a *radial rescale* of the
displacement vector: ``shift *= min(1, cap*h / |shift|)`` in
``transport_velocity_correction()``) and it is untested at solver level, so the
port's only documented mitigation is currently an unverified recommendation.
This leg tests it.

The analytic handle comes from ``SURFACE_ANCHOR.md``: at a plane free surface
the gradient theorem gives ``|g| = 2 I(h,d)`` with ``I(h,0) = 15/(7 pi h)``, so
the *dominant* (surface) displacement of the operator is

    |delta r|/h = 30 alpha / (7 pi) = 1.3641 alpha          (1)

-- the largest shift the scheme wants to make is proportional to the
coefficient.  A cap ``c`` (in units of ``h``) therefore truncates exactly that
surface population whenever ``c < 1.3641 alpha``, and does nothing at all when
``c > max_t |delta r|/h``.  Between the two it acts as an **effective
coefficient**

    alpha_eff(c) = min(alpha, 7 pi c / 30) = min(alpha, c / 1.3641)      (2)

*if* the surface population really carries the maximum.  (2) is the derived
prediction tested here; it is falsifiable, and it is the reason a cap is not a
free lunch: it does not remove the bad part of a too-large ``alpha``, it
*uniformly rescales* it, so it can only be equivalent to some smaller
``alpha`` -- which is exactly what the earlier rounds measured the error to be
non-monotone in.

Predeclared hypotheses (written before any run of this grid)
------------------------------------------------------------
H1 (**cap = effective coefficient**).  A capped run with ``(alpha, c)`` and
    ``c / 1.3641 < alpha`` matches the **uncapped** run at
    ``alpha_eff = c / 1.3641``: mean relative front error within 2 percentage
    points, and ``max_tvc_shift/h`` equal to the cap to within 1 %.  Tested
    twice, on two different ``alpha`` (0.30 and 0.20) hitting the *same*
    ``alpha_eff = 0.147`` -- the two capped runs must agree with each other and
    with the uncapped ``alpha = 0.15`` row.
H2 (**the cap is inert above the observed maximum**).  At ``alpha = 0.10``
    (measured ``max_tvc_shift/h ~ 0.155``) a cap ``c = 0.20`` changes neither
    the shift peak nor the front error (< 1 pp, ``max_tvc_clipped == 0``).
H3 (**the cap below the surface scale destroys the mechanism**).  At
    ``alpha = 0.10`` a cap ``c = 0.10 < 1.3641 * 0.10 = 0.136`` clips the
    surface population, ``alpha_eff = 0.073``, and the front error moves toward
    the uncapped ``alpha ~ 0.073`` behaviour rather than improving.
H4 (**no resolution-independent setting follows from the cap**).  Because the
    cap only realises ``alpha_eff``, and the accuracy-optimal ``alpha`` itself
    moves with ``dx`` (``RESOLUTION_LADDER.md`` section 3), no single
    ``(alpha, c)`` pair in this grid meets the owner gate at all four
    resolutions.  If H4 is falsified the cap *is* the missing
    resolution-independent knob and that is the headline result.

Gates
-----
* owner accuracy gate: mean relative front error ``<= 15 %`` (``OWNER_GATE_REL``)
* ledger gate: ``MAE(Z/a) <= 0.15`` (``LEDGER_GATE_ABS``)
* H1 agreement gate: ``<= 2`` percentage points in mean relative error, and
  ``<= 1 %`` in ``max_tvc_shift/h``.
* digitisation band: the reference ``Z/a`` carries ``+-0.05``; a verdict that
  flips inside the band is reported as marginal, never as PASS.

Run (single core, ~5 min)::

    python benchmarks/xval_sphinx_tvc/shift_cap.py --resolutions 0.04 0.03 0.02 0.01

The payload is written incrementally (a re-run skips measured points), so a
resolution can be added later without redoing the coarse levels.
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

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(HERE))

import torch  # noqa: E402

from run import (OWNER_GATE_REL, LEDGER_GATE_ABS, load_experiment,  # noqa: E402
                 run_variant)

OUT = ROOT / "docs" / "xval_sphinx_tvc" / "data" / "shift_cap.json"

#: ``|dr|/h = SURFACE_FACTOR * alpha`` from ``SURFACE_ANCHOR.md``
#: (``30/(7 pi)``; planar free surface, ``I(h, 0) = 15/(7 pi h)``).
SURFACE_FACTOR = 30.0 / (7.0 * math.pi)
H_OVER_DX = 1.3

#: variant -> (alpha, cap in units of h, free-form note). ``cap=None`` is the
#: uncapped control.
GRID: dict[str, dict] = {
    "nocap_a010": dict(alpha=0.10, cap=None),
    "nocap_a015": dict(alpha=0.15, cap=None),
    "cap010_a010": dict(alpha=0.10, cap=0.10),
    "cap020_a010": dict(alpha=0.10, cap=0.20),
    "cap020_a020": dict(alpha=0.20, cap=0.20),
    "cap020_a030": dict(alpha=0.30, cap=0.20),
    "cap030_a030": dict(alpha=0.30, cap=0.30),
}


def variant_kwargs(spec: dict) -> dict:
    kw = dict(tvc_every=1, tvc_alpha=spec["alpha"], tvc_scope="all")
    if spec["cap"] is not None:
        kw["tvc_max_shift"] = spec["cap"]
    return kw


def alpha_eff_predicted(spec: dict) -> float:
    """(2): the coefficient the capped run should be equivalent to."""
    if spec["cap"] is None:
        return spec["alpha"]
    return min(spec["alpha"], spec["cap"] / SURFACE_FACTOR)


def environment() -> dict:
    return {
        "python": platform.python_version(),
        "torch": torch.__version__,
        "machine": platform.machine(),
        "threads": torch.get_num_threads(),
        "git_head": subprocess.run(["git", "rev-parse", "HEAD"], cwd=ROOT,
                                   capture_output=True, text=True).stdout.strip(),
    }


def load_payload() -> dict:
    if OUT.exists():
        return json.loads(OUT.read_text())
    return {"anchor": "XVAL-SPH-TRANSPORT-VELOCITY-CORRECTION",
            "reference": load_experiment()["source"],
            "surface_factor": SURFACE_FACTOR,
            "h_over_dx": H_OVER_DX,
            "gates": {"owner_rel": OWNER_GATE_REL, "ledger_abs": LEDGER_GATE_ABS},
            "grid": GRID,
            "environment": environment(),
            "records": []}


def key(dx: float, name: str) -> str:
    return f"{dx:g}|{name}"


def run_point(dx: float, name: str, spec: dict, seconds: float,
              samples: int) -> dict:
    kw = variant_kwargs(spec)
    rec = run_variant(name, kw, dx, seconds, output_samples=samples)
    # ``run_variant`` (``run.py``) supplies the front score, the distribution
    # metrics and the cap diagnostics (``max_tvc_shift_over_h``,
    # ``max_tvc_clipped``); this wrapper adds the coefficient bookkeeping of
    # equation (2).
    rec["variant"] = name
    rec["params"] = kw
    rec["alpha"] = spec["alpha"]
    rec["cap_over_h"] = spec["cap"]
    rec["alpha_eff_predicted"] = alpha_eff_predicted(spec)
    rec["alpha_eff_realised"] = (rec["max_tvc_shift_over_h"] / SURFACE_FACTOR
                                if rec["max_tvc_shift_over_h"] else 0.0)
    return rec


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--resolutions", type=float, nargs="+",
                    default=[0.04, 0.03, 0.02, 0.01])
    ap.add_argument("--variants", nargs="+", default=list(GRID))
    ap.add_argument("--seconds", type=float, default=0.45)
    ap.add_argument("--output-samples", type=int, default=240)
    ap.add_argument("--out", type=Path, default=OUT)
    args = ap.parse_args(argv)

    payload = load_payload()
    have = {key(r["dx"], r["variant"]) for r in payload["records"]}
    payload["seconds"] = args.seconds
    payload["environment"] = environment()

    for dx in args.resolutions:
        for name in args.variants:
            k = key(dx, name)
            if k in have:
                print(f"skip {k} (already measured)", flush=True)
                continue
            t0 = time.time()
            rec = run_point(dx, name, GRID[name], args.seconds,
                            args.output_samples)
            rec["dx"] = dx
            sc = rec["score"]
            print(f"dx={dx:<5} {name:<14} alpha={rec['alpha']:.2f} "
                  f"cap={rec['cap_over_h']} "
                  f"rel={sc.get('mean_rel_error', float('nan')):.3f} "
                  f"mae={sc.get('mae', float('nan')):.4f} "
                  f"shift/h={rec['max_tvc_shift_over_h']:.3f} "
                  f"eff={rec['alpha_eff_realised']:.3f} "
                  f"clipped={rec.get('max_tvc_clipped', 0.0):.3f} "
                  f"({time.time() - t0:.1f}s)", flush=True)
            payload["records"].append(rec)
            args.out.parent.mkdir(parents=True, exist_ok=True)
            args.out.write_text(json.dumps(payload, indent=2))
    print(f"wrote {args.out} ({len(payload['records'])} records)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())