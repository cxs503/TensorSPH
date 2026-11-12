"""Is the closure read-out inadmissible because the drop has not settled yet?

Motivation
----------
`docs/verification_laplace_pressure_closure.md` measured every closure variant
of the Laplace drop over **1.5 capillary times**.  With the free-surface pressure
closure on (interior donors, 2 layers) the read-out is inside the +/-15 % band at
every ladder point, but at ``(dx = 2 mm, c0 = 20 m/s)`` the *same* run is
inadmissible: ``v_max/v_ref = 0.388`` (3.9x Adami et al. 2012's static-drop
limit of 0.1) and the final snapshot is +142.8 % off -- "the surface is
breathing, so the time average is not a static Laplace measurement"
(`benchmarks/laplace_pressure_closure/verdict_audit.py`, rule 2).

That leaves two mutually exclusive explanations, and the evidence does not
distinguish them because no run was ever advanced past 1.5 ``t_sigma``:

  H_settle  the breathing is a *startup transient*.  The drop is initialised at
            ``p = 0`` (Tait is normalised at ``rho0``) while it must carry
            ``sigma/R``; releasing it compresses the core and excites a
            capillary/breathing mode of period ~ ``t_sigma``.  1.5 ``t_sigma``
            is one and a half periods, i.e. a short window, and the tail average
            is taken over the transient.  Lengthening the window (the viscosity
            of this case is 0.01 m^2/s, 100x water, so the mode is strongly
            damped) should make the spurious current decay and the read-out both
            in-band *and* admissible.

  H_persistent  the spurious current is the persistent parasitic mode of
            sum-density wcSPH + CSF (Adami et al. 2012 sec. 4).  Then the
            amplitude does not decay with time, the closure variant stays
            inadmissible at that point, and the Laplace capability stays
            WITHHELD for a *scheme* reason rather than a measurement reason.

The discriminator is the decay of ``v_max(t)``, which is cheap to read: the
transient hypothesis predicts a monotone-ish fall after the first period, the
persistent hypothesis predicts a plateau.

What is measured
----------------
The drop of ``benchmarks/laplace_pressure_floor`` (`relax()` is reused as is, so
the case definition cannot drift), advanced for ``--t-end`` capillary times **at
the stable dt** (``max_steps`` is deliberately large: `relax()` trades dt for
step count when ``max_steps`` binds, which would silently change the scheme).
Per case the record holds

* ``tail40`` / ``tail25`` -- ``p_core - 0`` averaged over the last 40 % / 25 %
  of the window, the C1-corrected absolute definition of the jump;
* ``vmax`` -- maximum spurious current over the *late* window (conservative;
  Adami's criterion is on the maximum) and its mean, plus the early-window
  maximum for the decay ratio;
* ``final_snapshot`` -- the last instantaneous ``p_core``, kept because
  `verdict_audit.py` treats a |final| > 20 % in-band average as "breathing".

Usage
-----
    PYTHONPATH=src python benchmarks/laplace_pressure_closure/settling.py \
        --dx 0.002 --c0 20 --t-end 6

One record per ``(dx, c0)`` in ``raw/settling_dx<N>mm_c<c0>.json``; cases are
appended as they finish, so a partial run still leaves usable numbers.
"""
from __future__ import annotations

import argparse
import importlib.util
import json
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
RAW = HERE / "raw"

#: The case set that matters for the open gate: the candidate closure variant
#: (the only one that is inside the band) and its control.
CASES = (
    ("closure_on_interior_layers2", dict(closure=True, donors="interior", layers=2)),
    ("closure_off", dict(closure=False)),
)

#: The four acceptance points of the P0 in ``backlog_update_m2971.md``.
ACCEPTANCE_POINTS = ((0.002, 20.0), (0.002, 5.0), (0.0015, 20.0), (0.0015, 5.0))


def load_floor():
    spec = importlib.util.spec_from_file_location(
        "lpf", HERE.parent / "laplace_pressure_floor" / "run.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def out_path(dx: float, c0: float) -> Path:
    return RAW / f"settling_dx{dx * 1e3:.4g}mm_c{c0:g}.json"


def _window(trace: dict, lo: float, hi: float) -> int:
    n = len(trace["t"])
    i0, i1 = int(lo * n), max(int(lo * n) + 1, int(hi * n))
    return slice(i0, i1)


def summarise(run: dict, v_ref_limit: float = 0.1) -> dict:
    """Late-window statistics of one relaxed drop, from its own trace."""
    trace = run["trace"]
    t = trace["t"]
    t_end = max(t) if t else float("nan")

    def mean(key, lo, hi):
        sl = _window(trace, lo, hi)
        vals = trace[key][sl]
        return sum(vals) / len(vals)

    def vmax(key, lo, hi):
        sl = _window(trace, lo, hi)
        return max(trace[key][sl])

    analytic = None  # filled by the caller from lpf.DP_ANALYTIC via `score`
    late_max = vmax("vmax_over_vref", 0.75, 1.0)
    early_max = vmax("vmax_over_vref", 0.0, 0.20)
    return {
        "tail40_Pa": mean("absolute", 0.60, 1.0),
        "tail25_Pa": mean("absolute", 0.75, 1.0),
        "tail40_rho_core_rel": mean("rho_core_rel", 0.60, 1.0),
        "vmax_over_vref_early20": early_max,
        "vmax_over_vref_late_max": late_max,
        "vmax_over_vref_late_mean": mean("vmax_over_vref", 0.75, 1.0),
        "decay_ratio": late_max / early_max if early_max > 0 else float("inf"),
        "passes_adami_late": bool(late_max < v_ref_limit),
        "final_abs_Pa": trace["absolute"][-1] if trace["absolute"] else float("nan"),
        "t_over_t_sigma_max": t_end,
        "n_samples": len(t),
        "vmax_trace_over_t_sigma": list(zip(t, trace["vmax_over_vref"])),
        "p_core_trace_over_t_sigma": list(zip(t, trace["absolute"])),
        "analytic": analytic,
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dx", type=float, default=0.002)
    ap.add_argument("--c0", type=float, default=20.0)
    ap.add_argument("--t-end", type=float, default=6.0,
                    help="end time in units of the capillary time t_sigma")
    ap.add_argument("--max-steps", type=int, default=20000,
                    help="must not bind, otherwise relax() coarsens dt")
    ap.add_argument("--sample-every", type=int, default=0,
                    help="0 = pick ~60 samples over the window")
    ap.add_argument("--cases", nargs="+", default=[c[0] for c in CASES])
    args = ap.parse_args()

    lpf = load_floor()
    out = out_path(args.dx, args.c0)
    out.parent.mkdir(parents=True, exist_ok=True)
    record = {"dx_m": args.dx, "sound_speed": args.c0,
              "analytic_dp_Pa": lpf.DP_ANALYTIC,
              "t_end_over_t_sigma": args.t_end, "cases": []}
    if out.exists():
        previous = json.loads(out.read_text())
        if (previous.get("dx_m"), previous.get("sound_speed")) == (args.dx, args.c0):
            record["cases"] = previous.get("cases", [])
        else:
            print(f"ignoring {out.name}: recorded point differs", flush=True)

    # dt is chosen by relax() from stable_dt(); the sample interval that gives
    # ~60 samples depends on the step count, known only after the first call, so
    # it is derived from an analytic estimate of the step count here.
    sample_every = args.sample_every
    for name, kw in CASES:
        if name not in args.cases:
            continue
        t0 = time.time()
        run = lpf.relax(args.dx, args.c0, t_end_over_t_sigma=args.t_end,
                        max_steps=args.max_steps,
                        sample_every=sample_every or 100, **kw)
        if sample_every == 0:
            # resample once with the right interval if the default over-sampled
            nsamp = len(run["trace"]["t"])
            if nsamp > 200:
                sample_every = max(1, run["steps"] // 60)
                t0 = time.time()
                run = lpf.relax(args.dx, args.c0, t_end_over_t_sigma=args.t_end,
                                max_steps=args.max_steps,
                                sample_every=sample_every, **kw)
        summary = summarise(run)
        summary["analytic"] = lpf.DP_ANALYTIC
        dp = lpf.DP_ANALYTIC
        summary["tail40_rel_error_pct"] = (summary["tail40_Pa"] - dp) / dp * 100.0
        summary["tail25_rel_error_pct"] = (summary["tail25_Pa"] - dp) / dp * 100.0
        summary["final_rel_error_pct"] = (summary["final_abs_Pa"] - dp) / dp * 100.0
        entry = {
            "case": name, "config": kw,
            "wall_clock_s": round(time.time() - t0, 1),
            "steps": run["steps"], "dt_s": run["dt_s"],
            "n_particles": run["n_particles"],
            "dt_was_clipped": bool(run["steps"] >= args.max_steps),
            **summary,
        }
        record["cases"] = [c for c in record["cases"] if c["case"] != name] + [entry]
        out.write_text(json.dumps(record, indent=2))
        print(f"{name:30s} t_end {entry['t_over_t_sigma_max']:.2f} t_sigma "
              f"({entry['steps']} steps)  tail40 {entry['tail40_rel_error_pct']:+7.1f} %  "
              f"tail25 {entry['tail25_rel_error_pct']:+7.1f} %  "
              f"final {entry['final_rel_error_pct']:+7.1f} %  "
              f"vmax early {entry['vmax_over_vref_early20']:.3f} -> "
              f"late {entry['vmax_over_vref_late_max']:.3f} "
              f"({'ADMISSIBLE' if entry['passes_adami_late'] else 'inadmissible'})  "
              f"{entry['wall_clock_s']}s", flush=True)
    print("wrote", out)


if __name__ == "__main__":
    main()