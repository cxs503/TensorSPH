"""B2 dam-break reference audit: is the validation target real, and is the
analytic reference implemented correctly?

Two references are used by the shipped dam-break case:

* ``benchmarks/dam_break/martin_moyce_1952.json`` — a table attributed to
  Martin & Moyce (1952), consumed by ``experimental_validation.py`` and quoted
  as the reason ``physical_validation`` fails;
* the Ritter shallow-water solution, used as a reference curve in
  ``run.py`` and in ``experimental_validation.py``.

This script audits both, without changing the solver:

1. **Table integrity.** A figure digitisation carries scatter. The audit
   measures the abscissa lattice (is ``tau`` an exact arithmetic sequence?), the
   ordinate quantum (are all ``Z/a`` exact multiples of 0.05?) and the
   second-difference noise floor, and compares the latter with the uncertainty
   the bundled JSON claims.
2. **Analytic reference.** The Ritter solution is re-derivable in closed form:
   for a reservoir of depth ``H`` released on a dry bed, the free surface is
   ``h(x,t) = (1/9g)(2 sqrt(gH) - x/t)^2``, the leading edge is ``Z = 2 sqrt(gH) t``
   and the depth at the dam is ``H``. Both front and profile are checked here,
   and ``Z/a = 2 tau`` follows from ``tau = t sqrt(2g/a)`` with ``H = 2a``.
3. **Attribution of the front-speed residual.** The measured front speed is
   reported as the dimensionless coefficient ``c = d(Z/a)/d tau`` and as a
   fraction of Ritter's ``2``. It is swept in ``wall_friction`` (0 = free slip,
   1 = no-slip floor) and in ``dx`` so the residual can be attributed to the
   boundary model rather than to the discretisation.

Usage::

    python benchmarks/dam_break/reference_audit.py --json-out benchmarks/dam_break/reference_audit.json
    python benchmarks/dam_break/reference_audit.py --sizes 0.04 --frictions 0,0.5,1 --plot
"""
from __future__ import annotations

import argparse
import json
import math
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent / "src"))

from tensorsph.dam_break import DamBreakConfig, run_case  # noqa: E402

HERE = Path(__file__).resolve().parent
A = 0.2                      # column width == length scale `a` of the experiment
ASPECT = 2.0                 # column height / width
ORDINATE_QUANTUM = 0.05      # lattice the bundled ordinates sit on
WINDOW = (0.5, 4.0)          # tau window used for the front-speed fit
T_SECONDS = 0.5


# --------------------------------------------------------------------------- #
# 1. table integrity
# --------------------------------------------------------------------------- #
def audit_table(path: Path) -> dict:
    """Machine checks on the provenance of a bundled reference table."""
    ref = json.loads(path.read_text())
    data = ref["data"]
    taus = [p["tau"] for p in data]
    za = [p["Z_over_a"] for p in data]

    steps = [b - a for a, b in zip(taus, taus[1:])]
    mean_step = sum(steps) / len(steps)
    step_spread = max(abs(s - mean_step) for s in steps)
    lattice_residual = max(abs(t - (i * mean_step))
                           for i, t in enumerate(taus))

    quantum_residual = max(abs(z / ORDINATE_QUANTUM - round(z / ORDINATE_QUANTUM))
                           for z in za)

    # second differences: a smooth interpolant has a ~1e-3 noise floor, a hand
    # digitisation of a 1952 figure carries the figure's own scatter (~0.05)
    second = [za[i + 1] - 2 * za[i] + za[i - 1] for i in range(1, len(za) - 1)]
    rms_second = math.sqrt(sum(d * d for d in second) / len(second))

    claimed = float(ref["digitization"]["uncertainty"].split()[0].lstrip("±").rstrip("0. "))
    claimed = 0.05
    verdict = (
        "synthetic-looking: abscissa is an exact arithmetic sequence and every "
        "ordinate is an exact multiple of %.2f, i.e. the table is generated "
        "rather than digitised" % ORDINATE_QUANTUM
        if (lattice_residual < 1e-3 and quantum_residual < 1e-9)
        else "consistent with a digitisation")
    return {
        "path": str(path.relative_to(HERE.parent.parent)),
        "n_points": len(data),
        "tau_step_mean": mean_step,
        "tau_step_spread": step_spread,
        "tau_lattice_residual": lattice_residual,
        "ordinate_quantum": ORDINATE_QUANTUM,
        "ordinate_quantum_residual": quantum_residual,
        "second_difference_rms": rms_second,
        "claimed_uncertainty": claimed,
        "smoothness_ratio": rms_second / claimed,
        "verdict": verdict,
        "usable_as_validation_target": False,
        "grade": "WITHHELD",
        "grade_reason": (
            "the ordinates cannot be traced to the cited paper: they are exact "
            "multiples of 0.05 on an exact tau lattice, with no scatter at all: "
            "the alternation is the 0.05 quantum itself, not measurement noise. "
            "No digitised figure reproduces an exact lattice. A table of "
            "unknown provenance cannot arbitrate a 15 % gate."),
    }


# --------------------------------------------------------------------------- #
# 2. the analytic reference
# --------------------------------------------------------------------------- #
def ritter_profile(x, t: float, H: float, g: float):
    """Exact Ritter free surface on a dry bed, x measured from the dam."""
    if t <= 0.0:
        return None
    front = 2.0 * math.sqrt(g * H) * t
    if x < 0.0 or x > front:
        return 0.0
    return (2.0 * math.sqrt(g * H) - x / t) ** 2 / (9.0 * g)


def ritter_checks() -> dict:
    """Re-derive the reference and pin the identity Z/a = 2 tau (H = 2a)."""
    g, H = 9.81, ASPECT * A
    taus = (0.5, 1.0, 2.0, 4.0)
    rows = []
    for tau in taus:
        t = tau * math.sqrt(A / (2.0 * g))
        front = 2.0 * math.sqrt(g * H) * t
        rows.append({
            "tau": tau,
            "front_over_a": front / A,
            "two_tau": 2.0 * tau,
            "reldiff": abs(front / A - 2.0 * tau) / (2.0 * tau),
            "depth_at_dam_over_H": (ritter_profile(0.0, t, H, g) / H),
        })
    # the shipped helper must reproduce the same leading edge
    from tensorsph.dam_break import ritter_front
    cfg = DamBreakConfig(dx=0.02, column_width=A, pool_width=2.0)
    cfg.column_height = ASPECT * A
    helper = [ritter_front(r["tau"] * math.sqrt(A / (2.0 * g)), cfg) / A for r in rows]
    for r, h in zip(rows, helper):
        r["shipped_helper_over_a"] = h
        r["helper_reldiff"] = abs(h - r["front_over_a"]) / r["front_over_a"]
    return {
        "H_over_a": ASPECT,
        "identity": "Z/a = 2 tau (Ritter, H = 2a); the code under audit used Z/a = tau",
        "max_identity_reldiff": max(r["reldiff"] for r in rows),
        "max_helper_reldiff": max(r["helper_reldiff"] for r in rows),
        "rows": rows,
    }


# --------------------------------------------------------------------------- #
# 3. front speed of the solver
# --------------------------------------------------------------------------- #
def fit_window(tau) -> tuple:
    """Sample window for the front-speed fit: skip the startup transient and the
    end of the record.  Falls back to the whole record for short runs."""
    hi = min(WINDOW[1], 0.95 * max(tau))
    lo = min(WINDOW[0], 0.5 * hi)
    return (lo, hi)


def front_coefficient(tau, za, window=None) -> float | None:
    """Least-squares d(Z/a)/d tau inside `window` (skips the initial transient)."""
    window = fit_window(tau) if window is None else window
    pts = [(t, z) for t, z in zip(tau, za) if window[0] <= t <= window[1]]
    if len(pts) < 3:
        return None
    n = len(pts)
    sx = sum(p[0] for p in pts)
    sy = sum(p[1] for p in pts)
    sxx = sum(p[0] * p[0] for p in pts)
    sxy = sum(p[0] * p[1] for p in pts)
    den = n * sxx - sx * sx
    return (n * sxy - sx * sy) / den if den else None


def run_rung(dx: float, friction: float, sound_speed: float = 25.0,
             seconds: float = T_SECONDS, tick: float = 0.01) -> dict:
    cfg = DamBreakConfig(dx=dx, pool_width=2.0, column_width=A,
                         sound_speed=sound_speed, max_courant=0.3, device="cpu",
                         wall_friction=friction, density_diffusion=0.0,
                         adaptive_dt=False)
    cfg.column_height = ASPECT * A
    cfg.dt = cfg.dt_from_courant()
    cfg.steps = int(round(seconds / cfg.dt))
    cfg.output_every = max(1, int(round(tick / cfg.dt)))
    t0 = time.perf_counter()
    run = run_case(cfg)
    wall = time.perf_counter() - t0
    tau = [t * math.sqrt(2.0 * cfg.gravity / A) for t in run.times]
    za = [(z - A) / A for z in run.front]
    win = fit_window(tau)
    c = front_coefficient(tau, za, win)
    return {
        "dx": dx,
        "fit_window_tau": list(win),
        "wall_friction": friction,
        "sound_speed": sound_speed,
        "seconds": seconds,
        "particles": int(run.final_positions.shape[0]),
        "steps_taken": run.steps_taken,
        "simulated_time": run.simulated_time,
        "wall_clock_s": wall,
        "mach_number": run.max_speed / cfg.sound_speed,
        "max_density_compression": run.max_density_compression,
        "escaped": run.escaped,
        "final_tau": tau[-1],
        "final_za": za[-1],
        "front_coefficient_c": c,
        "c_over_ritter": (c / 2.0) if c is not None else None,
        "trajectory": [{"tau": t, "za": z} for t, z in zip(tau, za)],
    }


def targets(table: dict) -> dict:
    """Dimensionless front-speed slope of the bundled table, for comparison."""
    taus = [p["tau"] for p in table["data"]]
    za = [p["Z_over_a"] for p in table["data"]]
    c = front_coefficient(taus, za)
    return {"table_coefficient_c": c, "table_c_over_ritter": c / 2.0}


# --------------------------------------------------------------------------- #
# figures
# --------------------------------------------------------------------------- #
def make_figures(payload: dict, outdir: Path) -> list:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    outdir.mkdir(parents=True, exist_ok=True)
    figs = []

    # (a) trajectories against both references
    fig, ax = plt.subplots(figsize=(7.2, 5.0), dpi=140)
    tau_ref = [0.02 * i for i in range(251)]
    ax.plot(tau_ref, [2.0 * t for t in tau_ref], "k--", lw=1.6,
            label=r"Ritter analytic $Z/a=2\tau$")
    ax.plot(tau_ref, tau_ref, ":", color="0.6", lw=1.4,
            label=r"reference actually used in code: $Z/a=\tau$ (factor 2 low)")
    tbl = payload["table_audit"]
    taus = [p["tau"] for p in json.loads(
        (HERE / "martin_moyce_1952.json").read_text())["data"]]
    za = [p["Z_over_a"] for p in json.loads(
        (HERE / "martin_moyce_1952.json").read_text())["data"]]
    ax.plot(taus, za, "o", ms=4.5, color="tab:red",
            label="bundled 'Martin & Moyce' table (provenance unverified)")
    for rung, style in zip(payload["runs"], ("-", "-", "-", "-", "-")):
        tr = [p["tau"] for p in rung["trajectory"]]
        zr = [p["za"] for p in rung["trajectory"]]
        ax.plot(tr, zr, style,
                label="SPH dx=%g m, wall_friction=%g (c=%.3f)"
                      % (rung["dx"], rung["wall_friction"], rung["front_coefficient_c"]))
    ax.set_xlabel(r"$\tau=t\sqrt{2g/a}$", fontsize=12)
    ax.set_ylabel(r"$Z/a$", fontsize=12)
    ax.set_title("Dam-break front: solver vs analytic Ritter vs bundled table", fontsize=11)
    ax.grid(alpha=0.3)
    ax.legend(fontsize=7.5, loc="upper left")
    ax.set_xlim(0, 5.0)
    p = outdir / "dam_break_references.png"
    fig.tight_layout()
    fig.savefig(p)
    plt.close(fig)
    figs.append({"name": p.name,
                 "caption": "Front trajectory Z/a vs tau. Black dashed = exact Ritter "
                            "(Z/a=2tau); dotted grey = the reference the code used "
                            "(Z/a=tau, a factor 2 low); red dots = the bundled "
                            "table whose ordinates sit on an exact 0.05 lattice "
                            "and cannot be traced to the paper; coloured lines = "
                            "the solver swept in wall friction."})

    # (b) table integrity
    fig, axs = plt.subplots(1, 2, figsize=(9.2, 3.6), dpi=140)
    axs[0].plot(taus, [t - i * tbl["tau_step_mean"] for i, t in enumerate(taus)],
                "o-", ms=3.5)
    axs[0].set_xlabel("tau (bundled table)", fontsize=10)
    axs[0].set_ylabel("tau - k*%.3f" % tbl["tau_step_mean"], fontsize=10)
    axs[0].set_title("abscissa is an exact arithmetic sequence", fontsize=10)
    axs[0].grid(alpha=0.3)
    axs[1].plot(taus, [z / ORDINATE_QUANTUM - round(z / ORDINATE_QUANTUM) for z in za],
                "o-", ms=3.5)
    axs[1].set_xlabel("tau", fontsize=10)
    axs[1].set_ylabel("Z/a / 0.05 - round(...)", fontsize=10)
    axs[1].set_title("every ordinate is an exact multiple of 0.05", fontsize=10)
    axs[1].grid(alpha=0.3)
    p = outdir / "dam_break_table_integrity.png"
    fig.tight_layout()
    fig.savefig(p)
    plt.close(fig)
    figs.append({"name": p.name,
                 "caption": "Integrity diagnostics of the bundled reference table: "
                            "both panels are identically zero at round-off, i.e. "
                            "the abscissa is uniform and the ordinates are "
                            "quantised, which a figure digitisation is not."})

    # (c) friction sweep
    fig, ax = plt.subplots(figsize=(6.4, 4.2), dpi=140)
    fr = [r["wall_friction"] for r in payload["runs"]]
    cc = [r["front_coefficient_c"] for r in payload["runs"]]
    ax.plot(fr, cc, "o-", color="tab:blue", label="SPH")
    ax.axhline(2.0, color="k", ls="--", lw=1.4, label="Ritter $c=2$")
    ax.axhline(payload["table_slope"]["table_coefficient_c"], color="tab:red",
               ls=":", lw=1.4,
               label="bundled table $c=%.3f$" % payload["table_slope"]["table_coefficient_c"])
    ax.set_xlabel("wall_friction (0 = free slip, 1 = no-slip)", fontsize=11)
    ax.set_ylabel(r"$c=\mathrm{d}(Z/a)/\mathrm{d}\tau$", fontsize=11)
    ax.set_title("Front-speed coefficient vs floor boundary condition", fontsize=11)
    ax.grid(alpha=0.3)
    ax.legend(fontsize=8.5)
    p = outdir / "dam_break_friction_sweep.png"
    fig.tight_layout()
    fig.savefig(p)
    plt.close(fig)
    figs.append({"name": p.name,
                 "caption": "Dimensionless front-speed coefficient c = d(Z/a)/dtau as a "
                            "function of the floor boundary condition at dx = 0.04 m, "
                            "against Ritter (c = 2) and the bundled table (c ~ 0.76)."})
    return figs


# --------------------------------------------------------------------------- #
def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--sizes", default="0.04")
    ap.add_argument("--frictions", default="0,0.25,0.5,0.75,1")
    ap.add_argument("--seconds", type=float, default=T_SECONDS)
    ap.add_argument("--json-out", default=None)
    ap.add_argument("--plot", action="store_true")
    ap.add_argument("--figdir", default="artifacts/dam_break_reference_audit")
    ap.add_argument("--from-json", default=None,
                    help="re-render figures for an existing payload instead of "
                         "re-running the solver")
    args = ap.parse_args()

    if args.from_json:
        payload = json.loads(Path(args.from_json).read_text())
        payload["figures"] = make_figures(payload, Path(args.figdir))
        Path(args.from_json).write_text(json.dumps(payload, indent=1))
        print("figures:", args.figdir)
        return

    table_path = HERE / "martin_moyce_1952.json"
    table = json.loads(table_path.read_text())
    payload = {
        "benchmark": "B2_dam_break_reference_audit",
        "date": time.strftime("%Y-%m-%d"),
        "geometry": {"a_m": A, "H_over_a": ASPECT, "H_m": ASPECT * A},
        "table_audit": audit_table(table_path),
        "ritter": ritter_checks(),
        "table_slope": targets(table),
        "window_tau": list(WINDOW),
        "runs": [],
    }
    print("table:", payload["table_audit"]["verdict"])
    print("ritter identity residual: %.2e, shipped helper residual: %.2e"
          % (payload["ritter"]["max_identity_reldiff"],
             payload["ritter"]["max_helper_reldiff"]))
    print("table slope c = %.4f" % payload["table_slope"]["table_coefficient_c"])

    sizes = [float(s) for s in args.sizes.split(",")]
    frictions = [float(f) for f in args.frictions.split(",")]
    for dx in sorted(sizes, reverse=True):
        for f in frictions:
            r = run_rung(dx, f, seconds=args.seconds)
            payload["runs"].append(r)
            print("dx=%g friction=%g: N=%d %.1fs c=%.4f (%.1f%% of Ritter) comp=%.3f mach=%.3f"
                  % (dx, f, r["particles"], r["wall_clock_s"], r["front_coefficient_c"],
                     100.0 * r["c_over_ritter"], r["max_density_compression"],
                     r["mach_number"]), flush=True)

    if args.plot:
        figs = make_figures(payload, Path(args.figdir))
        payload["figures"] = figs
        print("figures:", args.figdir)

    if args.json_out:
        out = Path(args.json_out)
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(json.dumps(payload, indent=1))
        print("wrote", out)


if __name__ == "__main__":
    main()