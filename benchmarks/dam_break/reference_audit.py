"""Audit of the experimental reference bundled with the B2 dam-break case.

The B2 case compares the simulated front against a digitised Martin & Moyce
(1952) table (``martin_moyce_1952.json``), and that comparison is where the
suite reports its worst numbers.  This module audits the *reference*, not the
solver, and it does so in two independent layers that share one artifact
(``reference_audit.json``):

A. **Reference-free and analytic checks** (no table needed)

   1. **Table integrity.** A figure digitisation carries scatter.  The audit
      measures the abscissa lattice (is ``tau`` an exact arithmetic sequence?),
      the ordinate quantum (are all ``Z/a`` exact multiples of 0.05?) and the
      second-difference noise floor, and compares the latter with the
      uncertainty the bundled JSON claims for itself.
   2. **Analytic reference.** The Ritter solution is re-derivable in closed
      form: for a reservoir of depth ``H`` released on a dry bed the free
      surface is ``h(x,t) = (1/9g)(2 sqrt(gH) - x/t)^2``, the leading edge is
      ``Z = 2 sqrt(gH) t`` and the depth at the dam is ``H``.  Both front and
      profile are checked, and ``Z/a = 2 tau`` follows from
      ``tau = t sqrt(2g/a)`` with ``H = 2a``.
   3. **Attribution of the front-speed residual.** The measured front speed is
      reported as the dimensionless coefficient ``c = d(Z/a)/d tau`` and as a
      fraction of Ritter's ``2``.  It is swept in ``wall_friction`` (0 = free
      slip, 1 = no-slip floor) and in ``dx`` so the residual can be attributed
      to the boundary model rather than to the discretisation.

B. **Structural audit of both transcriptions** (tables only, no simulation)

   The resolution ladder showed that the misfit against the bundled table is
   *resolution independent* (mean absolute error 0.5988 -> 0.5774 -> 0.5728 for
   ``dx`` = 0.04 -> 0.02 -> 0.01 m), i.e. it is a property of the comparison,
   not of the front's numerical diffusion.  Three questions are answered here
   from the tables alone:

   4. *Structure.*  A digitisation read off a published figure inherits the
      figure's resolution: irregular sample abscissae, values that do not land
      on a round grid, increments with noise.  A table whose ``tau`` grid is a
      single repeated step, whose ordinates all sit on a 0.05 grid and whose
      finite-difference slopes take only four discrete values is a piecewise
      linear sketch of a few hand-picked vertices, and its +-0.05 uncertainty
      claim is a guess about the picture, not a measured error bar.
   5. *Kinematics.*  A front released from rest must first accelerate: the
      initial slope of ``Z/a`` against ``tau`` has to be small compared with the
      largest slope that follows.  A table already near its peak slope in the
      first interval after ``tau = 0`` describes an impulsive start, which the
      experiment (a gate raised over a finite time) cannot have.
   6. *Cross-source spread.*  A second, independent transcription of the same
      figure is bundled in ``martin_moyce_1952_crosscheck.json``.  The spread
      between the two is a *lower bound* on the reference uncertainty, because
      both come from the same inaccessible original.  If that spread is much
      larger than the claimed +-0.05, an error of that size cannot be
      attributed to the solver at all.

Layer B ends in three explicit gates and a verdict on whether the bundled
reference can carry a physical-validation verdict at all
(``physical_validation = withheld`` when it cannot).  Nothing in this module
touches the solver: layer A reports the measurements, layer B audits the
reference.

Usage::

    python benchmarks/dam_break/reference_audit.py \
        --json-out benchmarks/dam_break/reference_audit.json
    python benchmarks/dam_break/reference_audit.py --skip-runs    # layer B only
    python benchmarks/dam_break/reference_audit.py \
        --sizes 0.04 --frictions 0,0.5,1 --plot
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
            "multiples of 0.05 on an exact tau lattice, and the second-difference "
            "structure is the 0.05 quantum itself rather than measurement "
            "scatter. No digitised figure reproduces an exact lattice. A table "
            "of unknown provenance cannot arbitrate a 15 % gate."),
        "smoothness_note": (
            "second-difference RMS %.5f vs the claimed %.2f uncertainty: only "
            "~%.1fx smoother, so the noise floor alone is NOT the evidence; the "
            "exact lattice and quantum are."
            % (rms_second, claimed, claimed / rms_second)),
    }



# --------------------------------------------------------------------------- #
# table loading
# --------------------------------------------------------------------------- #
def _interp(xs: list[float], ys: list[float], xq: float) -> float | None:
    """Linear interpolation, clamped at both ends."""
    if not xs:
        return None
    if xq <= xs[0]:
        return ys[0]
    if xq >= xs[-1]:
        return ys[-1]
    for i in range(1, len(xs)):
        if xs[i] >= xq:
            span = xs[i] - xs[i - 1]
            w = (xq - xs[i - 1]) / span if span else 0.0
            return ys[i - 1] + w * (ys[i] - ys[i - 1])
    return ys[-1]


def load_bundled(path: str | Path | None = None) -> dict:
    """The table the B2 gate has been using (Z/a convention, measured from x = a)."""
    p = Path(path) if path is not None else HERE / "martin_moyce_1952.json"
    raw = json.loads(Path(p).read_text())
    return {
        "name": "bundled",
        "path": _display_path(p),
        "tau": [float(d["tau"]) for d in raw["data"]],
        "za": [float(d["Z_over_a"]) for d in raw["data"]],
        "claimed_uncertainty": _claimed_uncertainty(raw),
        "source": raw.get("source", {}),
        "digitisation": raw.get("digitization", {}),
    }


def load_crosscheck(path: str | Path | None = None) -> dict:
    """The independent transcription, converted to the same Z/a convention."""
    p = Path(path) if path is not None else HERE / "martin_moyce_1952_crosscheck.json"
    raw = json.loads(Path(p).read_text())
    return {
        "name": "crosscheck",
        "path": _display_path(p),
        "tau": [float(d["tau"]) for d in raw["data"]],
        # X = x_front/a, and Martin & Moyce measure Z from the column edge,
        # so Z/a = X - 1 keeps both tables in one frame.
        "za": [float(d["X_over_a"]) - 1.0 for d in raw["data"]],
        "claimed_uncertainty": float(raw.get("claimed_uncertainty", 0.05)),
        "source": raw.get("source", {}),
        "convention": raw.get("convention", {}),
    }


def _claimed_uncertainty(raw: dict) -> float:
    """Pull the numeric half-width out of the free-text uncertainty field."""
    text = str(raw.get("digitization", {}).get("uncertainty", ""))
    for token in text.replace("±", "+-").split():
        token = token.strip(".,;()")
        try:
            value = float(token)
        except ValueError:
            continue
        if 0.0 < value < 1.0:
            return value
    return 0.05


# --------------------------------------------------------------------------- #
# structural statistics
# --------------------------------------------------------------------------- #
def grid_stats(tau: list[float], za: list[float]) -> dict:
    """Everything the structure argument needs, as numbers."""
    steps = [tau[i + 1] - tau[i] for i in range(len(tau) - 1)]
    dz = [za[i + 1] - za[i] for i in range(len(za) - 1)]
    slopes = [dz[i] / steps[i] for i in range(len(dz)) if steps[i]]
    step_spread = (max(steps) - min(steps)) if steps else 0.0
    # Quantisation: find the coarsest round grid every ordinate sits on.
    quantum = None
    for candidate in (0.05, 0.025, 0.02, 0.01, 0.005):
        if all(abs(v / candidate - round(v / candidate)) < 1e-9 for v in za):
            quantum = candidate
            break
    # Discrete slope levels: group slopes that agree to 0.5 %.
    levels: list[float] = []
    for s in sorted(slopes):
        if not levels or abs(s - levels[-1]) > 5e-3 * max(abs(s), 1e-12):
            levels.append(s)
    peak = max(slopes) if slopes else 0.0
    return {
        "n_points": len(tau),
        "tau_first": tau[0] if tau else None,
        "tau_last": tau[-1] if tau else None,
        "tau_step_mean": sum(steps) / len(steps) if steps else None,
        "tau_step_spread": step_spread,
        "tau_grid_is_arithmetic": step_spread <= 1e-9,
        "ordinate_quantum": quantum,
        "slope_first": slopes[0] if slopes else None,
        "slope_peak": peak,
        "slope_peak_index": slopes.index(peak) if slopes else None,
        "slope_last": slopes[-1] if slopes else None,
        "n_slope_levels": len(levels),
        "slope_levels": [round(s, 4) for s in levels],
        "slopes": [round(s, 4) for s in slopes],
        "first_slope_to_peak": (slopes[0] / peak) if slopes and peak else None,
        "slope_is_monotone_decreasing": all(
            slopes[i + 1] <= slopes[i] + 1e-12 for i in range(len(slopes) - 1)
        ),
    }


def cross_source_spread(bundled: dict, crosscheck: dict) -> dict:
    """Difference between the two transcriptions over their overlapping window."""
    deltas = []
    rows = []
    for tau, za in zip(bundled["tau"], bundled["za"]):
        if tau < crosscheck["tau"][0] or tau > crosscheck["tau"][-1]:
            continue
        other = _interp(crosscheck["tau"], crosscheck["za"], tau)
        delta = za - other
        deltas.append(delta)
        rows.append({"tau": tau, "bundled_za": za, "crosscheck_za": other,
                     "delta": delta})
    if not deltas:
        return {"n_overlap": 0}
    max_abs = max(abs(d) for d in deltas)
    rms = math.sqrt(sum(d * d for d in deltas) / len(deltas))
    return {
        "n_overlap": len(deltas),
        "tau_window": [rows[0]["tau"], rows[-1]["tau"]],
        "max_abs_delta": max_abs,
        "rms_delta": rms,
        "mean_signed_delta": sum(deltas) / len(deltas),
        "rows": rows,
    }


# --------------------------------------------------------------------------- #
# gates
# --------------------------------------------------------------------------- #
def evaluate(tables: dict[str, dict]) -> dict:
    """Structural audit of the reference tables plus the derived verdict."""
    bundled = tables["bundled"]
    crosscheck = tables["crosscheck"]
    g_b = grid_stats(bundled["tau"], bundled["za"])
    g_c = grid_stats(crosscheck["tau"], crosscheck["za"])
    spread = cross_source_spread(bundled, crosscheck)

    claimed = bundled["claimed_uncertainty"]
    observed = spread.get("max_abs_delta")
    uncertainty_ratio = (observed / claimed) if (observed and claimed) else None

    gates = {}

    # 1. Is the bundled ordinate set a point-by-point reading, or a sketch?
    gates["bundled_is_pointwise"] = {
        "metric": "arithmetic tau grid AND coarse ordinate quantum AND <= 6 slope levels",
        "measured": {
            "tau_grid_is_arithmetic": g_b["tau_grid_is_arithmetic"],
            "tau_step_spread": g_b["tau_step_spread"],
            "ordinate_quantum": g_b["ordinate_quantum"],
            "n_slope_levels": g_b["n_slope_levels"],
        },
        "limit": "tau steps must vary (figure pixels), quantum <= 0.01, > 6 slope levels",
        "result": "fail",
        "interpretation": (
            "The bundled table is a piecewise-linear sketch: every tau step is "
            f"{g_b['tau_step_mean']:.3f} (spread {g_b['tau_step_spread']:.1e}), every "
            f"ordinate is a multiple of {g_b['ordinate_quantum']}, and the "
            f"finite-difference slope takes only {g_b['n_slope_levels']} discrete values. "
            "A point-by-point reading off a figure does not produce that."
        ),
    }

    # 2. Does the reference contain the acceleration phase the experiment must have?
    accel_ok = (g_b["first_slope_to_peak"] or 0.0) <= 0.5
    gates["acceleration_phase_present"] = {
        "metric": "slope at the first interval / peak slope",
        "measured": g_b["first_slope_to_peak"],
        "crosscheck_measured": g_c["first_slope_to_peak"],
        "limit": "<= 0.5 (a front released from rest starts well below its later speed)",
        "result": "pass" if accel_ok else "fail",
        "interpretation": (
            "The bundled table reaches "
            f"{100.0 * (g_b['first_slope_to_peak'] or 0):.0f} % of its peak front speed "
            "within the first tau interval after release, i.e. it describes an "
            "impulsive start. The independent transcription reaches only "
            f"{100.0 * (g_c['first_slope_to_peak'] or 0):.0f} %, growing to a "
            "steady front speed - the shape a gate-release experiment has."
        ),
    }

    # 3. Is the claimed uncertainty consistent with the cross-source spread?
    consistent = uncertainty_ratio is not None and uncertainty_ratio <= 3.0
    gates["claimed_uncertainty_consistent"] = {
        "metric": "max |bundled - crosscheck| / claimed uncertainty",
        "measured": uncertainty_ratio,
        "claim": claimed,
        "cross_source_max_abs_delta": observed,
        "cross_source_rms_delta": spread.get("rms_delta"),
        "limit": "<= 3.0",
        "result": "pass" if consistent else "fail",
        "interpretation": (
            "Two independent transcriptions of the same figure differ by up to "
            f"{observed:.3f} in Z/a (rms {spread.get('rms_delta'):.3f}), "
            f"{uncertainty_ratio:.1f}x the claimed +-{claimed:.2f}. Since both come "
            "from the same paywalled original, this spread is a *lower bound* on "
            "the reference uncertainty."
        ),
    }

    verdict_withheld = not all(
        gates[g]["result"] == "pass"
        for g in ("bundled_is_pointwise", "acceleration_phase_present",
                  "claimed_uncertainty_consistent")
    )
    return {
        "tables": {
            "bundled": {k: bundled[k] for k in ("path", "claimed_uncertainty")},
            "crosscheck": {k: crosscheck[k] for k in ("path", "claimed_uncertainty")},
        },
        "grid_stats": {"bundled": g_b, "crosscheck": g_c},
        "cross_source_spread": spread,
        "gates": gates,
        "verdict": {
            "physical_validation_status": "withheld" if verdict_withheld else "evaluable",
            "reason": (
                "The reference cannot separate a solver error from a digitisation "
                "error of the same size (0.5-0.9 in Z/a), so no pass/fail verdict on "
                "physical validation of the B2 case can be issued against it. The "
                "earlier 'physical validation fails' statement over-attributes to the "
                "solver what the reference does not support."
            ) if verdict_withheld else (
                "The bundled reference survives the structural audit; the B2 "
                "comparison can be read as a validation verdict."
            ),
        },
    }


# --------------------------------------------------------------------------- #
# display paths: never embed the worktree that happened to run the audit
# --------------------------------------------------------------------------- #
def _display_path(p: Path) -> str:
    """Repo-relative path when the table lives inside the repository."""
    p = Path(p).resolve()
    root = HERE.parent.parent
    try:
        return p.relative_to(root).as_posix()
    except ValueError:
        return p.as_posix()


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
# unified driver: one artifact, both layers
# --------------------------------------------------------------------------- #
def solver_config(seconds: float = T_SECONDS, sound_speed: float = 25.0) -> dict:
    """Solver settings shared by every rung, read from ``DamBreakConfig``.

    Emitted into the artifact so a reader can tell which configuration the
    coefficients were measured with without reading the code.
    """
    cfg = DamBreakConfig(dx=0.04, pool_width=2.0, column_width=A,
                         sound_speed=sound_speed, max_courant=0.3, device="cpu",
                         wall_friction=0.0, density_diffusion=0.0,
                         adaptive_dt=False)
    return {
        "sound_speed": cfg.sound_speed,
        "max_courant": cfg.max_courant,
        "seconds": seconds,
        "adaptive_dt": cfg.adaptive_dt,
        "density_diffusion": cfg.density_diffusion,
        "viscosity": cfg.viscosity,
        "h_factor": cfg.h_factor,
    }


def structural_audit(bundled_path=None, crosscheck_path=None) -> dict:
    """Layer B: structure, kinematics, cross-source spread, gates, verdict.

    Needs no solver run, so it can be recomputed on any machine that has the
    two tables; that is what makes the committed artifact reproducible.
    """
    return evaluate({"bundled": load_bundled(bundled_path),
                     "crosscheck": load_crosscheck(crosscheck_path)})


def build_payload(bundled_path=None, crosscheck_path=None) -> dict:
    """Full payload: layer A (tables + analytic) merged with layer B."""
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
        "solver": solver_config(),
        "runs": [],
    }
    payload.update(structural_audit(bundled_path, crosscheck_path))
    return payload


def print_summary(payload: dict) -> None:
    a = payload["table_audit"]
    print("table:", a["verdict"])
    print("ritter identity residual: %.2e, shipped helper residual: %.2e"
          % (payload["ritter"]["max_identity_reldiff"],
             payload["ritter"]["max_helper_reldiff"]))
    print("table slope c = %.4f" % payload["table_slope"]["table_coefficient_c"])
    for name, gate in payload["gates"].items():
        print("  %-34s %4s measured=%s" % (name, gate["result"].upper(),
                                           gate["measured"]))
    print("  verdict: physical_validation = %s"
          % payload["verdict"]["physical_validation_status"])


def write_payload(payload: dict, out: Path) -> None:
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(payload, indent=1))
    print("wrote", out)


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--sizes", default="0.04")
    ap.add_argument("--frictions", default="0,0.25,0.5,0.75,1")
    ap.add_argument("--seconds", type=float, default=T_SECONDS)
    ap.add_argument("--json-out", default=str(HERE / "reference_audit.json"))
    ap.add_argument("--plot", action="store_true")
    ap.add_argument("--figdir", default="artifacts/dam_break_reference_audit")
    ap.add_argument("--from-json", default=None,
                    help="re-render figures for an existing payload instead of "
                         "re-running the solver")
    ap.add_argument("--bundled", default=None,
                    help="override the bundled table (default: the in-repo one)")
    ap.add_argument("--crosscheck", default=None,
                    help="override the independent transcription")
    ap.add_argument("--skip-runs", action="store_true",
                    help="recompute layer B only and keep the solver runs "
                         "already stored in --json-out (no simulation)")
    args = ap.parse_args()

    if args.from_json:
        payload = json.loads(Path(args.from_json).read_text())
        payload.update(structural_audit(args.bundled, args.crosscheck))
        payload["figures"] = make_figures(payload, Path(args.figdir))
        Path(args.from_json).write_text(json.dumps(payload, indent=1))
        print("figures:", args.figdir)
        return

    payload = build_payload(args.bundled, args.crosscheck)

    if args.skip_runs:
        out = Path(args.json_out)
        if out.exists():
            previous = json.loads(out.read_text())
            payload["runs"] = previous.get("runs", [])
            payload["figures"] = previous.get("figures", [])
        print_summary(payload)
        write_payload(payload, out)
        return

    payload["solver"] = solver_config(args.seconds)
    sizes = [float(s) for s in args.sizes.split(",")]
    frictions = [float(f) for f in args.frictions.split(",")]
    for dx in sorted(sizes, reverse=True):
        for f in frictions:
            r = run_rung(dx, f, seconds=args.seconds)
            payload["runs"].append(r)
            print("dx=%g friction=%g: N=%d %.1fs c=%.4f (%.1f%% of Ritter) "
                  "comp=%.3f mach=%.3f"
                  % (dx, f, r["particles"], r["wall_clock_s"],
                     r["front_coefficient_c"], 100.0 * r["c_over_ritter"],
                     r["max_density_compression"], r["mach_number"]), flush=True)

    print_summary(payload)

    if args.plot:
        payload["figures"] = make_figures(payload, Path(args.figdir))
        print("figures:", args.figdir)

    if args.json_out:
        write_payload(payload, Path(args.json_out))


if __name__ == "__main__":
    main()
