"""CASE-POISEUILLE-SPH -- oscillatory wall shear in the harmonic regime.

The graded ladder of ``run.py`` certifies the *kinematics* of the Womersley
periodic state (amplitude and phase of every row, gates G1-G8).  It never
grades the quantity a hull designer actually asks for in a pulsatile duct: the
**oscillatory wall shear stress** (pulsatile drag), i.e. the complex amplitude
``tau_hat`` of ``tau_w(t) = Re{ tau_hat exp(i omega t) }`` at the wall.

This module adds that layer as pure post-processing of the committed artifact
``out/result.json`` -- no SPH rerun, no new solver knob.

Why not just differentiate the measured profile at the wall?
------------------------------------------------------------
`benchmarks/case_poiseuille/combined.py` recovers *both* steady wall shears
from a least-squares quadratic through the interior band, and
``docs/verification_poiseuille_wall_shear_estimator.md`` shows that this is the
only local estimator that survives the near-wall template/ghost geometry for
the steady parabola.  That estimator is **exact only because the steady profile
is a parabola**.  In the harmonic regime the profile is
``cosh(alpha (y - H/2))`` with ``alpha = sqrt(i omega / nu)`` -- an exponential
whose curvature at the wall is not a quadratic.  Applied to the *analytic*
Womersley profile it returns a bias of ``-5.3 %`` in amplitude and ``-24 deg``
in phase at ``Wo = 1.8``, and ``-23 % / -52 deg`` at ``Wo = 2.8``, independent
of resolution.  That is model bias, not discretisation; gate **W5** below
records it as a *control* (the local estimator must be shown to be
non-transferable, otherwise this layer would be redundant).

Kernel-free estimator actually used
-----------------------------------
Integrate the momentum equation over the channel, ``u_t = nu u_yy + g(t)``:

    d<u>/dt = g(t) + [ tau(H) - tau(0) ] / (rho H),   tau(y) = mu u_y

For the anti-symmetric Womersley profile ``tau(H) = -tau(0)``, so in the
periodic state (single harmonic, ``d/dt -> i omega``):

    tau_hat = rho H ( g_hat - i omega <u>_hat ) / 2

This involves **no kernel and no wall normal**, only the cross-section mean
velocity ``<u>_hat`` -- the functional the lattice rows integrate anyway.  It
therefore asks the sharp question: does the mirror/ghost wall absorb exactly
the analytic oscillatory momentum flux?  (For the steady case the same budget
is what proved the -49.7 % "wall model deficit" was an estimator bias, see
``docs/verification_poiseuille_wall_shear_audit.md``.)

Predeclared gates
-----------------
Thresholds follow the case convention of ``DESIGN.md`` (5 % amplitude,
5 % of a quarter turn = 4.5 deg for phase) and were **not** moved after the
numbers were seen: the first run of this script on the committed artifact was
an exploratory probe (`docs/case_poiseuille_wall_shear_harmonic.md` Sec. 1
says so), and every observed error is below 1.6 %, i.e. a tighter threshold
would also have passed -- none was retuned.

| gate | metric | threshold |
|---|---|---|
| **W0** | cross-section mean velocity transfer function at ``omega``: ``<u>_hat`` amplitude, finest rung | <= 5 % |
| **W0b** | same, phase | <= 4.5 deg (5 % of pi/2) |
| **W1** | wall shear amplitude ``|tau_hat|``, finest rung | <= 5 % |
| **W2** | wall shear phase ``arg tau_hat``, finest rung | <= 4.5 deg |
| **W3** | ladder: W1/W2 hold at *every* rung (8/12/16) | pass |
| **W4** | frequency discrimination at ``H/dx = 12``: ``|tau_fast|/|tau_slow|`` and the phase difference | <= 5 % / 4.5 deg |
| **W5** | control: interior-quadratic local estimator applied to the *analytic* profile is biased by > 5 % in amplitude at ``omega`` | pass (= the budget is not redundant) |
| **W6** | reference self-check: budget closed form == ``mu alpha (g_hat/i omega) tanh(alpha H/2)`` to 1e-10, and the ``omega -> 0`` limit == ``rho g H / 2`` | pass |

A quasi-steady solver (one that follows ``g(t)`` with the steady parabola) would
give ``|tau_fast|/|tau_slow| = 1`` and a zero phase difference; the analytic
values are ``0.580`` and ``-6.6 deg``.  W4 is therefore a falsification lever
for "the run resolves the unsteady shear layer" and cannot be passed by a mean
momentum balance.

Usage
-----
    PYTHONPATH=.:src python3 benchmarks/case_poiseuille/wall_shear_harmonic.py
    PYTHONPATH=.:src python3 benchmarks/case_poiseuille/make_wall_shear_harmonic_doc.py
"""

from __future__ import annotations

import argparse
import cmath
import json
import math
from pathlib import Path

import numpy as np

#: Case-pool identifier serviced by this layer.
CASE_KEY = "CASE-POISEUILLE-SPH"

HERE = Path(__file__).resolve().parent
DEFAULT_ARTIFACT = HERE / "out" / "result.json"
DEFAULT_OUT = HERE / "out" / "wall_shear_harmonic.json"

#: Amplitude tolerance, case convention (``DESIGN.md``).
TOL_AMP_PCT = 5.0
#: Phase tolerance expressed as a percentage of a quarter turn.
TOL_PHASE_PCT_QUARTER_TURN = 5.0
#: Quarter turn in degrees; ``5 %`` of it is the ``4.5 deg`` of ``DESIGN.md``.
QUARTER_TURN_DEG = 90.0
#: Toleranced resolution for the frequency set.
TOL_REFERENCE = 1.0e-10
#: Interior band used by the steady wall-shear estimator (``combined.py``).
INTERIOR_BAND = (0.15, 0.85)
#: Resolution at which the two-frequency discrimination is graded.
DISCRIMINATION_RESOLUTION = 12


# --------------------------------------------------------------------------- #
# Closed forms (analytic reference, independent of any SPH code)
# --------------------------------------------------------------------------- #

def _alpha(omega: float, viscosity: float) -> complex:
    if not omega > 0.0:
        raise ValueError(f"omega must be positive, got {omega!r}")
    if not viscosity > 0.0:
        raise ValueError(f"viscosity must be positive, got {viscosity!r}")
    return cmath.sqrt(1j * omega / viscosity)


def analytic_wall_shear(omega: float, height: float, body_force: float,
                        viscosity: float, rho0: float) -> complex:
    """``tau_hat`` at the ``y = 0`` wall, for ``g(t) = g0 cos(omega t)``.

    ``tau = mu u_y`` and ``Uhat = (g_hat/i omega)[1 - cosh(alpha(y-H/2))/
    cosh(alpha H/2)]``, so

        dUhat/dy|_0 = (g_hat / i omega) alpha tanh(alpha H / 2)

    giving ``tau_hat = mu (g_hat/i omega) alpha tanh(alpha H/2)``.  The
    ``y = H`` wall carries ``-tau_hat`` (anti-symmetry).
    """
    g_hat = complex(body_force)
    a = _alpha(omega, viscosity)
    mu = rho0 * viscosity
    return mu * (g_hat / (1j * omega)) * a * cmath.tanh(a * 0.5 * height)


def analytic_mean_amplitude(omega: float, height: float, body_force: float,
                            viscosity: float) -> complex:
    """``<Uhat>``: the cross-section mean of the Womersley profile.

    ``<Uhat> = (g_hat/i omega) [ 1 - (2/(alpha H)) tanh(alpha H / 2) ]``.
    """
    g_hat = complex(body_force)
    a = _alpha(omega, viscosity)
    return (g_hat / (1j * omega)) * (1.0 - 2.0 / (a * height) * cmath.tanh(a * 0.5 * height))


def analytic_profile(y, omega: float, height: float, body_force: float,
                     viscosity: float):
    """``Uhat(y)`` -- the same closed form as ``tensorsph.womersley``."""
    a = _alpha(omega, viscosity)
    y = np.asarray(y, dtype=float)
    g_hat = complex(body_force)
    return (g_hat / (1j * omega)) * (
        1.0 - np.cosh(a * (y - 0.5 * height)) / np.cosh(a * 0.5 * height))


# --------------------------------------------------------------------------- #
# Estimators
# --------------------------------------------------------------------------- #

def measured_mean_amplitude(rows) -> complex:
    """``<u>_hat`` from the committed per-row harmonic fits.

    Each lattice row is one ``dx`` thick and carries the same number of
    columns, so the cross-section mean is the arithmetic mean of the row means
    (that is what makes ``bin_rows`` a mean over the section and not a
    particle-weighted average of a non-uniform lattice).

    A row record is ``{"y": float, "amplitude": float, "phase": float}`` with
    the ``u = A cos(omega t + phase)`` convention of
    ``tensorsph.stokes_second.harmonic_fit``; the complex amplitude is
    ``A exp(i phase)``.
    """
    if not rows:
        raise ValueError("no rows to average")
    total = 0j
    for row in rows:
        amp = float(row["amplitude"])
        phase = float(row.get("phase", math.radians(row.get("phase_deg", 0.0))))
        total += amp * cmath.exp(1j * phase)
    return total / len(rows)


def budget_wall_shear(mean_amplitude: complex, omega: float, height: float,
                      body_force: float, rho0: float) -> complex:
    """``tau_hat = rho H (g_hat - i omega <u>_hat) / 2`` (momentum budget).

    Exact for the periodic state of ``u_t = nu u_yy + g(t)`` with
    anti-symmetric wall flux; no kernel, no wall normal, no fit.
    """
    g_hat = complex(body_force)
    return rho0 * height * (g_hat - 1j * omega * complex(mean_amplitude)) / 2.0


def quadratic_local_wall_shear(y, uhat, height: float, rho0: float,
                               viscosity: float,
                               band=INTERIOR_BAND) -> complex:
    """``tau_hat`` at ``y = 0`` from a complex least-squares quadratic.

    The steady-regime estimator of ``combined.py`` lifted to complex
    amplitudes: ``u = a y^2 + b y + c`` fitted on ``band`` (fractions of ``H``),
    wall shear ``= rho nu b``.  Kept here **as a control** (gate W5), not as
    the graded estimator.
    """
    y = np.asarray(y, dtype=float)
    uhat = np.asarray(uhat, dtype=complex)
    lo, hi = band
    sel = (y / height >= lo) & (y / height <= hi)
    if int(sel.sum()) < 3:
        raise ValueError(f"band {band} holds {int(sel.sum())} rows, need >= 3")
    # normal equations, solved in real arithmetic for both parts at once
    design = np.vstack([y[sel] ** 2, y[sel], np.ones(int(sel.sum()))]).T
    coef, *_ = np.linalg.lstsq(design, uhat[sel], rcond=None)
    return rho0 * viscosity * complex(coef[1])


# --------------------------------------------------------------------------- #
# Comparison helpers
# --------------------------------------------------------------------------- #

def wrap_deg(angle: float) -> float:
    """Wrap an angle in degrees to ``[-180, 180)``."""
    return (angle + 180.0) % 360.0 - 180.0


def compare(measured: complex, analytic: complex) -> dict:
    """Amplitude error [%], phase error [deg] and phase error in % of pi/2."""
    if abs(analytic) == 0.0:
        raise ValueError("analytic reference is zero")
    amp_pct = 100.0 * (abs(measured) / abs(analytic) - 1.0)
    phase_deg = wrap_deg(math.degrees(cmath.phase(measured / analytic)))
    return {
        "measured_abs": abs(measured),
        "measured_phase_deg": math.degrees(cmath.phase(measured)),
        "analytic_abs": abs(analytic),
        "analytic_phase_deg": math.degrees(cmath.phase(analytic)),
        "amp_error_pct": amp_pct,
        "phase_error_deg": phase_deg,
        "phase_error_pct_quarter_turn": 100.0 * abs(phase_deg) / QUARTER_TURN_DEG,
    }


def _amp_ok(cmp: dict, tol: float = TOL_AMP_PCT) -> bool:
    return abs(cmp["amp_error_pct"]) <= tol


def _phase_ok(cmp: dict, tol: float = TOL_PHASE_PCT_QUARTER_TURN) -> bool:
    return cmp["phase_error_pct_quarter_turn"] <= tol


# --------------------------------------------------------------------------- #
# Build
# --------------------------------------------------------------------------- #

def run_record(run, geometry: dict) -> dict:
    """One harmonic run -> the mean-velocity and wall-shear comparison."""
    height = float(geometry["height"])
    rho0 = float(geometry["rho0"])
    nu = float(geometry["viscosity"])
    g0 = float(geometry["body_force"])
    omega = float(run["omega"])
    rows = run["rows"]

    uhat_rows = [float(r["amplitude"]) * cmath.exp(1j * float(r["phase"]))
                 for r in rows]
    y = np.array([float(r["y"]) for r in rows])

    mean_meas = measured_mean_amplitude(rows)
    mean_ana = analytic_mean_amplitude(omega, height, g0, nu)
    tau_meas = budget_wall_shear(mean_meas, omega, height, g0, rho0)
    tau_ana = analytic_wall_shear(omega, height, g0, nu, rho0)
    tau_local = quadratic_local_wall_shear(y, uhat_rows, height, rho0, nu)

    return {
        "resolution_H_over_dx": int(run["resolution_H_over_dx"]),
        "omega": omega,
        "womersley_number": float(run.get("womersley_number", float("nan"))),
        "stokes_layer": float(run.get("stokes_layer", float("nan"))),
        "n_rows": len(rows),
        "mean_velocity": compare(mean_meas, mean_ana),
        "wall_shear_budget": compare(tau_meas, tau_ana),
        "wall_shear_local_quadratic": compare(tau_local, tau_ana),
        "mean_amplitude_complex": [mean_meas.real, mean_meas.imag],
        "wall_shear_budget_complex": [tau_meas.real, tau_meas.imag],
        "wall_shear_analytic_complex": [tau_ana.real, tau_ana.imag],
        "wall_shear_local_complex": [tau_local.real, tau_local.imag],
    }


def reference_selfcheck(height: float, nu: float, g0: float, rho0: float,
                        omega: float) -> dict:
    """W6: the budget identity and the steady limit, without any measurement."""
    g_hat = complex(g0)
    mean_ana = analytic_mean_amplitude(omega, height, g0, nu)
    via_budget = budget_wall_shear(mean_ana, omega, height, g0, rho0)
    closed = analytic_wall_shear(omega, height, g0, nu, rho0)
    rel = abs(via_budget - closed) / abs(closed)
    # omega -> 0 limit of mu (g/i omega) alpha tanh(alpha H/2) is rho g H / 2
    steady_limit = rho0 * g0 * height / 2.0

    # omega -> 0 limit of the closed form must be the steady wall shear
    small = 1e-9
    limit = abs(analytic_wall_shear(small, height, g0, nu, rho0))
    limit_rel = abs(limit - steady_limit) / steady_limit
    return {
        "budget_identity_rel_error": rel,
        "steady_wall_shear": steady_limit,
        "omega_to_zero_limit": limit,
        "omega_to_zero_rel_error": limit_rel,
        "g_hat": [g_hat.real, g_hat.imag],
        "pass": rel <= TOL_REFERENCE and limit_rel <= 1.0e-6,
    }


def estimator_transferability(omega: float, geometry: dict,
                              resolutions=(8, 12, 16)) -> dict:
    """W5 control: bias of the local quadratic estimator on the *analytic* profile.

    Resolution-independent by construction (the profile is smooth and the band
    is the same fraction of ``H``), which is what makes it model bias rather
    than discretisation error.
    """
    height = float(geometry["height"])
    rho0 = float(geometry["rho0"])
    nu = float(geometry["viscosity"])
    g0 = float(geometry["body_force"])
    tau_ana = analytic_wall_shear(omega, height, g0, nu, rho0)
    per_rung = {}
    for n in resolutions:
        dx = height / n
        y = dx * 0.5 + dx * np.arange(n)
        uhat = analytic_profile(y, omega, height, g0, nu)
        tau = quadratic_local_wall_shear(y, uhat, height, rho0, nu)
        per_rung[str(n)] = compare(tau, tau_ana)
    amp = [abs(v["amp_error_pct"]) for v in per_rung.values()]
    spread = max(amp) - min(amp)
    return {
        "omega": omega,
        "womersley_number": 0.5 * height * math.sqrt(omega / nu),
        "per_rung": per_rung,
        "amp_bias_max_pct": max(amp),
        "amp_bias_spread_pct": spread,
        "is_biased_beyond_case_tolerance": max(amp) > TOL_AMP_PCT,
        "is_resolution_independent": spread < 1.0,
    }


def sweep_estimator_bias(geometry: dict, omegas=None) -> list:
    """Bias of the local estimator vs ``Wo`` (analytic profile, finest lattice).

    Used by the figure; not graded.
    """
    if omegas is None:
        omegas = np.linspace(2.0, 40.0, 25)
    out = []
    for omega in omegas:
        rec = estimator_transferability(float(omega), geometry, resolutions=(16,))
        cmp = rec["per_rung"]["16"]
        out.append({
            "omega": float(omega),
            "womersley_number": rec["womersley_number"],
            "amp_error_pct": cmp["amp_error_pct"],
            "phase_error_deg": cmp["phase_error_deg"],
        })
    return out


def build(artifact: dict) -> dict:
    """Post-process one ``out/result.json`` into the wall-shear verdict."""
    geometry = dict(artifact["geometry"])
    height = float(geometry["height"])
    rho0 = float(geometry["rho0"])
    nu = float(geometry["viscosity"])
    g0 = float(geometry["body_force"])

    runs = list(artifact["harmonic_runs"])
    fast = artifact.get("harmonic_fast")
    omega_slow = float(runs[0]["omega"])

    slow = [run_record(r, geometry) for r in runs]
    fast_rec = run_record(fast, geometry) if fast else None
    finest = max(slow, key=lambda r: r["resolution_H_over_dx"])

    # two-frequency discrimination at one common resolution
    disc = None
    if fast_rec is not None:
        pool = {r["resolution_H_over_dx"]: r for r in slow}
        n = DISCRIMINATION_RESOLUTION
        if n not in pool:
            n = min(pool)
        slow_at_n = pool[n]
        meas_ratio = abs(complex(*fast_rec["wall_shear_budget_complex"])) / abs(
            complex(*slow_at_n["wall_shear_budget_complex"]))
        ana_ratio = abs(complex(*fast_rec["wall_shear_analytic_complex"])) / abs(
            complex(*slow_at_n["wall_shear_analytic_complex"]))
        meas_dphi = wrap_deg(
            math.degrees(cmath.phase(complex(*fast_rec["wall_shear_budget_complex"])
                                     / complex(*slow_at_n["wall_shear_budget_complex"]))))
        ana_dphi = wrap_deg(
            math.degrees(cmath.phase(complex(*fast_rec["wall_shear_analytic_complex"])
                                     / complex(*slow_at_n["wall_shear_analytic_complex"]))))
        disc = {
            "resolution_H_over_dx": n,
            "omega_slow": slow_at_n["omega"],
            "omega_fast": fast_rec["omega"],
            "ratio_measured": meas_ratio,
            "ratio_analytic": ana_ratio,
            "ratio_error_pct": 100.0 * (meas_ratio / ana_ratio - 1.0),
            "phase_difference_measured_deg": meas_dphi,
            "phase_difference_analytic_deg": ana_dphi,
            "phase_difference_error_deg": wrap_deg(meas_dphi - ana_dphi),
            "quasi_steady_ratio_would_be": 1.0,
            "quasi_steady_phase_difference_deg": 0.0,
        }

    # control: a solver that followed the forcing with the *steady* parabola
    # would report the section mean <u> = g0 H^2 / (12 nu) at zero phase, so its
    # budget wall shear is rho H g0 / 2 (fully in phase with the forcing).
    mean_steady = g0 * height * height / (12.0 * nu)
    tau_quasi = budget_wall_shear(complex(mean_steady, 0.0), omega_slow, height, g0, rho0)
    tau_ana_slow = analytic_wall_shear(omega_slow, height, g0, nu, rho0)
    quasi_steady = {
        "description": ("quasi-steady control: section mean of the steady parabola "
                        "<u> = g0 H^2/(12 nu) at zero phase, fed through the same "
                        "budget (1)"),
        "mean_amplitude": mean_steady,
        "measured": compare(tau_quasi, tau_ana_slow),
        "note": "this is what a mean-momentum-balance-only solver would report",
    }

    transferability = estimator_transferability(omega_slow, geometry)
    selfcheck = reference_selfcheck(height, nu, g0, rho0, omega_slow)

    gates = {
        "W0_mean_amplitude_5pct": _amp_ok(finest["mean_velocity"]),
        "W0b_mean_phase_5pct_of_quarter_turn": _phase_ok(finest["mean_velocity"]),
        "W1_wall_shear_amplitude_5pct": _amp_ok(finest["wall_shear_budget"]),
        "W2_wall_shear_phase_5pct_of_quarter_turn": _phase_ok(finest["wall_shear_budget"]),
        "W3_all_rungs_5pct": all(
            _amp_ok(r["wall_shear_budget"]) and _phase_ok(r["wall_shear_budget"])
            for r in slow),
        "W4_frequency_ratio_5pct": bool(
            disc is not None
            and abs(disc["ratio_error_pct"]) <= TOL_AMP_PCT
            and 100.0 * abs(disc["phase_difference_error_deg"]) / QUARTER_TURN_DEG
            <= TOL_PHASE_PCT_QUARTER_TURN),
        "W5_local_estimator_control_biased_gt_5pct":
            bool(transferability["is_biased_beyond_case_tolerance"]),
        "W6_reference_selfcheck_1e-10": bool(selfcheck["pass"]),
    }

    return {
        "case": CASE_KEY,
        "layer": "harmonic wall shear (pulsatile drag) from the momentum budget",
        "note": ("post-processing of out/result.json; no SPH rerun. "
                 "tau_hat = rho H (g_hat - i omega <u>_hat)/2, exact for the "
                 "periodic state, kernel-free."),
        "inputs": {
            "artifact": "benchmarks/case_poiseuille/out/result.json",
            "geometry": geometry,
            "body_force": g0,
            "omega_slow": omega_slow,
            "omega_fast": None if fast_rec is None else fast_rec["omega"],
        },
        "thresholds": {
            "amp_pct": TOL_AMP_PCT,
            "phase_pct_of_quarter_turn": TOL_PHASE_PCT_QUARTER_TURN,
            "phase_deg_equivalent": TOL_PHASE_PCT_QUARTER_TURN * QUARTER_TURN_DEG / 100.0,
            "interior_band_y_over_H": list(INTERIOR_BAND),
            "declared": ("case convention of DESIGN.md; not tuned to the observed "
                         "values (exploratory probe recorded in the note)"),
        },
        "harmonic_runs": slow,
        "harmonic_fast": fast_rec,
        "frequency_discrimination": disc,
        "quasi_steady_control": quasi_steady,
        "estimator_transferability": transferability,
        "estimator_bias_sweep": sweep_estimator_bias(geometry),
        "reference_selfcheck": selfcheck,
        "gates": gates,
        "all_gates_pass": all(gates.values()),
        "limitations": [
            "The budget uses only the cross-section mean velocity: it certifies the "
            "integral momentum flux at the wall, not the shape of the Stokes layer.",
            "The phase reference is the absolute time base of the harmonic fit; a "
            "common shift of the forcing history would move every phase equally, "
            "which the measured-vs-analytic agreement rules out.",
            "Single harmonic only: a secondary-harmonic (nonlinear) wall shear is "
            "not measurable from the stored per-row fits.",
        ],
    }


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--artifact", default=str(DEFAULT_ARTIFACT))
    parser.add_argument("--out", default=str(DEFAULT_OUT))
    parser.add_argument("--quiet", action="store_true")
    args = parser.parse_args(argv)

    artifact = json.loads(Path(args.artifact).read_text())
    verdict = build(artifact)
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(json.dumps(verdict, indent=1) + "\n")

    if not args.quiet:
        print(f"{verdict['case']} -- harmonic wall shear")
        hdr = f"{'H/dx':>5} {'|tau| [Pa]':>11} {'analytic':>9} {'amp %':>8} {'phase deg':>10}"
        print(hdr)
        for rec in verdict["harmonic_runs"]:
            c = rec["wall_shear_budget"]
            print(f"{rec['resolution_H_over_dx']:>5} {c['measured_abs']:>11.4f} "
                  f"{c['analytic_abs']:>9.4f} {c['amp_error_pct']:>+8.3f} "
                  f"{c['phase_error_deg']:>+10.3f}")
        disc = verdict["frequency_discrimination"]
        if disc:
            print(f"frequency ratio {disc['ratio_measured']:.4f} vs "
                  f"{disc['ratio_analytic']:.4f} "
                  f"({disc['ratio_error_pct']:+.2f} %)")
        for name, ok in verdict["gates"].items():
            print(f"  {'PASS' if ok else 'FAIL'}  {name}")
        print("all gates pass:", verdict["all_gates_pass"])
    return 0 if verdict["all_gates_pass"] else 1


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())