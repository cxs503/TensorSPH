#!/usr/bin/env python3
"""3-D hydrostatic column against ``p(z) = rho0 g (H - z)``.

Why this case
-------------
The shear-wave benchmark has no wall and no free surface, so it says nothing
about the *boundary* half of the 3-D machinery.  A column of fluid resting on a
static wall lattice is the cheapest configuration whose exact answer is known
without approximation, and it exercises exactly the pieces that a 3-D port has to
get right:

* static wall particles on the same lattice as the fluid,
* the Adami et al. (2012) wall-pressure extrapolation and the Tait-consistent wall
  density (switched off in the ``tait`` control runs, which use the wall's own
  summation density instead),
* the density summation in the presence of a free surface (the top layer is
  under-supported and its pressure is clamped to zero by the Tait law).

Reading the number
------------------
In weakly compressible SPH the pressure is a *small* density difference:
``p = B[(rho/rho0)^gamma - 1]``, so at ``c0 = 20`` m/s a bottom pressure of
``rho0 g H`` corresponds to ``delta rho / rho0 = 1.8e-3`` and a 0.08 per mille
density error already costs 4.5 % of the pressure.  The ``c0`` sweep therefore
measures how much of the residual pressure error is compressibility-limited
rather than operator-limited -- that is the parameter-robustness question this
benchmark exists to answer.

Usage::

    PYTHONPATH=src python benchmarks/threed_hydrostatic/run.py \
        --only adami-c0_10-nz16 --out /tmp/h.json
    PYTHONPATH=src python benchmarks/threed_hydrostatic/run.py --merge --figure
"""
from __future__ import annotations

import argparse
import glob
import json
import math
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))

from tensorsph.threed import acoustic_dt, run_hydrostatic  # noqa: E402

REPO = Path(__file__).resolve().parents[2]
OUT = REPO / "benchmarks" / "threed_hydrostatic"
DX = 0.01
NX = NY = 8
T_END = 0.60
#: Tait exponent used by the solver (see :func:`tensorsph.threed.tait_pressure`).
GAMMA = 7.0
#: WCSPH envelope: ``delta rho / rho0 = gamma g (H - z) / c0^2`` is the density
#: perturbation the pressure is read from.  Above this limit the answer is set by
#: the stiffness ``B = rho0 c0^2 / gamma``, not by the SPH operators, so the run is
#: a *probe*: it is reported with its numbers, it may not set a capability gate,
#: and the gate it must satisfy is the one that says it is worse than the gated
#: runs.  See ``docs/verification_threed_hydrostatic_envelope.md``.
ENVELOPE_LIMIT = 0.15

#: name -> (sound speed, wall model, nz)
CONFIGS = {
    "adami-c0_20-nz16": (20.0, "adami", 16),
    "adami-c0_10-nz16": (10.0, "adami", 16),
    "adami-c0_5-nz16": (5.0, "adami", 16),
    "tait-c0_10-nz16": (10.0, "tait", 16),
    "adami-c0_10-nz12": (10.0, "adami", 12),
    "adami-c0_10-nz24": (10.0, "adami", 24),
    # Validated envelope: the standard acoustic-speed rule c0 >= 10 sqrt(g H)
    # (Monaghan; Violent-SPH practice) puts this column at c0 = 39 m/s, so every
    # run above sits *below* the rule.  Without one run inside the envelope the
    # benchmark cannot separate "operator defect" from "stiffness too small".
    "adami-c0_40-nz16": (40.0, "adami", 16),
}


def envelope(sound_speed: float, height: float, gravity: float,
             gamma: float = GAMMA) -> dict:
    """Density-perturbation envelope of a run, from its own parameters only."""
    delta = gamma * gravity * height / sound_speed ** 2
    return {
        "delta_rho_over_rho0_at_bottom": delta,
        "c0_over_10sqrt_gH": sound_speed / (10.0 * math.sqrt(gravity * height)),
        "in_gated_envelope": bool(delta <= ENVELOPE_LIMIT),
        "limit": ENVELOPE_LIMIT,
    }


def run_case(name: str, *, t_end: float = T_END, viscosity: float = 0.02) -> dict:
    sound_speed, wall_model, nz = CONFIGS[name]
    h = 1.3 * DX
    dt = acoustic_dt(DX, h, sound_speed)
    steps = int(round(t_end / dt))
    result = run_hydrostatic(nx=NX, ny=NY, nz=nz, dx=DX, sound_speed=sound_speed,
                             viscosity=viscosity, wall_model=wall_model,
                             steps=steps, dt=dt, sample_every=max(1, steps // 40),
                             settle_fraction=0.5)
    profile = result.profile
    scale = profile["rho0"] * result.gravity * result.height
    bottom = profile["rows"][0]
    half = len(result.bottom_pressure) // 2
    first_half = sum(result.bottom_pressure[:half]) / max(half, 1)
    second_half = sum(result.bottom_pressure[half:]) / max(len(result.bottom_pressure) - half, 1)
    return {
        "case": name,
        "sound_speed": sound_speed,
        "wall_model": wall_model,
        "nz": nz,
        "dx": DX,
        "h": h,
        "height": result.height,
        "dt": dt,
        "steps": steps,
        "t_end": steps * dt,
        "fluid_particles": result.fluid_particles,
        "wall_particles": result.wall_particles,
        "envelope": envelope(sound_speed, result.height, result.gravity),
        "compressibility": {
            "rho_perturbation_at_bottom": (
                profile["rows"][0]["rho_analytic"] - profile["rho0"]) / profile["rho0"],
            "mach_from_initial_velocity": 0.0,
        },
        "pressure": {
            "bottom_measured": bottom["p_measured"],
            "bottom_analytic": bottom["p_analytic"],
            "bottom_rel_error": bottom["p_rel_error"],
            "core_layers": profile["core_layers"],
            "max_rel_error_core": profile["max_rel_error_core"],
            "mean_rel_error_core": profile["mean_rel_error_core"],
            "max_abs_error_over_rho0_g_H": profile["max_abs_error_over_scale"],
            "abs_error_scale_Pa": scale,
            "layers_within_15pct": profile["layers_within_15pct"],
            "integral_measured": profile["integral_p_measured"],
            "integral_analytic": profile["integral_p_analytic"],
            "integral_rel_error": profile["integral_rel_error"],
            "odd_even_amplitude_over_scale": profile[
                "odd_even_amplitude_over_scale"],
        },
        "profile": profile["rows"],
        "conservation": {
            "centre_of_mass_drift_over_height": (
                result.centre_of_mass_z[-1] - result.centre_of_mass_z[0])
            / (2.0 * 0.01 + result.height),
            "wall_displacement": profile.get("wall_displacement"),
            "momentum_not_conserved_reason": (
                "gravity and the static-wall reaction are external forces; the "
                "column is in balance, not in free flight"),
            "bottom_pressure_window_drift": (
                (second_half - first_half) / first_half if first_half else None),
        },
        "settling": {
            "bottom_pressure_first_half": first_half,
            "bottom_pressure_second_half": second_half,
            "relative_change": ((second_half - first_half) / first_half
                                if first_half else None),
        },
    }


def _per_run_status(run: dict) -> dict:
    """The four gates of a single run, evaluated identically for every run."""
    pressure = run["pressure"]
    return {
        "bottom_pressure_within_15pct": bool(
            abs(pressure["bottom_rel_error"]) < 0.15),
        "depth_integrated_load_within_15pct": bool(
            abs(pressure["integral_rel_error"]) < 0.15),
        "settled_window_drift_below_2pct": bool(
            abs(run["settling"]["relative_change"] or 0.0) < 0.02),
        "centre_of_mass_drift_below_1pct": bool(
            abs(run["conservation"]["centre_of_mass_drift_over_height"]) < 0.01),
    }


def _compressibility_fit(runs: list[dict]) -> dict:
    """Least-squares ``|e| = a + b / c0^2`` over the fixed-geometry adami runs.

    The residual pressure error of a weakly compressible column is expected to be
    dominated by the stiffness ``B = rho0 c0^2 / gamma``, i.e. to fall as
    ``1 / c0^2``; the intercept ``a`` is the part that does *not* move with the
    sound speed (operator and boundary bias).  Fitting both from the sweep is the
    difference between "the error is compressibility-limited" and "the error is
    the operators".
    """
    pts = [(r["sound_speed"], abs(r["pressure"]["bottom_rel_error"]))
           for r in runs]
    if len(pts) < 3:
        return {"points": len(pts), "a": None, "b": None}
    xs = [1.0 / (c * c) for c, _ in pts]
    ys = [e for _, e in pts]
    n = len(xs)
    sx, sy = sum(xs), sum(ys)
    sxx = sum(x * x for x in xs)
    sxy = sum(x * y for x, y in zip(xs, ys))
    denom = n * sxx - sx * sx
    if denom == 0.0:
        return {"points": n, "a": None, "b": None}
    b = (n * sxy - sx * sy) / denom
    a = (sy - b * sx) / n
    return {
        "model": "|bottom_rel_error| = a + b / c0^2",
        "points": n,
        "a": a,
        "b": b,
        "residuals": [y - (a + b * x) for x, y in zip(xs, ys)],
        "c0_values": [c for c, _ in pts],
    }


#: Why a run may not set a capability gate.  Two independent reasons exist and
#: conflating them was a defect of the first version of this file: a run can be
#: outside the *density-perturbation envelope* (``c0`` too small for the column
#: height) or it can be a deliberate *control* (the wall model under test switched
#: off).  Only the second kind is required to fail; the first kind is reported with
#: its numbers and must simply not be allowed to gate a claim either way.
CONTROL_WALL_MODELS = ("tait",)


def classify(run: dict) -> str:
    """``"gated"``, ``"control"`` or ``"envelope_probe"``, from parameters only."""
    if run["wall_model"] in CONTROL_WALL_MODELS:
        return "control"
    return "gated" if run["envelope"]["in_gated_envelope"] else "envelope_probe"


def _per_run_status(run: dict) -> dict:
    """The four gates of a single run, evaluated identically for every run."""
    pressure = run["pressure"]
    return {
        "bottom_pressure_within_15pct": bool(
            abs(pressure["bottom_rel_error"]) < 0.15),
        "depth_integrated_load_within_15pct": bool(
            abs(pressure["integral_rel_error"]) < 0.15),
        "settled_window_drift_below_2pct": bool(
            abs(run["settling"]["relative_change"] or 0.0) < 0.02),
        "centre_of_mass_drift_below_1pct": bool(
            abs(run["conservation"]["centre_of_mass_drift_over_height"]) < 0.01),
    }


def _compressibility_fit(runs: list[dict]) -> dict:
    """Least-squares ``|e| = a + b / c0^2`` over the fixed-geometry adami runs.

    The residual pressure error of a weakly compressible column is expected to be
    dominated by the stiffness ``B = rho0 c0^2 / gamma``, i.e. to fall as
    ``1 / c0^2``; the intercept ``a`` is the part that does *not* move with the
    sound speed (operator and boundary bias).  Fitting both from the sweep is the
    difference between "the error is compressibility-limited" and "the error is
    the operators".
    """
    pts = [(r["sound_speed"], abs(r["pressure"]["bottom_rel_error"]))
           for r in runs]
    if len(pts) < 3:
        return {"points": len(pts), "a": None, "b": None}
    xs = [1.0 / (c * c) for c, _ in pts]
    ys = [e for _, e in pts]
    n = len(xs)
    sx, sy = sum(xs), sum(ys)
    sxx = sum(x * x for x in xs)
    sxy = sum(x * y for x, y in zip(xs, ys))
    denom = n * sxx - sx * sx
    if denom == 0.0:
        return {"points": n, "a": None, "b": None}
    b = (n * sxy - sx * sy) / denom
    a = (sy - b * sx) / n
    return {
        "model": "|bottom_rel_error| = a + b / c0^2",
        "points": n,
        "a": a,
        "b": b,
        "residuals": [y - (a + b * x) for x, y in zip(xs, ys)],
        "c0_values": [c for c, _ in pts],
    }


def merge(results: list[dict]) -> dict:
    by_name = {r["case"]: r for r in results}
    rows = [by_name[name] for name in CONFIGS if name in by_name]
    roles = {r["case"]: classify(r) for r in rows}
    gated = [r for r in rows if roles[r["case"]] == "gated"]
    probes = [r for r in rows if roles[r["case"]] == "envelope_probe"]
    controls = [r for r in rows if roles[r["case"]] == "control"]
    adami = by_name.get("adami-c0_10-nz16")
    best = min(gated, key=lambda r: abs(r["pressure"]["bottom_rel_error"])) \
        if gated else None
    best_gated_error = (abs(best["pressure"]["bottom_rel_error"])
                        if best else None)
    # Fixed-geometry adami ladder (nz = 16, wall model = adami): the c0 sweep.
    sweep = sorted((r for r in rows
                    if r["wall_model"] == "adami" and r["nz"] == 16),
                   key=lambda r: r["sound_speed"])
    monotone = bool(len(sweep) >= 2) and all(
        abs(b["pressure"]["bottom_rel_error"])
        <= abs(a["pressure"]["bottom_rel_error"]) + 1e-12
        for a, b in zip(sweep, sweep[1:]))
    # steps = round(t_end / dt) is an integer, so the realised window differs from
    # run to run by up to one dt: the window counts as common within 1 %.
    t_end_raw = [r["t_end"] for r in rows]
    t_end_spread = ((max(t_end_raw) - min(t_end_raw)) / max(t_end_raw)
                    if t_end_raw else 0.0)
    control_status = {r["case"]: _per_run_status(r) for r in controls}
    probe_status = {r["case"]: _per_run_status(r) for r in probes}
    criteria = {
        # ---- capability gates: gated runs only -------------------------------
        "bottom_pressure_within_15pct_all_gated": bool(
            gated and all(abs(r["pressure"]["bottom_rel_error"]) < 0.15
                          for r in gated)),
        "bottom_pressure_within_3pct_best_gated": bool(
            best_gated_error is not None and best_gated_error < 0.03),
        "depth_integrated_load_within_15pct_all_gated": bool(
            gated and all(abs(r["pressure"]["integral_rel_error"]) < 0.15
                          for r in gated)),
        "column_settled_window_drift_below_2pct_all_gated": bool(
            gated and all(abs(r["settling"]["relative_change"] or 0.0) < 0.02
                          for r in gated)),
        "centre_of_mass_drift_below_1pct_all_gated": bool(
            gated and all(abs(r["conservation"]["centre_of_mass_drift_over_height"])
                          < 0.01 for r in gated)),
        # ---- positive gates about the runs that may not gate ----------------
        "wall_displacement_zero": bool(
            rows and all((r["conservation"]["wall_displacement"] or 0.0) == 0.0
                         for r in rows)),
        "common_time_window": bool(t_end_raw and t_end_spread <= 0.01),
        "control_run_fails_as_designed": bool(
            controls and all(not all(st.values())
                             for st in control_status.values())),
        "adami_wall_beats_tait_wall": bool(
            adami and controls
            and all(abs(adami["pressure"]["bottom_rel_error"])
                    < abs(c["pressure"]["bottom_rel_error"]) for c in controls)),
        "compressibility_error_monotone_in_sound_speed": monotone,
        "probes_are_worse_than_gated_best": bool(
            probes and best_gated_error is not None
            and all(abs(r["pressure"]["bottom_rel_error"]) > best_gated_error
                    for r in probes)),
        "probe_labels_are_derived_not_chosen": bool(
            all(roles[r["case"]] != "gated" for r in probes)
            and all(not r["envelope"]["in_gated_envelope"] for r in probes)),
    }
    missing = sorted(set(CONFIGS) - set(by_name))
    gated_verdict = ("pass" if all(criteria.values()) and not missing else "fail")
    return {
        "benchmark": "threed_hydrostatic",
        "case": "3-D wall-bounded column at rest, free surface at z = H",
        "reference": "p(z) = rho0 g (H - z), p = 0 at the free surface",
        "params": {"dx": DX, "nx": NX, "ny": NY, "rho0": 1000.0, "gravity": 9.81,
                   "t_end": 0.5 * (min(t_end_raw) + max(t_end_raw)) if t_end_raw else None,
                   "t_end_spread_over_mean": t_end_spread if t_end_raw else None,
                   "viscosity": 0.02, "gamma": GAMMA,
                   "envelope_limit": ENVELOPE_LIMIT},
        "runs": rows,
        "roles": roles,
        "best_case": best["case"] if best else None,
        "gated_cases": [r["case"] for r in gated],
        "probe_cases": [r["case"] for r in probes],
        "control_cases": [r["case"] for r in controls],
        # Honest FAIL stays in the record: the probe and control runs keep their
        # numbers and their per-run gate status, next to the verdict they do not set.
        "probe_outcomes": {
            r["case"]: {"role": roles[r["case"]],
                        "sound_speed": r["sound_speed"], "nz": r["nz"],
                        "delta_rho_over_rho0_at_bottom":
                            r["envelope"]["delta_rho_over_rho0_at_bottom"],
                        "status": _per_run_status(r)}
            for r in probes
        },
        "control_outcomes": {
            r["case"]: {"role": "control", "wall_model": r["wall_model"],
                        "delta_rho_over_rho0_at_bottom":
                            r["envelope"]["delta_rho_over_rho0_at_bottom"],
                        "status": _per_run_status(r)}
            for r in controls
        },
        "worst_probe_bottom_rel_error": max(
            (abs(r["pressure"]["bottom_rel_error"]) for r in probes + controls),
            default=None),
        "best_gated_bottom_rel_error": best_gated_error,
        "compressibility_fit": _compressibility_fit(sweep),
        "criteria": criteria,
        "missing_cases": missing,
        "verdict": gated_verdict,
        "verdict_including_probes": (
            "pass" if gated_verdict == "pass"
            and all(all(st.values()) for st in
                    list(probe_status.values()) + list(control_status.values()))
            else "fail"),
    }


def figure(report: dict, path: Path) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    runs = report["runs"]
    fig, axes = plt.subplots(1, 3, figsize=(13.5, 3.9))
    reference = None
    for run in runs:
        z = [r["depth_below_surface"] for r in run["profile"]]
        p = [r["p_measured"] for r in run["profile"]]
        reference = [r["p_analytic"] for r in run["profile"]]
        axes[0].plot(p, z, "o-", ms=3,
                     label=f"{run['case']} ({run['wall_model']})")
    if reference is not None:
        axes[0].plot(reference, z, "k--", lw=1.5, label=r"$\rho_0 g(H-z)$")
    axes[0].invert_xaxis()
    axes[0].set_xlabel("pressure (Pa)")
    axes[0].set_ylabel("depth below free surface (m)")
    axes[0].set_title("hydrostatic profile")
    axes[0].legend(fontsize=6)
    axes[0].grid(alpha=0.3)

    axes[1].axhline(0.0, color="k", lw=0.8)
    axes[1].axhline(0.03, color="k", ls=":", lw=1, label="3 %")
    axes[1].axhline(-0.03, color="k", ls=":", lw=1)
    axes[1].axhline(0.15, color="r", ls="--", lw=1, label="15 % platform limit")
    axes[1].axhline(-0.15, color="r", ls="--", lw=1)
    for run in runs:
        core = [r for r in run["profile"] if r["support_complete"]]
        axes[1].plot([r["depth_below_surface"] for r in core],
                     [r["p_rel_error"] for r in core], "o-", ms=3, label=run["case"])
    axes[1].set_xlabel("depth below free surface (m)")
    axes[1].set_ylabel("relative pressure error")
    axes[1].set_title("pressure error per layer")
    axes[1].legend(fontsize=6)
    axes[1].grid(alpha=0.3)

    for run in runs:
        axes[2].plot([run["sound_speed"]], [abs(run["pressure"]["bottom_rel_error"])],
                     "o", label=run["case"])
    axes[2].axhline(0.03, color="k", ls=":", lw=1, label="3 %")
    axes[2].set_xlabel("sound speed $c_0$ (m/s)")
    axes[2].set_ylabel("|bottom pressure relative error|")
    axes[2].set_title("compressibility sweep")
    axes[2].legend(fontsize=8)
    axes[2].grid(alpha=0.3, which="both")

    fig.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=130)
    plt.close(fig)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--only", action="append", default=None)
    parser.add_argument("--out", default=None)
    parser.add_argument("--merge", nargs="*", default=None)
    parser.add_argument("--figure", action="store_true")
    parser.add_argument("--t-end", type=float, default=T_END)
    args = parser.parse_args()

    if args.merge is not None:
        pattern = args.merge or [str(OUT / "cases" / "*.json")]
        files = [Path(p) for pat in pattern for p in glob.glob(pat)]
        results = [json.loads(f.read_text()) for f in files]
        report = merge(results)
        OUT.mkdir(parents=True, exist_ok=True)
        (OUT / "report.json").write_text(json.dumps(report, indent=2) + "\n")
        if args.figure:
            figure(report, OUT / "figures" / "threed_hydrostatic.png")
        print(json.dumps({"report": str(OUT / "report.json"),
                          "verdict": report["verdict"],
                          "criteria": report["criteria"],
                          "missing": report["missing_cases"]}, indent=2))
        return 0 if report["verdict"] == "pass" else 1

    torch.set_default_dtype(torch.float64)
    names = args.only or list(CONFIGS)
    out_path = Path(args.out) if args.out else None
    if out_path and len(names) != 1:
        raise SystemExit("--out needs exactly one --only case")
    for name in names:
        payload = run_case(name, t_end=args.t_end)
        if out_path:
            out_path.parent.mkdir(parents=True, exist_ok=True)
            out_path.write_text(json.dumps(payload, indent=2) + "\n")
        print(json.dumps({"case": name, "pressure": payload["pressure"],
                          "settling": payload["settling"]}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
