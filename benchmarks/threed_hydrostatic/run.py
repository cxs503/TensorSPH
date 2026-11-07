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

#: name -> (sound speed, wall model, nz)
CONFIGS = {
    "adami-c0_20-nz16": (20.0, "adami", 16),
    "adami-c0_10-nz16": (10.0, "adami", 16),
    "adami-c0_5-nz16": (5.0, "adami", 16),
    "tait-c0_10-nz16": (10.0, "tait", 16),
    "adami-c0_10-nz12": (10.0, "adami", 12),
    "adami-c0_10-nz24": (10.0, "adami", 24),
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


def merge(results: list[dict]) -> dict:
    by_name = {r["case"]: r for r in results}
    rows = [by_name[name] for name in CONFIGS if name in by_name]
    best = min(rows, key=lambda r: abs(r["pressure"]["max_rel_error_core"] or 1e9))
    adami = by_name.get("adami-c0_10-nz16")
    tait = by_name.get("tait-c0_10-nz16")
    # The Adami family is the claim; ``tait-c0_10-nz16`` is the control run whose
    # whole purpose is to fail, and it must not be allowed to set a gate.
    adami_runs = [r for r in rows if r["wall_model"] == "adami"]
    criteria = {
        "bottom_pressure_within_15pct_all_adami": bool(
            adami_runs
            and all(abs(r["pressure"]["bottom_rel_error"]) < 0.15 for r in adami_runs)),
        "bottom_pressure_within_3pct_best": bool(
            best and abs(best["pressure"]["bottom_rel_error"]) < 0.03),
        "depth_integrated_load_within_15pct_all_adami": bool(
            adami_runs
            and all(abs(r["pressure"]["integral_rel_error"]) < 0.15
                    for r in adami_runs)),
        "wall_displacement_zero": bool(
            rows and all((r["conservation"]["wall_displacement"] or 0.0) == 0.0
                         for r in rows)),
        "column_settled_window_drift_below_2pct": bool(
            adami_runs and all(abs(r["settling"]["relative_change"] or 0.0) < 0.02
                               for r in adami_runs)),
        "centre_of_mass_drift_below_1pct": bool(
            rows and all(abs(r["conservation"]["centre_of_mass_drift_over_height"])
                         < 0.01 for r in rows)),
        "adami_wall_beats_tait_wall": bool(
            adami and tait and abs(adami["pressure"]["bottom_rel_error"])
            < abs(tait["pressure"]["bottom_rel_error"])),
    }
    missing = sorted(set(CONFIGS) - set(by_name))
    return {
        "benchmark": "threed_hydrostatic",
        "case": "3-D wall-bounded column at rest, free surface at z = H",
        "reference": "p(z) = rho0 g (H - z), p = 0 at the free surface",
        "params": {"dx": DX, "nx": NX, "ny": NY, "rho0": 1000.0, "gravity": 9.81,
                   "t_end": T_END, "viscosity": 0.02},
        "runs": rows,
        "best_case": best["case"] if best else None,
        "criteria": criteria,
        "missing_cases": missing,
        "verdict": "pass" if all(criteria.values()) and not missing else "fail",
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
