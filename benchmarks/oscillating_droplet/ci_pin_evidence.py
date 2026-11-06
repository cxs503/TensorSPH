"""Evidence artefact for the m4802 main-red fix of the mode-2 frequency pins.

Main was red at 799310b on two physics-layer tests of
``tests/test_oscillating_droplet.py`` (plus a missing figure artefact in
``tests/test_laplace_pressure_jump.py``).  The cause was not the physics record
but the *form* of the pin: the CI release case was pinned with
``xfail(strict=True)`` against a value recorded on loongarch64, and CI now runs
on a loongarch64 runner, so the run reproduced the pin and XPASSed.

This script measures the release case once and writes the numbers that the
tests and the document quote, so nothing is transcribed by hand:

* the pseudo-frequency error of the release case in the two estimators
  (damped-cosine fit / zero crossing) and their disagreement;
* the divergence evidence that explains why the value is not portable -- the
  equivalent radius ``r_eff`` growth over the simulated span and ``max |v|``
  against the capillary velocity (a *valid* capillary oscillation keeps
  ``r_eff`` and ``|v|`` bounded);
* the platform-independent verdict taken from the committed
  ``validation_report.json`` (release scale, R = 2 cm).

    PYTHONPATH=src python benchmarks/oscillating_droplet/ci_pin_evidence.py \
        --json-out benchmarks/oscillating_droplet/ci_pin_evidence.json \
        --fig /tmp/m4802_ci_pin.png

No solver, threshold or verdict is modified here: the release-scale case is a
documented FAIL (-43.5 %) and stays so.
"""
from __future__ import annotations

import argparse
import importlib.util
import json
import math
import platform
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]


def _load_run():
    spec = importlib.util.spec_from_file_location("oscillating_run", HERE / "run.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def measure() -> dict:
    run = _load_run()
    t0 = time.time()
    rep = run.run_case(radius=0.01, dx=0.001, sigma=7.2, rho0=1000.0,
                       sound_speed=20.0, viscosity=5e-5, eps=0.10, periods=1.8)
    wall = time.time() - t0
    freq = rep["frequency"]
    setup = rep["setup"]
    trace = rep["diagnostics"]["trace"]
    r_eff0 = rep["initial_state"]["r_eff"]
    r_eff1 = rep["final_state"]["r_eff"]
    return {
        "platform": platform.machine(),
        "python": platform.python_version(),
        "release_case": {
            "radius_m": setup["radius_m"], "dx_m": setup["dx_m"],
            "n_particles": setup["n_particles"], "steps": setup["steps"],
            "periods_simulated": setup["periods_simulated"],
            "wall_time_s": wall,
            "omega_reference_rad_per_s": rep["reference"]["omega_rad_per_s"],
            "fit_omega_rad_per_s": freq["fit_second_moment_rad_per_s"],
            "fit_error_pct": freq["fit_second_moment_rel_error_pct"],
            "zero_crossing_omega_rad_per_s": freq["zero_crossing_rad_per_s"],
            "zero_crossing_error_pct": freq["zero_crossing_rel_error_pct"],
            "estimator_agreement_pct": freq["estimator_agreement_pct"],
            "fit_r2": freq["fit"]["r2"],
            "fit_lambda_per_s": freq["fit"]["lambda_per_s"],
            "within_15pct": freq["within_15pct"],
        },
        "divergence": {
            "r_eff_initial_m": r_eff0,
            "r_eff_final_m": r_eff1,
            "r_eff_growth_factor": r_eff1 / r_eff0,
            "max_speed_m_per_s": rep["diagnostics"]["max_speed_m_per_s"],
            "capillary_velocity_m_per_s":
                rep["diagnostics"]["capillary_velocity_m_per_s"],
            "max_speed_over_capillary":
                rep["diagnostics"]["max_speed_m_per_s"]
                / rep["diagnostics"]["capillary_velocity_m_per_s"],
            "mass_relative_drift": rep["diagnostics"]["mass_relative_drift"],
        },
        "trace": {
            "t_s": trace["t"],
            "s_second_moment": trace["s_second_moment"],
            "kinetic_energy_J": trace["kinetic_energy_J"],
        },
    }


def committed_verdict() -> dict:
    rep = json.loads((HERE / "validation_report.json").read_text())
    freq = rep["frequency"]
    return {
        "case": {"radius_m": freq["setup"]["radius_m"],
                 "dx_m": freq["setup"]["dx_m"],
                 "steps": freq["setup"]["steps"],
                 "periods_simulated": freq["setup"]["periods_simulated"]},
        "omega_reference_rad_per_s": freq["reference"]["omega_rad_per_s"],
        "fit_omega_rad_per_s": freq["frequency"]["fit_second_moment_rad_per_s"],
        "fit_error_pct": freq["frequency"]["fit_second_moment_rel_error_pct"],
        "verdict": rep["verdict"],
        "final_r_eff_m": freq["final_state"]["r_eff"],
        "initial_r_eff_m": freq["initial_state"]["r_eff"],
    }


def make_figure(meas: dict, committed: dict, path: Path,
                recorded: dict, dpi: int = 130) -> Path:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(1, 3, figsize=(16.5, 4.4))

    # (a) the release-case trace: the shape that makes the fit a pseudo-frequency
    ax = axes[0]
    t = meas["trace"]["t_s"]
    s = meas["trace"]["s_second_moment"]
    ax.plot(t, s, lw=0.9, color="C0", label=r"$s(t)$ (2nd-moment mode 2)")
    omega_fit = meas["release_case"]["fit_omega_rad_per_s"]
    omega_ref = meas["release_case"]["omega_reference_rad_per_s"]
    t_grid = [t[0] + (t[-1] - t[0]) * i / 400.0 for i in range(401)]
    ax.plot(t_grid, [s[0] * math.cos(omega_ref * (ti - t[0])) for ti in t_grid],
            "--", color="k", lw=1.0, label=r"$\cos(\omega_2 t)$, analytic")
    ax.plot(t_grid, [s[0] * math.cos(omega_fit * (ti - t[0])) for ti in t_grid],
            ":", color="C3", lw=1.4, label=r"$\cos(\omega_{\rm fit} t)$")
    ax.set_xlabel("t [s]")
    ax.set_ylabel(r"$s = (A-B)/(A+B)$ [-]")
    ax.set_title("(a) CI release case: diverging trace")
    ax.legend(fontsize=8)
    ax.grid(alpha=0.3)

    # (b) the divergence: r_eff grows ~5x, so the "frequency" is not capillary
    ax = axes[1]
    ax.bar(["initial", "final"],
           [meas["divergence"]["r_eff_initial_m"] * 1e3,
            meas["divergence"]["r_eff_final_m"] * 1e3], color=["C0", "C3"])
    for i, v in enumerate([meas["divergence"]["r_eff_initial_m"],
                           meas["divergence"]["r_eff_final_m"]]):
        ax.text(i, v * 1e3, f"{v * 1e3:.1f} mm", ha="center", va="bottom",
                fontsize=9)
    ax.set_ylabel(r"$r_{\rm eff}$ [mm]")
    ax.set_title(
        f"(b) drop spreads x{meas['divergence']['r_eff_growth_factor']:.1f} in "
        f"{meas['release_case']['periods_simulated']:.1f} T")
    ax.grid(alpha=0.3, axis="y")

    # (c) the error record: per-platform pin vs the platform-independent verdict
    ax = axes[2]
    labels = [f"release case\nR={committed['case']['radius_m'] * 1e3:.0f} mm\n"
              "(committed verdict)",
              f"CI case\nR={meas['release_case']['radius_m'] * 1e3:.0f} mm\n"
              f"({meas['platform']})"]
    vals = [committed["fit_error_pct"], meas["release_case"]["fit_error_pct"]]
    ax.bar(labels, vals, color=["C3", "C1"])
    for lbl, v in zip(labels, vals):
        ax.text(lbl, v, f"{v:+.1f} %", ha="center",
                va="top" if v < 0 else "bottom", fontsize=9)
    for arch, v in recorded.items():
        ax.axhline(v, ls=":", lw=1.0, color="gray")
        ax.text(1.42, v, f"{arch}: {v:+.2f} %", fontsize=7.5, va="center")
    ax.axhline(15, color="r", ls="--", lw=1.2, label="15 % gate")
    ax.axhline(-15, color="r", ls="--", lw=1.2)
    ax.axhline(0, color="k", lw=0.8)
    ax.set_ylabel("mode-2 frequency error [%]")
    ax.set_title("(c) recorded record vs 15 % gate")
    ax.legend(fontsize=8)
    ax.grid(alpha=0.3, axis="y")

    fig.tight_layout()
    fig.savefig(path, dpi=dpi)
    plt.close(fig)
    return path


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--json-out",
                    default=str(HERE / "ci_pin_evidence.json"))
    ap.add_argument("--fig", default="/tmp/m4802_ci_pin.png")
    ap.add_argument("--dpi", type=int, default=130)
    ap.add_argument("--skip-run", action="store_true",
                    help="only re-render from an existing evidence JSON")
    args = ap.parse_args()

    out = Path(args.json_out)
    recorded = {"loongarch64": -31.78, "x86_64": 8.64}
    if args.skip_run and out.exists():
        ev = json.loads(out.read_text())
    else:
        ev = measure()
        ev["committed_record"] = committed_verdict()
        ev["recorded_pins_pct"] = recorded
        out.write_text(json.dumps(ev, indent=1))
        print(f"wrote {out}")
    fig = make_figure(ev, ev["committed_record"], Path(args.fig), recorded,
                      dpi=args.dpi)
    print(f"wrote {fig}")
    print(f"CI case  : {ev['release_case']['fit_error_pct']:+.2f} % "
          f"(r2 = {ev['release_case']['fit_r2']:.3f}, "
          f"estimators differ by "
          f"{ev['release_case']['estimator_agreement_pct']:+.1f} pp)")
    print(f"release  : {ev['committed_record']['fit_error_pct']:+.2f} %, "
          f"within_15pct = "
          f"{ev['committed_record']['verdict']['worst_within_15pct']}")
    print(f"divergence: r_eff x{ev['divergence']['r_eff_growth_factor']:.1f}, "
          f"max|v|/{ev['divergence']['capillary_velocity_m_per_s']:.2f} = "
          f"{ev['divergence']['max_speed_over_capillary']:.2e} cap. velocities")


if __name__ == "__main__":
    main()
