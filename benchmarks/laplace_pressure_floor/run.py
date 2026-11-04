"""Decompose the WITHHELD Young-Laplace pressure jump into its error sources.

Why this benchmark exists
-------------------------
Three successive rounds of work on the CSF surface-tension model all failed the
same end-to-end check and all reported roughly the same number:

    ``docs/verification_surface_tension_laplace.md``   -93.6 %  (legacy CSF)
    ``docs/verification_surface_tension_consistent.md`` -75.6 %  (consistent CSF)
    ``docs/verification_surface_tension_curvature.md``  -87.6 %  (fit curvature)

Those documents attribute the miss to the *surface-tension discretisation* and
hand a "fix the EOS / background pressure" note to the next mission.  This
benchmark replaces that guess with a measurement: it isolates the four error
sources that are simultaneously present in the check, using only quantities that
can be read off the run.

  C1  measurement definition.  ``Delta p`` was read as
      ``p(r < 0.5R) - p(0.80R..0.95R)``.  The upper band is *inside* the region
      where SPH sum-density is deficient (the free surface), so the band mean is
      a density artefact, not a fluid pressure.  Absolute reference: the Tait
      law is normalised so that ``p(rho0) = 0``; a drop in vacuum must therefore
      satisfy ``p_core = sigma/R`` directly.

  C2  initialisation bias.  A hex lattice carries a uniform sum-density offset
      ``rho/rho0 = 1 + b`` with ``b ~ 5.9e-3`` (h = 1.3 dx, independent of c0),
      because the discrete kernel sum is not exactly one.  With c0 = 20 m/s that
      is a spurious *initial* pressure of ``B[(1+b)^gamma - 1]`` = 2.4 kPa,
      i.e. 6.7 x the Laplace signal.  Calibrating the particle mass so the
      lattice sums to exactly rho0 removes it.

  C3  free-surface pressure closure.  ``tait_pressure`` clamps p to >= 0, and
      the outer shell's density deficit (-11.6 % at 0.85-0.95R, -34 % at
      0.95-1.05R) puts every particle there at p = 0.  The interface region
      therefore cannot hold ``sigma/R``; only the interior part of the pressure
      profile can, and the interior settles at a fraction of the analytic jump.

  C4  EOS resolution floor.  ``Delta p = sigma/R`` needs ``rho/rho0 - 1`` of
      9.0e-4 (c0 = 20) while the *dynamical* sum-density noise is ~5e-4, so
      pressure noise ``rho0 c0^2 s_rho`` is of the order of the signal.  Lowering
      c0 raises the required compression: a c0 ladder separates C4 from C3.

The c0 ladder is the discriminator: C4 predicts the error shrinks as c0 falls,
C3 predicts it does not.

The decomposition pointed at C3, and C3 is now fixed: ``rhs`` closes the
free-surface pressure with the Adami/Hu/Adams extrapolation from *interior*
neighbours (``Config.free_surface_pressure``). The same grid is therefore run
twice -- ``runs`` (closure off, the historical record that isolates C1-C4) and
``closure_runs`` (closure on, the acceptance set: ``|Delta p|`` within 15 % of
``sigma/R`` at every point).

Usage
-----
    PYTHONPATH=src python benchmarks/laplace_pressure_floor/run.py
    PYTHONPATH=src python benchmarks/laplace_pressure_floor/run.py --quick
"""
from __future__ import annotations

import argparse
import importlib.util
import json
import math
from pathlib import Path

import torch

from tensorsph.droplet_impact import (
    Config,
    density_from_neighbor_pairs,
    rhs,
    stable_dt,
    tait_pressure,
)
from tensorsph.neighbor_search import cell_linked_pairs

torch.set_num_threads(2)

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]

RADIUS = 0.02          # m
SIGMA = 7.2            # N/m  (10 x water: shortens the capillary time)
RHO0 = 1000.0          # kg/m^3
GAMMA = 7.0
H_FACTOR = 1.3
VISCOSITY = 0.01       # m^2/s
DP_ANALYTIC = SIGMA / RADIUS                      # 360 Pa
T_SIGMA = math.sqrt(RHO0 * RADIUS ** 3 / SIGMA)   # 0.0333 s

BANDS = ((0.0, 0.50), (0.50, 0.70), (0.70, 0.85), (0.85, 0.95), (0.95, 1.05))


def _load_sibling(name: str, filename: str):
    spec = importlib.util.spec_from_file_location(name, HERE.parent / filename)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


_law = _load_sibling("laplace_law_run", "laplace_law/run.py")
hexagonal_droplet = _law.hexagonal_droplet


# --------------------------------------------------------------------------- #
# C2: the lattice bias, and how to remove it
# --------------------------------------------------------------------------- #
def calibrated_droplet(dx: float, rho0: float = RHO0):
    """Hex drop whose sum-density is exactly ``rho0`` in the interior.

    The lattice-constant mass ``rho0 * dx * dy`` makes the discrete kernel sum
    ``rho/rho0 = 1 + b`` with b > 0, i.e. the drop starts pre-compressed.  One
    pass of ``m -> m / (1 + b)`` (b measured in the core, where the kernel is
    complete) removes the offset to machine precision.
    """
    h = H_FACTOR * dx
    pos, mass0 = hexagonal_droplet(RADIUS, dx, rho0)
    pairs = cell_linked_pairs(pos, 2.0 * h)
    rho = density_from_neighbor_pairs(pos, mass0, h, pairs)
    r = torch.linalg.vector_norm(pos, dim=1) / RADIUS
    bias = float(rho[r < 0.5].mean() / rho0)
    mass = mass0 / bias
    rho_cal = density_from_neighbor_pairs(pos, mass, h, pairs)
    return pos, mass, h, pairs, bias, float(rho_cal[r < 0.5].mean() / rho0)


def density_structure(dx: float, sound_speeds: list[float]):
    """Lattice sum-density: noise floor (C4) and radial trend (C3)."""
    h = H_FACTOR * dx
    pos, mass = hexagonal_droplet(RADIUS, dx, RHO0)
    pairs = cell_linked_pairs(pos, 2.0 * h)
    rho = density_from_neighbor_pairs(pos, mass, h, pairs)
    rel = rho / RHO0 - 1.0
    r = torch.linalg.vector_norm(pos, dim=1) / RADIUS
    core = rel[r < 0.5]
    s_rho = float(core.std()) if core.numel() > 1 else 0.0

    bands = {}
    for lo, hi in BANDS:
        m = (r >= lo) & (r < hi)
        bands[f"{lo:.2f}-{hi:.2f}"] = {
            "count": int(m.sum()),
            "mean_rel_density": float(rel[m].mean()) if bool(m.any()) else float("nan"),
            "std_rel_density": float(rel[m].std()) if int(m.sum()) > 1 else 0.0,
        }

    per_c0 = {}
    for c0 in sound_speeds:
        bulk = RHO0 * c0 ** 2 / GAMMA
        comp_exact = (1.0 + DP_ANALYTIC / bulk) ** (1.0 / GAMMA) - 1.0
        p_noise = RHO0 * c0 ** 2 * s_rho
        per_c0[f"{c0:g}"] = {
            "bulk_modulus_Pa": bulk,
            "compression_required": comp_exact,
            "p_noise_1sigma_Pa": p_noise,
            "p_noise_over_signal": p_noise / DP_ANALYTIC,
            "initial_spurious_pressure_Pa": bulk * (
                (1.0 + float(core.mean())) ** GAMMA - 1.0),
        }
    return {
        "dx_m": dx, "h_m": h, "n_particles": int(pos.shape[0]),
        "lattice_bias_rel_density": float(core.mean()),
        "lattice_bias_std": s_rho,
        "bands": bands,
        "per_sound_speed": per_c0,
    }


# --------------------------------------------------------------------------- #
# C1 + C3 + C4: relax and read the jump three ways
# --------------------------------------------------------------------------- #
def radial_profile(pos, values, n_bins: int = 12):
    r = torch.linalg.vector_norm(pos, dim=1) / RADIUS
    rows = []
    for b in range(n_bins):
        lo, hi = b / n_bins, (b + 1) / n_bins
        m = (r >= lo) & (r < hi)
        if not bool(m.any()):
            continue
        rows.append({"r_over_R_mid": 0.5 * (lo + hi), "count": int(m.sum()),
                     "mean": float(values[m].mean()),
                     "std": float(values[m].std()) if int(m.sum()) > 1 else 0.0})
    return rows


def relax(dx: float, c0: float, scheme: str = "fit_quadratic",
          calibrate_mass: bool = True, t_end_over_t_sigma: float = 1.5,
          max_steps: int = 4000, sample_every: int = 50, closure: bool = False):
    """Advance the drop and read ``Delta p`` with three definitions.

    ``absolute``  ``p_core - 0``  (Tait is normalised at ``rho0``; the vacuum is
                  the p = 0 reference, so this is the physical jump).
    ``bandA``     ``p_core - p(0.80..0.95R)``, the definition used by the three
                  previous verification rounds (contaminated by C3).
    ``bandB``     ``p_core - p(0.60..0.80R)``, inside the deficit zone's edge.
    """
    cfg = Config(dx=dx, h_factor=H_FACTOR, rho0=RHO0, sound_speed=c0,
                 gamma=GAMMA, gravity=0.0, viscosity=VISCOSITY,
                 surface_tension_coefficient=SIGMA, csf_scheme="consistent",
                 csf_curvature=scheme,
                 free_surface_pressure=closure)
    h = cfg.h_factor * cfg.dx
    dt = float(stable_dt(cfg))
    t_end = t_end_over_t_sigma * T_SIGMA
    steps = min(max_steps, int(math.ceil(t_end / dt)))
    dt = t_end / steps

    if calibrate_mass:
        pos, mass, h, _, bias, bias_after = calibrated_droplet(dx)
    else:
        pos, mass = hexagonal_droplet(RADIUS, dx, RHO0)
        bias, bias_after = float("nan"), float("nan")
    vel = torch.zeros_like(pos)
    pairs = cell_linked_pairs(pos, 2.0 * h)
    rho = density_from_neighbor_pairs(pos, mass, h, pairs)
    pressure = tait_pressure(rho, cfg)
    v_ref = math.sqrt(SIGMA / (RHO0 * RADIUS))

    def read(pos, pressure):
        r = torch.linalg.vector_norm(pos, dim=1) / RADIUS
        core = r < 0.5
        shellA = (r >= 0.80) & (r < 0.95)
        shellB = (r >= 0.60) & (r < 0.80)
        return {
            "absolute": float(pressure[core].mean()),
            "bandA": float(pressure[core].mean() - pressure[shellA].mean()),
            "bandB": float(pressure[core].mean() - pressure[shellB].mean()),
        }

    trace = {"t": [], "absolute": [], "bandA": [], "bandB": [],
             "vmax_over_vref": [], "rho_core_rel": []}
    for step in range(steps):
        pairs = cell_linked_pairs(pos, 2.0 * h)
        rho, pressure, acc = rhs(pos, vel, mass, cfg, pairs=pairs, wall=None,
                                 dt=dt)
        vel = vel + dt * acc
        pos = pos + dt * vel
        if step % sample_every == 0 or step == steps - 1:
            r = torch.linalg.vector_norm(pos, dim=1) / RADIUS
            vals = read(pos, pressure)
            trace["t"].append((step + 1) * dt / T_SIGMA)
            trace["absolute"].append(vals["absolute"])
            trace["bandA"].append(vals["bandA"])
            trace["bandB"].append(vals["bandB"])
            trace["rho_core_rel"].append(float(rho[r < 0.5].mean() / RHO0 - 1.0))
            trace["vmax_over_vref"].append(
                float(torch.linalg.vector_norm(vel, dim=1).max()) / v_ref)

    final = read(pos, pressure)
    r = torch.linalg.vector_norm(pos, dim=1) / RADIUS

    # The breathing mode (period ~ t_sigma) means a single snapshot is not the
    # equilibrium value; average over the last 40 % of the window as well.
    n_tail = max(1, int(0.4 * len(trace["absolute"])))
    tail_abs = sum(trace["absolute"][-n_tail:]) / n_tail

    def score(value):
        err = (value - DP_ANALYTIC) / DP_ANALYTIC
        return {"delta_p_Pa": value, "rel_error_pct": err * 100.0,
                "within_3pct": abs(err) < 0.03, "within_15pct": abs(err) < 0.15}

    return {
        "dx_m": dx, "h_m": h, "n_particles": int(pos.shape[0]),
        "sound_speed": c0, "curvature_scheme": scheme,
        "mass_calibrated": calibrate_mass,
        "lattice_bias": bias, "lattice_bias_after_calibration": bias_after,
        "dt_s": dt, "steps": steps, "t_end_s": t_end,
        "t_over_t_sigma": t_end / T_SIGMA,
        "definition_absolute": score(final["absolute"]),
        "definition_bandA": score(final["bandA"]),
        "definition_bandB": score(final["bandB"]),
        "absolute_time_average_tail": score(tail_abs),
        "density": {
            "compression_ratio": (float(rho[r < 0.5].mean() / RHO0) - 1.0)
            / ((1.0 + DP_ANALYTIC / (RHO0 * c0 ** 2 / GAMMA)) ** (1.0 / GAMMA)
               - 1.0),
            "rho_core_over_rho0": float(rho[r < 0.5].mean() / RHO0),
            "rho_core_rel_std": float((rho[r < 0.5] / RHO0 - 1.0).std()),
            "rho_shell_0p80_0p95_over_rho0": float(
                rho[(r >= 0.80) & (r < 0.95)].mean() / RHO0),
            "compression_required": (1.0 + DP_ANALYTIC / (RHO0 * c0 ** 2 / GAMMA))
            ** (1.0 / GAMMA) - 1.0,
        },
        "spurious_current": {
            "v_ref_m_per_s": v_ref,
            "vmax_over_vref_final": trace["vmax_over_vref"][-1],
            "passes_adami_0p1": trace["vmax_over_vref"][-1] < 0.1,
        },
        "finite": bool(torch.isfinite(pos).all() and torch.isfinite(vel).all()),
        "density_profile": radial_profile(pos, rho / RHO0),
        "pressure_profile": radial_profile(pos, pressure),
        "trace": trace,
    }


def closure_acceptance(runs):
    """Machine-checkable verdict on the P0 acceptance criterion."""
    if not runs:
        return {"evaluated": False}
    worst = max(runs, key=lambda r: abs(r["absolute_time_average_tail"]
                                        ["rel_error_pct"]))
    return {
        "evaluated": True,
        "criterion": ("|absolute_time_average_tail rel. error| < 15 % of "
                      "sigma/R at every recorded case"),
        "worst_case": {"dx_m": worst["dx_m"],
                       "sound_speed": worst["sound_speed"],
                       "rel_error_pct":
                           worst["absolute_time_average_tail"]["rel_error_pct"]},
        "max_abs_rel_error_pct": abs(
            worst["absolute_time_average_tail"]["rel_error_pct"]),
        "within_15pct": abs(worst["absolute_time_average_tail"]
                            ["rel_error_pct"]) < 15.0,
        "within_3pct": all(r["absolute_time_average_tail"]["within_3pct"]
                           for r in runs),
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--quick", action="store_true")
    ap.add_argument("--dx", type=float, default=0.002)
    ap.add_argument("--sound-speeds", type=float, nargs="+",
                    default=[20.0, 10.0, 5.0])
    ap.add_argument("--t-end", type=float, default=1.5,
                    help="window length in capillary times")
    ap.add_argument("--skip-closure", action="store_true",
                    help="skip the free-surface-closure acceptance grid")
    args = ap.parse_args()

    dxs = [args.dx] if args.quick else [args.dx, 0.0015]
    c0s = [20.0, 5.0] if args.quick else list(args.sound_speeds)

    structures = [density_structure(dx, c0s) for dx in dxs]
    runs = []
    for dx in dxs:
        for c0 in c0s:
            run = relax(dx, c0, t_end_over_t_sigma=args.t_end)
            runs.append(run)
            print(f"dx={dx:g} c0={c0:5g}: abs {run['definition_absolute']['rel_error_pct']:+7.1f} %"
                  f"  bandA {run['definition_bandA']['rel_error_pct']:+7.1f} %"
                  f"  tail-avg {run['absolute_time_average_tail']['rel_error_pct']:+7.1f} %"
                  f"  comp {run['density']['rho_core_over_rho0'] - 1.0:+.2e}"
                  f"/{run['density']['compression_required']:.2e}"
                  f"  steps {run['steps']}", flush=True)

    closure_runs = []
    if not args.skip_closure:
        for dx in dxs:
            for c0 in c0s:
                run = relax(dx, c0, t_end_over_t_sigma=args.t_end, closure=True)
                closure_runs.append(run)
                print(f"closure dx={dx:g} c0={c0:5g}: "
                      f"tail-avg {run['absolute_time_average_tail']['rel_error_pct']:+7.1f} %"
                      f"  abs {run['definition_absolute']['rel_error_pct']:+7.1f} %"
                      f"  comp ratio {run['density']['compression_ratio']:.2f}"
                      f"  steps {run['steps']}", flush=True)

    # controls: (a) uncalibrated mass at the highest c0 (isolates C2),
    #           (b) legacy curvature operator at the lowest c0 (isolates the
    #               operator from the closure problem)
    ref_c0 = c0s[0]
    uncal = relax(args.dx, ref_c0, calibrate_mass=False,
                  t_end_over_t_sigma=args.t_end)
    print(f"control uncalibrated dx={args.dx:g} c0={ref_c0:g}: "
          f"abs {uncal['definition_absolute']['rel_error_pct']:+7.1f} %", flush=True)
    ctrl_scheme = relax(args.dx, c0s[-1], scheme="mls",
                        t_end_over_t_sigma=args.t_end)
    print(f"control mls dx={args.dx:g} c0={c0s[-1]:g}: "
          f"abs {ctrl_scheme['definition_absolute']['rel_error_pct']:+7.1f} %",
          flush=True)

    report = {
        "benchmark": "laplace_pressure_floor",
        "reference": {"delta_p_Pa": DP_ANALYTIC, "sigma_N_per_m": SIGMA,
                      "radius_m": RADIUS, "rho0": RHO0, "gamma": GAMMA,
                      "t_sigma_s": T_SIGMA},
        "structures": structures,
        "runs": runs,
        "controls": {"uncalibrated_c0_reference": uncal,
                     "mls_scheme_lowest_c0": ctrl_scheme},
        "closure_runs": closure_runs,
        "closure_acceptance": closure_acceptance(closure_runs),
    }
    (HERE / "report.json").write_text(json.dumps(report, indent=2))
    report["figures"] = [make_figures(report)]
    (HERE / "report.json").write_text(json.dumps(report, indent=2))
    print("wrote", HERE / "report.json", report["figures"])


# --------------------------------------------------------------------------- #
# figures
# --------------------------------------------------------------------------- #
def make_figures(report, out_dir=HERE):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    runs = report["runs"]
    structs = report["structures"]
    dx0 = structs[0]["dx_m"]
    primary = [r for r in runs if r["dx_m"] == dx0]
    primary.sort(key=lambda r: -r["sound_speed"])

    fig, axes = plt.subplots(2, 2, figsize=(11.5, 8.6))

    ax = axes[0][0]
    run = primary[0]
    r = [p["r_over_R_mid"] for p in run["density_profile"]]
    ax.plot(r, [(p["mean"] - 1.0) * 100.0 for p in run["density_profile"]],
            "o-", ms=3, label=r"relaxed $\rho/\rho_0-1$")
    st = structs[0]
    r0 = []
    for key, val in st["bands"].items():
        lo, hi = (float(x) for x in key.split("-"))
        r0.append(0.5 * (lo + hi))
    ax.plot(r0, [st["bands"][k]["mean_rel_density"] * 100.0
                 for k in st["bands"]], "s--", ms=4,
            label="initial lattice (C2/C3)")
    ax.axhline(0.0, color="k", lw=0.8)
    ax.set_xlabel(r"$r/R$")
    ax.set_ylabel(r"$\rho/\rho_0 - 1$ [%]")
    ax.set_title(f"(a) sum-density: +{st['lattice_bias_rel_density'] * 100:.2f} % bias,"
                 " surface deficit")
    ax.legend(fontsize=7)
    ax.grid(alpha=0.3)

    ax = axes[0][1]
    for run in primary:
        ax.plot(run["trace"]["t"], run["trace"]["absolute"], lw=1,
                label=rf"$c_0$={run['sound_speed']:g} m/s, no closure")
    closure_primary = [r for r in report.get("closure_runs", [])
                       if r["dx_m"] == dx0]
    closure_primary.sort(key=lambda r: -r["sound_speed"])
    for run in closure_primary:
        ax.plot(run["trace"]["t"], run["trace"]["absolute"], lw=1, ls="--",
                label=rf"$c_0$={run['sound_speed']:g} m/s, closure")
    ax.axhline(DP_ANALYTIC, color="k", ls="--", lw=1,
               label=rf"$\sigma/R$={DP_ANALYTIC:.0f} Pa")
    ax.set_xlabel(r"$t/t_\sigma$")
    ax.set_ylabel(r"$p_{\rm core}$ [Pa]")
    ax.set_title("(b) absolute core pressure: closure on vs off")
    ax.legend(fontsize=6)
    ax.grid(alpha=0.3)

    ax = axes[1][0]
    labels = ("band A\n(0.80-0.95R)", "band B\n(0.60-0.80R)", "absolute\n$p_{core}$")
    keys = ("definition_bandA", "definition_bandB", "definition_absolute")
    width = 0.8 / max(1, len(primary))
    idx = torch.arange(len(labels)).numpy()
    for k, run in enumerate(primary):
        vals = [run[key]["rel_error_pct"] for key in keys]
        ax.bar(idx + (k - len(primary) / 2 + 0.5) * width, vals, width,
               label=rf"$c_0$={run['sound_speed']:g}")
    ax.axhline(15.0, color="r", ls=":", lw=1, label="+15 %")
    ax.axhline(-15.0, color="r", ls=":", lw=1)
    ax.axhline(0.0, color="k", lw=0.8)
    ax.set_xticks(idx)
    ax.set_xticklabels(labels)
    ax.set_ylabel(r"$(\Delta p - \sigma/R)/(\sigma/R)$ [%]")
    ax.set_title("(c) C1: the read-out definition costs ~60 points")
    ax.legend(fontsize=7)
    ax.grid(alpha=0.3, axis="y")

    ax = axes[1][1]
    c0_axis = sorted({r["sound_speed"] for r in primary})
    ax.plot(c0_axis, [r["density"]["compression_required"] for r in primary],
            "k^--", ms=5, label="required by Young-Laplace")
    ax.plot(c0_axis, [r["density"]["rho_core_over_rho0"] - 1.0 for r in primary],
            "o-", label="achieved (interior)")
    ax.plot(c0_axis, [r["density"]["rho_core_rel_std"] for r in primary],
            "s:", label=r"dynamical $\rho$ noise $\sigma_\rho$ (C4)")
    if closure_primary:
        c0c = sorted({r["sound_speed"] for r in closure_primary})
        ax.plot(c0c, [r["density"]["rho_core_over_rho0"] - 1.0
                      for r in closure_primary], "D-.", ms=5,
                label="achieved with closure")
    ax.set_yscale("log")
    ax.set_xlabel(r"$c_0$ [m/s]")
    ax.set_ylabel(r"$\rho/\rho_0 - 1$")
    ax.set_title("(d) compression: needed vs achieved (C3 broken by closure)")
    ax.legend(fontsize=7)
    ax.grid(alpha=0.3, which="both")

    fig.suptitle("TensorSPH: decomposing the Young-Laplace verification miss")
    fig.tight_layout()
    path = Path(out_dir) / "laplace_pressure_floor.png"
    fig.savefig(path, dpi=130)
    plt.close(fig)
    return path.name


if __name__ == "__main__":
    main()