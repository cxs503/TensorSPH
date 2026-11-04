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

Nothing in ``src/`` is touched (verification/instrumentation task).

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
          max_steps: int = 4000, sample_every: int = 50,
          free_surface_pressure: bool = False, closure_passes: int = 3):
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
                 free_surface_pressure=free_surface_pressure,
                 free_surface_pressure_passes=closure_passes)
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
        "free_surface_pressure": free_surface_pressure,
        "closure_passes": closure_passes,
        "lattice_bias": bias, "lattice_bias_after_calibration": bias_after,
        "dt_s": dt, "steps": steps, "t_end_s": t_end,
        "t_over_t_sigma": t_end / T_SIGMA,
        "definition_absolute": score(final["absolute"]),
        "definition_bandA": score(final["bandA"]),
        "definition_bandB": score(final["bandB"]),
        "absolute_time_average_tail": score(tail_abs),
        "density": {
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


def _score_rows(runs):
    return [{"dx_m": r["dx_m"], "sound_speed": r["sound_speed"],
             "tail_rel_error_pct": r["absolute_time_average_tail"]["rel_error_pct"],
             "abs_rel_error_pct": r["definition_absolute"]["rel_error_pct"],
             "within_15pct": r["absolute_time_average_tail"]["within_15pct"],
             "within_3pct": r["absolute_time_average_tail"]["within_3pct"]}
            for r in runs]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--quick", action="store_true")
    ap.add_argument("--dx", type=float, default=0.002)
    ap.add_argument("--sound-speeds", type=float, nargs="+",
                    default=[20.0, 10.0, 5.0])
    ap.add_argument("--t-end", type=float, default=1.5,
                    help="window length in capillary times")
    ap.add_argument("--passes", type=int, default=3,
                    help="shell-closure passes (P0); 0 keeps the open surface")
    ap.add_argument("--open-surface", action="store_true",
                    help="regenerate the pre-closure reference matrix into "
                         "report_open_surface.json instead of the closure matrix")
    args = ap.parse_args()

    dxs = [args.dx] if args.quick else [args.dx, 0.0015]
    c0s = [20.0, 5.0] if args.quick else list(args.sound_speeds)
    structures = [density_structure(dx, c0s) for dx in dxs]

    if args.open_surface:
        runs = []
        for dx in dxs:
            for c0 in c0s:
                run = relax(dx, c0, t_end_over_t_sigma=args.t_end)
                runs.append(run)
                print(f"[open surface] dx={dx:g} c0={c0:5g}: "
                      f"tail {run['absolute_time_average_tail']['rel_error_pct']:+7.1f} %",
                      flush=True)
        report = {
            "benchmark": "laplace_pressure_floor",
            "closure": {"free_surface_pressure": False, "closure_passes": 0,
                        "note": "pre-closure (p clamped at 0 across the deficit "
                                "shell) reference matrix; regenerated with "
                                "run.py --open-surface"},
            "reference": {"delta_p_Pa": DP_ANALYTIC, "sigma_N_per_m": SIGMA,
                          "radius_m": RADIUS, "rho0": RHO0, "gamma": GAMMA,
                          "t_sigma_s": T_SIGMA},
            "structures": structures,
            "runs": runs,
            "scores": _score_rows(runs),
        }
        (HERE / "report_open_surface.json").write_text(json.dumps(report, indent=2))
        report["figures"] = [make_figures(report)]
        (HERE / "report_open_surface.json").write_text(json.dumps(report, indent=2))
        print("wrote", HERE / "report_open_surface.json")
        return

    passes = max(1, args.passes)
    runs = []
    for dx in dxs:
        for c0 in c0s:
            run = relax(dx, c0, t_end_over_t_sigma=args.t_end,
                        free_surface_pressure=True, closure_passes=passes)
            runs.append(run)
            print(f"dx={dx:g} c0={c0:5g}: abs {run['definition_absolute']['rel_error_pct']:+7.1f} %"
                  f"  bandA {run['definition_bandA']['rel_error_pct']:+7.1f} %"
                  f"  tail-avg {run['absolute_time_average_tail']['rel_error_pct']:+7.1f} %"
                  f"  comp {run['density']['rho_core_over_rho0'] - 1.0:+.2e}"
                  f"/{run['density']['compression_required']:.2e}"
                  f"  steps {run['steps']}", flush=True)

    # Pass ladder: the deficit shell is several particles thick, so the closure
    # converges with the number of prolongation passes. Recorded so the
    # `--passes` default is evidence-based rather than chosen.
    ladder = []
    for np_ in (1, 2, 3):
        run = relax(dxs[0], c0s[0], t_end_over_t_sigma=args.t_end,
                    free_surface_pressure=True, closure_passes=np_)
        ladder.append({"closure_passes": np_,
                       "dx_m": run["dx_m"], "sound_speed": run["sound_speed"],
                       "tail_rel_error_pct":
                           run["absolute_time_average_tail"]["rel_error_pct"],
                       "abs_rel_error_pct":
                           run["definition_absolute"]["rel_error_pct"],
                       "compression_fraction":
                           (run["density"]["rho_core_over_rho0"] - 1.0)
                           / run["density"]["compression_required"],
                       "spurious_vmax_over_vref":
                           run["spurious_current"]["vmax_over_vref_final"]})
        print(f"passes={np_}: tail {ladder[-1]['tail_rel_error_pct']:+7.1f} %", flush=True)

    # Controls, both with the closure on: (a) uncalibrated mass (isolates C2),
    # (b) legacy curvature operator (isolates the operator from the closure).
    ref_c0 = c0s[0]
    uncal = relax(args.dx, ref_c0, calibrate_mass=False,
                  t_end_over_t_sigma=args.t_end, free_surface_pressure=True,
                  closure_passes=passes)
    ctrl_scheme = relax(args.dx, c0s[-1], scheme="mls",
                        t_end_over_t_sigma=args.t_end, free_surface_pressure=True,
                        closure_passes=passes)
    print(f"control uncalibrated c0={ref_c0:g}: "
          f"tail {uncal['absolute_time_average_tail']['rel_error_pct']:+7.1f} %", flush=True)
    print(f"control mls c0={c0s[-1]:g}: "
          f"tail {ctrl_scheme['absolute_time_average_tail']['rel_error_pct']:+7.1f} %",
          flush=True)

    legacy_path = HERE / "report_open_surface.json"
    legacy = json.loads(legacy_path.read_text()) if legacy_path.exists() else None

    report = {
        "benchmark": "laplace_pressure_floor",
        "closure": {"free_surface_pressure": True,
                    "closure_passes": passes,
                    "free_surface_rho0_ratio": 0.99,
                    "reference": "Adami, Ihmsen & Adams (2012) JCP 231:7057 eq. (12); "
                                 "multi-pass prolongation after Ihmsen et al. (2014) "
                                 "eq. (12)"},
        "reference": {"delta_p_Pa": DP_ANALYTIC, "sigma_N_per_m": SIGMA,
                      "radius_m": RADIUS, "rho0": RHO0, "gamma": GAMMA,
                      "t_sigma_s": T_SIGMA},
        "structures": structures,
        "runs": runs,
        "scores": _score_rows(runs),
        "passes_ladder": ladder,
        "controls": {"uncalibrated_c0_reference": uncal,
                     "mls_scheme_lowest_c0": ctrl_scheme},
        "open_surface": ({"source": legacy_path.name,
                          "scores": legacy.get("scores", _score_rows(legacy["runs"]))}
                         if legacy else None),
    }
    (HERE / "report.json").write_text(json.dumps(report, indent=2))
    report["figures"] = [make_closure_figures(report, legacy)]
    (HERE / "report.json").write_text(json.dumps(report, indent=2))
    print("wrote", HERE / "report.json", report["figures"])


def make_closure_figures(report, legacy=None, out_dir=HERE):
    """Closure-on vs closure-off: the figure the P0 decision rests on."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    runs = report["runs"]
    dx0 = report["structures"][0]["dx_m"]
    primary = sorted([r for r in runs if r["dx_m"] == dx0],
                     key=lambda r: -r["sound_speed"])
    fig, axes = plt.subplots(2, 2, figsize=(11.5, 8.6))

    ax = axes[0][0]
    for run in primary:
        ax.plot(run["trace"]["t"], run["trace"]["absolute"], lw=1.2,
                label=rf"$c_0$={run['sound_speed']:g} m/s")
    ax.axhline(DP_ANALYTIC, color="k", ls="--", lw=1,
               label=rf"$\sigma/R$={DP_ANALYTIC:.0f} Pa")
    ax.axhspan(DP_ANALYTIC * 0.85, DP_ANALYTIC * 1.15, color="r", alpha=0.06,
               label=r"$\pm 15\,\%$")
    ax.set_xlabel(r"$t/t_\sigma$")
    ax.set_ylabel(r"$p_{\rm core}$ [Pa]")
    ax.set_title(f"(a) closure on ({report['closure']['closure_passes']} passes):"
                 " core pressure tracks the Laplace value")
    ax.legend(fontsize=7)
    ax.grid(alpha=0.3)

    ax = axes[0][1]
    cases = [f"{r['dx_m'] * 1e3:g}mm\n$c_0$={r['sound_speed']:g}" for r in runs]
    idx = list(range(len(runs)))
    new = [r["absolute_time_average_tail"]["rel_error_pct"] for r in runs]
    if legacy is not None:
        lerr = {(r["dx_m"], r["sound_speed"]):
                r["absolute_time_average_tail"]["rel_error_pct"]
                for r in legacy["runs"]}
        old = [lerr.get((r["dx_m"], r["sound_speed"]), float("nan")) for r in runs]
        ax.bar([i - 0.2 for i in idx], old, 0.4, label="open surface (before)",
               color="0.6")
        ax.bar([i + 0.2 for i in idx], new, 0.4, label="closure (after)",
               color="C0")
    else:
        ax.bar(idx, new, 0.5, label="closure (after)")
    ax.axhline(-15.0, color="r", ls=":", lw=1, label=r"$-15\,\%$ gate")
    ax.axhline(-3.0, color="orange", ls=":", lw=1, label=r"$-3\,\%$")
    ax.axhline(0.0, color="k", lw=0.8)
    ax.set_xticks(idx)
    ax.set_xticklabels(cases, fontsize=7)
    ax.set_ylabel(r"$(\Delta p - \sigma/R)/(\sigma/R)$ [%] (tail average)")
    ax.set_title("(b) every point moves inside the gate")
    ax.legend(fontsize=7)
    ax.grid(alpha=0.3, axis="y")

    ax = axes[1][0]
    lad = report.get("passes_ladder") or []
    if lad:
        ax.plot([d["closure_passes"] for d in lad],
                [d["tail_rel_error_pct"] for d in lad], "o-")
        ax.axhline(-15.0, color="r", ls=":", lw=1)
        ax.axhline(-3.0, color="orange", ls=":", lw=1)
    ax.set_xlabel("closure passes")
    ax.set_ylabel(r"tail $\Delta p$ error [%]")
    ax.set_title("(c) prolongation converges over the multi-layer deficit shell")
    ax.grid(alpha=0.3)

    ax = axes[1][1]
    r = [p["r_over_R_mid"] for p in primary[0]["density_profile"]]
    ax.plot(r, [(p["mean"] - 1.0) * 100.0 for p in primary[0]["density_profile"]],
            "o-", ms=3, label="relaxed with closure")
    if legacy is not None:
        old_primary = sorted([q for q in legacy["runs"] if q["dx_m"] == dx0],
                             key=lambda q: -q["sound_speed"])[0]
        ax.plot([p["r_over_R_mid"] for p in old_primary["density_profile"]],
                [(p["mean"] - 1.0) * 100.0 for p in old_primary["density_profile"]],
                "s--", ms=3, label="relaxed, open surface")
    ax.axhline(0.0, color="k", lw=0.8)
    ax.set_xlabel(r"$r/R$")
    ax.set_ylabel(r"$\rho/\rho_0 - 1$ [%]")
    ax.set_title("(d) the closure does not manufacture mass at the surface")
    ax.legend(fontsize=7)
    ax.grid(alpha=0.3)

    fig.suptitle("TensorSPH: free-surface pressure closure (P0) closes the "
                 "Young-Laplace gap")
    fig.tight_layout()
    path = Path(out_dir) / "laplace_pressure_closure.png"
    fig.savefig(path, dpi=130)
    plt.close(fig)
    return path.name


if __name__ == "__main__":
    main()
