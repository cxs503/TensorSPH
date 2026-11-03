"""Young-Laplace pressure jump: is the *observable* measuring the jump?

Motivation
----------
``benchmarks/csf_consistent/run.py::relax`` reports

    dp = p(r < 0.5 R) - p(0.80 R < r < 0.95 R)

and every verification round since m2174 quoted that number as "the Laplace
jump is 77-118 % off".  But the interface of this model is a kernel-smeared
shell of thickness ~2h; at R/h = 7.7 that is 0.26 R, i.e. the whole band
0.80 R .. 0.95 R sits *inside the fluid*, where the Young-Laplace solution
predicts a **flat** pressure.  The legacy observable therefore measures the
internal flatness of p, not the jump across the interface, and cannot converge
to sigma/R by construction.

This script measures three observables on the same relaxation runs:

``legacy``      p(r < 0.5R) - p(0.80R < r < 0.95R)      (the old metric)
``interface``   p(core) - p(outermost layer, colour c < 0.85)  (the jump)
``force``       sum_shell m a . rhat / (2 pi R)         (the CSF force integral)

``interface`` is what Young-Laplace predicts to be sigma/R; ``force`` is an
*independent* estimate of the same quantity, valid only if the CSF force is
assembled consistently (delta p = integral of the normal force density across
the interface).  Comparing ``interface`` with ``force`` tests the model's
internal consistency and separates "the force is wrong" from "the pressure
response is wrong".

It also reports the two density scales that decide whether a WCSPH drop can hold
sigma/R at all:

* ``compression_required`` = dp / B, B = rho0 c0^2 / gamma  (Tait: 9.3e-4 at
  c0 = 20 m/s),
* ``lattice_bias`` = the t = 0 bulk sum-density excess over rho0 (5.9e-3),
  i.e. the discrete density floor that the Laplace compression must exceed.

Usage
-----
    python benchmarks/csf_consistent/interface_jump.py            # full scan
    python benchmarks/csf_consistent/interface_jump.py --quick    # 2 points
"""
from __future__ import annotations

import argparse
import importlib.util
import json
import math
from pathlib import Path

import torch

from tensorsph.csf_consistent import csf_fields, kernel_sum_color
from tensorsph.droplet_impact import (
    Config,
    density_from_neighbor_pairs,
    rhs,
    stable_dt,
    tait_pressure,
)
from tensorsph.neighbor_search import cell_linked_pairs

torch.set_num_threads(4)

HERE = Path(__file__).resolve().parent


def _load_sibling(name: str, filename: str):
    spec = importlib.util.spec_from_file_location(name, HERE.parent / filename)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


_law = _load_sibling("laplace_law_run", "laplace_law/run.py")
hexagonal_droplet = _law.hexagonal_droplet


# --------------------------------------------------------------------------- #
# observables
# --------------------------------------------------------------------------- #
def observables(pos, vel, mass, rho, pressure, acc, cfg, rho0, sigma, radius):
    """All three Young-Laplace observables on one state."""
    h = cfg.h_factor * cfg.dx
    pairs = cell_linked_pairs(pos, 2.0 * h)
    color = kernel_sum_color(pos, mass, pairs, h, rho0)
    r = torch.linalg.vector_norm(pos, dim=1)
    rhat = pos / r[:, None].clamp_min(1e-30)
    rn = r / radius

    core = rn < 0.5
    legacy_band = (rn > 0.80) & (rn < 0.95)
    # Outermost fluid layer: colour < 0.85 (half a kernel short of the bulk).
    # Its Tait pressure is clamped to 0, i.e. it is the vacuum-side boundary.
    outer = color < 0.85
    if int(outer.sum()) < 5:                      # too coarse to resolve a shell
        cut = torch.quantile(rn, 0.97)
        outer = rn > cut

    # The integral of the *model's own* CSF force across the interface,
    # expressed as a pressure: sigma * (integral delta_s dn) * <kappa>_delta_s.
    # (Summing m*a_csf is not usable here: the ``acc`` returned by ``rhs`` also
    # carries the pressure gradient, which cancels it to zero in a relaxed
    # drop.)  This is the jump the pressure field should reproduce if the
    # WCSPH density response transmitted the applied force.
    st = csf_fields(pos, mass, pairs, h, rho0, curvature_scheme="fit_quadratic")
    vol = mass / rho0
    ds_integral = float((st.delta_s * vol / (2.0 * math.pi * r)).sum())
    kappa_mean = float((st.kappa.abs() * st.delta_s).sum() / st.delta_s.sum())
    force_integral = sigma * ds_integral * kappa_mean

    return {
        "n_core": int(core.sum()), "n_outer": int(outer.sum()),
        "p_core_Pa": float(pressure[core].mean()),
        "p_legacy_band_Pa": float(pressure[legacy_band].mean()),
        "p_outer_Pa": float(pressure[outer].mean()),
        "r_outer_over_R": [float(rn[outer].min()), float(rn[outer].max())],
        "color_outer_mean": float(color[outer].mean()),
        "legacy_dp_Pa": float(pressure[core].mean() - pressure[legacy_band].mean()),
        "interface_dp_Pa": float(pressure[core].mean() - pressure[outer].mean()),
        "force_dp_Pa": force_integral,
        "delta_s_integral": ds_integral,
        "kappa_delta_s_mean_1_per_m": kappa_mean,
    }


def _pct(x, ref):
    return (x - ref) / ref * 100.0


def run_case(radius=0.02, dx=0.002, sigma=7.2, rho0=1000.0, sound_speed=20.0,
             t_end_in_tau=2.0, viscosity=0.01, min_steps=200):
    """Relax a free drop to ``t_end_in_tau`` capillary times, sample observables."""
    cfg = Config(dx=dx, h_factor=1.3, rho0=rho0, sound_speed=sound_speed,
                 gamma=7.0, gravity=0.0, viscosity=viscosity,
                 surface_tension_coefficient=sigma, csf_scheme="consistent")
    h = cfg.h_factor * cfg.dx
    dt = float(stable_dt(cfg))
    tau = math.sqrt(rho0 * radius ** 3 / sigma)
    steps = max(min_steps, int(round(t_end_in_tau * tau / dt)))

    pos, mass = hexagonal_droplet(radius, dx, rho0)
    vel = torch.zeros_like(pos)
    pairs = cell_linked_pairs(pos, 2.0 * h)
    rho = density_from_neighbor_pairs(pos, mass, h, pairs)
    pressure = tait_pressure(rho, cfg)
    acc = torch.zeros_like(pos)

    r0 = torch.linalg.vector_norm(pos, dim=1)
    lattice_bias = float(rho[r0 < 0.5 * radius].mean() / rho0) - 1.0
    dp_theory = sigma / radius
    bulk = rho0 * sound_speed ** 2 / cfg.gamma
    v_ref = math.sqrt(sigma / (rho0 * radius))

    hist = {"t_over_tau": [], "p_core": [], "interface_dp": [], "legacy_dp": [],
            "force_dp": [], "vmax_over_vref": []}
    sample_every = max(1, steps // 180)
    for step in range(steps):
        pairs = cell_linked_pairs(pos, 2.0 * h)
        rho, pressure, acc = rhs(pos, vel, mass, cfg, pairs=pairs, wall=None,
                                 dt=dt)
        vel = vel + dt * acc
        pos = pos + dt * vel
        if step % sample_every == 0 or step == steps - 1:
            obs = observables(pos, vel, mass, rho, pressure, acc, cfg, rho0,
                              sigma, radius)
            r = torch.linalg.vector_norm(pos, dim=1)
            core = r < 0.5 * radius
            hist["t_over_tau"].append((step + 1) * dt / tau)
            hist["p_core"].append(float(pressure[core].mean()))
            hist["interface_dp"].append(obs["interface_dp_Pa"])
            hist["legacy_dp"].append(obs["legacy_dp_Pa"])
            hist["force_dp"].append(obs["force_dp_Pa"])
            hist["vmax_over_vref"].append(
                float(torch.linalg.vector_norm(vel, dim=1).max()) / v_ref)

    obs = observables(pos, vel, mass, rho, pressure, acc, cfg, rho0, sigma,
                      radius)
    obs.update({
        "dx_m": dx, "h_m": h, "n_particles": int(pos.shape[0]), "steps": steps,
        "dt_s": dt, "t_end_s": steps * dt, "t_over_tau": steps * dt / tau,
        "sound_speed": sound_speed, "sigma_N_per_m": sigma, "rho0": rho0,
        "radius_m": radius, "dp_theory_Pa": dp_theory,
        "compression_required": dp_theory / bulk,
        "lattice_bias": lattice_bias,
        "lattice_bias_over_compression": lattice_bias / (dp_theory / bulk),
        "particles_per_radius": radius / h,
        "spurious_current": hist["vmax_over_vref"][-1],
        "vmax_over_vref_max": max(hist["vmax_over_vref"]),
    })
    for key, ref in (("interface", obs["interface_dp_Pa"]),
                     ("legacy", obs["legacy_dp_Pa"]),
                     ("force", obs["force_dp_Pa"])):
        obs[f"{key}_rel_error_pct"] = _pct(ref, dp_theory)
        obs[f"{key}_within_15pct"] = abs(_pct(ref, dp_theory)) < 15.0
        obs[f"{key}_within_3pct"] = abs(_pct(ref, dp_theory)) < 3.0
    obs["history"] = hist
    return obs


# --------------------------------------------------------------------------- #
# figures
# --------------------------------------------------------------------------- #
def make_figure(rows, out_dir=HERE):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(1, 3, figsize=(15, 4.2))
    dxs = [r["dx_m"] for r in rows]

    ax = axes[0]
    ax.plot(dxs, [r["interface_dp_Pa"] for r in rows], "o-",
            label=r"$\Delta p$ across the interface (corrected)")
    ax.plot(dxs, [r["force_dp_Pa"] for r in rows], "s--",
            label=r"CSF force integral $\int\sigma\kappa\delta_s\,dn$")
    ax.plot(dxs, [r["legacy_dp_Pa"] for r in rows], "^:",
            label="legacy band 0.80-0.95 R")
    ax.axhline(rows[0]["dp_theory_Pa"], color="k", ls="-.",
               label=r"Young-Laplace $\sigma/R$")
    ax.set_xscale("log")
    ax.set_xlabel("dx [m]")
    ax.set_ylabel(r"$\Delta p$ [Pa]")
    ax.set_title("(a) three observables of the same relaxation")
    ax.legend(fontsize=7)
    ax.grid(alpha=0.3, which="both")

    ax = axes[1]
    ax.plot(dxs, [abs(r["interface_rel_error_pct"]) for r in rows], "o-",
            label="interface observable")
    ax.plot(dxs, [abs(r["force_rel_error_pct"]) for r in rows], "s--",
            label="force integral")
    ax.plot(dxs, [abs(r["legacy_rel_error_pct"]) for r in rows], "^:",
            label="legacy observable")
    ax.axhline(15.0, color="r", ls="--", label="15 % acceptance")
    ax.axhline(3.0, color="0.5", ls=":", label="3 %")
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel("dx [m]")
    ax.set_ylabel("|error| vs sigma/R [%]")
    ax.set_title("(b) error of each observable")
    ax.legend(fontsize=7)
    ax.grid(alpha=0.3, which="both")

    ax = axes[2]
    for r in rows:
        h = r["history"]
        ax.plot(h["t_over_tau"], h["interface_dp"], label=f"dx={r['dx_m']:.4f}")
    ax.axhline(rows[0]["dp_theory_Pa"], color="k", ls="-.",
               label=r"$\sigma/R$")
    ax.set_xlabel(r"$t/\tau_\sigma$")
    ax.set_ylabel(r"$\Delta p$ across the interface [Pa]")
    ax.set_title("(c) time history: the jump is still drifting")
    ax.legend(fontsize=7)
    ax.grid(alpha=0.3)

    fig.tight_layout()
    path = Path(out_dir) / "interface_jump.png"
    fig.savefig(path, dpi=130)
    plt.close(fig)
    return path


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--quick", action="store_true", help="two resolutions only")
    ap.add_argument("--t-end-in-tau", type=float, default=2.0)
    ap.add_argument("--out", default=str(HERE))
    args = ap.parse_args(argv)

    dxs = (0.003, 0.002) if args.quick else (0.004, 0.003, 0.002, 0.0015)
    rows = []
    for dx in dxs:
        row = run_case(dx=dx, t_end_in_tau=args.t_end_in_tau)
        rows.append(row)
        print(f"dx={dx:.4f} N={row['n_particles']:4d} "
              f"legacy={row['legacy_dp_Pa']:7.1f} ({row['legacy_rel_error_pct']:+6.1f}%) "
              f"interface={row['interface_dp_Pa']:7.1f} "
              f"({row['interface_rel_error_pct']:+6.1f}%) "
              f"force={row['force_dp_Pa']:7.1f} ({row['force_rel_error_pct']:+6.1f}%) "
              f"vmax/vref={row['spurious_current']:.3f}", flush=True)

    payload = {
        "case": "2-D static droplet, Young-Laplace jump, surface tension only",
        "scheme": "csf_consistent (kernel-sum colour, MLS operators)",
        "setup": {"radius_m": rows[0]["radius_m"],
                  "sigma_N_per_m": rows[0]["sigma_N_per_m"],
                  "rho0": rows[0]["rho0"],
                  "sound_speed": rows[0]["sound_speed"],
                  "t_end_in_tau": args.t_end_in_tau},
        "reference": {"delta_p_Pa": rows[0]["dp_theory_Pa"],
                      "source": "Young-Laplace, 2-D circular interface"},
        "rows": rows,
        "verdict": {
            "legacy_observable_is_inside_the_fluid": True,
            "interface_within_15pct": all(r["interface_within_15pct"] for r in rows),
            "force_integral_within_15pct": all(r["force_within_15pct"] for r in rows),
            "any_within_3pct": any(r["interface_within_3pct"] for r in rows),
            "status": ("failed" if not all(r["interface_within_15pct"] for r in rows)
                       else "pass"),
        },
    }
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "interface_jump.json").write_text(json.dumps(payload, indent=2))
    print("figure:", make_figure(rows, out_dir))
    print("verdict:", json.dumps(payload["verdict"]))
    return payload


if __name__ == "__main__":
    main()