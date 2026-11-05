"""Colour-function study of the CSF model (0.9.0, merged from agent/m2191).

The Laplace-law *gate* for the shipped scheme lives in the sibling script
``benchmarks/surface_tension/laplace.py``; this file is the colour-function
comparison it grew out of, kept whole because its results are cited in
``docs/benchmarks/surface-tension.md``.

B3 -- surface-tension verification against the 2-D Laplace law.

The gap register (``docs/algorithm-gap-analysis.md``, B3) asks for a static
droplet test *of the pressure jump against curvature and surface tension*. This
script is that test. ``benchmarks/static_droplet/run.py`` checks only the
*direction* of the force, which a force of the wrong magnitude also passes.

Physics
-------
For a 2-D circular droplet of radius ``R`` in vacuum the Young-Laplace relation
with one principal curvature gives

    dp = sigma / R .

In the CSF (continuum surface force) model the surface tension enters as a
volume force ``f = sigma * kappa * delta_s * n``, so the pressure jump it can
support is the radial integral of that force across the interface,

    dp_force = int |f_r| dr ,

and the measured jump after the acoustic transient is
``dp_sim = <p>_core - <p>_shell``. Both are reported as a ratio to ``sigma/R``.

Three discretisation details make the test meaningful instead of decorative.

1. **Density renormalisation.** WCSPH obtains density by kernel summation, which
   has a lattice bias. Through the Tait law that bias becomes a pressure offset
   that swamps the Laplace signal, so the reference density is taken from the
   measured interior density of the unstrained lattice and the bulk pressure of
   a droplet without surface tension is zero by construction. The factor is
   reported.

2. **Pre-stressed start.** With ``--prestress`` the particle masses are scaled so
   that the interior already carries the pressure ``sigma/R`` the interface is
   supposed to hold: ``m = s rho_ref dx^2`` with
   ``s = (1 + (sigma/R)/B)^(1/gamma)``, ``B = rho0 c0^2 / gamma``. Without it the
   droplet starts at zero pressure, which is not its equilibrium, and the
   resulting capillary oscillation overshoots and (measured) destroys the
   lattice before anything can be read.

3. **Resolvable equation of state.** The pressure jump must be a measurable
   fraction of the bulk modulus ``B``. At water-like ``c0 = 30 m/s`` the signal
   is ``(sigma/R)/B`` = 1.6e-6 in density, two orders of magnitude below the
   summation noise, so the defaults use a soft EOS of ``c0 = 3 m/s`` where the
   signal is ~2 % of ``B``.

Usage
-----
    python benchmarks/surface_tension/laplace.py --plot
    python benchmarks/surface_tension/laplace.py --models support_sum --delta-scales 1.0,2.0
"""
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import torch

from tensorsph import Config
from tensorsph.droplet_impact import (
    density_from_neighbor_pairs,
    rhs,
)
from tensorsph.neighbor_search import cell_linked_pairs
from tensorsph.surface_tension import capillary_dt_limit, surface_tension_field


# --------------------------------------------------------------------------- #
# Case construction
# --------------------------------------------------------------------------- #

def droplet_lattice(radius: float, dx: float, dtype, device):
    """Square lattice clipped to a circle of radius ``radius``, centred at 0."""
    half = int(math.ceil(radius / dx)) + 1
    axis = torch.arange(-half, half + 1, dtype=dtype, device=device) * dx
    gx, gy = torch.meshgrid(axis, axis, indexing="xy")
    pos = torch.stack((gx.reshape(-1), gy.reshape(-1)), dim=1)
    inside = torch.linalg.vector_norm(pos, dim=1) <= radius
    return pos[inside].contiguous()


def build_case(radius: float, dx: float, rho0: float, sound_speed: float,
               gamma: float, sigma: float, dtype, device, prestress: bool):
    """Lattice droplet, reference density renormalised, optionally pre-stressed."""
    pos = droplet_lattice(radius, dx, dtype, device)
    n = pos.shape[0]
    probe = Config(dx=dx, rho0=rho0, sound_speed=sound_speed, gamma=gamma)
    h = probe.h_factor * dx
    mass0 = torch.full((n,), rho0 * dx * dx, dtype=dtype, device=device)
    rho_lattice = density_from_neighbor_pairs(pos, mass0, h)
    core = torch.linalg.vector_norm(pos, dim=1) < 0.6 * radius
    rho_ref = float(rho_lattice[core].mean())

    bulk = rho_ref * sound_speed ** 2 / gamma
    dp_theory = sigma / radius
    scale = (1.0 + dp_theory / bulk) ** (1.0 / gamma) if prestress else 1.0

    cfg = Config(dx=dx, rho0=rho_ref, sound_speed=sound_speed, gamma=gamma,
                 gravity=0.0, viscosity=0.005)
    mass = torch.full((n,), scale * rho_ref * dx * dx, dtype=dtype, device=device)
    info = {
        "n_particles": int(n),
        "rho0_nominal": rho0,
        "rho_reference": rho_ref,
        "rho_renormalisation": rho_ref / rho0,
        "bulk_modulus_Pa": bulk,
        "prestress_scale": scale,
        "expected_compression": scale ** gamma - 1.0,
        "h": h,
    }
    return pos, mass, cfg, info


# --------------------------------------------------------------------------- #
# Static (t = 0) field diagnostics
# --------------------------------------------------------------------------- #

def field_diagnostics(pos, mass, cfg, radius: float, sigma: float, color: str,
                      delta_scale: float):
    """Curvature, normal orientation, delta-function integral and dp_force."""
    dtype, device = pos.dtype, pos.device
    h = cfg.h_factor * cfg.dx
    cfg.surface_tension_coefficient = sigma
    cfg.surface_tension_color = color
    cfg.surface_tension_delta_scale = delta_scale
    pairs = cell_linked_pairs(pos, 2.0 * h)
    rho = density_from_neighbor_pairs(pos, mass, h, pairs)
    normal, kappa, delta_s, acc = surface_tension_field(pos, rho, mass, pairs, cfg)

    r = torch.linalg.vector_norm(pos, dim=1)
    rhat = pos / r.clamp_min(1e-30)[:, None]
    band = (r > radius - 1.5 * cfg.dx) & (r < radius + 0.5 * cfg.dx)
    target = 1.0 / radius

    kappa_band = kappa[band]
    dot = (normal[band] * rhat[band]).sum(dim=1)

    # Surface delta function: int delta_s dr over the interface. A correct CSF
    # discretisation integrates to the colour jump (1); the volume of the
    # particles carrying it is m/rho, and the interface length is 2*pi*R.
    vol = mass / rho
    ds_integral = float((delta_s * vol).sum()) / (2.0 * math.pi * radius)

    # Radial integral of the inward force density: the pressure jump the sheet
    # can sustain. Bins are area-weighted (mean over the particles in the bin).
    f_in = -(sigma * kappa * delta_s * (normal * rhat).sum(dim=1))
    order = torch.argsort(r)
    r_sorted = r[order]
    f_sorted = f_in[order]
    bins = torch.arange(0.0, float(r_sorted.max()) + cfg.dx, cfg.dx,
                        dtype=dtype, device=device)
    idx = (torch.bucketize(r_sorted, bins) - 1).clamp(0, len(bins) - 2)
    sums = torch.zeros(len(bins) - 1, dtype=dtype, device=device)
    counts = torch.zeros(len(bins) - 1, dtype=dtype, device=device)
    sums = sums.index_add(0, idx, f_sorted)
    counts = counts.index_add(0, idx, torch.ones_like(f_sorted))
    f_bar = sums / counts.clamp_min(1.0)
    # The pressure a force sheet sustains is its line integral along the normal,
    # i.e. total force / interface length. Summing f*V and dividing by 2*pi*R
    # gives exactly that; averaging per radial bin instead (as an earlier
    # version did) halves the answer, because the annulus also contains bulk
    # particles whose force is zero.
    dp_force = float(-(f_in * vol).sum()) / (2.0 * math.pi * radius)

    return {
        "kappa_mean_band": float(kappa_band.mean()),
        "kappa_target": target,
        "kappa_ratio": float(kappa_band.mean()) / target,
        "kappa_rel_error": float((kappa_band.mean() - target) / target),
        "normal_outward_dot_mean": float(dot.mean()),
        "normal_angle_deg_mean": float(
            torch.rad2deg(torch.acos(dot.clamp(-1.0, 1.0))).mean()),
        "delta_s_integral": ds_integral,
        "dp_force_Pa": dp_force,
        "dp_force_ratio": dp_force / (sigma / radius),
        "n_band_particles": int(band.sum()),
        "_profile": {
            "bins": bins.detach().cpu().numpy().tolist(),
            "f_bar": f_bar.detach().cpu().numpy().tolist(),
        },
    }


# --------------------------------------------------------------------------- #
# Time integration
# --------------------------------------------------------------------------- #

def run_case(pos, mass, cfg, sigma: float, color: str, delta_scale: float,
             steps: int):
    """Symplectic Euler, no wall, no gravity; returns the pressure history."""
    cfg.surface_tension_coefficient = sigma
    cfg.surface_tension_color = color
    cfg.surface_tension_delta_scale = delta_scale
    h = cfg.h_factor * cfg.dx
    c0 = cfg.sound_speed
    dt = min(0.25 * h / c0, capillary_dt_limit(cfg),
             0.125 * h * h / max(cfg.viscosity, 1e-12))
    cfg.dt = dt
    vel = torch.zeros_like(pos)
    r_start = float(torch.linalg.vector_norm(pos, dim=1).max())
    history = []
    for step in range(steps):
        pairs = cell_linked_pairs(pos, 2.0 * h)
        rho, pressure, acc = rhs(pos, vel, mass, cfg, pairs=pairs, wall=None)
        vel = vel + dt * acc
        pos = pos + dt * vel
        if step % 10 == 0 or step == steps - 1:
            r = torch.linalg.vector_norm(pos, dim=1)
            core_mask = r < 0.5 * r_start
            shell_mask = (r > 0.9 * r_start) & (r < r_start + 0.5 * cfg.dx)
            history.append({
                "step": step,
                "t": (step + 1) * dt,
                "p_core": float(pressure[core_mask].mean()),
                "p_shell": float(pressure[shell_mask].mean()) if bool(shell_mask.any()) else 0.0,
                "rho_core_over_rho0": float((rho[core_mask] / cfg.rho0).mean()),
                "radius": float(r.mean()),
                "vmax": float(torch.linalg.vector_norm(vel, dim=1).max()),
            })
    return dt, history


# --------------------------------------------------------------------------- #
# Driver
# --------------------------------------------------------------------------- #

def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--radius", type=float, default=0.02)
    ap.add_argument("--resolutions", type=str, default="8,12,16",
                    help="R/dx values, comma separated")
    ap.add_argument("--models", type=str, default="neighbor_count,support_sum")
    ap.add_argument("--delta-scales", type=str, default="1.0",
                    help="support_sum only; 2.0 doubles the delta function")
    ap.add_argument("--sigma", type=float, default=0.5)
    ap.add_argument("--sound-speed", type=float, default=3.0,
                    help="soft EOS so that sigma/R is a measurable part of B")
    ap.add_argument("--rho0", type=float, default=1000.0)
    ap.add_argument("--gamma", type=float, default=7.0)
    ap.add_argument("--steps", type=int, default=4000)
    ap.add_argument("--no-prestress", action="store_true")
    ap.add_argument("--device", type=str, default="cpu")
    ap.add_argument("--out", type=str, default="benchmarks/surface_tension/results")
    ap.add_argument("--plot", action="store_true")
    args = ap.parse_args()

    dtype = torch.float64
    device = torch.device(args.device)
    models = [m.strip() for m in args.models.split(",") if m.strip()]
    resolutions = [int(v) for v in args.resolutions.split(",") if v.strip()]
    delta_scales = [float(v) for v in args.delta_scales.split(",") if v.strip()]
    theory = args.sigma / args.radius

    cases = []
    for model in models:
        for scale in delta_scales:
            if model != "support_sum" and scale != delta_scales[0]:
                continue  # the delta scale only exists in the support_sum model
            for res in resolutions:
                dx = args.radius / res
                pos, mass, cfg, info = build_case(
                    args.radius, dx, args.rho0, args.sound_speed, args.gamma,
                    args.sigma, dtype, device, not args.no_prestress)
                field = field_diagnostics(pos, mass, cfg, args.radius, args.sigma,
                                          model, scale)
                profile = field.pop("_profile")
                dt, hist = run_case(pos, mass, cfg, args.sigma, model, scale,
                                    args.steps)
                tail = hist[len(hist) // 2:]
                dp_sim = (sum(h["p_core"] for h in tail) / len(tail)
                          - sum(h["p_shell"] for h in tail) / len(tail))
                case = {
                    "model": model,
                    "delta_scale": scale,
                    "R_over_dx": res,
                    "dx": dx,
                    "dt": dt,
                    "t_end": hist[-1]["t"],
                    "capillary_period_s": 2.0 * math.pi * math.sqrt(
                        cfg.rho0 * args.radius ** 3 / args.sigma),
                    "dp_theory_Pa": theory,
                    "dp_sim_Pa": dp_sim,
                    "dp_sim_ratio": dp_sim / theory,
                    "dp_sim_rel_error": (dp_sim - theory) / theory,
                    "vmax_end": hist[-1]["vmax"],
                    "radius_drift": hist[-1]["radius"]
                                     - hist[0]["radius"],
                    **info, **field,
                    "history": hist,
                    "_profile": profile,
                }
                cases.append(case)
                print(f"[{model:>14s} ds={scale:.1f}] R/dx={res:>3d} "
                      f"N={info['n_particles']:>5d} "
                      f"kappa/(1/R)={field['kappa_ratio']:+.3f} "
                      f"int(ds)={field['delta_s_integral']:.3f} "
                      f"dp_force/(s/R)={field['dp_force_ratio']:+.3f} "
                      f"dp_sim/(s/R)={dp_sim / theory:+.3f}")

    outdir = Path(args.out)
    outdir.mkdir(parents=True, exist_ok=True)
    payload = {
        "benchmark": "surface_tension_laplace_2d",
        "case": {
            "radius_m": args.radius,
            "sigma_N_per_m": args.sigma,
            "sound_speed_m_per_s": args.sound_speed,
            "rho0_nominal": args.rho0,
            "gamma": args.gamma,
            "steps": args.steps,
            "prestress": not args.no_prestress,
            "dtype": "float64",
            "device": args.device,
        },
        "theory": {
            "dp_laplace_Pa": theory,
            "kappa_target_1_per_m": 1.0 / args.radius,
            "note": "2-D droplet: one principal curvature, dp = sigma / R",
        },
        "cases": cases,
    }
    report = outdir / "laplace_report.json"
    report.write_text(json.dumps(payload, indent=2))
    print(f"\nwrote {report}")
    if args.plot:
        make_figures(payload, outdir)
    return 0


def make_figures(payload: dict, outdir: Path) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    cases = payload["cases"]
    R = payload["case"]["radius_m"]
    labels = sorted({(c["model"], c["delta_scale"]) for c in cases})
    colors = plt.cm.tab10.colors

    fig, axes = plt.subplots(1, 3, figsize=(14.0, 4.1))

    def series(model, scale, key_fn):
        pts = [c for c in cases if c["model"] == model and c["delta_scale"] == scale]
        return [c["R_over_dx"] for c in pts], [key_fn(c) for c in pts]

    for ax, key_fn, ylabel, title, ref in (
        (axes[0], lambda c: c["kappa_ratio"], "$\\langle\\kappa\\rangle/(1/R)$",
         "(a) curvature vs analytic $1/R$", 1.0),
        (axes[1], lambda c: c["dp_sim_ratio"], "$\\Delta p_{sim}/(\\sigma/R)$",
         "(b) measured Laplace jump", 1.0),
    ):
        for idx, (model, scale) in enumerate(labels):
            x, y = series(model, scale, key_fn)
            ax.plot(x, y, marker="osD"[idx % 3], color=colors[idx % 10],
                    label=f"{model}, $\\delta_s$x{scale:g}")
        ax.axhline(ref, color="k", lw=1, ls="--", label="analytic")
        ax.set_xlabel("$R/\\Delta x$")
        ax.set_ylabel(ylabel)
        ax.set_title(title)
        ax.grid(alpha=0.3)
        ax.legend(fontsize=7)

    ax = axes[2]
    for idx, (model, scale) in enumerate(labels):
        x, y = series(model, scale, lambda c: c["dp_force_ratio"])
        ax.plot(x, y, marker="osD"[idx % 3], color=colors[idx % 10],
                label=f"{model}, $\\delta_s$x{scale:g}")
    ax.axhline(1.0, color="k", lw=1, ls="--", label="analytic")
    ax.set_xlabel("$R/\\Delta x$")
    ax.set_ylabel("$\\Delta p_{force}/(\\sigma/R)$")
    ax.set_title("(c) pressure the force sheet sustains")
    ax.grid(alpha=0.3)
    ax.legend(fontsize=7)

    fig.suptitle("2-D Laplace test: $\\Delta p=\\sigma/R$, "
                 "$\\sigma=%.2f$ N/m, $R=%.3f$ m, $c_0=%.1f$ m/s"
                 % (payload["case"]["sigma_N_per_m"], R,
                    payload["case"]["sound_speed_m_per_s"]), fontsize=10)
    fig.tight_layout()
    path = outdir / "laplace.png"
    fig.savefig(path, dpi=140)
    print(f"wrote {path}")

    # second figure: radial force profile and the pressure history of one case
    fig, axes = plt.subplots(1, 2, figsize=(11.0, 4.0))
    ax = axes[0]
    for idx, (model, scale) in enumerate(labels):
        pts = [c for c in cases if c["model"] == model and c["delta_scale"] == scale]
        if not pts:
            continue
        fine = max(pts, key=lambda c: c["R_over_dx"])
        ax.plot(fine["_profile"]["bins"][:-1], fine["_profile"]["f_bar"],
                color=colors[idx % 10], label=f"{model}, x{scale:g} "
                                             f"($R/\\Delta x$={fine['R_over_dx']})")
    ax.axvline(R, color="k", lw=1, ls="--", label="$r=R$")
    ax.set_xlabel("$r$ [m]")
    ax.set_ylabel("$f_r$ [Pa/m]")
    ax.set_title("(d) radial CSF force density")
    ax.grid(alpha=0.3)
    ax.legend(fontsize=7)

    ax = axes[1]
    for idx, (model, scale) in enumerate(labels):
        pts = [c for c in cases if c["model"] == model and c["delta_scale"] == scale]
        if not pts:
            continue
        fine = max(pts, key=lambda c: c["R_over_dx"])
        t = [h["t"] for h in fine["history"]]
        dp = [(h["p_core"] - h["p_shell"]) / fine["dp_theory_Pa"] for h in fine["history"]]
        ax.plot(t, dp, color=colors[idx % 10],
                label=f"{model}, x{scale:g}")
    ax.axhline(1.0, color="k", lw=1, ls="--", label="analytic")
    ax.set_xlabel("$t$ [s]")
    ax.set_ylabel("$(p_{core}-p_{shell})/(\\sigma/R)$")
    ax.set_title("(e) pressure-jump trace, finest resolution")
    ax.grid(alpha=0.3)
    ax.legend(fontsize=7)
    fig.tight_layout()
    path = outdir / "laplace_fields.png"
    fig.savefig(path, dpi=140)
    print(f"wrote {path}")


if __name__ == "__main__":
    raise SystemExit(main())
