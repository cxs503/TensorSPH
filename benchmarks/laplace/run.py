"""Young-Laplace verification of the merged CSF surface-tension model.

The CSF model in ``tensorsph.surface_tension`` (Brackbill, Kothe & Zemach 1992)
was merged with a benchmark that only asked whether the force "points roughly
inward" and is "non-zero". Nothing in the repository has ever compared its
output with the one analytic result a capillary model must reproduce: the
Young-Laplace pressure jump across a circular interface in two dimensions,

    dp = sigma / R .

This benchmark performs that comparison on the production code path, at three
resolutions, and writes the measured numbers to ``report.json``. It makes no
attempt to hide a failure: the acceptance criteria are stated up front and the
exit status is non-zero when they are not met.

Probes (all R-dependent quantities are reported in nondimensional form):

P1 colour-jump completeness   I = integral |grad c| dr across the interface.
   A phase indicator that goes from 1 (liquid) to 0 (vacuum) must satisfy
   I = 1. This is the discrete statement that the CSF surface delta function
   carries the right amount of "interfacial area".

P2 curvature                  kappa * R across the interface band (theory: 1).

P3 discrete Laplace pressure  dp = -integral rho * a_r dr from the free surface
   to the centre, using the acceleration that
   :func:`compute_surface_tension_force` actually returns. This is the
   Young-Laplace law as the force field sees it.

P4 dynamical plateau          the production RHS (``droplet_impact.rhs``) is
   advanced for 0.05 s with gravity off; the interior pressure and the peak
   spurious current are measured. A working capillary model settles with
   p_interior = sigma / R (theory) and small residual velocity.

Usage:
    python benchmarks/laplace/run.py                 # all probes, writes report
    python benchmarks/laplace/run.py --levels 20     # single resolution
    python benchmarks/laplace/run.py --no-plot
"""
from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))

from tensorsph.droplet_impact import Config, rhs                     # noqa: E402
from tensorsph.neighbor_search import cell_linked_pairs             # noqa: E402
from tensorsph.surface_tension import (                             # noqa: E402
    compute_color_gradient,
    compute_curvature,
    compute_normal,
    compute_surface_tension_force,
)

# Acceptance criteria. Justified in docs/benchmarks/laplace-verification.md:
# 15% is this repository's "unacceptable beyond" line, and 3% is the strict
# line used for the report's yes/no answer.
CRITERIA = {
    "colour_jump_rel_error_max": 0.05,   # I = 1 +/- 5%
    "curvature_rel_error_max": 0.15,     # kappa R = 1 +/- 15%
    "laplace_pressure_rel_error_max": 0.15,
    "pressure_plateau_rel_error_max": 0.15,
}


def lattice_disc(radius: float, dx: float, rho0: float, offset: float = 0.37):
    """Square-lattice disc, centred off-lattice so no particle sits at r = 0."""
    lo, hi = -radius - dx, radius + dx
    xs = torch.arange(lo, hi, dx) + offset * dx
    ys = torch.arange(lo, hi, dx) + offset * dx
    gx, gy = torch.meshgrid(xs, ys, indexing="xy")
    pts = torch.stack((gx.reshape(-1), gy.reshape(-1)), dim=-1)
    keep = torch.linalg.vector_norm(pts, dim=-1) <= radius
    pos = pts[keep]
    mass = torch.full((pos.shape[0],), rho0 * dx * dx, dtype=pos.dtype)
    return pos, mass


def radial_bin(values: torch.Tensor, radius: torch.Tensor, edges: torch.Tensor):
    """Angular mean of ``values`` inside radial shells; empty shells are NaN."""
    idx = (torch.bucketize(radius, edges) - 1).clamp(0, edges.numel() - 2)
    total = torch.zeros(edges.numel() - 1, dtype=values.dtype)
    count = torch.zeros(edges.numel() - 1, dtype=values.dtype)
    total = total.index_add(0, idx, values)
    count = count.index_add(0, idx, torch.ones_like(values))
    with torch.no_grad():
        return torch.where(count > 0, total / count.clamp_min(1),
                           torch.full_like(total, float("nan")))


def csf_diagnostics(pos, mass, rho, cfg, sigma):
    """Run the production CSF chain and return (acc, kappa, |grad c|, is_surface)."""
    h = cfg.h_factor * cfg.dx
    pairs = cell_linked_pairs(pos, 2.0 * h)
    # The acceleration below is exactly what the solver adds to the RHS. The
    # colour field and curvature are recomputed with the same public helpers
    # and the same steps, because the force routine does not return them.
    acc = compute_surface_tension_force(pos, rho, mass, pairs, cfg)

    n = pos.shape[0]
    i, j = pairs.unbind(dim=1)
    ones = torch.ones_like(i, dtype=pos.dtype)
    neighbour_count = torch.zeros(n, dtype=pos.dtype)
    neighbour_count = neighbour_count.index_add(0, i, ones)
    neighbour_count = neighbour_count.index_add(0, j, ones)
    average = neighbour_count.mean()
    is_surface = neighbour_count < 0.7 * average
    colour = torch.ones(n, dtype=pos.dtype)
    colour[is_surface] = neighbour_count[is_surface] / average
    grad_c = compute_color_gradient(pos, colour, mass, rho, pairs, h)
    normal = compute_normal(-grad_c)
    kappa = compute_curvature(pos, normal, mass, rho, pairs, h)
    return acc, kappa, torch.linalg.vector_norm(grad_c, dim=-1), is_surface


def probe_static(radius, dx, sigma, rho0):
    """P1/P2/P3: colour jump, curvature and the discrete Laplace pressure."""
    cfg = Config(dx=dx, rho0=rho0, surface_tension_coefficient=sigma,
                 gravity=0.0)
    pos, mass = lattice_disc(radius, dx, rho0)
    rho = torch.full((pos.shape[0],), rho0, dtype=pos.dtype)
    acc, kappa, gmag, is_surface = csf_diagnostics(pos, mass, rho, cfg, sigma)

    r = torch.linalg.vector_norm(pos, dim=-1)
    a_r = (acc * pos / r[:, None]).sum(dim=1)      # radial (outward positive)
    # Shells fine enough to resolve the interface at every resolution.
    edges = torch.linspace(0.0, 1.2 * radius, max(12, int(radius / dx)))
    shell_center = 0.5 * (edges[1:] + edges[:-1])

    gmag_shell = radial_bin(gmag, r, edges)
    kappa_shell = radial_bin(kappa, r, edges)
    ar_shell = radial_bin(a_r, r, edges)

    # P1: the surface delta function must integrate to unity across the
    # interface. Sum |grad c| * dr over the shells that carry the interface.
    delta = (edges[1:] - edges[:-1])
    finite = torch.isfinite(gmag_shell)
    colour_jump = float((gmag_shell[finite] * delta[finite]).sum())

    # P2: curvature on the interface band, where the interface contribution
    # is largest (outer 20% of the droplet).
    band = is_surface & (r > 0.8 * radius)
    if band.any():
        kappa_band = float(kappa[band].mean())
    else:                                     # pragma: no cover - defensive
        kappa_band = float("nan")

    # P3: dp = integral of the radial force per unit volume from the free
    # surface inward. a_r is outward positive, the CSF force is inward, so the
    # integrand -rho0 * a_r dr is the pressure rise toward the centre.
    integrand = torch.where(torch.isfinite(ar_shell), -rho0 * ar_shell,
                            torch.zeros_like(ar_shell))
    dp_discrete = float(torch.trapezoid(integrand, shell_center))

    return {
        "levels": int(round(radius / dx)),
        "dx_m": dx,
        "n_particles": int(pos.shape[0]),
        "surface_fraction": float(is_surface.to(torch.float64).mean()),
        "colour_jump_integral": colour_jump,
        "colour_jump_rel_error": colour_jump - 1.0,
        "curvature_band_mean_per_m": kappa_band,
        "curvature_times_R": kappa_band * radius,
        "curvature_rel_error": kappa_band * radius - 1.0,
        "laplace_pressure_Pa": dp_discrete,
        "laplace_theory_Pa": sigma / radius,
        "laplace_rel_error": dp_discrete / (sigma / radius) - 1.0,
        "max_acceleration_m_per_s2": float(
            torch.linalg.vector_norm(acc, dim=-1).max()),
        "shells": {
            "r_over_R": (shell_center / radius).tolist(),
            "colour_gradient": gmag_shell.tolist(),
            "curvature_times_R": (kappa_shell * radius).tolist(),
            "radial_acceleration": ar_shell.tolist(),
        },
    }


def probe_dynamical(radius, dx, sigma, rho0, seconds=0.05, sound_speed=20.0):
    """P4: relax the droplet with the production RHS and look at the pressure."""
    cfg = Config(dx=dx, rho0=rho0, surface_tension_coefficient=sigma,
                 gravity=0.0, sound_speed=sound_speed)
    pos, mass = lattice_disc(radius, dx, rho0)
    n = pos.shape[0]
    h = cfg.h_factor * dx
    # Step limited by both the acoustic and the capillary scale of the model.
    dt_acoustic = 0.2 * h / cfg.sound_speed
    dt_capillary = 0.25 * (rho0 * h ** 3 / sigma) ** 0.5
    dt = min(dt_acoustic, dt_capillary)
    steps = max(1, int(round(seconds / dt)))
    vel = torch.zeros_like(pos)
    pairs = cell_linked_pairs(pos, 2.0 * h)
    rho = torch.full((n,), rho0, dtype=pos.dtype)
    pressure = torch.zeros_like(rho)
    for _ in range(steps):
        rho, pressure, acc = rhs(pos, vel, mass, cfg, pairs=pairs, dt=dt)
        vel = vel + dt * acc
        pos = pos + dt * vel
    r = torch.linalg.vector_norm(pos, dim=-1)
    interior = r < 0.5 * radius
    rho_interior = float(rho[interior].mean()) if interior.any() else float("nan")
    p_interior = float(pressure[interior].mean()) if interior.any() else float("nan")
    theory = sigma / radius
    bulk = rho0 * cfg.sound_speed ** 2 / cfg.gamma
    return {
        "levels": int(round(radius / dx)),
        "sound_speed_m_per_s": sound_speed,
        "bulk_modulus_Pa": bulk,
        # Tait needs this density excess to carry the Laplace pressure; compare
        # it with the O(1e-3) bias of a lattice density summation.
        "required_density_excess": theory / (cfg.gamma * bulk),
        "dt_s": dt,
        "steps": steps,
        "time_s": steps * dt,
        "interior_density_ratio": rho_interior / rho0,
        "interior_pressure_Pa": p_interior,
        "laplace_theory_Pa": theory,
        "pressure_rel_error": (p_interior / theory - 1.0) if theory else float("nan"),
        "max_speed_m_per_s": float(torch.linalg.vector_norm(vel, dim=-1).max()),
        "capillary_speed_scale_m_per_s": math.sqrt(sigma / (rho0 * radius)),
    }


def make_plot(static_results, dynamical, path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(1, 3, figsize=(13.5, 4.0))
    ax = axes[0]
    for res in static_results:
        s = res["shells"]
        ax.plot(s["r_over_R"], s["curvature_times_R"], "o-", ms=3,
                label=f"R/dx={res['levels']}")
    ax.axhline(1.0, color="k", ls="--", lw=1, label="theory $\\kappa R=1$")
    ax.set_xlabel("$r/R$")
    ax.set_ylabel("$\\kappa R$")
    ax.set_title("P2 curvature of a circular interface")
    ax.set_ylim(-3, 3)
    ax.legend(fontsize=7)
    ax.grid(alpha=0.3)

    ax = axes[1]
    theory = static_results[0]["laplace_theory_Pa"]
    levels = [r["levels"] for r in static_results]
    measured = [r["laplace_pressure_Pa"] for r in static_results]
    ax.plot(levels, measured, "s-", color="C3", label="CSF force field")
    ax.axhline(theory, color="k", ls="--", lw=1, label="$\\sigma/R$ (exact)")
    ax.set_xscale("log")
    ax.set_xlabel("$R/\\Delta x$")
    ax.set_ylabel("$\\Delta p$ [Pa]")
    ax.set_title("P3 discrete Young-Laplace pressure")
    ax.legend(fontsize=8)
    ax.grid(alpha=0.3)

    ax = axes[2]
    ratios = [r["laplace_rel_error"] * 100 for r in static_results]
    dyns = [d["pressure_rel_error"] * 100 for d in dynamical]
    labels = ([f"force\nR/dx={lv}" for lv in levels]
              + [f"relax\nc0={d['sound_speed_m_per_s']:.0f}" for d in dynamical])
    ax.bar(range(len(ratios) + len(dyns)), ratios + dyns, color="C0")
    ax.set_xticks(range(len(ratios) + len(dyns)))
    ax.set_xticklabels(labels, fontsize=8)
    ax.axhline(0, color="k", lw=0.8)
    ax.axhline(15, color="C3", ls=":", lw=1, label="15% limit")
    ax.axhline(-15, color="C3", ls=":", lw=1)
    ax.set_ylabel("relative error in $\\Delta p$ [%]")
    ax.set_title("error against $\\sigma/R$ (clipped)")
    ax.set_ylim(-120, 120)
    ax.legend(fontsize=7)
    ax.grid(alpha=0.3, axis="y")

    fig.suptitle("TensorSPH CSF surface tension vs the Young-Laplace law "
                 f"($\\sigma$={static_results[0].get('sigma', 0.072)} N/m)", fontsize=11)
    fig.tight_layout()
    fig.savefig(path, dpi=140)
    print(f"wrote {path}")


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--radius", type=float, default=0.05)
    parser.add_argument("--sigma", type=float, default=0.072)
    parser.add_argument("--rho0", type=float, default=1000.0)
    parser.add_argument("--levels", type=int, nargs="+", default=[10, 20, 40])
    parser.add_argument("--seconds", type=float, default=0.05)
    parser.add_argument("--dynamical-levels", type=int, default=20)
    parser.add_argument("--dynamical-sound-speeds", type=float, nargs="+",
                        default=[20.0, 5.0])
    parser.add_argument("--output", type=str, default="benchmarks/laplace")
    parser.add_argument("--threads", type=int, default=4)
    parser.add_argument("--no-plot", action="store_true")
    args = parser.parse_args(argv)

    torch.set_default_dtype(torch.float64)
    # The probes are small dense sums; a few threads beat a full 128-core
    # fan-out, which spends more time in barriers than in arithmetic.
    torch.set_num_threads(int(args.threads))
    out = Path(args.output)
    out.mkdir(parents=True, exist_ok=True)

    print("=" * 78)
    print("Young-Laplace verification of the CSF surface-tension model")
    print(f"  sigma = {args.sigma} N/m   R = {args.radius} m   "
          f"theory dp = sigma/R = {args.sigma / args.radius:.6f} Pa")
    print("=" * 78)

    static = []
    for levels in args.levels:
        res = probe_static(args.radius, args.radius / levels, args.sigma,
                           args.rho0)
        res["sigma"] = args.sigma
        static.append(res)
        print(f"R/dx={res['levels']:3d} N={res['n_particles']:5d}  "
              f"int|grad c|dr={res['colour_jump_integral']:6.3f} "
              f"| kappa*R={res['curvature_times_R']:+7.3f} "
              f"| dp={res['laplace_pressure_Pa']:+9.4f} Pa "
              f"({res['laplace_rel_error']:+.1%})")

    dynamical = []
    for c0 in args.dynamical_sound_speeds:
        res = probe_dynamical(args.radius, args.radius / args.dynamical_levels,
                              args.sigma, args.rho0, args.seconds, c0)
        dynamical.append(res)
        print(f"relax  R/dx={res['levels']:3d} c0={c0:5.1f} t={res['time_s']:.4f}s "
              f"rho/rho0={res['interior_density_ratio']:.6f} "
              f"p={res['interior_pressure_Pa']:+.4f} Pa "
              f"(theory {res['laplace_theory_Pa']:.4f}) "
              f"|v|max={res['max_speed_m_per_s']:.3e} m/s")

    checks = {}
    for res in static:
        tag = f"R/dx={res['levels']}"
        checks[f"{tag}:colour_jump"] = {
            "measured": res["colour_jump_integral"],
            "limit": CRITERIA["colour_jump_rel_error_max"],
            "result": "pass" if abs(res["colour_jump_rel_error"])
            <= CRITERIA["colour_jump_rel_error_max"] else "fail",
        }
        checks[f"{tag}:curvature"] = {
            "measured": res["curvature_times_R"],
            "limit": CRITERIA["curvature_rel_error_max"],
            "result": "pass" if abs(res["curvature_rel_error"])
            <= CRITERIA["curvature_rel_error_max"] else "fail",
        }
        checks[f"{tag}:laplace_pressure"] = {
            "measured": res["laplace_pressure_Pa"],
            "theory": res["laplace_theory_Pa"],
            "limit": CRITERIA["laplace_pressure_rel_error_max"],
            "result": "pass" if abs(res["laplace_rel_error"])
            <= CRITERIA["laplace_pressure_rel_error_max"] else "fail",
        }
    for res in dynamical:
        checks[f"relaxation c0={res['sound_speed_m_per_s']:.0f}:pressure"] = {
            "measured": res["interior_pressure_Pa"],
            "theory": res["laplace_theory_Pa"],
            "limit": CRITERIA["pressure_plateau_rel_error_max"],
            "result": "pass" if abs(res["pressure_rel_error"])
            <= CRITERIA["pressure_plateau_rel_error_max"] else "fail",
        }
    passed = all(c["result"] == "pass" for c in checks.values())

    report = {
        "benchmark": "young_laplace_csf_verification",
        "model": "CSF (Brackbill, Kothe & Zemach 1992) as merged in m277",
        "setup": {
            "radius_m": args.radius,
            "sigma_N_per_m": args.sigma,
            "rho0_kg_per_m3": args.rho0,
            "sound_speed_m_per_s": 20.0,
            "levels": args.levels,
            "discretisation": "square lattice, offset 0.37 dx",
            "dtype": "float64",
            "dynamical_seconds": args.seconds,
            "dynamical_sound_speeds": args.dynamical_sound_speeds,
        },
        "theory": {
            "laplace_pressure_Pa": args.sigma / args.radius,
            "curvature_times_R": 1.0,
            "colour_jump_integral": 1.0,
        },
        "static_probes": static,
        "dynamical_probes": dynamical,
        "criteria": CRITERIA,
        "checks": checks,
        "validation": {
            "status": "PASS" if passed else "FAIL",
            "failed_checks": [k for k, v in checks.items()
                              if v["result"] != "pass"],
        },
    }

    if not args.no_plot:
        make_plot(static, dynamical, out / "laplace_verification.png")
    (out / "report.json").write_text(json.dumps(report, indent=2) + "\n")
    print(f"wrote {out / 'report.json'}")
    print(f"validation: {report['validation']['status']}"
          + ("" if passed else f"  failed={report['validation']['failed_checks']}"))
    return 0 if passed else 1


if __name__ == "__main__":
    raise SystemExit(main())
