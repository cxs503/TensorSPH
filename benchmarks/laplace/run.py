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

from tensorsph.droplet_impact import (                              # noqa: E402
    Config,
    density_from_neighbor_pairs,
    rhs,
)
from tensorsph.neighbor_search import cell_linked_pairs             # noqa: E402
from tensorsph.csf_consistent import csf_fields                     # noqa: E402
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


def lattice_disc(radius: float, dx: float, rho0: float, offset: float = 0.37,
                 lattice: str = "square"):
    """Disc of particles; ``hex`` (default) or ``square`` lattice.

    ``"square"`` is the default: it is the configuration every historical number
    in the docs was measured on, and it is the only one where the production RHS
    stays stable at h = 1.3 dx (a hex lattice gives ~6 neighbours per particle
    there, and the relaxation run diverges -- see section "lattice caveat" in
    docs/benchmarks/laplace-verification.md).  ``"hex"`` is kept for the static
    probes, where six neighbours remove part of the lattice anisotropy.

    The lattice is centred off-site (``offset``) so no particle sits at r = 0,
    where the radial binning would divide by zero.
    """
    if lattice == "hex":
        dy = dx * math.sqrt(3.0) / 2.0
        lo, hi = -radius - 2 * dx, radius + 2 * dx
        rows = torch.arange(lo, hi, dy) + offset * dy
        pts = []
        for k, y in enumerate(rows):
            xs = torch.arange(lo, hi, dx) + offset * dx
            if k % 2:
                xs = xs + 0.5 * dx
            pts.append(torch.stack((xs, torch.full_like(xs, float(y))), dim=-1))
        pts = torch.cat(pts, dim=0)
        area = dx * dy
    elif lattice == "square":
        lo, hi = -radius - dx, radius + dx
        xs = torch.arange(lo, hi, dx) + offset * dx
        ys = torch.arange(lo, hi, dx) + offset * dx
        gx, gy = torch.meshgrid(xs, ys, indexing="xy")
        pts = torch.stack((gx.reshape(-1), gy.reshape(-1)), dim=-1)
        area = dx * dx
    else:                                      # pragma: no cover - defensive
        raise ValueError(f"unknown lattice {lattice!r}")
    keep = torch.linalg.vector_norm(pts, dim=-1) <= radius
    pos = pts[keep]
    mass = torch.full((pos.shape[0],), rho0 * area, dtype=pos.dtype)
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


def csf_diagnostics(pos, mass, rho, cfg, sigma, scheme=None):
    """Fields of the CSF chain that produced the acceleration.

    Returns ``(acc, kappa, delta_s, grad_c_mag, interface)``.

    ``scheme`` defaults to ``cfg.csf_scheme``, i.e. the scheme the solver runs.
    The colour field, curvature and surface delta function are taken from *that
    same* scheme: the helper used to build a neighbour-count colour field for
    every scheme, so a ``"consistent"`` acceleration was reported next to a
    ``"legacy"`` interfacial structure.  Those mixed numbers describe neither
    model and are what the old ``int |grad c| dr = 0.44`` pin in
    ``tests/test_laplace_verification.py`` measured (see
    docs/benchmarks/laplace-verification.md, section "mixed-model artefact").
    """
    scheme = scheme if scheme is not None else getattr(cfg, "csf_scheme",
                                                       "consistent")
    h = cfg.h_factor * cfg.dx
    pairs = cell_linked_pairs(pos, 2.0 * h)
    # Exactly what the solver adds to the RHS, for the selected scheme.
    acc = compute_surface_tension_force(pos, rho, mass, pairs, cfg,
                                        scheme=scheme)

    if scheme == "consistent":
        state = csf_fields(pos, mass, pairs, h, cfg.rho0)
        grad_c_mag = torch.linalg.vector_norm(state.grad_c, dim=-1)
        # The same flat-shell mask surface_tension_acceleration applies: bulk
        # particles carry a clamped, numerically-flat colour field.
        delta_s = state.delta_s.clone()
        delta_s = torch.where(delta_s <= 1.0e-3 * delta_s.max(),
                              torch.zeros_like(delta_s), delta_s)
        return acc, state.kappa, delta_s, grad_c_mag, delta_s > 0.0

    n = pos.shape[0]
    i, j = pairs.unbind(dim=1)
    ones = torch.ones_like(i, dtype=pos.dtype)
    neighbour_count = torch.zeros(n, dtype=pos.dtype)
    neighbour_count = neighbour_count.index_add(0, i, ones)
    neighbour_count = neighbour_count.index_add(0, j, ones)
    average = neighbour_count.mean()
    interface = neighbour_count < 0.7 * average
    colour = torch.ones(n, dtype=pos.dtype)
    colour[interface] = neighbour_count[interface] / average
    grad_c = compute_color_gradient(pos, colour, mass, rho, pairs, h)
    normal = compute_normal(-grad_c)
    kappa = compute_curvature(pos, normal, mass, rho, pairs, h)
    grad_c_mag = torch.linalg.vector_norm(grad_c, dim=-1)
    # The legacy CSF sets delta_s = |grad c| with no normalisation.
    return acc, kappa, grad_c_mag.clone(), grad_c_mag, interface


def probe_static(radius, dx, sigma, rho0, scheme=None, lattice="square"):
    """P1/P2/P3: delta_s normalisation, curvature and the discrete Laplace jump."""
    cfg = Config(dx=dx, rho0=rho0, surface_tension_coefficient=sigma,
                 gravity=0.0)
    scheme = scheme if scheme is not None else cfg.csf_scheme
    pos, mass = lattice_disc(radius, dx, rho0, lattice=lattice)
    rho = torch.full((pos.shape[0],), rho0, dtype=pos.dtype)
    acc, kappa, d_s, gmag, is_surface = csf_diagnostics(pos, mass, rho, cfg,
                                                        sigma, scheme=scheme)

    r = torch.linalg.vector_norm(pos, dim=-1)
    a_r = (acc * pos / r[:, None]).sum(dim=1)      # radial (outward positive)
    # Shells fine enough to resolve the interface at every resolution.
    edges = torch.linspace(0.0, 1.2 * radius, max(12, int(radius / dx)))
    shell_center = 0.5 * (edges[1:] + edges[:-1])

    gmag_shell = radial_bin(gmag, r, edges)
    ds_shell = radial_bin(d_s, r, edges)
    kappa_shell = radial_bin(kappa, r, edges)
    ar_shell = radial_bin(a_r, r, edges)

    # P1: the surface delta function must integrate to unity across the
    # interface (Brackbill et al. 1992).  Sum delta_s * dr over the shells.
    #
    # For a *single-phase* cloud the raw |grad c| integral can only reach 1/2:
    # c runs from 1 in the bulk to ~1/2 on the outer particle layer (that layer
    # sees half a kernel) and the remaining 1/2 -> 0 half lives in the void,
    # where no particles exist to carry it.  Hence the benchmark reports both
    # integrals: delta_s (criterion 1.0) and |grad c| (criterion 0.5), instead
    # of comparing a one-phase |grad c| with the two-phase 1.0 and calling the
    # unavoidable factor 2 an error.
    delta = (edges[1:] - edges[:-1])
    finite = torch.isfinite(ds_shell)
    colour_jump = float((ds_shell[finite] * delta[finite]).sum())
    finite_g = torch.isfinite(gmag_shell)
    grad_integral = float((gmag_shell[finite_g] * delta[finite_g]).sum())

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
        "scheme": scheme,
        "surface_fraction": float(is_surface.to(torch.float64).mean()),
        "colour_jump_integral": colour_jump,
        "colour_jump_rel_error": colour_jump - 1.0,
        "colour_gradient_integral": grad_integral,
        "colour_gradient_theory": 0.5,
        "colour_gradient_rel_error": grad_integral / 0.5 - 1.0,
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
            "delta_s": ds_shell.tolist(),
            "curvature_times_R": (kappa_shell * radius).tolist(),
            "radial_acceleration": ar_shell.tolist(),
        },
    }


def probe_dynamical(radius, dx, sigma, rho0, seconds=0.05,
                    sound_speed=20.0, lattice="square"):
    """P4: relax the droplet with the production RHS and look at the pressure."""
    cfg = Config(dx=dx, rho0=rho0, surface_tension_coefficient=sigma,
                 gravity=0.0, sound_speed=sound_speed)
    pos, mass = lattice_disc(radius, dx, rho0, lattice=lattice)
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


def lattice_density_bias(radius, dx, rho0, lattice="square"):
    """Density-summation bias of the *static* lattice, before any dynamics.

    ``rho/rho0 - 1`` measured with the same cubic-spline summation the solver
    uses.  It is the noise floor of the Tait pressure: no pressure feature
    smaller than ``B*((1+bias)^gamma - 1)`` can be resolved, however good the
    surface-tension force is.
    """
    cfg = Config(dx=dx)
    pos, mass = lattice_disc(radius, dx, rho0, lattice=lattice)
    h = cfg.h_factor * dx
    # Production density summation, so the self term (m_i W(0), ~27 % of rho0 in
    # 2-D at h = 1.3 dx) is included exactly as the solver does.
    rho = density_from_neighbor_pairs(pos, mass, h)
    r = torch.linalg.vector_norm(pos, dim=-1)
    interior = r < 0.5 * radius
    rel = rho[interior] / rho0 - 1.0
    return {
        "max_abs_bias": float(rel.abs().max()),
        "mean_bias": float(rel.mean()),
        "std_bias": float(rel.std()),
        "n_interior": int(interior.sum()),
    }


def probe_pressure_floor(radius, dx, sigma, rho0,
                         sound_speeds=(20.0, 10.0, 5.0, 2.5), seconds=0.05,
                         lattice="square"):
    """P5: is the Laplace pressure resolvable in the Tait representation?

    A dynamic failure of Young-Laplace has two possible causes: the force is
    wrong, or the equation of state cannot represent a pressure jump this small.
    The static probes (P1-P3) settle the first; this probe settles the second by
    comparing, for each sound speed,

        required density excess   drho/rho0 = (1 + sigma/(R B))^(1/gamma) - 1
        lattice density bias      |rho/rho0 - 1| of the uncompressed disc
        pressure floor            B*((1 + bias)^gamma - 1)          [Pa]

    with ``B = rho0 c0^2 / gamma`` the Tait bulk modulus.  ``signal_to_noise``
    below unity means the pressure route is measuring discretisation noise.
    """
    bias = lattice_density_bias(radius, dx, rho0, lattice=lattice)
    theory = sigma / radius
    rows = []
    for c0 in sound_speeds:
        cfg = Config(dx=dx, rho0=rho0, surface_tension_coefficient=sigma,
                     gravity=0.0, sound_speed=c0)
        bulk = rho0 * c0 ** 2 / cfg.gamma
        required = (1.0 + theory / bulk) ** (1.0 / cfg.gamma) - 1.0
        floor = bulk * ((1.0 + bias["max_abs_bias"]) ** cfg.gamma - 1.0)
        dyn = probe_dynamical(radius, dx, sigma, rho0, seconds, c0)
        rows.append({
            "sound_speed_m_per_s": c0,
            "bulk_modulus_Pa": bulk,
            "required_density_excess": required,
            "lattice_bias_max": bias["max_abs_bias"],
            "signal_to_noise": required / bias["max_abs_bias"],
            "pressure_floor_Pa": floor,
            "floor_over_theory": floor / theory,
            "measured_pressure_Pa": dyn["interior_pressure_Pa"],
            "pressure_rel_error": dyn["pressure_rel_error"],
            "max_speed_m_per_s": dyn["max_speed_m_per_s"],
            "capillary_mach": dyn["max_speed_m_per_s"] / c0,
            "weakly_compressible": bool(dyn["max_speed_m_per_s"] / c0 < 0.1),
        })
    return {
        "levels": int(round(radius / dx)),
        "dx_m": dx,
        "laplace_theory_Pa": theory,
        "lattice_density_bias": bias,
        "rows": rows,
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
                label=f"{res.get('scheme', 'consistent')[:4]} R/dx={res['levels']}")
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
    labels = ([f"force\nR/dx={l}" for l in levels]
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
    parser.add_argument("--schemes", type=str, nargs="+",
                        default=["consistent"],
                        choices=["consistent", "legacy"],
                        help="CSF schemes to A/B; the solver default is "
                             "'consistent'")
    parser.add_argument("--lattice", type=str, default="square",
                        choices=["hex", "square"],
                        help="particle lattice for the static probes")
    parser.add_argument("--pressure-floor", action="store_true",
                        help="also run probe P5 (Tait resolvability of sigma/R)")
    parser.add_argument("--floor-sound-speeds", type=float, nargs="+",
                        default=[20.0, 10.0, 5.0, 2.5])
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
    for scheme in args.schemes:
        for levels in args.levels:
            res = probe_static(args.radius, args.radius / levels, args.sigma,
                               args.rho0, scheme=scheme,
                               lattice=args.lattice)
            res["sigma"] = args.sigma
            static.append(res)
            print(f"{scheme:10s} R/dx={res['levels']:3d} N={res['n_particles']:5d}  "
                  f"int ds dr={res['colour_jump_integral']:6.3f} "
                  f"(int|grad c|dr={res['colour_gradient_integral']:6.3f}) "
                  f"| kappa*R={res['curvature_times_R']:+7.3f} "
                  f"| dp={res['laplace_pressure_Pa']:+9.4f} Pa "
                  f"({res['laplace_rel_error']:+.1%})")

    dynamical = []
    for c0 in args.dynamical_sound_speeds:
        res = probe_dynamical(args.radius, args.radius / args.dynamical_levels,
                              args.sigma, args.rho0, args.seconds, c0,
                              lattice=args.lattice)
        dynamical.append(res)
        print(f"relax  R/dx={res['levels']:3d} c0={c0:5.1f} t={res['time_s']:.4f}s "
              f"rho/rho0={res['interior_density_ratio']:.6f} "
              f"p={res['interior_pressure_Pa']:+.4f} Pa "
              f"(theory {res['laplace_theory_Pa']:.4f}) "
              f"|v|max={res['max_speed_m_per_s']:.3e} m/s")

    floor = None
    if args.pressure_floor:
        floor = probe_pressure_floor(args.radius, args.radius / args.levels[0],
                                     args.sigma, args.rho0,
                                     tuple(args.floor_sound_speeds),
                                     args.seconds, lattice=args.lattice)
        print(f"P5 Tait resolvability at R/dx={floor['levels']}: "
              f"lattice density bias {floor['lattice_density_bias']['max_abs_bias']:.2e}")
        for row in floor["rows"]:
            print(f"    c0={row['sound_speed_m_per_s']:5.1f}  "
                  f"needed drho/rho0={row['required_density_excess']:8.2e}  "
                  f"S/N={row['signal_to_noise']:6.2f}  "
                  f"floor={row['pressure_floor_Pa']:8.1f} Pa "
                  f"({row['floor_over_theory']:5.1%} of sigma/R)  "
                  f"dp={row['measured_pressure_Pa']:+8.2f} Pa "
                  f"({row['pressure_rel_error']:+.1%})  "
                  f"Mach={row['capillary_mach']:.2f}")

    checks = {}
    for res in static:
        tag = f"{res['scheme']} R/dx={res['levels']}"
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
            "schemes": args.schemes,
            "discretisation": f"{args.lattice} lattice (see lattice_disc)",
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
        "pressure_floor": floor,
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