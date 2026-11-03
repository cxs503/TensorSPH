"""B3 surface-tension verification: Laplace law on a static 2-D droplet.

Theory
------
For a 2-D circular droplet (an infinite cylinder in 3-D) held by surface
tension ``sigma`` and radius ``R`` the Young-Laplace balance gives a pressure
jump

    dp = sigma / R .

The solver implements the Continuum Surface Force (CSF) model of Brackbill,
Kothe & Zemach (1992):  f = -sigma * kappa * n * |grad c| , applied to the
particles the model classifies as surface particles.  In the continuum the
surface delta function satisfies  integral |grad c| dn = 1  across the
interface, so the pressure jump produced by the discrete force field is

    dp_model = integral f_r dr = sigma * <kappa>_delta * N_delta ,
    N_delta  = integral |grad c| dr .

Both factors are measured here.  The model passes only if dp_model/dp_theory
is within the platform tolerance (+/-15 %) at every resolution tested and
converges under refinement.

Method
------
Static, no time integration: the droplet lattice is built, summation density
and the production surface-tension acceleration are evaluated (src code path),
and the radial force profile is integrated.  A dynamic pressure-jump
measurement is *not* used: the weakly-compressible Tait EOS resolves a
Laplace pressure only as  drho = dp / c0^2 , so the density signal is
1.8e-6 relative for water at c0 = 20 m/s, far below SPH density noise.  See
``laplace_resolvability`` below, which records the numbers.

Two schemes are analysed:
- **legacy**: neighbour-count colour function + uncorrected SPH operators
  (the original m277 implementation, known to fail the Laplace law).
- **consistent**: kernel-sum colour + MLS-consistent gradient/divergence
  (the m2188 improvement; force direction and cut-force identity pass,
  curvature still ~25 % high).

Usage
-----
    PYTHONPATH=src python benchmarks/laplace_droplet/run.py [--plot]
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import torch

from tensorsph.droplet_impact import Config, density_from_neighbor_pairs
from tensorsph.neighbor_search import cell_linked_pairs
from tensorsph.surface_tension import (compute_color_gradient, compute_curvature,
                                       compute_normal, compute_surface_tension_force)

RHO0 = 1000.0
CENTER = 0.5
TOL = 0.15


# --------------------------------------------------------------------------- #
# Geometry
# --------------------------------------------------------------------------- #

def droplet_lattice(radius: float, dx: float, lattice: str = "hex",
                    center: tuple[float, float] = (CENTER, CENTER),
                    rho0: float = RHO0):
    """Particles inside a disc, with the correct 2-D number density per lattice."""
    if lattice == "hex":
        dy = dx * math.sqrt(3.0) / 2.0
        rows = torch.arange(center[1] - radius - 2 * dx, center[1] + radius + 2 * dx,
                            dy, dtype=torch.float64)
        pts = []
        for k, y in enumerate(rows):
            xs = torch.arange(center[0] - radius - 2 * dx, center[0] + radius + 2 * dx,
                              dx, dtype=torch.float64)
            xs = xs + (0.5 * dx if k % 2 else 0.0)
            pts.extend((float(x), float(y)) for x in xs)
        pos = torch.tensor(pts, dtype=torch.float64)
        area_per_particle = dx * dy
    elif lattice == "square":
        xs = torch.arange(center[0] - radius - 2 * dx, center[0] + radius + 2 * dx,
                          dx, dtype=torch.float64)
        ys = torch.arange(center[1] - radius - 2 * dx, center[1] + radius + 2 * dx,
                          dx, dtype=torch.float64)
        xx, yy = torch.meshgrid(xs, ys, indexing="ij")
        pos = torch.stack([xx.flatten(), yy.flatten()], dim=1)
        area_per_particle = dx * dx
    else:
        raise ValueError(f"unknown lattice {lattice!r}")
    keep = (pos[:, 0] - center[0]).square() + (pos[:, 1] - center[1]).square() <= radius ** 2
    pos = pos[keep].clone()
    mass = torch.full((pos.shape[0],), rho0 * area_per_particle, dtype=torch.float64)
    return pos, mass, area_per_particle


def jitter_lattice(pos: torch.Tensor, dx: float, amplitude: float, seed: int = 0) -> torch.Tensor:
    g = torch.Generator().manual_seed(seed)
    noise = torch.rand(pos.shape, generator=g, dtype=torch.float64) - 0.5
    return pos + amplitude * dx * noise


# --------------------------------------------------------------------------- #
# Radial bookkeeping
# --------------------------------------------------------------------------- #

def radial_mean(values: torch.Tensor, radius: torch.Tensor, r_max: float,
                dr: float) -> tuple[torch.Tensor, torch.Tensor, float]:
    """Particle-count mean of ``values`` in radial shells of width ``dr``."""
    nbins = int(math.ceil(r_max / dr))
    idx = torch.clamp((radius / dr).floor().long(), 0, nbins - 1)
    total = torch.zeros(nbins, dtype=torch.float64).index_add(0, idx, values)
    count = torch.zeros(nbins, dtype=torch.float64).index_add(
        0, idx, torch.ones_like(values))
    mean = torch.where(count > 0, total / count.clamp_min(1.0), torch.zeros_like(total))
    return mean, count, nbins * dr


def _integrate_radial(f_r: torch.Tensor, r: torch.Tensor, r_max: float,
                      dr: float) -> float:
    """Integrate a radial profile by summing shell means."""
    f_mean, _, _ = radial_mean(f_r, r, r_max, dr)
    return -float(f_mean.sum() * dr)


def _interface_stats(kappa: torch.Tensor, delta_s: torch.Tensor,
                     r: torch.Tensor, radius: float, h: float) -> dict:
    """Curvature and delta_s statistics in the interface band."""
    band = (r - radius).abs() < 1.5 * h
    k_band, ds_band = kappa[band], delta_s[band]
    weight = ds_band.clamp_min(0.0)
    k_delta = float((k_band * weight).sum() / weight.sum().clamp_min(1e-30))
    return {
        "kappa_median": float(k_band.median()),
        "kappa_median_times_R": float(k_band.median() * radius),
        "kappa_mean": float(k_band.mean()),
        "kappa_std": float(k_band.std()),
        "kappa_positive_fraction": float((k_band > 0).double().mean()),
        "kappa_delta_weighted": k_delta,
        "kappa_delta_weighted_times_R": k_delta * radius,
        "N_delta": float(ds_band.sum() * 1.5 * h * 2 / max(band.sum(), 1)),
        "n_interface_band": int(band.sum()),
    }


# --------------------------------------------------------------------------- #
# One case
# --------------------------------------------------------------------------- #

def run_case(radius: float = 0.1, dx: float = 0.005, sigma: float = 0.072,
             lattice: str = "hex", jitter: float = 0.0, seed: int = 0,
             surface_factor: float = 0.7,
             scheme: str = "consistent") -> dict:
    """Run a single Laplace-droplet case and return diagnostics.

    Parameters
    ----------
    scheme : str
        ``"consistent"`` (default) uses the kernel-sum colour + MLS operators
        from ``csf_consistent``.  ``"legacy"`` uses the neighbour-count colour
        function and uncorrected SPH operators (the original m277 model).
    """
    cfg = Config(dx=dx, surface_tension_coefficient=sigma, gravity=0.0, rho0=RHO0)
    h = cfg.h_factor * cfg.dx
    pos, mass, area = droplet_lattice(radius, dx, lattice)
    if jitter > 0.0:
        pos = jitter_lattice(pos, dx, jitter, seed)

    pairs = cell_linked_pairs(pos, 2.0 * h)
    rho = density_from_neighbor_pairs(pos, mass, h, pairs)

    rel = pos - torch.tensor([CENTER, CENTER], dtype=torch.float64)
    r = rel.norm(dim=1)
    rhat = rel / r.clamp_min(1e-12)[:, None]

    dr = 0.25 * h

    # --- Production force (always from the chosen scheme) ----------------
    acc = compute_surface_tension_force(pos, rho, mass, pairs, cfg, scheme=scheme)
    f_r = (acc * rhat).sum(dim=1) * rho                      # N/m^3, negative = inward
    dp_model = _integrate_radial(f_r, r, 1.5 * radius, dr)

    # --- Legacy field analysis (for backward compatibility) --------------
    count = torch.zeros(pos.shape[0], dtype=torch.float64)
    i, j = pairs.unbind(dim=1)
    count = count.index_add(0, i, torch.ones_like(i, dtype=torch.float64))
    count = count.index_add(0, j, torch.ones_like(j, dtype=torch.float64))
    is_surface = count < surface_factor * count.mean()
    color_legacy = torch.ones(pos.shape[0], dtype=torch.float64)
    color_legacy[is_surface] = count[is_surface] / count.mean()

    grad_c_leg = compute_color_gradient(pos, color_legacy, mass, rho, pairs, h)
    delta_s_leg = torch.linalg.vector_norm(grad_c_leg, dim=1)
    normal_leg = compute_normal(-grad_c_leg)
    kappa_leg = compute_curvature(pos, normal_leg, mass, rho, pairs, h)

    face_leg = (normal_leg * rhat).sum(dim=1)
    band = (r - radius).abs() < 1.5 * h
    stats_leg = _interface_stats(kappa_leg, delta_s_leg, r, radius, h)
    stats_leg["normal_alignment_delta_weighted"] = float(
        (face_leg[band] * delta_s_leg[band]).sum() / delta_s_leg[band].sum().clamp_min(1e-30))
    stats_leg["normal_coherent_fraction"] = float((face_leg[band] > 0.5).double().mean())
    stats_leg["n_surface"] = int(is_surface.sum())

    # N_delta for legacy
    ds_mean_leg, _, _ = radial_mean(delta_s_leg, r, 1.5 * radius, dr)
    stats_leg["N_delta"] = float(ds_mean_leg.sum() * dr)

    # --- Consistent field analysis (when scheme == "consistent") ---------
    if scheme == "consistent":
        from tensorsph.csf_consistent import csf_fields
        vol = mass / RHO0
        state = csf_fields(pos, mass, pairs, h, RHO0)
        face_con = (state.normal * rhat).sum(dim=1)
        stats_con = _interface_stats(state.kappa, state.delta_s, r, radius, h)
        stats_con["normal_alignment_delta_weighted"] = float(
            (face_con[band] * state.delta_s[band]).sum() / state.delta_s[band].sum().clamp_min(1e-30))
        stats_con["normal_coherent_fraction"] = float((face_con[band] > 0.5).double().mean())
        stats_con["color_min"] = float(state.color.min())
        stats_con["color_max"] = float(state.color.max())
        # N_delta for consistent: integrate |grad c| across the interface
        ds_mean_con, _, _ = radial_mean(state.delta_s, r, 1.5 * radius, dr)
        stats_con["N_delta"] = float(ds_mean_con.sum() * dr)
    else:
        stats_con = None

    # --- Error decomposition (using the production force's scheme) -------
    if scheme == "consistent":
        # Use consistent fields for instruments
        from tensorsph.csf_consistent import csf_fields
        state = csf_fields(pos, mass, pairs, h, RHO0)
        delta_s_use = state.delta_s
        normal_use = state.normal
        face_use = (normal_use * rhat).sum(dim=1)
    else:
        delta_s_use = delta_s_leg
        normal_use = normal_leg
        face_use = face_leg

    f_r_exact = -sigma * (1.0 / radius) * delta_s_use * face_use
    dp_exact_kappa = _integrate_radial(f_r_exact, r, 1.5 * radius, dr)
    f_r_exact_n = -sigma * (1.0 / radius) * delta_s_use
    dp_exact_kappa_exact_normal = _integrate_radial(f_r_exact_n, r, 1.5 * radius, dr)
    n_delta_use = float(radial_mean(delta_s_use, r, 1.5 * radius, dr)[0].sum() * dr)
    f_r_norm = f_r_exact_n / max(n_delta_use, 1e-30)
    dp_exact_kappa_norm = _integrate_radial(f_r_norm, r, 1.5 * radius, dr)

    dp_theory = sigma / radius
    result = {
        "radius_m": radius, "dx_m": dx, "resolution_R_over_dx": round(radius / dx, 2),
        "lattice": lattice, "jitter": jitter, "sigma_N_per_m": sigma,
        "scheme": scheme,
        "n_particles": int(pos.shape[0]), "n_pairs": int(pairs.shape[0]),
        "mass_error_vs_rho0_area": float(
            mass.sum() / (RHO0 * math.pi * radius ** 2) - 1.0),
        "density_min_ratio": float((rho / RHO0).min()),
        "density_max_ratio": float((rho / RHO0).max()),
        "dp_theory_Pa": dp_theory,
        "dp_model_Pa": dp_model,
        "dp_ratio": dp_model / dp_theory,
        "relative_error": abs(dp_model / dp_theory - 1.0),
        "within_15pct": abs(dp_model / dp_theory - 1.0) <= TOL,
        "instrument_A_exact_kappa_model_normal_ratio": dp_exact_kappa / dp_theory,
        "instrument_B_exact_kappa_exact_normal_ratio": dp_exact_kappa_exact_normal / dp_theory,
        "instrument_C_exact_operators_ratio": dp_exact_kappa_norm / dp_theory,
        # Legacy field analysis (always included for comparison)
        "legacy_N_delta": stats_leg["N_delta"],
        "legacy_kappa_median": stats_leg["kappa_median"],
        "legacy_kappa_median_times_R": stats_leg["kappa_median_times_R"],
        "legacy_kappa_delta_weighted_times_R": stats_leg["kappa_delta_weighted_times_R"],
        "legacy_normal_alignment": stats_leg["normal_alignment_delta_weighted"],
        "legacy_n_surface": stats_leg["n_surface"],
    }
    if scheme == "consistent":
        result.update({
            "consistent_N_delta": stats_con["N_delta"],
            "consistent_kappa_median": stats_con["kappa_median"],
            "consistent_kappa_median_times_R": stats_con["kappa_median_times_R"],
            "consistent_kappa_delta_weighted_times_R": stats_con["kappa_delta_weighted_times_R"],
            "consistent_normal_alignment": stats_con["normal_alignment_delta_weighted"],
            "consistent_color_min": stats_con["color_min"],
            "consistent_color_max": stats_con["color_max"],
        })
    return result


def laplace_resolvability(rho0: float = RHO0, radius: float = 0.1,
                          sigma: float = 0.072, c0: float = 20.0,
                          gamma: float = 7.0) -> dict:
    """Why the pressure jump cannot be read off a WCI-SPH density field.

    Tait EOS: p = B[(rho/rho0)^gamma - 1], B = rho0 c0^2 / gamma, so
    dp/drho = c0^2 at rho0.  The density signal of a Laplace jump is
    therefore dp / c0^2 and has to be compared with the summation-density
    noise of a lattice, measured here and in docs/benchmarks/hydrostatic.md
    at the ~1e-3 level.
    """
    dp = sigma / radius
    bulk = rho0 * c0 ** 2 / gamma
    return {
        "dp_theory_Pa": dp, "bulk_modulus_Pa": bulk, "c0_m_per_s": c0,
        "dp_over_bulk": dp / bulk,
        "density_ratio_signal": ((1.0 + dp / bulk) ** (1.0 / gamma) - 1.0),
        "density_noise_reference": 1.0e-3,
        "signal_to_noise": ((1.0 + dp / bulk) ** (1.0 / gamma) - 1.0) / 1.0e-3,
        "verdict": "not resolvable" if ((1.0 + dp / bulk) ** (1.0 / gamma) - 1.0) < 3e-3
                   else "resolvable",
    }


# --------------------------------------------------------------------------- #
# Sweep
# --------------------------------------------------------------------------- #

def run_benchmark(plot: bool = True, output_dir: str | Path = "benchmarks/laplace_droplet") -> dict:
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)
    cases = []
    for lattice in ("hex", "square"):
        for dx in (0.01, 0.005, 0.0025):
            cases.append(run_case(radius=0.1, dx=dx, sigma=0.072, lattice=lattice))
    # disorder sensitivity (SPH lattices are artificial equilibria)
    cases.append(run_case(radius=0.1, dx=0.005, sigma=0.072, lattice="hex", jitter=0.2, seed=1))
    cases.append(run_case(radius=0.1, dx=0.005, sigma=0.072, lattice="hex", jitter=0.0, seed=0, scheme="legacy"))

    results = {"cases": cases, "resolvability": laplace_resolvability()}
    (out / "results.json").write_text(json.dumps(results, indent=2))

    if plot:
        _make_plot(cases, out)

    return results


def _make_plot(cases: list[dict], out: Path):
    """Quick comparison plot: dp_ratio vs resolution for both schemes."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(8, 5))
    for scheme_label, key in [("consistent", "dp_ratio"), ("legacy", "dp_ratio")]:
        xs, ys = [], []
        for c in cases:
            if c.get("scheme", "consistent") == scheme_label and c["lattice"] == "hex" and c["jitter"] == 0:
                xs.append(c["resolution_R_over_dx"])
                ys.append(c[key])
        if xs:
            ax.plot(xs, ys, "o-", label=scheme_label)
    ax.axhline(1.0, color="k", ls="--", alpha=0.5, label="theory")
    ax.axhline(0.85, color="r", ls=":", alpha=0.5, label="15% band")
    ax.axhline(1.15, color="r", ls=":", alpha=0.5)
    ax.set_xlabel("R / dx")
    ax.set_ylabel("dp_model / dp_theory")
    ax.set_title("Laplace-law verification: consistent vs legacy CSF")
    ax.legend()
    ax.set_xscale("log")
    fig.savefig(out / "laplace_droplet.png", dpi=150)
    plt.close(fig)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--plot", action="store_true")
    parser.add_argument("--output-dir", default="benchmarks/laplace_droplet")
    args = parser.parse_args()
    results = run_benchmark(plot=args.plot, output_dir=args.output_dir)
    print(json.dumps(results, indent=2)[:2000])
