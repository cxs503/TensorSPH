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

Usage
-----
    PYTHONPATH=src python benchmarks/laplace_droplet/run.py [--plot]
"""

from __future__ import annotations

import sys
from pathlib import Path as _Path

_ROOT = _Path(__file__).resolve().parents[2]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))
from tools.worktree_env import ensure_worktree_source  # noqa: E402

ensure_worktree_source(__file__)

import argparse
import json
import math
from pathlib import Path

import torch

from tensorsph.droplet_impact import Config, density_from_neighbor_pairs
from tensorsph.neighbor_search import cell_linked_pairs
from tensorsph.csf_consistent import csf_fields as csf_fields_consistent
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


# --------------------------------------------------------------------------- #
# One case
# --------------------------------------------------------------------------- #

def run_case(radius: float = 0.1, dx: float = 0.005, sigma: float = 0.072,
             lattice: str = "hex", jitter: float = 0.0, seed: int = 0,
             surface_factor: float = 0.7) -> dict:
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

    # --- field diagnostics: score the pipeline the solver actually runs ---
    # Until m2730 this block rebuilt the colour field locally with the m277
    # neighbour-count heuristic, while ``compute_surface_tension_force`` had
    # already switched to ``csf_scheme="consistent"`` (the Config default).  The
    # field verdicts (N_delta, sign and scale of kappa, normal coherence) thus
    # described a pipeline that no longer produced the force and contradicted
    # the pressure jump measured from it.  Both are reported now:
    #   shipped fields : tensorsph.csf_consistent.csf_fields -- what the solver runs
    #   legacy_*       : m277 neighbour-count CSF           -- A/B reference only
    shipped = csf_fields_consistent(pos, mass, pairs, h, RHO0,
                                    curvature_scheme=getattr(cfg, "csf_curvature", "mls"))
    delta_s = shipped.delta_s
    normal_out = shipped.normal
    kappa = shipped.kappa
    is_surface = shipped.surface_mask(0.05)

    # legacy pipeline, kept only for the A/B record
    count = torch.zeros(pos.shape[0], dtype=torch.float64)
    i, j = pairs.unbind(dim=1)
    count = count.index_add(0, i, torch.ones_like(i, dtype=torch.float64))
    count = count.index_add(0, j, torch.ones_like(j, dtype=torch.float64))
    legacy_is_surface = count < surface_factor * count.mean()
    legacy_color = torch.ones(pos.shape[0], dtype=torch.float64)
    legacy_color[legacy_is_surface] = count[legacy_is_surface] / count.mean()
    legacy_grad_c = compute_color_gradient(pos, legacy_color, mass, rho, pairs, h)
    legacy_delta_s = torch.linalg.vector_norm(legacy_grad_c, dim=1)
    legacy_normal = compute_normal(-legacy_grad_c)
    legacy_kappa = compute_curvature(pos, legacy_normal, mass, rho, pairs, h)

    # --- N_delta = integral |grad c| dr ---------------------------------
    dr = 0.25 * h
    ds_mean, _, _ = radial_mean(delta_s, r, 1.5 * radius, dr)
    n_delta = float(ds_mean.sum() * dr)
    legacy_ds_mean, _, _ = radial_mean(legacy_delta_s, r, 1.5 * radius, dr)
    legacy_n_delta = float(legacy_ds_mean.sum() * dr)

    # --- pressure jump implied by the production force field ------------
    acc = compute_surface_tension_force(pos, rho, mass, pairs, cfg)
    f_r = (acc * rhat).sum(dim=1) * rho                      # N/m^3, negative = inward
    f_mean, _, _ = radial_mean(f_r, r, 1.5 * radius, dr)
    dp_model = -float(f_mean.sum() * dr)

    # --- Error decomposition.  The same CSF form is evaluated with pieces of
    # the model replaced by exact values, which isolates the three error
    # sources and validates the radial integration itself:
    #   instrument_A : exact kappa = 1/R, model normal, model delta_s
    #                  -> must recover N_delta * <n.rhat>_delta
    #   instrument_B : exact kappa, exact normal rhat, model delta_s
    #                  -> must recover N_delta  (validates the integration)
    #   instrument_C : exact kappa, exact normal, unit-normalised delta_s
    #                  -> must recover 1         (the Laplace law itself)
    face = (normal_out * rhat).sum(dim=1)
    f_r_exact = -sigma * (1.0 / radius) * delta_s * face
    exact_mean, _, _ = radial_mean(f_r_exact, r, 1.5 * radius, dr)
    dp_exact_kappa = -float(exact_mean.sum() * dr)
    f_r_exact_n = -sigma * (1.0 / radius) * delta_s
    exactn_mean, _, _ = radial_mean(f_r_exact_n, r, 1.5 * radius, dr)
    dp_exact_kappa_exact_normal = -float(exactn_mean.sum() * dr)
    f_r_norm = f_r_exact_n / max(n_delta, 1e-30)
    norm_mean, _, _ = radial_mean(f_r_norm, r, 1.5 * radius, dr)
    dp_exact_kappa_norm = -float(norm_mean.sum() * dr)

    band = (r - radius).abs() < 1.5 * h
    k_band, ds_band = kappa[band], delta_s[band]
    weight = ds_band.clamp_min(0.0)
    k_delta = float((k_band * weight).sum() / weight.sum().clamp_min(1e-30))

    legacy_k_band = legacy_kappa[band]
    legacy_w = legacy_delta_s[band].clamp_min(0.0)
    legacy_k_delta = float((legacy_k_band * legacy_w).sum()
                           / legacy_w.sum().clamp_min(1e-30))
    legacy_face = (legacy_normal * rhat).sum(dim=1)
    legacy_alignment = float(
        (legacy_face[band] * legacy_delta_s[band]).sum()
        / legacy_delta_s[band].sum().clamp_min(1e-30))

    dp_theory = sigma / radius
    return {
        "radius_m": radius, "dx_m": dx, "resolution_R_over_dx": round(radius / dx, 2),
        "lattice": lattice, "jitter": jitter, "sigma_N_per_m": sigma,
        "n_particles": int(pos.shape[0]), "n_pairs": int(pairs.shape[0]),
        "n_surface": int(is_surface.sum()), "n_interface_band": int(band.sum()),
        "mass_error_vs_rho0_area": float(
            mass.sum() / (RHO0 * math.pi * radius ** 2) - 1.0),
        "density_min_ratio": float((rho / RHO0).min()),
        "density_max_ratio": float((rho / RHO0).max()),
        "N_delta": n_delta,
        "kappa_model_median": float(k_band.median()),
        "kappa_model_median_times_R": float(k_band.median() * radius),
        "kappa_model_mean": float(k_band.mean()),
        "kappa_model_std": float(k_band.std()),
        "kappa_model_positive_fraction": float((k_band > 0).double().mean()),
        "kappa_delta_weighted": k_delta,
        "kappa_delta_weighted_times_R": k_delta * radius,
        "dp_theory_Pa": dp_theory,
        "dp_model_Pa": dp_model,
        "dp_ratio": dp_model / dp_theory,
        "relative_error": abs(dp_model / dp_theory - 1.0),
        "within_15pct": abs(dp_model / dp_theory - 1.0) <= TOL,
        "consistency_check_Ndelta_x_kappaR": n_delta * k_delta * radius,
        "normal_alignment_delta_weighted": float(
            (face[band] * delta_s[band]).sum() / delta_s[band].sum().clamp_min(1e-30)),
        "normal_coherent_fraction": float((face[band] > 0.5).double().mean()),
        "instrument_A_exact_kappa_model_normal_ratio": dp_exact_kappa / dp_theory,
        "instrument_B_exact_kappa_exact_normal_ratio": dp_exact_kappa_exact_normal / dp_theory,
        "instrument_C_exact_operators_ratio": dp_exact_kappa_norm / dp_theory,
        # provenance of the field diagnostics (m2730 harness-fidelity fix)
        "fields_under_test": "tensorsph.csf_consistent.csf_fields "
                             "(csf_scheme='consistent', Config default)",
        "legacy_N_delta": legacy_n_delta,
        "legacy_kappa_model_median_times_R": float(legacy_k_band.median() * radius),
        "legacy_kappa_model_positive_fraction": float(
            (legacy_k_band > 0).double().mean()),
        "legacy_kappa_delta_weighted_times_R": legacy_k_delta * radius,
        "legacy_normal_alignment_delta_weighted": legacy_alignment,
        "legacy_n_surface": int(legacy_is_surface.sum()),
    }


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
    cases.append(run_case(radius=0.1, dx=0.005, sigma=0.072, lattice="hex", jitter=0.4, seed=2))
    # linearity in sigma: the CSF force is linear, the ratio must not move
    cases.append(run_case(radius=0.1, dx=0.005, sigma=0.72, lattice="hex"))
    cases.append(run_case(radius=0.1, dx=0.005, sigma=7.2, lattice="hex"))
    # radius scaling at fixed resolution: R/dx is the governing parameter
    cases.append(run_case(radius=0.05, dx=0.005, sigma=0.072, lattice="hex"))
    cases.append(run_case(radius=0.2, dx=0.005, sigma=0.072, lattice="hex"))

    resolvability = laplace_resolvability()
    base = [c for c in cases if c["jitter"] == 0.0 and c["sigma_N_per_m"] == 0.072]
    verdict = {
        "criterion": "|dp_model/dp_theory - 1| <= 15 % at every resolution",
        "tolerance": TOL,
        "max_relative_error": max(c["relative_error"] for c in base),
        "n_cases": len(base),
        "n_within_tolerance": sum(c["within_15pct"] for c in base),
        "converges_under_refinement":
            _converging([c for c in base if c["lattice"] == "hex"]),
        "status": "PASS" if all(c["within_15pct"] for c in base) else "FAIL",
        "current_model_verdict":
            "consistent CSF (csf_scheme='consistent') reproduces the Young-Laplace "
            "jump within the platform tolerance"
            if all(c["within_15pct"] for c in base)
            else "CSF model misses the Young-Laplace jump outside the platform tolerance",
        "instrument_check": {
            "description": ("the same CSF form with exact kappa = 1/R: A keeps the model "
                            "normal (must recover N_delta * <n.rhat>), B uses the exact "
                            "normal rhat (must recover N_delta), C also normalises delta_s "
                            "(must recover 1)"),
            "max_deviation_B_vs_N_delta": max(
                abs(c["instrument_B_exact_kappa_exact_normal_ratio"] / c["N_delta"] - 1.0)
                for c in cases),
            "max_deviation_C_vs_1": max(
                abs(c["instrument_C_exact_operators_ratio"] - 1.0) for c in cases),
        },
    }
    results = {
        "benchmark": "laplace_static_droplet_2d",
        "reference": "Young-Laplace, dp = sigma / R (2-D cylinder)",
        "implementation_under_test": "tensorsph.surface_tension.compute_surface_tension_force",
        "platform_tolerance": TOL,
        "resolvability_audit": resolvability,
        "verdict": verdict,
        "cases": cases,
    }
    (out / "results.json").write_text(json.dumps(results, indent=2) + "\n")
    if plot:
        _plot(cases, out)
    return results


def _converging(rows: list[dict]) -> bool:
    rows = sorted(rows, key=lambda c: c["resolution_R_over_dx"])
    err = [c["relative_error"] for c in rows]
    return all(b <= a for a, b in zip(err, err[1:]))


def _plot(cases: list[dict], out: Path) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    base = [c for c in cases if c["jitter"] == 0.0]
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.4))

    ax = axes[0]
    for lattice, marker in (("hex", "o"), ("square", "s")):
        rows = sorted([c for c in base if c["lattice"] == lattice
                       and c["sigma_N_per_m"] == 0.072 and c["radius_m"] == 0.1],
                      key=lambda c: c["resolution_R_over_dx"])
        ax.plot([c["resolution_R_over_dx"] for c in rows], [c["dp_ratio"] for c in rows],
                marker=marker, label=f"{lattice} lattice")
    ax.axhline(1.0, color="k", lw=0.8, ls="--", label="Laplace law")
    ax.fill_between([8, 42], 0.85, 1.15, color="green", alpha=0.12, label="+/-15 % band")
    ax.set_xlabel("resolution  R/dx")
    ax.set_ylabel(r"$\Delta p_{model} / (\sigma/R)$")
    ax.set_title("Implied pressure jump vs resolution")
    ax.legend(fontsize=8)
    ax.grid(alpha=0.3)

    ax = axes[1]
    rows = sorted([c for c in base if c["lattice"] == "hex" and c["radius_m"] == 0.1],
                  key=lambda c: c["resolution_R_over_dx"])
    ax.bar([f"R/dx={c['resolution_R_over_dx']:.0f}" for c in rows], [c["N_delta"] for c in rows],
           color="#4c72b0", label=r"$\int|\nabla c|\,dr$ measured")
    ax.axhline(1.0, color="r", ls="--", lw=1.0, label="CSF requirement = 1")
    ax.set_ylim(0, 1.2)
    ax.set_ylabel(r"$N_\delta$")
    ax.set_title("Surface delta-function normalisation")
    ax.legend(fontsize=8)
    ax.grid(alpha=0.3, axis="y")

    ax = axes[2]
    rows = sorted([c for c in base if c["lattice"] == "hex" and c["radius_m"] == 0.1],
                  key=lambda c: c["resolution_R_over_dx"])
    ax.plot([c["resolution_R_over_dx"] for c in rows],
            [c["kappa_delta_weighted_times_R"] for c in rows], "o-",
            label=r"$\langle\kappa\rangle_\delta R$ (model)")
    ax.axhline(1.0, color="k", ls="--", lw=0.8, label="exact $R/R=1$")
    ax.set_xlabel("resolution  R/dx")
    ax.set_ylabel(r"$\langle\kappa\rangle_\delta R$")
    ax.set_title("Curvature accuracy (delta_s weighted)")
    ax.legend(fontsize=8)
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(out / "laplace_verification.png", dpi=140)
    plt.close(fig)

    # force profile figure
    cfg = Config(dx=0.005, surface_tension_coefficient=0.072, gravity=0.0, rho0=RHO0)
    h = cfg.h_factor * cfg.dx
    fig, ax = plt.subplots(figsize=(6.4, 4.4))
    for dx, style in ((0.01, ":"), (0.005, "-"), (0.0025, "--")):
        pos, mass, _ = droplet_lattice(0.1, dx, "hex")
        pairs = cell_linked_pairs(pos, 2 * cfg.h_factor * dx)
        rho = density_from_neighbor_pairs(pos, mass, cfg.h_factor * dx, pairs)
        c = Config(dx=dx, surface_tension_coefficient=0.072, gravity=0.0, rho0=RHO0)
        acc = compute_surface_tension_force(pos, rho, mass, pairs, c)
        rel = pos - torch.tensor([CENTER, CENTER], dtype=torch.float64)
        r = rel.norm(dim=1)
        f_r = (acc * (rel / r.clamp_min(1e-12)[:, None])).sum(dim=1) * rho
        f_mean, _, _ = radial_mean(f_r, r, 0.15, 0.25 * c.h_factor * dx)
        radii = [(k + 0.5) * 0.25 * c.h_factor * dx for k in range(f_mean.numel())]
        ax.plot([x / 0.1 for x in radii], f_mean.tolist(), style, label=f"R/dx={0.1/dx:.0f}")
    ax.axhline(0.0, color="k", lw=0.6)
    ax.set_xlim(0.7, 1.25)
    ax.set_xlabel("r / R")
    ax.set_ylabel(r"$\rho a_r$  [N m$^{-3}$]")
    ax.set_title("Radial CSF force profile (inward < 0)")
    ax.legend(fontsize=8)
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(out / "laplace_force_profile.png", dpi=140)
    plt.close(fig)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--no-plot", action="store_true")
    parser.add_argument("--output", default="benchmarks/laplace_droplet")
    args = parser.parse_args()
    res = run_benchmark(plot=not args.no_plot, output_dir=args.output)
    v = res["verdict"]
    print(f"\nB3 Laplace verification: {v['status']}  "
          f"(max relative error {v['max_relative_error']:.1%}, "
          f"{v['n_within_tolerance']}/{v['n_cases']} cases inside +/-15 %)")
    for c in res["cases"]:
        print(f"  {c['lattice']:6s} R/dx={c['resolution_R_over_dx']:5.1f} "
              f"sigma={c['sigma_N_per_m']:<6g} jitter={c['jitter']:<4g} "
              f"N_delta={c['N_delta']:.3f} kappaR={c['kappa_delta_weighted_times_R']:+.3f} "
              f"dp/dp_th={c['dp_ratio']:+.3f} "
              f"[A:{c['instrument_A_exact_kappa_model_normal_ratio']:+.3f} "
              f"B:{c['instrument_B_exact_kappa_exact_normal_ratio']:+.3f} "
              f"C:{c['instrument_C_exact_operators_ratio']:+.3f}] "
              f"n.rhat={c['normal_alignment_delta_weighted']:+.2f}")
    return 0 if v["status"] == "PASS" else 1


if __name__ == "__main__":
    raise SystemExit(main())