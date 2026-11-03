"""Laplace-law verification of the CSF surface-tension model.

Why this benchmark exists
-------------------------
Surface tension (CSF, Brackbill et al. 1992) was merged into TensorSPH without
ever being checked against the analytic Young-Laplace law.  The older
``benchmarks/static_droplet`` run prints ``Δp_theory = σ/R`` but only tests the
*sign* of the force, so that theory value is decorative.  This harness measures
the pressure jump and compares it, number for number, with ``σ/R``.

Three analytic references (2-D, single-phase drop)
--------------------------------------------------
1. **Young-Laplace**: a circular drop of radius ``R`` has one non-zero
   principal curvature, hence

       Δp = p_in - p_out = σ / R                                    (1)

2. **Cut-force identity**: a cut through the drop exposes two interfaces, so
   the total surface-tension force transmitted across the cut is exactly
   ``2σ`` per unit out-of-plane depth (the classical "surface tension =
   force per unit length").  This scores the *magnitude* of the CSF force
   without relying on any pressure measurement:

       F_cut = Σ_{x>0} m_i a_st,i,x  ->  -2σ                        (2)

3. **Curvature**: the CSF pipeline must return ``κ = 1/R`` on a circle.

The spurious-current level is scored against
``max|v| / sqrt(σ/(ρR)) << 1`` (Adami, Ihmsen & Adams 2012, JCP 231:7057).

Usage::

    PYTHONPATH=src python benchmarks/laplace_law/run.py --mode instant
    PYTHONPATH=src python benchmarks/laplace_law/run.py --mode relax
"""
from __future__ import annotations

import argparse
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
from tensorsph.csf_consistent import surface_tension_acceleration
from tensorsph.surface_tension import (
    compute_color_gradient,
    compute_curvature,
    compute_normal,
    compute_surface_tension_force,
)

torch.set_num_threads(2)


# --------------------------------------------------------------------------- #
# geometry
# --------------------------------------------------------------------------- #
def hexagonal_droplet(radius: float, dx: float, rho0: float,
                      center: tuple[float, float] = (0.0, 0.0)):
    """Hexagonally packed circular drop.

    A square lattice has 4 nearest neighbours, hex packing has 6, which halves
    the noise of the neighbour-count surface detector and of the SPH gradient
    operators.  The lattice constant is set so ``mass * number_density = rho0``.
    """
    dy = dx * math.sqrt(3.0) / 2.0
    mass = torch.tensor(rho0 * dx * dy, dtype=torch.float64)
    pts = []
    j = 0
    y = -radius
    while y <= radius:
        offset = 0.5 * dx if (j % 2) else 0.0
        x = -radius + offset
        while x <= radius:
            if x * x + y * y <= radius * radius:
                pts.append((center[0] + x, center[1] + y))
            x += dx
        y += dy
        j += 1
    pos = torch.tensor(pts, dtype=torch.float64)
    return pos, torch.full((pos.shape[0],), float(mass), dtype=torch.float64)


# --------------------------------------------------------------------------- #
# CSF pipeline introspection (mirrors compute_surface_tension_force)
# --------------------------------------------------------------------------- #
def csf_fields(pos, rho, mass, pairs, cfg):
    """LEGACY-scheme introspection only.

    This is a copy of the *original* CSF field construction (neighbour-count
    colour with a 0.7 x mean threshold, uncorrected operators, ``kappa = div n``,
    no factor 2 on ``delta_s``).  It describes the force only when
    ``cfg.csf_scheme == "legacy"``; for the production ("consistent") scheme use
    :func:`production_fields`, which reads the very state the force was built
    from.  Keeping both is the point of the scheme ablation below: before
    m2857 this function was used unconditionally, so the reported curvature and
    ``delta_s`` did **not** describe the force that was actually applied.
    """
    h = cfg.h_factor * cfg.dx
    n = pos.shape[0]
    i, j = pairs.unbind(dim=1)
    count = torch.zeros(n, dtype=pos.dtype)
    one = torch.ones_like(i, dtype=pos.dtype)
    count = count.index_add(0, i, one).index_add(0, j, one)
    avg = count.mean()
    is_surface = count < 0.7 * avg
    color = torch.ones(n, dtype=pos.dtype)
    color[is_surface] = count[is_surface] / avg
    grad_c = compute_color_gradient(pos, color, mass, rho, pairs, h)
    normal = compute_normal(-grad_c)
    kappa = compute_curvature(pos, normal, mass, rho, pairs, h)
    delta_s = torch.linalg.vector_norm(grad_c, dim=1)
    return color, grad_c, normal, kappa, delta_s, is_surface


def apply_scheme(cfg: Config, scheme: str) -> Config:
    """Translate a ``scheme`` label into the two ``Config`` switches.

    ``"legacy"`` is the pre-consistent CSF; ``"consistent:mls"`` is the shipped
    default (kernel-sum colour + MLS operators, ``kappa = div n``);
    ``"consistent:fit_quadratic"`` / ``"consistent:fit_circle"`` swap in the
    weighted local interface fit of :mod:`tensorsph.csf_curvature`.
    """
    if scheme == "legacy":
        cfg.csf_scheme, cfg.csf_curvature = "legacy", "mls"
    else:
        kind, _, curv = scheme.partition(":")
        assert kind == "consistent", scheme
        cfg.csf_scheme, cfg.csf_curvature = "consistent", curv or "mls"
    return cfg


def production_fields(pos, rho, mass, pairs, cfg):
    """``(kappa, delta_s, is_surface, acc_st)`` from the production CSF path.

    The force and the fields come from one call, so the curvature scored in the
    report is the curvature the force was actually built with.  The "consistent"
    path also widens the pair list to the fit stencil internally
    (``surface_tension_acceleration``); feeding the 2h list straight into
    ``csf_fields`` under-samples the stencil and inflates ``kappa`` by 3-10x, so
    the two paths are not interchangeable.
    """
    if cfg.csf_scheme == "legacy":
        color, _, _, kappa, delta_s, is_surface = csf_fields(pos, rho, mass,
                                                             pairs, cfg)
        acc = compute_surface_tension_force(pos, rho, mass, pairs, cfg)
        return kappa, delta_s, is_surface, acc
    h = cfg.h_factor * cfg.dx
    acc, state = surface_tension_acceleration(
        pos, rho, mass, pairs, h, cfg.surface_tension_coefficient, cfg.rho0,
        curvature_scheme=cfg.csf_curvature)
    return state.kappa, state.delta_s, state.surface_mask(0.05), acc


def radial_band(pos, values, radius, lo=0.6, hi=0.98):
    r = torch.linalg.vector_norm(pos, dim=1) / radius
    m = (r >= lo) & (r <= hi)
    if not bool(m.any()):
        return {"count": 0, "mean": float("nan"), "median": float("nan"),
                "std": float("nan")}
    v = values[m]
    return {
        "count": int(m.sum()),
        "mean": float(v.mean()),
        "median": float(v.median()),
        "std": float(v.std()) if v.numel() > 1 else 0.0,
    }


def pressure_floor(rho, pos, radius, cfg, dp_theory):
    """Is the measured pressure jump resolvable at all?

    ``Delta p`` is read as ``p_core - p_shell``, and ``tait_pressure`` clamps at
    zero, so a shell whose *summation* density sits below ``rho0`` reports
    ``p_shell = 0`` no matter how far below it is.  Kernel truncation alone puts
    the free-surface layer several percent under ``rho0``; the unclamped Tait
    pressure there is then tens of kPa negative, i.e. tens to hundreds of times
    the ``sigma/R`` signal that is being claimed as measured.  The clamp is a
    legitimate free-surface treatment, but it makes the *difference* of two
    pressures a measurement of the clamp, not of the Laplace law.

    The floor is quantified by ``signal_to_floor = Delta p_theory /
    |p_shell_unclamped|``; ``delta_p_valid`` requires the suppressed pressure to
    be small compared with the signal (``<= 10 %``, i.e. it could not move the
    answer by more than one gate width of the 15 % acceptance band).
    """
    r = torch.linalg.vector_norm(pos, dim=1) / radius
    shell = (r > 0.8) & (r < 0.95)
    rho_s = float(rho[shell].mean())
    bulk = cfg.rho0 * cfg.sound_speed ** 2 / cfg.gamma
    p_raw = bulk * ((rho_s / cfg.rho0) ** cfg.gamma - 1.0)
    deficit = (1.0 - rho_s / cfg.rho0) * 100.0
    return {
        "rho_shell_over_rho0": rho_s / cfg.rho0,
        "shell_density_deficit_pct": deficit,
        "tait_bulk_modulus_Pa": bulk,
        "p_shell_unclamped_Pa": p_raw,
        "p_shell_reported_by_tait_Pa": max(p_raw, 0.0),
        "zero_clamp_active": p_raw < 0.0,
        "delta_p_signal_Pa": dp_theory,
        "signal_to_floor": abs(dp_theory / p_raw) if p_raw else float("inf"),
        "delta_p_valid": abs(dp_theory / p_raw) >= 10.0 if p_raw else False,
        "criterion": "delta_p is only a measurement of sigma/R when the "
                     "pressure the Tait clamp suppresses at the shell "
                     "(|p_shell_unclamped|) is <= 10 % of sigma/R",
    }


def pressure_profile(pos, pressure, radius, nbins=20):
    r = torch.linalg.vector_norm(pos, dim=1) / radius
    edges = torch.linspace(0.0, 1.0, nbins + 1, dtype=pos.dtype)
    prof = []
    for k in range(nbins):
        m = (r >= edges[k]) & (r < edges[k + 1])
        if bool(m.any()):
            prof.append({
                "r_over_R": float(0.5 * (edges[k] + edges[k + 1])),
                "p_mean": float(pressure[m].mean()),
                "p_std": float(pressure[m].std()) if int(m.sum()) > 1 else 0.0,
                "n": int(m.sum()),
            })
    return prof


def instant_diagnostics(radius, dx, sigma, rho0, sound_speed, viscosity,
                        scheme="consistent:mls"):
    """Everything that can be scored on a single static evaluation."""
    cfg = Config(dx=dx, h_factor=1.3, rho0=rho0, sound_speed=sound_speed,
                 gamma=7.0, gravity=0.0, viscosity=viscosity,
                 surface_tension_coefficient=sigma)
    apply_scheme(cfg, scheme)
    h = cfg.h_factor * cfg.dx
    pos, mass = hexagonal_droplet(radius, dx, rho0)
    n = pos.shape[0]
    pairs = cell_linked_pairs(pos, 2.0 * h)
    rho = density_from_neighbor_pairs(pos, mass, h, pairs)

    kappa, delta_s, is_surface, acc_st = production_fields(pos, rho, mass,
                                                           pairs, cfg)

    # -- cut-force identity (2): force transmitted across the y-axis cut ------- #
    right = pos[:, 0] > 0.0
    f_cut = float((mass[right, None] * acc_st[right]).sum(dim=0)[0])
    kappa_exact = 1.0 / radius
    f_cut_theory = -2.0 * sigma

    kappa_band = radial_band(pos, kappa.abs(), radius)
    # Interface curvature: the physically meaningful average is weighted by the
    # surface delta function |grad c|, because that is the weight the CSF force
    # itself uses.  An unweighted band average mixes interior particles
    # (kappa ~ 0) with interface particles (kappa large) and is not a fair
    # score of the estimator.
    w = delta_s.abs()
    kappa_w = float((kappa.abs() * w).sum() / w.sum().clamp_min(1e-30))
    ds_band = radial_band(pos, delta_s, radius)
    # delta_s normalisation, stated as the integral it has to satisfy:
    #   int_over_the_interface delta_s dn = 1   (Brackbill et al. 1992)
    # In 2-D the 1-D integral across the interface times the interface length is
    # the area integral, so with V_i = m_i/rho0 and a circle of length 2 pi R
    #       S = sum_i delta_s_i V_i / (2 pi R)
    # must be 1.  A "band mean x 2h" number is a proxy that depends on where the
    # band edges fall; S does not.
    vol = mass / cfg.rho0
    ds_integral = float((delta_s * vol).sum() / (2.0 * math.pi * radius))
    # inward alignment: a_st should oppose the radial unit vector
    r_hat = pos / torch.linalg.vector_norm(pos, dim=1, keepdim=True)
    mag = torch.linalg.vector_norm(acc_st, dim=1)
    # true cosine: +1 means the CSF force points exactly inward (opposite r_hat)
    cos_in = -(acc_st * r_hat).sum(dim=1) / mag.clamp_min(1e-30)
    aligned = mag > 0
    r_norm = torch.linalg.vector_norm(pos, dim=1) / radius
    band = aligned & (r_norm >= 0.6) & (r_norm <= 0.98)
    cb = cos_in[band]
    cos_in_band = {"count": int(band.sum()),
                   "mean": float(cb.mean()) if cb.numel() else float("nan"),
                   "median": float(cb.median()) if cb.numel() else float("nan")}

    # pressure implied by the cut force, for comparison with (1)
    dp_from_force = abs(f_cut) / (2.0 * radius)

    return {
        "scheme": scheme,
        "dx_m": dx,
        "h_m": h,
        "n_particles": n,
        "dt_acoustic_s": float(stable_dt(cfg)),
        "n_surface_particles": int(is_surface.sum()),
        "surface_fraction": float(is_surface.float().mean()),
        "curvature": {
            "exact_1_per_m": kappa_exact,
            "band": kappa_band,
            "rel_error_pct": (kappa_band["mean"] - kappa_exact)
                             / kappa_exact * 100.0,
            "delta_s_weighted_mean_1_per_m": kappa_w,
            "delta_s_weighted_rel_error_pct": (kappa_w - kappa_exact)
                                              / kappa_exact * 100.0,
        },
        "delta_s": {
            "band": ds_band,
            "ideal_value_1_per_m": 1.0 / (2.0 * h),   # |grad c| across one h
            "line_integral": ds_integral,
            "line_integral_ideal": 1.0,
            "line_integral_rel_error_pct": (ds_integral - 1.0) * 100.0,
        },
        "cut_force": {
            "theory_N_per_m": f_cut_theory,
            "measured_N_per_m": f_cut,
            "ratio_measured_over_theory": f_cut / f_cut_theory,
            "implied_delta_p_Pa": dp_from_force,
        },
        "inward_alignment": {
            "cos_band_mean": cos_in_band["mean"],
            "cos_band_median": cos_in_band["median"],
            "fraction_inward": float((cos_in[aligned] > 0).float().mean()),
            "n_aligned": int(aligned.sum()),
        },
        "max_acc_st_m_per_s2": float(mag.max()),
        "mean_acc_st_m_per_s2": float(mag.mean()),
    }


def relax(radius=0.02, dx=0.002, sigma=7.2, rho0=1000.0, sound_speed=20.0,
          steps=900, viscosity=0.01, scheme="consistent:mls"):
    cfg = Config(dx=dx, h_factor=1.3, rho0=rho0, sound_speed=sound_speed,
                 gamma=7.0, gravity=0.0, viscosity=viscosity,
                 surface_tension_coefficient=sigma)
    apply_scheme(cfg, scheme)
    h = cfg.h_factor * cfg.dx
    dt = float(stable_dt(cfg))
    t_sigma = math.sqrt(rho0 * radius ** 3 / sigma)
    v_ref = math.sqrt(sigma / (rho0 * radius))
    dp_theory = sigma / radius

    pos, mass = hexagonal_droplet(radius, dx, rho0)
    n = pos.shape[0]
    vel = torch.zeros_like(pos)
    pairs = cell_linked_pairs(pos, 2.0 * h)
    rho = density_from_neighbor_pairs(pos, mass, h, pairs)
    pressure = tait_pressure(rho, cfg)

    hist = {"t": [], "vmax": [], "ke": [], "p_center": [], "rho_center": []}
    for step in range(steps):
        pairs = cell_linked_pairs(pos, 2.0 * h)
        rho, pressure, acc = rhs(pos, vel, mass, cfg, pairs=pairs, wall=None,
                                 dt=dt)
        vel = vel + dt * acc
        pos = pos + dt * vel
        if step % 50 == 0 or step == steps - 1:
            r = torch.linalg.vector_norm(pos, dim=1)
            core = r < 0.3 * radius
            hist["t"].append((step + 1) * dt)
            hist["vmax"].append(float(torch.linalg.vector_norm(vel, dim=1).max()))
            hist["ke"].append(float(0.5 * (mass @ (vel * vel).sum(dim=1))))
            hist["p_center"].append(float(pressure[core].mean()))
            hist["rho_center"].append(float(rho[core].mean()))

    pairs = cell_linked_pairs(pos, 2.0 * h)
    rho_f = density_from_neighbor_pairs(pos, mass, h, pairs)
    p_f = tait_pressure(rho_f, cfg)
    r = torch.linalg.vector_norm(pos, dim=1) / radius
    p_in = float(p_f[r < 0.5].mean())
    p_band = float(p_f[(r > 0.8) & (r < 0.95)].mean())
    dp_meas = p_in - p_band
    err = (dp_meas - dp_theory) / dp_theory * 100.0
    vmax = float(torch.linalg.vector_norm(vel, dim=1).max())
    kappa_f, ds_f, _, _ = production_fields(pos, rho_f, mass, pairs, cfg)
    acc_st_f = compute_surface_tension_force(pos, rho_f, mass, pairs, cfg)
    right = pos[:, 0] > 0.0
    f_cut = float((mass[right, None] * acc_st_f[right]).sum(dim=0)[0])
    comp_needed = (1.0 + dp_theory / (rho0 * sound_speed ** 2 / 7.0)) ** (1 / 7) - 1
    return {
        "setup": {
            "radius_m": radius, "dx_m": dx, "h_m": h, "dt_s": dt,
            "n_particles": n, "sigma_N_per_m": sigma, "rho0": rho0,
            "sound_speed": sound_speed, "viscosity": viscosity,
            "steps": steps, "physical_time_s": steps * dt,
            "t_over_t_sigma": steps * dt / t_sigma,
            "lattice": "hexagonal", "gravity": 0.0,
            "csf_scheme": cfg.csf_scheme, "csf_curvature": cfg.csf_curvature,
            "scheme": scheme,
        },
        "reference": {
            "delta_p_Pa": dp_theory,
            "curvature_1_per_m": 1.0 / radius,
            "capillary_time_s": t_sigma,
            "capillary_velocity_m_per_s": v_ref,
            "cut_force_N_per_m": -2.0 * sigma,
            "spurious_current_criterion": "max|v|/sqrt(sigma/(rho R)) << 1 "
                                          "(Adami et al. 2012)",
        },
        "laplace": {
            "delta_p_theory_Pa": dp_theory,
            "p_center_Pa": p_in,
            "p_interface_band_Pa": p_band,
            "delta_p_measured_Pa": dp_meas,
            "rel_error_pct": err,
            "within_3pct": abs(err) < 3.0,
            "within_15pct": abs(err) < 15.0,
        },
        "curvature_relaxed": {
            "band": radial_band(pos, kappa_f.abs(), radius),
            "delta_s_weighted_mean_1_per_m": float(
                (kappa_f.abs() * ds_f).sum() / ds_f.sum().clamp_min(1e-30)),
            "delta_s_weighted_rel_error_pct": float(
                ((kappa_f.abs() * ds_f).sum() / ds_f.sum().clamp_min(1e-30)
                 - 1.0 / radius) / (1.0 / radius) * 100.0),
        },
        "spurious_current": {
            "v_ref_m_per_s": v_ref,
            "vmax_final_m_per_s": vmax,
            "ratio": vmax / v_ref,
            "passes_adami_0p1": vmax / v_ref < 0.1,
        },
        "density": {
            "rho_center_over_rho0": float(rho_f[r < 0.5].mean() / rho0),
            "rho_band_over_rho0": float(rho_f[(r > 0.8) & (r < 0.95)].mean() / rho0),
            "compression_required_by_law": comp_needed,
        },
        "pressure_floor": pressure_floor(rho_f, pos, radius, cfg, dp_theory),
        "delta_p_from_cut_force": {
            "Pa": abs(f_cut) / (2.0 * radius),
            "rel_error_pct": (abs(f_cut) / (2.0 * radius) - dp_theory)
                             / dp_theory * 100.0,
            "within_15pct": abs(abs(f_cut) / (2.0 * radius) - dp_theory)
                            / dp_theory < 0.15,
            "note": "Delta p implied by the cut-force identity "
                    "F_cut = Delta p * 2R; independent of the pressure field, "
                    "so it is the resolvable stand-in while the pressure floor "
                    "is above the signal",
        },
        "pressure_profile_final": pressure_profile(pos, p_f, radius),
        "history": hist,
    }


def make_figures(relax_report, instant_rows, out_dir):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    out = Path(out_dir)
    fig, axes = plt.subplots(1, 4, figsize=(19, 4.2))

    prof = relax_report["pressure_profile_final"]
    axes[0].errorbar([p["r_over_R"] for p in prof], [p["p_mean"] for p in prof],
                     yerr=[p["p_std"] for p in prof], fmt="o-", ms=4,
                     label="SPH pressure")
    axes[0].axhline(relax_report["reference"]["delta_p_Pa"], color="k", ls="--",
                    label=r"analytic $\Delta p=\sigma/R$")
    axes[0].axhline(relax_report["laplace"]["delta_p_measured_Pa"], color="r",
                    ls=":", label="measured jump")
    axes[0].set_xlabel("r / R")
    axes[0].set_ylabel("pressure [Pa]")
    axes[0].set_title("(a) pressure profile vs Young-Laplace")
    axes[0].legend(fontsize=8)
    axes[0].grid(alpha=0.3)

    hist = relax_report["history"]
    vr = relax_report["spurious_current"]["v_ref_m_per_s"]
    axes[1].semilogy(hist["t"], [max(v, 1e-12) / vr for v in hist["vmax"]], "o-",
                     ms=3, label=r"$|v|_{max}/\sqrt{\sigma/\rho R}$")
    axes[1].axhline(0.1, color="r", ls="--", label="Adami threshold 0.1")
    axes[1].set_xlabel("t [s]")
    axes[1].set_title("(b) spurious current")
    axes[1].legend(fontsize=8)
    axes[1].grid(alpha=0.3, which="both")

    kx = [r["curvature"]["exact_1_per_m"] for r in instant_rows]
    axes[2].plot([r["dx_m"] for r in instant_rows],
                 [r["curvature"]["band"]["mean"] for r in instant_rows], "o-",
                 label=r"measured $|\kappa|$")
    axes[2].plot([r["dx_m"] for r in instant_rows], kx, "k--",
                 label=r"analytic $\kappa=1/R$")
    axes[2].set_xscale("log")
    axes[2].set_xlabel("dx [m]")
    axes[2].set_ylabel(r"$|\kappa|$ [1/m]")
    axes[2].set_title("(c) CSF curvature vs exact (resolution scan)")
    axes[2].legend(fontsize=8)
    axes[2].grid(alpha=0.3)

    axes[3].plot([r["dx_m"] for r in instant_rows],
                 [r["cut_force"]["ratio_measured_over_theory"]
                  for r in instant_rows], "o-", label=r"$F_{cut}/(2\sigma)$")
    axes[3].axhline(1.0, color="k", ls="--", label="identity: 1.0")
    axes[3].set_xscale("log")
    axes[3].set_xlabel("dx [m]")
    axes[3].set_title("(d) cut-force identity vs resolution")
    axes[3].legend(fontsize=8)
    axes[3].grid(alpha=0.3)

    fig.tight_layout()
    path = out / "laplace_law.png"
    fig.savefig(path, dpi=140)
    print(f"figure -> {path}")
    return path


SCHEME_LABELS = {
    "legacy": "legacy (neighbour-count colour)",
    "consistent:mls": "consistent, kappa = div n",
    "consistent:fit_quadratic": "consistent, quadratic fit",
    "consistent:fit_circle": "consistent, circle fit",
}


def make_scheme_figure(payload, relax_report, out_dir):
    """Curvature-scheme ablation + pressure-floor evidence."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    out = Path(out_dir)
    fig, axes = plt.subplots(1, 3, figsize=(15.5, 4.2))
    ablation = payload["scheme_ablation"]
    styles = {"legacy": "s--", "consistent:mls": "o-",
              "consistent:fit_quadratic": "^-", "consistent:fit_circle": "v:"}

    for scheme, rows in ablation.items():
        label = SCHEME_LABELS.get(scheme, scheme)
        st = styles.get(scheme, "o-")
        dx = [r["dx_m"] for r in rows]
        axes[0].plot(dx, [r["curvature"]["delta_s_weighted_rel_error_pct"]
                          for r in rows], st, ms=5, label=label)
        axes[1].plot(dx, [r["cut_force"]["ratio_measured_over_theory"]
                          for r in rows], st, ms=5, label=label)
    for ax in axes[:2]:
        ax.set_xscale("log")
        ax.set_xlabel("dx [m]")
        ax.grid(alpha=0.3)
    axes[0].axhspan(-15.0, 15.0, color="g", alpha=0.10,
                    label="15 % acceptance band")
    axes[0].set_ylabel(r"$\kappa$ error [%]")
    axes[0].set_title(r"(a) $\kappa$ vs analytic $1/R$ (weighted)")
    axes[0].legend(fontsize=7)
    axes[1].axhspan(0.85, 1.15, color="g", alpha=0.10,
                    label="15 % acceptance band")
    axes[1].axhline(1.0, color="k", ls="--")
    axes[1].set_ylabel(r"$F_{cut}/(2\sigma)$")
    axes[1].set_title("(b) cut-force identity")
    axes[1].legend(fontsize=7)

    # (c) the pressure jump: direct probe vs cut-force proxy, with the floor
    names, probe, proxy, valid = [], [], [], []
    for entry in relax_report["scheme_ablation"]:
        names.append(entry["scheme"].replace("consistent:", ""))
        probe.append(entry["laplace"]["rel_error_pct"])
        proxy.append(entry["delta_p_from_cut_force"]["rel_error_pct"])
        valid.append(entry["pressure_floor"]["delta_p_valid"])
    y = range(len(names))
    axes[2].barh([i + 0.2 for i in y], proxy, height=0.38, color="tab:blue",
                 label=r"via $F_{cut}/(2R)$")
    axes[2].barh([i - 0.2 for i in y], probe, height=0.38, color="tab:red",
                 label=r"direct $p_{core}-p_{shell}$")
    axes[2].set_yticks(list(y))
    axes[2].set_yticklabels([f"{n}{'' if v else ' (invalid)'}"
                             for n, v in zip(names, valid)], fontsize=8)
    axes[2].axvspan(-15.0, 15.0, color="g", alpha=0.10)
    axes[2].axvline(0.0, color="k", ls="--", lw=0.8)
    axes[2].set_xlabel(r"$\Delta p$ error vs $\sigma/R$ [%]")
    axes[2].set_title(r"(c) $\Delta p$ at dx = 4 mm: probe vs proxy")
    axes[2].legend(fontsize=7)
    axes[2].grid(alpha=0.3, axis="x")

    fig.tight_layout()
    path = out / "curvature_scheme_ablation.png"
    fig.savefig(path, dpi=140)
    print(f"figure -> {path}")
    return path


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", choices=["instant", "relax", "both"], default="both")
    ap.add_argument("--radius", type=float, default=0.02)
    ap.add_argument("--dx", type=float, default=0.002)
    ap.add_argument("--sigma", type=float, default=7.2)
    ap.add_argument("--steps", type=int, default=900)
    ap.add_argument("--schemes", default="legacy,consistent:mls,"
                                        "consistent:fit_quadratic,"
                                        "consistent:fit_circle")
    ap.add_argument("--abl_steps", type=int, default=380,
                    help="relaxation steps of the scheme ablation (coarse dx)")
    ap.add_argument("--out", default="benchmarks/laplace_law")
    a = ap.parse_args()

    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    instant_path = out / "instant_report.json"
    relax_path = out / "validation_report.json"

    schemes = [x for x in a.schemes.split(",") if x]
    dxs = (0.001, 0.0015, 0.002, 0.003)
    if a.mode in ("instant", "both"):
        ablation = {}
        for scheme in schemes:
            ablation[scheme] = [instant_diagnostics(a.radius, dx, a.sigma,
                                                    1000.0, 20.0, 0.01,
                                                    scheme=scheme)
                                for dx in dxs]
        rows = ablation.get("consistent:mls", next(iter(ablation.values())))
        payload = {
            "benchmark": "csf_surface_tension_instant_diagnostics",
            "reference": {
                "curvature_1_per_m": 1.0 / a.radius,
                "cut_force_N_per_m": -2.0 * a.sigma,
                "note": "cut-force identity: surface tension = force per unit "
                        "length, so a cut through the drop transmits 2*sigma",
            },
            "setup": {"radius_m": a.radius, "sigma_N_per_m": a.sigma,
                      "rho0": 1000.0, "sound_speed": 20.0,
                      "default_scheme": "consistent:mls",
                      "introspection": "production: force and kappa/delta_s "
                                       "come from one surface_tension_"
                                       "acceleration call"},
            "resolutions": rows,
            "scheme_ablation": ablation,
        }
        instant_path.write_text(json.dumps(payload, indent=2))
        for scheme, drows in ablation.items():
            for r in drows:
                print(f"{scheme:22s} dx={r['dx_m']:.4f} N={r['n_particles']:5d} "
                      f"kappa_w={r['curvature']['delta_s_weighted_mean_1_per_m']:7.1f} "
                      f"(w err {r['curvature']['delta_s_weighted_rel_error_pct']:+7.1f}%) "
                      f"Fcut/2sigma={r['cut_force']['ratio_measured_over_theory']:+7.3f} "
                      f"cos={r['inward_alignment']['cos_band_median']:+.3f} "
                      f"surf={r['surface_fraction']:.2f}")

    if a.mode in ("relax", "both"):
        rep = relax(radius=a.radius, dx=a.dx, sigma=a.sigma, steps=a.steps)
        # Scheme ablation in time: same coarse case as the fast regression test.
        rep["scheme_ablation"] = []
        for scheme in schemes:
            sub_rep = relax(radius=a.radius, dx=0.004, sigma=a.sigma,
                            steps=a.abl_steps, scheme=scheme)
            rep["scheme_ablation"].append({
                "scheme": scheme,
                "setup": sub_rep["setup"],
                "laplace": sub_rep["laplace"],
                "pressure_floor": sub_rep["pressure_floor"],
                "delta_p_from_cut_force": sub_rep["delta_p_from_cut_force"],
                "spurious_current": sub_rep["spurious_current"],
                "curvature_relaxed": sub_rep["curvature_relaxed"],
            })
            print(f"ablation {scheme:22s} dx=0.004 dp={sub_rep['laplace']['delta_p_measured_Pa']:9.2f} "
                  f"({sub_rep['laplace']['rel_error_pct']:+7.1f}%) "
                  f"valid={sub_rep['pressure_floor']['delta_p_valid']} "
                  f"F/2sigma->dp={sub_rep['delta_p_from_cut_force']['Pa']:7.1f} "
                  f"({sub_rep['delta_p_from_cut_force']['rel_error_pct']:+6.1f}%)")
        relax_path.write_text(json.dumps(rep, indent=2))
        print(f"RELAX dx={a.dx} steps={a.steps} "
              f"t/t_sigma={rep['setup']['t_over_t_sigma']:.2f}: "
              f"dp theory {rep['laplace']['delta_p_theory_Pa']:.1f} Pa, "
              f"measured {rep['laplace']['delta_p_measured_Pa']:.2f} Pa, "
              f"err {rep['laplace']['rel_error_pct']:+.1f}%, "
              f"spurious {rep['spurious_current']['ratio']:.3f}, "
              f"pressure floor ratio "
              f"{rep['pressure_floor']['signal_to_floor']:.4f} "
              f"(valid={rep['pressure_floor']['delta_p_valid']}), "
              f"dp(cut force) "
              f"{rep['delta_p_from_cut_force']['rel_error_pct']:+.1f}%")

    if a.mode == "both":
        inst_payload = json.loads(instant_path.read_text())
        make_figures(json.loads(relax_path.read_text()),
                     inst_payload["resolutions"], out)
        make_scheme_figure(inst_payload, json.loads(relax_path.read_text()), out)


if __name__ == "__main__":
    main()