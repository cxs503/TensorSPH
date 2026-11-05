"""Capillary-wave dispersion and damping: quantitative validation vs Lamb (1932).

Why this harness exists
-----------------------
``tests/test_capillary_wave.py`` only checks the *formula* ``omega^2 =
sigma k^3 / rho`` against itself, ``benchmarks/capillary_wave/run.py`` never ran
at HEAD (it called the removed ``Config.dt_from_courant``) and its amplitude
estimator was unsigned, so ``find_period`` could never report a period.  In
other words the "capillary-wave" capability of TensorSPH had **no** end-to-end
validation: nothing ever compared a simulated oscillation with the analytic
frequency.  This file supplies that comparison.

Two analytic references, both classical
---------------------------------------
Dispersion (Lamb 1932, Hydrodynamics 6th ed., Art. 267 for deep water and
Art. 348 for the finite-depth factor):

    omega^2 = (g k + sigma k^3 / rho) tanh(k H) ,        k = 2 pi / lambda

The harness runs with ``g = 0`` (pure capillary regime) and ``k H ~ 6`` so the
finite-depth factor is 0.99999; it is nevertheless kept in the reference because
the propagation of the code must not silently assume deep water.

Viscous damping (Lamb 1932, Art. 348; Landau & Lifshitz, Fluid Mechanics §25):

    gamma = 2 nu k^2

for a weakly damped capillary wave.  This is the one analytic handle that the
viscous operator of the solver has: unlike the dam-break or impact cases it does
not need a tuned coefficient, because the decay rate of a standing wave is a
closed-form function of ``nu`` and ``k``.

Standing-wave set-up
--------------------
Domain width ``lambda/2`` with a free-slip (``clamp``) side boundary, so the
eigenmode ``eta = A cos(k x)`` has zero horizontal velocity at ``x = 0`` and
``x = lambda/2`` (crest/trough), which is exactly what the side clamp imposes.
The bottom is a free-slip clamp at ``y = 0``; the free surface is at
``y = H + eta(x, t)`` with ``H`` the water depth; no wall particles are used, so
the case exercises the free-surface path of ``rhs``.

The interface displacement is measured from the **column mass**,

    eta_b(t) = sum_{i in bin b} m_i / (rho0 * dx_bin) - H ,

which is exact up to the (static) lattice quadrature of the initial state and
does not quantise the surface to the top particle row.  The mode amplitude is
the cosine projection

    a(t) = (2 / W) * sum_b eta_b(t) cos(k x_b) dx_bin      ( -> A at t = 0 ),

and ``(omega, gamma)`` follow from a least-squares fit of
``a(t) = offset + exp(-gamma t)[c cos(omega t) + s sin(omega t)]``.  The fit is
a two-stage grid search (no scipy), and every fit reports its relative residual
so an unconverged or noise-dominated run is visible instead of being scored as a
number.

Usage::

    PYTHONPATH=src python benchmarks/capillary_wave/validation.py --case all
    PYTHONPATH=src python benchmarks/capillary_wave/validation.py --case dispersion --periods 3

Exit status is non-zero when a measured quantity misses its analytic value by
more than ``--gate`` (default 15 %).  ``--gate 3`` applies the platform's
strict tier; the JSON always records both.
"""
from __future__ import annotations

import argparse
import json
import math
import time
from pathlib import Path

import numpy as np
import torch

from tensorsph.droplet_impact import (
    Config,
    apply_boundary,
    resolve_device,
    rhs,
    stable_dt,
    suggested_dt,
)
from tensorsph.neighbor_search import PairListCache

HERE = Path(__file__).resolve().parent
torch.set_num_threads(2)

GATE_PCT = 15.0
STRICT_PCT = 3.0


# --------------------------------------------------------------------------- #
# analytic references
# --------------------------------------------------------------------------- #
def angular_frequency(sigma: float, k: float, rho: float, depth: float,
                      gravity: float = 0.0) -> float:
    """``omega^2 = (g k + sigma k^3 / rho) tanh(k H)`` (Lamb 1932, Art. 267/348)."""
    return math.sqrt((gravity * k + sigma * k**3 / rho) * math.tanh(k * depth))


def damping_rate(nu: float, k: float) -> float:
    """``gamma = 2 nu k^2`` for a weakly damped capillary wave (Lamb 1932, Art. 348)."""
    return 2.0 * nu * k**2


def rel_error_pct(measured: float, exact: float) -> float:
    return (measured - exact) / exact * 100.0


# --------------------------------------------------------------------------- #
# set-up and diagnostics
# --------------------------------------------------------------------------- #
def make_case(wavelength: float, depth: float, dx: float, amplitude: float,
              rho0: float, sigma: float, viscosity: float, sound_speed: float,
              gamma_eos: float = 7.0, h_factor: float = 1.3,
              lattice: str = "hex",
              density_diffusion: float = 0.0,
              ) -> tuple[torch.Tensor, torch.Tensor, Config]:
    """Lattice fluid under a ``eta = A cos(k x)`` interface, width ``lambda/2``.

    ``lattice="hex"`` (default) packs rows at ``dx sqrt(3)/2`` with alternating
    half-spacing offsets, so each particle owns the area ``dx^2 sqrt(3)/2`` and
    uniform ``rho0`` masses follow.  A square lattice puts the interface on a
    particle-scale staircase whose curvature noise is far larger than the wave
    curvature ``A k^2``; the triangular packing is the standard cure and is what
    the droplet benchmarks of this repo already use.
    """
    if lattice not in ("hex", "square"):
        raise ValueError("lattice must be 'hex' or 'square'")
    k = 2.0 * math.pi / wavelength
    width = wavelength / 2.0
    nx = max(4, int(round(width / dx)))
    dy = dx * math.sqrt(3.0) / 2.0 if lattice == "hex" else dx
    ny = max(4, int(round((depth + amplitude) / dy)))
    ys = (torch.arange(ny, dtype=torch.float64) + 0.5) * dy
    eta_y = depth + amplitude * torch.cos(k * (
        (torch.arange(nx, dtype=torch.float64) + 0.5) * (width / nx)))
    rows = []
    hx = width / nx
    for j, y in enumerate(ys):
        if lattice == "hex" and (j % 2):
            # half-spacing stagger; the odd rows drop their last column so that
            # no particle starts on (or outside) the clamp boundary
            xs = (torch.arange(nx - 1, dtype=torch.float64) + 1.0) * hx
        else:
            xs = (torch.arange(nx, dtype=torch.float64) + 0.5) * hx
        # the interface is evaluated at the actual (shifted) x of each particle
        keep = y < depth + amplitude * torch.cos(k * xs)
        rows.append(torch.stack((xs[keep], torch.full_like(xs[keep], y)), dim=1))
    pos = torch.cat(rows, dim=0).contiguous()
    area = dx * dx * (math.sqrt(3.0) / 2.0 if lattice == "hex" else 1.0)
    mass = torch.full((pos.shape[0],), rho0 * area, dtype=torch.float64)
    _ = eta_y

    # drop_center_y / drop_radius only set domain_top(); the box top is passed
    # explicitly to apply_boundary, so any sane value works here.
    cfg = Config(dx=dx, rho0=rho0, sound_speed=sound_speed, gamma=gamma_eos,
                 gravity=0.0, viscosity=viscosity,
                 surface_tension_coefficient=sigma, boundary="clamp",
                 pool_width=width, pool_depth=depth,
                 drop_center_y=depth + 3.0 * dx, drop_radius=dx,
                 h_factor=h_factor, adaptive_dt=False,
                 density_diffusion=density_diffusion)
    cfg.dt = suggested_dt(cfg)
    return pos, mass, cfg


def column_elevation(pos: torch.Tensor, mass: torch.Tensor, width: float,
                     depth: float, rho0: float, k: float,
                     n_bins: int) -> float:
    """Cosine-projected interface amplitude from the column mass of each/bin."""
    x = pos[:, 0]
    bin_width = width / n_bins
    idx = torch.clamp((x / bin_width).floor().long(), 0, n_bins - 1)
    column_mass = torch.zeros(n_bins, dtype=pos.dtype)
    column_mass.index_add_(0, idx, mass)
    height = column_mass / (rho0 * bin_width)
    centres = (torch.arange(n_bins, dtype=pos.dtype) + 0.5) * bin_width
    weight = torch.cos(k * centres)
    return float((2.0 / width) * ((height - depth) * weight * bin_width).sum())


def run_case(wavelength: float, depth: float, dx: float, amplitude: float,
             periods: float, rho0: float = 1000.0, sigma: float = 7.2,
             viscosity: float = 0.0, sound_speed: float = 10.0,
             samples_per_period: int = 120, device: str = "cpu",
             store_field: bool = False, lattice: str = "hex",
             h_factor: float = 1.3, density_diffusion: float = 0.0) -> dict:
    """March one standing capillary wave and return the measured amplitude history."""
    pos, mass, cfg = make_case(wavelength, depth, dx, amplitude, rho0, sigma,
                               viscosity, sound_speed, h_factor=h_factor,
                               lattice=lattice,
                               density_diffusion=density_diffusion)
    device = resolve_device(device)
    pos = pos.to(device=device, dtype=torch.float32)
    mass = mass.to(device=device, dtype=torch.float32)

    k = 2.0 * math.pi / wavelength
    omega_ref = angular_frequency(sigma, k, rho0, depth)
    gamma_ref = damping_rate(viscosity, k)
    period_ref = 2.0 * math.pi / omega_ref
    t_end = periods * period_ref

    dt = float(cfg.dt)
    steps = int(math.ceil(t_end / dt))
    output_every = max(1, int(round(steps / (periods * samples_per_period))))
    n_bins = max(8, int(round((wavelength / 2.0) / (2.0 * dx))))
    top = depth + amplitude + 4.0 * dx

    vel = torch.zeros_like(pos)
    h = cfg.h_factor * cfg.dx
    cache = PairListCache(2.0 * h, cfg.pair_skin_factor * h,
                          fluid_count=pos.shape[0])

    times: list[float] = []
    amps: list[float] = []
    peak_speed = 0.0
    t0 = time.perf_counter()
    elapsed = 0.0
    for step in range(steps + 1):
        pairs, _ = cache.get(pos)
        density, _, acc = rhs(pos, vel, mass, cfg, pairs=pairs, wall=None, dt=dt)
        if not torch.isfinite(acc).all():
            raise FloatingPointError(f"non-finite acceleration at step {step}")
        if step % output_every == 0 or step == steps:
            times.append(elapsed)
            amps.append(column_elevation(pos, mass, wavelength / 2.0, depth,
                                         rho0, k, n_bins))
        vel = vel + dt * acc
        pos = pos + dt * vel
        escaped = apply_boundary(pos, vel, cfg, top=top)
        if escaped:
            raise RuntimeError(
                f"{escaped} particle(s) left the box at step {step}; the case "
                "is not a free-surface standing wave any more")
        peak_speed = max(peak_speed, float(torch.linalg.vector_norm(vel, dim=1).max()))
        elapsed += dt
    wall = time.perf_counter() - t0

    hist = np.asarray(amps, dtype=float)
    t = np.asarray(times, dtype=float)
    fit = fit_damped_oscillation(t, hist, omega_ref)
    field = None
    if store_field:
        speed = torch.linalg.vector_norm(vel, dim=1).cpu().numpy()
        field = {"x": pos[:, 0].cpu().numpy().tolist(),
                 "y": pos[:, 1].cpu().numpy().tolist(),
                 "speed": speed.tolist()}

    return {
        "case": {
            "wavelength_m": wavelength, "depth_m": depth, "dx_m": dx,
            "amplitude_m": amplitude, "amplitude_over_dx": amplitude / dx,
            "points_per_wavelength": wavelength / dx,
            "k_1_per_m": k, "kH": k * depth, "tanh_kH": math.tanh(k * depth),
            "sigma_N_per_m": sigma, "rho0": rho0, "viscosity_m2_per_s": viscosity,
            "sound_speed_m_per_s": sound_speed, "gravity": 0.0,
            "lattice": lattice, "h_factor": h_factor,
            "density_diffusion": density_diffusion,
            "csf_scheme": cfg.csf_scheme, "csf_curvature": cfg.csf_curvature,
            "n_particles": int(pos.shape[0]), "n_bins": n_bins,
            "dt_s": dt, "steps": steps, "output_every": output_every,
            "t_end_s": steps * dt, "periods": periods,
            "initial_amplitude_measured_m": float(hist[0]) if hist.size else None,
        },
        "analytic": {"omega_rad_per_s": omega_ref, "period_s": period_ref,
                     "damping_rate_1_per_s": gamma_ref},
        "measured": {
            "omega_rad_per_s": fit["omega"], "gamma_1_per_s": fit["gamma"],
            "period_s": (2.0 * math.pi / fit["omega"]) if fit["omega"] else None,
            "omega_rel_error_pct": (rel_error_pct(fit["omega"], omega_ref)
                                    if fit["omega"] else None),
            "gamma_rel_error_pct": (rel_error_pct(fit["gamma"], gamma_ref)
                                    if (gamma_ref > 0.0 and fit["gamma"] is not None)
                                    else None),
            "fit_residual_rel": fit["residual_rel"],
            "fit_amplitude_m": fit["amplitude"],
            "peak_speed_m_per_s": peak_speed,
            "wall_clock_s": wall,
        },
        "history": {"t": times, "amplitude_m": amps, "fit": fit["curve"]},
        "field": field,
    }


# --------------------------------------------------------------------------- #
# damped-oscillation fit
# --------------------------------------------------------------------------- #
def _fit_at(t: np.ndarray, a: np.ndarray, omega: float, gamma: float):
    """Linear least squares for ``offset + e^{-gamma t}(c cos + s sin)``."""
    env = np.exp(-gamma * t)
    basis = np.stack([env * np.cos(omega * t), env * np.sin(omega * t),
                      np.ones_like(t)], axis=1)
    coef, *_ = np.linalg.lstsq(basis, a, rcond=None)
    resid = a - basis @ coef
    return coef, float(np.sqrt(np.mean(resid**2)))


def fit_damped_oscillation(t, a, omega_guess: float,
                           omega_span: float = 0.6,
                           gamma_max_rel: float = 1.5) -> dict:
    """Two-stage grid search for ``(omega, gamma)``; returns residual and curve.

    Deterministic, dependency-free and robust to the missing first half period:
    the search is over the two nonlinear parameters only, the amplitudes and the
    offset come from a linear solve.  ``residual_rel`` (RMS residual over the RMS
    of ``a``) is the honesty metric -- a noise-dominated signal gives ~1.
    """
    t = np.asarray(t, dtype=float)
    a = np.asarray(a, dtype=float)
    a_rms = float(np.sqrt(np.mean(a**2))) or 1.0

    best = (math.inf, omega_guess, 0.0, np.zeros(3))
    for gamma in np.linspace(0.0, gamma_max_rel * omega_guess, 41):
        for omega in np.linspace((1.0 - omega_span) * omega_guess,
                                 (1.0 + omega_span) * omega_guess, 161):
            coef, resid = _fit_at(t, a, omega, gamma)
            if resid < best[0]:
                best = (resid, omega, gamma, coef)
    # local refinement around the coarse optimum
    _, w0, g0, _ = best
    dw = omega_span * omega_guess / 40.0
    dg = gamma_max_rel * omega_guess / 20.0
    for gamma in np.linspace(max(0.0, g0 - dg), g0 + dg, 21):
        for omega in np.linspace(w0 - dw, w0 + dw, 41):
            coef, resid = _fit_at(t, a, omega, gamma)
            if resid < best[0]:
                best = (resid, omega, gamma, coef)

    resid, omega, gamma, coef = best
    c, s, offset = coef
    curve = offset + np.exp(-gamma * t) * (c * np.cos(omega * t)
                                           + s * np.sin(omega * t))
    return {"omega": float(omega), "gamma": float(gamma),
            "amplitude": float(math.hypot(c, s)),
            "offset": float(offset),
            "residual_rms": float(resid),
            "residual_rel": float(resid / a_rms),
            "curve": curve.tolist()}


def zero_crossing_period(t, a) -> float | None:
    """Independent estimator: period from interpolated zero crossings of ``a(t)``."""
    t = np.asarray(t, dtype=float)
    a = np.asarray(a, dtype=float) - float(np.mean(a))
    crossings = []
    for i in range(1, a.size):
        if a[i - 1] == 0.0 or a[i - 1] * a[i] < 0.0:
            f = a[i - 1] / (a[i - 1] - a[i])
            crossings.append(t[i - 1] + f * (t[i] - t[i - 1]))
    if len(crossings) < 3:
        return None
    half = np.diff(crossings)
    return float(2.0 * np.mean(half))


# --------------------------------------------------------------------------- #
# case suite
# --------------------------------------------------------------------------- #
CASES = {
    # name:  wavelength, depth, dx,      amplitude/dx, periods, viscosity
    "dispersion": dict(wavelength=0.02, depth=0.02, dx=0.0005, amp_over_dx=0.5,
                       periods=5.0, viscosity=0.0),
    "viscous": dict(wavelength=0.02, depth=0.02, dx=0.0005, amp_over_dx=0.5,
                    periods=4.0, viscosity=2.4e-4),
    "resolution": dict(wavelength=0.02, depth=0.02, dx=0.001, amp_over_dx=0.5,
                       periods=5.0, viscosity=0.0),
    "scaling": dict(wavelength=0.01, depth=0.02, dx=0.0005, amp_over_dx=0.5,
                    periods=5.0, viscosity=0.0),
}


def _summary_row(name: str, rep: dict) -> dict:
    m, an = rep["measured"], rep["analytic"]
    return {
        "case": name,
        "dx_m": rep["case"]["dx_m"],
        "wavelength_m": rep["case"]["wavelength_m"],
        "omega_analytic": an["omega_rad_per_s"],
        "omega_measured": m["omega_rad_per_s"],
        "omega_error_pct": m["omega_rel_error_pct"],
        "omega_within_3pct": abs(m["omega_rel_error_pct"]) < STRICT_PCT,
        "omega_within_15pct": abs(m["omega_rel_error_pct"]) < GATE_PCT,
        "gamma_analytic": an["damping_rate_1_per_s"],
        "gamma_measured": m["gamma_1_per_s"],
        "gamma_error_pct": m["gamma_rel_error_pct"],
        "fit_residual_rel": m["fit_residual_rel"],
        "wall_clock_s": m["wall_clock_s"],
    }


def make_figures(payload: dict, out_dir: Path = HERE) -> Path:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    runs = payload["runs"]
    fig, axes = plt.subplots(2, 2, figsize=(13, 8.4))

    ax = axes[0][0]
    for name, rep in runs.items():
        t = np.asarray(rep["history"]["t"]) / rep["analytic"]["period_s"]
        a = np.asarray(rep["history"]["amplitude_m"]) / rep["case"]["amplitude_m"]
        fit = np.asarray(rep["history"]["fit"]) / rep["case"]["amplitude_m"]
        ax.plot(t, a, ".", ms=3, alpha=0.65, label=f"{name} (measured)")
        ax.plot(t, fit, "-", lw=1.2,
                label=(f"{name}: T={2*math.pi/rep['measured']['omega_rad_per_s']:.4f}s "
                       f"({rep['measured']['omega_rel_error_pct']:+.1f}%)"))
    ax.set_xlabel("t / T_analytic")
    ax.set_ylabel(r"$a(t)/A_0$")
    ax.set_title("(a) standing capillary wave: measured vs damped-oscillation fit")
    ax.legend(fontsize=6)
    ax.grid(alpha=0.3)

    ax = axes[0][1]
    names = list(runs)
    for i, name in enumerate(names):
        rep = runs[name]
        om, an = rep["measured"]["omega_rad_per_s"], rep["analytic"]["omega_rad_per_s"]
        ax.bar(i, om, color="steelblue", width=0.6, label="measured" if i == 0 else None)
        ax.plot([i - 0.35, i + 0.35], [an, an], "k--", lw=1.4,
                label="Lamb $\\omega=\\sqrt{\\sigma k^3/\\rho\\,\\tanh kH}$"
                if i == 0 else None)
        ax.text(i, max(om, an) * 1.02, f"{rep['measured']['omega_rel_error_pct']:+.1f}%",
                ha="center", fontsize=8)
    ax.set_xticks(range(len(names)))
    ax.set_xticklabels(names, fontsize=8)
    ax.set_ylabel(r"$\omega$ [rad/s]")
    ax.set_title("(b) angular frequency vs analytic")
    ax.legend(fontsize=7)
    ax.grid(alpha=0.3, axis="y")

    ax = axes[1][0]
    vis = runs.get("viscous")
    if vis is not None:
        t = np.asarray(vis["history"]["t"])
        a = np.asarray(vis["history"]["amplitude_m"])
        ax.semilogy(t, np.abs(a), ".", ms=3, label="|a(t)| measured")
        g = vis["measured"]["gamma_1_per_s"]
        ax.semilogy(t, vis["case"]["amplitude_m"] * np.exp(-g * t), "-",
                    label=f"fit $e^{{-\\gamma t}}$, $\\gamma$={g:.1f} 1/s")
        gth = vis["analytic"]["damping_rate_1_per_s"]
        ax.semilogy(t, vis["case"]["amplitude_m"] * np.exp(-gth * t), "k--",
                    label=f"Lamb $2\\nu k^2$={gth:.1f} 1/s")
        ax.set_xlabel("t [s]")
        ax.set_ylabel(r"$|a|$ [m]")
        ax.set_title("(c) viscous damping of the standing wave")
        ax.legend(fontsize=7)
        ax.grid(alpha=0.3, which="both")
    else:
        ax.axis("off")

    ax = axes[1][1]
    disp = [runs[n] for n in runs if runs[n]["case"]["viscosity_m2_per_s"] == 0.0]
    if disp:
        ks = np.array([r["case"]["k_1_per_m"] for r in disp])
        oms = np.array([r["measured"]["omega_rad_per_s"] for r in disp])
        oth = np.array([r["analytic"]["omega_rad_per_s"] for r in disp])
        order = np.argsort(ks)
        ax.loglog(ks[order], oms[order], "o", label="measured")
        ax.loglog(ks[order], oth[order], "k--", label=r"$\omega\propto k^{3/2}$")
        if len(disp) > 1:
            for i, r in enumerate(disp):
                ax.annotate(f"dx={r['case']['dx_m']*1e3:.2f} mm", (ks[i], oms[i]),
                            fontsize=7, xytext=(4, -10), textcoords="offset points")
        ax.set_xlabel(r"$k$ [1/m]")
        ax.set_ylabel(r"$\omega$ [rad/s]")
        ax.set_title("(d) wavenumber scaling (includes resolution variation)")
        ax.legend(fontsize=7)
        ax.grid(alpha=0.3, which="both")

    fig.tight_layout()
    path = out_dir / "capillary_wave_validation.png"
    fig.savefig(path, dpi=140)
    print(f"figure -> {path}")
    return path


def make_field_figure(payload: dict, out_dir: Path = HERE) -> Path | None:
    """Field cloud: speed of the wave at the last stored instant."""
    reps = [r for r in payload["runs"].values() if r.get("field")]
    if not reps:
        return None
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    rep = reps[0]
    f = rep["field"]
    fig, ax = plt.subplots(figsize=(6.4, 5.0))
    sc = ax.scatter(f["x"], f["y"], c=f["speed"], s=2.2, cmap="viridis")
    ax.set_aspect("equal")
    ax.set_xlabel("x [m]")
    ax.set_ylabel("y [m]")
    lam = rep["case"]["wavelength_m"]
    ax.set_title(f"capillary standing wave, $\\lambda$={lam*1e3:.0f} mm, "
                 f"dx={rep['case']['dx_m']*1e3:.2f} mm")
    fig.colorbar(sc, ax=ax, label=r"$|v|$ [m/s]")
    fig.tight_layout()
    path = out_dir / "capillary_wave_field.png"
    fig.savefig(path, dpi=140)
    print(f"field figure -> {path}")
    return path


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--case", default="all",
                    help="all or one of: " + ", ".join(CASES))
    ap.add_argument("--periods", type=float, default=None,
                    help="override the number of periods simulated")
    ap.add_argument("--gate", type=float, default=GATE_PCT)
    ap.add_argument("--out", default=str(HERE))
    a = ap.parse_args()

    names = list(CASES) if a.case == "all" else [a.case]
    unknown = [n for n in names if n not in CASES]
    if unknown:
        ap.error(f"unknown case(s) {unknown}; choose from {list(CASES)}")

    runs: dict[str, dict] = {}
    rows = []
    failures = []
    for name in names:
        spec = dict(CASES[name])
        if a.periods is not None:
            spec["periods"] = a.periods
        amp_over_dx = spec.pop("amp_over_dx")
        dx = spec["dx"]
        rep = run_case(amplitude=amp_over_dx * dx, store_field=True, **spec)
        runs[name] = rep
        row = _summary_row(name, rep)
        cycles = zero_crossing_period(rep["history"]["t"], rep["history"]["amplitude_m"])
        row["period_zero_crossing_s"] = cycles
        rows.append(row)
        print(f"{name:11s} dx={row['dx_m']*1e3:.2f} mm  N={rep['case']['n_particles']:5d}  "
              f"omega={row['omega_measured']:8.2f} / {row['omega_analytic']:8.2f} rad/s "
              f"({row['omega_error_pct']:+6.1f}%)  "
              f"gamma={row['gamma_measured']:+7.2f} / {row['gamma_analytic']:+7.2f} 1/s  "
              f"resid={row['fit_residual_rel']:.3f}")
        for key in ("omega_error_pct", "gamma_error_pct"):
            val = row[key]
            if val is not None and abs(val) > a.gate:
                failures.append(f"{name}:{key}={val:+.1f}%")

    payload = {
        "benchmark": "capillary_wave_dispersion_damping",
        "references": {
            "dispersion": "Lamb (1932) Art. 267/348: omega^2=(g k + sigma k^3/rho) tanh(kH)",
            "damping": "Lamb (1932) Art. 348: gamma = 2 nu k^2",
        },
        "gates_pct": {"platform_15": GATE_PCT, "strict_3": STRICT_PCT,
                      "applied": a.gate},
        "rows": rows,
        "runs": runs,
        "errors": failures,
        "pass": not failures,
    }
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    (out / "validation_report.json").write_text(json.dumps(payload, indent=2))
    make_figures(payload, out)
    make_field_figure(payload, out)
    print(f"report -> {out / 'validation_report.json'}")
    if failures:
        print("GATE FAILURES: " + "; ".join(failures))
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())