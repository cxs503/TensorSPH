"""Oscillating drop: dynamic validation of the CSF surface-tension force.

.. note::
   Preserved verbatim from branch ``agent/m2919`` (PR #87) when it was merged
   into ``main`` -- mission m3820.  This file used to be
   ``benchmarks/oscillating_droplet/run_lamb.py``; ``main`` independently grew a
   *different* harness for the same benchmark at that path, so both are kept
   side by side instead of one silently overwriting the other.  Companion
   artifacts: ``validation_report_lamb.json``, ``oscillating_droplet_lamb*.png``,
   ``tests/test_oscillating_droplet_lamb.py``,
   ``docs/verification_oscillating_droplet_lamb.md``.

What is missing in this repository
----------------------------------
Every surface-tension number TensorSPH publishes is *static*: the Young-Laplace
pressure jump, the cut-force identity ``F_cut = 2 sigma`` and the curvature of a
circular interface (``docs/verification_surface_tension_{laplace,consistent,
curvature}.md``).  Nothing tested the force on a *moving* interface, so an error
in the direction/phase of the force, or in the effective inertia of the CSF
acceleration, would not have been noticed.  This benchmark supplies the dynamic
reference: a slightly deformed drop is a harmonic oscillator with a closed-form
frequency.

Analytic reference (Lamb 1932, Hydrodynamics, Art. 275; Rayleigh 1879)
----------------------------------------------------------------------
A 2-D inviscid drop of radius ``R``, density ``rho``, surface tension ``sigma``
oscillates in the mode ``r(theta) = R [1 + eps cos(n theta)]`` at

    omega_n^2 = n (n^2 - 1) sigma / (rho R^3) ,        n >= 2

so ``n = 2`` gives ``6 sigma / (rho R^3)``.

**Do not use the 3-D Rayleigh formula here.**  A 3-D drop oscillates at
``omega^2 = n (n-1)(n+2) sigma / (rho R^3)``, which at ``n = 2`` is 8 instead of
6 -- 15 % higher, i.e. exactly the size of the acceptance gate.  The two branches
agree only for large ``n`` (both ``~ n^3``), which is why the harness scores
``n = 2`` and ``n = 3`` separately.

Derivation check (paper, not code): the curvature perturbation of
``r = R + a cos n theta`` is ``delta kappa = a (n^2 - 1) cos n theta / R^2``
(it vanishes for ``n = 1``, as it must for a translation); with the interior
potential flow ``phi = C r^n cos n theta`` the unsteady Bernoulli pressure at the
interface is ``-rho (R/n) ddot{a} cos n theta``.  Matching the two yields the
relation above.

Measurement
-----------
The mode amplitude is read from the inertia tensor,

    M(t) = sum_i m_i (x_i^2 - y_i^2) / sum_i m_i r_i^2  ->  (3/2) eps cos(omega t)

which needs no surface-particle detection and -- unlike a "topmost particle"
estimator -- is not quantised by the lattice.  ``(omega, gamma)`` come from a
least-squares fit of ``M(t) = offset + exp(-gamma t) [c cos + s sin]`` (two-stage
grid search, no scipy); the fit reports its relative residual so a noise-
dominated run is visible instead of being scored as a number.

Usage::

    PYTHONPATH=src python benchmarks/oscillating_droplet/run_lamb.py --case all
    PYTHONPATH=src python benchmarks/oscillating_droplet/run_lamb.py --case mode2 --periods 2

Exit status is non-zero when a measured frequency misses its analytic value by
more than ``--gate`` (default 15 %); the 3 % tier is recorded in every row.
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
    suggested_dt,
)

HERE = Path(__file__).resolve().parent
torch.set_num_threads(2)

GATE_PCT = 15.0
STRICT_PCT = 3.0


# --------------------------------------------------------------------------- #
# analytic references
# --------------------------------------------------------------------------- #
def drop_frequency_2d(sigma: float, rho: float, radius: float, mode: int) -> float:
    """``omega_n^2 = n (n^2 - 1) sigma / (rho R^3)`` (Lamb 1932, Art. 275, 2-D)."""
    if mode < 2:
        raise ValueError("mode n = 1 is a translation, it has no restoring force")
    return math.sqrt(mode * (mode * mode - 1) * sigma / (rho * radius**3))


def drop_frequency_3d(sigma: float, rho: float, radius: float, mode: int) -> float:
    """3-D Rayleigh frequency -- kept only to document the formula trap above."""
    return math.sqrt(mode * (mode - 1) * (mode + 2) * sigma / (rho * radius**3))


def rel_error_pct(measured: float, exact: float) -> float:
    return (measured - exact) / exact * 100.0


# --------------------------------------------------------------------------- #
# geometry
# --------------------------------------------------------------------------- #
def lobed_droplet(radius: float, dx: float, rho0: float, eps: float,
                  mode: int) -> tuple[torch.Tensor, torch.Tensor]:
    """Hexagonal drop with ``r(theta) = R (1 + eps cos n theta)`` (centred at 0)."""
    dy = dx * math.sqrt(3.0) / 2.0
    mass = rho0 * dx * dy
    reach = radius * (1.0 + abs(eps))
    pts = []
    j = 0
    y = -reach
    while y <= reach:
        offset = 0.5 * dx if (j % 2) else 0.0
        x = -reach + offset
        while x <= reach:
            r = math.hypot(x, y)
            if r > 1e-12:
                theta = math.atan2(y, x)
                if r <= radius * (1.0 + eps * math.cos(mode * theta)):
                    pts.append((x, y))
            x += dx
        y += dy
        j += 1
    pos = torch.tensor(pts, dtype=torch.float64)
    return pos, torch.full((pos.shape[0],), mass, dtype=torch.float64)


def mode_amplitude(pos: torch.Tensor, mass: torch.Tensor,
                   centre: tuple[float, float] = (0.0, 0.0)) -> float:
    """``sum m (x^2 - y^2) / sum m r^2`` about ``centre``: n = 2 inertia mode."""
    rel = pos - torch.tensor(centre, dtype=pos.dtype, device=pos.device)
    r2 = (rel**2).sum(dim=1)
    quad = mass * (rel[:, 0] ** 2 - rel[:, 1] ** 2)
    return float(quad.sum() / (mass * r2).sum())


# --------------------------------------------------------------------------- #
# damped-oscillation fit
# --------------------------------------------------------------------------- #
def _fit_at(t: np.ndarray, a: np.ndarray, omega: float, gamma: float):
    env = np.exp(-gamma * t)
    basis = np.stack([env * np.cos(omega * t), env * np.sin(omega * t),
                      np.ones_like(t)], axis=1)
    coef, *_ = np.linalg.lstsq(basis, a, rcond=None)
    resid = a - basis @ coef
    return coef, float(np.sqrt(np.mean(resid**2)))


def fit_damped_oscillation(t, a, omega_guess: float, omega_span: float = 0.6,
                           gamma_max_rel: float = 1.5) -> dict:
    """Deterministic two-stage grid search for ``(omega, gamma)`` of a damped cosine.

    Only the two nonlinear parameters are searched; amplitudes and offset come
    from a linear solve.  ``residual_rel`` is the RMS residual over the RMS of
    ``a`` -- a noise-dominated signal gives ~1, a clean one ~1e-2.
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
    _, w0, g0, _ = best
    for _ in range(2):  # two refinement passes: 8x finer each time
        dw = omega_span * omega_guess / 40.0
        dg = gamma_max_rel * omega_guess / 20.0
        for gamma in np.linspace(max(0.0, g0 - dg), g0 + dg, 21):
            for omega in np.linspace(w0 - dw, w0 + dw, 41):
                coef, resid = _fit_at(t, a, omega, gamma)
                if resid < best[0]:
                    best = (resid, omega, gamma, coef)
        _, w0, g0, _ = best
        omega_span /= 8.0
        gamma_max_rel /= 8.0

    resid, omega, gamma, coef = best
    c, s, offset = coef
    curve = offset + np.exp(-gamma * t) * (c * np.cos(omega * t)
                                          + s * np.sin(omega * t))
    return {"omega": float(omega), "gamma": float(gamma),
            "amplitude": float(math.hypot(c, s)), "offset": float(offset),
            "residual_rms": float(resid),
            "residual_rel": float(resid / a_rms),
            "curve": curve.tolist()}


def zero_crossing_period(t, a) -> float | None:
    """Independent estimator: period from interpolated zero crossings."""
    t = np.asarray(t, dtype=float)
    a = np.asarray(a, dtype=float) - float(np.mean(a))
    crossings = []
    for i in range(1, a.size):
        if a[i - 1] * a[i] < 0.0:
            f = a[i - 1] / (a[i - 1] - a[i])
            crossings.append(t[i - 1] + f * (t[i] - t[i - 1]))
    if len(crossings) < 3:
        return None
    return float(2.0 * np.mean(np.diff(crossings)))


# --------------------------------------------------------------------------- #
# simulation
# --------------------------------------------------------------------------- #
def run_case(radius: float = 0.02, dx: float = 0.001, mode: int = 2,
             eps: float = 0.1, periods: float = 3.0, sigma: float = 7.2,
             rho0: float = 1000.0, viscosity: float = 1e-3,
             sound_speed: float = 10.0, samples_per_period: int = 120,
             device: str = "cpu", store_field: bool = False) -> dict:
    pos, mass = lobed_droplet(radius, dx, rho0, eps, mode)
    device = resolve_device(device)
    dt = suggested_dt(Config(dx=dx, rho0=rho0, sound_speed=sound_speed,
                             h_factor=1.3, max_courant=0.25))
    cfg = Config(dx=dx, rho0=rho0, sound_speed=sound_speed, gamma=7.0,
                 gravity=0.0, viscosity=viscosity,
                 surface_tension_coefficient=sigma, boundary="clamp",
                 pool_width=4.0 * radius, pool_depth=4.0 * radius,
                 drop_center_y=2.0 * radius, drop_radius=radius,
                 adaptive_dt=False, density_diffusion=0.0, dt=dt)
    centre = (2.0 * radius, 2.0 * radius)   # drop centred inside the clamp box
    pos = pos + torch.tensor(centre, dtype=pos.dtype)
    pos = pos.to(device=device, dtype=torch.float32)
    mass = mass.to(device=device, dtype=torch.float32)
    h = cfg.h_factor * cfg.dx

    omega_ref = drop_frequency_2d(sigma, rho0, radius, mode) if sigma > 0 else 0.0
    # the sigma = 0 negative control is still marched for the same wall time
    omega_clock = (drop_frequency_2d(7.2, rho0, radius, mode)
                   if sigma <= 0 else omega_ref)
    period_ref = 2.0 * math.pi / omega_clock
    t_end = periods * period_ref
    steps = int(math.ceil(t_end / dt))
    output_every = max(1, int(round(steps / (periods * samples_per_period))))
    top = 2.0 * radius + 4.0 * radius  # generous: the box must not clip the drop

    vel = torch.zeros_like(pos)
    from tensorsph.neighbor_search import PairListCache
    cache = PairListCache(2.0 * h, cfg.pair_skin_factor * h,
                          fluid_count=pos.shape[0])

    times: list[float] = []
    amps: list[float] = []
    peak_speed = 0.0
    elapsed = 0.0
    stability = "stable"
    t0 = time.perf_counter()
    for step in range(steps + 1):
        pairs, _ = cache.get(pos)
        density, _, acc = rhs(pos, vel, mass, cfg, pairs=pairs, wall=None, dt=dt)
        if not torch.isfinite(acc).all():
            raise FloatingPointError(f"non-finite acceleration at step {step}")
        if step % output_every == 0 or step == steps:
            times.append(elapsed)
            amps.append(mode_amplitude(pos, mass, centre))
        vel = vel + dt * acc
        pos = pos + dt * vel
        if apply_boundary(pos, vel, cfg, top=top):
            # A particle leaving the box means the free-surface discretisation
            # is no longer a drop oscillation.  Do not hide it: stop, keep the
            # samples collected so far, and flag the run as unstable.
            stability = f"particle left the box at step {step} ({elapsed/period_ref:.2f} T)"
            break
        peak_speed = max(peak_speed, float(torch.linalg.vector_norm(vel, dim=1).max()))
        elapsed += dt
    wall = time.perf_counter() - t0

    hist = np.asarray(amps, dtype=float)
    if sigma > 0.0:
        fit = fit_damped_oscillation(times, hist, omega_ref)
    else:  # negative control: no restoring force, the fit must not invent one
        fit = fit_damped_oscillation(times, hist, omega_ref or 100.0)
    field = None
    if store_field:
        field = {"x": pos[:, 0].cpu().numpy().tolist(),
                 "y": pos[:, 1].cpu().numpy().tolist(),
                 "speed": torch.linalg.vector_norm(vel, dim=1).cpu().numpy().tolist()}

    measured = {
        "omega_rad_per_s": fit["omega"],
        "gamma_1_per_s": fit["gamma"],
        "period_s": 2.0 * math.pi / fit["omega"] if fit["omega"] else None,
        "fit_amplitude": fit["amplitude"],
        "fit_residual_rel": fit["residual_rel"],
        "initial_mode_m": float(hist[0]) if hist.size else None,
        "mode_swing_m": float(hist.max() - hist.min()) if hist.size else None,
        "peak_speed_m_per_s": peak_speed,
        "wall_clock_s": wall,
        "stability": stability,
    }
    measured["omega_rel_error_pct"] = (rel_error_pct(fit["omega"], omega_ref)
                                       if omega_ref else None)
    return {
        "case": {"radius_m": radius, "dx_m": dx, "radius_over_h": radius / h,
                 "mode_n": mode, "eps": eps, "sigma_N_per_m": sigma,
                 "rho0": rho0, "viscosity_m2_per_s": viscosity,
                 "sound_speed_m_per_s": sound_speed, "gravity": 0.0,
                 "csf_scheme": cfg.csf_scheme, "csf_curvature": cfg.csf_curvature,
                 "n_particles": int(pos.shape[0]), "dt_s": dt, "steps": steps,
                 "output_every": output_every, "t_end_s": steps * dt,
                 "periods": periods},
        "analytic": {"omega_rad_per_s": omega_ref, "period_s": period_ref,
                     "sigma_scored": sigma,
                     "omega_3d_wrong_rad_per_s":
                         drop_frequency_3d(sigma, rho0, radius, mode)
                         if sigma > 0 else None},
        "measured": measured,
        "history": {"t": times, "mode": amps, "fit": fit["curve"]},
        "field": field,
    }


CASES = {
    # name        radius  dx      mode eps   periods sigma
    "mode2": dict(radius=0.02, dx=0.0015, mode=2, eps=0.05, periods=3.0),
    "mode3": dict(radius=0.02, dx=0.0015, mode=3, eps=0.05, periods=3.0),
    "coarse": dict(radius=0.02, dx=0.0020, mode=2, eps=0.05, periods=3.0),
    "no_sigma": dict(radius=0.02, dx=0.0015, mode=2, eps=0.05, periods=1.0,
                     sigma=0.0),
}


def summary_row(name: str, rep: dict) -> dict:
    m, an = rep["measured"], rep["analytic"]
    err = m["omega_rel_error_pct"]
    return {
        "case": name, "mode_n": rep["case"]["mode_n"], "dx_m": rep["case"]["dx_m"],
        "radius_over_h": rep["case"]["radius_over_h"],
        "omega_analytic": an["omega_rad_per_s"], "omega_measured": m["omega_rad_per_s"],
        "omega_error_pct": err,
        "omega_within_3pct": (abs(err) < STRICT_PCT) if err is not None else None,
        "omega_within_15pct": (abs(err) < GATE_PCT) if err is not None else None,
        "gamma_measured": m["gamma_1_per_s"],
        "fit_residual_rel": m["fit_residual_rel"],
        "stability": m["stability"],
        "n_particles": rep["case"]["n_particles"],
        "steps": rep["case"]["steps"], "wall_clock_s": m["wall_clock_s"],
    }


def make_figures(payload: dict, out_dir: Path = HERE) -> list[Path]:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    runs = payload["runs"]
    paths = []
    fig, axes = plt.subplots(1, 3, figsize=(16, 4.4))
    ax = axes[0]
    for name, rep in runs.items():
        per = rep["analytic"]["period_s"] or 1.0
        t = np.asarray(rep["history"]["t"]) / per
        a = np.asarray(rep["history"]["mode"])
        fit = np.asarray(rep["history"]["fit"])
        ax.plot(t, a / rep["case"]["eps"], ".", ms=3, alpha=0.6, label=f"{name}")
        ax.plot(t, fit / rep["case"]["eps"], "-", lw=1.1)
    ax.set_xlabel(r"$t / T_{analytic}$")
    ax.set_ylabel(r"$M(t)/\varepsilon$ (mode-2 inertia amplitude)")
    ax.set_title("(a) mode amplitude: data (dots) and damped-cosine fit (line)")
    ax.legend(fontsize=7)
    ax.grid(alpha=0.3)

    ax = axes[1]
    names = [n for n in runs]
    for i, name in enumerate(names):
        rep = runs[name]
        err = rep["measured"]["omega_rel_error_pct"]
        ax.bar(i, rep["measured"]["omega_rad_per_s"], color="steelblue", width=0.55)
        if rep["analytic"]["omega_rad_per_s"]:
            ax.plot([i - 0.34, i + 0.34],
                    [rep["analytic"]["omega_rad_per_s"]] * 2, "k--", lw=1.5)
            ax.text(i, rep["analytic"]["omega_rad_per_s"] * 1.03,
                    f"{err:+.1f}%", ha="center", fontsize=8)
    ax.set_xticks(range(len(names)))
    ax.set_xticklabels(names, fontsize=8)
    ax.set_ylabel(r"$\omega$ [rad/s]")
    ax.set_title(r"(b) $\omega$ vs Lamb 2-D $\omega_n=\sqrt{n(n^2-1)\sigma/\rho R^3}$")
    ax.grid(alpha=0.3, axis="y")

    ax = axes[2]
    reps = [r for r in runs.values() if r.get("field")]
    if reps:
        f = reps[0]["field"]
        sc = ax.scatter(f["x"], f["y"], c=f["speed"], s=2.0, cmap="viridis")
        fig.colorbar(sc, ax=ax, label=r"$|v|$ [m/s]")
        ax.set_aspect("equal")
        ax.set_title("(c) velocity field, final instant")
    else:
        ax.axis("off")
    fig.tight_layout()
    path = out_dir / "oscillating_droplet_lamb.png"
    fig.savefig(path, dpi=140)
    paths.append(path)

    fig2, ax2 = plt.subplots(figsize=(6.4, 3.6))
    for name, rep in runs.items():
        t = np.asarray(rep["history"]["t"])
        a = np.abs(np.asarray(rep["history"]["mode"]))
        ax2.semilogy(t, np.maximum(a, 1e-9), ".", ms=3, label=name)
    ax2.set_xlabel("t [s]")
    ax2.set_ylabel(r"$|M(t)|$")
    ax2.set_title("mode decay (measured damping, no analytic claim)")
    ax2.legend(fontsize=7)
    ax2.grid(alpha=0.3, which="both")
    fig2.tight_layout()
    path2 = out_dir / "oscillating_droplet_lamb_decay.png"
    fig2.savefig(path2, dpi=140)
    paths.append(path2)
    for p in paths:
        print(f"figure -> {p}")
    return paths


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--case", default="all",
                    help="all or one of: " + ", ".join(CASES))
    ap.add_argument("--periods", type=float, default=None)
    ap.add_argument("--gate", type=float, default=GATE_PCT)
    ap.add_argument("--out", default=str(HERE))
    a = ap.parse_args()

    names = list(CASES) if a.case == "all" else [a.case]
    unknown = [n for n in names if n not in CASES]
    if unknown:
        ap.error(f"unknown case(s) {unknown}; choose from {list(CASES)}")

    rows, runs, failures = [], {}, []
    for name in names:
        spec = dict(CASES[name])
        if a.periods is not None:
            spec["periods"] = a.periods
        rep = run_case(store_field=True, **spec)
        runs[name] = rep
        row = summary_row(name, rep)
        row["period_zero_crossing_s"] = zero_crossing_period(
            rep["history"]["t"], rep["history"]["mode"])
        rows.append(row)
        err_pct = row["omega_error_pct"]
        err_txt = "None" if err_pct is None else round(err_pct, 1)
        print(f"{name:9s} n={row['mode_n']} dx={row['dx_m']*1e3:.2f} mm "
              f"N={row['n_particles']:5d} omega={row['omega_measured']:8.2f}/"
              f"{row['omega_analytic']:8.2f} rad/s "
              f"({err_txt}%) "
              f"gamma={row['gamma_measured']:.2f} 1/s resid={row['fit_residual_rel']:.3f} "
              f"t={row['wall_clock_s']:.0f}s")
        if row["omega_error_pct"] is not None and abs(row["omega_error_pct"]) > a.gate:
            failures.append(f"{name}:omega={row['omega_error_pct']:+.1f}%")

    payload = {
        "benchmark": "oscillating_droplet_dynamic_surface_tension",
        "references": {
            "2d_drop": "Lamb (1932) Art. 275 / Rayleigh (1879): "
                       "omega_n^2 = n(n^2-1) sigma/(rho R^3)",
            "3d_drop_wrong_here": "n(n-1)(n+2) sigma/(rho R^3) -- 15% high at n=2",
        },
        "gates_pct": {"platform_15": GATE_PCT, "strict_3": STRICT_PCT,
                      "applied": a.gate},
        "rows": rows, "runs": runs, "errors": failures, "pass": not failures,
    }
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    (out / "validation_report_lamb.json").write_text(json.dumps(payload, indent=2))
    make_figures(payload, out)
    print(f"report -> {out / 'validation_report_lamb.json'}")
    if failures:
        print("GATE FAILURES: " + "; ".join(failures))
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
