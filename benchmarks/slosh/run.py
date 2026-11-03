"""B5 -- standing gravity-wave (sloshing mode) dispersion verification.

Every other benchmark in this repository checks either a static balance
(hydrostatics, Young-Laplace), a steady flow (Ritter) or a transient in which no
analytic reference exists (the droplet impact, whose numbers cannot be scored).
None of them measures the one thing a free-surface solver has to get right for
seakeeping work: **does the surface oscillate at the analytic frequency?**

For a rectangular tank of length ``L`` and still-water depth ``H``, linear
potential-flow theory gives the natural frequencies of the standing modes

    omega_n^2 = g k_n tanh(k_n H),        k_n = n pi / L,   n = 1, 2, ...

(Lamb 1932, *Hydrodynamics*, Art. 231; Abramson 1966, NASA SP-106 *The dynamic
behavior of liquids in moving containers*, eq. 2-9; Faltinsen & Timokha 2009,
*Sloshing*, eq. 4.3).  The initial condition is the corresponding mode shape,
``eta_0(x) = A cos(k_n x)``, with the mass of every column taken from the
analytic hydrostatic profile of that column, so the run starts from a discrete
equilibrium of the perturbed free surface rather than from free fall.

Measurement protocol
--------------------
Two independent estimators of the free-surface elevation are computed on every
sample, and their disagreement is reported as the measurement uncertainty:

1. **Column volume (primary).**  Each particle contributes its own volume and a
   one-dimensional Gaussian in ``x``, so the vertical direction is integrated
   exactly instead of being binned:

       h(x_p) = sum_j V_j G_sigma(x_p - x_j),      V_j = m_j / rho_j,
       eta(x_p) = h(x_p) - H.

   ``sigma = 1.2 dx`` smooths away the particle-scale noise; the resulting
   attenuation of a mode of wavenumber ``k`` is ``exp(-(sigma k)^2 / 2)`` and is
   divided out, which is 0.4 % for mode 1 of this tank.  Reading the elevation
   as ``max(y)`` in a bin would instead quantise it to ``dx``, which is a tenth
   of the amplitude used here.

2. **Colour level set (cross-check).**  ``c(x, y) = sum_j V_j W(|x - x_j|)``
   with the cubic-spline kernel; the surface is the ``c = 1/2`` contour,
   interpolated in ``y`` inside a per-probe window taken from estimator 1.

The modal coordinate is the projection onto the mode shape,

    q_n(t) = (2 / L) * int_0^L eta(x, t) cos(k_n x) dx,

which is exactly the quantity the analytic solution predicts to oscillate
sinusoidally at ``omega_n``.

The frequency is measured by a least-squares frequency scan -- ``q(t)`` is fitted
with ``a cos(omega t) + b sin(omega t) + c`` on a fine grid of ``omega`` and the
residual minimiser is reported -- not by a raw FFT peak, which the window length
limits to ``1/T`` and which would be coarser than the acceptance band.  The
scan is calibrated on synthetic sinusoids by
``tests/test_slosh.py::test_frequency_scan_recovers_known_signals``.

Usage::

    PYTHONPATH=src python benchmarks/slosh/run.py              # full ladder
    PYTHONPATH=src python benchmarks/slosh/run.py --quick      # one short run
    PYTHONPATH=src python benchmarks/slosh/run.py --dx 0.005 --modes 1

Output: ``results.json`` next to this file, plus a figure.  Exit status is
non-zero when the acceptance gate fails, so the script can be used as a gate in
the same way as the other benchmarks.
"""
from __future__ import annotations

import argparse
import json
import math
import time
from pathlib import Path

import torch

from tensorsph.droplet_impact import (
    Config,
    apply_boundary,
    cubic_spline_kernel,
    hydrostatic_density,
    rhs,
    wall_lattice,
)
from tensorsph.neighbor_search import PairListCache

HERE = Path(__file__).resolve().parent
TOL = 0.15          # platform acceptance bar: 15 % on every key quantity
TARGET = 0.03       # the accuracy this benchmark is aiming for


# --------------------------------------------------------------------------- #
# analytic reference
# --------------------------------------------------------------------------- #
def analytic_omega(mode: int, length: float, depth: float,
                   gravity: float = 9.81) -> float:
    """``omega^2 = g k tanh(k H)``, ``k = n pi / L``."""
    k = mode * math.pi / length
    return math.sqrt(gravity * k * math.tanh(k * depth))


def nonlinear_shift(k: float, amplitude: float, depth: float) -> float:
    """Second-order (Stokes) correction to the frequency, relative.

    A finite-amplitude standing wave is not exactly linear; the leading
    correction for a gravity wave is of order ``(k A)^2`` (Lamb 1932 Art. 250;
    Faltinsen & Timokha 2009 sec. 5).  The coefficient is not universal -- it
    depends on depth and mode -- so this returns the *scale* of the correction,
    ``(k A)^2 / 8``, and the report quotes it rather than correcting the
    measurement with a fitted constant.
    """
    return (k * amplitude) ** 2 / 8.0


# --------------------------------------------------------------------------- #
# initial condition
# --------------------------------------------------------------------------- #
def build_tank(cfg: Config, mode: int, amplitude: float,
               device: torch.device | None = None):
    """Perturbed tank: fluid below ``y = H + A cos(k_n x)``, hydrostatic masses.

    Returns ``(pos, mass, surface_fn)`` where ``surface_fn(x)`` is the analytic
    initial elevation, so the reconstruction can be checked against it without a
    solver step.
    """
    device = torch.device("cpu") if device is None else device
    dx, length, depth = cfg.dx, cfg.pool_width, cfg.pool_depth
    k = mode * math.pi / length

    def surface_fn(x: torch.Tensor) -> torch.Tensor:
        return depth + amplitude * torch.cos(k * x)

    xs = torch.arange(dx * 0.5, length, dx, device=device)
    # The lattice has to reach past the crest; the cut below removes the rest.
    ys = torch.arange(dx * 0.5, depth + amplitude + 2.0 * dx, dx, device=device)
    gx, gy = torch.meshgrid(xs, ys, indexing="xy")
    pos = torch.stack((gx.reshape(-1), gy.reshape(-1)), dim=-1)
    keep = pos[:, 1] < surface_fn(pos[:, 0])
    pos = pos[keep].contiguous()
    # Column-local hydrostatic pressure: a resting column of local depth
    # H + eta(x) is compressed according to its own depth, so every column
    # starts close to its own equilibrium.
    rho = hydrostatic_density(pos[:, 1], cfg, surface=surface_fn(pos[:, 0]))
    mass = (rho * dx * dx).contiguous()
    return pos, mass, surface_fn


# --------------------------------------------------------------------------- #
# surface reconstruction: estimator 1, column volume
# --------------------------------------------------------------------------- #
def probe_xs(cfg: Config, n_probe: int, device=None) -> torch.Tensor:
    """Probe abscissae, kept clear of the side walls by two spacings."""
    dx = cfg.dx
    device = cfg_device(device)
    return torch.linspace(2.0 * dx, cfg.pool_width - 2.0 * dx, n_probe,
                          dtype=torch.float64, device=device)


def cfg_device(device=None) -> torch.device:
    return torch.device("cpu") if device is None else torch.device(device)


def column_volume(pos: torch.Tensor, mass: torch.Tensor, rho: torch.Tensor,
                  cfg: Config, xs: torch.Tensor, smooth_dx: float = 1.2):
    """Local water column volume per unit width, ``h(x) = sum_j V_j G(x - x_j)``.

    Vertical integration is exact (each particle contributes its volume at its
    own abscissa), so the estimator has no binning noise; the only smoothing is
    the 1-D Gaussian of width ``smooth_dx * dx``.  Returns ``(h, sigma)``.
    """
    dx = cfg.dx
    sigma = smooth_dx * dx
    vol = (mass / rho).to(torch.float64)
    px = pos[:, 0].to(torch.float64)
    g = torch.exp(-0.5 * ((xs[:, None] - px[None, :]) / sigma) ** 2)
    g = g / (math.sqrt(2.0 * math.pi) * sigma)
    return g @ vol, sigma


def attenuation(sigma: float, k: float) -> float:
    """Transfer function of the Gaussian smoother at wavenumber ``k``."""
    return math.exp(-0.5 * (sigma * k) ** 2)


def modal_amplitude(xs: torch.Tensor, eta: torch.Tensor, mode: int,
                    length: float) -> float:
    """``q_n = (2/L) int eta cos(k_n x) dx`` over the valid probes."""
    k = mode * math.pi / length
    ok = torch.isfinite(eta)
    if int(ok.sum()) < 4:
        return float("nan")
    x, e = xs[ok].to(torch.float64), eta[ok].to(torch.float64)
    return float(2.0 / length * torch.trapezoid(e * torch.cos(k * x), x))


# --------------------------------------------------------------------------- #
# surface reconstruction: estimator 2, colour level set
# --------------------------------------------------------------------------- #
def level_set_surface(pos: torch.Tensor, mass: torch.Tensor, rho: torch.Tensor,
                      cfg: Config, xs: torch.Tensor,
                      h_estimate: torch.Tensor, n_level: int = 17,
                      window: float = 4.0):
    """``c = 1/2`` contour of the kernel-sum colour, per probe column.

    The vertical window of each probe is centred on the column-volume estimate
    ``h_estimate`` with half-width ``window * dx``, so a stray spray particle
    cannot stretch the grid and coarsen the interpolation.
    """
    dx, h = cfg.dx, cfg.h_factor * cfg.dx
    vol = (mass / rho).to(torch.float64)
    px = pos[:, 0].to(torch.float64)
    py = pos[:, 1].to(torch.float64)
    half = window * dx
    dy = 2.0 * half / (n_level - 1)
    levels = h_estimate[:, None] + torch.linspace(-half, half, n_level,
                                                  dtype=torch.float64,
                                                  device=pos.device)[None, :]
    qx = xs[:, None].expand(levels.shape)
    pts = torch.stack((qx.reshape(-1), levels.reshape(-1)), dim=-1)

    colour = torch.zeros(pts.shape[0], dtype=torch.float64, device=pos.device)
    chunk = 4096
    for start in range(0, pts.shape[0], chunk):
        block = pts[start:start + chunk]
        d = torch.linalg.vector_norm(block[:, None, :] - pos[None, :, :].to(torch.float64),
                                     dim=2)
        support = d < 2.0 * h
        w = torch.where(support, cubic_spline_kernel(d.clamp_max(2.0 * h), h), 
                        torch.zeros_like(d))
        colour[start:start + chunk] = (w * vol[None, :]).sum(dim=1)
    c_grid = colour.reshape(xs.numel(), n_level)

    eta = torch.full((xs.numel(),), float("nan"), dtype=torch.float64,
                     device=pos.device)
    for i in range(xs.numel()):
        column = c_grid[i]
        inside = (column >= 0.5).nonzero().flatten()
        if inside.numel() == 0:
            continue
        last = int(inside.max())
        if last >= n_level - 1:
            continue      # the column never leaves the fluid inside the window
        c_in, c_out = float(column[last]), float(column[last + 1])
        frac = (0.5 - c_in) / (c_out - c_in) if c_out != c_in else 0.0
        eta[i] = float(levels[i, last]) + frac * dy - float(h_estimate[i])
    return eta


# --------------------------------------------------------------------------- #
# frequency estimation
# --------------------------------------------------------------------------- #
def frequency_scan(t: torch.Tensor, q: torch.Tensor, guess: float,
                   span: float = 0.40, n_scan: int = 601) -> dict:
    """Least-squares frequency: fit ``a cos + b sin + c`` on an omega grid."""
    t = t.to(torch.float64)
    q = q.to(torch.float64)
    base = q - q.mean()
    denom = float((base * base).sum())
    omegas = torch.linspace((1.0 - span) * guess, (1.0 + span) * guess, n_scan,
                            dtype=torch.float64)
    best = None
    for omega in omegas:
        basis = torch.stack((torch.cos(omega * t), torch.sin(omega * t),
                             torch.ones_like(t)), dim=1)
        coef, *_ = torch.linalg.lstsq(basis, q)
        resid = q - basis @ coef
        rss = float((resid * resid).sum())
        if best is None or rss < best["rss"]:
            best = {"omega": float(omega), "rss": rss,
                    "amplitude": math.hypot(float(coef[0]), float(coef[1])),
                    "r2": 1.0 - rss / denom if denom > 0 else float("nan")}
    best["period"] = 2.0 * math.pi / best["omega"]
    best["at_scan_edge"] = bool(best["omega"] < float(omegas[1])
                                or best["omega"] > float(omegas[-2]))
    return best


def decay_rate(t: torch.Tensor, q: torch.Tensor) -> float:
    """Envelope damping rate [1/s] from a windowed RMS decay fit."""
    t = t.to(torch.float64)
    q = q.to(torch.float64)
    n = int(t.numel())
    w = max(n // 8, 6)
    times, levels = [], []
    for start in range(0, n - w, w):
        seg = q[start:start + w]
        times.append(float(t[start:start + w].mean()))
        levels.append(math.sqrt(float((seg * seg).mean())))
    if len(times) < 3:
        return float("nan")
    tt = torch.tensor(times, dtype=torch.float64)
    ll = torch.log(torch.tensor(levels, dtype=torch.float64).clamp_min(1e-30))
    a = torch.stack((tt, torch.ones_like(tt)), dim=1)
    coef, *_ = torch.linalg.lstsq(a, ll)
    return float(-coef[0])


# --------------------------------------------------------------------------- #
# one case
# --------------------------------------------------------------------------- #
def run_mode(mode: int = 1, dx: float = 0.01, length: float = 0.4,
             depth: float = 0.2, amplitude_ratio: float = 0.10,
             periods: float = 4.0, skip_periods: float = 0.5,
             sound_speed: float = 20.0, viscosity: float = 0.0,
             n_probe: int = 48, device: str = "cpu",
             max_courant: float = 0.25, samples_per_period: int = 60) -> dict:
    """Integrate one standing mode and measure its frequency."""
    amp = amplitude_ratio * depth
    cfg = Config(dx=dx, pool_width=length, pool_depth=depth,
                 sound_speed=sound_speed, gravity=9.81, viscosity=viscosity,
                 density_diffusion=0.0, wall_friction=0.0,
                 hydrostatic_init=True, device=device, boundary="wall",
                 max_courant=max_courant,
                 drop_center_y=depth + 10.0 * amp)  # unused: no drop here
    cfg.dt = cfg.max_courant * cfg.h_factor * cfg.dx / cfg.sound_speed
    omega_theory = analytic_omega(mode, length, depth, cfg.gravity)
    t_end = periods * 2.0 * math.pi / omega_theory
    steps = max(int(round(t_end / cfg.dt)), 4)
    sample_every = max(int(round((2.0 * math.pi / omega_theory)
                                 / samples_per_period / cfg.dt)), 1)
    top = depth + amp + 6.0 * dx
    k = mode * math.pi / length

    pos, mass, surface_fn = build_tank(cfg, mode, amp, cfg_device(device))
    wall = wall_lattice(cfg, cfg_device(device), top=top)
    vel = torch.zeros_like(pos)
    initial_mass = float(mass.sum())
    xs = probe_xs(cfg, n_probe, device)

    h = cfg.h_factor * cfg.dx
    fluid_count = pos.shape[0]
    cache = (PairListCache(2.0 * h, cfg.pair_skin_factor * h,
                           fluid_count=fluid_count)
             if cfg.pair_skin_factor > 0 else None)
    all_positions = torch.cat((pos, wall), dim=0)

    times, modal, modal_level, elevations = [], [], [], []
    measure_spread = []
    escaped = 0
    started = time.perf_counter()
    rho = mass / (cfg.dx * cfg.dx)
    for step in range(steps):
        if cache is None:
            pairs = None
        else:
            all_positions[:fluid_count] = pos
            pairs, _ = cache.get(all_positions)
        rho, _, acc = rhs(pos, vel, mass, cfg, pairs=pairs, wall=wall)
        if not torch.isfinite(acc).all():
            raise SystemExit(f"ERROR: non-finite acceleration at step {step}")
        vel = vel + cfg.dt * acc
        pos = pos + cfg.dt * vel
        escaped += apply_boundary(pos, vel, cfg, top=top)
        if step % sample_every == 0:
            h_col, sigma = column_volume(pos, mass, rho, cfg, xs)
            eta = (h_col - depth) / attenuation(sigma, k)
            eta_ls = level_set_surface(pos, mass, rho, cfg, xs, h_col)
            times.append((step + 1) * cfg.dt)
            modal.append(modal_amplitude(xs, eta, mode, length))
            modal_level.append(modal_amplitude(xs, eta_ls, mode, length))
            ok = torch.isfinite(eta_ls)
            if int(ok.sum()) >= 4:
                measure_spread.append(float((eta_ls[ok] - eta[ok]).abs().max()))
            if len(elevations) < 8:
                elevations.append({"t": times[-1], "x": xs.tolist(),
                                   "eta": eta.tolist(),
                                   "eta_level_set": eta_ls.tolist()})
    wall_clock = time.perf_counter() - started

    t = torch.tensor(times)
    q = torch.tensor(modal)
    q_level = torch.tensor(modal_level)
    t_skip = skip_periods * 2.0 * math.pi / omega_theory
    use = torch.isfinite(q) & (t >= t_skip)
    t, q = t[use], q[use]
    fit = frequency_scan(t, q, omega_theory)
    use_l = torch.isfinite(q_level) & (torch.tensor(times) >= t_skip)
    fit_level = frequency_scan(torch.tensor(times)[use_l], q_level[use_l],
                               omega_theory)
    rho, pressure, _ = rhs(pos, vel, mass, cfg, pairs=None, wall=wall)
    max_speed = float(torch.linalg.vector_norm(vel, dim=1).max())
    err = (fit["omega"] - omega_theory) / omega_theory
    return {
        "mode": mode,
        "dx_m": dx,
        "h_m": cfg.h_factor * dx,
        "layers_over_depth": depth / dx,
        "length_m": length,
        "depth_m": depth,
        "amplitude_m": amp,
        "amplitude_over_depth": amp / depth,
        "k_m": k,
        "k_times_amplitude": k * amp,
        "nonlinear_shift_scale": nonlinear_shift(k, amp, depth),
        "n_particles": int(pos.shape[0]),
        "n_wall": int(wall.shape[0]),
        "steps": steps,
        "dt_s": cfg.dt,
        "t_end_s": t_end,
        "sample_every": sample_every,
        "sound_speed": sound_speed,
        "viscosity": viscosity,
        "omega_theory": omega_theory,
        "period_theory": 2.0 * math.pi / omega_theory,
        "omega_measured": fit["omega"],
        "period_measured": fit["period"],
        "frequency_error_pct": err * 100.0,
        "fit_r2": fit["r2"],
        "scan_at_edge": fit["at_scan_edge"],
        "fit_amplitude_over_initial": fit["amplitude"] / amp,
        "omega_level_set": fit_level["omega"],
        "frequency_error_level_set_pct": (fit_level["omega"] - omega_theory)
                                          / omega_theory * 100.0,
        "estimator_max_difference_m": (max(measure_spread)
                                       if measure_spread else float("nan")),
        "damping_rate_1_s": decay_rate(t, q),
        "max_speed_m_s": max_speed,
        "mach": max_speed / sound_speed,
        "mass_drift": (float(mass.sum()) - initial_mass) / initial_mass,
        "density_compression": float((rho / cfg.rho0 - 1.0).abs().max()),
        "escaped_particles": int(escaped),
        "pair_rebuilds": (0 if cache is None else cache.rebuilds),
        "within_15pct": abs(err) < TOL,
        "within_3pct": abs(err) < TARGET,
        "wall_clock_s": wall_clock,
        # ``t``/``q`` are the post-skip samples; the level-set series is cut
        # with the same mask so the three arrays stay aligned for plotting.
        "_series": {"t": t.tolist(), "q": q.tolist(),
                    "q_level_set": q_level[use].tolist()},
        "_elevations": elevations,
        "_final_pos": pos.tolist(),
        "_final_pressure": pressure.reshape(-1).tolist(),
        "_initial_surface": {"x": torch.linspace(0, length, 96).tolist(),
                             "eta": surface_fn(torch.linspace(0, length, 96)).tolist()},
    }


# --------------------------------------------------------------------------- #
# figures
# --------------------------------------------------------------------------- #
def make_figures(results: dict, out_png: Path) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    rows = results["cases"]
    fig, axes = plt.subplots(2, 2, figsize=(11.5, 7.8))
    colors = {1: "tab:blue", 2: "tab:red"}

    # (a) modal time series, both estimators, finest resolution
    ax = axes[0, 0]
    for row in rows:
        if row["dx_m"] != results["finest_dx"]:
            continue
        s = row["_series"]
        t = torch.tensor(s["t"])
        ax.plot(t, torch.tensor(s["q"]) * 1e3, color=colors[row["mode"]], lw=1.2,
                label=f"mode {row['mode']} column volume")
        ql = torch.tensor(s["q_level_set"])
        ok = torch.isfinite(ql)
        if bool(ok.any()):
            ax.plot(t[ok], ql[ok] * 1e3, ".", ms=2, color=colors[row["mode"]],
                    alpha=0.45, label=f"mode {row['mode']} level set")
        omega = row["omega_measured"]
        tt = torch.linspace(0, float(t.max()), 400)
        amp = row["fit_amplitude_over_initial"] * row["amplitude_m"]
        ax.plot(tt, amp * torch.cos(omega * tt) * 1e3, "k--", lw=0.8, alpha=0.6)
    ax.axhline(0.0, color="k", lw=0.4)
    ax.set_xlabel("t [s]")
    ax.set_ylabel(r"$q_n(t)$ [mm]")
    ax.set_title("(a) modal coordinate, finest dx (dashed: cos fit)")
    ax.legend(fontsize=6, ncol=2)
    ax.grid(alpha=0.3)

    # (b) measured vs analytic frequency
    ax = axes[0, 1]
    for mode in sorted({r["mode"] for r in rows}):
        sub = sorted((r for r in rows if r["mode"] == mode),
                     key=lambda r: r["dx_m"])
        ax.plot([r["dx_m"] for r in sub], [r["omega_measured"] for r in sub],
                "o-", color=colors[mode], label=f"mode {mode} measured")
        ax.axhline(sub[0]["omega_theory"], color=colors[mode], ls=":",
                   label=f"mode {mode} analytic")
    ax.set_xscale("log")
    ax.set_xlabel("dx [m]")
    ax.set_ylabel(r"$\omega$ [rad/s]")
    ax.set_title(r"(b) $\omega_n^2 = g k_n \tanh(k_n H)$ vs measured")
    ax.legend(fontsize=6)
    ax.grid(alpha=0.3, which="both")

    # (c) frequency error with the acceptance band
    ax = axes[1, 0]
    for mode in sorted({r["mode"] for r in rows}):
        sub = sorted((r for r in rows if r["mode"] == mode),
                     key=lambda r: r["dx_m"])
        ax.plot([r["dx_m"] for r in sub],
                [abs(r["frequency_error_pct"]) for r in sub], "o-",
                color=colors[mode], label=f"mode {mode}")
    ax.axhline(15.0, color="r", ls="--", label="15 % acceptance")
    ax.axhline(3.0, color="g", ls=":", label="3 % target")
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel("dx [m]")
    ax.set_ylabel("|frequency error| [%]")
    ax.set_title("(c) error over the resolution ladder")
    ax.legend(fontsize=7)
    ax.grid(alpha=0.3, which="both")

    # (d) reconstructed free surface vs the initial mode shape
    ax = axes[1, 1]
    cand = [r for r in rows if r["mode"] == 1 and r["_elevations"]]
    if cand:
        row = max(cand, key=lambda r: r["dx_m"])
        snap = row["_elevations"][0]
        ax.plot(snap["x"], [e * 1e3 for e in snap["eta"]], "o-", ms=3,
                label=f"t = {snap['t']:.3f} s, column volume")
        ax.plot(snap["x"], [e * 1e3 for e in snap["eta_level_set"]], "x",
                ms=4, alpha=0.7, label="same instant, level set")
        ini = row["_initial_surface"]
        ax.plot(ini["x"], [e * 1e3 for e in ini["eta"]], "k:", label="initial mode shape")
    ax.set_xlabel("x [m]")
    ax.set_ylabel(r"$\eta$ [mm]")
    ax.set_title("(d) free-surface reconstruction, mode 1")
    ax.legend(fontsize=6)
    ax.grid(alpha=0.3)

    fig.suptitle("TensorSPH B5: standing gravity-wave dispersion in a tank")
    fig.tight_layout()
    fig.savefig(out_png, dpi=110)
    plt.close(fig)


# --------------------------------------------------------------------------- #
# entry point
# --------------------------------------------------------------------------- #
def strip_private(result: dict) -> dict:
    return {k: v for k, v in result.items() if not k.startswith("_")}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dx", type=float, nargs="+", default=[0.010, 0.005])
    parser.add_argument("--modes", type=int, nargs="+", default=[1, 2])
    parser.add_argument("--length", type=float, default=0.4)
    parser.add_argument("--depth", type=float, default=0.2)
    parser.add_argument("--periods", type=float, default=4.0)
    parser.add_argument("--amplitude-ratio", type=float, default=0.10)
    parser.add_argument("--sound-speed", type=float, default=20.0)
    parser.add_argument("--viscosity", type=float, default=0.0)
    parser.add_argument("--n-probe", type=int, default=48)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--threads", type=int, default=1,
                        help="torch CPU threads; small SPH tensors are much "
                             "faster single-threaded (17x measured here)")
    parser.add_argument("--quick", action="store_true",
                        help="one short coarse run (smoke test)")
    parser.add_argument("--out", default=str(HERE / "results.json"))
    args = parser.parse_args()

    torch.set_num_threads(max(args.threads, 1))
    if args.quick:
        args.dx, args.modes, args.periods = [0.02], [1], 1.5

    cases = []
    for dx in args.dx:
        for mode in args.modes:
            case = run_mode(mode=mode, dx=dx, length=args.length,
                            depth=args.depth, periods=args.periods,
                            amplitude_ratio=args.amplitude_ratio,
                            sound_speed=args.sound_speed,
                            viscosity=args.viscosity, n_probe=args.n_probe,
                            device=args.device)
            print(f"mode {case['mode']} dx {case['dx_m']:.4f}: "
                  f"omega {case['omega_measured']:.4f} vs "
                  f"{case['omega_theory']:.4f} rad/s  "
                  f"({case['frequency_error_pct']:+.2f} %)  "
                  f"r2 {case['fit_r2']:.3f}  "
                  f"amp ratio {case['fit_amplitude_over_initial']:.2f}  "
                  f"{case['wall_clock_s']:.1f} s")
            cases.append(case)

    # dispersion check: the measured *ratio* omega_2/omega_1 must follow the
    # analytic ratio; this is insensitive to a common offset of the method.
    ratios = {}
    for dx in args.dx:
        one = next((c for c in cases if c["mode"] == 1 and c["dx_m"] == dx), None)
        two = next((c for c in cases if c["mode"] == 2 and c["dx_m"] == dx), None)
        if one and two:
            meas = two["omega_measured"] / one["omega_measured"]
            theo = two["omega_theory"] / one["omega_theory"]
            ratios[f"{dx:g}"] = {"measured": meas, "analytic": theo,
                                 "error_pct": (meas - theo) / theo * 100.0,
                                 "within_15pct": abs(meas - theo) / theo < TOL}

    key_failures = [c for c in cases if not c["within_15pct"]]
    results = {
        "benchmark": "B5 standing gravity-wave dispersion",
        "reference": "omega^2 = g k tanh(k H); Lamb (1932) Art. 231; "
                     "Abramson (1966) NASA SP-106 eq. 2-9; "
                     "Faltinsen & Timokha (2009) eq. 4.3",
        "tolerance_pct": TOL * 100.0,
        "target_pct": TARGET * 100.0,
        "finest_dx": min(args.dx),
        "config": {"length_m": args.length, "depth_m": args.depth,
                   "periods": args.periods,
                   "amplitude_ratio": args.amplitude_ratio,
                   "sound_speed": args.sound_speed,
                   "viscosity": args.viscosity, "wall_friction": 0.0,
                   "density_diffusion": 0.0, "n_probe": args.n_probe,
                   "threads": args.threads},
        "dispersion_ratio": ratios,
        "gate": {"n_cases": len(cases), "n_failed": len(key_failures),
                 "result": "pass" if not key_failures else "fail",
                 "n_within_3pct": sum(1 for c in cases if c["within_3pct"])},
        "cases": [strip_private(c) for c in cases],
        "series": [{"mode": c["mode"], "dx_m": c["dx_m"],
                    "t": c["_series"]["t"], "q": c["_series"]["q"],
                    "q_level_set": c["_series"]["q_level_set"]}
                   for c in cases],
    }
    out = Path(args.out)
    out.write_text(json.dumps(results, indent=2, sort_keys=True))
    print(f"wrote {out}")

    png = HERE / "slosh.png"
    make_figures({**results, "cases": cases}, png)
    print(f"wrote {png}")

    if key_failures:
        print(f"GATE FAIL: {len(key_failures)}/{len(cases)} cases outside "
              f"{TOL * 100:.0f} %")
        raise SystemExit(1)
    print(f"GATE PASS: {len(cases)}/{len(cases)} cases inside {TOL * 100:.0f} %")


if __name__ == "__main__":
    main()