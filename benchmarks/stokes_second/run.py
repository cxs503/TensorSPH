#!/usr/bin/env python3
"""Oscillating plate (Stokes' second problem) in a 2-D weakly-compressible SPH channel.

Physics
-------
A viscous fluid fills the gap ``0 < y < H`` between two parallel plates.  The
lower plate oscillates tangentially, ``u_wall(t) = U cos(omega t)``, the upper
plate is at rest, and the flow is invariant in ``x``.  The exact solution of the
incompressible Navier-Stokes equations for this geometry is the difference of
two Stokes layers (Landau & Lifshitz, *Fluid Mechanics*, §24; the second image
term enforces the rest condition on the upper plate):

    k = (1 + i) / delta,      delta = sqrt(2 nu / omega)
    W(y, t) = U * [exp(-k y) - exp(-k (2H - y))] / [1 - exp(-2 k H)] * exp(i omega t)
    u(y, t) = Re W(y, t)

so the amplitude decays as ``A(y) = |W|`` and the phase lags by ``arg W``.  For
``H >> delta`` both reduce to the familiar ``A = U exp(-y/delta)``,
``phase = y/delta``; the finite-domain form is used here because it is exact for
the geometry that is actually simulated (the upper wall really is a no-slip
wall in the particle layout).

What is verified
----------------
The case tests the *no-slip wall condition of the viscous operator* under an
unsteady load, which neither ``benchmarks/couette`` (steady, straight profile)
nor ``benchmarks/poiseuille`` (steady, curved profile) can: the answer is a
decaying amplitude and a phase lag, and a wall model that is wrong in the
tangential direction changes one or both.  The particle solver is the unchanged
production path (``tensorsph.droplet_impact.rhs``): the oscillating plate enters
through ``wall_vel`` -- the same input the wave-flume wavemaker uses -- with
``wall_friction = 1`` for the no-slip blend, and the wall velocity is updated
every step from the analytic ``U cos(omega t)``.  No parameter is calibrated
against the run.

Acceptance gates (predeclared, ``benchmarks/stokes_second/DESIGN.md``)
---------------------------------------------------------------------
* ``G1`` amplitude L2 error inside the Stokes layer (rows ``0 < y <= delta``,
  first fluid row excluded -- see below) < 5 % of ``U``;
* ``G2`` decay-rate error ``|d ln A/dy * delta + 1|`` < 5 %;
* ``G3`` phase-lag slope error ``|d phase/dy * delta - 1|`` < 5 %;
* ``G4`` free-slip negative control: amplitude of the first row < 20 % of the
  analytic value (a wall that does not transfer tangential momentum cannot
  produce the Stokes layer);
* the Stokes layer must contain at least 5 particle rows (resolution
  requirement of the anchor: "Stokes 层内 >=5 层粒子").

The first fluid row is excluded from G1 because the wall/fluid kernel support
there is one-sided and the wall pressure is extrapolated (Adami et al. 2012);
its measured amplitude is reported, not gated.  ``G2``/``G3`` are unaffected by
a constant wall offset (they are slopes), so they carry the physics claim while
``G1`` measures the absolute level.

Usage::

    PYTHONPATH=src python benchmarks/stokes_second/run.py --out benchmarks/stokes_second/out
    PYTHONPATH=src python benchmarks/stokes_second/run.py --quick
    PYTHONPATH=src python benchmarks/stokes_second/run.py --figure
"""
from __future__ import annotations

import argparse
import json
import math
import sys
import time
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))

from tensorsph.droplet_impact import Config, resolve_device, rhs  # noqa: E402

OUT = Path(__file__).resolve().parent / "out"

# --------------------------------------------------------------------------- #
# Case definition
# --------------------------------------------------------------------------- #
HEIGHT = 0.5                 # m, gap between the plates
NU = 0.05                    # m^2/s, kinematic viscosity (input, not fitted)
PERIOD = 0.5                 # s, oscillation period of the plate
OMEGA = 2.0 * math.pi / PERIOD
DELTA = math.sqrt(2.0 * NU / OMEGA)   # Stokes layer thickness, 8.92 cm
U0 = 0.02                    # m/s, plate velocity amplitude
SOUND_SPEED = 5.0            # m/s, c0 (Mach = U/c0 = 4e-3, weakly compressible)
RHO0 = 1000.0
H_FACTOR = 1.3
WALL_LAYERS = 3
WALL_FRICTION = 1.0
COLUMNS = 8                  # fluid columns in one periodic cell
LENGTH = COLUMNS * (HEIGHT / 40.0)   # 8 dx at the reference resolution
DX_REF = HEIGHT / 40.0       # 1.25 cm -> 7.1 rows inside the Stokes layer
PERIODS_DISCARD = 2          # transient decay: delta^2/nu = 0.159 s = 0.32 T
PERIODS_AVERAGE = 2

GATES = {
    "amplitude_l2": 0.05,
    "decay_rate": 0.05,
    "phase_slope": 0.05,
    "freeslip_ratio": 0.20,
    "min_rows_in_layer": 5,
}

REFERENCE_NOTE = (
    "Landau & Lifshitz, Fluid Mechanics, §24 (oscillating plate); finite-gap "
    "image form W = U[exp(-ky) - exp(-k(2H-y))]/[1-exp(-2kH)] exp(i omega t), "
    "k=(1+i)/delta, delta=sqrt(2 nu/omega). Secondary: Macia et al. (2011) "
    "Prog. Theor. Phys. 125:1091, doi:10.1143/ptp.125.1091 (SPH for this case)."
)


# --------------------------------------------------------------------------- #
# Exact reference
# --------------------------------------------------------------------------- #

def analytic(y, height: float = HEIGHT, nu: float = NU,
             omega: float = OMEGA, u0: float = U0):
    """Complex amplitude of the exact finite-gap Stokes solution.

    Returns ``(amplitude, phase_lag)`` arrays; ``u(y, t) = A cos(omega t - phase)``.
    """
    y = np.asarray(y, dtype=float)
    delta = math.sqrt(2.0 * nu / omega)
    k = (1.0 + 1.0j) / delta
    w = u0 * (np.exp(-k * y) - np.exp(-k * (2.0 * height - y))) / \
        (1.0 - np.exp(-2.0 * k * height))
    # u = Re(W e^{i omega t}) = A cos(omega t + arg W) -> lag = -arg W
    return np.abs(w), -np.angle(w)


def analytic_wall_amplitude(u0: float = U0) -> float:
    """Analytic plate amplitude (the exact form equals ``u0`` at ``y = 0``)."""
    return float(analytic(0.0, u0=u0)[0])


# --------------------------------------------------------------------------- #
# Setup
# --------------------------------------------------------------------------- #

def make_config(dx: float, viscosity: float, friction: float, device) -> Config:
    return Config(
        dx=dx,
        h_factor=H_FACTOR,
        rho0=RHO0,
        sound_speed=SOUND_SPEED,
        gamma=7.0,
        gravity=0.0,             # no hydrostatic loading: pure shear problem
        viscosity=viscosity,
        dt=0.0,                  # filled by the computed stable step
        steps=0,
        output_every=1000,
        pool_width=LENGTH,
        pool_depth=HEIGHT,
        boundary="wall",
        wall_layers=WALL_LAYERS,
        wall_friction=friction,
        fail_fast=False,
        device=str(device),
    )


def build_fluid(cfg: Config, height: float, length: float, device):
    """Fluid lattice, rows ``y = dx/2, 3dx/2, ... < height``, ``x`` in [0, L)."""
    dx = cfg.dx
    xs = torch.arange(dx * 0.5, length, dx, device=device)
    ys = torch.arange(dx * 0.5, height, dx, device=device)
    gx, gy = torch.meshgrid(xs, ys, indexing="xy")
    pos = torch.stack((gx.reshape(-1), gy.reshape(-1)), dim=-1)
    mass = torch.full((pos.shape[0],), cfg.rho0 * dx * dx, device=device)
    return pos, mass


def build_walls(cfg: Config, height: float, length: float, layers: int,
                device):
    """Bottom (oscillating) and top (rest) wall bands; velocities start at zero.

    ``wall_vel`` is updated in the time loop: the bottom band carries
    ``u_x = U cos(omega t)``, the top band stays at rest.  The bands are extended
    three spacings past both ``x`` ends so periodic images of the fluid always
    see full wall support (same construction as ``benchmarks/couette``).
    """
    dx = cfg.dx
    margin = 3.0 * dx
    xs = torch.arange(-margin, length + margin, dx, device=device)
    bottom, top = [], []
    for layer in range(1, layers + 1):
        yb = torch.full_like(xs, -dx * 0.5 - (layer - 1) * dx)
        bottom.append(torch.stack((xs, yb), dim=-1))
        yt = torch.full_like(xs, height + dx * 0.5 + (layer - 1) * dx)
        top.append(torch.stack((xs, yt), dim=-1))
    # Bottom band first, so ``wall_vel[:n_bottom]`` is exactly the moving plate.
    pos = torch.cat(bottom + top, dim=0)
    n_bottom = xs.numel() * layers
    return pos, torch.zeros_like(pos), n_bottom


def periodic_images(pos, length: float):
    shift = torch.tensor([length, 0.0], dtype=pos.dtype, device=pos.device)
    return torch.cat((pos, pos + shift, pos - shift), dim=0)


# --------------------------------------------------------------------------- #
# Time integration
# --------------------------------------------------------------------------- #

def run_case(cfg: Config, height: float, length: float, u0: float,
             omega: float, periods_discard: float, periods_average: float,
             device, snapshot: bool = True):
    """Integrate the oscillating-plate channel and project onto the fundamental."""
    dx = cfg.dx
    pos, mass = build_fluid(cfg, height, length, device)
    n = pos.shape[0]
    wall, wall_vel, n_bottom = build_walls(cfg, height, length,
                                              cfg.wall_layers, device)
    vel = torch.zeros_like(pos)

    h = cfg.h_factor * dx
    dt_acoustic = 0.3 * h / cfg.sound_speed
    dt_viscous = 0.125 * h * h / cfg.viscosity
    dt = min(dt_acoustic, dt_viscous)
    cfg.dt = dt

    period = 2.0 * math.pi / omega
    steps = int(round((periods_discard + periods_average) * period / dt))
    window_start = int(round(periods_discard * period / dt))
    window_steps = steps - window_start
    tw = window_steps * dt                     # exact averaging window length

    # Row bins from the lattice construction (the flow is one-dimensional in y,
    # so the label never has to be recomputed).
    n_rows = int(round(height / dx))
    row = torch.round((pos[:, 1] - dx * 0.5) / dx).to(torch.long).clamp_(0, n_rows - 1)
    row_sum = torch.zeros(n_rows, dtype=pos.dtype, device=device)
    row_count = torch.bincount(row, minlength=n_rows).clamp_min(1)

    acc_cos = torch.zeros(n_rows, dtype=torch.float64, device=device)
    acc_sin = torch.zeros(n_rows, dtype=torch.float64, device=device)
    probe_rows = [1, max(2, int(round(0.5 * math.sqrt(2 * cfg.viscosity / omega) / dx)))]
    probe = {k: [] for k in probe_rows}

    sample_every = max(1, steps // 400)
    wall_clock = time.time()
    snap = None

    for step in range(steps):
        t = step * dt
        wall_vel[:n_bottom, 0] = u0 * math.cos(omega * t)
        wall_vel[n_bottom:, 0] = 0.0

        pos_ext = periodic_images(pos, length)
        vel_ext = torch.cat((vel, vel, vel), dim=0)
        mass_ext = torch.cat((mass, mass, mass), dim=0)
        _, _, acc = rhs(pos_ext, vel_ext, mass_ext, cfg, wall=wall,
                        wall_vel=wall_vel)
        acc = acc[:n].clone()

        if step >= window_start:
            w = 2.0 * dt / tw
            row_sum.zero_()
            row_sum.index_add_(0, row, vel[:, 0])
            row_mean = (row_sum / row_count).to(torch.float64)
            acc_cos += w * row_mean * math.cos(omega * t)
            acc_sin += w * row_mean * math.sin(omega * t)

        if step % sample_every == 0 or step == steps - 1:
            u = vel[:, 0]
            entry = {"t": round(t, 6), "u_top": float(u.max()),
                     "u_mean": float(u.mean())}
            for k in probe_rows:
                sel = row == k
                entry[f"u_row{k}"] = float(u[sel].mean()) if bool(sel.any()) else 0.0
            probe_entry = entry
            for k in probe_rows:
                probe[k].append({"t": entry["t"], "u": entry[f"u_row{k}"]})
            if step >= steps - 1 or (snapshot and snap is None and t >= 0.9 * steps * dt):
                snap = {"x": pos[:, 0].tolist(), "y": pos[:, 1].tolist(),
                        "u": u.tolist(), "t": t}

        vel = vel + dt * acc
        pos = pos + dt * vel
        pos[:, 0] = torch.remainder(pos[:, 0], length)

    y_rows = np.array([dx * 0.5 + k * dx for k in range(n_rows)])
    amp = torch.sqrt(acc_cos ** 2 + acc_sin ** 2).cpu().numpy()
    phase = torch.atan2(acc_sin, acc_cos).cpu().numpy()
    a_ref, p_ref = analytic(y_rows, height=height, nu=cfg.viscosity, omega=omega,
                            u0=u0)

    profile = []
    for k in range(n_rows):
        profile.append({
            "row": k,
            "y": float(y_rows[k]),
            "y_over_delta": float(y_rows[k] / math.sqrt(2 * cfg.viscosity / omega)),
            "amplitude": float(amp[k]),
            "amplitude_ref": float(a_ref[k]),
            "phase": float(phase[k]),
            "phase_ref": float(p_ref[k]),
        })

    return {
        "n_fluid": int(n),
        "n_wall": int(wall.shape[0]),
        "dx": dx,
        "dt": dt,
        "period": period,
        "steps": steps,
        "window_steps": window_steps,
        "window_s": tw,
        "delta": math.sqrt(2.0 * cfg.viscosity / omega),
        "rows_in_layer": int(np.sum((y_rows > 0) & (y_rows <= math.sqrt(2.0 * cfg.viscosity / omega)))),
        "wall_friction": cfg.wall_friction,
        "profile": profile,
        "probe": {f"row{k}": v for k, v in probe.items()},
        "snapshot": snap,
        "max_mach": float(max(p["u_top"] for p in [probe_entry])) / cfg.sound_speed,
        "wall_clock_s": time.time() - wall_clock,
    }


# --------------------------------------------------------------------------- #
# Analysis
# --------------------------------------------------------------------------- #

def layer_metrics(result: dict, gate_delta: float) -> dict:
    """Compare the measured amplitude/phase field with the exact solution."""
    prof = result["profile"]
    delta = result["delta"]
    # Rows strictly inside the Stokes layer, first fluid row excluded.
    sel = [p for p in prof if 0.0 < p["y"] <= delta and p["row"] >= 1]
    first = prof[0]
    if len(sel) < 2:
        raise RuntimeError("fewer than two usable rows inside the Stokes layer")

    y = np.array([p["y"] for p in sel])
    a = np.array([p["amplitude"] for p in sel])
    ar = np.array([p["amplitude_ref"] for p in sel])
    ph = np.array([p["phase"] for p in sel])
    phr = np.array([p["phase_ref"] for p in sel])

    u0 = ar[0] / math.exp(-y[0] / delta) if y[0] < delta else float(ar[0])
    l2 = float(np.sqrt(np.mean((a - ar) ** 2)) / u0)

    # Decay rate: least squares on ln A (analytic slope is -1/delta).
    lna = np.log(np.maximum(a, 1e-12))
    slope_a = float(np.polyfit(y, lna, 1)[0])
    slope_ar = float(np.polyfit(y, np.log(ar), 1)[0])
    decay_err = abs((slope_a - slope_ar) * delta)

    # Phase lag: analytic phase grows as y/delta (modulo 2*pi wraps).
    dph = np.unwrap(ph - phr)
    phase_err = abs(float(np.polyfit(y, ph, 1)[0]) * delta - 1.0)

    # Effective wall offset implied by the amplitude decay (diagnostic only).
    intercept = float(np.polyfit(y, lna, 1)[1])
    y_at_u0 = (math.log(u0) - intercept) / slope_a if slope_a != 0 else float("nan")
    wall_offset = float(-y_at_u0) if math.isfinite(y_at_u0) else float("nan")

    return {
        "n_rows_in_layer": len(sel),
        "amplitude_l2": l2,
        "decay_rate": slope_a * delta,
        "decay_rate_ref": slope_ar * delta,
        "decay_err": decay_err,
        "phase_slope": float(np.polyfit(y, ph, 1)[0]) * delta,
        "phase_err": phase_err,
        "phase_wrap_maxdev": float(np.max(np.abs(dph))),
        "first_row_amplitude": float(first["amplitude"]),
        "first_row_amplitude_ref": float(first["amplitude_ref"]),
        "first_row_ratio": float(first["amplitude"] / first["amplitude_ref"]),
        "wall_offset_over_dx": wall_offset / result["dx"] if math.isfinite(wall_offset) else float("nan"),
        "u0_est": u0,
        "max_abs_phase_dev_rad": float(np.max(np.abs(dph))),
    }


def verdict(result: dict, metrics: dict, control_metrics: dict | None) -> dict:
    reasons = []
    ok = True

    def gate(name, passed, detail):
        nonlocal ok
        ok = ok and passed
        if not passed:
            reasons.append(f"{name}: {detail}")

    gate("G1 amplitude L2", metrics["amplitude_l2"] < GATES["amplitude_l2"],
         f"{100 * metrics['amplitude_l2']:.2f}% >= 5%")
    gate("G2 decay rate", metrics["decay_err"] < GATES["decay_rate"],
         f"slope*delta = {metrics['decay_rate']:.4f} (ref "
         f"{metrics['decay_rate_ref']:.4f})")
    gate("G3 phase slope", metrics["phase_err"] < GATES["phase_slope"],
         f"phase slope*delta = {metrics['phase_slope']:.4f}")
    gate("G5 rows in layer",
         metrics["n_rows_in_layer"] >= GATES["min_rows_in_layer"],
         f"{metrics['n_rows_in_layer']} < 5")
    if control_metrics is not None:
        gate("G4 free-slip control",
             control_metrics["first_row_ratio"] < GATES["freeslip_ratio"],
             f"free-slip first-row amplitude ratio "
             f"{control_metrics['first_row_ratio']:.3f} >= 0.20")
    return {"status": "pass" if ok else "fail", "reasons": reasons,
            "gates": GATES}


# --------------------------------------------------------------------------- #
# Driver
# --------------------------------------------------------------------------- #

def run_one(dx: float, friction: float, viscosity: float = NU,
            periods_discard: float = PERIODS_DISCARD,
            periods_average: float = PERIODS_AVERAGE, device=None,
            label: str = "") -> dict:
    device = resolve_device(device or "cpu")
    cfg = make_config(dx, viscosity, friction, device)
    res = run_case(cfg, HEIGHT, LENGTH, U0, OMEGA, periods_discard,
                   periods_average, device)
    res["metrics"] = layer_metrics(res, GATES["amplitude_l2"])
    res["label"] = label
    return res


def make_figures(payload: dict, out: Path) -> list:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    names = []
    ref = min(payload["ladder"], key=lambda r: r["dx"])
    prof = ref["profile"]
    y = np.array([p["y_over_delta"] for p in prof])
    a = np.array([p["amplitude"] for p in prof])
    ar = np.array([p["amplitude_ref"] for p in prof])
    ph = np.array([p["phase"] for p in prof])
    phr = np.array([p["phase_ref"] for p in prof])

    fig, axes = plt.subplots(1, 3, figsize=(13.5, 4.2))
    ax = axes[0]
    ax.plot(y, a / U0, "o-", label="SPH (no-slip)")
    ax.plot(y, ar / U0, "k--", label="exact")
    ctrl = payload.get("control")
    if ctrl:
        yc = np.array([p["y_over_delta"] for p in ctrl["profile"]])
        ac = np.array([p["amplitude"] for p in ctrl["profile"]])
        ax.plot(yc, ac / U0, "s:", color="tab:red", label="free-slip control")
    ax.axvline(1.0, color="0.7", lw=0.8)
    ax.set_xlabel(r"$y/\delta$")
    ax.set_ylabel(r"amplitude $A/U$")
    ax.set_title(f"amplitude decay, dx/H=1/{round(HEIGHT / ref['dx'])}; "
                 f"L2={100 * ref['metrics']['amplitude_l2']:.2f}%")
    ax.legend()
    ax.grid(alpha=0.3)

    ax = axes[1]
    ax.plot(y, np.unwrap(ph), "o-", label="SPH")
    ax.plot(y, np.unwrap(phr), "k--", label="exact")
    ax.plot(y, y, ":", color="0.6", label=r"$y/\delta$ (H$\to\infty$)")
    ax.set_xlabel(r"$y/\delta$")
    ax.set_ylabel("phase lag [rad]")
    ax.set_title(f"phase lag; slope*delta={ref['metrics']['phase_slope']:.3f}")
    ax.legend()
    ax.grid(alpha=0.3)

    ax = axes[2]
    snap = ref.get("snapshot")
    if snap:
        s = ax.scatter(snap["x"], snap["y"], c=snap["u"], s=6, cmap="coolwarm")
        fig.colorbar(s, ax=ax, label=r"$u_x$ [m/s]")
        ax.set_aspect("equal")
        ax.set_xlabel("x [m]")
        ax.set_ylabel("y [m]")
        ax.set_title(f"u_x field at t={snap['t']:.3f} s")
    fig.tight_layout()
    path = out / "stokes_second.png"
    fig.savefig(path, dpi=140)
    plt.close(fig)
    names.append(path.name)

    # Time series at two probe rows vs the exact forcing.
    fig, ax = plt.subplots(figsize=(7.0, 3.6))
    for key, colour in (("row1", "tab:blue"), ("row2", "tab:green")):
        if key not in ref["probe"]:
            continue
        tr = ref["probe"][key]
        t = np.array([p["t"] for p in tr])
        u = np.array([p["u"] for p in tr])
        k = int(key.replace("row", ""))
        yy = ref["dx"] * (0.5 + k)
        plt_a, plt_p = analytic(np.array([yy]), u0=U0)
        m = t >= (PERIODS_DISCARD * ref["period"])
        ax.plot(t[m], u[m], ".", ms=3, color=colour,
                label=f"SPH row {k}")
        ax.plot(t[m], plt_a[0] * np.cos(OMEGA * t[m] - plt_p[0]), "-",
                color=colour, lw=0.9, label=f"exact row {k}")
    ax.set_xlabel("t [s]")
    ax.set_ylabel(r"$u_x$ [m/s]")
    ax.set_title("phase-resolved velocity at two rows")
    ax.grid(alpha=0.3)
    ax.legend(fontsize=8, ncol=2)
    fig.tight_layout()
    path = out / "stokes_second_timeseries.png"
    fig.savefig(path, dpi=140)
    plt.close(fig)
    names.append(path.name)
    return names


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, default=OUT)
    parser.add_argument("--resolutions", type=float, nargs="+",
                        default=[DX_REF, DX_REF / 2])
    parser.add_argument("--device", default="auto")
    parser.add_argument("--threads", type=int, default=2)
    parser.add_argument("--quick", action="store_true")
    parser.add_argument("--figure", action="store_true")
    parser.add_argument("--no-control", action="store_true")
    args = parser.parse_args(argv)

    args.out.mkdir(parents=True, exist_ok=True)
    if args.figure:
        payload = json.loads((args.out / "results.json").read_text())
        print("[stokes2] figures:", make_figures(payload, args.out))
        return 0

    if args.device in ("auto", "cpu"):
        torch.set_num_threads(args.threads)
    device = resolve_device(args.device)

    periods_discard, periods_average = PERIODS_DISCARD, PERIODS_AVERAGE
    resolutions = list(args.resolutions)
    if args.quick:
        resolutions = [DX_REF * 2]
        periods_discard, periods_average = 1, 1

    ladder = []
    for dx in resolutions:
        print(f"[stokes2] ladder dx={dx:.5f} (H/dx={HEIGHT / dx:.0f})", flush=True)
        res = run_one(dx, WALL_FRICTION, device=device,
                      periods_discard=periods_discard,
                      periods_average=periods_average,
                      label=f"dx{HEIGHT / dx:.0f}")
        m = res["metrics"]
        print(f"[stokes2]   ampL2={100 * m['amplitude_l2']:.2f}% "
              f"decay*delta={m['decay_rate']:.4f} phase*delta={m['phase_slope']:.4f} "
              f"rows={m['n_rows_in_layer']} ({res['wall_clock_s']:.0f}s,"
              f" {res['steps']} steps)", flush=True)
        ladder.append(res)

    control = None
    if not args.no_control:
        print("[stokes2] free-slip control", flush=True)
        control = run_one(resolutions[0], 0.0, device=device,
                          periods_discard=periods_discard,
                          periods_average=periods_average, label="freeslip")
        print(f"[stokes2]   first-row amplitude ratio "
              f"{control['metrics']['first_row_ratio']:.3f}", flush=True)

    payload = {
        "case": "stokes_second_problem",
        "description": ("2-D channel 0<y<H, bottom plate oscillates at "
                        "U cos(omega t), top plate at rest, periodic in x"),
        "reference": REFERENCE_NOTE,
        "reference_form": "u = Re{ U [exp(-ky)-exp(-k(2H-y))]/[1-exp(-2kH)] e^{i omega t} }, "
                          "k=(1+i)/delta, delta=sqrt(2 nu/omega)",
        "height": HEIGHT,
        "nu": NU,
        "period": PERIOD,
        "omega": OMEGA,
        "delta": DELTA,
        "u0": U0,
        "sound_speed": SOUND_SPEED,
        "rho0": RHO0,
        "h_factor": H_FACTOR,
        "wall_layers": WALL_LAYERS,
        "wall_friction": WALL_FRICTION,
        "periods_discard": periods_discard,
        "periods_average": periods_average,
        "gates": GATES,
        "device": str(device),
        "torch": torch.__version__,
        "ladder": ladder,
        "control": control,
        "verdict": verdict(ladder[0], ladder[0]["metrics"],
                           control["metrics"] if control else None),
        "wall_clock_s": sum(r["wall_clock_s"] for r in ladder + ([control] if control else [])),
    }
    (args.out / "results.json").write_text(json.dumps(payload, indent=2))
    print(f"[stokes2] verdict: {payload['verdict']['status']}")
    for r in payload["verdict"]["reasons"]:
        print(f"[stokes2]   ! {r}")
    names = make_figures(payload, args.out) if ladder else []
    print(f"[stokes2] figures: {names}")
    return 0 if payload["verdict"]["status"] == "pass" else 1


if __name__ == "__main__":
    raise SystemExit(main())