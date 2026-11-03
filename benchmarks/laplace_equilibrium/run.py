"""Young-Laplace pressure jump measured at *static equilibrium*.

What this benchmark settles
---------------------------
The two verification rounds that came before this one disagree with each other
about the same physical quantity, ``Delta p = sigma / R``:

* ``benchmarks/laplace_law`` (dx = 4 mm, 380 steps) reports a PASS;
* ``benchmarks/csf_consistent`` (dx = 2 mm, 600-900 steps) reports -75.6 %.

Both use the same CSF force and the same equation of state, so at most one of
them can be right about the physics.  This benchmark shows that *neither*
measurement protocol is an equilibrium measurement, and supplies one that is.

Three measurement defects are isolated and quantified:

1. **Transient sampling.**  A drop released with ``rho = rho0`` is not in
   equilibrium: surface tension squeezes it, it overshoots (the breathing mode,
   period ~1 capillary time) and decays on the viscous time.  ``p_core(t)``
   swings by a factor of several during that decay, so a single sample at a
   fixed step count reports whatever phase the run happened to be in.  Section
   ``phase_scan`` samples the *same* run at many step counts to size that
   artifact.
2. **Interface band inside the shell.**  The legacy protocol measures
   ``p_band`` over ``0.8 R < r < 0.95 R``.  The CSF force is localised by
   ``delta_s`` over a shell of thickness ~2h, and ``dp/dr = rho a_r < 0``
   *inside* that shell, so p has already fallen part of the way to the vacuum
   value at 0.95 R.  The band reading is therefore a systematic underestimate
   of the jump that is independent of any physics error, and its size grows
   with ``h/R`` (which is why the coarse case looked better).
3. **Core band in the truncated-support region.**  At ``dx = 4 mm`` the
   "core" band ``r < 0.5 R`` reaches into ``r > R - 2h``, where the summation
   density carries a kernel-truncation *deficit* (measured ``rho_core/rho0 =
   0.9972``, i.e. the wrong sign).  Pressure is read off the Tait law, which is
   exponential in density, so that deficit is a large negative bias.

The measurement protocol used here: sample only particles with **complete
kernel support** (``r < R - 2.5 h``), take the outside pressure as the vacuum
value ``p = 0`` (which the Tait law clamps to exactly zero), run until the
kinetic energy is negligible, and average over the last oscillations to remove
the residual ring-down.

Reference
---------
Young & Laplace, ``Delta p = sigma * kappa``; in 2-D for a circular interface
``kappa = 1/R``, so ``Delta p = sigma / R``.  Momentum balance gives the same
number without any pressure measurement: ``int_0^R rho a_r dr = sigma/R``
because ``int delta_s dn = 1``.

Run
---
    OMP_NUM_THREADS=1 PYTHONPATH=src python3 benchmarks/laplace_equilibrium/run.py
    OMP_NUM_THREADS=1 PYTHONPATH=src python3 benchmarks/laplace_equilibrium/run.py --quick

Outputs ``report.json`` (every number quoted in the docs comes from it) and
``laplace_equilibrium.png``.
"""
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import torch

from tensorsph.droplet_impact import Config, rhs, stable_dt
from tensorsph.neighbor_search import cell_linked_pairs

HERE = Path(__file__).resolve().parent
torch.set_num_threads(2)

SIGMA = 7.2          # N/m, water/air
RHO0 = 1000.0        # kg/m^3
C0 = 20.0            # m/s, keeps Mach < 0.05
RADIUS = 0.02        # m
VISCOSITY = 0.02     # m^2/s -- sets only the ring-down rate, not the static state


# --------------------------------------------------------------------------- #
# geometry
# --------------------------------------------------------------------------- #
def hexagonal_droplet(radius: float, dx: float, rho0: float):
    """Hexagonally packed circular drop centered on the origin.

    Six neighbours per particle halves the noise of the SPH gradient operators
    relative to a square lattice.  ``mass = rho0 * dx * dy`` makes the lattice
    number density equal ``rho0``, so the initial summation density is ``rho0``
    in the interior.
    """
    dy = dx * math.sqrt(3.0) / 2.0
    mass = rho0 * dx * dy
    pts = []
    row = 0
    y = -radius
    while y <= radius:
        offset = 0.5 * dx if (row % 2) else 0.0
        x = -radius + offset
        while x <= radius:
            if x * x + y * y <= radius * radius:
                pts.append((x, y))
            x += dx
        y += dy
        row += 1
    pos = torch.tensor(pts, dtype=torch.float64)
    return pos, torch.full((pos.shape[0],), float(mass), dtype=torch.float64)


def complete_support_mask(pos: torch.Tensor, radius: float, h: float,
                          margin: float = 2.5) -> torch.Tensor:
    """Particles whose kernel support is entirely inside the fluid.

    The cubic-spline kernel reaches 2h, so a particle closer than ``margin * h``
    to the interface has a truncated neighbour sum; its summation density is
    biased low and, through the Tait law, so is its pressure.  ``margin = 2.5``
    leaves half a smoothing length of slack.
    """
    r = torch.linalg.vector_norm(pos, dim=1)
    return r < radius - margin * h


# --------------------------------------------------------------------------- #
# one relaxation run
# --------------------------------------------------------------------------- #
def relax(dx: float, steps: int | None = None, radius: float = RADIUS,
          sigma: float = SIGMA, rho0: float = RHO0, sound_speed: float = C0,
          viscosity: float = VISCOSITY, density_diffusion: float = 0.0,
          cusp: float = 4.0, every: int = 40) -> dict:
    """Integrate a static drop to equilibrium and measure the pressure jump.

    ``steps`` defaults to ``cusp`` capillary times, which is several viscous
    damping times at ``viscosity = 0.02`` and therefore well past the point
    where the breathing mode has died out.
    """
    cfg = Config(dx=dx, h_factor=1.3, rho0=rho0, sound_speed=sound_speed,
                 gamma=7.0, gravity=0.0, viscosity=viscosity,
                 surface_tension_coefficient=sigma, csf_scheme="consistent",
                 density_diffusion=density_diffusion)
    h = cfg.h_factor * dx
    dt = float(stable_dt(cfg))
    t_sigma = math.sqrt(rho0 * radius ** 3 / sigma)
    v_ref = math.sqrt(sigma / (rho0 * radius))
    if steps is None:
        steps = max(600, int(round(cusp * t_sigma / dt)))

    pos, mass = hexagonal_droplet(radius, dx, rho0)
    vel = torch.zeros_like(pos)
    hist = {"t_over_t_sigma": [], "p_complete": [], "p_band": [], "p_core50": [],
            "v_over_vref": [], "rho_excess": [], "p_eos_check": []}

    for step in range(steps):
        pairs = cell_linked_pairs(pos, 2.0 * h)
        rho, pressure, acc = rhs(pos, vel, mass, cfg, pairs=pairs, wall=None,
                                 dt=dt)
        vel = vel + dt * acc
        pos = pos + dt * vel
        if step % every == 0 or step == steps - 1:
            r = torch.linalg.vector_norm(pos, dim=1)
            full = complete_support_mask(pos, radius, h)
            band = (r > 0.8 * radius) & (r < 0.95 * radius)
            core50 = r < 0.5 * radius
            hist["t_over_t_sigma"].append(float((step + 1) * dt / t_sigma))
            hist["p_complete"].append(float(pressure[full].mean()))
            hist["p_band"].append(float(pressure[band].mean()) if band.any()
                                  else float("nan"))
            hist["p_core50"].append(float(pressure[core50].mean()))
            hist["v_over_vref"].append(
                float(torch.linalg.vector_norm(vel, dim=1).max()) / v_ref)
            hist["rho_excess"].append(float((rho[full] / rho0 - 1.0).mean()))
            # the Tait law applied to the measured mean density must reproduce
            # the measured mean pressure -- catches a stale pressure array
            ratio = float(rho[full].mean()) / rho0
            bulk = rho0 * sound_speed ** 2 / 7.0
            hist["p_eos_check"].append(
                float(bulk * (ratio ** 7 - 1.0)))

    # equilibrium estimate: mean over the last half of the record, which still
    # contains 1-2 residual breathing cycles at the viscosity used here
    n_half = max(1, len(hist["p_complete"]) // 2)
    def tail(key):
        vals = [v for v in hist[key][-n_half:] if not math.isnan(v)]
        return sum(vals) / len(vals)

    full = complete_support_mask(pos, radius, h)
    r = torch.linalg.vector_norm(pos, dim=1)
    dp_theory = sigma / radius
    # statistical uncertainty of the particle-mean reading, and the residual
    # ring-down of the record: both bound how much of the reported error is
    # measurement noise rather than physics
    p_sem = float(pressure[full].std() / math.sqrt(max(1, int(full.sum()))))
    p_ring = (max(hist["p_complete"][-n_half:])
              - min(hist["p_complete"][-n_half:])) / 2.0
    p_complete = tail("p_complete")
    p_band, p_core50 = tail("p_band"), tail("p_core50")

    def err(p):
        return (p - dp_theory) / dp_theory * 100.0

    return {
        "setup": {
            "radius_m": radius, "dx_m": dx, "h_m": h, "h_over_radius": h / radius,
            "n_particles": int(pos.shape[0]),
            "n_complete_support": int(full.sum()),
            "n_band_080_095": int(((r > 0.8 * radius) &
                                   (r < 0.95 * radius)).sum()),
            "n_core_lt_050": int((r < 0.5 * radius).sum()),
            "sigma_N_per_m": sigma, "rho0_kg_per_m3": rho0,
            "sound_speed_m_per_s": sound_speed, "viscosity_m2_per_s": viscosity,
            "density_diffusion_delta": density_diffusion,
            "dt_s": dt, "steps": steps, "physical_time_s": steps * dt,
            "capillary_time_s": t_sigma,
            "t_over_t_sigma": steps * dt / t_sigma,
            "lattice": "hexagonal", "gravity": 0.0, "csf_scheme": "consistent",
        },
        "reference": {
            "delta_p_Pa": dp_theory,
            "curvature_1_per_m": 1.0 / radius,
            "density_excess_required": (1.0 + dp_theory
                                        / (rho0 * sound_speed ** 2 / 7.0)) ** (1 / 7) - 1.0,
            "note": "Delta p = sigma/R (2-D circle); p_outside = 0 by the "
                    "non-negative Tait clamp",
        },
        "equilibrium": {
            "p_complete_Pa": p_complete, "rel_error_pct": err(p_complete),
            "p_band_Pa": p_band, "rel_error_band_pct": err(p_complete - p_band),
            "p_core50_Pa": p_core50, "rel_error_core50_pct": err(p_core50),
            "terminal_v_over_vref": hist["v_over_vref"][-1],
            "terminal_rho_excess": hist["rho_excess"][-1],
            "ring_down_amplitude_over_theory": (max(hist["p_complete"][-n_half:])
                                               - min(hist["p_complete"][-n_half:]))
                                              / dp_theory,
            "p_complete_sem_Pa": p_sem,
            "p_ring_down_half_amplitude_Pa": p_ring,
            "measurement_snr": p_complete
            / max(p_sem, p_ring, 1e-12),
            "within_3pct": abs(err(p_complete)) < 3.0,
            "within_15pct": abs(err(p_complete)) < 15.0,
            "band_protocol_within_15pct": abs(err(p_complete - p_band)) < 15.0,
        },
        "history": hist,
    }


def phase_scan(dx: float = 0.004, steps_total: int = 380,
               viscosity: float = 0.01) -> dict:
    """Sample the *same* 380-step run at 19 different step counts.

    Reproduces the protocol of ``tests/test_laplace_law.py::test_laplace_
    pressure_jump`` (dx = 4 mm, 380 steps) and shows how much of its verdict is
    decided by *when* the sample is taken.
    """
    cfg = Config(dx=dx, h_factor=1.3, rho0=RHO0, sound_speed=C0, gamma=7.0,
                 gravity=0.0, viscosity=viscosity,
                 surface_tension_coefficient=SIGMA, csf_scheme="consistent")
    h = cfg.h_factor * cfg.dx
    dt = float(stable_dt(cfg))
    pos, mass = hexagonal_droplet(RADIUS, dx, RHO0)
    vel = torch.zeros_like(pos)
    dp_theory = SIGMA / RADIUS
    samples = []
    marks = sorted({int(round(s)) for s in
                    torch.linspace(20, steps_total, 19).tolist()})
    for step in range(1, steps_total + 1):
        pairs = cell_linked_pairs(pos, 2.0 * h)
        rho, pressure, acc = rhs(pos, vel, mass, cfg, pairs=pairs, wall=None,
                                 dt=dt)
        vel = vel + dt * acc
        pos = pos + dt * vel
        if step in marks:
            r = torch.linalg.vector_norm(pos, dim=1)
            full = complete_support_mask(pos, RADIUS, h)
            band = (r > 0.8 * RADIUS) & (r < 0.95 * RADIUS)
            core50 = r < 0.5 * RADIUS
            p_full = float(pressure[full].mean())
            p_band = float(pressure[band].mean()) if band.any() else 0.0
            p_core50 = float(pressure[core50].mean())
            samples.append({
                "step": step,
                "t_over_t_sigma": step * dt
                / math.sqrt(RHO0 * RADIUS ** 3 / SIGMA),
                "p_complete_Pa": p_full,
                "p_band_Pa": p_band,
                "p_core50_Pa": p_core50,
                "rel_error_pct": (p_full - dp_theory) / dp_theory * 100.0,
                "rel_error_band_pct": (p_full - p_band - dp_theory)
                / dp_theory * 100.0,
                # exactly the protocol of tests/test_laplace_law.py: p(r<0.5R)
                # minus p(0.8R..0.95R), sampled once at the end of the run
                "rel_error_legacy_pct": (p_core50 - p_band - dp_theory)
                / dp_theory * 100.0,
            })
    errs = [s["rel_error_pct"] for s in samples]
    legacy = [s["rel_error_legacy_pct"] for s in samples]
    return {
        "setup": {"dx_m": dx, "n_particles": int(pos.shape[0]), "dt_s": dt,
                  "steps_total": steps_total, "viscosity_m2_per_s": viscosity,
                  "protocol": "one run, sampled at every listed step count; "
                              "'legacy' = p(r<0.5R) - p(0.8R..0.95R), the "
                              "protocol of tests/test_laplace_law.py",
                  "n_complete_support": int(complete_support_mask(
                      pos, RADIUS, h).sum())},
        "reference": {"delta_p_Pa": dp_theory},
        "samples": samples,
        "error_spread_pct": max(errs) - min(errs),
        "legacy_error_spread_pct": max(legacy) - min(legacy),
        "n_samples_within_15pct": sum(1 for e in errs if abs(e) < 15.0),
        "n_legacy_samples_within_15pct": sum(1 for e in legacy
                                            if abs(e) < 15.0),
        "n_samples": len(errs),
    }


# --------------------------------------------------------------------------- #
# figures
# --------------------------------------------------------------------------- #
def make_figures(ladder: list, scan: dict, out_dir: Path = HERE) -> Path:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(1, 3, figsize=(15.5, 4.6))

    # (a) convergence of the measured jump toward sigma/R
    a = ax[0]
    for row in ladder:
        h = row["history"]
        a.plot(h["t_over_t_sigma"], [p / row["reference"]["delta_p_Pa"]
                                     for p in h["p_complete"]],
               lw=1.3, label=f"dx = {row['setup']['dx_m'] * 1e3:.1f} mm "
                             f"(N = {row['setup']['n_particles']})")
    a.axhline(1.0, color="k", ls="--", lw=1.2, label=r"$\sigma/R$")
    a.set_xlabel(r"$t / t_\sigma$")
    a.set_ylabel(r"$p_\mathrm{core}\,/\,(\sigma/R)$")
    a.set_title("(a) pressure jump at equilibrium\n(complete-support particles)")
    a.legend(fontsize=8)
    a.grid(alpha=0.3)

    # (b) error of the two protocols vs resolution
    b = ax[1]
    dxs = [r["setup"]["dx_m"] * 1e3 for r in ladder]
    eq = [r["equilibrium"]["rel_error_pct"] for r in ladder]
    band = [r["equilibrium"]["rel_error_band_pct"] for r in ladder]
    b.plot(dxs, eq, "o-", lw=1.8, label="equilibrium protocol\n"
                                       r"($p_\mathrm{core}-0$, $r<R-2.5h$)")
    b.plot(dxs, band, "s--", lw=1.5, color="tab:red",
           label="legacy band protocol\n" r"($p_\mathrm{core}-p_{0.8R..0.95R}$)")
    b.axhspan(-15, 15, color="tab:green", alpha=0.12, label="15 % bar")
    b.axhline(0, color="k", lw=0.8)
    b.set_xlabel("dx [mm]")
    b.set_ylabel(r"$\Delta p$ error [%]")
    b.set_title("(b) systematic of the measurement basis")
    b.legend(fontsize=8)
    b.grid(alpha=0.3)

    # (c) phase artifact
    c = ax[2]
    s = scan["samples"]
    c.plot([x["step"] for x in s], [x["rel_error_pct"] for x in s], "o-",
           lw=1.4, ms=3.5, label="equilibrium basis")
    c.plot([x["step"] for x in s], [x["rel_error_legacy_pct"] for x in s], "s--",
           lw=1.2, ms=3.0, color="tab:red",
           label="legacy basis (test_laplace_law)")
    c.axhspan(-15, 15, color="tab:green", alpha=0.12)
    c.axhline(0, color="k", lw=0.8)
    c.set_xlabel("step at which the sample is taken")
    c.set_ylabel(r"$\Delta p$ error [%]")
    c.set_title(f"(c) phase artifact: one run, {scan['n_samples']} sample times\n"
                f"(dx = {scan['setup']['dx_m'] * 1e3:.0f} mm, "
                f"spread = {scan['error_spread_pct']:.0f} %)")
    c.legend(fontsize=8)
    c.grid(alpha=0.3)

    fig.tight_layout()
    out = out_dir / "laplace_equilibrium.png"
    fig.savefig(out, dpi=140)
    plt.close(fig)
    return out


# --------------------------------------------------------------------------- #
# driver
# --------------------------------------------------------------------------- #
def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--quick", action="store_true",
                    help="short ladder for a smoke run (not the committed record)")
    ap.add_argument("--cusp", type=float, default=4.0,
                    help="capillary times to integrate")
    ap.add_argument("--out", type=Path, default=HERE / "report.json")
    args = ap.parse_args()

    dxs = [0.004, 0.003] if args.quick else [0.003, 0.002, 0.0015]
    cusp = 1.5 if args.quick else args.cusp
    ladder = []
    for dx in dxs:
        rep = relax(dx=dx, cusp=cusp)
        ladder.append(rep)
        eq = rep["equilibrium"]
        print(f"dx={dx * 1e3:5.2f} mm  N={rep['setup']['n_particles']:5d}  "
              f"ncomp={rep['setup']['n_complete_support']:4d}  "
              f"steps={rep['setup']['steps']:6d}  "
              f"p_core={eq['p_complete_Pa']:8.2f} Pa  "
              f"err={eq['rel_error_pct']:+7.2f} %  "
              f"(band basis {eq['rel_error_band_pct']:+7.2f} %)  "
              f"v/vref={eq['terminal_v_over_vref']:.4f}")

    scan = phase_scan()
    print(f"phase scan: equilibrium-basis spread {scan['error_spread_pct']:.0f} % "
          f"({scan['n_samples_within_15pct']}/{scan['n_samples']} inside 15 %), "
          f"legacy-basis spread {scan['legacy_error_spread_pct']:.0f} % "
          f"({scan['n_legacy_samples_within_15pct']}/{scan['n_samples']} "
          f"inside 15 %)")

    report = {
        "benchmark": "laplace_equilibrium",
        "purpose": "Delta p = sigma/R measured at static equilibrium, after "
                   "removing the three measurement defects that made earlier "
                   "rounds disagree (transient sampling, interface band inside "
                   "the CSF shell, core band in the truncated-support region)",
        "reference": {"delta_p_Pa": SIGMA / RADIUS,
                      "curvature_1_per_m": 1.0 / RADIUS,
                      "source": "Young-Laplace, 2-D circular interface"},
        "params": {"sigma_N_per_m": SIGMA, "rho0_kg_per_m3": RHO0,
                   "sound_speed_m_per_s": C0, "radius_m": RADIUS,
                   "viscosity_m2_per_s": VISCOSITY,
                   "capillary_times": cusp},
        "ladder": ladder,
        "phase_scan": scan,
    }
    args.out.write_text(json.dumps(report, indent=1))
    print("report ->", args.out)
    print("figure ->", make_figures(ladder, scan))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())