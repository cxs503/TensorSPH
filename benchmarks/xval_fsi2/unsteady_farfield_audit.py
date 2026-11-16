"""Unsteady far-field (control-volume) force-from-fluid audit.

Anchor ``XVAL-SPH-FSI2-BENCHMARK-AND-FORCE-FROM-FLUID`` (Xiangyu-Hu/SPHinXsys
``test_2d_fsi2`` + ``FSI::ForceFromFluid``).  Round 3 of this anchor added the
*steady* control-volume channel (:mod:`tensorsph.farfield_force`) and had to
stop at the accumulation term ``-d/dt int rho u dV`` because in 2-D its uniform
part grows like ``R**2`` and cancels against the pressure flux.  This driver
audits the perturbation form that repairs that, on a field whose load is known
in closed form *before* the run.

Predeclared decision and gates (fixed before any number was looked at)
---------------------------------------------------------------------
D1  Can the far-field channel be used for an **unsteady** load (added mass,
    inertia phase, oscillating-stream force) at a control-surface radius the
    coupled ``fsi2`` runs can afford?
    PASS  if the inertia amplitude is within **3 %** of ``2 rho pi a**2 u_dot``
          at every ``R/a <= 10`` and ``dx/a <= 0.05``;
    ACCEPT (owner gate) if within **15 %**; otherwise FAIL.
D2  Wiring null: an accelerating *uniform* stream with **no body** must give
    exactly zero force -- relative residual < 1e-9 of ``rho R**2 u_dot``.
D3  The literal (unsplit) form of (1) must be shown to be the blocker it was
    claimed to be: its error is expected to grow like ``(R/a)**2``.

Field (analytic, no solver)
---------------------------
Fixed cylinder ``r <= a`` in an oscillating stream ``U(t) = U0 cos(Omega t)``
along ``+x``; velocity from ``phi = U(t)(r + a**2/r) cos t``, pressure from the
unsteady Bernoulli equation (:func:`tensorsph.farfield_force.
unsteady_potential_flow_cylinder`).  Exact load: ``F_x = 2 rho pi a**2 u_dot``
(``C_M = 2``), ``F_y = 0``.  The estimator never sees the body: it is handed
the body's *area* only.

Arms
----
``oss-perturbation``  the new unsteady CV form (default ``reconstruct="fit"``).
``mean``              same, with the plain bin average on the surface.
``naive``             literal unsplit (1): raw momentum rate + total pressure.
``steady-only``       the round-3 estimator (no accumulation term) on the same
                      field -- the arm whose error the new term must remove.

Every number in ``docs/xval_fsi2/UNSTEADY_FARFIELD_TERM.md`` comes from the
``--out`` JSON of this driver.
"""
from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path

import torch

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(HERE))

from tensorsph.farfield_force import (  # noqa: E402
    control_volume_force,
    control_volume_force_unsteady,
    cv_momentum,
    d_dt_periodic,
    unsteady_potential_flow_cylinder,
)

from farfield_force_audit import annulus_lattice  # noqa: E402

DT = torch.float64

# ---- predeclared case parameters (dimensionless) -------------------------- #
A = 0.5                 # cylinder radius
RHO = 1.0
U0 = 1.0                # stream amplitude
OMEGA = 4.0             # -> KC = U0/(Omega a) = 0.5 (potential flow regime)
PHASES = 32             # samples per period (2nd-order d/dt)
TARGET = 0.03           # D1 target gate
OWNER_GATE = 0.15       # D1 owner gate
NULL_GATE = 1e-9        # D2

EXACT_CM = 2.0
BODY_AREA = math.pi * A * A
EXACT_FX_AMP = EXACT_CM * RHO * BODY_AREA * U0 * OMEGA


def harmonic_amplitude(series: torch.Tensor, dt: float, omega: float):
    """Least-squares ``a cos + b sin`` fit; returns (amplitude, phase_deg).

    Phase is measured against the driving ``u_dot = -U0 Omega sin(Omega t)``:
    a load in phase with ``u_dot`` has ``amp`` on the sine channel.  The returned
    phase is the lag of ``F`` behind ``u_dot`` in degrees (``0`` = in phase).
    """
    t = torch.arange(series.shape[0], dtype=series.dtype) * dt
    design = torch.stack((torch.cos(omega * t), torch.sin(omega * t),
                          torch.ones_like(t)), dim=-1)
    sol = torch.linalg.lstsq(design, series.reshape(-1, 1)).solution[:, 0]
    amp = float(torch.hypot(sol[0], sol[1]))
    f_dot = torch.atan2(sol[0], sol[1])          # F = amp sin(wt + f_dot)
    phase = math.degrees(f_dot) + 180.0          # u_dot = -sin(wt) -> +180 deg
    phase = (phase + 180.0) % 360.0 - 180.0
    return amp, phase


def one_config(radius: float, dx: float, n_phases: int = PHASES) -> dict:
    """Sweep one phase per sample over the analytic unsteady field."""
    centre = torch.zeros(2, dtype=DT)
    band = 1.0 * dx
    pos = annulus_lattice(centre, A + 0.5 * dx, radius + band, dx)
    if pos.shape[0] == 0:
        raise RuntimeError("empty annulus lattice")
    dt = (2.0 * math.pi / OMEGA) / n_phases
    rows = {"force": [], "mean": [], "naive": [], "steady": [],
            "displaced": [], "unsteady": [], "flux": [], "press": [],
            "mom_pert": [], "mom_total": [], "exact_udot": []}
    for k in range(n_phases):
        t = k * dt
        u = U0 * math.cos(OMEGA * t)
        udot = -U0 * OMEGA * math.sin(OMEGA * t)
        vel, pres = unsteady_potential_flow_cylinder(pos, A, u, udot, RHO,
                                                     centre=centre)
        frame_vel = torch.tensor([u, 0.0], dtype=DT)
        frame_acc = torch.tensor([udot, 0.0], dtype=DT)
        rows["mom_pert"].append(cv_momentum(pos, vel, RHO, centre, radius, dx,
                                            frame_vel=frame_vel))
        rows["mom_total"].append(cv_momentum(pos, vel, RHO, centre, radius, dx))
        rows["exact_udot"].append(float(udot))
        rows["_state"] = (pos, vel, pres, frame_vel, frame_acc, t)
    rate = d_dt_periodic(torch.stack(rows["mom_pert"]), dt, periodic=True)
    rate_total = d_dt_periodic(torch.stack(rows["mom_total"]), dt, periodic=True)
    dt_max = float(torch.linalg.norm(rate, dim=1).max())
    for k in range(n_phases):
        t = k * dt
        u = U0 * math.cos(OMEGA * t)
        udot = -U0 * OMEGA * math.sin(OMEGA * t)
        vel, pres = unsteady_potential_flow_cylinder(pos, A, u, udot, RHO,
                                                     centre=centre)
        frame_vel = torch.tensor([u, 0.0], dtype=DT)
        frame_acc = torch.tensor([udot, 0.0], dtype=DT)
        out = control_volume_force_unsteady(
            pos, vel, pres, RHO, centre, radius, dx,
            momentum_rate=rate[k], frame_vel=frame_vel, frame_acc=frame_acc,
            body_area=BODY_AREA, momentum_rate_total=rate_total[k])
        out_m = control_volume_force_unsteady(
            pos, vel, pres, RHO, centre, radius, dx,
            momentum_rate=rate[k], frame_vel=frame_vel, frame_acc=frame_acc,
            body_area=BODY_AREA, reconstruct="mean")
        steady = control_volume_force(pos, vel, pres, RHO, centre, radius, dx)
        rows["force"].append(out["force"])
        rows["mean"].append(out_m["force"])
        rows["naive"].append(out["force_naive"])
        rows["steady"].append(steady["force"])
        rows["displaced"].append(out["force_displaced_mass"].clone())
        rows["unsteady"].append(out["force_unsteady"].clone())
        rows["flux"].append(out["force_momentum_flux"].clone())
        rows["press"].append(out["force_pressure"].clone())
    series = {k: torch.stack(v) for k, v in rows.items()
              if isinstance(v, list) and k != "exact_udot"}
    res = {
        "radius_over_a": radius / A,
        "dx_over_a": dx / A,
        "n_particles": int(pos.shape[0]),
        "coverage": float(out["coverage"]),
        "n_bins": int(out["n_bins"]),
        "phases": n_phases,
        "dt": dt,
        "exact_fx_amp": EXACT_FX_AMP,
        "dt_truncation_rel": 1.0 - math.sin(OMEGA * dt / 2.0) / (OMEGA * dt / 2.0),
        "momentum_rate_scale": dt_max,
    }
    for name, series_t in series.items():
        fx = harmonic_amplitude(series_t[:, 0], dt, OMEGA)
        fy = harmonic_amplitude(series_t[:, 1], dt, OMEGA)
        res[name] = {
            "fx_amp": fx[0],
            "fx_amp_rel_err": (fx[0] - EXACT_FX_AMP) / EXACT_FX_AMP,
            "fx_phase_deg": fx[1],
            "fy_amp": fy[0],
            "fy_amp_rel_err": fy[0] / EXACT_FX_AMP,
            "cm": fx[0] / (RHO * BODY_AREA * U0 * OMEGA),
        }
    # phase-resolved series, so a figure can show the decomposition without
    # re-running anything (values, not fitted amplitudes)
    res["series"] = {
        "t": [k * dt for k in range(n_phases)],
        "u": [U0 * math.cos(OMEGA * k * dt) for k in range(n_phases)],
        "u_dot": [float(x) for x in rows["exact_udot"]],
    }
    for name in ("force", "displaced", "unsteady", "flux", "press", "naive",
                 "steady"):
        res["series"][name] = series[name].tolist()
    return res


def gate_u0_null() -> dict:
    """D2: accelerating uniform stream, no body -> force must vanish.

    The whole fluid moves rigidly with the free stream, so ``u' = 0`` and
    ``p' = 0`` by construction: every term of the perturbation form is zero
    *except* the two uniform pieces, which must cancel analytically
    (``-rho A_fluid a_f + rho pi R**2 a_f`` with ``A_body = 0``).  If the
    cancellation is mis-wired -- e.g. the ``pi R**2 a_f`` surface piece omitted,
    or ``A_fluid`` taken as the sampled lattice area instead of the analytic
    ``pi R**2`` -- the residual is ``O(rho R**2 a_f)`` and this leg fails by
    many orders of magnitude.
    """
    centre = torch.zeros(2, dtype=DT)
    rows = []
    worst = 0.0
    for radius in (1.0, 2.0):
        for dx in (0.05, 0.025):
            pos = annulus_lattice(centre, 0.2, radius + 1.0 * dx, dx)
            dt = (2.0 * math.pi / OMEGA) / PHASES
            scale = RHO * radius ** 2 * U0 * OMEGA
            mom = []
            for k in range(PHASES):
                t = k * dt
                u = U0 * math.cos(OMEGA * t)
                vel = torch.tensor([u, 0.0], dtype=DT).expand_as(pos).clone()
                mom.append(cv_momentum(pos, vel, RHO, centre, radius, dx,
                                       frame_vel=torch.tensor([u, 0.0], dtype=DT)))
            rate = d_dt_periodic(torch.stack(mom), dt, periodic=True)
            leg = 0.0
            for k in range(PHASES):
                t = k * dt
                u = U0 * math.cos(OMEGA * t)
                udot = -U0 * OMEGA * math.sin(OMEGA * t)
                vel = torch.tensor([u, 0.0], dtype=DT).expand_as(pos).clone()
                pres = -RHO * udot * pos[:, 0]
                out = control_volume_force_unsteady(
                    pos, vel, pres, RHO, centre, radius, dx,
                    momentum_rate=rate[k],
                    frame_vel=torch.tensor([u, 0.0], dtype=DT),
                    frame_acc=torch.tensor([udot, 0.0], dtype=DT),
                    body_area=0.0)
                leg = max(leg, float(torch.linalg.norm(out["force"])) / scale)
                cov = float(out["coverage"])
            worst = max(worst, leg)
            rows.append({"radius": radius, "dx": dx,
                         "n_particles": int(pos.shape[0]),
                         "coverage": cov,
                         "null_rel_to_scale": leg,
                         "complete_surface": bool(cov == 1.0)})
    return {"worst_null_rel": worst, "rows": rows,
            "gate": NULL_GATE, "pass": bool(worst < NULL_GATE)}


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--out", default=str(HERE / "unsteady_farfield.json"))
    ap.add_argument("--phases", type=int, default=PHASES)
    ap.add_argument("--only", default="",
                    help="comma list of 'R_over_a:dx_over_a' configs to run "
                         "(default: the full predeclared sweep)")
    ap.add_argument("--merge", action="store_true",
                    help="keep rows already present in --out and recompute the summary")
    args = ap.parse_args(argv)

    results = {"parameters": {
        "a": A, "rho": RHO, "u0": U0, "omega": OMEGA,
        "kc": U0 / (OMEGA * A), "phases": args.phases,
        "exact_cm": EXACT_CM, "exact_fx_amp": EXACT_FX_AMP,
        "gates": {"target": TARGET, "owner": OWNER_GATE, "null": NULL_GATE},
    }}
    existing = {}
    if args.merge and Path(args.out).exists():
        existing = json.loads(Path(args.out).read_text())
    results["u0_null"] = existing.get("u0_null") or gate_u0_null()
    print(f"U0 null          worst rel = {results['u0_null']['worst_null_rel']:.3e}"
          f"  pass={results['u0_null']['pass']}")
    grid = [(3.0, 0.1), (3.0, 0.05), (3.0, 0.025),
            (5.0, 0.1), (5.0, 0.05), (10.0, 0.1), (10.0, 0.05)]
    if args.only:
        grid = [tuple(float(v) for v in item.split(":"))
                for item in args.only.split(",") if item]
    rows = [r for r in existing.get("u1_oscillating_stream", [])
            if (r["radius_over_a"], r["dx_over_a"]) not in set(grid)]
    for r_over_a, dx_over_a in grid:
        r = one_config(r_over_a * A, dx_over_a * A, args.phases)
        rows.append(r)
        print(f"U1 R/a={r_over_a:4.1f} dx/a={dx_over_a:5.3f} "
              f"n={r['n_particles']:6d} cov={r['coverage']:.2f} | "
              f"pert {100*r['force']['fx_amp_rel_err']:+7.3f}%  "
              f"mean {100*r['mean']['fx_amp_rel_err']:+7.3f}%  "
              f"naive {100*r['naive']['fx_amp_rel_err']:+9.2f}%  "
              f"steady {100*r['steady']['fx_amp_rel_err']:+9.2f}%  "
              f"C_M {r['force']['cm']:.4f}")
    results["u1_oscillating_stream"] = rows

    def worst(arm, field="fx_amp_rel_err"):
        return max(abs(r[arm][field]) for r in rows)

    summary = {
        "worst_perturbation": worst("force"),
        "worst_perturbation_at_benchmark_scale": max(
            abs(r["force"]["fx_amp_rel_err"]) for r in rows
            if r["dx_over_a"] <= 0.05),
        "worst_mean_reconstruction": worst("mean"),
        "worst_naive": worst("naive"),
        "worst_steady_only": worst("steady"),
        "worst_cross_component": max(r["force"]["fy_amp_rel_err"] for r in rows),
        "worst_phase_deg": max(abs(r["force"]["fx_phase_deg"]) for r in rows),
        "gates": {"target": TARGET, "owner": OWNER_GATE},
    }
    summary["pass_target"] = bool(summary["worst_perturbation_at_benchmark_scale"] <= TARGET)
    summary["pass_owner"] = bool(summary["worst_perturbation_at_benchmark_scale"] <= OWNER_GATE)
    results["summary"] = summary
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(results, indent=2, sort_keys=True))
    print(f"\nworst perturbation error (dx/a<=0.05): "
          f"{100*summary['worst_perturbation_at_benchmark_scale']:.3f}% "
          f"target<{100*TARGET:.0f}% owner<{100*OWNER_GATE:.0f}% "
          f"-> target={summary['pass_target']} owner={summary['pass_owner']}")
    print(f"worst steady-only (round-3 arm)       : {100*summary['worst_steady_only']:.1f}%")
    print(f"worst naive (unsplit)                 : {100*summary['worst_naive']:.1f}%")
    print(f"worst cross-component F_y             : {100*summary['worst_cross_component']:.4f}%")
    print(f"worst phase lag                       : {summary['worst_phase_deg']:.3f} deg")
    print(f"json -> {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())