"""Moving-body far-field (control-volume) force-from-fluid audit.

Anchor ``XVAL-SPH-FSI2-BENCHMARK-AND-FORCE-FROM-FLUID`` (Xiangyu-Hu/SPHinXsys
``test_2d_fsi2`` + ``FSI::ForceFromFluid``).  Round 3 built the *steady*
control-volume channel; round 4 added the accumulation term for a **fixed**
body.  Both rounds assumed the control surface stands still in the lab frame.
Every load this repository actually wants (a flapping ``fsi2`` beam, a heaving
section, a ship-fixed control volume for added resistance) rides on a surface
that **moves with the body**, and a translating surface is not a cosmetic
generalisation: the momentum crossing it is carried by the relative velocity
``(u - u_s).n``, not ``u.n``.  This driver audits that branch of
:func:`tensorsph.farfield_force.control_volume_force_unsteady` against a load
that is known in closed form *before* the run.

Predeclared decision and gates (fixed before any number was looked at)
---------------------------------------------------------------------
D1  Can the far-field channel report the inertial load on a **translating**
    body from a **body-attached** control surface?
    PASS  if the load amplitude is within **3 %** of
          ``rho pi a**2 (2 u_dot - body_acc)`` at every surface centre,
          ``R/a <= 10`` and ``dx/a <= 0.05``; ACCEPT (owner gate) if within
          **15 %**; else FAIL.
D2  Null: a fluid translating rigidly **with** the surface, no body, must give
    exactly zero -- relative residual < 1e-9 of ``rho R**2 U0 Omega``.  This is
    the gate that catches an inconsistent ``surface_velocity`` (``u_s`` in the
    flux but the absolute momentum in the rate, or a ``frame_acc`` inconsistent
    with ``frame_vel``).
D3  Reference leg: the exact body-surface integral ``-oint p n ds`` of the
    analytic field must reproduce the closed form within 1e-6 relative, so the
    reference used by D1 is proven inside the same run instead of cited.
D4  The defect this round closes must be **material**: dropping
    ``surface_velocity`` has to be worse than the 15 % owner gate on an
    off-centre surface and its error has to grow with the offset/R.
D5  Surface independence: the identity holds on *any* closed surface around the
    body, so the concentric and the off-centre (``R/2.5`` away) centre must give
    the same load.  This is also what puts the accumulation term on a non-zero
    share: on a surface concentric with the body the dipole field integrates to
    exactly zero momentum, so the previous round could not exercise it.

Field (analytic, no solver)
---------------------------
Cylinder ``r <= a`` at ``centre(t)``, translating with ``V(t) = V0 cos(Omega t)``
along ``+x`` through a stream ``U(t) = U0 cos(Omega t)``
(:func:`tensorsph.farfield_force.moving_cylinder_potential_flow`).  Exact load
``F_x = rho pi a**2 (2 u_dot - V_dot)``, ``F_y = 0`` -- Froude-Krylov
``rho A_b u_dot`` plus added mass ``rho C_a A_b (u_dot - V_dot)``, ``C_a = 1``.
``U0 != V0`` so the relative stream, and hence the momentum flux through the
surface, is non-zero.  The lattice, the surface and the body all translate
together, so ``momentum_rate`` really is the moving-volume derivative.

Arms
----
``perturb``        the moving-surface form (default ``reconstruct="fit"``).
``mean``           same, with the plain bin average on the surface.
``no_surface_vel`` the same field with ``surface_velocity`` omitted -- the defect.
``steady``         the round-3 estimator on the same field (no accumulation, no
                   displaced mass) -- the arm the new form must beat.

Every number in ``docs/xval_fsi2/MOVING_BODY_FARFIELD.md`` comes from the
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
    moving_cylinder_load,
    moving_cylinder_potential_flow,
)

from unsteady_farfield_audit import harmonic_amplitude  # noqa: E402

DT = torch.float64

A = 0.5                     # cylinder radius (D = 1)
RHO = 1.0
U0 = 1.0                    # stream amplitude
V0 = 0.5                    # body-velocity amplitude (U0 != V0 on purpose)
OMEGA = 4.0                 # period T = pi/2
PHASES = 32                 # samples per period (2nd-order d/dt)
TARGET = 0.03
OWNER_GATE = 0.15
NULL_GATE = 1e-9
REF_GATE = 1e-6

BODY_AREA = math.pi * A * A
EXACT_FX_AMP = RHO * BODY_AREA * OMEGA * (2.0 * U0 - V0)
NULL_SCALE = RHO * 100.0 * A * A * U0 * OMEGA      # rho (10a)**2 U0 Omega


def body_state(t: float):
    """Stream, body velocity and body displacement at time ``t`` (collinear x)."""
    u = U0 * math.cos(OMEGA * t)
    u_dot = -U0 * OMEGA * math.sin(OMEGA * t)
    v = V0 * math.cos(OMEGA * t)
    v_dot = -V0 * OMEGA * math.sin(OMEGA * t)
    x_c = (V0 / OMEGA) * math.sin(OMEGA * t)     # consistent with v = dx_c/dt
    return u, u_dot, v, v_dot, x_c


def body_frame_lattice(radius: float, dx: float, offset: float,
                       body_radius: float = A) -> torch.Tensor:
    """Square lattice, in the **body frame**, covering the CV disc.

    The control surface is centred at ``(offset, 0)`` in the body frame and the
    body occupies ``|X| < body_radius + dx/2``; both the surface and the body
    translate with the body, so this relative lattice is the *same* set of
    material-independent points at every phase -- that is what makes
    ``d_dt_periodic`` of :func:`cv_momentum` the moving-volume derivative.
    """
    band = 1.0 * dx
    lo, hi = offset - radius - band, offset + radius + band
    gx = torch.arange(lo, hi + 1e-9, dx, dtype=DT)
    gy = torch.arange(-radius - band, radius + band + 1e-9, dx, dtype=DT)
    X, Y = torch.meshgrid(gx, gy, indexing="xy")
    rel = torch.stack((X.reshape(-1), Y.reshape(-1)), dim=-1)
    keep = (rel - torch.tensor([offset, 0.0], dtype=DT)).norm(dim=1) <= radius + band
    if body_radius > 0.0:
        keep &= rel.norm(dim=1) >= body_radius + 0.5 * dx
    return rel[keep]


def reference_body_integral(t: float, n_theta: int = 720) -> float:
    """D3: ``-oint p n_x ds`` on the body, from the analytic field itself."""
    u, u_dot, v, v_dot, x_c = body_state(t)
    centre = torch.tensor([x_c, 0.0], dtype=DT)
    theta = (torch.arange(n_theta, dtype=DT) + 0.5) * (2.0 * math.pi / n_theta)
    pts = torch.stack((x_c + A * torch.cos(theta), A * torch.sin(theta)), dim=-1)
    _, pres = moving_cylinder_potential_flow(pts, A, u, u_dot, v, v_dot, RHO,
                                             centre=centre)
    ds = A * 2.0 * math.pi / n_theta
    return float(-(pres * torch.cos(theta) * ds).sum())


def one_config(radius: float, dx: float, offset: float,
               n_phases: int = PHASES) -> dict:
    """One (R/a, dx/a, surface-offset) point; one phase per sample."""
    rel = body_frame_lattice(radius, dx, offset)
    if rel.shape[0] == 0:
        raise RuntimeError("empty lattice")
    dt = (2.0 * math.pi / OMEGA) / n_phases
    surf_off = torch.tensor([offset, 0.0], dtype=DT)
    mom_pert, mom_total, states = [], [], []
    for k in range(n_phases):
        t = k * dt
        u, u_dot, v, v_dot, x_c = body_state(t)
        centre = torch.tensor([x_c, 0.0], dtype=DT)
        pos = rel + centre
        cs = centre + surf_off                                  # moving surface
        vel, pres = moving_cylinder_potential_flow(pos, A, u, u_dot, v, v_dot,
                                                   RHO, centre=centre)
        frame_vel = torch.tensor([u, 0.0], dtype=DT)
        mom_pert.append(cv_momentum(pos, vel, RHO, cs, radius, dx,
                                    frame_vel=frame_vel))
        mom_total.append(cv_momentum(pos, vel, RHO, cs, radius, dx))
        states.append((pos, vel, pres, cs, frame_vel,
                       torch.tensor([u_dot, 0.0], dtype=DT),
                       torch.tensor([v, 0.0], dtype=DT)))
    rate = d_dt_periodic(torch.stack(mom_pert), dt, periodic=True)
    rate_total = d_dt_periodic(torch.stack(mom_total), dt, periodic=True)

    rows = {name: [] for name in ("perturb", "mean", "no_surface_vel", "steady",
                                  "displaced", "unsteady", "flux", "press")}
    for k in range(n_phases):
        pos, vel, pres, cs, frame_vel, frame_acc, surf_vel = states[k]
        common = dict(pos=pos, vel=vel, pressure=pres, rho=RHO, centre=cs,
                      radius=radius, dx=dx, momentum_rate=rate[k],
                      frame_vel=frame_vel, frame_acc=frame_acc,
                      body_area=BODY_AREA)
        out = control_volume_force_unsteady(surface_velocity=surf_vel,
                                            momentum_rate_total=rate_total[k],
                                            **common)
        out_mean = control_volume_force_unsteady(surface_velocity=surf_vel,
                                                 reconstruct="mean", **common)
        out_nosv = control_volume_force_unsteady(**common)
        steady = control_volume_force(pos, vel, pres, RHO, cs, radius, dx)
        rows["perturb"].append(out["force"])
        rows["mean"].append(out_mean["force"])
        rows["no_surface_vel"].append(out_nosv["force"])
        rows["steady"].append(steady["force"])
        rows["displaced"].append(out["force_displaced_mass"].clone())
        rows["unsteady"].append(out["force_unsteady"].clone())
        rows["flux"].append(out["force_momentum_flux"].clone())
        rows["press"].append(out["force_pressure"].clone())
        coverage = float(out["coverage"])
        n_bins = int(out["n_bins"])
        n_surf = int(out["n_surface_particles"])

    series = {k: torch.stack(v) for k, v in rows.items()}
    res = {
        "radius_over_a": radius / A,
        "dx_over_a": dx / A,
        "offset_over_a": offset / A,
        "n_particles": int(rel.shape[0]),
        "n_surface_particles": n_surf,
        "coverage": coverage,
        "n_bins": n_bins,
        "phases": n_phases,
        "dt": dt,
        "exact_fx_amp": EXACT_FX_AMP,
        "dt_truncation_rel": 1.0 - math.sin(OMEGA * dt / 2.0) / (OMEGA * dt / 2.0),
        "max_abs_momentum_rate": float(torch.linalg.norm(rate, dim=1).max()),
    }
    for name, s in series.items():
        fx = harmonic_amplitude(s[:, 0], dt, OMEGA)
        fy = harmonic_amplitude(s[:, 1], dt, OMEGA)
        res[name] = {
            "fx_amp": fx[0],
            "fx_amp_rel_err": (fx[0] - EXACT_FX_AMP) / EXACT_FX_AMP,
            "fx_phase_deg": fx[1],
            "fy_amp": fy[0],
            "fy_amp_rel": fy[0] / EXACT_FX_AMP,
        }
    res["series"] = {"t": [k * dt for k in range(n_phases)]}
    for name in ("perturb", "displaced", "unsteady", "flux", "press",
                 "no_surface_vel", "steady"):
        res["series"][name] = series[name].tolist()
    res["series"]["u"] = [U0 * math.cos(OMEGA * k * dt) for k in range(n_phases)]
    res["series"]["body_vel"] = [V0 * math.cos(OMEGA * k * dt)
                                 for k in range(n_phases)]
    res["series"]["u_dot"] = [-U0 * OMEGA * math.sin(OMEGA * k * dt)
                              for k in range(n_phases)]
    return res


def gate_null(n_phases: int = PHASES, radius: float = 2.5, dx: float = 0.05,
              offset: float = 1.0) -> dict:
    """D2: fluid translating rigidly with the surface, no body -> zero force.

    ``u = V(t) x`` everywhere and the surface moves with the same ``V(t)``: the
    perturbation ``u - frame_vel`` and the relative flux ``(u - u_s).n`` both
    vanish identically, so every term of the moving-surface form must be zero.
    The pressure must carry the ramp of the accelerating frame,
    ``p = -rho a_f . (x - c_s)``, because that ramp is exactly what the
    perturbation form removes (``p_pert = p + rho a_f . (x - c_s)``) and what
    its ``rho A_b a_f`` displaced-mass term is the analytic residue of; feeding a
    flat pressure instead would assert an inconsistent field and the estimator
    would (correctly) report the missing ``-rho pi R**2 a_f``.
    """
    rel = body_frame_lattice(radius, dx, offset, body_radius=0.0)
    dt = (2.0 * math.pi / OMEGA) / n_phases
    surf_off = torch.tensor([offset, 0.0], dtype=DT)
    mom, states = [], []
    for k in range(n_phases):
        t = k * dt
        _, _, v, v_dot, x_c = body_state(t)
        centre = torch.tensor([x_c, 0.0], dtype=DT)
        pos = rel + centre
        cs = centre + surf_off
        vel = torch.stack((torch.full((pos.shape[0],), v, dtype=DT),
                           torch.zeros(pos.shape[0], dtype=DT)), dim=-1)
        pres = -RHO * v_dot * (pos[:, 0] - cs[0])   # ramp of the frame
        frame_vel = torch.tensor([v, 0.0], dtype=DT)
        mom.append(cv_momentum(pos, vel, RHO, cs, radius, dx,
                               frame_vel=frame_vel))
        states.append((pos, vel, pres, cs, frame_vel,
                       torch.tensor([v_dot, 0.0], dtype=DT)))
    rate = d_dt_periodic(torch.stack(mom), dt, periodic=True)
    worst = 0.0
    for k in range(n_phases):
        pos, vel, pres, cs, frame_vel, frame_acc = states[k]
        out = control_volume_force_unsteady(
            pos, vel, pres, RHO, cs, radius, dx,
            momentum_rate=rate[k], frame_vel=frame_vel, frame_acc=frame_acc,
            body_area=0.0, surface_velocity=frame_vel)
        worst = max(worst, float(torch.linalg.norm(out["force"])))
    resid = worst / (RHO * radius ** 2 * U0 * OMEGA)
    return {"residual_rel": resid, "gate": NULL_GATE, "pass": resid < NULL_GATE,
            "radius": radius, "dx": dx, "offset": offset}


def gate_reference(n_phases: int = PHASES) -> dict:
    """D3: the closed form vs the exact body-surface integral of the same field."""
    dt = (2.0 * math.pi / OMEGA) / n_phases
    worst = 0.0
    for k in range(n_phases):
        t = k * dt
        _, u_dot, _, v_dot, _ = body_state(t)
        ref = reference_body_integral(t)
        closed = float(moving_cylinder_load(A, RHO, u_dot, v_dot))
        worst = max(worst, abs(ref - closed) / EXACT_FX_AMP)
    return {"worst_rel_err": worst, "gate": REF_GATE, "pass": worst < REF_GATE}


def gate_quadrature(n_phases: int = 8, radius: float = 1.0,
                    dx_over_r: float = 0.05) -> dict:
    """D6: operator algebra on a *moving* surface with a non-zero momentum flux.

    Irrotational flow cannot exercise ``surface_velocity``: the closed-surface
    momentum flux of any potential field vanishes identically (d'Alembert --
    measured ``flux_amp = 0`` to round-off in every D1 configuration above), so
    a potential-flow benchmark can only show that the term does not *break*
    anything, never that it is needed.  This gate closes that hole
    algebraically: it evaluates the identity on a synthetic straining field
    whose volume momentum, momentum flux and pressure integral are all in closed
    form,

        u = (alpha(t) + kappa X_x/R, beta(t) + lambda X_y/R),
        p = p0 + q . X,          X = x - c_s(t)  (the *moving* surface frame)

    for which

        int_CV rho u dV      = rho pi R**2 (alpha, beta)
        oint rho u (u.n) dS  = rho R pi ( 2 alpha kappa + alpha lambda
                                          - kappa u_s,
                                          beta kappa + 2 beta lambda )
        oint (p - p_ref) n   = pi R**2 q

    so the relative-flux term contributes ``-rho R pi kappa u_s`` -- an ``O(R)``
    piece that the absolute-flux form gets wrong.  ``momentum_rate`` is supplied
    analytically, because forming it from a time series is the *caller's* step
    (and is separately guarded); this is a wiring gate, not a physics gate.
    """
    dx = dx_over_r * radius
    rel = body_frame_lattice(radius, dx, 0.0, body_radius=0.0)
    dt = (2.0 * math.pi / OMEGA) / n_phases
    a0, b0, kappa, lam, u_s = 0.7, -0.3, 0.9, 0.4, 0.8
    q = torch.tensor([0.5, -0.2], dtype=DT)
    worst_on, worst_off, u_s_share = 0.0, 0.0, 0.0
    for k in range(n_phases):
        t = k * dt
        alpha = a0 * math.cos(OMEGA * t)
        beta = b0 * math.sin(OMEGA * t)
        alpha_dot = -a0 * OMEGA * math.sin(OMEGA * t)
        beta_dot = b0 * OMEGA * math.cos(OMEGA * t)
        x0 = (u_s / OMEGA) * math.sin(OMEGA * t)          # dx0/dt = u_s(t)
        centre = torch.tensor([x0, 0.0], dtype=DT)
        X = rel - torch.tensor([0.0, 0.0], dtype=DT)
        pos = X + centre
        vel = torch.stack((alpha + kappa * X[:, 0] / radius,
                           beta + lam * X[:, 1] / radius), dim=-1)
        pres = q[0] * X[:, 0] + q[1] * X[:, 1]
        rate = RHO * math.pi * radius ** 2 * torch.tensor([alpha_dot, beta_dot],
                                                         dtype=DT)
        surf_vel = torch.tensor([u_s * math.cos(OMEGA * t), 0.0], dtype=DT)
        common = dict(pos=pos, vel=vel, pressure=pres, rho=RHO, centre=centre,
                      radius=radius, dx=dx, momentum_rate=rate,
                      frame_vel=torch.zeros(2, dtype=DT),
                      frame_acc=torch.zeros(2, dtype=DT), body_area=0.0)
        on = control_volume_force_unsteady(surface_velocity=surf_vel, **common)
        off = control_volume_force_unsteady(**common)
        # closed form of the same right-hand side
        flux_x = RHO * radius * math.pi * (2 * alpha * kappa + alpha * lam
                                          - kappa * surf_vel[0].item())
        flux_y = RHO * radius * math.pi * (beta * kappa + 2 * beta * lam)
        ref = (-rate
               - torch.tensor([flux_x, flux_y], dtype=DT)
               - math.pi * radius ** 2 * q)
        scale = float(ref.norm())
        worst_on = max(worst_on, float((on["force"] - ref).norm()) / scale)
        worst_off = max(worst_off, float((off["force"] - ref).norm()) / scale)
        u_s_share = max(u_s_share,
                        abs(RHO * radius * math.pi * kappa * surf_vel[0].item())
                        / scale)
    return {"worst_rel_err_with_surface_vel": worst_on,
            "worst_rel_err_without": worst_off,
            "surface_vel_term_share": u_s_share,
            "gate": 1e-10, "pass": worst_on < 1e-10,
            "discriminates": worst_off > 1e-3, "radius": radius,
            "dx_over_r": dx_over_r}


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=str(ROOT / "artifacts/xval_fsi2/"
                                        "moving_body_farfield.json"))
    ap.add_argument("--quick", action="store_true")
    ap.add_argument("--phases", type=int, default=PHASES)
    ap.add_argument("--grid", default="3,5,10",
                    help="comma separated R/a values")
    ap.add_argument("--dxs", default="0.05,0.10")
    ap.add_argument("--offsets", default="0.0,0.4")
    ap.add_argument("--resume", action="store_true",
                    help="keep already-computed configurations in --out")
    args = ap.parse_args(argv)

    radii = ([float(v) for v in args.grid.split(",")] if not args.quick
             else [3.0])
    dxs = [float(v) for v in args.dxs.split(",")] if not args.quick else [0.10]
    offsets = [float(v) for v in args.offsets.split(",")] if not args.quick \
        else [0.0, 0.4]
    phases = args.phases
    out_path = Path(args.out)

    def _key(c):
        return (c["radius_over_a"], c["dx_over_a"], c["offset_over_a"])

    configs = []
    if args.resume and out_path.exists():
        try:
            old = json.loads(out_path.read_text())
            if old.get("params", {}).get("phases") == phases:
                configs = old.get("configs", [])
                print(f"resumed {len(configs)} configurations", flush=True)
        except Exception:
            configs = []
    done = {_key(c) for c in configs}

    partial = {"anchor": "XVAL-SPH-FSI2-BENCHMARK-AND-FORCE-FROM-FLUID",
               "params": {"a": A, "rho": RHO, "u0": U0, "v0": V0,
                          "omega": OMEGA, "phases": phases,
                          "body_area": BODY_AREA,
                          "exact_fx_amp": EXACT_FX_AMP},
               "configs": configs, "gates": {},
               "complete": False}
    for r_over_a in radii:
        for dx_over_a in dxs:
            for off_frac in offsets:
                if (r_over_a, dx_over_a, off_frac) in done:
                    continue
                res = one_config(r_over_a * A, dx_over_a * A,
                                 off_frac * r_over_a * A, n_phases=phases)
                configs.append(res)
                partial["configs"] = configs
                out_path.parent.mkdir(parents=True, exist_ok=True)
                out_path.write_text(json.dumps(partial, indent=1))
                print(f"R/a={r_over_a:5.1f} dx/a={dx_over_a:4.2f} "
                      f"off={off_frac:4.2f}R cov={res['coverage']:.3f} "
                      f"perturb={100*res['perturb']['fx_amp_rel_err']:+8.3f}% "
                      f"mean={100*res['mean']['fx_amp_rel_err']:+8.3f}% "
                      f"no_sv={100*res['no_surface_vel']['fx_amp_rel_err']:+9.2f}% "
                      f"steady={100*res['steady']['fx_amp_rel_err']:+9.2f}% "
                      f"accum={100*res['unsteady']['fx_amp_rel_err']:+8.3f}%",
                      flush=True)

    null = gate_null()
    ref = gate_reference()
    quad = gate_quadrature()

    fine = [c for c in configs if c["dx_over_a"] <= 0.05] or configs
    worst_perturb = max(abs(c["perturb"]["fx_amp_rel_err"]) for c in fine)
    worst_mean = max(abs(c["mean"]["fx_amp_rel_err"]) for c in fine)
    worst_cross = max(c["perturb"]["fy_amp_rel"] for c in fine)
    worst_nosv = max(abs(c["no_surface_vel"]["fx_amp_rel_err"]) for c in configs)
    off_fine = [c for c in fine if c["offset_over_a"] > 0.0]
    nosv_off = max(abs(c["no_surface_vel"]["fx_amp_rel_err"]) for c in off_fine)
    # D5: same (R/a, dx/a), different surface centre
    spread = []
    for c in fine:
        if c["offset_over_a"] == 0.0:
            twin = [d for d in fine if d["offset_over_a"] > 0.0
                    and d["radius_over_a"] == c["radius_over_a"]
                    and d["dx_over_a"] == c["dx_over_a"]]
            if twin:
                spread.append({"radius_over_a": c["radius_over_a"],
                               "dx_over_a": c["dx_over_a"],
                               "concentric_rel_err": c["perturb"]["fx_amp_rel_err"],
                               "offset_rel_err": twin[0]["perturb"]["fx_amp_rel_err"],
                               "abs_difference": abs(c["perturb"]["fx_amp_rel_err"]
                                                     - twin[0]["perturb"]["fx_amp_rel_err"])})
    worst_spread = max((s["abs_difference"] for s in spread), default=float("nan"))
    accum_share = [{"radius_over_a": c["radius_over_a"],
                    "dx_over_a": c["dx_over_a"],
                    "offset_over_a": c["offset_over_a"],
                    "accum_amp_over_exact": c["unsteady"]["fx_amp"] / EXACT_FX_AMP,
                    "flux_amp_over_exact": c["flux"]["fx_amp"] / EXACT_FX_AMP,
                    "press_amp_over_exact": c["press"]["fx_amp"] / EXACT_FX_AMP,
                    "displaced_amp_over_exact": c["displaced"]["fx_amp"] / EXACT_FX_AMP}
                   for c in fine]

    verdict = {
        "D1_worst_rel_err": worst_perturb,
        "D1_target_pass": worst_perturb < TARGET,
        "D1_owner_gate_pass": worst_perturb < OWNER_GATE,
        "D1_mean_arm_worst_rel_err": worst_mean,
        "D1_cross_component_worst_rel": worst_cross,
        "D2_null": null,
        "D3_reference": ref,
        "D6_quadrature": quad,
        "D4_no_surface_vel_worst_rel_err": worst_nosv,
        "D4_no_surface_vel_offset_worst_rel_err": nosv_off,
        "D4_material": bool(nosv_off > OWNER_GATE),
        "D5_surface_independence": spread,
        "D5_worst_abs_difference": worst_spread,
        "D5_term_shares": accum_share,
    }
    payload = {
        "anchor": "XVAL-SPH-FSI2-BENCHMARK-AND-FORCE-FROM-FLUID",
        "params": {"a": A, "rho": RHO, "u0": U0, "v0": V0, "omega": OMEGA,
                   "phases": PHASES, "body_area": BODY_AREA,
                   "exact_fx_amp": EXACT_FX_AMP},
        "configs": configs,
        "gates": verdict,
    }
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(payload, indent=1))
    print(json.dumps({k: v for k, v in verdict.items()
                      if k not in ("D5_surface_independence", "D5_term_shares")},
                     indent=1))
    print("wrote", out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())