"""Static force balance: how much Laplace pressure can the discrete CSF force hold?

No time stepping. On the calibrated drop lattice of ``laplace_pressure_floor`` we
evaluate the CSF surface-tension acceleration and integrate the pressure gradient
it implies,

    Delta p_force = int_0^R rho a_r dr,

which must equal ``sigma * kappa = sigma / R`` if the discrete surface delta
function is normalised (``int delta_s dn = 1``). Reported together with the
Tait/clamped and closure-extrapolated core pressures on the same lattice.

The point of the measurement: after the free-surface pressure closure lands, the
residual Young-Laplace deficit must be attributed. If ``Delta p_force`` is short
of ``sigma/R`` by the same amount as the relaxed core pressure, the remaining
error is the force-side normalisation and *not* the pressure closure.

    PYTHONPATH=src python benchmarks/laplace_pressure_closure/analysis.py
"""
from __future__ import annotations

import importlib.util
import json
import math
from pathlib import Path

import torch

from tensorsph.droplet_impact import (
    Config,
    density_from_neighbor_pairs,
    free_surface_pressure,
    pair_kernels,
    tait_pressure,
)
from tensorsph.surface_tension import compute_surface_tension_force

torch.set_num_threads(2)

HERE = Path(__file__).resolve().parent
OUT = HERE / "report_force_balance.json"
N_BINS = 400
BIN_SENSITIVITY = (100, 200, 400, 800)


def floor_module():
    spec = importlib.util.spec_from_file_location(
        "lpf", HERE.parent / "laplace_pressure_floor" / "run.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def case(lpf, dx: float, c0: float = 20.0, n_bins: int = N_BINS) -> dict:
    R, SIGMA, RHO0 = lpf.RADIUS, lpf.SIGMA, lpf.RHO0
    dp_analytic = SIGMA / R
    pos, mass, h, pairs, bias, after = lpf.calibrated_droplet(dx)
    cfg = Config(dx=dx, h_factor=lpf.H_FACTOR, rho0=RHO0, sound_speed=c0,
                 gamma=lpf.GAMMA, gravity=0.0, viscosity=lpf.VISCOSITY,
                 surface_tension_coefficient=SIGMA, csf_scheme="consistent",
                 csf_curvature="fit_quadratic")
    rho = density_from_neighbor_pairs(pos, mass, h, pairs)
    acc_st = compute_surface_tension_force(pos, rho, mass, pairs, cfg)
    r = torch.linalg.vector_norm(pos, dim=1)
    rhat = pos / r.clamp_min(1e-30).unsqueeze(-1)
    a_r = (acc_st * rhat).sum(dim=1)

    # Radial force balance by the flux (divergence-theorem) estimator.
    #
    # dp/dr = rho a_r, so  Delta p = int_0^R rho a_r dr.  Estimating the
    # integrand by the mean of rho a_r over a radial shell is badly conditioned:
    # a shell holds a handful of particles, and the mean-field value times the
    # bin width is not a volume integral (the naive estimator drifts by a factor
    # two between 100 and 800 bins; see ``naive_mean_estimator`` below, kept as
    # the negative control).  The flux form does not have that problem: for an
    # axisymmetric field the force crossing an annulus of radius r is
    #     2 pi r f(r) dr = int_annulus f dA  ~=  sum_p f_p V_p ,
    # so each annulus contributes  sum_p f_p V_p / (2 pi r_c)  to Delta p.
    volume = mass / rho
    f_r = rho * a_r
    edges = torch.linspace(0.0, 1.1 * R, n_bins + 1, dtype=pos.dtype)
    dr = float(edges[1] - edges[0])
    profile, jump = [], 0.0
    for k in range(n_bins):
        m = (r >= edges[k]) & (r < edges[k + 1])
        r_c = float(0.5 * (edges[k] + edges[k + 1]))
        flux = float((f_r[m] * volume[m]).sum()) if bool(m.any()) else 0.0
        contribution = flux / (2.0 * math.pi * max(r_c, 0.5 * dr))
        jump += contribution
        profile.append({
            "r_over_R": r_c / R, "count": int(m.sum()),
            "flux_Pa_m": flux, "delta_p_contribution_Pa": contribution,
            "rho_a_r_mean_Pa_per_m": float(f_r[m].mean()) if bool(m.any()) else 0.0,
        })
    # negative control: the naive shell-mean estimator
    naive = 0.0
    for p_row in profile:
        naive += p_row["rho_a_r_mean_Pa_per_m"] * dr

    delta = pos[pairs[:, 0]] - pos[pairs[:, 1]]
    wij, _ = pair_kernels(delta, h)
    ii, jj = pairs.unbind(dim=1)
    p_tait = tait_pressure(rho, cfg)
    pressure_side = {}
    for name, kw in (("tait_clamped", None),
                     ("closure_all_donors", dict(surface_pressure_donors="all")),
                     ("closure_interior_layers2",
                      dict(surface_pressure_donors="interior",
                           surface_pressure_layers=2))):
        if kw is None:
            p = p_tait
            thresh = None
        else:
            c = Config(**{**cfg.__dict__, "surface_pressure_closure": True, **kw})
            p = free_surface_pressure(pos, rho, p_tait, delta, wij, ii, jj, c)
            thresh = c.surface_pressure_threshold
        core = r < 0.5 * R
        shell = (r >= 0.8 * R) & (r < 0.95 * R)
        pressure_side[name] = {
            "threshold": thresh,
            "p_core_Pa": float(p[core].mean()),
            "p_core_over_analytic": float(p[core].mean()) / dp_analytic,
            "p_shell_0p8_0p95_over_analytic": float(p[shell].mean()) / dp_analytic,
        }

    return {
        "dx_m": dx, "h_m": h, "n_particles": int(pos.shape[0]),
        "sound_speed": c0,
        "lattice_bias": bias, "lattice_bias_after_calibration": after,
        "analytic_dp_Pa": dp_analytic,
        "force_side": {
            # a_r is inward, so dp/dr = rho a_r < 0 and the jump read at the
            # core is minus the integral.
            "delta_p_force_Pa": -jump,
            "delta_p_force_over_analytic": -jump / dp_analytic,
            "deficit_pct": 100.0 * (-jump / dp_analytic - 1.0),
            "spurious_acc_max_m_per_s2": float(a_r.abs().max()),
            "naive_shell_mean_estimator_Pa": -naive,
            "profile": profile,
        },
        "pressure_side": pressure_side,
    }


def main():
    lpf = floor_module()
    report = {
        "benchmark": "laplace_pressure_closure",
        "question": "after the free-surface pressure closure, what is left of the "
                    "Young-Laplace deficit, and is it on the pressure side or on "
                    "the surface-tension force side?",
        "reference": {"sigma_N_per_m": lpf.SIGMA, "radius_m": lpf.RADIUS,
                      "rho0": lpf.RHO0, "analytic_dp_Pa": lpf.DP_ANALYTIC},
        "method": "int_0^R rho * a_r dr from a single CSF evaluation on the "
                  "calibrated lattice (no time stepping); N_BINS=%d" % N_BINS,
        "bin_sensitivity": {
            f"dx={dx:g}": {str(nb): round(case(lpf, dx, n_bins=nb)["force_side"]["delta_p_force_Pa"], 2)  # noqa: E501
                           for nb in BIN_SENSITIVITY} for dx in (0.002,)},
        "cases": [case(lpf, dx) for dx in (0.002, 0.0015)],
    }
    for c in report["cases"]:
        print(f"dx={c['dx_m']:g}: -int(rho a_r)dr = "
              f"{c['force_side']['delta_p_force_Pa']:+7.1f} Pa "
              f"({c['force_side']['delta_p_force_over_analytic']:.3f} sigma/R)", flush=True)
    OUT.write_text(json.dumps(report, indent=2))
    print("wrote", OUT)


if __name__ == "__main__":
    main()
