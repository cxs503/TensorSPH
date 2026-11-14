"""Tests for CASE-SPH-POISEUILLE-MORRIS (Morris 1997 low-Re channel).

The case's credibility rests on two independent legs, so both are tested:

* the *reference* -- the steady parabola and the start-up series are re-derived
  symbolically and certified numerically (a transcription error must show);
* the *measurement* -- the metric pipeline returns zero error on an exact
  profile, and the WCSPH + mirror-wall run agrees with the reference inside the
  predeclared 5 % band on a CI-sized configuration.
"""

from __future__ import annotations

import json
import math

import numpy as np
import pytest

from tensorsph.case_poiseuille_morris import (
    CASE_KEY,
    MorrisConfig,
    averaged_profile,
    build_channel,
    derive_reference_symbolically,
    evaluate,
    main,
    run_case,
    startup_profile,
    steady_profile,
    verify_series,
)

CFG = MorrisConfig()


# --------------------------------------------------------------------------- #
# Reference solution
# --------------------------------------------------------------------------- #

def test_symbolic_reference_has_zero_residuals():
    ref = derive_reference_symbolically()
    if not ref.get("available"):
        pytest.skip(f"sympy unavailable: {ref.get('reason')}")
    assert ref["ode_residual_is_zero"], ref["ode_residual_nu_uyy_plus_g"]
    assert ref["bc_both_zero"]
    assert ref["u_max_matches_gH2_over_8nu"]
    assert ref["tau_wall_matches_rho_gH_over_2"]
    assert ref["flow_rate_matches_gH3_over_12nu"]


def test_startup_series_is_the_exact_solution():
    out = verify_series(CFG)
    assert out["pass"], out
    assert out["initial_condition_residual_over_umax"] < 1e-6
    assert out["pde_residual_over_g"] < 1e-3
    # exp(-10) = 4.5e-5 of the leading mode is still alive at t = 10 tau
    assert out["steady_limit_residual_over_umax"] < 1e-4


def test_series_agrees_with_isph_channel_implementation():
    """The same closed form exists in ``isph_channel`` (scipy-gated)."""
    try:
        from tensorsph.isph_channel import analytic_startup_profile
    except Exception as exc:  # pragma: no cover - scipy not installed
        pytest.skip(f"isph_channel not importable: {exc}")
    import torch

    y = np.linspace(0.02, 0.98, 17) * CFG.height
    t = 0.3 * CFG.tau_diffusive
    mine = startup_profile(y, t, CFG)
    their = analytic_startup_profile(torch.tensor(y, dtype=torch.float64), t,
                                     CFG.height, CFG.body_force, CFG.nu).numpy()
    assert np.max(np.abs(mine - their)) / CFG.u_max < 1e-12


def test_averaged_profile_reduces_to_steady_for_a_settled_window():
    y = np.linspace(0.1, 0.9, 9) * CFG.height
    avg = averaged_profile(y, 8.0 * CFG.tau_diffusive, 10.0 * CFG.tau_diffusive, CFG)
    assert np.max(np.abs(avg - steady_profile(y, CFG))) / CFG.u_max < 1e-2
    # and it is a genuine average early in the transient: below the steady value
    early = averaged_profile(y, 0.0, 0.2 * CFG.tau_diffusive, CFG)
    assert np.all(early < steady_profile(y, CFG))
    # over [0, 0.2 tau] the channel is still accelerating: the average is far
    # below the steady peak but well above zero
    assert 0.01 < float(np.max(early)) / CFG.u_max < 0.2


def test_derived_dimensional_groups_are_consistent():
    cfg = MorrisConfig(reynolds=0.5, mu=2.0, height=2.0e-3, rho0=900.0)
    assert math.isclose(cfg.nu, cfg.mu / cfg.rho0)
    assert math.isclose(cfg.reynolds, cfg.rho0 * cfg.u_mean * cfg.height / cfg.mu)
    assert math.isclose(cfg.u_max, 1.5 * cfg.u_mean, rel_tol=1e-12)
    assert math.isclose(cfg.tau_wall, cfg.rho0 * cfg.body_force * cfg.height / 2.0)
    # Q from the profile must equal the closed form
    y = np.linspace(0.0, cfg.height, 200001)
    q = float(np.trapezoid(steady_profile(y, cfg), y)) if hasattr(np, "trapezoid") \
        else float(np.trapz(steady_profile(y, cfg), y))
    assert math.isclose(q, cfg.q_ref, rel_tol=1e-6)
    dims = cfg.dimensionless()
    assert dims["Re_mean"] == cfg.reynolds
    assert 0.0 < dims["mach_max"] < 0.1
    assert dims["courant_acoustic"] <= 0.3 + 1e-12


def test_channel_lattice_geometry():
    pos, mass = build_channel(MorrisConfig(resolution=8), "cpu")
    cfg = MorrisConfig(resolution=8)
    assert int(pos.shape[0]) == 8 * int(round(cfg.length / cfg.dx))
    assert math.isclose(float(mass[0]), cfg.rho0 * cfg.dx ** 2)
    y = pos[:, 1].numpy()
    assert y.min() > 0.0 and y.max() < cfg.height


# --------------------------------------------------------------------------- #
# Metrics
# --------------------------------------------------------------------------- #

def _exact_case(cfg: MorrisConfig) -> dict:
    """A synthetic 'measurement' that is exactly the reference solution."""
    n_rows = cfg.resolution
    y = np.array([0.5 * cfg.dx + k * cfg.dx for k in range(n_rows)])
    win = [cfg.t_end * (1.0 - cfg.average_fraction), cfg.t_end]
    u = averaged_profile(y, win[0], win[1], cfg)
    rows = [{"row": k, "y": float(y[k]), "y_over_H": float(y[k] / cfg.height),
             "u_mean": float(u[k]), "u_std_x": 0.0, "count": 8}
            for k in range(n_rows)]
    return {"profile": rows, "avg_window": win, "steady_drift": 0.0,
            "snapshots": [], "dt": cfg.dt, "steps": 10, "n_fluid": 8 * n_rows,
            "n_ghost": 16, "n_averaged_samples": 10, "wall_clock_s": 0.0}


def test_metrics_are_zero_on_the_exact_profile():
    cfg = MorrisConfig(resolution=16)
    ev = evaluate(_exact_case(cfg), cfg)
    assert abs(ev["gates"]["A_centre_line"]["value_pct"]) < 0.05
    assert ev["gates"]["B_profile_shape"]["value_pct"] < 0.05
    assert ev["gates"]["C_wall_shear"]["value_pct"] < 0.2
    # gate E is estimator-matched; the estimator's own bias against the closed
    # form (missing wall strips, ~1 %) is reported separately
    assert ev["gates"]["E_flow_rate"]["value_pct"] < 0.05
    assert -2.0 < ev["flow_rate_error_vs_closed_form_pct"] < 0.0
    assert ev["gates"]["D_wall_slip"]["value_pct"] < 0.05
    assert ev["pass"]


def test_metrics_flag_a_defective_profile():
    """A profile scaled by 12 % must fail gate A, and only through its value."""
    cfg = MorrisConfig(resolution=16)
    case = _exact_case(cfg)
    for r in case["profile"]:
        r["u_mean"] *= 1.12
    ev = evaluate(case, cfg)
    assert not ev["gates"]["A_centre_line"]["pass"]
    assert abs(ev["gates"]["A_centre_line"]["value_pct"] - 12.0) < 0.5


# --------------------------------------------------------------------------- #
# End to end
# --------------------------------------------------------------------------- #

def test_wcsph_mirror_channel_matches_the_exact_solution():
    cfg = MorrisConfig(resolution=8, t_end_over_tau=2.0)
    case = run_case(cfg)
    ev = evaluate(case, cfg)
    assert ev["gates"]["A_centre_line"]["value_pct"] < 5.0, ev["gates"]
    assert ev["gates"]["B_profile_shape"]["value_pct"] < 5.0, ev["gates"]
    assert ev["gates"]["C_wall_shear"]["value_pct"] < 5.0, ev["gates"]
    # the start-up snapshots are recorded at t/tau = 0.1, 0.2, 0.4
    assert len(case["snapshots"]) == 3
    assert [s["fraction_of_tau"] for s in case["snapshots"]] == [0.1, 0.2, 0.4]


def test_cli_writes_a_complete_artifact(tmp_path):
    out = tmp_path / "case"
    rc = main(["--quick", "--no-figures", "--out", str(out), "--threads", "2"])
    payload = json.loads((out / "results.json").read_text())
    assert payload["case_key"] == CASE_KEY
    assert payload["repro_cmd"].startswith("PYTHONPATH=src python -m tensorsph")
    entry = payload["ladder"]["entries"][0]
    assert entry["evaluation"]["n_fluid"] > 0
    assert entry["evaluation"]["n_ghost"] > 0
    assert len(entry["evaluation"]["nodes"]) >= 6
    node = {n["name"]: n for n in entry["evaluation"]["nodes"]}["u_max"]
    assert node["reference"] > 0.0 and node["got"] > 0.0
    assert payload["reference"]["series"]["pass"]
    assert rc == 0
