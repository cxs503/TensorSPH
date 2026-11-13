"""Verification of the exact Tait-consistent hydrostatic solution and of the
CASE-HYDROSTATIC error decomposition.

The closed form in :mod:`tensorsph.eos_hydrostatics` is a *claim* ("this is the
exact solution of ``dp/dy = -rho(p) g`` for the Tait EOS").  It is checked here
against an independent numerical integration of the same ODE (RK4, different
code path), against the repository's own EOS helper, and against the shipped
case evidence.  The decomposition artifact is checked for *reproducibility*
(an artifact that disagrees with its own recomputation fails), not merely for
existence.
"""
from __future__ import annotations

import importlib.util
import json
import math
from pathlib import Path
import sys

import pytest
import torch

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))

from tensorsph.eos_hydrostatics import (GAMMA_WATER, acoustic_mach,  # noqa: E402
                                        acoustic_parameter, bulk_modulus,
                                        compressibility_bias_pct,
                                        eos_hydrostatic_density,
                                        eos_hydrostatic_pressure,
                                        pointwise_eos_pressure, tait_density,
                                        tait_pressure)

CHANNEL = REPO / "tests" / "data" / "case_hydrostatic"
DECOMP = REPO / "tests" / "data" / "case_hydrostatic_eos" / "decomposition.json"
DEPTH = 0.4
RHO0 = 1000.0
G = 9.81


# --------------------------------------------------------------------------- #
# Independent numerical reference (RK4 on the barotropic balance)
# --------------------------------------------------------------------------- #

def _rk4_profile(depths, depth=DEPTH, rho0=RHO0, sound_speed=20.0,
                 gamma=GAMMA_WATER, gravity=G, steps=200_000):
    """Integrate ``dp/ds = rho(p) g`` (``s = depth - y``, ``p(0) = 0``) with RK4."""
    b = rho0 * sound_speed ** 2 / gamma

    def deds(p, s=None):
        rho = rho0 * (1.0 + max(p, 0.0) / b) ** (1.0 / gamma)
        return rho * gravity

    ds = depth / steps
    grid_s = [0.0]
    grid_p = [0.0]
    p = 0.0
    for i in range(steps):
        k1 = deds(p)
        k2 = deds(p + 0.5 * ds * k1)
        k3 = deds(p + 0.5 * ds * k2)
        k4 = deds(p + ds * k3)
        p = p + ds / 6.0 * (k1 + 2 * k2 + 2 * k3 + k4)
        grid_s.append((i + 1) * ds)
        grid_p.append(p)
    out = []
    for y in depths:
        s = depth - y
        # linear interpolation on the fine grid (the step is 2e-6 m)
        idx = s / ds
        i0 = min(int(idx), steps - 1)
        frac = idx - i0
        out.append(grid_p[i0] * (1 - frac) + grid_p[i0 + 1] * frac)
    return out


@pytest.mark.parametrize("sound_speed,gamma", [(5.0, 7.0), (20.0, 7.0),
                                               (30.0, 7.0), (20.0, 1.4)])
def test_closed_form_solves_the_barotropic_balance(sound_speed, gamma):
    """Closed form == independent RK4 integration of ``dp/dy = -rho(p) g``."""
    depths = [0.02, 0.1, 0.2, 0.3, 0.39, DEPTH]
    numeric = _rk4_profile(depths, sound_speed=sound_speed, gamma=gamma)
    for y, p_num in zip(depths, numeric):
        p_exact = float(eos_hydrostatic_pressure(y, DEPTH, sound_speed=sound_speed,
                                                 gamma=gamma))
        assert p_num == pytest.approx(p_exact, rel=1e-9, abs=1e-7)


def test_pressure_gradient_matches_local_density():
    """``dp/dy == -rho(p) g`` by central differences of the closed form itself."""
    c0 = 20.0
    ys = torch.tensor([0.05, 0.15, 0.25, 0.35], dtype=torch.float64)
    eps = 1e-6
    p_up = eos_hydrostatic_pressure(ys + eps, DEPTH, sound_speed=c0)
    p_dn = eos_hydrostatic_pressure(ys - eps, DEPTH, sound_speed=c0)
    dpdy = (p_up - p_dn) / (2 * eps)
    rho = tait_density(eos_hydrostatic_pressure(ys, DEPTH, sound_speed=c0),
                       sound_speed=c0)
    assert torch.allclose(dpdy, -rho * G, rtol=1e-6)


def test_eos_identity_pressure_of_closed_form_density():
    """``p(rho_eos(y)) == p_eos(y)``: the profile is on the EOS surface."""
    ys = torch.linspace(0.0, DEPTH, 17, dtype=torch.float64)
    rho = eos_hydrostatic_density(ys, DEPTH, sound_speed=20.0)
    p_back = tait_pressure(rho, sound_speed=20.0)
    p = eos_hydrostatic_pressure(ys, DEPTH, sound_speed=20.0)
    assert torch.allclose(p_back, p, rtol=1e-10, atol=1e-7)
    # the column is compressed: density >= rho0, strictly below the surface
    assert float(rho.max()) > RHO0
    assert torch.all(rho >= RHO0 - 1e-12)


@pytest.mark.parametrize("sound_speed", [1e4, 1e6])
def test_incompressible_limit(sound_speed):
    ys = torch.linspace(0.0, DEPTH, 9, dtype=torch.float64)
    ideal = RHO0 * G * (DEPTH - ys)
    p = eos_hydrostatic_pressure(ys, DEPTH, sound_speed=sound_speed)
    # float64 keeps ~1e-16 relative; the residual here is the
    # cancellation budget of ``B * expm1(...)`` at these huge c0
    assert torch.allclose(p, ideal, rtol=1e-7, atol=1e-9)


def test_free_surface_gauge_and_monotonicity():
    assert float(eos_hydrostatic_pressure(DEPTH, DEPTH, sound_speed=20.0)) == 0.0
    assert float(eos_hydrostatic_density(DEPTH, DEPTH, sound_speed=20.0)) == RHO0
    ys = torch.linspace(0.0, DEPTH, 41, dtype=torch.float64)
    p = eos_hydrostatic_pressure(ys, DEPTH, sound_speed=8.0)  # low c: high Mach
    assert torch.all(p[1:] < p[:-1])
    assert float(p.min()) >= 0.0
    # valid for any physical column: the branch factor stays > 1
    assert float((1 + (GAMMA_WATER - 1) * acoustic_parameter(0.0, DEPTH, 2.0))) > 1.0


def test_bias_is_half_g_h_over_c2_and_gamma_independent():
    """Leading order: ``bias % -> 50 g h / c0**2``; log-log slope in x is 1.

    ``gamma`` does not enter at first order (checked by the exact second-order
    coefficient ``1 + (2 - gamma) x / 3`` below), which is why the bias is a
    property of the acoustic Mach number and not of the liquid's stiffness.
    """
    y = 0.0
    xs, bs = [], []
    for c0 in (400.0, 800.0, 1600.0, 3200.0):
        xs.append(math.log(float(acoustic_parameter(y, DEPTH, c0))))
        bs.append(math.log(float(compressibility_bias_pct(y, DEPTH, sound_speed=c0))))
    slope = (bs[-1] - bs[0]) / (xs[-1] - xs[0])
    assert slope == pytest.approx(1.0, abs=0.02)
    # absolute level at small x, gamma-free to first order
    for c0 in (400.0, 800.0):
        x = float(acoustic_parameter(y, DEPTH, c0))
        for gamma in (1.4, 7.0):
            bias = float(compressibility_bias_pct(y, DEPTH, sound_speed=c0, gamma=gamma))
            # first order only: the gamma term below is O(x) relative
            assert bias / (50.0 * x) == pytest.approx(1.0, abs=1e-4)
    # the second-order coefficient carries the only gamma dependence
    for c0, tol in ((400.0, 1e-7), (20.0, 1e-2)):
        x = float(acoustic_parameter(y, DEPTH, c0))
        for gamma in (1.4, 4.0, 7.0):
            bias = float(compressibility_bias_pct(y, DEPTH, sound_speed=c0, gamma=gamma))
            predicted = 50.0 * x * (1.0 + (2.0 - gamma) * x / 3.0)
            assert bias == pytest.approx(predicted, rel=tol)
    # gamma sensitivity is weak at small x (Mach 0.004) and O(x) at Mach 0.14
    for c0, bound in ((400.0, 1e-4), (20.0, 3e-2)):
        b_water = float(compressibility_bias_pct(y, DEPTH, sound_speed=c0, gamma=7.0))
        b_gas = float(compressibility_bias_pct(y, DEPTH, sound_speed=c0, gamma=1.4))
        assert abs(b_water / b_gas - 1.0) < bound


def test_case_configuration_bias_consumes_half_the_tolerance():
    """At ``c0 = 20 m/s`` the bottom bias is ~0.5 % -- half of the 1 % gate."""
    bias = float(compressibility_bias_pct(0.0, DEPTH, sound_speed=20.0))
    assert 0.40 < bias < 0.55
    assert acoustic_mach(DEPTH, 20.0) == pytest.approx(0.14, abs=0.005)


def test_pointwise_helper_returns_the_ideal_reference_identically():
    """EOS inversion + EOS cancel: the first-order helper *is* ``rho0 g h``.

    This is the load-bearing reason that helper must not be used as a
    verification reference for pressure: it can never see the compressibility
    bias, however small it is made.
    """
    ys = torch.tensor([0.0, 0.1, 0.25, 0.39], dtype=torch.float64)
    for c0 in (5.0, 20.0, 30.0):
        ideal = RHO0 * G * (DEPTH - ys)
        assert torch.allclose(pointwise_eos_pressure(ys, DEPTH, sound_speed=c0),
                              ideal, rtol=1e-12, atol=1e-9)
        # while the exact equilibrium is strictly above it below the surface
        assert bool((eos_hydrostatic_pressure(ys, DEPTH, sound_speed=c0) > ideal).all())


def test_repo_first_order_helper_agrees_with_module():
    """``droplet_impact.hydrostatic_density`` vs the exact equilibrium density."""
    from types import SimpleNamespace

    from tensorsph.droplet_impact import hydrostatic_density

    cfg = SimpleNamespace(rho0=RHO0, sound_speed=20.0, gamma=GAMMA_WATER,
                          pool_depth=DEPTH, gravity=G)
    ys = torch.tensor([0.0, 0.1, 0.2, 0.3], dtype=torch.float64)
    rho_helper = hydrostatic_density(ys, cfg)
    rho_exact = eos_hydrostatic_density(ys, DEPTH, sound_speed=20.0)
    rel = float(((rho_helper - rho_exact) / rho_exact).abs().max())
    assert rel < 1e-4  # first-order density helper: O(x**2) off the exact profile
    devs = []
    for c0 in (20.0, 10.0):
        cfg.sound_speed = c0
        rho_h = hydrostatic_density(ys, cfg)
        rho_e = eos_hydrostatic_density(ys, DEPTH, sound_speed=c0)
        devs.append(float(((rho_h - rho_e) / rho_e).abs().max()))
    assert devs[1] > devs[0]  # grows as the acoustic parameter grows
    # ... but its *pressure* is the ideal reference exactly (see the dedicated
    # test): density agreement does not make it an equilibrium solution.


def test_the_case_gate_reference_was_not_changed():
    """Adding the exact reference must not silently redefine the case's gate."""
    from tensorsph import case_hydrostatic as ch

    assert ch.TOLERANCE_PCT == 1.0
    assert ch.REFERENCE["formula"] == "p(y) = rho0 * g * (H - y)"
    ys = torch.tensor([0.1, 0.3], dtype=torch.float64)
    assert torch.allclose(ch.analytic_pressure(ys), RHO0 * G * (DEPTH - ys))
    # the two references differ by the compressibility bias at c0 = 20 -- i.e.
    # the case measures the ideal reference on purpose, and the difference is
    # quantified here rather than assumed to be zero
    bias = compressibility_bias_pct(ys, DEPTH, sound_speed=20.0)
    assert bool((bias > 0.1).all())


# --------------------------------------------------------------------------- #
# Shipped evidence: tie the decomposition to the run files it claims to read
# --------------------------------------------------------------------------- #

def _harness():
    path = REPO / "benchmarks" / "case_hydrostatic" / "eos_decomposition.py"
    spec = importlib.util.spec_from_file_location("eos_decomposition_harness", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _shipped_runs():
    report = json.loads((CHANNEL / "report.json").read_text())
    assert report["case_id"] == "CASE-HYDROSTATIC"
    runs = []
    for path in sorted((CHANNEL / "runs").glob("*.json")):
        rec = json.loads(path.read_text())
        if rec.get("kind") == "dynamic_time_mean":
            runs.append(rec)
    assert runs, "no dynamic runs in the shipped evidence channel"
    return runs


def test_shipped_ideal_reference_is_reproducible_from_band_centres():
    """The gate quantity is recomputable: ``p_ideal = rho0 g (H - y_band)``."""
    for rec in _shipped_runs():
        ref = rec["reference"]
        depth = rec["config"]["depth"]
        for band in rec["profile"]:
            y = band["y"]
            p_ideal = ref["rho0"] * ref["g"] * (depth - y)
            assert band["pressure_analytic"] == pytest.approx(p_ideal, rel=1e-9)
            err = (band["pressure"] - p_ideal) / p_ideal * 100.0
            assert err == pytest.approx(band["rel_error_pct"], rel=1e-9, abs=1e-12)
        gated = [b for b in rec["profile"] if b.get("gated", not b["free_surface"])]
        worst = max(abs(b["rel_error_pct"]) for b in gated)
        assert worst == pytest.approx(rec["gate"]["max_band_error_pct"], rel=1e-9)


def _close(a, b, rel=1e-12):
    """Deep compare that tolerates last-bit float drift but nothing else."""
    if isinstance(a, dict):
        return isinstance(b, dict) and a.keys() == b.keys() and all(
            _close(a[k], b[k], rel) for k in a)
    if isinstance(a, list):
        return isinstance(b, list) and len(a) == len(b) and all(
            _close(x, y, rel) for x, y in zip(a, b))
    if isinstance(a, bool) or isinstance(b, bool):
        return a == b
    if isinstance(a, (int, float)) and isinstance(b, (int, float)):
        if a == b:
            return True
        return abs(a - b) <= rel * max(abs(a), abs(b), 1e-300)
    return a == b


def test_decomposition_artifact_is_current_and_internally_consistent():
    """Artifact == recomputation from the shipped runs; channel gate value ties."""
    harness = _harness()
    recomputed = harness.build()
    shipped = json.loads(DECOMP.read_text())
    assert _close(shipped, recomputed), (
        "eos_decomposition.json is stale; re-run "
        "`python benchmarks/case_hydrostatic/eos_decomposition.py`")
    channel = json.loads((CHANNEL / "channel.json").read_text())
    gate_run = next(r for r in shipped["runs"] if r["config_name"] == channel["gate_config"])
    assert gate_run["gate_shipped_pct"] == pytest.approx(channel["gate_value_pct"], rel=1e-12)


def test_decomposition_closes_the_gate_error_budget():
    """``err_ideal = bias + residual`` (to first order) on every gated band."""
    shipped = json.loads(DECOMP.read_text())
    for run in shipped["runs"]:
        if abs(run["gate_shipped_pct"]) > 20.0:  # the g-off negative control
            gated = [b for b in run["bands"] if b["gated"]]
            assert all(b["err_ideal_pct"] < -50.0 for b in gated)
            continue
        gated = [b for b in run["bands"] if b["gated"]]
        assert gated
        for b in gated:
            # bias is strictly positive below the free surface
            assert b["bias_pct"] > 0.0
            assert b["p_eos"] > b["p_ideal"]
            # residual is the rest of the gate error, up to O(bias**2)
            assert (b["residual_pct"] + b["bias_pct"]) == pytest.approx(
                b["err_ideal_pct"], abs=0.02)
            assert b["residual_pct"] == pytest.approx(
                (b["p_sph"] - b["p_eos"]) / b["p_eos"] * 100.0, rel=1e-9)
        assert run["mean_bias_pct"] > 0.0
        assert run["max_abs_residual_pct"] >= abs(run["mean_residual_pct"])


def test_gate_config_budget_reads_bias_plus_residual():
    """The gate value of the shipped gate run is decomposable with its own numbers."""
    shipped = json.loads(DECOMP.read_text())
    channel = json.loads((CHANNEL / "channel.json").read_text())
    gate_run = next(r for r in shipped["runs"] if r["config_name"] == channel["gate_config"])
    w = gate_run["worst_band_ideal"]
    assert w["bias_pct"] > 0.0
    assert gate_run["gate_shipped_pct"] == pytest.approx(abs(w["err_ideal_pct"]), rel=1e-12)
    # the bias alone is a non-trivial share of the 1 % limit
    assert 0.3 < w["bias_pct"] < 0.55
    # and the bias floor is resolution-independent (same c0 -> same bottom bias)
    same_c = [r for r in shipped["runs"] if r["sound_speed"] == gate_run["sound_speed"]]
    bottoms = {round(r["bias_bottom_pct"], 10) for r in same_c}
    assert len(bottoms) == 1


def test_residual_shrinks_with_the_averaging_window():
    """8 s > 4 s > 2 s: the residual is dominated by finite time averaging."""
    shipped = json.loads(DECOMP.read_text())
    by_name = {r["config_name"]: r for r in shipped["runs"]}
    windows = ["window_2s", "window_4s", "window_8s"]
    if not all(w in by_name for w in windows):
        pytest.skip("window ladder not in the shipped evidence")
    res = [by_name[w]["max_abs_residual_pct"] for w in windows]
    assert res[0] > res[1] > res[2], res
    assert by_name["window_8s"]["max_abs_residual_pct"] < 0.5


def test_higher_sound_speed_does_not_monotonically_improve_the_ideal_gate():
    """Falsifies the naive ``+c0 => closer to the ideal reference`` lever."""
    shipped = json.loads(DECOMP.read_text())
    by_name = {r["config_name"]: r for r in shipped["runs"]}
    if "sound_c30" not in by_name:
        pytest.skip("sound-speed sensitivity point not shipped")
    c30 = by_name["sound_c30"]
    gate = by_name["window_8s"]
    # bias shrinks  (Mach**-2 scaling) ...
    assert c30["bias_bottom_pct"] < gate["bias_bottom_pct"] / 2.0
    # ... yet the gate reading is worse than the gate configuration's
    assert abs(c30["gate_shipped_pct"]) > abs(gate["gate_shipped_pct"])
    # because the residual does not shrink with it (it turns negative)
    assert c30["mean_residual_pct"] < 0.0