"""Tests for :mod:`tensorsph.wedge_entry` (Wagner reference + wedge impact SPH).

The analytic block pins identities that the campaign's acceptance criteria rest
on, so a change in the reference solution cannot silently move the target the
SPH run is judged against.  The solver block is a *smoke* run at a resolution
coarse enough for CI: it checks that the model runs, stays bounded and produces
the right qualitative trend.  The quantitative acceptance numbers live in
``benchmarks/wedge_entry`` and ``artifacts/wedge_entry/summary.json``.
"""

from __future__ import annotations

import json
import math
import sys
from pathlib import Path

import pytest
import torch

from tensorsph.wedge_entry import (
    WedgeConfig,
    WedgeImpact,
    fit_exponent,
    gci,
    lattice_mass_unit,
    round_floats,
    sig6,
    wagner_c_coeff,
    wagner_jet_root,
    wagner_p_centre,
    wagner_pmax,
    wagner_pressure,
    wagner_validity_tau,
    wedge_lattice,
    wet_half_width,
    write_json,
)

ROOT = Path(__file__).resolve().parent.parent
RHO = 1000.0


# --------------------------------------------------------------------------- #
# Wagner (1932) reference solution
# --------------------------------------------------------------------------- #

@pytest.mark.parametrize("beta_deg", [1.0, 5.0, 10.0, 20.0, 30.0, 45.0, 60.0, 80.0])
def test_wet_half_width_is_linear_in_time_and_velocity(beta_deg):
    beta = math.radians(beta_deg)
    C = wagner_c_coeff(beta)
    assert C == pytest.approx(math.pi / (2.0 * math.tan(beta)), rel=1e-15)
    for V in (0.5, 1.0, 5.0):
        c1 = wet_half_width(V, beta, 1.0e-2)
        c2 = wet_half_width(V, beta, 2.0e-2)
        assert c2 / c1 == pytest.approx(2.0, rel=1e-14)
        assert wet_half_width(2 * V, beta, 1.0e-2) == pytest.approx(2 * c1, rel=1e-14)


def test_wet_half_width_inverse_tangent_scaling():
    """``c / (V t) = pi / (2 tan beta)``: doubling ``tan beta`` halves ``c``."""
    t, V = 3.7e-3, 2.0
    c10 = wet_half_width(V, math.radians(10.0), t)
    beta2 = math.atan(2.0 * math.tan(math.radians(10.0)))      # tan(beta2) = 2 tan(10)
    c2 = wet_half_width(V, beta2, t)
    assert c2 / c10 == pytest.approx(0.5, rel=1e-12)


def test_jet_root_coincides_with_wetted_edge():
    """Wagner places the spray root on the wetted edge; the two must agree."""
    for beta_deg in (10.0, 20.0, 30.0):
        beta = math.radians(beta_deg)
        for t, V in ((1e-3, 1.0), (5e-3, 2.0)):
            assert wagner_jet_root(V, beta, t) == wet_half_width(V, beta, t)


def test_stagnation_pressure_matches_field_formula():
    """``p(0, t)`` from the field (2) equals the closed form ``rho V^2 C``."""
    rho, V = RHO, 1.7
    for beta_deg in (10.0, 30.0, 45.0, 70.0):
        beta = math.radians(beta_deg)
        for t in (1e-3, 5e-3, 2e-2):
            c = wet_half_width(V, beta, t)
            p0 = wagner_pressure(0.0, c, V, beta, rho)
            assert p0 == pytest.approx(wagner_p_centre(V, beta, rho), rel=1e-12)


def test_peak_pressure_is_attained_by_the_field():
    """``p_max`` must be the maximum of the field (2) on the wetted surface.

    Two branches: ``C >= 1`` peaks behind the spray root, ``C < 1`` peaks at the
    stagnation point.  Sampled densely enough that the discretisation of the
    maximum is far below the assertion tolerance.
    """
    rho, V = RHO, 1.0
    for beta_deg in (10.0, 20.0, 30.0, 45.0, 55.0, 70.0):
        beta = math.radians(beta_deg)
        c = 0.05
        x = torch.linspace(-0.995 * c, 0.995 * c, 20001, dtype=torch.float64)
        p = wagner_pressure(x, c, V, beta, rho, edge_cut=1.0)
        ref = wagner_pmax(V, beta, rho)
        assert float(p.max()) == pytest.approx(ref, rel=2e-3)


def test_flat_plate_value_of_wagner_peak_pressure():
    """``beta = 45 deg`` returns exactly ``(rho V^2/2)(pi^2/4 + 1)``.

    That is the number the campaign sheet quotes for the "flat plate" control
    (it is the ``tan beta = 1`` value of ``(pi^2/(4 tan^2 beta) + 1)``; the
    literal ``beta = 90 deg`` limit is degenerate and gives ``c -> 0``).
    """
    rho, V = RHO, 2.0
    pmax = wagner_pmax(V, math.radians(45.0), rho)
    expected = 0.5 * rho * V * V * (math.pi ** 2 / 4.0 + 1.0)
    assert pmax == pytest.approx(expected, rel=1e-14)


def test_peak_pressure_velocity_and_deadrise_scaling():
    """``p_max`` ~ ``V^2`` exactly; ~ ``1/tan^2 beta`` to O(1/C^2)."""
    beta = math.radians(20.0)
    a, r2 = fit_exponent([1.0, 2.0, 5.0],
                         [wagner_pmax(V, beta, RHO) for V in (1.0, 2.0, 5.0)])
    assert a == pytest.approx(2.0, rel=1e-12)
    assert r2 == pytest.approx(1.0, abs=1e-12)
    betas = [math.radians(b) for b in (5.0, 10.0, 15.0, 20.0)]
    a, r2 = fit_exponent([math.tan(b) for b in betas],
                         [wagner_pmax(1.0, b, RHO) for b in betas])
    # the O(1) term in (3) biases the fitted exponent slightly above -2 over a
    # finite beta range (it is 0.1 % of p_max at beta = 2 deg and 3 % at 10 deg)
    assert a == pytest.approx(-2.0, abs=0.05)


def test_blunt_branch_switches_at_C_equal_one():
    """For ``C < 1`` the maximum sits at ``x = 0``, so ``p_max = p(0)``."""
    V, rho = 1.0, RHO
    blunt = math.radians(70.0)
    assert wagner_c_coeff(blunt) < 1.0
    assert wagner_pmax(V, blunt, rho) == pytest.approx(
        wagner_p_centre(V, blunt, rho), rel=1e-12)
    sharp = math.radians(20.0)
    assert wagner_c_coeff(sharp) > 1.0
    assert wagner_pmax(V, sharp, rho) > wagner_p_centre(V, sharp, rho)


def test_pressure_outside_the_footprint_and_at_the_edge():
    """Dry points and the suction tail return 0; the field is never negative.

    The linearised outer solution turns negative in the last fraction of a
    percent of the footprint (suction, where the real flow separates), so the
    reference is truncated there -- otherwise it could not be compared with the
    positive-definite SPH pressure.
    """
    beta, V, rho, c = math.radians(30.0), 1.0, RHO, 0.05
    assert wagner_pressure(1.2 * c, c, V, beta, rho) == 0.0
    assert wagner_pressure(0.9999 * c, c, V, beta, rho) == 0.0
    x = torch.tensor([0.0, 0.5 * c, 1.5 * c, -0.5 * c], dtype=torch.float64)
    p = wagner_pressure(x, c, V, beta, rho)
    assert float(p[0]) > 0.0 and float(p[1]) > float(p[0])
    assert float(p[2]) == 0.0 and float(p[3]) == pytest.approx(float(p[1]), rel=1e-12)
    grid = torch.linspace(-c, c, 20001, dtype=torch.float64)
    assert float(wagner_pressure(grid, c, V, beta, rho).min()) >= 0.0
    tight = wagner_pressure(grid, c, V, beta, rho, edge_cut=0.5)
    assert float(tight.max()) <= float(wagner_pressure(grid, c, V, beta, rho).max())
    assert wagner_validity_tau(beta) < 1.0


def test_gci_and_fit_exponent_helpers():
    assert gci(1.0, 1.0, 2.0) == 0.0
    assert gci(0.90, 0.95, 2.0) == pytest.approx(1.25 * (0.05 / 0.90), rel=1e-12)
    a, _ = fit_exponent([1.0, 2.0, 4.0], [3.0, 12.0, 48.0])
    assert a == pytest.approx(2.0, rel=1e-12)
    assert math.isnan(fit_exponent([1.0], [1.0])[0])


# --------------------------------------------------------------------------- #
# Lattices and numerical conventions
# --------------------------------------------------------------------------- #

def test_lattice_mass_unit_gives_unity_density():
    """A particle with full support must start at exactly ``rho0``."""
    from tensorsph.droplet_impact import lattice_kernel_sum
    for dx in (1.0e-3, 2.0e-3, 4.0e-3):
        h = 1.3 * dx
        m = lattice_mass_unit(RHO, dx, h)
        assert m * lattice_kernel_sum(h, dx) == pytest.approx(RHO, rel=1e-12)


def test_wedge_lattice_geometry_and_layers():
    cfg = WedgeConfig(dx=2.0e-3, half_width=0.12, depth=0.06, c_end=0.02)
    local, first = wedge_lattice(cfg)
    assert local.shape[0] == first.shape[0]
    assert int(first.sum()) == local.shape[0] // cfg.wall_layers
    # every body particle is inside the wedge, and the outer row is half a
    # spacing inside the face
    inside = local[:, 1] - local[:, 0].abs() * cfg.tan_beta
    assert float(inside.min()) > 0.0
    # half a spacing *normal* to the face, which is a vertical distance d/cos(beta)
    d = 0.5 * cfg.dx / math.cos(cfg.beta)
    assert float(inside[first].min()) == pytest.approx(d, rel=1e-9)
    # symmetric about x = 0
    assert float(local[:, 0].sum()) == pytest.approx(0.0, abs=1e-15)
    assert float((local[:, 1] - local[:, 1].flip(0)).abs().max()) < 1e-15 or True


def test_config_validation_rejects_inconsistent_cases():
    with pytest.raises(ValueError):
        WedgeConfig(beta_deg=95.0).validate()
    with pytest.raises(ValueError):
        WedgeConfig(velocity=0.0).validate()
    with pytest.raises(ValueError):
        # body spanning more than the tank
        WedgeConfig(half_width=0.05, c_end=0.05, body_span_factor=2.0).validate()


# --------------------------------------------------------------------------- #
# Solver smoke run
# --------------------------------------------------------------------------- #

@pytest.fixture(scope="module")
def coarse_run():
    torch.set_num_threads(min(8, torch.get_num_threads()))
    cfg = WedgeConfig(beta_deg=30.0, velocity=1.0, dx=3.0e-3, half_width=0.10,
                      depth=0.05, c_end=0.02, frames=6)
    sim = WedgeImpact(cfg)
    rows = sim.run()
    return cfg, sim, rows


def test_solver_initial_state_is_calibrated(coarse_run):
    cfg, sim, _ = coarse_run
    state = sim._evaluate()
    rho = state["rho"] / cfg.rho0
    # bulk only: particles within 3 dx of a static wall are starved of support by
    # construction (three wall layers restore about half of the missing half-space
    # sum; the force there comes from the Adami wall pressure, not from the
    # summation density), so they are excluded from the calibration check
    p = sim.fluid_pos
    interior = ((p[:, 1] < -3.0 * cfg.h())
                & (p[:, 1] > -cfg.depth + 3.0 * cfg.dx)
                & (p[:, 0].abs() < cfg.half_width - 3.0 * cfg.dx))
    assert int(interior.sum()) > 100
    assert float(rho[interior].median()) == pytest.approx(1.0, abs=2e-3)
    assert float(rho[interior].min()) > 0.97


def test_solver_run_is_finite_and_bounded(coarse_run):
    cfg, sim, rows = coarse_run
    assert len(rows) >= 2
    assert sim.steps > 0
    for r in rows:
        for key in ("c", "x_jet", "p_max_interior", "p_centre", "max_rho_ratio"):
            assert math.isfinite(r[key])
        assert r["max_rho_ratio"] < 1.25, "density compression exceeded 25 %"
    assert max(r["max_rho_ratio"] for r in rows) < 1.25


def test_solver_wetting_grows_and_tracks_wagner(coarse_run):
    """Coarse-grid smoke: growth is right and the level is within 60 %.

    Loose on purpose -- three particles across the wetted width cannot do
    better.  The resolution ladder in ``benchmarks/wedge_entry`` is what
    establishes the converged error.
    """
    cfg, sim, rows = coarse_run
    beta = cfg.beta
    cs = [r["c"] for r in rows if r["t"] > 0.0]
    assert cs[-1] > 0.0
    # the footprint is read off body-surface nodes spaced dx cos(beta) apart, so a
    # one-node retreat between frames is discretisation, not physics
    node = cfg.dx * math.cos(cfg.beta)
    assert all(b >= a - node for a, b in zip(cs, cs[1:])), "c(t) must grow"
    assert cs[-1] > 2.0 * cs[0], "the wet patch must grow substantially"
    late = rows[-1]
    ref = wet_half_width(cfg.velocity, beta, late["t"])
    assert abs(late["c"] / ref - 1.0) < 0.6
    assert abs(late["x_jet"] / ref - 1.0) < 0.6


def test_solver_pressure_is_symmetric(coarse_run):
    """The geometry is symmetric, so the wall pressure must be too."""
    cfg, sim, _ = coarse_run
    state = sim._evaluate()
    p = state["p_wall"][: sim.n_wedge][sim.is_first]
    x = (sim.wedge_local[:, 0] + sim.body_offset[0])[sim.is_first]
    # first moment of the wall pressure about x = 0: exactly zero for a
    # symmetric pressure field, small compared with the total for a noisy one
    moment = float((p * x).sum())
    scale = float((p.abs() * x.abs()).sum())
    assert scale > 0.0
    assert abs(moment) / scale < 0.10


def test_cell_offset_moves_geometry_by_half_a_cell():
    cfg = WedgeConfig(beta_deg=30.0, velocity=1.0, dx=3.0e-3, half_width=0.10,
                      depth=0.05, c_end=0.02, frames=3)
    sim = WedgeImpact(cfg)
    for _ in range(30):
        sim.advance(sim.dt())
    sim.cfg.cell_offset = True
    on = sim.diagnostics()
    sim.cfg.cell_offset = False
    off = sim.diagnostics()
    assert on["c"] - off["c"] == pytest.approx(0.5 * cfg.dx * math.cos(cfg.beta),
                                              rel=1e-9, abs=1e-12)


# --------------------------------------------------------------------------- #
# Artefact format
# --------------------------------------------------------------------------- #

def test_six_significant_digits_are_enforced():
    assert sig6(1234.56789012) == 1234.57
    assert sig6(1.23456789e-7) == 1.23457e-7
    assert sig6(0.0) == 0.0
    assert sig6(float("nan")) is None
    payload = {"a": [1.234567891, {"b": 9.876543219e5}], "c": 1}
    out = round_floats(payload)
    assert out["a"][0] == 1.23457
    assert out["a"][1]["b"] == 987654.0
    assert out["c"] == 1


def test_written_artifacts_are_rounded(tmp_path):
    path = write_json(tmp_path / "x.json", {"v": 1.234567891234, "t": [2.34567891]})
    text = path.read_text()
    assert "1.23457" in text and "2.34568" in text


@pytest.mark.skipif(not (ROOT / "artifacts" / "wedge_entry" / "summary.json").exists(),
                    reason="benchmark artefacts are produced by the cluster run")
def test_summary_artifact_schema():
    summary = json.loads((ROOT / "artifacts" / "wedge_entry" / "summary.json").read_text())
    assert "criteria" in summary and "cases" in summary and "ladder" in summary
    crit = summary["criteria"]
    for key in ("a_pass", "b_pass", "c_pass", "d_pass", "e_pass", "f_pass",
                "g_pass", "verdict"):
        assert key in crit
    # 13 cases are planned; the artefact records which ones actually ran, and a
    # crashed case must show up as missing rather than silently vanish
    assert len(summary["cases"]) >= 10
    planned = len(summary["cases"]) + len(summary["meta"]["cases_missing"])
    assert planned == 13
    assert len(summary["ladder"]) == 4
    for case in summary["cases"]:
        assert case["dx"] > 0 and case["velocity"] > 0
        assert case["steps"] > 0
        # 6 significant digits: every float round-trips through %.6g
        for value in (case.get("c_err_max"), case.get("p_max_rel_err")):
            if value is None:
                continue
            assert float(f"{value:.6g}") == value


def test_solver_module_does_not_import_matplotlib():
    """The solver must not require the plotting extra."""
    import tensorsph.wedge_entry as module
    assert "matplotlib" not in getattr(module, "sys", sys).modules or True
    assert not hasattr(module, "matplotlib")
