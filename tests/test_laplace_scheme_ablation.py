"""Verification-debt tests for the Laplace benchmark (m2857).

Two things are pinned here, both of them *findings* rather than features:

1. **The pressure jump is not resolvable at the tested resolutions.**  ``Delta
   p`` is read as ``p_core - p_shell`` and ``tait_pressure`` clamps at zero, so a
   shell whose summation density sits below ``rho0`` (kernel truncation, not
   surface tension) reports ``p_shell = 0`` while the unclamped value is tens of
   kPa.  ``benchmarks/laplace_law/run.py::pressure_floor`` quantifies that: the
   signal ``sigma/R`` was 1/60 .. 1/120 of the pressure the clamp suppressed in
   every case.  The coarse case that used to be recorded as a *pass*
   (``test_laplace_law.py::test_laplace_pressure_jump``, +5.6 %) is therefore a
   false pass, and that test is now a strict xfail carrying this evidence.

2. **The curvature calculus is measurable only if the introspection runs through
   the production pipeline.**  The benchmark used to copy the *legacy* CSF field
   construction while the force came from the consistent one, so the reported
   ``kappa``/``delta_s`` did not describe the force.  ``production_fields``
   fixes that; the tests below check the scheme ablation that the fix enabled.

Numbers are read from the committed artifacts, except for the staleness guard,
which recomputes one case and compares -- so a stale artifact fails the suite
instead of silently ageing.
"""
from __future__ import annotations

import importlib.util
import json
from pathlib import Path

import pytest
import torch

ROOT = Path(__file__).resolve().parents[1]
BENCH = ROOT / "benchmarks" / "laplace_law"

GATE = 15.0      # platform acceptance bar [%]
TIGHT = 3.0      # "publishable" bar [%]


def _bench_module():
    spec = importlib.util.spec_from_file_location("laplace_law_run",
                                                  BENCH / "run.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


@pytest.fixture(scope="module")
def bench():
    return _bench_module()


@pytest.fixture(scope="module")
def inst():
    return json.loads((BENCH / "instant_report.json").read_text())


@pytest.fixture(scope="module")
def rel():
    return json.loads((BENCH / "validation_report.json").read_text())


# --------------------------------------------------------------------------- #
# 1. scheme ablation, from the committed artifact
# --------------------------------------------------------------------------- #
def test_ablation_covers_every_scheme_and_resolution(inst):
    abl = inst["scheme_ablation"]
    assert set(abl) == {"legacy", "consistent:mls", "consistent:fit_quadratic",
                        "consistent:fit_circle"}
    for scheme, rows in abl.items():
        assert [r["dx_m"] for r in rows] == pytest.approx(
            [0.001, 0.0015, 0.002, 0.003]), scheme
        assert all(r["scheme"] == scheme for r in rows)


def test_quadratic_fit_curvature_meets_the_gate(inst):
    """kappa from the fitted interface, dx <= 2 mm: inside 15 %, and inside 3 %
    at dx = 1 mm (docs/verification_laplace_pressure_floor.md section 3.1)."""
    rows = inst["scheme_ablation"]["consistent:fit_quadratic"][:3]
    errs = [abs(r["curvature"]["delta_s_weighted_rel_error_pct"]) for r in rows]
    assert max(errs) < GATE
    assert errs[0] < TIGHT


def test_default_div_n_curvature_still_misses_the_gate(inst):
    """The shipped default (csf_curvature="mls") is +25..+31 %: FAIL, recorded,
    not papered over.  Also guard against the two schemes silently becoming the
    same operator again."""
    mls = inst["scheme_ablation"]["consistent:mls"]
    fit = inst["scheme_ablation"]["consistent:fit_quadratic"]
    for i, (a, b) in enumerate(zip(mls, fit)):
        a_err = a["curvature"]["delta_s_weighted_rel_error_pct"]
        b_err = b["curvature"]["delta_s_weighted_rel_error_pct"]
        assert a_err > GATE
        assert a_err - b_err > 5.0          # the operators really differ
        if i < 3:                           # dx <= 2 mm: by more than a gate
            assert a_err - b_err > GATE


def test_circle_fit_is_not_usable_on_a_lattice_droplet(inst):
    """fit_circle was documented as a supported option; on the lattice droplet
    it reads +78..+477 % and drives the force 1.5..5.1x too high, getting worse
    with refinement.  Pinned so the claim cannot come back without the fix."""
    rows = inst["scheme_ablation"]["consistent:fit_circle"]
    assert all(r["curvature"]["delta_s_weighted_rel_error_pct"] > 50.0
               for r in rows)
    assert all(r["cut_force"]["ratio_measured_over_theory"] > 1.4
               for r in rows)


def test_delta_s_normalisation_is_short(inst):
    """int delta_s dn must be 1; the consistent scheme delivers 0.66..0.70 and
    does not converge with dx (the free-surface layer sees a fixed fraction of
    the kernel).  This is the amplitude gap behind the cut-force error."""
    for scheme in ("consistent:mls", "consistent:fit_quadratic"):
        for row in inst["scheme_ablation"][scheme]:
            s = row["delta_s"]["line_integral"]
            assert 0.60 < s < 0.80, (scheme, row["dx_m"], s)


@pytest.mark.xfail(strict=True, reason="delta_s is 30-34 % short of its "
                                      "normalisation int(delta_s) dn = 1; "
                                      "see docs/verification_laplace_pressure_"
                                      "floor.md section 5.4")
def test_delta_s_normalisation_is_satisfied(inst):
    for row in inst["scheme_ablation"]["consistent:fit_quadratic"]:
        assert abs(row["delta_s"]["line_integral"] - 1.0) < 0.15


# --------------------------------------------------------------------------- #
# 2. the introspection must describe the force (regression on the old defect)
# --------------------------------------------------------------------------- #
def test_production_fields_differ_from_the_legacy_copy(bench):
    """The legacy copy is kept for A/B, but for the default ("consistent")
    scheme it must not be what the report scores: the two disagree on kappa by
    more than the 15 % gate at dx = 2 mm."""
    from tensorsph.droplet_impact import Config, density_from_neighbor_pairs
    from tensorsph.neighbor_search import cell_linked_pairs

    dx = 0.002
    cfg = Config(dx=dx, h_factor=1.3, rho0=1000.0, sound_speed=20.0, gamma=7.0,
                 gravity=0.0, viscosity=0.01, surface_tension_coefficient=7.2)
    pos, mass = bench.hexagonal_droplet(0.02, dx, 1000.0)
    h = cfg.h_factor * cfg.dx
    pairs = cell_linked_pairs(pos, 2.0 * h)
    rho = density_from_neighbor_pairs(pos, mass, h, pairs)

    kappa_prod, ds_prod, _, acc_prod = bench.production_fields(pos, rho, mass,
                                                               pairs, cfg)
    _, _, _, kappa_leg, ds_leg, _ = bench.csf_fields(pos, rho, mass, pairs, cfg)
    # the force agrees (production path is the same call), the fields do not
    from tensorsph.surface_tension import compute_surface_tension_force
    acc_ref = compute_surface_tension_force(pos, rho, mass, pairs, cfg)
    assert torch.allclose(acc_prod, acc_ref)

    def weighted(kappa, delta_s):
        w = delta_s.abs()
        return float((kappa.abs() * w).sum() / w.sum().clamp_min(1e-30))

    k_prod = weighted(kappa_prod, ds_prod)
    k_leg = weighted(kappa_leg, ds_leg)
    assert abs(k_prod - k_leg) / k_leg > 0.15


def test_introspection_is_not_stale(bench, inst):
    """Recompute the coarsest fit case and compare with the committed artifact,
    to 1e-9 relative -- the benchmark is deterministic, so any drift means the
    JSON no longer describes the code."""
    fresh = bench.instant_diagnostics(0.02, 0.003, 7.2, 1000.0, 20.0, 0.01,
                                      scheme="consistent:fit_quadratic")
    committed = inst["scheme_ablation"]["consistent:fit_quadratic"][-1]
    for key in ("delta_s_weighted_rel_error_pct",):
        assert fresh["curvature"][key] == pytest.approx(
            committed["curvature"][key], rel=1e-9)
    assert fresh["delta_s"]["line_integral"] == pytest.approx(
        committed["delta_s"]["line_integral"], rel=1e-9)
    assert fresh["cut_force"]["ratio_measured_over_theory"] == pytest.approx(
        committed["cut_force"]["ratio_measured_over_theory"], rel=1e-9)


# --------------------------------------------------------------------------- #
# 3. the pressure jump is measurement-limited (the "false pass" correction)
# --------------------------------------------------------------------------- #
def test_pressure_floor_is_above_the_signal(rel):
    floor = rel["pressure_floor"]
    assert floor["zero_clamp_active"] is True
    assert floor["p_shell_unclamped_Pa"] < 0.0
    assert floor["delta_p_valid"] is False
    assert floor["signal_to_floor"] < 0.1
    # the clamp really is what the difference sees
    assert floor["p_shell_reported_by_tait_Pa"] == 0.0
    assert floor["shell_density_deficit_pct"] > 1.0


def test_every_ablated_case_is_flagged_invalid(rel):
    """dx = 4 mm used to be recorded as a pass.  All four schemes are invalid
    there by the floor criterion; the recorded numbers stay in the artifact."""
    for entry in rel["scheme_ablation"]:
        pf = entry["pressure_floor"]
        assert pf["delta_p_valid"] is False, entry["scheme"]
        assert pf["signal_to_floor"] < 0.1, entry["scheme"]


def test_pressure_floor_matches_the_tait_equation(rel):
    floor = rel["pressure_floor"]
    rho_s = floor["rho_shell_over_rho0"]
    bulk = floor["tait_bulk_modulus_Pa"]
    assert bulk == pytest.approx(rel["setup"]["rho0"]
                                 * rel["setup"]["sound_speed"] ** 2 / 7.0,
                                 rel=1e-12)
    expected = bulk * (rho_s ** 7.0 - 1.0)
    assert floor["p_shell_unclamped_Pa"] == pytest.approx(expected, rel=1e-9)


def test_cut_force_proxy_is_reported_next_to_the_probe(rel):
    """The resolvable stand-in for Delta p is F_cut/(2R); it must be recorded
    with its own error and stay an *independent* reading, not a relabelling of
    the probe."""
    proxy = rel["delta_p_from_cut_force"]
    assert proxy["Pa"] > 0.0
    assert proxy["rel_error_pct"] == pytest.approx(
        (proxy["Pa"] / rel["laplace"]["delta_p_theory_Pa"] - 1.0) * 100.0,
        rel=1e-9)
    assert proxy["within_15pct"] is (abs(proxy["rel_error_pct"]) < GATE)
    # the two routes must not agree to better than the gate at this resolution
    assert abs(proxy["rel_error_pct"]) > 1.0