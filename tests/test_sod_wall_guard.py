"""Regression tests for the mirror-ghost wall guard of the compressible Sod harness.

Anchor ``XVAL-SPH-METHOD-SELECTION-2026-11-12`` / mission m22096.

The guard exists because the banked `compressible_riemann` bound is a *single*
Sod measurement, and the harness that produced it is only valid while the outer
plateaus are at rest.  These tests pin the three facts the benchmark rests on:

* Sod is clean (guard silent, plateau error 0.00 %);
* a pure Galilean boost of the same physics destroys the measurement and the
  guard says so *before* anyone publishes the inflated number;
* the Lax family, measured in the frame where the left plateau is at rest,
  reproduces the bound (clean quantities < 1 %) -- i.e. the bound transfers.
"""
from __future__ import annotations

import json
import os

import pytest

from tensorsph.compressible import sod as S

REPO = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
TRANSFER_JSON = os.path.join(REPO, "benchmarks", "xval_method_selection",
                             "riemann_bound_transfer.json")
TRANSFER_MD = os.path.join(REPO, "docs",
                           "xval_method_selection_bound_transferability_m22096.md")

LAX = dict(rho_L=0.445, p_L=3.528, u_L=0.698, rho_R=0.5, p_R=0.571, u_R=0.0,
           t_end=0.14)


def _metrics(cfg):
    res = S.run_sod_sph(cfg, output_times=[cfg.t_end])
    return S.evaluate_metrics(res["snapshots"][-1], cfg)


def test_sod_windows_are_clean_and_guard_is_silent():
    cfg = S.SodSPHConfig(n_particles=400)
    met = _metrics(cfg)
    guard = met["wall_guard"]
    assert guard["clean"] is True
    assert guard["contaminated"] == []
    assert met["windows_clean"] is True
    # both outer plateaus are at rest -> the mirror ghosts are exact
    assert all(flag["wall"] is None for flag in guard["flags"].values())
    assert all(flag["amplitude_pct"] == 0.0 for flag in guard["flags"].values())
    # and the measured plateau error is at, or just above, round-off level:
    # left plateau 3.9e-6 (n=400) / 4.3e-8 (n=800), right plateau 1.1e-3 at
    # n=400 -- the latter is the *smeared foot of the shock*, which at n=400 is
    # 0.1 % of p_R, not a wall effect (it drops to 3.3e-6 at n=800).
    assert abs(met["rel_err"]["rho_L"]) < 1e-4
    assert abs(met["rel_err"]["p_R"]) < 2e-3
    fine = _metrics(S.SodSPHConfig(n_particles=800))
    assert abs(fine["rel_err"]["p_R"]) < 1e-4
    # require_clean_windows must not raise on Sod
    S.evaluate_metrics(S.run_sod_sph(cfg, output_times=[cfg.t_end])["snapshots"][-1],
                       cfg, require_clean_windows=True)


def test_galilean_boost_is_caught_before_it_is_published():
    """Same physics, moving walls: the harness error explodes and the guard fires."""
    nominal = _metrics(S.SodSPHConfig(n_particles=400))
    boosted = _metrics(S.SodSPHConfig(n_particles=400, u_L=0.5, u_R=0.5))
    assert nominal["max_abs_rel_err"] < 0.01          # ~0.78 %
    assert boosted["max_abs_rel_err"] > 0.20          # > 20 %: pure artefact
    assert boosted["windows_clean"] is False
    assert {"left_plateau", "right_plateau"}.issubset(
        set(boosted["wall_guard"]["contaminated"]))
    # predicted amplitude is the linear-acoustics estimate gamma |u| / c
    left = boosted["wall_guard"]["flags"]["left_plateau"]
    assert left["wall"] == "left"
    assert left["amplitude_pct"] == pytest.approx(
        100.0 * 1.4 * 0.5 / (1.4 * 1.0 / 1.0) ** 0.5, rel=1e-12)

    # the flag is always reported; refusing to publish is opt-in
    cfg = S.SodSPHConfig(n_particles=800, u_L=0.5, u_R=0.5)
    snap = S.run_sod_sph(cfg, output_times=[cfg.t_end])["snapshots"][-1]
    assert S.evaluate_metrics(snap, cfg)["windows_clean"] is False
    with pytest.raises(S.WallContaminationError):
        S.evaluate_metrics(snap, cfg, require_clean_windows=True)


def test_lax_textbook_frame_inflation_is_a_wall_artefact():
    """The 26 % 'error' of Lax in a [0,1] box is the wall, not the scheme."""
    cfg = S.SodSPHConfig(n_particles=400, **LAX)
    met = _metrics(cfg)
    inflated = met["max_abs_rel_err"]
    assert inflated > 0.20                       # would read as a scheme failure
    assert set(met["wall_guard"]["contaminated"]) == {"left_plateau", "star_left"}
    # the windows the guard calls clean really are clean, to <1 %
    for q in ("rho_R", "p_R", "rho_star_R", "p_star_R", "x_contact", "x_shock"):
        assert abs(met["rel_err"][q]) < 0.01, (q, met["rel_err"][q])


def test_lax_left_rest_frame_transfers_the_banked_bound():
    """Left-rest frame + long box: the second family reproduces the bound."""
    cfg = S.SodSPHConfig(n_particles=1200, x_left=-1.0, x_right=6.0,
                         x_diaphragm=2.0, **dict(LAX, u_L=0.0, u_R=-0.698))
    met = _metrics(cfg)
    guard = met["wall_guard"]
    assert guard["contaminated"] == ["right_plateau"]     # only the moving wall
    clean = [q for q in met["gated"]
             if q not in ("rho_R", "p_R")]
    clean_max = max(abs(met["rel_err"][q]) for q in clean)
    assert clean_max < 0.01            # banked bound 0.7768 % transfers
    assert met["max_abs_rel_err"] < 0.03    # even counting the dirty window
    assert guard["flags"]["right_plateau"]["wall"] == "right"


def test_committed_transfer_artifacts_agree_with_each_other():
    """The published number in the JSON is the one printed in the report."""
    payload = json.load(open(TRANSFER_JSON))
    summary = payload["summary"]
    verdict = summary["verdict"]
    assert verdict["within_owner_standard"] is True
    assert verdict["cross_family_bound_pct"] < verdict["record_gate_pct"] < 15.0
    assert (summary["lax_rest_frame"]["max_pct_clean_quantities"]
            < verdict["cross_family_bound_pct"])
    assert summary["sod_reproduction"]["reproduction_rel_diff_pct"] < 1.0
    md = open(TRANSFER_MD).read()
    assert f"{verdict['cross_family_bound_pct']:.2f} %" in md
    assert f"{summary['banked_record']['measured_pct']:.4f} %" in md
    assert "是否 < 3 %" in md          # the owner-mandated explicit answer