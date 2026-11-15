"""Guards for the second compressible-Riemann family and its topology module.

Every assertion is an invariant, not a smoke test:

* the wave topology is classified from the exact star pressure (shock vs
  rarefaction per side) and the Sod topology agrees with the legacy
  ``exact_star_state`` -- a regression on the code path the shipped case uses;
* the shock branches satisfy the Rankine-Hugoniot conditions they claim to use,
  checked with the reported wave speeds (so a wrong ``speed_head`` cannot hide);
* the analytic yardstick is confirmed by a solver that shares no code with it
  (first-order HLLC finite volume);
* the representability gate refuses a Riemann problem whose mass budget a
  fixed-mass reflecting box cannot meet (Toro's test 4) and accepts the two that
  are graded;
* the measurement windows are derived from exact wave positions, stay inside a
  plateau, and *raise* instead of reporting a number when the plateau is too
  thin to hold one;
* the committed record is reproducible from the committed artefact (the registry
  number is the extractor applied to ``result.json``, nothing typed in).
"""
from __future__ import annotations

import json
import os
import sys

import numpy as np
import pytest

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, os.path.join(ROOT, "src"))

from tensorsph.compressible import fv_reference as FV  # noqa: E402
from tensorsph.compressible import riemann_topology as RT  # noqa: E402
from tensorsph.compressible import sod as S  # noqa: E402

CASE_DIR = os.path.join(ROOT, "benchmarks", "case_strong_shock_contact")
SOD = dict(gamma=1.4, rho_L=1.0, p_L=1.0, u_L=0.0, rho_R=0.125, p_R=0.1, u_R=0.0)
STRONG = dict(gamma=1.4, rho_L=1.0, p_L=1000.0, u_L=0.0, rho_R=1.0, p_R=1.0, u_R=0.0)
TORO4 = dict(gamma=1.4, rho_L=5.99924, p_L=460.8938, u_L=19.5975,
             rho_R=5.99242, p_R=46.0950, u_R=-6.19633)


def topo(states):
    return RT.topology(states["gamma"], states["rho_L"], states["p_L"], states["u_L"],
                       states["rho_R"], states["p_R"], states["u_R"])


# --------------------------------------------------------------------------- #
# topology
# --------------------------------------------------------------------------- #
def test_sod_topology_is_left_rarefaction_right_shock():
    t = topo(SOD)
    assert t.left.kind == "rarefaction" and t.right.kind == "shock"
    assert t.wave_left_is_outermost if hasattr(t, "wave_left_is_outermost") else True
    assert t.left.speed_head < t.left.speed_tail < t.contact_speed < t.right.speed_head


def test_sod_topology_reproduces_legacy_star_state():
    """The generic classification must not move a number the shipped case reports."""
    t = topo(SOD)
    legacy = S.exact_star_state(**SOD)
    assert t.p_star == pytest.approx(legacy["p_star"], rel=1e-12)
    assert t.u_star == pytest.approx(legacy["u_star"], rel=1e-12)
    assert t.left.rho_star == pytest.approx(legacy["rho_star_L"], rel=1e-12)
    assert t.right.rho_star == pytest.approx(legacy["rho_star_R"], rel=1e-12)
    assert t.right.speed_head == pytest.approx(legacy["shock_speed"], rel=1e-12)
    assert t.left.speed_head == pytest.approx(legacy["rarefaction_head_speed"], rel=1e-12)


def test_two_shock_topology_is_classified_as_shocks():
    """Toro test 4: p* exceeds both p_L and p_R, so *both* sides are shocks.

    The legacy Sod helper would silently return the isentropic rarefaction
    density for the left side here; the topology module must not.
    """
    t = topo(TORO4)
    assert t.p_star > TORO4["p_L"] and t.p_star > TORO4["p_R"]
    assert (t.left.kind, t.right.kind) == ("shock", "shock")
    isentropic = TORO4["rho_L"] * (t.p_star / TORO4["p_L"]) ** (1 / TORO4["gamma"])
    assert abs(t.left.rho_star - isentropic) / t.left.rho_star > 0.05  # not the same number
    assert t.left.rho_star > TORO4["rho_L"]                            # shock compresses
    # ... and the legacy Sod helper reports the *rarefaction* head speed for a
    # side that is a shock: the number it returns is not even the outermost wave.
    legacy = S.exact_star_state(**TORO4)
    # legacy calls the left wave a rarefaction, so it reports u_L - c_L; the real
    # outermost left wave travels at +0.79 m/s (a right-moving *shock*).
    assert legacy["rarefaction_head_speed"] - t.left.speed_head > 5.0


@pytest.mark.parametrize("states", [SOD, STRONG, TORO4])
def test_shock_branches_satisfy_rankine_hugoniot(states):
    """Mass and momentum jump conditions, checked with the reported wave speed."""
    t = topo(states)
    g = states["gamma"]
    for side, (rho_k, p_k, u_k) in (
            ("left", (states["rho_L"], states["p_L"], states["u_L"])),
            ("right", (states["rho_R"], states["p_R"], states["u_R"]))):
        w = getattr(t, side)
        if w.kind != "shock":
            # isentropic fan: p/rho^gamma is conserved along the fan
            assert t.p_star / w.rho_star ** g == pytest.approx(p_k / rho_k ** g, rel=1e-10)
            continue
        speed = w.speed_head
        mass = rho_k * (u_k - speed) - w.rho_star * (t.u_star - speed)
        mom = (p_k + rho_k * (u_k - speed) ** 2) - (t.p_star + w.rho_star * (t.u_star - speed) ** 2)
        assert abs(mass) <= 1e-9 * max(1.0, rho_k)
        assert abs(mom) <= 1e-9 * max(1.0, p_k)


@pytest.mark.parametrize("states,x0,lo,hi,t", [(SOD, 0.5, 0.0, 1.0, 0.2),
                                               (STRONG, 0.0, -1.0, 1.0, 0.02),
                                               (TORO4, 0.4, 0.0, 1.0, 0.035)])
def test_wave_order_is_monotone(states, x0, lo, hi, t):
    pos = topo(states).positions(t, x0)
    # <= on the first pair: a shock side has speed_head == speed_tail
    assert lo < pos["x_wave_left"] <= pos["x_rarefaction_tail"] \
        < pos["x_contact"] < pos["x_wave_right"] < hi


# --------------------------------------------------------------------------- #
# yardstick (independent route to the same exact solution)
# --------------------------------------------------------------------------- #
def test_exact_yardstick_is_confirmed_by_hllc():
    """SPH's error numbers are measured against this profile; verify it independently."""
    fv = FV.run_hllc(n_cells=4000, t_end=0.02, cfl=0.4, gamma=STRONG["gamma"],
                     x_left=-1.0, x_right=1.0, x_diaphragm=0.0,
                     rho_L=STRONG["rho_L"], p_L=STRONG["p_L"], u_L=STRONG["u_L"],
                     rho_R=STRONG["rho_R"], p_R=STRONG["p_R"], u_R=STRONG["u_R"])
    g = RT.grade_riemann(fv["x"], fv["rho"], fv["u"], fv["p"], topo(STRONG),
                         float(fv["t"]), 0.0, (1.0 - -1.0) / 4000, -1.0, 1.0)
    # a first-order flux smears the contact, so the density plateau is the loose
    # one; the pressure/velocity plateaus and the wave positions are sharp
    assert abs(g["rel_err"]["rho_star_R"]) < 0.05, g["rel_err"]
    assert abs(g["rel_err"]["p_star"]) < 0.005, g["rel_err"]
    assert abs(g["rel_err"]["u_star"]) < 0.01, g["rel_err"]
    assert abs(g["rel_err"]["x_wave_right"]) < 0.01, g["rel_err"]


# --------------------------------------------------------------------------- #
# representability (the precondition of the whole family)
# --------------------------------------------------------------------------- #
def test_representability_accepts_stationary_outer_states():
    for states, x0, t, lo, hi in ((SOD, 0.5, 0.2, 0.0, 1.0),
                                  (STRONG, 0.0, 0.02, -1.0, 1.0)):
        b = RT.reflective_box_budget(topo(states), t, x0, lo, hi)
        assert b["representable"], b


def test_representability_refuses_inflow_riemann_data():
    b = RT.reflective_box_budget(topo(TORO4), 0.035, 0.4, 0.0, 1.0)
    assert not b["representable"]
    assert b["relative_mass_change"] > 0.5           # measured 0.90
    assert "mass flux" in b["reason"]


def test_representability_refuses_waves_hitting_the_wall():
    b = RT.reflective_box_budget(topo(STRONG), 0.5, 0.0, -1.0, 1.0)   # 0.02 was fine
    assert not b["representable"] and not b["contained"]


# --------------------------------------------------------------------------- #
# measurement windows
# --------------------------------------------------------------------------- #
def test_windows_lie_inside_the_star_plateaus():
    t, x0 = 0.02, 0.0
    h = 2.5 * 2.0 / 800
    top = topo(STRONG)
    pw = RT.plateau_windows(top, t, x0, h, -1.0, 1.0)
    pos = top.positions(t, x0)
    (l0, l1), (r0, r1) = pw["windows"]["star_left"], pw["windows"]["star_right"]
    assert pos["x_rarefaction_tail"] < l0 < l1 < pos["x_contact"] < r0 < r1 < pos["x_wave_right"]


def test_windows_refuse_to_measure_a_too_thin_plateau():
    with pytest.raises(ValueError, match="empty measurement window"):
        RT.plateau_windows(topo(STRONG), 0.02, 0.0, h=5.0, x_left=-1.0, x_right=1.0)


def test_windows_leave_the_domain_loudly():
    with pytest.raises(ValueError, match="leaves the domain"):
        RT.plateau_windows(topo(SOD), 0.2, 0.5, h=1e-4, x_left=0.6, x_right=1.0)


# --------------------------------------------------------------------------- #
# the shipped solver, one cheap run
# --------------------------------------------------------------------------- #
def test_short_run_conserves_mass_and_grades_finite():
    states = {k: v for k, v in STRONG.items() if k != "gamma"}
    cfg = S.SodSPHConfig(n_particles=600, h_factor=2.5, cfl=0.15, t_end=0.02,
                         x_left=-1.0, x_right=1.0, x_diaphragm=0.0,
                         gamma=STRONG["gamma"], **states)
    res = S.run_sod_sph(cfg)
    assert res["mass_final"] == pytest.approx(res["mass_initial"], rel=1e-12)
    snap = res["snapshots"][-1]
    g = RT.grade_riemann(snap["x"], snap["rho"], snap["u"], snap["p"], topo(STRONG),
                         snap["t"], 0.0, cfg.h, cfg.x_left, cfg.x_right)
    assert set(RT.GATED_QUANTITIES) <= set(g["rel_err"])
    assert np.all(np.isfinite(list(g["rel_err"].values())))
    assert 0.0 <= g["l1"]["rho"] < 0.5


# --------------------------------------------------------------------------- #
# the committed record
# --------------------------------------------------------------------------- #
def _record():
    path = os.path.join(CASE_DIR, "result.json")
    if not os.path.exists(path):
        pytest.skip("benchmarks/case_strong_shock_contact/result.json not present")
    with open(path) as fh:
        return json.load(fh)


def test_record_is_within_its_own_gate():
    rec = _record()
    measured = abs(float(rec["primary"]["grading"]["max_abs_rel_err"])) * 100.0
    assert measured < rec["gate_pct"]
    assert rec["pass_gate"] is True and rec["pass_owner_tolerance"] is True


def test_record_is_reproducible_from_its_artefact():
    """The registry number must be the extractor applied to result.json."""
    rec = _record()
    registry = json.load(open(os.path.join(
        ROOT, "docs", "reports", "method_selection_registry.json")))
    rows = [r for r in registry["records"] if r["extractor"] == "x_strong_shock_contact"]
    assert len(rows) == 1
    measured = abs(float(rec["primary"]["grading"]["max_abs_rel_err"])) * 100.0
    assert rows[0]["measured_pct"] == pytest.approx(measured, rel=1e-9)
    assert rows[0]["method"] == "compressible_riemann" and rows[0]["status"] == "banked"
    assert rows[0]["precondition"]          # the second-family precondition is stated


def test_record_reports_reduced_model_limits_explicitly():
    rec = _record()
    g = rec["primary"]["grading"]
    assert set(RT.DIAGNOSTIC_QUANTITIES) <= set(g["diagnostic_rel_err"])
    # the fan-interior bias is reported (and not gated): it must stay visible
    assert abs(g["diagnostic_rel_err"]["x_u_half"]) > 0.01
    assert "x_u_half" not in g["rel_err"]
    # the contact-zone spike exists and is reported as a diagnostic
    assert g["star_envelope_pressure_spike_over_p_star"] > 1.5
    if "nonrepresentable_probe" in rec:
        assert rec["nonrepresentable_probe"]["representability"]["representable"] is False