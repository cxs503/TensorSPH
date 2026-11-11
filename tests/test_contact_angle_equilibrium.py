"""The contact-angle equilibrium benchmark: protocol, instrument, and verdict.

Three things are pinned here, in the order the document argues them:

1. the *shipped* physics initial condition cannot distinguish the set angle
   (the defect this benchmark was built to expose) -- pinned so that fixing
   ``make_drop_on_wall`` turns the test red and forces a rewrite;
2. the *replacement* cap family reproduces the set angle on the shipped
   instrument to better than 2 deg at production resolution (the device floor);
3. the physics reading is still FAIL and every run still violates the
   admissibility gates -- pinned so that a real fix has to update the record.

The numbers are read from ``results.json`` / ``wall_color_deficit.json``; the
document is checked to quote them (no hand-copied numbers).
"""
from __future__ import annotations

import json
import math
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
BENCH = ROOT / "benchmarks" / "contact_angle_equilibrium"
DOC = ROOT / "docs" / "verification_contact_angle_equilibrium.md"


@pytest.fixture(scope="module")
def result():
    return json.loads((BENCH / "results.json").read_text())


@pytest.fixture(scope="module")
def probe():
    return json.loads((BENCH / "wall_color_deficit.json").read_text())


def _by(result, theta, bc):
    for row in result["rows"]:
        if row["theta_set_deg"] == theta and row["bc"] == bc:
            return row
    raise AssertionError(f"no row for theta={theta} bc={bc}")


# --------------------------------------------------------------------------- #
# 1. the shipped initial condition is angle-independent
# --------------------------------------------------------------------------- #

def test_shipped_initial_condition_ignores_the_set_angle(result):
    audit = result["summary"]["ic_audit"]
    assert audit["theta_spread_deg"] == pytest.approx(60.0)
    assert audit["n_particles_identical"] is True
    assert audit["reading_spread_deg"] < 1.0
    # the geometry is the theta = 180 deg limit: a full disc tangent to the wall
    for row in audit["rows"]:
        assert row["centre_y_over_R"] <= 1.0
        assert row["centre_y_over_R"] > 0.9


def test_a_fixed_initial_condition_must_retire_this_pin():
    """Guard: if ``make_drop_on_wall`` starts honouring ``theta``, this fails."""
    import importlib.util

    import torch

    path = ROOT / "benchmarks" / "contact_angle" / "run.py"
    spec = importlib.util.spec_from_file_location("ca_run_guard", path)
    ca = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(ca)
    counts = set()
    for theta_deg in (60.0, 90.0, 120.0):
        _, _, pos, _, _, _ = ca.make_drop_on_wall(math.radians(theta_deg),
                                                  torch.device("cpu"))
        counts.add(int(pos.shape[0]))
    assert len(counts) == 1, (
        "make_drop_on_wall now depends on theta -- the protocol defect recorded "
        "in docs/verification_contact_angle_equilibrium.md is fixed, rewrite the record"
    )


# --------------------------------------------------------------------------- #
# 2. the replacement cap family + the instrument floor
# --------------------------------------------------------------------------- #

def test_cap_geometry_is_area_preserving():
    import importlib.util

    path = BENCH / "run.py"
    spec = importlib.util.spec_from_file_location("ca_eq_geo", path)
    eq = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(eq)
    R = eq.CA.RADIUS
    for theta in (math.radians(60.0), math.radians(90.0), math.radians(120.0),
                  math.pi):
        a = eq.cap_radius(theta)
        area = a * a * (theta - math.sin(theta) * math.cos(theta))
        assert area == pytest.approx(math.pi * R * R, rel=1e-12)
    # theta = 180 deg reduces to the shipped geometry exactly
    assert eq.cap_radius(math.pi) == pytest.approx(R, rel=1e-12)


def test_cap_family_reproduces_the_set_angle_on_the_shipped_instrument(result):
    for theta in (60.0, 90.0, 120.0):
        row = _by(result, theta, "on")
        assert abs(row["theta_ic_deg"] - theta) < 2.0, (
            f"instrument reads {row['theta_ic_deg']:.2f} on an exact {theta} deg cap")


def test_cap_particle_count_matches_the_area():
    import importlib.util

    import torch

    path = BENCH / "run.py"
    spec = importlib.util.spec_from_file_location("ca_eq_count", path)
    eq = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(eq)
    expected = math.pi * eq.CA.RADIUS ** 2 / 0.0015 ** 2
    theta = math.radians(90.0)
    _, _, pos, _, _, _ = eq.make_cap_on_wall(theta, 0.0015, torch.device("cpu"))
    assert int(pos.shape[0]) == pytest.approx(expected, rel=0.02)
    assert float(pos[:, 1].min()) >= 0.0


# --------------------------------------------------------------------------- #
# 3. the physics verdict: direction correct, magnitude far short, runs inadmissible
# --------------------------------------------------------------------------- #

def test_boundary_condition_shifts_the_angle_towards_wetting(result):
    """A positive control: the BC has to move the answer, and in this direction."""
    for theta in (60.0, 120.0):
        on = _by(result, theta, "on")["theta_final_deg"]
        off = _by(result, theta, "off")["theta_final_deg"]
        assert on < off - 5.0, (
            f"BC at theta={theta} fails to wet the interface ({on:.1f} vs {off:.1f})")


def test_boundary_condition_still_misses_the_gate(result):
    """Current honest state: no run reaches the 10 deg gate (all FAIL)."""
    gate = result["gate_deg"]
    for row in result["rows"]:
        assert abs(row["error_deg"]) > gate
        assert row["within_gate"] is False
    assert result["summary"]["n_within_gate"] == 0


def test_every_run_is_inadmissible(result):
    gates = result["admissibility"]
    for row in result["rows"]:
        assert row["admissible"] is False
        assert (row["wall_penetration_dx"] >= gates["penetration_gate_dx"]
                or row["spurious_current_ratio"] >= gates["spurious_gate"])


def test_null_solution_would_have_been_a_degeneracy(result):
    """Both BC states land in the non-wetting attractor band (>120 deg)."""
    for row in result["rows"]:
        assert row["theta_final_deg"] > 120.0


# --------------------------------------------------------------------------- #
# 4. the mechanism probe and the document
# --------------------------------------------------------------------------- #

def test_probe_finds_a_wall_contaminated_interface_layer(probe):
    assert probe["interface_bulk_fraction"] > 0.2
    bands = {tuple(b["d_over_dx"]): b for b in probe["rows"]}
    near = [b for k, b in bands.items() if k[1] <= 2.5]
    assert near, "no near-wall band in the probe"
    assert max(b["delta_s_over_max_mean"] for b in near) > 0.2
    # the analytic curvature of the 90 deg cap is 1/a with a = R sqrt(2)
    analytic = 1.0 / (probe["R_over_dx"] * probe["dx_m"] * math.sqrt(2.0))
    far = [b for k, b in bands.items() if 2.5 < k[1] <= 4.0]
    assert far and max(b["kappa_abs_mean"] for b in far) > 5.0 * analytic


def test_document_quotes_the_artifacts(result):
    assert DOC.exists()
    text = DOC.read_text()
    assert "contact_angle_equilibrium.png" in text
    for theta in (60.0, 120.0):
        on = _by(result, theta, "on")
        assert f"{on['theta_final_deg']:.1f}" in text
        assert f"{on['theta_ic_deg']:.2f}" in text
    for section in ("案例", "怎么做的", "计算结果", "结果图", "注意点", "误差判定"):
        assert section in text
    assert (BENCH / "contact_angle_equilibrium.png").exists()