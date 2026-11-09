"""Pins of the contact-angle BC diagnosis (``benchmarks/contact_angle/diagnose_bc.py``).

The tests read only the committed artifacts, so they stay cheap; they are
written to *turn red when the physics changes*, which is the point: the day the
BC starts delivering the set angle, ``test_no_admissible_run_is_within_the_gate``
fails and forces the record (this file, ``docs/verification_contact_angle_bc.md``
and the ledger entry) to be updated instead of silently going stale.
"""
from __future__ import annotations

import json
import math
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
BENCH = ROOT / "benchmarks" / "contact_angle"
DIAG = json.loads((BENCH / "bc_diagnosis.json").read_text())
DOC = (ROOT / "docs" / "verification_contact_angle_bc.md").read_text()


def test_ladder_covers_the_written_closure_resolution():
    """m6503's closure path asked for R/dx >= 13; it has to be in the ladder."""
    rungs = [s["R_over_dx"] for s in DIAG["ladder"]]
    assert min(rungs) < 7.0 and max(rungs) >= 13.0
    fine = max(DIAG["ladder"], key=lambda s: s["R_over_dx"])
    assert fine["n_admissible"] == fine["n_runs"] >= 3, (
        "the closure resolution must be admissible on every set angle, otherwise "
        "the reading is WITHHELD by the repo rule and cannot be graded")


def test_no_admissible_run_is_within_the_gate():
    """Current state of the physics: an admissible run still misses every angle.

    This is the assertion that must fail first when the BC is fixed.
    """
    bad = [r for r in DIAG["ladder_rows"]
           if r["admissible"] and abs(r["error_deg"]) <= DIAG["gate_deg"]]
    assert bad == [], f"contact-angle BC now delivers the angle: {bad}"
    fine = max(DIAG["ladder"], key=lambda s: s["R_over_dx"])
    assert fine["worst_abs_error_deg"] > 3.0 * DIAG["gate_deg"]


def test_reading_barely_responds_to_the_set_angle():
    """A BC that delivers theta must have dtheta_measured/dtheta_set ~ 1."""
    fine = max(DIAG["ladder"], key=lambda s: s["R_over_dx"])
    assert fine["slope"] is not None
    assert abs(fine["slope"]) < 0.2, fine
    # and the whole 60 deg sweep of set angles moves the reading by a few degrees
    assert fine["theta_measured_range"] < 5.0


def test_missing_angle_grows_as_the_set_angle_drops():
    """R/dx = 13.3 reads ~152 deg for all three set angles: the *error* is the
    set angle itself, so it must shrink monotonically as theta_set -> 152."""
    fine = [r for r in DIAG["ladder_rows"] if r["R_over_dx"] > 13.0 and r["admissible"]]
    by_set = sorted(fine, key=lambda r: r["theta_set_deg"])
    errs = [r["error_deg"] for r in by_set]
    assert errs == sorted(errs, reverse=True), errs


def test_reach_study_records_a_false_pass_trap():
    """A within-gate reading that is not admissible is an artefact, not a result.

    The only way the shipped BC reaches the set angle is by letting the blend
    cover the whole drop; the study has to keep that row and flag it.
    """
    rows = DIAG["reach_study"]
    assert rows, "reach study missing"
    traps = DIAG["reach_false_passes"]
    assert traps, "the reach study no longer produces its within-gate/inadmissible row"
    for row in traps:
        assert row["blend_fraction"] is not None and row["blend_fraction"] > 0.9
        assert not row["admissible"]
    # the admissible thickness must not be the one that hits the gate
    adm = [r for r in rows if r["admissible"]]
    assert adm and all(not r["within_gate"] for r in adm)


def test_doc_numbers_are_injected_from_the_artifact():
    """Every graded reading has to appear in the document, taken from the JSON."""
    for r in DIAG["ladder_rows"]:
        if not r["admissible"]:
            continue
        assert f"{r['theta_measured_deg']:.1f}" in DOC
    assert f"**{DIAG['verdict']}**" in DOC
    assert "WITHHELD" not in DOC.split("## 6")[0].split("## 3")[1].split("判定")[-1]


def test_gate_constants_match_the_harness():
    """The diagnosis must not invent its own gates."""
    src = (BENCH / "run.py").read_text()
    assert "WALL_PENETRATION_GATE_DX = 0.5" in src
    assert "SPURIOUS_CURRENT_GATE = 0.1" in src
    assert "ANGLE_GATE_DEG = 10.0" in src
    assert DIAG["gate_deg"] == 10.0
    assert DIAG["penetration_gate_dx"] == 0.5
    assert DIAG["spurious_gate"] == 0.1


def test_reach_default_is_unchanged_by_the_new_config_field():
    """The only src/ change is an exposed knob; its default must not move physics."""
    from tensorsph.csf_consistent import CONTACT_ANGLE_REACH, apply_contact_angle_boundary
    from tensorsph.droplet_impact import Config
    import inspect

    assert CONTACT_ANGLE_REACH == 2.0
    assert Config().contact_angle_reach == CONTACT_ANGLE_REACH
    assert inspect.signature(apply_contact_angle_boundary).parameters["reach"].default == 2.0


def test_config_reach_reaches_the_boundary_condition(monkeypatch):
    """Config.contact_angle_reach must flow into the BC (not be silently ignored)."""
    import tensorsph.csf_consistent as cc
    import tensorsph.surface_tension as st
    import torch

    seen = {}
    orig = cc.apply_contact_angle_boundary

    def spy(grad_c, pos, wall_pos, h, theta, reach=2.0):
        seen["reach"] = reach
        return orig(grad_c, pos, wall_pos, h, theta, reach=reach)

    monkeypatch.setattr(cc, "apply_contact_angle_boundary", spy)
    monkeypatch.setattr(st, "apply_contact_angle_boundary", spy)

    class Cfg:
        h_factor = 1.3
        dx = 0.01
        rho0 = 1000.0
        surface_tension_coefficient = 0.072
        csf_scheme = "consistent"
        csf_curvature = "mls"
        contact_angle = math.pi / 2
        contact_angle_reach = 3.5

    # three particles in a line + a wall row: enough for a non-zero gradient
    pos = torch.tensor([[0.0, 0.02], [0.0, 0.03], [0.0, 0.04]])
    mass = torch.full((3,), 1.0)
    pairs = torch.tensor([[0, 1], [1, 2], [0, 2]])
    wall = torch.tensor([[-0.01, 0.0], [0.0, 0.0], [0.01, 0.0]])
    st.compute_surface_tension_force(pos, torch.full((3,), 1000.0), mass, pairs,
                                    Cfg(), wall=wall)
    assert seen.get("reach") == pytest.approx(3.5)