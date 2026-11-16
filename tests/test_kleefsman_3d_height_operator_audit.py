"""Admissibility of the 3-D gauge operators (CASE-SPH-KLEEFSMAN-DAMBREAK-OBST).

The two gauge operators screen the fluid with the same ``_station_mask``, so
they are two functions of one particle set at one instant.  These tests pin the
consequence -- a record that reports metres of water while the highest wetted
point is 0.0 is a record defect -- and pin it against the *committed* recovered
records, re-derived from the raw CSV (no hand-copied numbers).
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
HERE = ROOT / "benchmarks" / "kleefsman_obstacle_3d"
sys.path.insert(0, str(HERE))

from tensorsph.kleefsman_obstacle_3d import (  # noqa: E402
    Kleefsman3DConfig,
    Kleefsman3DRun,
    _station_mask,
    peak_height,
    slab_height,
)
import height_operator_audit as audit  # noqa: E402


def _cfg(dx: float = 0.05, **kw):
    return Kleefsman3DConfig(dx=dx, **kw)


def _band_lattice(cfg: Kleefsman3DConfig, layers: int = 4, dx: float = 0.05):
    """Particles on a lattice inside the gauge band, ``layers`` high."""
    x0 = cfg.station_x() - 0.5 * cfg.gauge_aperture + 0.5 * dx
    z0 = cfg.obstacle_z + 0.5 * dx
    pts = []
    for i in range(2):                                   # streamwise columns
        for j in range(layers):
            for k in range(4):                            # spanwise columns
                pts.append([x0 + i * dx, 0.5 * dx + j * dx, z0 + k * dx])
    return torch.tensor(pts, dtype=torch.float32)


def test_operators_share_one_mask_so_wet_band_implies_positive_height():
    cfg = _cfg()
    pos = _band_lattice(cfg)
    assert bool(_station_mask(pos, cfg).all())
    assert slab_height(pos, cfg) > 0.0
    assert peak_height(pos, cfg) > 0.0                  # the witness cannot be 0


def test_gate_flags_water_with_a_floor_level_surface():
    cfg = _cfg()
    run = Kleefsman3DRun(cfg=cfg, slab=[0.0, 0.05, 0.12, 0.12],
                         height=[0.0, 0.0, 0.0, 0.0])
    gate = run.operator_consistency()
    assert gate["wet_samples"] == 3
    assert gate["impossible_samples"] == 2              # 0.05 and 0.12 > 2*dx
    assert gate["height_column_admissible"] is False
    assert gate["reason"]


def test_gate_accepts_a_healthy_record_and_the_reproduced_floor_trap():
    cfg = _cfg()
    healthy = Kleefsman3DRun(cfg=cfg, slab=[0.0, 0.05, 0.12],
                             height=[0.0, 0.03, 0.11])
    assert healthy.operator_consistency()["height_column_admissible"] is True
    # a single particle on the floor is real (aperture narrower than a spacing
    # at dx = 0.10), so it must not be called impossible
    trap = Kleefsman3DRun(cfg=Kleefsman3DConfig(dx=0.10), slab=[0.031],
                          height=[0.0])
    assert trap.operator_consistency()["height_column_admissible"] is True


def test_empty_band_is_not_a_defect():
    cfg = _cfg()
    run = Kleefsman3DRun(cfg=cfg, slab=[0.0, 0.0], height=[0.0, 0.0])
    gate = run.operator_consistency()
    assert gate["wet_samples"] == 0
    assert gate["height_column_admissible"] is True


@pytest.mark.parametrize("dx", [0.02, 0.04, 0.05])
def test_floor_bound_is_monotone_and_never_below_one_column(dx):
    cfg = _cfg(dx=dx)
    tight = audit.floor_columns(dx, cfg)
    generous = audit.floor_columns_generous(dx, cfg)
    assert 1 <= tight <= generous
    # a finer lattice cannot explain *less* floor area in the same band
    assert audit.floor_columns_generous(0.5 * dx, cfg) >= generous


def test_verdict_survives_the_most_generous_floor_packing():
    """The impossibility must not hinge on how tightly the floor is counted."""
    for run in audit.RUNS:
        rec = audit.audit_run(run)
        assert (rec["particles_needed_at_floor_for_worst_sample"]
                > rec["floor_columns_generous"]), run["job"]


def test_recovered_records_are_inadmissible_and_the_figure_of_merit_is_geometric():
    for run in audit.RUNS:
        rec = audit.audit_run(run)
        assert rec["height_max_m"] == 0.0
        assert rec["wet_samples"] > 0
        assert rec["height_column_admissible"] is False
        # the worst impossible sample needs more particles on the floor plane
        # than there are floor positions in the band
        assert (rec["particles_needed_at_floor_for_worst_sample"]
                > rec["floor_columns_available"])


def test_committed_audit_json_matches_the_raw_series():
    committed = json.loads((audit.OUT / "audit.json").read_text())
    for run in audit.RUNS:
        rec = audit.audit_run(run)
        rec.pop("_series")
        want = [r for r in committed["records"] if r["job"] == run["job"]][0]
        for key, value in want.items():
            assert rec[key] == value, (run["job"], key)
    assert committed["any_inadmissible"] is True


def test_writer_stamps_the_code_revision_into_the_manifest():
    import importlib.util
    spec = importlib.util.spec_from_file_location("k3_run", HERE / "run.py")
    runner = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(runner)
    rev = runner._code_revision()
    assert rev["git"] is None or len(rev["git"]) == 40
    assert rev["operator_sha1"] is not None and len(rev["operator_sha1"]) == 40
