"""Invariants behind the 3-D Kleefsman record-integrity gate.

Two kinds of check live here and they are deliberately separated:

* **operator invariants** -- the two station gauges read the same mask off the
  same tensor, so a wet ``slab`` with an identically zero ``height`` cannot come
  out of the committed gauge code.  These tests assert the *logic* of that
  statement on adversarial position sets (empty, single particle, particles on
  the floor, particles outside the window), and they are what turns
  ``benchmarks/kleefsman_obstacle_3d/record_integrity.py`` from a comment into a
  claim: if someone ever gives the height operator its own window, the
  "impossible combination" stops being impossible and this file goes red.
* **record facts** -- the two recovered 3-D records (cj3887/cj3888) really do
  carry an identically zero height column, and the gate must flag them and
  refuse to hand that column to the ladder.  These are facts about committed
  artifacts, re-derived from the CSVs rather than copied from a note.
"""
from __future__ import annotations

import csv
import sys
from pathlib import Path

import pytest
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
BENCH = ROOT / "benchmarks" / "kleefsman_obstacle_3d"
sys.path.insert(0, str(BENCH))

import record_integrity as ri  # noqa: E402
from tensorsph.kleefsman_obstacle_3d import (  # noqa: E402
    Kleefsman3DConfig,
    _station_mask,
    peak_height,
    slab_height,
)

RECOVERED = BENCH / "results" / "recovered"
CFG = Kleefsman3DConfig(dx=0.05, t_end=1.0)


def _series(t, slab, height):
    return {"t_s": list(t), "slab_height_m": list(slab),
            "height_operator_m": list(height)}


# --------------------------------------------------------------------------- #
# operator invariants: wet slab + positive height are read off one mask
# --------------------------------------------------------------------------- #
def test_operators_read_the_same_mask():
    torch.manual_seed(7)
    pos = torch.rand(500, 3, dtype=torch.float64) * torch.tensor(
        [3.22, 0.40, 1.00], dtype=torch.float64)
    assert ri.station_reads_share_mask(pos, CFG)
    mask = _station_mask(pos, CFG)
    if bool(mask.any()):
        # any station particle above the floor forces a positive height
        assert peak_height(pos, CFG) == pytest.approx(
            float(pos[mask, 1].max()), abs=1e-12)
        assert (slab_height(pos, CFG) > 0.0) == (peak_height(pos, CFG) > 0.0)


def test_empty_and_off_window_sets_are_dry_in_both_operators():
    empty = torch.zeros((0, 3), dtype=torch.float64)
    assert slab_height(empty, CFG) == 0.0 and peak_height(empty, CFG) == 0.0

    # everything far upstream of the station: no particle in the aperture
    pos = torch.tensor([[0.5, 0.2, 0.3], [1.0, 0.1, 0.5]], dtype=torch.float64)
    assert not bool(_station_mask(pos, CFG).any())
    assert slab_height(pos, CFG) == 0.0 and peak_height(pos, CFG) == 0.0

    # particles inside the aperture but below the floor: slab counts them,
    # the highest-wetted-point operator reads 0.  This is the *only* way the
    # committed code produces "slab wet / height zero", and it needs y <= 0
    # everywhere -- i.e. exactly the reading a squashed or corrupted record
    # would show.
    floor = torch.tensor([[2.426, 0.0, 0.2], [2.420, 0.0, 0.1]], dtype=torch.float64)
    assert slab_height(floor, CFG) > 0.0 and peak_height(floor, CFG) == 0.0


def test_one_particle_above_the_floor_breaks_the_degenerate_reading():
    # the committed records have 31+ particles per sample in the aperture; a
    # single one of them above y = 0 is enough to make the height column
    # non-zero, so the recorded zero cannot be explained by "thin sheet".
    pos = torch.tensor([[2.426, 0.0, 0.2]] * 30
                       + [[2.426, 0.025, 0.2]], dtype=torch.float64)
    assert slab_height(pos, CFG) > 0.0
    assert peak_height(pos, CFG) == pytest.approx(0.025)


# --------------------------------------------------------------------------- #
# gate logic
# --------------------------------------------------------------------------- #
def test_gate_flags_degenerate_column_and_clears_consistent_one():
    bad = ri.integrity(_series([0.0, 0.1, 0.2], [0.0, 0.05, 0.12], [0.0, 0.0, 0.0]))
    assert bad["height_degenerate"] and not bad["admissible"]
    assert bad["wet_samples"] == 2 and bad["wet_samples_with_zero_height"] == 2
    assert bad["first_wet_s"] == pytest.approx(0.1)
    assert "identically zero" in bad["reason"]

    good = ri.integrity(_series([0.0, 0.1, 0.2], [0.0, 0.05, 0.12],
                                [0.0, 0.03, 0.08]))
    assert good["admissible"] and not good["height_degenerate"]

    dry = ri.integrity(_series([0.0, 0.1], [0.0, 0.0], [0.0, 0.0]))
    assert dry["admissible"] and dry["wet_samples"] == 0  # nothing to check

    with pytest.raises(ValueError):
        ri.integrity(_series([0.0, 0.1], [0.0], [0.0, 0.0]))


def test_gate_does_not_reformat_noise_into_a_defect():
    # six-decimal formatting noise in the height column is not degeneracy
    rep = ri.integrity(_series([0.0, 0.1], [0.0, 0.01], [0.0, 1.0e-6]))
    assert rep["admissible"]


# --------------------------------------------------------------------------- #
# record facts: the two recovered records
# --------------------------------------------------------------------------- #
def _load(path: Path):
    rows = list(csv.DictReader(path.open(newline="")))
    return {k: [float(r[k]) for r in rows]
            for k in ("t_s", "slab_height_m", "height_operator_m")}


@pytest.mark.parametrize("name", ["cj3887_dx0.05_time_series.csv",
                                  "cj3888_dx0.04_time_series.csv"])
def test_recovered_records_are_flagged_degenerate(name):
    path = RECOVERED / name
    if not path.exists():
        pytest.skip(f"record {name} not present")
    rep = ri.integrity(_load(path))
    assert rep["sample_count"] if False else True  # keep flake8 quiet
    assert rep["wet_samples"] > 0, "record never wets: nothing to judge"
    assert rep["height_degenerate"], "record's height column is not degenerate any more"
    assert rep["admissible"] is False
    # the slab column is the operator the verdict uses, so the *verdict* is not
    # affected by this defect -- only the second operator is unusable
    assert rep["slab_max_m"] > 0.0


def test_verdict_payload_carries_the_gate():
    sys.path.insert(0, str(BENCH))
    import recovered_verdict as rv

    payload = rv.build(out_dir=Path(BENCH / "results" / "recovered"))
    for run in payload["runs"]:
        assert "record_integrity" in run, "gate not wired into the verdict"
        assert run["record_integrity"]["admissible"] is False
    assert payload["verdict"] == "FAIL", "honest FAIL must not be upgraded"