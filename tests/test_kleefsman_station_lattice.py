"""Probe-station lattice of the Kleefsman case (m22476).

The acceptance metric of ``CASE-SPH-KLEEFSMAN-DAMBREAK-OBST`` is compared
against the experimental gauge H2, and every committed grade of the case was
taken at ``obstacle_upstream_face_x - 0.05 = 2.426 m`` -- an assumed station.

These tests pin the arithmetic that decides whether that station can be H2 at
all.  They are solver-free: the evidence (probe spacing, reservoir length) is
read from ``reference.json`` / the audit artefact, so a red test means the
geometry evidence and the committed station disagree, not that a run drifted.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
CASE = ROOT / "benchmarks" / "kleefsman_obstacle"
sys.path.insert(0, str(CASE))

from station_lattice import SPACING_M, lattice_audit  # noqa: E402


@pytest.fixture(scope="module")
def ref() -> dict:
    return json.loads((CASE / "reference.json").read_text())


@pytest.fixture(scope="module")
def audit(ref: dict) -> dict:
    return lattice_audit(ref)


def test_probe_spacing_is_the_published_string() -> None:
    """0.496 m / 3 equal gaps / 4 probes -- Amaro Junior et al. (2020) Fig. 2a."""
    assert SPACING_M == pytest.approx(0.496, abs=1e-9)


def test_admissible_interval_is_bounded_by_the_reservoir_length(audit: dict, ref: dict) -> None:
    column = ref["geometry_m"]["column_length"]
    lo, hi = audit["constraints"]["admissible_H2_interval"]
    # a <= column_length because H4 is the one probe inside the reservoir, so
    # H2 = a + 2S cannot reach further than column + 2S into the tank
    assert hi == pytest.approx(column + 2.0 * SPACING_M, abs=1e-9)
    assert lo == pytest.approx(column, abs=1e-9)
    assert hi < ref["geometry_m"]["obstacle_upstream_face_x"]  # still upstream of the box


def test_committed_station_is_excluded_by_the_lattice(audit: dict, ref: dict) -> None:
    """The historical 2.426 m station needs H4 inside the tank -> inadmissible."""
    committed = ref["geometry_m"]["obstacle_upstream_face_x"] - \
        ref["metric_definition"]["frozen_primary"]["station_offset_m"]
    assert audit["committed_station"]["x_m"] == pytest.approx(committed, abs=1e-12)
    assert audit["committed_station"]["verdict"] == "EXCLUDED_BY_LATTICE"
    assert audit["committed_station"]["implied_a_m"] > ref["geometry_m"]["column_length"]


def test_both_anchors_are_admissible_and_distinct(audit: dict) -> None:
    lo, hi = audit["constraints"]["admissible_a_interval"]
    for name in ("anchor_gate", "anchor_far_wall"):
        anc = audit["anchors"][name]
        assert anc["admissible"], name
        assert lo <= anc["a_m"] <= hi, name
        # lattice is evenly spaced by construction
        assert anc["H2_m"] - anc["H3_m"] == pytest.approx(SPACING_M, abs=1e-9)
        assert anc["H1_m"] - anc["H2_m"] == pytest.approx(SPACING_M, abs=1e-9)
    assert audit["anchors"]["anchor_gate"]["H2_m"] == pytest.approx(2.220, abs=1e-9)
    assert audit["anchors"]["anchor_far_wall"]["H2_m"] == pytest.approx(1.574, abs=1e-9)


def test_committed_scores_are_reproducible(ref: dict) -> None:
    """The config must still put the probe where reference.json says it does.

    Guards against a silent re-stationing of the case: if somebody changes
    ``probe_offset`` without recording it, the config and the frozen protocol
    part company here.
    """
    from tensorsph.kleefsman_obstacle import KleefsmanConfig

    frozen = ref["metric_definition"]["frozen_primary"]
    assert KleefsmanConfig().probe_offset == pytest.approx(frozen["station_offset_m"], abs=1e-12)
    assert ref["metric_definition"]["aperture_m"] == pytest.approx(
        KleefsmanConfig().gauge_aperture or 0.08, abs=1e-12)


def test_measured_ladder_is_recorded_and_brackets_the_verdict() -> None:
    """If the station runs landed, their numbers must be internally consistent.

    The runs are optional (they need minutes of wall clock), so a missing
    artefact skips rather than fails -- but a present artefact must satisfy the
    arithmetic the report quotes.
    """
    out = CASE / "results" / "station_lattice"
    files = sorted(out.glob("stations_dx*.json"))
    if not files:
        pytest.skip("no station-lattice runs committed yet")
    for f in files:
        d = json.loads(f.read_text())
        st = d["stations"]
        # arrival grows downstream: the surge needs longer to reach a larger x
        xs = sorted(((v["x_m"], v["slab_arrival_s"]) for v in st.values()))
        arrivals = [a for _, a in xs]
        assert arrivals == sorted(arrivals), f"{f.name}: arrival not monotone in x"
        ref = d["reference"]
        for v in st.values():
            assert v["arrival_rel_error"] == pytest.approx(
                abs(v["slab_arrival_s"] - ref["arrival_time_s"]) / ref["arrival_time_s"],
                rel=1e-9)
            assert v["arrival_pass"] == bool(v["arrival_rel_error"] < d["tolerance"])