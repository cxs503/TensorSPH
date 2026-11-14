"""Evidence channel of ``CASE-SPH-LAKE-AT-REST``: is the anchor decidable here?

The case's acceptance is a machine-precision *identity* (``u = 0``, ``eta =
const``, ``max|v|/c0 < 1e-9``).  Before this file, the proof lived in
``artifacts/lake_at_rest/*.json`` (``.gitignore``'d -> absent on a fresh
checkout) and in prose inside ``docs/verification_lake_at_rest*.md`` (not
machine-decidable, free to drift from the code).  ``benchmarks/lake_at_rest/
freeze.py`` ships the text records into ``tests/data/lake_at_rest``; this file
is what makes them *load-bearing*:

* **tamper-evidence** -- every shipped byte is checked against
  ``MANIFEST.sha256`` and the channel manifest against the files present;
* **the shipped gate table is re-graded from the shipped numbers** with the
  library's own ``evaluate_gates``, so a report whose rows disagree with its own
  run record fails here (a hand-edited ``report.json`` cannot launder a pass);
* **the anchor verdict** -- all five acceptance gates pass in the x-periodic
  tank *at the published ``dx = 0.02``*, i.e. the tolerance is met by the
  shipped evidence and not only by the coarse ``dx = 0.04`` unit tests;
* **the honest FAIL is preserved** -- the wall-bounded tank still fails exactly
  ``{G2, G3, G4, G5, G3i, G4i}`` and the failure magnitudes are the documented
  ones; a future accidental "fix" of the side-wall residual shows up as a red
  test rather than as silently stale prose;
* **the evidence is re-derivable from source** -- the shipped geometry and the
  ``G7`` negative control (the analytic-profile residual, one ``rhs`` call) are
  recomputed here at ``dx = 0.02`` and compared with the frozen numbers.  That
  is the cheap half of the case that can run on every checkout (~2 s); the full
  reconstruction is reproducible with ``freeze.py --verify``;
* **the reproducibility claim is itself asserted** -- rounding-level
  quantities are declared as such and must be below the anchor tolerance; the
  stable ones are declared stable.  A channel that started shipping digits as
  if they were bit-reproducible would be lying, and this test names which is
  which.
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest
import torch

REPO = Path(__file__).resolve().parents[1]
CHANNEL = REPO / "tests" / "data" / "lake_at_rest"
PERIODIC = "periodic_dx020"
WALL = "wall_dx020"
ANCHOR_TOLERANCE = 1.0e-9
ACCEPTANCE = ("G1", "G2", "G3", "G4", "G5")

import sys  # noqa: E402

sys.path.insert(0, str(REPO / "src"))

from tensorsph.lake_at_rest import (LakeConfig, analytic_profile,  # noqa: E402
                                   build_state, evaluate_gates, residual_report)


def _load(name: str) -> dict:
    path = CHANNEL / name
    if not path.exists():
        pytest.fail(f"shipped evidence {path} is missing: the anchor's verdict "
                    f"is not decidable on this checkout -- regenerate with "
                    f"`PYTHONPATH=src python3 benchmarks/lake_at_rest/freeze.py`")
    return json.loads(path.read_text())


@pytest.fixture(scope="module")
def report() -> dict:
    return _load("report.json")


@pytest.fixture(scope="module")
def records() -> dict:
    return {name: _load(f"{name}.json") for name in (PERIODIC, WALL)}


def test_manifest_matches_every_shipped_byte():
    manifest = CHANNEL / "MANIFEST.sha256"
    assert manifest.exists(), "no MANIFEST.sha256: the shipped numbers are not tamper-evident"
    listed = {}
    for line in manifest.read_text().splitlines():
        digest, size, name = line.split()
        path = CHANNEL / name
        assert path.exists(), f"{name} is listed in the manifest but not shipped"
        payload = path.read_bytes()
        assert len(payload) == int(size), f"{name}: size drifted from the manifest"
        assert hashlib.sha256(payload).hexdigest() == digest, (
            f"{name}: sha256 drifted from the manifest -- the evidence was "
            f"edited without re-freezing")
        listed[name] = digest
    on_disk = {p.name for p in CHANNEL.glob("*.json")}
    assert on_disk <= set(listed) | {"channel.json"}, (
        f"unmanifested evidence files: {sorted(on_disk - set(listed))}")


def test_channel_declares_what_it_does_not_ship():
    channel = _load("channel.json")
    assert channel["case"] == "CASE-SPH-LAKE-AT-REST"
    assert channel["manifest"] == "MANIFEST.sha256"
    assert channel["frozen_by"].endswith("freeze.py")
    assert channel["read_by"].endswith("test_lake_at_rest_evidence.py")
    not_shipped = channel["not_shipped"]
    assert "png" in not_shipped and "artifacts" in not_shipped
    assert "gitignore" in not_shipped["why"].lower() or "artifact" in not_shipped["why"].lower()
    for name in (f"{PERIODIC}.json", f"{WALL}.json", "report.json"):
        assert name in channel["shipped"], f"{name} not declared as shipped"


def test_shipped_gate_rows_are_regraded_from_their_own_numbers(report, records):
    """A report that disagrees with its own run record must fail here."""
    for name, record in records.items():
        regraded = evaluate_gates(record)
        shipped = report["runs"][name]["gates"]
        assert regraded["verdicts"] == shipped["verdicts"], (
            f"{name}: shipped verdicts {shipped['verdicts']} do not re-grade "
            f"from the record ({regraded['verdicts']})")
        assert regraded["all_passed"] == shipped["all_passed"]
        assert len(regraded["rows"]) == len(shipped["rows"])
        for fresh, frozen in zip(regraded["rows"], shipped["rows"]):
            assert fresh["gate"] == frozen["gate"]
            assert fresh["threshold"] == frozen["threshold"]
            assert fresh["comparison"] == frozen["comparison"]
            assert fresh["passed"] == frozen["passed"]
            assert fresh["measured"] == pytest.approx(frozen["measured"],
                                                      rel=1e-12, abs=0.0)


def test_anchor_tolerance_is_the_analytic_identity(report):
    anchor = report["anchor"]
    assert anchor["tolerance"] == ANCHOR_TOLERANCE
    assert tuple(anchor["acceptance_gates"]) == ACCEPTANCE
    assert "analytic" in report["reference"]
    # no fitted constant / no experimental band anywhere in the record
    assert "identity" in anchor["note"] or "no experimental" in anchor["note"]


def test_periodic_tank_meets_the_anchor_tolerance_at_published_resolution(report):
    run = report["runs"][PERIODIC]
    geometry = run["geometry"]
    assert geometry["periodic_x"] is True
    assert geometry["dx"] == pytest.approx(0.02)
    assert geometry["particles"] == 300 and geometry["pairs"] == 3000
    assert geometry["h"] == pytest.approx(1.3 * geometry["dx"], rel=1e-12)
    gates = run["gates"]
    assert gates["all_passed"] is True
    for gate in ACCEPTANCE:
        assert gates["verdicts"][gate] is True, f"{gate} did not pass"
    acceptance = {row["gate"]: row["measured"] for row in gates["rows"]
                  if row["gate"] in ACCEPTANCE}
    for gate, value in acceptance.items():
        assert value < ANCHOR_TOLERANCE, f"{gate} = {value:.3e} is not machine precision"
    # the rest run actually moved the particles: the identity is measured over 0.5 s
    assert run["rest"]["t_end"] == pytest.approx(0.50, rel=1e-3)
    assert run["rest"]["steps"] > 1000
    assert run["equilibrium"]["converged"] is True
    assert report["verdict"][PERIODIC]["status"] == "PASS"


def test_wall_bounded_honest_fail_is_preserved(report):
    run = report["runs"][WALL]
    assert run["geometry"]["periodic_x"] is False
    assert run["geometry"]["wall_particles"] > 0
    gates = run["gates"]
    assert gates["all_passed"] is False
    assert report["verdict"][WALL]["status"] == "FAIL"
    measured = {row["gate"]: row["measured"] for row in gates["rows"]}
    # the vertical balance of the same operator is still machine precision ...
    assert measured["G1"] < ANCHOR_TOLERANCE
    # ... the horizontal one is not, and it is the side-wall band that breaks it
    assert measured["G2"] > 1e-2
    assert measured["G2"] < 1.0
    for gate in ("G3", "G4", "G5"):
        assert measured[gate] > ANCHOR_TOLERANCE, f"{gate} silently improved"
        assert measured[gate] < 1.0
    # the failing set is pinned: an accidental change of the defect is a red test
    failing = {gate for gate, ok in gates["verdicts"].items() if not ok}
    assert failing == {"G2", "G3", "G4", "G5", "G3i", "G4i"}


def test_negative_controls_are_alive_in_the_shipped_record(report):
    run = report["runs"][PERIODIC]
    controls = run["controls"]
    # uniform lattice: rho == rho0 everywhere -> p == 0 -> free fall at exactly g
    assert controls["uniform_lattice_mass"]["a_max_over_g"] == pytest.approx(1.0, rel=1e-9)
    assert controls["uniform_lattice_mass"]["p_max"] < 1e-6
    # analytic profile: the truncation residual a continuum prior cannot remove
    analytic = controls["analytic_hydrostatic_profile"]["a_max_over_g"]
    assert analytic > 0.05, "the operator looks dead: the analytic prior must not balance"
    assert analytic < 1.0
    # and the reconstruction's near-zero residual is a *balance*, not a dead path:
    response = _load(f"{PERIODIC}.json")["perturbation_response"]
    small, large = response["0.0001"], response["0.001"]
    assert 0.05 < small / large < 0.15, (
        "residual does not respond linearly to a mass perturbation")


def test_error_judgement_is_pinned_to_the_shipped_numbers(report):
    """The two auxiliary profile comparisons, and which side of 3 % they are on."""
    profiles = report["runs"][PERIODIC]["profiles"]
    # density: the discrete equilibrium realises the Tait profile to 0.03 %
    assert profiles["density_vs_eos_max_rel"] < 1e-3
    # pressure: 4.5 % -- above 3 %, reported as measured, NOT sold as a pass
    assert 0.01 < profiles["pressure_vs_eos_max_rel"] < 0.10
    assert profiles["pressure_vs_eos_max_rel"] > 0.03


def test_reproducibility_map_is_honest(report):
    repro = report["reproducibility"]
    stable = set(repro["stable"])
    rounding = set(repro["rounding_level"])
    assert rounding, "no rounding-level quantities declared"
    assert stable.isdisjoint(rounding)
    assert repro["stable_rtol"] < 1e-6
    low, high = repro["rounding_ratio_band"]
    assert low < 1.0 < high and high >= 1.3, (
        "the declared band must accommodate the measured run-to-run scatter")
    assert "not" in repro["note"] or "digits" in repro["note"]
    # every rounding-level quantity is graded by the anchor in both records
    for name in (PERIODIC, WALL):
        gates = report["runs"][name]["gates"]
        assert gates["verdicts"]["G1"] is True
        for key in ("equilibrium.a_max_over_g", "rest.max_v_over_c0"):
            assert key in rounding


def test_evidence_is_rederivable_from_source_on_a_fresh_checkout():
    """The cheap half of the case: geometry + the analytic negative control.

    ``G7`` is one ``rhs`` evaluation on the analytic profile -- no
    reconstruction -- so the shipped liveness number can be checked on every
    checkout (~2 s) instead of being taken on faith.  The full 300-particle
    reconstruction stays reproducible with ``freeze.py --verify``.
    """
    published = LakeConfig(dx=0.02, depth=0.30, width=0.40, t_end=0.50,
                           periodic_x=True)
    previous = torch.get_default_dtype()
    torch.set_default_dtype(published.torch_dtype)
    try:
        state = build_state(published)
        prior = analytic_profile(state, published)
        control = residual_report(state, published, prior["mass"])
    finally:
        torch.set_default_dtype(previous)

    record = _load(f"{PERIODIC}.json")
    geometry = record["geometry"]
    assert state["n_fluid"] == geometry["particles"]
    assert state["n_wall"] == geometry["wall_particles"]
    assert int(state["pairs"].shape[0]) == geometry["pairs"]
    assert published.h() == pytest.approx(geometry["h"], rel=1e-12)
    assert control["a_max_over_g"] == pytest.approx(
        record["controls"]["analytic_hydrostatic_profile"]["a_max_over_g"],
        rel=1e-8)
    assert control["a_max_over_g_horizontal"] < 1e-9, (
        "the periodic operator should be x-translation invariant on the "
        "y-dependent analytic profile")