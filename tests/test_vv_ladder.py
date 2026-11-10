"""pytest cases for the verification ladder (TensorSPH).

The suite is split so that CI cost stays bounded: the *contract* tests (schema,
provenance, negative-control wiring) are cheap and run on synthetic documents,
while the *physics* tests run each rung at the coarsest resolution that still
exercises the operator. The full-resolution ladder is a benchmark
(``benchmarks/verification_ladder/run.py``), not a unit test.
"""
from __future__ import annotations

import json
import math

import pytest
import torch

from tensorsph.vv.dambreak import (dam_break_case, filtered_front,
                                   jet_dispersion, load_martin_moyce)
from tensorsph.vv.hydrostatic import HydrostaticConfig, run_hydrostatic
from tensorsph.vv.ladder import LadderConfig, provenance, run_ladder
from tensorsph.vv.schema import (LADDER_SCHEMA_VERSION, validate_ladder_json)
from tensorsph.vv.tgv import TGVConfig, analytic_field, fit_decay_rate, run_tgv


@pytest.fixture(scope="module", autouse=True)
def _bounded_threads():
    """Keep torch from spawning one thread per core on these small tensors.

    The ladder's tensors are small (a few thousand particles), so a 128-thread
    pool spends more time in barriers than in work: measured 137 ms/step with
    the default pool against 20 ms/step with one thread for the same case.
    """
    previous = torch.get_num_threads()
    torch.set_num_threads(4)
    yield
    torch.set_num_threads(previous)


# --------------------------------------------------------------------------- #
# schema / contract
# --------------------------------------------------------------------------- #
def _synthetic_doc(error: float = 1.0) -> dict:
    anchors = [{
        "id": anchor_id,
        "title": "t",
        "physics": "p",
        "reference": {"kind": "analytic"},
        "measured": {},
        "error_pct": error,
        "gate": {"quantity": "q", "limit_pct": 3.0, "pass": error < 3.0},
        "verdict": "PASS" if error < 3.0 else "FAIL",
        "config": {},
        "series": [],
        "figures": [],
    } for anchor_id in ("C1_hydrostatic", "C2_tgv", "C3_dambreak")]
    controls = [{
        "id": "C4a", "title": "t", "mechanism": "m", "metric": "x",
        "expected": "e", "measured": {}, "verdict": "PASS",
    }]
    return {
        "schema_version": LADDER_SCHEMA_VERSION,
        "node": "p83w1-n01",
        "generated": "2026-01-01T00:00:00Z",
        "provenance": {"git_sha": "x", "python": "3", "torch": "2",
                       "package_version": "0", "command": "c"},
        "anchors": anchors,
        "negative_controls": controls,
        "summary": {"anchors_total": 3, "anchors_passed": 3,
                    "all_passed": True, "honest_failures": []},
        "campaign_criteria": {},
        "limitations": [],
    }


def test_schema_accepts_complete_document():
    assert validate_ladder_json(_synthetic_doc()) == []


def test_schema_rejects_missing_anchor():
    doc = _synthetic_doc()
    doc["anchors"] = doc["anchors"][:2]
    errors = validate_ladder_json(doc)
    assert any("C3_dambreak" in e for e in errors)


def test_schema_rejects_inconsistent_verdict():
    doc = _synthetic_doc()
    doc["anchors"][0]["gate"]["pass"] = True
    doc["anchors"][0]["error_pct"] = 10.0
    doc["anchors"][0]["verdict"] = "PASS"
    errors = validate_ladder_json(doc)
    assert any("verdict inconsistent" in e for e in errors)


def test_provenance_has_required_keys():
    prov = provenance("pytest")
    for key in ("git_sha", "python", "torch", "package_version", "command"):
        assert key in prov


# --------------------------------------------------------------------------- #
# rung 1: hydrostatic
# --------------------------------------------------------------------------- #
@pytest.mark.slow
@pytest.mark.timeout(300)
def test_hydrostatic_profile_matches_analytic():
    run = run_hydrostatic(HydrostaticConfig(dx=0.02, seconds=1.0,
                                            average_from=0.6))
    assert run["measured"]["bands"], "no bands were sampled"
    # The interior must sit on rho g h to a few percent at this reduced
    # resolution; the full 1 % gate is the benchmark's job (4 s run, dx = 0.02).
    assert run["error_pct"] < 5.0, run["measured"]["max_band_error_pct"]
    # Pressure must increase monotonically downwards.
    pressures = [b["pressure"] for b in run["measured"]["bands"]]
    assert pressures[0] > pressures[-1]
    assert run["gate"]["limit_pct"] == 1.0


def test_hydrostatic_gravity_off_is_the_negative_control():
    run = run_hydrostatic(HydrostaticConfig(dx=0.04, seconds=0.4),
                          gravity_override=0.0)
    assert run["reference"]["g"] == 9.81  # the reference is not moved
    # With no gravity the analytic reference is still rho g h, so a working
    # implementation must now be *wrong* by ~100 %.
    assert abs(run["error_pct"]) > 50.0


# --------------------------------------------------------------------------- #
# rung 2: Taylor-Green
# --------------------------------------------------------------------------- #
def test_tgv_initial_field_satisfies_analytic_identities():
    cfg = TGVConfig(dx=0.1)
    from tensorsph.vv.tgv import make_lattice
    pos, _mass = make_lattice(cfg)
    vel = analytic_field(pos, 0.0, cfg)
    # The exact field is divergence free on the lattice (spectral identity).
    k = cfg.k
    div = (cfg.U0 * k * torch.cos(k * pos[:, 0]) * torch.cos(k * pos[:, 1])
           - cfg.U0 * k * torch.cos(k * pos[:, 0]) * torch.cos(k * pos[:, 1]))
    assert float(div.abs().max()) < 1e-12
    assert float(vel.abs().max()) > 0.0


@pytest.mark.slow
@pytest.mark.timeout(300)
def test_viscous_operator_reproduces_the_laplacian():
    """The split-out Morris Laplacian must apply ``lap u = -2 k^2 u`` exactly."""
    from tensorsph.vv.tgv import make_lattice, viscous_acceleration
    cfg = TGVConfig(dx=0.025)
    pos, mass = make_lattice(cfg)
    vel = analytic_field(pos, 0.0, cfg)
    acc = viscous_acceleration(pos, vel, mass, cfg)
    exact = -2.0 * cfg.viscosity * cfg.k ** 2 * vel
    scale = float(exact.square().sum().sqrt())
    error = float((acc - exact).square().sum().sqrt()) / scale
    assert error < 0.05, error


@pytest.mark.slow
@pytest.mark.timeout(300)
def test_tgv_decay_rate_matches_analytic():
    run = run_tgv(TGVConfig(dx=0.05, t_end=0.3, U0=0.2, sound_speed=10.0))
    rate_a = run["reference"]["energy_decay_rate"]
    rate = run["measured"]["energy_decay_rate"]
    assert rate_a > 0.0
    assert abs(rate - rate_a) / rate_a < 0.06, (rate, rate_a)
    assert run["measured"]["energy_decay_rate_r2"] > 0.99


@pytest.mark.slow
@pytest.mark.timeout(300)
def test_tgv_viscosity_off_kills_the_decay():
    cfg = TGVConfig(dx=0.05, t_end=0.2, U0=0.2, sound_speed=10.0)
    nominal_rate = cfg.analytic_decay_rate()
    run = run_tgv(cfg, viscosity_override=0.0)
    # With nu = 0 only the (small) numerical dissipation floor may remain.
    assert abs(run["measured"]["energy_decay_rate"]) < 0.25 * nominal_rate


def test_fit_decay_rate_recovers_a_known_exponent():
    times = [i * 0.01 for i in range(101)]
    energies = [math.exp(-3.0 * t) for t in times]
    rate, r2 = fit_decay_rate(times, energies, 0.1, 0.9)
    assert rate == pytest.approx(3.0, rel=1e-6)
    assert r2 == pytest.approx(1.0, abs=1e-12)


# --------------------------------------------------------------------------- #
# rung 3: dam break
# --------------------------------------------------------------------------- #
def test_martin_moyce_table_loads_with_provenance():
    exp = load_martin_moyce()
    assert exp["data"]
    assert "1952" in exp["reference"]
    assert all({"tau", "Z_over_a"} <= set(row) for row in exp["data"])


@pytest.mark.slow
@pytest.mark.timeout(300)
def test_dam_break_case_is_stable_and_the_front_advances():
    cfg = dam_break_case(dx=0.05, seconds=0.15)
    assert cfg.steps > 0
    from tensorsph.dam_break import run_case
    run = run_case(cfg)
    assert run.front[-1] > run.front[0]
    assert abs(run.mass_drift) < 1e-9 * max(run.initial_mass, 1.0)
    assert run.max_speed / cfg.sound_speed < 0.3


def test_filtered_front_ignores_isolated_spray():
    import torch
    pos = torch.tensor([[0.0, 0.0], [0.0, 0.02], [0.02, 0.0], [0.02, 0.02],
                        [1.0, 1.0]])  # the last particle is a lone spray particle
    h = 0.02
    assert filtered_front(pos, h, min_neighbours=2) < 0.5


def test_jet_dispersion_reports_fragments():
    import torch
    pos = torch.tensor([[0.0, 0.0], [0.0, 0.02], [0.02, 0.0], [0.02, 0.02],
                        [1.0, 1.0]])
    stats = jet_dispersion(pos, 0.02, min_neighbours=2)
    assert stats["detached_particles"] >= 1
    assert stats["detached_fraction"] == pytest.approx(0.2)


# --------------------------------------------------------------------------- #
# whole ladder, quick mode
# --------------------------------------------------------------------------- #
@pytest.mark.slow
@pytest.mark.timeout(900)
def test_quick_ladder_emits_valid_json(tmp_path):
    cfg = LadderConfig().quickify()
    doc = run_ladder(cfg, command="pytest", log=lambda *a, **k: None)
    assert validate_ladder_json(doc) == []
    out = tmp_path / "ladder.json"
    out.write_text(json.dumps(doc))
    assert json.loads(out.read_text())["summary"]["anchors_total"] == 3