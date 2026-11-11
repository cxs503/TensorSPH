"""Tests of the acceptance-gate logic (pure functions, synthetic numbers)."""
from __future__ import annotations

import math

import pytest
import torch

from tensorsph.wcsph import chain


def test_gate_flips_exactly_at_its_limit():
    assert chain.gate("x", 0.10, 0.10)["verdict"] == "pass"
    assert chain.gate("x", 0.1000001, 0.10)["verdict"] == "fail"
    assert chain.gate("y", 1.0, 1.0, rule=">=")["verdict"] == "pass"
    assert chain.gate("y", 0.99, 1.0, rule=">=")["verdict"] == "fail"


def test_gate_withholds_when_the_measurement_is_missing_or_not_finite():
    assert chain.gate("x", None, 0.1)["verdict"] == "withheld"
    assert chain.gate("x", float("nan"), 0.1)["verdict"] == "withheld"


def test_summarise_never_reports_a_withheld_gate_as_a_pass():
    gates = [chain.gate("a", 0.0, 1.0), chain.gate("b", None, 1.0)]
    result = chain.summarise(gates)
    assert result["verdict"] == "withheld"
    assert result["pass"] == 1 and result["withheld"] == 1


def test_summarise_reports_fail_over_withheld():
    gates = [chain.gate("a", 2.0, 1.0), chain.gate("b", None, 1.0)]
    assert chain.summarise(gates)["verdict"] == "fail"


def test_convergence_order_of_a_known_rate():
    assert chain.convergence_order(0.04, 0.01, 0.02, 0.01) == pytest.approx(2.0)
    assert chain.convergence_order(0.03, 0.01, 0.02, 0.01) == pytest.approx(
        math.log(3.0) / math.log(2.0))
    assert math.isnan(chain.convergence_order(0.0, 0.1, 0.02, 0.01))


def test_relative_error_is_none_for_a_zero_reference():
    assert chain.relative_error(1.0, 0.0) is None
    assert chain.relative_error(1.1, 1.0) == pytest.approx(0.1)


def test_l2_profile_error_is_zero_for_identical_fields():
    dx = 0.01
    edges = torch.linspace(0.0, 1.0, 101, dtype=torch.float64)
    pos = torch.stack((torch.rand(200, dtype=torch.float64),
                       torch.rand(200, dtype=torch.float64)), dim=1)
    mass = torch.full((200,), 1.0, dtype=torch.float64)
    rho = torch.full((200,), 1000.0, dtype=torch.float64)
    assert chain.l2_profile_error(pos, mass, rho, pos.clone(), mass.clone(),
                                  rho.clone(), edges) == 0.0


def test_l2_profile_error_grows_with_the_displacement():
    edges = torch.linspace(0.0, 1.0, 101, dtype=torch.float64)
    base = torch.stack((torch.linspace(0.005, 0.995, 100, dtype=torch.float64),
                        torch.full((100,), 0.1, dtype=torch.float64)), dim=1)
    mass = torch.full((100,), 1.0, dtype=torch.float64)
    rho = torch.full((100,), 1000.0, dtype=torch.float64)
    # the profile is a per-bin volume sum: a *shift* by any distance leaves the
    # same L2 distance (one bin empties, one fills), so the discriminating
    # variable is how many particles cross a bin edge
    near = base.clone()
    near[50, 0] += 0.015
    far = base.clone()
    far[[50, 70], 0] += 0.015
    e_near = chain.l2_profile_error(base, mass, rho, near, mass, rho, edges)
    e_far = chain.l2_profile_error(base, mass, rho, far, mass, rho, edges)
    assert 0.0 < e_near < e_far


def test_evaluate_reports_every_criterion_and_marks_missing_ones_withheld():
    payload = {
        "pair": {"momentum_drift": 1e-13, "energy_drift": 0.001},
        "front_2d": {"max_relative_error": 0.05, "reference": "test"},
        "convergence": {"pairs": [{"label": "dx2", "order": 1.2}]},
        "tests": {"collected": 30, "coverage": 0.90},
    }
    result = chain.evaluate(payload)
    names = [g["criterion"] for g in result["gates"]]
    assert "a-momentum" in names and "e-convergence-dx2" in names
    assert result["summary"]["verdict"] == "withheld"   # 3-D + sloshing absent
    assert any(g["verdict"] == "withheld"
               for g in result["gates"] if g["criterion"] == "c-3d-front")


def test_evaluate_fails_loudly_when_a_number_is_out_of_spec():
    payload = {
        "pair": {"momentum_drift": 1e-3, "energy_drift": 0.05},
        "front_2d": {"max_relative_error": 0.5, "reference": "test"},
        "front_3d": {"max_relative_error": 0.5, "reference": "test"},
        "sloshing": {"rows": [{"frequency": 1.0, "relative_error": 0.4,
                               "reference": "test"}]},
        "convergence": {"pairs": [{"label": "dx2", "order": 0.2}]},
        "tests": {"collected": 5, "coverage": 0.5},
    }
    result = chain.evaluate(payload)
    assert result["summary"]["verdict"] == "fail"
    failed = {g["criterion"] for g in result["gates"] if g["verdict"] == "fail"}
    assert {"a-momentum", "a-energy", "b-2d-front", "c-3d-front",
            "d-sloshing-1.0Hz", "e-convergence-dx2", "f-test-count",
            "f-coverage"} <= failed