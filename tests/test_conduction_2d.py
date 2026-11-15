"""Tests for the 2-D conduction operator family (CASE-SPH-CONDUCTION-2D).

What is asserted (invariants, not just "the code ran"):

* the ``torch.roll`` fast path is the *same* operator as the dense pair sum;
* the 2-D code degenerates exactly to the shipped 1-D ``conduction_rhs``;
* the lattice product mode is an eigenfunction with rate ``symbol_rate`` -- this is
  what lets the symbol be used as a calibration at all;
* calibrating with the 2-D symbol makes the mode decay exact (RK4), while the 1-D
  calibration (dimensional mismatch) keeps a >100 % error: the trap is pinned;
* the 2-D-calibrated arm converges at ~2nd order and the committed artefact's main
  run is inside the 3 % gate, and the registry reads that same number;
* ``heat_conduction_operator`` now has two benchmark *families* (the verification
  debt this case was built to pay) and the refuted 1-D-lift record is present.
"""
from __future__ import annotations

import json
from pathlib import Path

import torch

from tensorsph.heat_conduction_2d import (
    ModeDecayConfig,
    degenerate_1d_agreement,
    fd_mode_decay,
    lattice_offsets,
    mode_field,
    run_mode_decay,
    stencil_rhs_2d,
    stencil_vs_dense_2d,
    symbol_rate,
)

ROOT = Path(__file__).resolve().parents[1]
ARTEFACT = ROOT / "benchmarks" / "case_conduction_2d" / "result.json"
REGISTRY = ROOT / "docs" / "reports" / "method_selection_registry.json"


def test_stencil_is_the_dense_pair_sum():
    assert stencil_vs_dense_2d() < 1e-12


def test_two_d_code_degenerates_to_shipped_one_d_operator():
    # exact, not approximate: same kernel, same harmonic mean, same 2/rho
    assert degenerate_1d_agreement() == 0.0


def test_product_mode_is_an_eigenfunction_of_the_lattice_operator():
    n, dx = 24, 1.0 / 24
    h = 1.3 * dx
    k = 2.0 * 3.141592653589793
    field = mode_field(n, dx, k, k)
    rate = stencil_rhs_2d(field, lattice_offsets(dx, h))
    lam = symbol_rate(dx, h, k, k)
    assert float((rate * field).sum() / (field * field).sum()) == \
        _approx(-lam, rel=1e-12)
    residual = rate + lam * field
    assert float(residual.abs().max()) < 1e-10 * float(field.abs().max())


def _approx(value, rel):
    class _C:
        def __eq__(self, other):
            return abs(float(other) - value) <= rel * abs(value)
    return _C()


def test_symbol_calibration_makes_the_decay_exact():
    res = run_mode_decay(ModeDecayConfig(n=32, calibration="mode_2d",
                                         integrator="rk4"))
    assert res.max_normalised_error < 1e-7


def test_one_d_calibration_does_not_transfer_to_two_d():
    res = run_mode_decay(ModeDecayConfig(n=32, calibration="lattice_1d",
                                         integrator="euler"))
    assert res.max_normalised_error > 1.0     # >100 %: O(dx) dimensional mismatch


def test_two_d_calibrated_arm_is_second_order_and_inside_the_gate():
    coarse = run_mode_decay(ModeDecayConfig(n=32, calibration="k0_2d",
                                            integrator="rk4"))
    fine = run_mode_decay(ModeDecayConfig(n=64, calibration="k0_2d",
                                          integrator="rk4"))
    assert fine.max_normalised_error < coarse.max_normalised_error
    order = (torch.log(torch.tensor(coarse.max_normalised_error / fine.max_normalised_error))
             / torch.log(torch.tensor(2.0)))
    assert 1.5 < float(order) < 2.5
    assert fine.max_normalised_error < 0.03


def test_committed_artefact_is_inside_the_gate():
    doc = json.loads(ARTEFACT.read_text())
    assert doc["case"] == "CASE-SPH-CONDUCTION-2D"
    assert doc["main_run"]["max_normalised_error"] <= doc["gate"]
    assert doc["equivalence"]["stencil_vs_dense_max_abs"] < 1e-12
    assert doc["equivalence"]["degenerate_1d_vs_shipped_conduction_rhs_max_abs"] == 0.0
    # the independent reference validates the closed form used as the target
    assert doc["fd_reference"][0]["max_normalised_error_vs_closed_form"] < 0.02


def test_registry_reads_the_same_number_and_the_operator_has_two_families():
    from tensorsph.method_selection import case_family, load_registry

    doc = json.loads(ARTEFACT.read_text())
    registry = load_registry()
    ev = registry["heat_conduction_operator"]
    families = {case_family(e.case) for e in ev}
    assert len(families) >= 2, families
    banked = [e for e in ev if not e.is_refuted and e.case.startswith("CASE-SPH-CONDUCTION-2D")]
    assert banked, "the 2-D record must be banked"
    assert abs(banked[0].measured_pct - doc["main_run"]["max_normalised_error"] * 100.0) < 1e-9
    refuted = [e for e in ev if e.is_refuted]
    assert refuted and all(e.measured_pct > 100.0 for e in refuted)


def test_fd_reference_converges_to_the_closed_form():
    coarse = fd_mode_decay(n=32)["max_normalised_error_vs_closed_form"]
    fine = fd_mode_decay(n=64)["max_normalised_error_vs_closed_form"]
    assert fine < coarse
    assert fine < 0.01
