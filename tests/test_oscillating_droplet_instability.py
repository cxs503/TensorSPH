"""Guards for the mode-2 oscillation instability finding.

The recorded oscillating-droplet cases are analysed in
``benchmarks/oscillating_droplet_instability/``;
``docs/verification_oscillating_droplet_instability.md`` is rendered from
``report.json``.  These tests keep the *claims* of that doc
falsifiable:

1. the surface energy released by the seeded ellipse is a hard upper bound on the
   kinetic energy of an energy-conserving solver, and both recorded traces blow
   through it (so the cases are numerically invalid, not merely inaccurate);
2. the fitted mode-2 frequency of the same trace moves by more than the 15 %
   gate depending on the fit window (so no single number can be quoted as the
   model's frequency error);
3. the recorded report is self-consistent with the traces (no hand-edited
   numbers).
"""
from __future__ import annotations

import importlib.util
import json
import math
from pathlib import Path

import numpy as np
import pytest
import torch

torch.set_num_threads(1)

ROOT = Path(__file__).resolve().parents[1]
BENCH = ROOT / "benchmarks" / "oscillating_droplet"
INST = ROOT / "benchmarks" / "oscillating_droplet_instability"
GATE_PCT = 15.0


def _load(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.fixture(scope="module")
def osc():
    return _load("oscillating_run", BENCH / "run.py")


@pytest.fixture(scope="module")
def inst():
    return _load("instability_run", INST / "run.py")


@pytest.fixture(scope="module")
def report():
    return json.loads((INST / "report.json").read_text())


def test_available_energy_is_a_hard_bound_and_both_cases_violate_it(report, inst):
    """KE_max > sigma * (P_ellipse - P_circle) is proof of energy creation."""
    for case in inst.load_cases():
        energy = inst.available_energy(case)
        ke_max = float(np.abs(case.ke).max())
        assert energy["available_energy_J"] > 0.0
        assert ke_max > energy["available_energy_J"], (
            f"{case.name}: recorded KE peak {ke_max:.3g} J does not exceed the "
            f"available surface energy {energy['available_energy_J']:.3g} J; "
            "the instability claim in docs/verification_oscillating_droplet_instability.md "
            "no longer holds for this trace")
        assert ke_max / energy["available_energy_J"] > 1.0


def test_reported_divergence_matches_the_traces(report):
    """Every number the doc quotes from report.json must follow from the traces."""
    for row in report["divergence"]:
        energy = report["energy_budget"][row["case"]]
        ratio = row["kinetic_energy_max_J"] / energy["available_energy_J"]
        assert ratio == pytest.approx(row["kinetic_energy_over_available"], rel=1e-9)
        assert row["diverged"] is (ratio > 1.0)
        assert math.log10(ratio) == pytest.approx(row["energy_creation_decades"],
                                                  abs=1e-9)


@pytest.mark.slow
def test_fitted_frequency_depends_on_the_window_far_beyond_the_gate(osc, inst):
    """Same trace, two admissible windows: the 'measured' error moves > 15 pt."""
    case = inst.load_cases()[0]  # reference case, documented as -43.5 %
    errs = {}
    for frac in (0.75, 1.8):
        mask = case.t <= frac * case.period
        fit = osc.damped_cosine_fit(list(case.t[mask]), list(case.s[mask]),
                                    case.omega_ref)
        errs[frac] = (fit["omega"] - case.omega_ref) / case.omega_ref * 100.0
    spread = abs(errs[1.8] - errs[0.75])
    assert spread > GATE_PCT, (
        f"window dependence collapsed to {spread:.1f} pt: if the case became "
        "stable, re-baseline mode-2 frequency and update the doc instead of "
        "keeping the instability note")
    assert errs[0.75] > errs[1.8], (
        "including the diverged tail must pull the fitted omega down "
        f"(got {errs[0.75]:+.1f} % -> {errs[1.8]:+.1f} %)")


def test_release_case_trace_is_the_ci_pin_configuration(report):
    """The probe must reproduce the pinned CI case, not some other configuration."""
    setup = report["setup"]["release_R1cm"]
    assert setup["radius_m"] == pytest.approx(0.01)
    assert setup["dx_m"] == pytest.approx(0.001)
    assert setup["periods_simulated"] == pytest.approx(1.8, abs=0.05)
    pinned = -31.78
    recorded = [row for row in report["window_study"]["release_R1cm"]][-1]
    assert abs(recorded["fit_rel_error_pct"] - pinned) < 2.0, (
        "the full-window fit of the probe trace must land on the number pinned "
        f"in tests/test_oscillating_droplet.py ({pinned} %), got "
        f"{recorded['fit_rel_error_pct']:.2f} %")
