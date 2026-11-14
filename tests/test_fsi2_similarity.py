"""Dimensionless-similarity audit of the Turek-Hron FSI2 case.

Anchor: ``XVAL-SPH-FSI2-BENCHMARK-AND-FORCE-FROM-FLUID``.

These tests pin the audit itself: the benchmark target groups, the group values
of the repository's default case (which does *not* reproduce the benchmark
stiffness group), and the similarity-corrected constructor that does.
"""
from __future__ import annotations

import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from tensorsph import fsi2_similarity as sim  # noqa: E402
from tensorsph.fsi2_turek_hron import FSI2Case  # noqa: E402


def test_benchmark_targets_are_the_documented_fsi2_values():
    """The target groups are the ones the module docstring claims to match."""
    t = sim.BENCHMARK_FSI2
    assert t["Re"] == pytest.approx(100.0)
    assert t["rho_s_over_rho_f"] == pytest.approx(1.0)
    # E = 5.6 MPa, rho_f = 1000, U_mean = 1.0 m/s
    assert t["Ae"] == pytest.approx(5.6e6 / (1000.0 * 1.0 ** 2), rel=1e-12)
    assert t["slenderness"] == pytest.approx(0.35 / 0.02, rel=1e-12)


def test_default_case_is_not_similarity_equivalent():
    """The shipped default (E = 4 MPa at U_mean = 0.2 m/s) is ~18x too stiff.

    This is the defect the audit exists to catch: the case reproduces Re and the
    density ratio but not ``Ae = E / (rho U^2)``, so a coupled run cannot match
    the benchmark displacement amplitudes no matter how good the solver is.
    """
    case = FSI2Case()
    rep = sim.similarity_report(case)
    by_name = {g["name"]: g for g in rep["groups"]}
    assert by_name["Re"]["passed"]
    assert by_name["rho_s_over_rho_f"]["passed"]
    assert by_name["slenderness"]["passed"]
    assert not by_name["Ae"]["passed"]
    assert by_name["Ae"]["value"] == pytest.approx(4.0e6 / (1000.0 * 0.2 ** 2))
    assert by_name["Ae"]["rel_error"] == pytest.approx(16.857, rel=1e-3)
    assert rep["failed"] == ["Ae"]


def test_poisson_ratio_is_informational_for_the_rod_model():
    """The rod solid never consumes nu, so a 0.3 vs 0.4 mismatch is not a fail."""
    rep = sim.similarity_report(FSI2Case())
    by_name = {g["name"]: g for g in rep["groups"]}
    assert not by_name["poisson_s"]["passed"]
    assert "poisson_s" in rep["informational"]
    assert "poisson_s" not in rep["failed"]


def test_consequence_scaling_of_wrong_Ae():
    """Tip amplitude ~ 1/Ae and bending frequency ~ sqrt(Ae) (first order)."""
    case = FSI2Case()
    cons = sim.amplitude_scaling_consequence(case)
    assert cons["tip_amplitude_ratio"] == pytest.approx(
        sim.BENCHMARK_FSI2["Ae"] / cons["Ae"])
    assert cons["tip_amplitude_ratio"] == pytest.approx(0.056, rel=1e-2)
    assert cons["tip_amplitude_rel_error"] == pytest.approx(-0.944, abs=1e-3)
    assert cons["bending_frequency_ratio"] == pytest.approx(
        (cons["Ae"] / sim.BENCHMARK_FSI2["Ae"]) ** 0.5)


def test_similarity_corrected_case_passes_all_consumed_groups():
    """At U_mean = 0.2 m/s the benchmark similarity gives E = 2.24e5 Pa."""
    case = sim.similarity_corrected_case(U_mean=0.2)
    rep = sim.similarity_report(case)
    assert rep["passed"], rep["failed"]
    assert case.beam_E == pytest.approx(sim.BENCHMARK_FSI2["Ae"] * 1000.0 * 0.2 ** 2)
    assert case.beam_E == pytest.approx(2.24e5, rel=1e-9)
    assert case.viscosity == pytest.approx(0.2, rel=1e-9)  # Re = 100 at U_mean = 0.2
    assert case.inlet_u_max == pytest.approx(0.3, rel=1e-12)
    assert case.reynolds() == pytest.approx(100.0, rel=1e-9)


def test_corrected_case_keeps_benchmark_geometry():
    ref, case = FSI2Case(), sim.similarity_corrected_case()
    for name in ("channel_length", "channel_height", "cylinder_d", "cylinder_x",
                 "cylinder_y", "beam_length", "beam_thickness", "n_beam_nodes"):
        assert getattr(case, name) == pytest.approx(getattr(ref, name)), name


def test_oss_case_constants_differ_from_the_benchmark():
    """The archived OSS set-up is not the FSI2 parameter point (documented finding)."""
    oss = sim.OSS_FSI2_CASE
    assert oss["Ae"] == pytest.approx(1.4e3)
    assert sim.BENCHMARK_FSI2["Ae"] / oss["Ae"] == pytest.approx(4.0)
    assert oss["rho0_s"] / oss["rho0_f"] == pytest.approx(10.0)
    assert sim.BENCHMARK_FSI2["rho_s_over_rho_f"] == pytest.approx(1.0)
    # length scale: D' = 1.0 vs benchmark D = 0.1 m
    assert oss["length_scale"] == pytest.approx(
        2.0 * 0.5 / FSI2Case().cylinder_d)


def test_corrected_case_is_a_valid_case():
    """The corrected preset must still pass the module's own validation."""
    case = sim.similarity_corrected_case(U_mean=0.2)
    case.validate()
    assert case.mach() <= 0.02
    assert case.rod_substeps_effective() >= 1