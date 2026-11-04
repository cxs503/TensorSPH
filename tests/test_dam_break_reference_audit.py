"""Regression guards for the B2 dam-break reference audit.

Two things are pinned here, both recorded in
``benchmarks/dam_break/reference_audit.json`` and
``docs/benchmarks/dam-break-reference-audit.md``:

1. **The bundled "Martin & Moyce (1952)" table is not a digitisation.** Its
   abscissa is an exact arithmetic sequence and every ordinate is an exact
   multiple of 0.05, so it must not be used as a validation target. The table is
   kept (it is still a plot reference) but flagged in the JSON itself; the tests
   fail if that flag is ever removed without new provenance, and they fail if
   the numbers are replaced by something *with* scatter without updating the
   flag -- in both directions the honest state has to be rewritten by hand.
2. **The analytic reference is implemented correctly.** Ritter's coefficient is
   ``c = 2`` in ``d(Z/a)/d tau``; the harness used ``c = 1`` until this audit.
   The solver's measured coefficient is pinned per resolution and per floor
   boundary condition so a solver change cannot silently move it.
"""
from __future__ import annotations

import json
import math
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
TABLE = ROOT / "benchmarks" / "dam_break" / "martin_moyce_1952.json"
ARTIFACT = ROOT / "benchmarks" / "dam_break" / "reference_audit.json"
DOC = ROOT / "docs" / "benchmarks" / "dam-break-reference-audit.md"

# measured least-squares d(Z/a)/d tau; rerun reference_audit.py and edit these
# together with the artifact
C_PINNED = {
    (0.04, 0.00): 1.2016,
    (0.04, 0.25): 1.1099,
    (0.04, 0.50): 1.0580,
    (0.04, 0.75): 1.0123,
    (0.04, 1.00): 0.9682,
    (0.02, 0.00): 1.2468,
    (0.02, 1.00): 0.8959,
}
C_TOLERANCE = 0.03
RITTER_COEFFICIENT = 2.0


@pytest.fixture(scope="module")
def table():
    return json.loads(TABLE.read_text())


@pytest.fixture(scope="module")
def payload():
    assert ARTIFACT.exists(), f"missing {ARTIFACT.relative_to(ROOT)}"
    return json.loads(ARTIFACT.read_text())


# --------------------------------------------------------------------------- #
# 1. provenance of the validation target
# --------------------------------------------------------------------------- #
def test_table_is_flagged_as_unverified(table):
    audit = table.get("integrity_audit")
    assert audit is not None, "the provenance flag was dropped from the table"
    assert audit["grade"] == "WITHHELD"
    assert audit["status"] == "provenance_unverified"


def test_table_abscissa_is_an_exact_arithmetic_sequence(table):
    taus = [p["tau"] for p in table["data"]]
    step = (taus[-1] - taus[0]) / (len(taus) - 1)
    assert max(abs(t - i * step) for i, t in enumerate(taus)) < 1e-12
    assert step == pytest.approx(0.226, abs=5e-4)


def test_table_ordinates_are_quantised(table):
    """Every Z/a is a multiple of 0.05 to round-off: a generated table."""
    za = [p["Z_over_a"] for p in table["data"]]
    assert max(abs(z / 0.05 - round(z / 0.05)) for z in za) < 1e-9


def test_table_has_no_measurement_scatter(table):
    """Second differences stay inside the 0.05 quantum: the residual structure is
    the lattice, not noise. The exact lattice and quantum are the decisive
    evidence; this only shows nothing contradicts them."""
    za = [p["Z_over_a"] for p in table["data"]]
    second = [za[i + 1] - 2 * za[i] + za[i - 1] for i in range(1, len(za) - 1)]
    rms = math.sqrt(sum(d * d for d in second) / len(second))
    assert 0.0 < rms < 0.60 * 0.05
    assert max(abs(d) for d in second) <= 0.05 + 1e-12


def test_audit_agrees_with_the_table_flag(payload, table):
    audit = payload["table_audit"]
    assert audit["usable_as_validation_target"] is False
    assert audit["grade"] == "WITHHELD"
    assert audit["tau_lattice_residual"] < 1e-12
    assert audit["ordinate_quantum_residual"] < 1e-9
    assert audit["grade"] == table["integrity_audit"]["grade"]


# --------------------------------------------------------------------------- #
# 2. the analytic reference
# --------------------------------------------------------------------------- #
def test_ritter_identity_residual_is_zero(payload):
    r = payload["ritter"]
    assert r["H_over_a"] == 2.0
    assert r["max_identity_reldiff"] < 1e-12, "Z/a = 2 tau no longer holds"
    assert r["max_helper_reldiff"] < 1e-12, "shipped helper disagrees with Ritter"


def test_table_slope_is_well_below_ritter(payload):
    c = payload["table_slope"]["table_coefficient_c"]
    assert 0.5 < c < 1.0
    assert c / RITTER_COEFFICIENT < 0.5


# --------------------------------------------------------------------------- #
# 3. the solver's dimensionless front coefficient
# --------------------------------------------------------------------------- #
def test_measured_coefficients_are_pinned(payload):
    measured = {(r["dx"], r["wall_friction"]): r["front_coefficient_c"]
                for r in payload["runs"]}
    assert set(measured) == set(C_PINNED)
    for key, pinned in C_PINNED.items():
        assert measured[key] == pytest.approx(pinned, abs=C_TOLERANCE), (
            f"dx={key[0]}, friction={key[1]}: c moved from {pinned}; if the "
            f"solver changed on purpose, rerun reference_audit.py and update "
            f"C_PINNED and the document")


def test_every_run_is_bounded_and_incompressible(payload):
    for r in payload["runs"]:
        assert r["escaped"] == 0
        assert r["mach_number"] < 0.25
        assert r["max_density_compression"] < 0.05


def test_coefficient_falls_with_floor_friction(payload):
    """No-slip floor must slow the front: the boundary model is doing work."""
    rows = sorted((r for r in payload["runs"] if r["dx"] == 0.04),
                  key=lambda r: r["wall_friction"])
    cs = [r["front_coefficient_c"] for r in rows]
    assert all(b < a for a, b in zip(cs, cs[1:])), cs


def test_friction_alone_cannot_reproduce_the_bundled_table(payload):
    """Extrapolating c(friction) to friction = 1 still overshoots the table.

    The table's slope is ~0.735; the free-slip and no-slip runs bracket it from
    above at both resolutions, so the residual cannot be blamed on the missing
    bed friction alone -- and, since the table is unverified, it must not be
    blamed on the solver either. Both statements are pinned so that neither
    drifts into folklore.
    """
    table_c = payload["table_slope"]["table_coefficient_c"]
    for dx in (0.04, 0.02):
        cs = [r["front_coefficient_c"] for r in payload["runs"]
              if r["dx"] == dx and r["wall_friction"] in (0.0, 1.0)]
        if len(cs) < 2:
            continue
        assert min(cs) > table_c


def test_document_is_generated_from_the_artifact(payload):
    assert DOC.exists(), f"missing {DOC.relative_to(ROOT)}"
    text = DOC.read_text()
    for r in payload["runs"]:
        assert "%.4f" % r["front_coefficient_c"] in text, (
            "the document must quote the artifact numbers verbatim; rerun "
            "make_audit_doc.py")
