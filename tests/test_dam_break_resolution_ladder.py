"""Guards on the B2 dam-break experimental resolution ladder.

The ladder artifact (``benchmarks/dam_break/resolution_ladder.json``) is produced
by a simulation run that costs minutes, so it is archived and pinned here
instead of being re-simulated by the test suite (same contract as
``tests/test_laplace_law.py``).

What is pinned:

1. the measured mean absolute error in ``Z/a`` at each resolution, so that any
   change to the solver, the boundary model or the non-dimensionalisation shows
   up as a test failure;
2. the *finding* that the error does not decay under refinement -- if a future
   fix removes the systematic bias this test fails and forces the verification
   record to be rewritten, rather than letting a stale "fails" claim survive;
3. that ``docs/benchmarks/dam-break-resolution-ladder.md`` is a rendering of the
   artifact and not hand-edited numbers.
"""
from __future__ import annotations

import importlib.util
import json
import math
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
ARTIFACT = ROOT / "benchmarks" / "dam_break" / "resolution_ladder.json"
DOCUMENT = ROOT / "docs" / "benchmarks" / "dam-break-resolution-ladder.md"

# Mean absolute error in Z/a measured at each dx (m); rerun the benchmark and
# edit these only together with the artifact.
MAE_PINNED = {
    0.04: 0.5988,
    0.02: 0.5774,
    0.01: 0.5728,
}
MAE_TOLERANCE = 0.02

GATE_LIMIT = 0.15          # --max-exp-error default of the benchmark
MAX_DECAY_ORDER = 0.30     # "converged" would be >= 0.5; measured is ~0.05


@pytest.fixture(scope="module")
def payload():
    assert ARTIFACT.exists(), f"missing artifact {ARTIFACT.relative_to(ROOT)}"
    return json.loads(ARTIFACT.read_text())


@pytest.fixture(scope="module")
def rungs(payload):
    """Results sorted coarse -> fine."""
    return sorted(payload["results"], key=lambda r: -r["dx"])


def _doc_generator():
    spec = importlib.util.spec_from_file_location(
        "make_ladder_doc", ROOT / "benchmarks" / "dam_break" / "make_ladder_doc.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def test_ladder_covers_the_expected_resolutions(rungs):
    assert [r["dx"] for r in rungs] == [0.04, 0.02, 0.01]
    for r in rungs:
        assert r["experimental_error"]["points_compared"] > 0
        assert r["mach_number"] < 0.25
        assert r["escaped"] == 0


def test_particle_count_scales_with_resolution(rungs):
    """dx halves -> particle count quadruples (2-D lattice)."""
    counts = [r["particles"] for r in rungs]
    assert counts == [50, 200, 800]
    for coarse, fine in zip(counts, counts[1:]):
        assert fine == pytest.approx(4 * coarse)


def test_measured_error_is_pinned(rungs):
    for r in rungs:
        pinned = MAE_PINNED[r["dx"]]
        measured = r["experimental_error"]["mean_absolute_error"]
        assert measured == pytest.approx(pinned, abs=MAE_TOLERANCE), (
            f"dx={r['dx']}: MAE {measured:.4f} moved away from the recorded "
            f"{pinned:.4f}; if the solver changed on purpose, regenerate the "
            f"artifact and update MAE_PINNED and the document")


def test_all_experimental_points_are_inside_the_simulation_window(rungs):
    for r in rungs:
        assert r["experimental_error"]["points_compared"] == 20


def test_error_does_not_decay_with_resolution(rungs):
    """The headline finding: the residual is a bias, not numerical diffusion.

    The front trajectory converges at order ~1 against the Ritter shallow-water
    solution, so refining dx must shrink *discretisation* error. Against the
    experiment it does not: the observed order is near zero. If this ever turns
    into a real convergence rate, the failure claim in the document has to be
    revisited -- which is exactly why it is asserted.
    """
    coarse, fine = rungs[0], rungs[-1]
    mae_c = coarse["experimental_error"]["mean_absolute_error"]
    mae_f = fine["experimental_error"]["mean_absolute_error"]
    p = math.log(mae_c / mae_f) / math.log(coarse["dx"] / fine["dx"])
    assert p < MAX_DECAY_ORDER, (
        f"error decay order {p:.3f} >= {MAX_DECAY_ORDER}: the experimental "
        f"comparison now converges, rewrite the verification record")


def test_the_gate_still_fails_at_the_finest_rung(rungs):
    """`physical_validation` stays `fail`: the platform bar is 15 %."""
    mae = rungs[-1]["experimental_error"]["mean_absolute_error"]
    assert mae > GATE_LIMIT
    assert rungs[-1]["experimental_error"]["within_uncertainty"] < 1.0


def test_the_document_is_a_rendering_of_the_artifact(payload):
    """No hand-copied numbers: the committed markdown must equal the render."""
    assert DOCUMENT.exists(), f"missing {DOCUMENT.relative_to(ROOT)}"
    rendered = _doc_generator().render(payload)
    assert DOCUMENT.read_text() == rendered, (
        "docs/benchmarks/dam-break-resolution-ladder.md is stale; rerun "
        "python benchmarks/dam_break/make_ladder_doc.py")
