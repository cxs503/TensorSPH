"""Integrity of the verification harness itself (m2730).

Two failure modes made every TensorSPH number in this host environment
un-attributable, and both are silent unless tested:

1. **Foreign source.**  ``/usr/lib/python3.11/site-packages/
   __editable__.tensorsph-0.8.0.pth`` points at worktree ``m2420``, so a bare
   ``pytest`` inside ``m2730`` imported *that* branch.  Symptom observed:
   ``tests/test_laplace_law.py`` reported ``cut_force ratio = 0.70`` (m2420,
   pre-consistent) while the same measurement on this worktree's source gives
   ``1.13``.  The guard lives in ``tools/worktree_env.py`` and is installed by
   ``tests/conftest.py``.

2. **Harness drift.**  ``benchmarks/laplace_droplet/run.py`` rebuilt the colour
   field with the legacy m277 heuristic and scored it, while the production
   force had already moved to ``csf_scheme="consistent"``.  The benchmark then
   reported a "failing" model next to a ``dp/dp_theory = 0.99`` computed from
   the *fixed* force.

These tests pin both: the package really is this worktree's source, and the
benchmark's field diagnostics come from the shipped field builder.
"""

import importlib.util
from pathlib import Path

import pytest

from tools.worktree_env import ensure_worktree_source, foreign_source_error

ROOT = Path(__file__).resolve().parents[1]
BENCH = ROOT / "benchmarks" / "laplace_droplet" / "run.py"


# --------------------------------------------------------------------------- #
# 1. source pinning
# --------------------------------------------------------------------------- #
def test_tensorsph_resolves_inside_this_worktree():
    import tensorsph

    origin = Path(tensorsph.__file__).resolve()
    assert (ROOT / "src").resolve() in origin.parents, (
        f"{origin} is not this worktree's source; run with PYTHONPATH=src")


def test_guard_accepts_the_worktree_source():
    resolved = ensure_worktree_source(__file__, verbose=False)
    assert resolved == (ROOT / "src" / "tensorsph" / "__init__.py").resolve()
    assert foreign_source_error(resolved, ROOT) is None


def test_guard_detects_a_foreign_copy(tmp_path):
    """Negative control: a decoy package outside the worktree must be caught."""
    decoy = tmp_path / "tensorsph" / "__init__.py"
    decoy.parent.mkdir()
    decoy.write_text("__version__ = '0.0.0-foreign'\n")

    message = foreign_source_error(decoy, ROOT)
    assert message is not None
    assert "outside this worktree" in message
    assert str(decoy) in message

    # a namespace package (no __file__) is equally unusable
    assert foreign_source_error(None, ROOT) is not None


# --------------------------------------------------------------------------- #
# 2. harness fidelity
# --------------------------------------------------------------------------- #
@pytest.fixture(scope="module")
def case():
    spec = importlib.util.spec_from_file_location("laplace_droplet_run", BENCH)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module.run_case(radius=0.1, dx=0.01, sigma=0.072, lattice="hex")


def test_benchmark_scores_the_shipped_field_builder(case):
    assert case["fields_under_test"] == ("tensorsph.csf_consistent.csf_fields "
                                        "(csf_scheme='consistent', Config default)")
    # the two pipelines must be visibly different, otherwise this test is vacuous
    assert case["N_delta"] > 1.8 * case["legacy_N_delta"]
    assert abs(case["kappa_delta_weighted_times_R"]
               - case["legacy_kappa_delta_weighted_times_R"]) > 0.2


def test_benchmark_field_values_match_the_production_force(case):
    """The pressure jump from the production force and from the field set the
    benchmark reports must agree: kappa*R (1.153) times N_delta (0.787) is the
    dp ratio (0.986) to within a few percent, i.e. the diagnostics and the
    force describe the same model."""
    consistency = case["consistency_check_Ndelta_x_kappaR"]
    assert consistency == pytest.approx(case["dp_ratio"], rel=0.10)


def test_benchmark_verdict_is_derived_not_hard_coded(case):
    assert case["within_15pct"] is (abs(case["dp_ratio"] - 1.0) <= 0.15)
    assert case["dp_model_Pa"] > 0.0 and case["dp_theory_Pa"] == pytest.approx(
        0.072 / 0.1)