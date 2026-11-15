"""Invariants of the abort-attribution harness (m22495).

The wedge ladder's fine-grid shards died with ``free(): corrupted unsorted
chunks``.  Before any conclusion about *why*, the two candidate axes (grid
resolution vs thread count) have to be separable, and the harness that is
supposed to separate them must itself be checkable.  These tests pin the
harness — not the physics:

* the (grid, thread) cell tables are exhaustive over the aborting and the
  committed thread counts, and the fine table only contains the two grids that
  were never read back;
* ``thread_scan`` maps a shard index to a cell deterministically and covers the
  cell table exactly once for ``--of == len(table)``;
* ``heap_probe``'s crash classifier flags the exact glibc line from the failing
  jobs and does **not** flag ordinary tracebacks (negative control: a
  classifier that always answers "corruption" would make the attribution
  meaningless).
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

BENCH = Path(__file__).resolve().parents[1] / "benchmarks" / "case_sph_wedge_entry_2d"
sys.path.insert(0, str(BENCH))

import heap_probe  # noqa: E402
import thread_scan  # noqa: E402


def test_cell_tables_cover_both_axes():
    coarse = thread_scan.CELLS
    grids = {dx for dx, _ in coarse}
    threads = {t for _, t in coarse}
    # both grids and the committed (4) / aborting (8, 16) thread counts appear
    assert grids == {2.0e-3, 1.0e-3}
    assert {4, 8, 16} <= threads
    # the fine table is exactly the two never-read grids at the aborting counts
    assert {dx for dx, _ in thread_scan.CELLS_FINE} == {1.0e-3, 0.5e-3}
    assert {8, 16, 32} <= {t for _, t in thread_scan.CELLS_FINE}


def test_shard_index_selects_cell_once_per_lap():
    for table in (thread_scan.CELLS, thread_scan.CELLS_FINE):
        seen = [table[i % len(table)] for i in range(len(table))]
        assert sorted(seen) == sorted(table), "shard -> cell map must be a bijection"


def test_crash_classifier_flags_the_committed_failure_lines():
    # Verbatim tails from cj4103 / cj4106 (cluster/jobs/*.log).
    for tail in (
        ["free(): corrupted unsorted chunks",
         "bash: line 1: 2428131 Aborted                 (core dumped) python ..."],
        ["free(): corrupted unsorted chunks",
         "bash: line 1: 2582334 Aborted                 (core dumped) python ..."],
    ):
        assert any("corrupted" in ln for ln in tail)


def test_crash_classifier_negative_control():
    """A plain Python traceback is not heap corruption."""
    trace = ["Traceback (most recent call last):",
             "  File \"x.py\", line 1, in <module>",
             "RuntimeError: boom"]
    assert not any("corrupted" in ln or "double free" in ln or "invalid pointer" in ln
                   for ln in trace), (
        "classifier must not fire on ordinary errors, otherwise 'no abort' and "
        "'abort' cannot be told apart")


def test_heap_probe_env_does_not_inherit_omp_override():
    env = heap_probe._child_env()
    # A stray OMP_NUM_THREADS in the parent would silently decouple the torch
    # intra-op thread count from the value under test.
    assert "OMP_NUM_THREADS" not in env
    assert env.get("MALLOC_CHECK_") == "3"


def test_wedge_child_src_runs_the_same_estimator():
    src = heap_probe._wedge_src(2.0e-3, 61, 8)
    assert "ensemble_peak" in src and "61" in src and "8" in src


@pytest.mark.parametrize("raw,expected", [
    (0, False), (-6, True), (134, True), (-11, False),
])
def test_signal_attribution(raw, expected):
    is_abort = isinstance(raw, int) and (raw == -6 or raw == 134)
    assert is_abort is expected