"""Tests for the fail-closed ladder assembler (`scripts/finalize_stokes_second.py`).

The assembler exists because the ladder is produced as independent per-rung
shards.  Its predecessor published a one-rung "ladder" with
``G6_not_growing: true``: it waited at most 2 s for a shard (a 5 s sleep inside
a 2 s guard) and assembled whatever happened to be on disk.  These tests pin the
replacement's contract:

* a complete, consistent set of rungs is assembled, hashed and graded by the
  same convergence code ``run.py --ladder`` uses;
* a missing rung, a duplicated rung, a disagreeing case configuration or a
  single-rung "ladder" is **refused** — nothing is published and the previously
  published ladder is not touched;
* a shard that lands late is *waited for*, not skipped (the root cause above).

Every payload here is synthetic: the assembler is a bookkeeping device, so it is
tested without running the solver (the physics is graded by
``tests/test_stokes_second.py`` and `docs/verification_stokes_second.md`).
"""

from __future__ import annotations

import importlib.util
import json
import math
import threading
import time
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[1]


def _load_assembler():
    path = REPO / "scripts" / "finalize_stokes_second.py"
    spec = importlib.util.spec_from_file_location("finalize_stokes_second", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


asm = _load_assembler()

NU, OMEGA, U0 = 0.01, 2.0 * math.pi, 0.01


def _probe_payload(res: int, l2: float, **case_overrides) -> dict:
    """A probe file with the fields the assembler reads and checks."""
    case = {"nu": NU, "omega": OMEGA, "U0": U0, "H_over_delta": 6.0,
            "columns": 6, "c0": 4.0, "periods": 6.0, "measure_periods": 2.0,
            "ladder": [8, 12, 18], "gamma": 7.0, "h_factor": 1.3,
            "wall_friction": 1.0}
    case.update(case_overrides)
    metrics = {
        "amplitude_l2_half_space_pct": l2,
        "amplitude_worst_row_half_space_pct": 1.5 * l2,
        "decay_rate_error_pct": -0.1 * l2,
        "amplitude_l2_finite_depth_pct": 1.02 * l2,
        "delta_ratio": 0.999,
        "first_row_amplitude_over_U0": 0.939,
        "first_row_amplitude_analytic_over_U0": 0.9394,
        "n_rows_stokes_layer": res,
    }
    return {"mode": "probe",
            "provenance": {"case": case, "gates": {"G1_amplitude_l2_pct": 5.0}},
            "runs": [{"resolution_delta_over_dx": res, "metrics": metrics,
                      "gates": {"G1": True, "G2": True, "G3": True,
                                "G4": True, "G5": True},
                      "wall_clock_s": 1.0 * res, "steps": 10 * res,
                      "dt": 6.876e-4, "rows": [], "label": f"probe_dx{res}"}]}


def _make_shards(root: Path, rungs, **case_overrides) -> list[str]:
    dirs = []
    for res, l2 in rungs:
        d = root / f"dx{res}"
        d.mkdir(parents=True, exist_ok=True)
        (d / "results_probe.json").write_text(
            json.dumps(_probe_payload(res, l2, **case_overrides)))
        dirs.append(str(d))
    return dirs


def _run(dirs, out, *extra):
    return asm.main(["--shard", *dirs, "--out", str(out),
                     "--poll", "0.05", *extra])


# --------------------------------------------------------------------------- #
# Publishing path
# --------------------------------------------------------------------------- #

def test_complete_ladder_is_assembled_graded_and_hashed(tmp_path):
    rungs = [(8, 0.0674), (12, 0.0182), (18, 0.0102)]
    dirs = _make_shards(tmp_path / "shards", rungs)
    out = tmp_path / "out"
    assert _run(dirs, out, "--timeout", "5") == 0

    ladder = json.loads((out / "results_ladder.json").read_text())
    conv = ladder["convergence"]
    assert ladder["mode"] == "ladder"
    assert [r["resolution_delta_over_dx"] for r in ladder["runs"]] == [8, 12, 18]
    assert conv["resolutions"] == [8, 12, 18]
    assert conv["complete"] is True and conv["G6_not_growing"] is True
    assert conv["all_gates"] is True
    # observed order of the synthetic L2 sequence: log(0.0674/0.0182)/log(1.5)
    assert conv["observed_order_l2_half_space"] == pytest.approx(
        [math.log(0.0674 / 0.0182) / math.log(1.5),
         math.log(0.0182 / 0.0102) / math.log(1.5)], rel=1e-3)

    # the machine-readable gate table and the hash manifest are part of the deal
    bench = json.loads((out / "benchmark.json").read_text())
    assert bench["primary_metric"] == "amplitude_l2_half_space_pct"
    assert bench["primary_value_pct"] == pytest.approx(0.0102)
    manifest = json.loads((out / "MANIFEST.json").read_text())
    import hashlib
    for name, info in manifest["artifacts"].items():
        digest = hashlib.sha256((out / name).read_bytes()).hexdigest()
        assert digest == info["sha256"], name
    # provenance of the assembly: which shard file contributed which rung
    sources = ladder["assembly"]["sources"]
    assert [s["resolution"] for s in sources] == [8, 12, 18]
    assert all(s["bytes"] > 0 and len(s["sha256"]) == 64 for s in sources)
    assert ladder["assembly"]["rungs_required"] == [8, 12, 18]
    # the case fingerprint travels with the artefact only if it agreed
    assert ladder["provenance"]["case"]["c0"] == 4.0
    assert ladder["provenance"]["case"]["periods"] == 6.0


def test_late_shard_is_waited_for_not_skipped(tmp_path):
    """The regression this script exists for: 2 s guard around a 5 s sleep."""
    rungs = [(8, 0.0674), (12, 0.0182)]
    dirs = _make_shards(tmp_path / "shards", rungs)
    late = tmp_path / "shards" / "dx18"
    late.mkdir(parents=True)

    def drop_late_shard():
        time.sleep(1.0)
        (late / "results_probe.json").write_text(
            json.dumps(_probe_payload(18, 0.0102)))

    thread = threading.Thread(target=drop_late_shard)
    thread.start()
    try:
        assert _run(dirs + [str(late)], tmp_path / "out", "--timeout", "60") == 0
    finally:
        thread.join()
    conv = json.loads((tmp_path / "out" / "results_ladder.json").read_text())["convergence"]
    assert conv["resolutions"] == [8, 12, 18]
    assert conv["complete"] is True


def test_partially_written_shard_is_not_read(tmp_path):
    """A shard caught mid-write must not be parsed as a one-rung ladder."""
    dirs = _make_shards(tmp_path / "shards", [(8, 0.0674), (12, 0.0182)])
    broken = tmp_path / "shards" / "dx18"
    broken.mkdir(parents=True)
    target = broken / "results_probe.json"
    target.write_text('{"mode": "probe", "runs": [{"resolution_delta_over_dx"')

    def finish_write():
        time.sleep(0.6)
        target.write_text(json.dumps(_probe_payload(18, 0.0102)))

    thread = threading.Thread(target=finish_write)
    thread.start()
    try:
        assert _run(dirs + [str(broken)], tmp_path / "out", "--timeout", "30") == 0
    finally:
        thread.join()
    assert not (tmp_path / "out" / "ladder_assembly_refused.json").exists()


# --------------------------------------------------------------------------- #
# Refusal path: nothing published, previous artefacts untouched
# --------------------------------------------------------------------------- #

def test_missing_rung_is_refused_and_publishes_nothing(tmp_path):
    dirs = _make_shards(tmp_path / "shards", [(8, 0.0674), (12, 0.0182)])
    out = tmp_path / "out"
    assert _run(dirs, out, "--timeout", "0.3") == 2
    assert not (out / "results_ladder.json").exists()
    assert not (out / "benchmark.json").exists()
    refusal = json.loads((out / "ladder_assembly_refused.json").read_text())
    assert refusal["refused"] is True and refusal["published"] is False
    assert "[18]" in refusal["reason"]


def test_refusal_does_not_overwrite_a_published_ladder(tmp_path):
    out = tmp_path / "out"
    out.mkdir()
    previous = out / "results_ladder.json"
    previous.write_text('{"mode": "ladder", "sentinel": true}')
    dirs = _make_shards(tmp_path / "shards", [(8, 0.0674), (12, 0.0182)])
    assert _run(dirs, out, "--timeout", "0.3") == 2
    assert json.loads(previous.read_text()) == {"mode": "ladder", "sentinel": True}


def test_disagreeing_case_configuration_is_refused(tmp_path):
    root = tmp_path / "shards"
    dirs = _make_shards(root, [(8, 0.0674), (12, 0.0182)])
    # a third rung run at a different sound speed is a different case, not a rung
    (root / "dx18").mkdir()
    (root / "dx18" / "results_probe.json").write_text(
        json.dumps(_probe_payload(18, 0.0102, c0=10.0)))
    out = tmp_path / "out"
    assert _run(dirs + [str(root / "dx18")], out, "--timeout", "5") == 2
    refusal = json.loads((out / "ladder_assembly_refused.json").read_text())
    assert "c0" in refusal["reason"]
    assert not (out / "results_ladder.json").exists()


def test_duplicate_rung_is_refused(tmp_path):
    a = _make_shards(tmp_path / "a", [(8, 0.0674), (12, 0.0182)])
    b = _make_shards(tmp_path / "b", [(18, 0.0102)])
    c = _make_shards(tmp_path / "c", [(18, 0.0101)])     # same rung, twice
    out = tmp_path / "out"
    assert _run(a + b + c, out, "--timeout", "5") == 2
    refusal = json.loads((out / "ladder_assembly_refused.json").read_text())
    assert "same rung" in refusal["reason"]
    assert not (out / "results_ladder.json").exists()


def test_single_rung_ladder_is_refused(tmp_path):
    dirs = _make_shards(tmp_path / "shards", [(8, 0.0674)])
    out = tmp_path / "out"
    assert _run(dirs, out, "--timeout", "1", "--expected-rungs", "8") == 2
    refusal = json.loads((out / "ladder_assembly_refused.json").read_text())
    assert "cannot test convergence" in refusal["reason"]
    assert not (out / "results_ladder.json").exists()


def test_reproduction_check_against_a_published_ladder(tmp_path):
    """--compare records got-vs-published per rung inside the artefact."""
    out = tmp_path / "out"
    out.mkdir()
    published = out / "results_ladder.json"
    reference = {"runs": [{"resolution_delta_over_dx": res,
                           "metrics": {"amplitude_l2_half_space_pct": l2},
                           "steps": 10 * res, "dt": 6.876e-4}
                          for res, l2 in [(8, 0.0670), (12, 0.0182)]],
                 "convergence": {}}
    published.write_text(json.dumps(reference))
    dirs = _make_shards(tmp_path / "shards", [(8, 0.0674), (12, 0.0182)])
    assert _run(dirs, out, "--timeout", "5", "--expected-rungs", "8", "12",
                "--compare", str(published)) == 0
    check = json.loads((out / "results_ladder.json")
                       .read_text())["assembly"]["reproduction_check"]
    row8 = check["per_rung"][0]
    assert row8["resolution"] == 8
    assert row8["reference_l2_pct"] == pytest.approx(0.0670)
    assert row8["delta_pp"] == pytest.approx(0.0004, abs=1e-12)
    assert row8["steps_match"] is True


# --------------------------------------------------------------------------- #
# The convergence code itself (shared with run.py --ladder)
# --------------------------------------------------------------------------- #

def test_ladder_convergence_flags_a_one_rung_ladder():
    from benchmarks.stokes_second import run as bench

    runs = [_probe_payload(8, 0.0674)["runs"][0]]
    conv = bench.ladder_convergence(runs, [8, 12, 18])
    assert conv["complete"] is False
    assert conv["G6_not_growing"] is False
    assert conv["all_gates"] is False
    assert "cannot be claimed" in conv["G6_reason"]


def test_ladder_convergence_fails_a_growing_error():
    from benchmarks.stokes_second import run as bench

    runs = [_probe_payload(res, l2)["runs"][0]
            for res, l2 in [(8, 0.10), (12, 0.50), (18, 0.90)]]
    conv = bench.ladder_convergence(runs, [8, 12, 18])
    assert conv["complete"] is True
    assert conv["G6_not_growing"] is False       # <= coarsest + 0.5 pp required
    assert conv["all_gates"] is False


def test_ladder_convergence_fails_when_a_rung_gate_fails():
    from benchmarks.stokes_second import run as bench

    runs = [_probe_payload(res, l2)["runs"][0]
            for res, l2 in [(8, 0.0674), (12, 0.0182), (18, 0.0102)]]
    runs[1]["gates"]["G2"] = False
    conv = bench.ladder_convergence(runs, [8, 12, 18])
    assert conv["all_gates"] is False