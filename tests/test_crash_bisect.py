"""Guard tests for the crash-onset bisect of the method-selection cells.

They pin three things that would otherwise silently turn a crashed cluster job
into a "result":

1. ``run_cell.common_argv`` may only substitute the horizon; geometry,
   resolution and δ-SPH arguments stay byte-identical to the protocol and
   ``COMMON`` is not mutated (a mutated module constant would leak the bisect
   horizon into later cells in the same process).
2. ``crash_bisect`` reports a horizon with no record as *not survived*, whatever
   the return code claims, and never grades a stale record left by an earlier
   run in the same directory.
3. A surviving record's row carries the diagnostics it came from, so the table
   can be traced back to a file.
"""

from __future__ import annotations

import importlib.util
import json
import subprocess
import sys
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[1]
HERE = REPO / "benchmarks" / "xval_method_selection"


def _load(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


run_cell = _load("xval_run_cell", HERE / "run_cell.py")
crash_bisect = _load("xval_crash_bisect", HERE / "crash_bisect.py")


def _protocol_value(argv, flag):
    return argv[argv.index(flag) + 1]


def test_common_argv_default_keeps_protocol_horizon():
    argv = run_cell.common_argv()
    assert _protocol_value(argv, "--simulate-time") == "5.5"
    assert argv == list(run_cell.COMMON)


def test_common_argv_override_touches_only_the_horizon():
    argv = run_cell.common_argv(1.6)
    assert _protocol_value(argv, "--simulate-time") == "1.6"
    diff = [i for i, (a, b) in enumerate(zip(argv, run_cell.COMMON)) if a != b]
    assert diff == [run_cell.COMMON.index("--simulate-time") + 1]
    assert len(argv) == len(run_cell.COMMON)


def test_common_argv_does_not_mutate_common():
    before = list(run_cell.COMMON)
    run_cell.common_argv(0.7)
    run_cell.common_argv(4.4)
    assert run_cell.COMMON == before
    assert _protocol_value(before, "--simulate-time") == "5.5"


def test_common_argv_keeps_the_pre_registered_cell_knobs():
    # The bisect must not become a different case: spot-check the knobs that the
    # protocol (docs/xval_method_selection_m17501/protocol.md §4) fixes.
    argv = run_cell.common_argv(2.6)
    assert _protocol_value(argv, "--depth") == "0.25"
    assert _protocol_value(argv, "--height-ratio") == "0.16"
    assert _protocol_value(argv, "--density-diffusion") == "0.1"
    assert _protocol_value(argv, "--sound-speed") == "30"
    # and the cell itself still carries the resolution
    assert run_cell.CELLS[1][1] == ["--depth-cells", "13"]


def test_record_path_is_stable_and_unique_per_horizon():
    outdir = Path("runs/xval/bisect")
    assert crash_bisect.record_path(outdir, 0.6).name == "t0p60.json"
    assert crash_bisect.record_path(outdir, 5.5).name == "t5p50.json"
    names = {crash_bisect.record_path(outdir, t).name for t in (0.6, 1.6, 2.6)}
    assert len(names) == 3


def test_describe_missing_record_is_not_a_result(tmp_path):
    row = crash_bisect.describe(255, tmp_path / "t1p60.json")
    assert row["record"] is False
    assert row["rc"] == 255
    assert "simulated_time" not in row  # no invented numbers


def test_describe_reads_a_surviving_record(tmp_path):
    path = tmp_path / "t2p60.json"
    path.write_text(json.dumps({
        "diagnostics": {"simulated_time": 2.6, "steps_taken": 12480,
                        "escaped": 0, "max_speed": 0.05,
                        "max_density_compression": 3e-3},
        "probe_x": [0.9964, 1.1594],
        "spec": {"celerity": 1.3038},
    }))
    row = crash_bisect.describe(0, path)
    assert row["record"] is True and row["solver_failed"] is False
    assert row["steps_taken"] == 12480
    assert row["gauges"] == 2
    assert row["first_arrival_s"] == pytest.approx(0.9964 / 1.3038, rel=1e-3)


def test_describe_marks_solver_side_failure(tmp_path):
    path = tmp_path / "t5p50.json"
    path.write_text(json.dumps({"failed": True,
                                "error": "FloatingPointError: non-finite"}))
    row = crash_bisect.describe(2, path)
    assert row["record"] is True and row["solver_failed"] is True
    assert "FloatingPointError" in row["error"]
    assert "steps_taken" not in row


def _fake_runner(behaviour):
    """A subprocess.run stand-in: ``behaviour(cmd, path)`` returns a returncode."""
    def runner(cmd, cwd=None, timeout=None, capture_output=True, text=True):
        path = Path(cmd[cmd.index("--out") + 1])
        rc = behaviour(cmd, path)
        return subprocess.CompletedProcess(cmd, rc, stdout="", stderr="")
    return runner


def test_bisect_reports_crash_without_record(tmp_path):
    def behaviour(cmd, path):
        return -6  # SIGABRT: free(): corrupted unsorted chunks

    summary = crash_bisect.bisect(1, [0.6, 5.5], tmp_path, runner=_fake_runner(behaviour))
    assert summary["counts"] == {"horizons": 2, "survived": 0, "died": 2}
    assert summary["longest_survived"] is None
    assert [r["record"] for r in summary["rows"]] == [False, False]


def test_bisect_never_grades_a_stale_record(tmp_path):
    # A record from an earlier run sits in the directory; the new child dies
    # without writing one.  It must not be counted as a survival.
    stale = crash_bisect.record_path(tmp_path, 2.6)
    stale.write_text(json.dumps({"diagnostics": {"steps_taken": 1}}))

    def behaviour(cmd, path):
        return 255

    summary = crash_bisect.bisect(1, [2.6], tmp_path, runner=_fake_runner(behaviour))
    assert summary["counts"]["survived"] == 0
    assert summary["rows"][0]["rc"] == 255


def test_bisect_finds_the_onset_bracket(tmp_path):
    def behaviour(cmd, path):
        horizon = float(cmd[cmd.index("--simulate-time") + 1])
        if horizon > 3.0:
            return -6
        path.write_text(json.dumps({"diagnostics": {"simulated_time": horizon,
                                                    "steps_taken": 1}}))
        return 0

    summary = crash_bisect.bisect(1, [1.6, 2.6, 3.6, 5.5], tmp_path,
                                  runner=_fake_runner(behaviour))
    assert summary["counts"] == {"horizons": 4, "survived": 2, "died": 2}
    assert summary["longest_survived"] == 2.6
    assert summary["first_death"] == 3.6
    assert summary["onset_bracket"] == [2.6, 3.6]


def test_bisect_passes_the_cell_resolution_to_the_child(tmp_path):
    seen = {}

    def behaviour(cmd, path):
        seen["args"] = cmd
        return 255

    crash_bisect.bisect(1, [1.6], tmp_path, runner=_fake_runner(behaviour))
    assert str(HERE / "run_cell.py") == seen["args"][1]
    assert seen["args"][2] == "1"
    assert seen["args"][seen["args"].index("--simulate-time") + 1] == "1.6"