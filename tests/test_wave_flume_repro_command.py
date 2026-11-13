"""Guard: the published reproduction command must rebuild the published record.

``docs/reports/wave_flume_case.md`` quotes a run command so someone else can
regenerate ``runs/wf_*.json``.  A command that silently omits a physics knob is
worse than no command: it reproduces *a different* case and dresses the mismatch
up as irreproducibility.  Before ``make_report.py`` was fixed the documented
command dropped ``--active-absorption``/``--output-every``/``--sponge-length``,
asked for ``--simulate-time 7.0`` against a 5 s record, and told the analysis to
load a nonexistent ``runs/wf_D32.json``.

These tests bind the generator (:func:`make_report.repro_command`) to the records
it claims to reproduce, so any future drift (a new solver knob that the command
forgets, a changed default) fails CI instead of a reviewer.
"""
from __future__ import annotations

import json
import math
import shlex
import sys
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[1]
RUNS = REPO / "runs"
sys.path.insert(0, str(REPO / "benchmarks" / "wave_flume"))

import make_report  # noqa: E402

#: Physics knobs whose value changes the simulation result and must be emitted.
#: (CLI flag, record-config key)
KNOBS = (
    ("viscosity", "viscosity"),
    ("density-diffusion", "density_diffusion"),
    ("sponge-strength", "sponge_strength"),
    ("active-absorption", "active_absorption"),
    ("sponge-length", "sponge_length"),
    ("xsph", "xsph"),
    ("xsph-settle", "xsph_settle"),
    ("viscosity-settle", "viscosity_settle"),
    ("settle", "settle_time"),
    ("flume-length", "flume_length"),
    ("sound-speed", "sound_speed"),
    ("gravity", "gravity"),
    ("ramp-periods", "ramp_periods"),
    ("dt", "dt_requested"),
)


def _records():
    return sorted(RUNS.glob("wf_*.json"))


def _parse(cmd: str) -> dict:
    """Split a backslash-continued shell command into a ``{flag: value}`` map."""
    tokens = shlex.split(cmd.replace("\\\n", " "))
    out = {}
    i = 0
    while i < len(tokens):
        tok = tokens[i]
        if tok.startswith("--"):
            key = tok[2:]
            if i + 1 < len(tokens) and not tokens[i + 1].startswith("--"):
                out[key] = tokens[i + 1]
                i += 2
            else:
                out[key] = True
                i += 1
        else:
            i += 1
    return out


@pytest.mark.parametrize("path", _records() or [None])
def test_repro_command_emits_every_physics_knob(path):
    """The command must carry the record's own physics settings, not defaults."""
    if path is None:
        pytest.skip("no wf_* record under runs/; nothing to reproduce")
    record = json.loads(path.read_text())
    cfg = record["config"]
    spec = record["spec"]
    args = _parse(make_report.repro_command(cfg, record["diagnostics"], spec, path))

    for flag, key in KNOBS:
        assert flag in args, f"{path.name}: repro command omits --{flag}"
        assert float(args[flag]) == pytest.approx(float(cfg[key]), rel=1e-6), (
            path.name, flag, args[flag], cfg[key])

    assert int(args["depth-cells"]) == int(round(cfg["depth_cells"]))
    assert float(args["depth"]) == pytest.approx(cfg["depth"], rel=1e-9)
    assert float(args["height-ratio"]) == pytest.approx(
        spec["height"] / spec["depth"], rel=1e-3)
    assert float(args["wave-period"]) == pytest.approx(spec["period"], rel=1e-6)
    assert int(args["output-every"]) == int(cfg["output_every"])
    assert "out" in args


@pytest.mark.parametrize("path", _records() or [None])
def test_repro_simulate_time_recovers_the_step_count(path):
    """``--simulate-time`` must reproduce the recorded step count (dt is clamped)."""
    if path is None:
        pytest.skip("no wf_* record under runs/; nothing to reproduce")
    record = json.loads(path.read_text())
    cfg = record["config"]
    args = _parse(make_report.repro_command(cfg, record["diagnostics"], record["spec"], path))
    h = 1.3 * cfg["depth"] / cfg["depth_cells"]
    step = min(cfg["dt_requested"], 0.25 * h / cfg["sound_speed"])
    steps = math.ceil(float(args["simulate-time"]) / step)
    assert abs(steps - record["diagnostics"]["steps_taken"]) <= 2, (
        path.name, steps, record["diagnostics"]["steps_taken"])


def test_default_block_is_derived_from_records_not_hardcoded():
    """The markdown repro block must reference real records only."""
    paths = _records()
    if not paths:
        pytest.skip("no wf_* record under runs/; nothing to reproduce")
    rows = []
    for p in paths:
        rec = json.loads(p.read_text())
        if not rec.get("failed"):
            rows.append({"config": rec["config"], "diagnostics": rec["diagnostics"],
                         "spec": rec["spec"], "record": str(p)})
    block = make_report.default_repro_block(rows or [{"failed": True}])
    assert "wf_D32" not in block, "repro block points at a record that does not exist"
    for knob in ("--active-absorption", "--output-every", "--sponge-length"):
        assert knob in block, f"repro block omits {knob}"
    for p in paths:
        assert f"--record {p}" in block
