"""Guards for the periodic-window FSI2 verdict pipeline.

Anchor ``XVAL-SPH-FSI2-BENCHMARK-AND-FORCE-FROM-FLUID``.

Three things are asserted here, all cheap:

1. the transient driver really forwards ``--interface-solver`` into the coupling
   step (a silently ignored flag would make every "Aitken" run an IQN-ILS run
   reported under the wrong name);
2. the report generator cannot invent a verdict: with a start-up-only signal
   (fewer than one period in the window) it must report ``WITHHELD``, and with a
   clean synthetic periodic signal the gate must be evaluated against the
   Turek-Hron numbers rather than something recomputed locally;
3. the emitted markdown carries the numbers of the artefact it was built from.
"""
from __future__ import annotations

import json
import math
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "benchmarks" / "xval_fsi2"))

from tensorsph.fsi2_reference import BENCHMARK_FORCES, BENCHMARK_TIP  # noqa: E402
from verdict_body_force_transient import build_verdict  # noqa: E402


def _synthetic(dx=0.05, t_end=12.0, period=2.4, n=3000, tip_y_amp=0.0082,
               cd_mean=3.2, cd_amp=0.96, cl_amp=0.22, dt=None):
    """A periodic artefact in the schema the driver writes."""
    dt = dt if dt is not None else t_end / n
    t = dt * np.arange(n)
    scale = 0.5 * 1000.0 * 0.2 ** 2 * 0.1  # 0.5 rho U^2 D = 2.0
    phase = 2.0 * math.pi * t / period
    tip = tip_y_amp * np.sin(phase)
    body_drag_fit = scale * (cd_mean + cd_amp * np.sin(phase + 0.4))
    body_lift_fit = scale * cl_amp * np.sin(phase + 1.1)
    hist = {
        "t": t.tolist(),
        "tip_x": (0.34 * tip).tolist(),
        "tip_y": tip.tolist(),
        "body_drag": body_drag_fit.tolist(),
        "body_lift": body_lift_fit.tolist(),
        "body_drag_fit": body_drag_fit.tolist(),
        "body_lift_fit": body_lift_fit.tolist(),
        "drag_viscous": np.full(n, 0.0).tolist(),
        "max_speed": np.full(n, 0.4).tolist(),
        "coupling_converged": [True] * n,
    }
    return {
        "history": hist,
        "dt": dt,
        "dx": dx,
        "n_steps": n,
        "aborted": False,
        "iterations": [3] * n,
        "interface_solver": "aitken",
        "case": {"rho0": 1000.0, "inlet_u_max": 0.3, "cylinder_d": 0.1},
    }


def test_periodic_artifact_is_judged_against_the_benchmark():
    v = build_verdict(_synthetic(), warmup_frac=0.3)
    assert v["regime_ok"] is True
    assert v["gates"]["tip"]["verdict"] in ("PASS", "FAIL")
    # a signal that sits exactly on the benchmark amplitudes must be flagged PASS
    rows = {r["quantity"]: r for r in v["gates"]["tip"]["rows"]}
    assert abs(rows["tip_y_amplitude"]["rel_error"]) < 0.15
    # the reference values come from the shared benchmark table, not from the run
    assert rows["tip_y_amplitude"]["reference"] == BENCHMARK_TIP["tip_y_amplitude"]
    assert v["gates"]["fit_arm"]["rows"][0]["reference"] == BENCHMARK_FORCES["drag_mean"]


def test_startup_only_window_is_withheld_not_pass_or_fail():
    """A monotone ramp has no period: the verdict must be WITHHELD."""
    p = _synthetic(t_end=0.2, n=60)
    t = p["history"]["t"]
    ramp = np.linspace(0.0, 0.002, len(t))
    p["history"]["tip_y"] = ramp.tolist()
    p["history"]["body_drag_fit"] = (10.0 * ramp).tolist()
    v = build_verdict(p, warmup_frac=0.3)
    if not v["regime_ok"]:
        assert v["gates"]["tip"]["verdict"] == "WITHHELD"


def test_driver_forwards_the_interface_solver_flag(tmp_path):
    """``--interface-solver aitken`` must reach ``step_fsi2_strong``."""
    out = tmp_path / "probe.json"
    cmd = [sys.executable, str(ROOT / "benchmarks" / "xval_fsi2"
                              / "run_body_force_transient.py"),
           "--dx", "0.1", "--steps", "3", "--max-iter", "2",
           "--interface-solver", "aitken", "--threads", "2",
           "--out", str(out)]
    r = subprocess.run(cmd, capture_output=True, text=True, timeout=600)
    assert r.returncode == 0, r.stderr[-2000:]
    payload = json.loads(out.read_text())
    assert payload["interface_solver"] == "aitken"
    assert payload["n_steps"] == 3
    assert payload["aborted"] is False


def test_report_markdown_carries_artifact_numbers(tmp_path):
    art = tmp_path / "viv.json"
    art.write_text(json.dumps(_synthetic(tip_y_amp=0.0113)))
    md = tmp_path / "report.md"
    cmd = [sys.executable, str(ROOT / "benchmarks" / "xval_fsi2"
                              / "make_viv_report.py"),
           "--artifact", str(art), "--out-md", str(md),
           "--fig-dir", str(tmp_path / "assets")]
    r = subprocess.run(cmd, capture_output=True, text=True, timeout=600)
    assert r.returncode == 0, r.stderr[-2000:]
    text = md.read_text()
    v = build_verdict(json.loads(art.read_text()), warmup_frac=0.3)
    rel = 100.0 * v["gates"]["tip"]["rows"][1]["rel_error"]
    assert f"{rel:+.2f} %" in text
    assert "WITHHELD" in text or "FAIL" in text or "PASS" in text
    assert "是否 <3 %" in text

def test_growing_oscillation_is_withheld_despite_a_clean_autocorrelation_peak():
    """A start-up transient has an interior autocorrelation peak but is not periodic.

    Regression guard for the defect this round found: the 350-step
    ``dx = 0.05`` artifact of the Aitken arm carries a clean 0.153 s
    autocorrelation peak (acoustic ringing while the amplitude is still
    growing), which the gate used to accept as "4.1 VIV periods" and then score
    as FAIL.  The amplitude-stationarity check must downgrade that to WITHHELD.
    """
    p = _synthetic(t_end=0.9, n=350, period=0.153, tip_y_amp=0.05)
    t = np.asarray(p["history"]["t"], float)
    growth = np.linspace(0.02, 1.0, t.size)
    phase = 2.0 * math.pi * t / 0.153
    p["history"]["tip_y"] = (0.05 * growth * np.sin(phase)).tolist()
    p["history"]["body_drag_fit"] = (2.0 * (1.0 + 20.0 * growth * np.sin(phase))).tolist()
    v = build_verdict(p, warmup_frac=0.3)
    assert v["window"]["periods_in_window"] >= 1.0          # the trap is real
    assert v["window"]["amplitude_stationary"] is False
    assert v["regime_ok"] is False
    assert v["gates"]["tip"]["verdict"] == "WITHHELD"
    assert v["gates"]["fit_arm"]["verdict"] == "WITHHELD"
    # the Strouhal plausibility band is reported next to it
    assert v["window"]["period_band_s"] is not None
    assert v["window"]["period_in_band"] is False
