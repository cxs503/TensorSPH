"""CASE-SPH-POISEUILLE-MORRIS: unit + regression tests for the benchmark driver.

Fast checks (each ISPH case here is one or two seconds):
* the reference solution is evaluated, not transcribed: u_max = g H^2/(8 nu) and
  u(H/2) of the reference profile agree to machine precision;
* ``case_from_reynolds`` really lands on the requested ``Re_c = u_max H/nu``;
* a real (short, coarse) ISPH run reproduces the analytic profile to < 5 % at
  H/dx = 8 -- the same gate the benchmark reports, so the test fails if the
  solver regresses;
* the zero-forcing control returns a zero field (the metric cannot manufacture
  agreement);
* the recorded artefacts, when present, satisfy the predeclared thresholds.
"""

from __future__ import annotations

import json
import math
from pathlib import Path

import pytest
import torch

from benchmarks.poiseuille_morris.run import (
    HEIGHT,
    case_from_reynolds,
    make_config,
    profile_metrics,
    reference_profile,
)
from tensorsph.isph_channel import fit_profile, gates, run_case

ARTEFACT = Path("runs/poiseuille_morris/results.json")


def test_reference_solution_is_consistent():
    nu, g, u_max = case_from_reynolds(1.0)
    assert u_max == pytest.approx(g * HEIGHT ** 2 / (8.0 * nu), rel=1e-12)
    assert reference_profile(HEIGHT / 2.0, HEIGHT, g, nu) == pytest.approx(u_max, rel=1e-12)
    # no-slip at both walls
    assert reference_profile(0.0, HEIGHT, g, nu) == 0.0
    assert reference_profile(HEIGHT, HEIGHT, g, nu) == 0.0


def test_case_from_reynolds_lands_on_requested_re():
    for re_c in (0.1, 1.0, 12.26):
        nu, g, _ = case_from_reynolds(re_c)
        assert (g * HEIGHT ** 2 / (8.0 * nu)) * HEIGHT / nu == pytest.approx(re_c, rel=1e-12)


def test_config_run_length_is_in_diffusive_times():
    cfg = make_config(1.0, 12)
    t_diff = HEIGHT ** 2 / (cfg.viscosity * math.pi ** 2)
    assert cfg.t_end / t_diff == pytest.approx(10.0, rel=1e-12)
    assert cfg.t_average == pytest.approx(0.1 * cfg.t_end, rel=1e-12)


def test_coarse_isph_run_meets_profile_gate():
    torch.set_num_threads(1)
    cfg = make_config(1.0, 8)
    cfg.t_end = 0.02                      # short: this is a gate test, not the benchmark
    cfg.t_average = 0.1 * cfg.t_end
    result = run_case(cfg, "cpu")
    metrics = profile_metrics(result["profile"], cfg)
    assert metrics["interior"]["n_rows"] > 3
    # coarse + short run must still land in the same ballpark as the reference
    assert metrics["interior"]["l2_over_umax_pct"] < 20.0


def test_zero_forcing_gives_zero_field():
    torch.set_num_threads(1)
    cfg = make_config(1.0, 8)
    cfg.body_force = 0.0
    cfg.t_end = 0.01
    cfg.t_average = 0.1 * cfg.t_end
    result = run_case(cfg, "cpu")
    assert max(abs(p["u_mean"]) for p in result["profile"]) < 1e-12


def test_artefact_thresholds_when_present():
    if not ARTEFACT.exists():
        pytest.skip("production artefacts not generated in this checkout")
    payload = json.loads(ARTEFACT.read_text())
    assert payload["anchor"] == "CASE-SPH-POISEUILLE-MORRIS"
    isph = payload["cases"]["isph_projection"]
    finest = {}
    for case in isph:
        if case["columns"] != 12 or case["h_over_dx"] != 1.90:
            continue
        cur = finest.get(case["reynolds_c"])
        if cur is None or case["resolution_H_over_dx"] > cur["resolution_H_over_dx"]:
            finest[case["reynolds_c"]] = case
    assert finest, "no canonical-ladder ISPH case recorded"
    for re_c, case in finest.items():
        assert case["metrics"]["interior"]["l2_over_umax_pct"] < 5.0, re_c
        assert abs(case["metrics"]["centre_error_pct"]) < 5.0, re_c
    # the reference block must record the access status of the paper
    assert payload["reference"]["authority_access"]["status"] == "WITHHELD"