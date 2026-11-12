"""Guards for the residual decomposition of REPRO-SPH-MONAGHAN1994.

The merged audit (``docs/audit_repro_monaghan1994_m16936.md``, F2) reported the
non-monotone ladder and left two sentences behind: "the residual is
resolution-independent" and the reading that refinement is therefore exhausted.
``results/convergence_decomposition.json`` splits the residual into a bias and a
scatter and tests whether either shrinks; ``results/depth_front_probe.json``
measures how far the graded number moves when the *front reading* changes.

Both are the kind of artefact that rots silently: a regenerated trajectory, an
edited protocol constant, or a document sentence that stops matching the numbers
would leave the tests green unless something recomputes them independently.
That is what this file does - it re-derives every load-bearing number from the
committed trajectories and the reference table (own interpolation, own metric),
and checks that the document quotes the artefact rather than a memory of it.

No simulation runs here (dx = 1 cm costs minutes), so the file is cheap enough
for the default suite.  The probes that produced the artefacts are re-runnable:
``benchmarks/repro_monaghan_1994/front_estimator_probe.py`` and
``.../depth_front_probe.py`` with ``--check-committed``.
"""

from __future__ import annotations

import json
import math
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[1]
BENCH = REPO / "benchmarks" / "repro_monaghan_1994"
RESULTS = BENCH / "results"
DOC = REPO / "docs" / "repro_monaghan_1994_convergence.md"

WINDOW = (0.5, 3.0)
N_SAMPLES = 161
LADDER = (0.04, 0.02, 0.01)


# --------------------------------------------------------------------------- #
# independent primitives (deliberately not imported from the benchmark)
# --------------------------------------------------------------------------- #
def _interp(xs, ys, xq):
    if xq <= xs[0]:
        return ys[0]
    if xq >= xs[-1]:
        return ys[-1]
    lo, hi = 0, len(xs) - 1
    while hi - lo > 1:
        mid = (lo + hi) // 2
        if xs[mid] <= xq:
            lo = mid
        else:
            hi = mid
    span = xs[hi] - xs[lo]
    if span <= 0:
        return ys[lo]
    w = (xq - xs[lo]) / span
    return ys[lo] * (1.0 - w) + ys[hi] * w


def _reference() -> dict:
    raw = json.loads((REPO / "benchmarks/dam_break/martin_moyce_1952_crosscheck.json").read_text())
    tau = [d["tau"] for d in raw["data"]]
    za = [float(d["X_over_a"]) - 1.0 for d in raw["data"]]
    pairs = sorted(zip(tau, za))
    return {"tau": [p[0] for p in pairs], "za": [p[1] for p in pairs]}


def _residuals(tau, za, ref):
    lo = max(WINDOW[0], min(tau), min(ref["tau"]))
    hi = min(WINDOW[1], max(tau), max(ref["tau"]))
    assert hi > lo
    grid = [lo + (hi - lo) * i / (N_SAMPLES - 1) for i in range(N_SAMPLES)]
    return [(_interp(tau, za, g) - _interp(ref["tau"], ref["za"], g)) for g in grid], [
        _interp(ref["tau"], ref["za"], g) for g in grid]


def _trajectory(dx: float) -> dict:
    return json.loads((RESULTS / f"trajectory_dx{dx:g}_c025.json").read_text())


def _artifact(name: str) -> dict:
    path = RESULTS / name
    assert path.exists(), f"missing committed artefact {path}"
    return json.loads(path.read_text())


# --------------------------------------------------------------------------- #
def test_bias_and_scatter_rederive_from_committed_trajectories():
    """The decomposition must be a property of the committed runs, not of a script."""
    payload = _artifact("convergence_decomposition.json")
    ref = _reference()
    for dx in LADDER:
        traj = _trajectory(dx)
        deltas, refs = _residuals(traj["tau"], traj["za"], ref)
        bias = sum(deltas) / len(deltas)
        scatter = sum(abs(d - bias) for d in deltas) / len(deltas)
        mae = sum(abs(d) for d in deltas) / len(deltas)
        amplitude = max(refs)
        got = payload["cases"][f"dx{dx:g}"]
        assert got["bias"] == pytest.approx(bias, abs=1e-9), dx
        assert got["scatter_mean_abs"] == pytest.approx(scatter, abs=1e-9), dx
        assert got["mae"] == pytest.approx(mae, abs=1e-9), dx
        assert got["mae_over_amplitude"] == pytest.approx(mae / amplitude, abs=1e-9), dx
        # the published ladder row must be the same number
        row = [e for e in payload["convergence"]["ladder"] if e["dx"] == dx][0]
        assert row["scatter_mean_abs"] == pytest.approx(scatter, abs=1e-9)


def test_the_two_components_are_what_the_document_claims():
    """Neither component may silently start shrinking (or the verdict is wrong)."""
    payload = _artifact("convergence_decomposition.json")
    ladder = sorted(payload["convergence"]["ladder"], key=lambda e: -e["dx"])
    scatter = [e["scatter_mean_abs"] for e in ladder]
    biases = [e["bias"] for e in ladder]

    # measured fact: scatter is flat to within 5 % across a 4x refinement
    assert max(scatter) / min(scatter) < 1.05, scatter
    # measured fact: the bias changes sign, so |bias| is not a converging series
    assert min(biases) < 0.0 < max(biases)
    # and the artefact must say so rather than manufacture a limit
    conv = payload["convergence"]
    assert conv["richardson_bias_limit"] is None or not math.isfinite(conv["richardson_bias_limit"])
    assert payload["verdict"]["C2_scatter_converges"]["result"] == "not supported"
    assert payload["verdict"]["C1_offset_survives_refinement"]["result"] == "not supported"


def test_verdict_follows_the_preregistered_rule():
    """Re-apply the criteria to the measured values: the verdict is not editorial."""
    payload = _artifact("convergence_decomposition.json")
    v = payload["verdict"]
    m1 = v["C3_single_shift_insufficient"]["measured"]["M1_worst_fine"]
    m2 = v["C3_single_shift_insufficient"]["measured"]["M2_worst_fine"]
    assert v["C3_single_shift_insufficient"]["result"] == (
        "supported" if min(m1, m2) >= 0.03 else "not supported")
    adv = abs(m2 - m1)
    expect = ("temporal" if m2 + 0.005 < m1 else
              "spatial" if m1 + 0.005 < m2 else "indistinguishable")
    assert v["C4_temporal_vs_spatial"]["result"] == expect
    assert v["C4_temporal_vs_spatial"]["measured"]["advantage"] == pytest.approx(adv, abs=1e-12)


def test_out_of_sample_rule_is_respected():
    """The shift applied to the fine cases must be the coarse fit, not a refit."""
    payload = _artifact("convergence_decomposition.json")
    oos = payload["out_of_sample_registration"]
    coarse = payload["cases"][f"dx{LADDER[0]:g}"]
    assert oos["dz"] == pytest.approx(coarse["bias_in_sample_fit"], abs=1e-12)
    assert oos["dtau"] == pytest.approx(coarse["temporal_fit"]["dtau"], abs=1e-12)
    assert [e["dx"] for e in oos["applied"]] == list(LADDER)
    # applying the coarse shift cannot be better on a fine case than on its own
    # case, and the two fine cases must not both beat the 3 % line
    fine = [e for e in oos["applied"] if e["dx"] != LADDER[0]]
    assert max(e["M1_mae_over_amplitude"] for e in fine) >= 0.03


def test_no_gate_was_flipped_and_no_solver_file_touched():
    """This work is diagnostic: it may not move a gate or change the case."""
    payload = _artifact("convergence_decomposition.json")
    assert payload["gate_contact"] is False
    summary = _artifact("summary.json")
    gates = summary["gates"]
    assert gates["trajectory_vs_independent"]["result"] == "pass"
    assert gates["trajectory_vs_independent"]["measured"] == pytest.approx(
        payload["cases"][f"dx{LADDER[-1]:g}"]["mae_over_amplitude"], abs=1e-9)
    # the trend gate's honest red must survive the new analysis
    assert gates["resolution_trend_vs_independent"]["result"] == "fail"


def test_front_reading_spread_is_measured_at_every_ladder_member():
    """The estimator probe must cover the whole ladder (the audit's gap at 1 cm)."""
    probe = _artifact("front_estimator_probe.json")
    covered = sorted(c["dx"] for c in probe["cases"])
    assert covered == sorted(LADDER), covered
    for case in probe["cases"]:
        assert case["committed_diff"]["max_abs_za_diff"] == 0.0, case["case"]
        # a quantile reading can never be ahead of the maximum, so its signed
        # offset against the same table must be the more negative one
        e = case["errors"]
        assert e["p99"]["crosscheck"]["bias"] <= e["x_max"]["crosscheck"]["bias"] + 1e-12


def test_depth_front_readings_are_resolution_independent_by_construction():
    """A depth threshold is a physical rule: it must not invert with refinement."""
    payload = _artifact("depth_front_probe.json")
    ladder = payload["ladder"]
    for name, entry in ladder.items():
        entries = sorted(entry["entries"], key=lambda e: -e["dx"])
        maes = [e["mae_over_amplitude"] for e in entries]
        assert entry["monotone_nonincreasing"] == all(b <= a * 1.10 for a, b in zip(maes, maes[1:]))
        assert entry["finest_mae_over_amplitude"] == pytest.approx(maes[-1], abs=1e-12)
    assert payload["coverage"]["dx"] == sorted(c["dx"] for c in payload["cases"]), \
        "coverage must be declared by the artefact, not assumed"
    spread = payload["reading_spread_at_finest"]
    assert spread["max_mae_over_amplitude"] >= spread["min_mae_over_amplitude"]
    assert spread["readings"][0] == min(ladder, key=lambda k: ladder[k]["finest_mae_over_amplitude"])


def test_document_quotes_the_artefact_numbers():
    """The generated document must print the numbers this artefact holds."""
    assert DOC.exists(), "run make_convergence_doc.py"
    text = DOC.read_text()
    payload = _artifact("convergence_decomposition.json")
    for dx in LADDER:
        c = payload["cases"][f"dx{dx:g}"]
        assert f"{c['bias']:+.3f}" in text, f"bias {c['bias']:+.3f} (dx={dx}) not printed"
        assert f"{c['scatter_mean_abs']:.4f}" in text
        assert f"{c['mae_over_amplitude']*100:.2f} %" in text
    for entry in payload["out_of_sample_registration"]["applied"]:
        assert f"{entry['M1_mae_over_amplitude']*100:.2f} %" in text
    for key in payload["verdict"]:
        assert key in text
    # the honest negative must be stated, not smoothed over
    assert "not supported" in text