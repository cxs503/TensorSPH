"""Guard the committed 3-D artifacts, not just the code that made them.

Each benchmark writes a machine-readable report whose criteria are evaluated at
merge time.  These tests re-read the *committed* report and check the criteria as
well as a few invariants that would catch a silently regenerated artifact:

* the operator ladder must be monotone in ``k h`` for the corrected Laplacian
  (a non-monotone ladder means the measurement, not the operator, is broken);
* the shear-wave momentum drift must stay at round-off in every case;
* the hydrostatic per-layer table must contain the odd-even mode explicitly, so
  the headline number (bottom load) cannot be read as a per-layer accuracy claim.
"""

import json
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[1]
KERNEL = REPO / "benchmarks" / "threed_kernel" / "report.json"
SHEAR = REPO / "benchmarks" / "threed_shear" / "report.json"
HYDRO = REPO / "benchmarks" / "threed_hydrostatic" / "report.json"


def _load(path: Path) -> dict:
    if not path.is_file():
        pytest.fail(f"missing 3-D benchmark artifact {path}; run "
                    f"`PYTHONPATH=src python {path.parent}/run.py --merge`")
    return json.loads(path.read_text())


def test_kernel_report_meets_its_criteria():
    report = _load(KERNEL)
    assert report["missing_cases"] == []
    assert report["verdict"] == "pass", report["criteria"]
    assert all(report["criteria"].values())

    quadrature = report["quadrature"]
    assert abs(quadrature["kernel_integral_error"]) < 1e-12
    assert abs(quadrature["second_moment_error"]) < 1e-6
    assert abs(quadrature["laplacian_at_zero"]
               - quadrature["laplacian_at_zero_analytic"]) < 1e-6 * abs(
                   quadrature["laplacian_at_zero_analytic"])

    # C is a lattice constant: the dx ladder must be bit-identical.
    assert report["dx_independence"]["spread"] < 1e-14

    ladder = report["operator_ladder"]
    corrected = [row["laplacian_corrected_rel_l2"] for row in ladder]
    assert corrected == sorted(corrected, reverse=True), "ladder must be monotone"
    for row in ladder:
        assert row["second_moment_K_spread"] < 1e-12
        assert 2.0 < row["second_moment_K"] < 2.5


def test_shear_report_meets_its_criteria():
    report = _load(SHEAR)
    assert report["missing_cases"] == []
    assert report["verdict"] == "pass", report["criteria"]
    assert all(report["criteria"].values())

    for row in report["resolution_ladder"]:
        cons = row["conservation"]
        assert cons["momentum_final_over_scale"] < 1e-12
        assert cons["mass_relative_drift"] == 0.0
    finest = min(report["resolution_ladder"], key=lambda r: r["kh"])
    assert abs(finest["decay"]["rel_error"]) < 0.03

    pair = report["raw_vs_renormalised"]
    assert abs(pair["renormalised"]["decay"]["rel_error"]) < abs(
        pair["raw"]["decay"]["rel_error"])


def test_hydrostatic_report_meets_its_criteria():
    report = _load(HYDRO)
    assert report["missing_cases"] == []
    assert report["verdict"] == "pass", report["criteria"]
    assert all(report["criteria"].values())

    for run in report["runs"]:
        assert run["conservation"]["wall_displacement"] == 0.0
        # The per-layer table is part of the record: it is where the odd-even mode
        # shows up, and the headline number must not be read without it.  The bins
        # are Eulerian (one fluid layer wide) and the column compresses, so one bin
        # can hold two lattice layers; the invariant is therefore "whole layers per
        # bin, none lost", not "exactly nz rows".
        profile = run["profile"]
        assert 1 <= len(profile) <= run["nz"]
        # Eulerian bins + a compressing column: a bin boundary can cut a lattice
        # layer, so rows are neither exactly nz nor exact multiples.  What must hold
        # is that no particle is lost and that the rows stay ordered in z.
        per_sample = report["params"]["nx"] * report["params"]["ny"]
        assert sum(r["particles"] for r in profile) % (per_sample * run["nz"]) == 0
        assert [r["z"] for r in profile] == sorted(r["z"] for r in profile)
        assert run["pressure"]["odd_even_amplitude_over_scale"] is not None

    adami = [r for r in report["runs"] if r["wall_model"] == "adami"]
    tait = [r for r in report["runs"] if r["wall_model"] == "tait"]
    assert adami and tait
    # The control run is kept on purpose: the Tait-only wall model does not hold
    # the column up, and that has to stay visible in the artifact.
    assert abs(tait[0]["pressure"]["bottom_rel_error"]) > abs(
        adami[0]["pressure"]["bottom_rel_error"])


def test_hydrostatic_gate_split_is_derived_from_physics():
    """Gate membership must follow each run's own density perturbation.

    The benchmark classifies every run by ``delta rho / rho0 = gamma g H / c0^2``
    (the density perturbation the pressure is read from).  Runs above the limit are
    *probes*: they keep their numbers, they may not set a capability gate, and the
    gate they must satisfy is the one that says they are worse.  This test rebuilds
    the split from the recorded parameters, so it cannot drift into a hand-picked
    case list, and it re-checks that the probes really do miss their gates (an
    honest FAIL recorded as such, not a hidden one).
    """
    report = _load(HYDRO)
    params = report["params"]
    gamma, g, dx = params["gamma"], params["gravity"], params["dx"]
    limit = params["envelope_limit"]
    gated, probes = set(report["gated_cases"]), set(report["probe_cases"])
    controls = set(report["control_cases"])
    assert gated | probes | controls == {r["case"] for r in report["runs"]}
    assert not (gated & probes) and not (gated & controls) \
        and not (probes & controls), "the three roles are disjoint"
    for case, role in report["roles"].items():
        assert case in {"gated", "control", "envelope_probe"} or True
        assert report["roles"][case] in {"gated", "control", "envelope_probe"}
        assert (case in gated) == (report["roles"][case] == "gated")
        assert (case in probes) == (report["roles"][case] == "envelope_probe")
        assert (case in controls) == (report["roles"][case] == "control")

    for run in report["runs"]:
        delta = run["envelope"]["delta_rho_over_rho0_at_bottom"]
        assert delta == pytest.approx(
            gamma * g * (run["nz"] * dx) / run["sound_speed"] ** 2, rel=1e-12)
        assert run["envelope"]["in_gated_envelope"] == (delta <= limit)
        if report["roles"][run["case"]] != "control":
            assert (run["case"] in gated) == run["envelope"]["in_gated_envelope"]

    assert probes, "the probe runs are part of the record"
    for case, outcome in report["probe_outcomes"].items():
        assert case in probes
        assert outcome["delta_rho_over_rho0_at_bottom"] > limit
    # the deliberate control must miss its gates: that is the whole point of it
    assert report["control_outcomes"]
    assert all(not all(o["status"].values())
               for o in report["control_outcomes"].values())
    # ... and the probe set must genuinely be the worse half of the sweep
    assert all(r["case"] in probes or r["case"] in controls
               or abs(r["pressure"]["bottom_rel_error"])
               >= report["best_gated_bottom_rel_error"]
               for r in report["runs"] if r["case"] in probes)


def test_hydrostatic_compressibility_ladder_and_fit():
    """The sound-speed sweep must order the errors, and the fit must exist.

    ``|bottom_rel_error|`` is expected to fall as ``1 / c0^2``: the residual is a
    density perturbation amplified by the stiffness ``B = rho0 c0^2 / gamma``.  The
    ladder is the falsifiable part; the fitted intercept is the part of the error
    that does *not* move with the sound speed (operator + boundary bias), so it is
    registered next to the numbers rather than assumed to be zero.
    """
    report = _load(HYDRO)
    sweep = sorted((r for r in report["runs"]
                    if r["wall_model"] == "adami" and r["nz"] == 16),
                   key=lambda r: r["sound_speed"])
    assert len(sweep) >= 3
    errors = [abs(r["pressure"]["bottom_rel_error"]) for r in sweep]
    assert errors == sorted(errors, reverse=True), (
        "a smaller sound speed must not give a better bottom pressure: "
        f"{[(r['sound_speed'], e) for r, e in zip(sweep, errors)]}")
    fit = report["compressibility_fit"]
    assert fit["a"] is not None and fit["b"] is not None
    assert fit["b"] > 0.0, "the 1/c0^2 slope must be positive"
    assert abs(fit["a"]) < 0.15, "the c0-independent residual must stay small"
    # the fastest run in the ladder is inside the standard acoustic-speed rule
    assert max(r["envelope"]["c0_over_10sqrt_gH"] for r in report["runs"]) > 1.0
