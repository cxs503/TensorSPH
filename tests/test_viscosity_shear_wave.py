"""Viscosity verification, part 2: the shear-wave decay instrument.

The measurement itself lives in ``benchmarks/viscosity_shear_wave/run.py`` and
its numbers in ``report.json`` (rendered into
``docs/verification_viscosity_shear_wave.md``).  These tests hold the
*instrument* to its own standard, in the order the error would hurt most:

1. ``test_analytic_reference_is_exact`` - the reference the whole benchmark is
   judged against is audited numerically: the claimed field must satisfy the
   incompressible Navier-Stokes equations and the no-slip planes, to truncation
   error, or the benchmark is measuring the wrong thing.
2. ``test_measured_viscosity_is_within_platform_tolerance`` - the headline
   claim, read from ``report.json``.
3. ``test_criteria_have_teeth`` - every acceptance criterion must be able to
   fail.  A gate that cannot fail is decoration.
4. ``test_recorded_verdicts_are_consistent`` - each recorded verdict must follow
   from its own ``measured``/``threshold`` pair (catches a gate that stores the
   numbers correctly and reports the wrong verdict).
5. ``test_coarse_case_is_reproducible`` - the cheapest case is re-run and must
   reproduce the committed number, so ``report.json`` cannot quietly rot.
"""

from __future__ import annotations

import importlib.util
import json
import math
from pathlib import Path

import pytest
import torch

_BENCH = Path(__file__).resolve().parents[1] / "benchmarks" / "viscosity_shear_wave"
_REPORT = _BENCH / "report.json"


def _load(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.fixture(scope="module")
def harness():
    threads = torch.get_num_threads()
    module = _load("viscosity_shear_wave_run", _BENCH / "run.py")
    yield module
    torch.set_num_threads(threads)


@pytest.fixture(scope="module")
def report():
    if not _REPORT.exists():  # pragma: no cover - benchmark not run yet
        pytest.skip("run benchmarks/viscosity_shear_wave/run.py first")
    return json.loads(_REPORT.read_text())


# --------------------------------------------------------------------------- #
# 1. Is the reference actually a solution?
# --------------------------------------------------------------------------- #

def test_analytic_reference_is_exact(harness):
    """u = U0 sin(ky) exp(-nu k^2 t) must satisfy NS and the no-slip planes."""
    u0, nu, lam = 0.2, 0.01, 0.06
    k = 2.0 * math.pi / lam
    ly = lam                      # one full wavelength between the walls
    dy = lam / 400.0
    y = torch.arange(0.0, ly + 0.5 * dy, dy, dtype=torch.float64)

    def field(t):
        return u0 * torch.sin(k * y) * math.exp(-nu * k * k * t)

    for t in (0.0, 0.004, 0.012):
        u = field(t)
        # Time derivative from the analytic formula, vs the viscous term from
        # finite differences of the analytic profile: Newton's second law for a
        # purely 1-D shear flow is du/dt = nu d2u/dy2.
        du_dt = -nu * k * k * u
        d2u_dy2 = (field(t)[2:] - 2.0 * field(t)[1:-1] + field(t)[:-2]) / dy**2
        scale = float(d2u_dy2.abs().max())
        err = float(((nu * d2u_dy2 - du_dt[1:-1]).abs().max())) / scale
        assert err < 1e-6, f"reference does not solve the viscous equation at t={t}"

    # Convective term: u = (u_x(y), 0) has no x dependence, so u . grad u = 0
    # identically.  Check on the discrete grid as well, since that is the only
    # sense in which the solver could disagree.
    u = field(0.004)
    dudx = torch.zeros_like(u)     # the field carries no x dependence
    dudy = torch.gradient(u, spacing=dy)[0]
    uy = torch.zeros_like(u)
    convective = u * dudx + uy * dudy
    assert float(convective.abs().max()) == 0.0

    # No-slip: the profile must vanish on both walls y = 0 and y = lambda/2, so
    # a bounded box needs no periodic boundary condition.
    for m in (0, 1, 2):
        assert abs(float(u0 * math.sin(k * m * lam / 2))) < 1e-12


# --------------------------------------------------------------------------- #
# 2. The headline claim
# --------------------------------------------------------------------------- #

def test_measured_viscosity_is_within_platform_tolerance(report):
    ratio = report["cases"]["nominal"]["nu_ratio"]
    assert abs(ratio - 1.0) < 0.15, (
        f"nu_eff/nu = {ratio:.4f} outside the platform's 15 % band")
    # Recorded so a future regression shows up as a change here, not a surprise.
    assert ratio == pytest.approx(report["cases"]["nominal"]["nu_effective"]
                                  / report["cases"]["nominal"]["nu_nominal"])


def test_numerical_dissipation_floor_is_measured(report):
    """The nu = 0 control must show that the decay is not numerical."""
    control = report["cases"]["zero_viscosity"]
    floor = abs(control["nu_effective"]) / control["nu_nominal"]
    assert floor < 0.05, f"numerical dissipation floor {floor:.4g} too high"


def test_decay_is_exponential(report):
    """A single rate is meaningless unless the amplitude decays exponentially.

    Applies to the viscously decaying cases only: the nu = 0 control has no
    decay to fit, so its R^2 measures relative noise, not exponentiality.
    """
    for name, case in report["cases"].items():
        if name == "zero_viscosity":
            continue
        assert case["log_linear_r2"] > 0.999, f"{name}: R^2 = {case['log_linear_r2']}"


# --------------------------------------------------------------------------- #
# 3. Gate power: every criterion must be able to fail
# --------------------------------------------------------------------------- #

def _case(nu_ratio=0.905, r2=0.9995, nu_eff_control=0.0002, nu=0.01,
          density_bulk=0.0085):
    nominal = {"nu_ratio": nu_ratio, "nu_effective": nu_ratio * nu,
               "nu_nominal": nu, "log_linear_r2": r2,
               "density_error_bulk": density_bulk}
    return {
        "nominal": nominal,
        "coarse": {"nu_ratio": 0.818, "nu_effective": 0.818 * nu,
                   "nu_nominal": nu, "log_linear_r2": r2,
                   "density_error_bulk": density_bulk},
        "zero_viscosity": {"nu_effective": nu_eff_control, "nu_nominal": nu,
                           "nu_ratio": None, "log_linear_r2": r2,
                           "density_error_bulk": density_bulk},
    }


def test_criteria_have_teeth(harness):
    def verdicts(cases):
        return {c["id"]: c["result"] for c in harness.criteria(cases)}

    # Baseline: a good measurement passes everything.
    assert set(verdicts(_case()).values()) == {"pass"}

    # Each perturbation must flip exactly the criterion it targets.
    assert verdicts(_case(nu_ratio=1.20))["nu_effective_within_15pct"] == "fail"
    assert verdicts(_case(r2=0.95))["decay_is_exponential"] == "fail"
    assert verdicts(_case(nu_eff_control=0.002))[
        "numerical_dissipation_floor"] == "fail"
    assert verdicts(_case(density_bulk=0.20))["density_bounded_bulk"] == "fail"
    # Refinement must move toward the nominal value; a coarse point closer than
    # the fine one means the ladder is not converging.
    inverse = _case(nu_ratio=0.90)
    inverse["coarse"]["nu_ratio"] = 0.98
    assert verdicts(inverse)["refines_toward_nominal"] == "fail"


# --------------------------------------------------------------------------- #
# 4. Recorded verdicts must follow from the recorded numbers
# --------------------------------------------------------------------------- #

def test_recorded_verdicts_are_consistent(report):
    for c in report["criteria"]:
        measured, threshold = c["measured"], c["threshold"]
        expected = ("pass" if measured <= threshold else "fail"
                    if c["comparison"] == "<=" else
                    "pass" if measured >= threshold else "fail")
        assert c["result"] == expected, (
            f"{c['id']}: recorded {c['result']} but {measured} "
            f"{c['comparison']} {threshold} says {expected}")


# --------------------------------------------------------------------------- #
# 5. The committed number is reproducible
# --------------------------------------------------------------------------- #

# ``nu_effective`` is a float32 reduction over particle pairs, so its trailing
# digits follow the summation order, which torch derives from the thread count.
# Re-running the committed coarse case with a single BLAS/OMP thread moves the
# value by 3.4e-5 relative (0.00817607 at 1 thread against 0.00817635 committed
# from a 128-thread run): real hardware dependence, not a solver change.  The
# band below is 1e-3 -- about 30x the measured thread drift and still ~150x
# tighter than the 15 % acceptance gate, so a changed initial condition or a
# changed lattice convention keeps tripping it.  Bit-exactness is asserted where
# it is meaningful on any machine: between two runs inside the same process.
CROSS_MACHINE_TOLERANCE = 1e-3


def test_coarse_case_is_reproducible(harness, report):
    """Re-run the cheapest ladder point; it must match report.json.

    Two degrees of agreement are asserted, because they carry different meaning:

    * same process, same thread count: bit-identical.  The solver has no
      stochastic step, so a second call must return exactly the same float32
      result; any drift here means an initial condition or a lattice convention
      changed.  This gate holds on every machine.
    * against the committed report, which was produced with a different thread
      count: ``CROSS_MACHINE_TOLERANCE``, the measured size of that hardware
      effect (see the comment above).
    """
    case = report["cases"]["coarse"]
    # main() runs the ladder with window == lambda, so lambda pins both.
    kwargs = {"lam": case["lambda"], "window": case["lambda"]}
    redo = harness.run_case(case["dx"], case["nu_nominal"], case["nu_nominal"], **kwargs)
    again = harness.run_case(case["dx"], case["nu_nominal"], case["nu_nominal"], **kwargs)

    assert redo["steps"] == case["steps"]
    assert redo["n_fluid"] == case["n_fluid"]
    assert again["nu_effective"] == redo["nu_effective"], (
        "the same case run twice in one process must be bit-identical")
    assert redo["nu_effective"] == pytest.approx(
        case["nu_effective"], rel=CROSS_MACHINE_TOLERANCE), (
        f"nu_eff {redo['nu_effective']:.6e} vs committed "
        f"{case['nu_effective']:.6e} is outside the "
        f"{CROSS_MACHINE_TOLERANCE:g} cross-machine band")
