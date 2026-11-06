"""Regenerate the CI-pin evidence ledger of the oscillating-droplet benchmark.

Why this file exists
--------------------
``benchmarks/oscillating_droplet/ci_pin_evidence.json`` is the ledger behind the
per-platform pins in ``tests/test_oscillating_droplet.py``
(``RECORDED_MODE2_ERR_PCT``).  It records that the mode-2 *pseudo*-frequency of
the CI case is -31.78 % on loongarch64 and +8.64 % on x86_64, i.e. that two
machines running the same commit disagree by ~40 pp -- which is exactly why no
single-platform number may be read as "the physics passes".

m4795 shipped the ledger described as "written programmatically (no hand-copied
numbers)" but did not commit the program, so the ledger was an un-reproducible
claim: nothing in the repository could regenerate it, and a hand edit would go
unnoticed (the test only cross-checks it against the pin table in the test
module, which a hand edit could change as well).  This script is that program.

Usage
-----
    python benchmarks/oscillating_droplet/ci_pin_evidence.py           # print the ledger
    python benchmarks/oscillating_droplet/ci_pin_evidence.py --write   # rewrite the ledger
    python benchmarks/oscillating_droplet/ci_pin_evidence.py --check   # compare with the ledger

``--check`` exits non-zero when the committed ledger no longer matches what this
machine measures, so drift (a changed model, a changed estimator, a different
thread count) is visible outside pytest as well.

``--write`` refreshes only the blocks this machine can measure: ``case``,
``analytic``, ``measured``, ``stability``, ``thread_determinism`` and *this*
platform's entry of ``recorded_platform_pins_pct``.  Pins of other platforms
were recorded on those machines and are carried over unchanged -- regenerating
them here would be fabrication.

What the ledger is not
----------------------
It is not a validation result.  The trace diverges (``max|v| > c0``), the
frequency is a pseudo-mode of that trace, and the recorded verdict stays
``WITHHELD``; see ``docs/verification_oscillating_droplet.md`` appendix B.
"""
from __future__ import annotations

import argparse
import importlib.util
import json
import math
import platform
import sys
from pathlib import Path

import torch

HERE = Path(__file__).resolve().parent
LEDGER = HERE / "ci_pin_evidence.json"

# The CI case: identical to the ``release_case`` fixture in
# ``tests/test_oscillating_droplet.py``.  Keeping one dict here (and having the
# test read it) is what stops the two from drifting apart.
CI_CASE = {
    "radius": 0.01,
    "dx": 0.001,
    "sigma": 7.2,
    "rho0": 1000.0,
    "sound_speed": 20.0,
    "viscosity": 5e-5,
    "eps": 0.10,
    "periods": 1.8,
}

GATE_PCT = 15.0
STRICT_PCT = 3.0
DETERMINISM_STEPS = 200
DETERMINISM_THREADS = (1, 4, 8, 16)
LAW = "omega^2 = 6 sigma / (rho R^3)  [2-D, m=2]"


def _bench():
    """Import the benchmark runner (``run.py``) as a module."""
    spec = importlib.util.spec_from_file_location("oscillating_run", HERE / "run.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def run_ci_case(bench=None, **overrides) -> dict:
    """Run the CI configuration once and return the runner's report."""
    bench = bench or _bench()
    kw = {**CI_CASE, **overrides}
    return bench.run_case(**kw)


def measured_blocks(rep: dict, periods: float = CI_CASE["periods"]) -> dict:
    """Map a ``run_case`` report onto the ledger's measured blocks.

    Pure function: no simulation, no I/O.  Every number in the ledger comes
    through here, which is what makes "no hand-copied numbers" checkable.
    """
    setup = rep["setup"]
    freq = rep["frequency"]
    diag = rep["diagnostics"]
    sound_speed = setup["sound_speed"]
    return {
        "case": {
            "radius_m": setup["radius_m"],
            "dx_m": setup["dx_m"],
            "eps": setup["initial_eps"],
            "sigma_N_per_m": setup["sigma_N_per_m"],
            "rho0": setup["rho0"],
            "sound_speed_m_per_s": sound_speed,
            "viscosity_m2_per_s": setup["viscosity_m2_per_s"],
            "periods": periods,
            "steps": setup["steps"],
            "dt_s": setup["dt_s"],
            "n_particles": setup["n_particles"],
        },
        "analytic": {
            "omega_2_rad_per_s": rep["reference"]["omega_rad_per_s"],
            "law": LAW,
        },
        "measured": {
            "omega_damped_fit_rad_per_s": freq["fit_second_moment_rad_per_s"],
            "fit_rel_error_pct": freq["fit_second_moment_rel_error_pct"],
            "fit_r2": freq["fit"]["r2"],
            "fit_lambda_per_s": freq["fit"]["lambda_per_s"],
            "omega_zero_crossing_rad_per_s": freq["zero_crossing_rad_per_s"],
            "zero_crossing_rel_error_pct": freq["zero_crossing_rel_error_pct"],
            # KE oscillates at 2*omega; halving makes it comparable with the
            # second-moment estimator (same convention as ``run.py``).
            "omega_kinetic_energy_halved_rad_per_s":
                0.5 * freq["fit_kinetic_energy_rad_per_s"],
            "kinetic_energy_rel_error_pct":
                freq["fit_kinetic_energy_rel_error_pct"],
        },
        "stability": {
            "max_speed_m_per_s": diag["max_speed_m_per_s"],
            "max_speed_over_c0": diag["max_speed_m_per_s"] / sound_speed,
            "mass_relative_drift": diag["mass_relative_drift"],
        },
    }


def thread_determinism_probe(bench=None, steps: int = DETERMINISM_STEPS,
                             threads=DETERMINISM_THREADS) -> dict:
    """Bit-for-bit check of the integrator across intra-op thread counts.

    The trace carries the -31.78 % pseudo-frequency, and a reader is entitled to
    ask whether *that number* is reproducible at all.  The integrator has no RNG
    and no atomic reduction, so it must be: this probe advances the same CI
    configuration ``steps`` steps under several ``torch.set_num_threads``
    values and reports the largest position difference.
    """
    bench = bench or _bench()
    cfg = bench.Config(
        dx=CI_CASE["dx"], h_factor=1.3, rho0=CI_CASE["rho0"],
        sound_speed=CI_CASE["sound_speed"], gamma=7.0, gravity=0.0,
        viscosity=CI_CASE["viscosity"],
        surface_tension_coefficient=CI_CASE["sigma"],
        csf_scheme="consistent", csf_curvature="fit_quadratic",
        density_diffusion=0.0,
    )
    dt = float(bench.stable_dt(cfg))
    h = cfg.h_factor * cfg.dx

    def advance(n_threads: int) -> torch.Tensor:
        torch.set_num_threads(n_threads)
        pos, mass = bench.elliptical_drop(CI_CASE["radius"], CI_CASE["dx"],
                                          CI_CASE["rho0"], CI_CASE["eps"])
        vel = torch.zeros_like(pos)
        for _ in range(steps):
            pairs = bench.cell_linked_pairs(pos, 2.0 * h)
            _, _, acc = bench.rhs(pos, vel, mass, cfg, pairs=pairs, wall=None,
                                  dt=dt)
            vel = vel + dt * acc
            pos = pos + dt * vel
        return pos

    try:
        reference = advance(threads[0])
        worst = 0.0
        for n in threads[1:]:
            worst = max(worst, float((advance(n) - reference).abs().max()))
    finally:
        # ``run.py`` sets 2 at import; leave the process as it found it.
        torch.set_num_threads(2)
    return {
        "threads_tested": list(threads),
        "steps": steps,
        "max_abs_position_diff_m": worst,
        "note": ("bit-for-bit identical; the integrator has no RNG and no "
                 "atomic reduction" if worst == 0.0 else
                 "thread-count dependent: the integrator is not deterministic "
                 "across intra-op threads"),
    }


def verdict_block(rel_error_pct: float) -> dict:
    """Platform verdict of the mode-2 frequency, derived from the measurement."""
    converges = abs(rel_error_pct) < STRICT_PCT
    return {
        "gate_pct": GATE_PCT,
        "strict_pct": STRICT_PCT,
        "passes_gate": abs(rel_error_pct) < GATE_PCT,
        "evidence_level": (
            "REAL (single-platform measurement inside the 3 % standard)"
            if converges else
            "WITHHELD (diverging run; pseudo-frequency)"),
    }


def build_evidence(rep: dict, determinism: dict, pins: dict,
                   arch: str | None = None) -> dict:
    """Assemble the full ledger from a run report, a probe and the pin table."""
    arch = arch or platform.machine()
    blocks = measured_blocks(rep)
    measured = blocks["measured"]
    pins = dict(pins)
    pins[arch] = round(measured["fit_rel_error_pct"], 2)
    return {
        "benchmark": "oscillating_droplet_ci_pin_evidence",
        "platform": arch,
        "generated_by": "benchmarks/oscillating_droplet/ci_pin_evidence.py",
        **blocks,
        "thread_determinism": determinism,
        "recorded_platform_pins_pct": pins,
        "verdict": verdict_block(measured["fit_rel_error_pct"]),
    }


def load_ledger(path: Path = LEDGER) -> dict:
    return json.loads(path.read_text())


# Tolerance for ``--check``: the run is deterministic on a fixed thread count,
# so this only has to absorb last-bit differences in the least-squares solve.
CHECK_TOL = {"rel_error_pct": 0.05, "ratio": 1e-6}


def _flatten(block, prefix=""):
    out = {}
    for key, value in block.items():
        path = f"{prefix}{key}"
        if isinstance(value, dict):
            out.update(_flatten(value, f"{path}."))
        elif isinstance(value, (int, float)) and not isinstance(value, bool):
            out[path] = float(value)
        else:
            out[path] = value
    return out


def compare_blocks(committed: dict, blocks: dict) -> list[str]:
    """Compare the measured blocks of a ledger against fresh ones.

    Split out from :func:`compare` so that a test which already owns a finished
    run report (the pytest module fixture) can verify the ledger without paying
    for a second simulation.
    """
    diffs = []
    for block in ("case", "analytic", "measured", "stability"):
        old, new = _flatten(committed.get(block, {}), f"{block}."), \
            _flatten(blocks[block], f"{block}.")
        for key in sorted(set(old) | set(new)):
            if key not in old or key not in new:
                where = "ledger" if key not in old else "run"
                diffs.append(f"{key}: missing in {where}")
                continue
            a, b = old[key], new[key]
            if isinstance(a, str) or isinstance(b, str):
                if a != b:
                    diffs.append(f"{key}: ledger {a!r} vs run {b!r}")
                continue
            tol = CHECK_TOL["rel_error_pct"] if key.endswith("_pct") else 0.0
            if tol:
                if abs(a - b) > tol:
                    diffs.append(f"{key}: ledger {a:+.6g} vs run {b:+.6g} "
                                 f"(tolerance {tol} pp)")
            elif not math.isclose(a, b, rel_tol=CHECK_TOL["ratio"], abs_tol=0.0):
                diffs.append(f"{key}: ledger {a:.12g} vs run {b:.12g}")
    return diffs


def compare(committed: dict, fresh: dict) -> list[str]:
    """Full ledger check: measured blocks, determinism and this platform's pin."""
    diffs = compare_blocks(committed, fresh)
    det = fresh["thread_determinism"]
    if det["max_abs_position_diff_m"] != 0.0:
        diffs.append("thread_determinism: the integrator is not bit-for-bit "
                     f"deterministic (max|dpos| = {det['max_abs_position_diff_m']:.3e} m)")
    pins_old = committed.get("recorded_platform_pins_pct", {})
    arch = fresh["platform"]
    if arch in pins_old and abs(pins_old[arch] - fresh["recorded_platform_pins_pct"][arch]) > \
            CHECK_TOL["rel_error_pct"]:
        diffs.append(f"recorded_platform_pins_pct.{arch}: ledger {pins_old[arch]:+.2f} "
                     f"vs run {fresh['recorded_platform_pins_pct'][arch]:+.2f}")
    return diffs


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    group = parser.add_mutually_exclusive_group()
    group.add_argument("--write", action="store_true",
                       help="rewrite the committed ledger")
    group.add_argument("--check", action="store_true",
                       help="compare with the committed ledger; nonzero exit on drift")
    parser.add_argument("--skip-determinism", action="store_true",
                        help="skip the thread-count probe (saves ~20 s)")
    parser.add_argument("--ledger", type=Path, default=LEDGER)
    args = parser.parse_args(argv)

    bench = _bench()
    rep = run_ci_case(bench)
    committed = load_ledger(args.ledger) if args.ledger.exists() else {}
    pins = committed.get("recorded_platform_pins_pct", {})
    if args.skip_determinism:
        determinism = committed.get("thread_determinism") or {
            "threads_tested": [], "steps": 0,
            "max_abs_position_diff_m": float("nan"),
            "note": "probe skipped",
        }
    else:
        determinism = thread_determinism_probe(bench)
    fresh = build_evidence(rep, determinism, pins)

    if args.write:
        args.ledger.write_text(json.dumps(fresh, indent=1, sort_keys=False) + "\n")
        print(f"wrote {args.ledger} ({fresh['platform']}, "
              f"fit_rel_error = {fresh['measured']['fit_rel_error_pct']:+.2f} %)")
        return 0

    if args.check:
        diffs = compare(committed, fresh)
        if diffs:
            print(f"DRIFT on {fresh['platform']} vs {args.ledger}:")
            for line in diffs:
                print("  " + line)
            return 1
        print(f"ledger matches this {fresh['platform']} run "
              f"(fit_rel_error = {fresh['measured']['fit_rel_error_pct']:+.2f} %)")
        return 0

    print(json.dumps(fresh, indent=1))
    return 0


if __name__ == "__main__":
    sys.exit(main())
