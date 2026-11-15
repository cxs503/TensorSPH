"""Freeze / verify the *shippable* text channel of ``CASE-SPH-LAKE-AT-REST``.

Why this tool exists
--------------------
The anchor's acceptance is a machine-precision identity (``u = 0``, flat free
surface, ``max|v|/c0 < 1e-9``).  The evidence for it was kept as

* ``artifacts/lake_at_rest/*.json`` -- ``.gitignore``'d, i.e. absent on every
  fresh checkout, and
* numbers quoted in ``docs/verification_lake_at_rest*.md`` -- prose, i.e. not
  machine-decidable and free to drift from the code.

So on a fresh checkout nothing failed when the shipped numbers stopped matching
the library, and the anchor's decision ("does the still lake stay still?") was
not decidable where it is read.  This tool closes that hole:

* it copies **only the text payload** of both runs -- the x-periodic tank (the
  configuration in which the tolerance is met) and the wall-bounded tank (the
  honest FAIL) -- into ``tests/data/lake_at_rest``, a path that is neither
  ``artifacts/`` nor artifact-class, so it survives push and merge;
* it writes ``report.json`` with the gate table re-graded from the shipped
  numbers (``evaluate_gates``), the anchor tolerance, and which quantities are
  reproducible vs which are rounding-level scatter;
* it writes ``MANIFEST.sha256`` (``sha256  bytes  relpath``, sorted) so the
  shipped numbers are tamper-evident, and ``channel.json`` recording what was
  shipped, what was not, and why;
* ``--verify`` re-runs both configurations and compares: stable quantities by
  relative tolerance, rounding-level quantities by *scale* (both sides below the
  anchor tolerance, ratio inside a generous band) -- because the equilibrium
  residual is a cancellation of double-precision sums and is **not** bit
  reproducible run to run (measured spread ~1.3x, see ``report.json``
  ``reproducibility``).  A verification that demanded equality would either be
  flaky or would have to be relaxed into meaninglessness; asserting the scale is
  the honest statement of the anchor.

Usage
-----
    python benchmarks/lake_at_rest/freeze.py            # freeze
    python benchmarks/lake_at_rest/freeze.py --verify   # fresh run vs frozen
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import sys

import torch

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[1]
if str(REPO / "src") not in sys.path:
    sys.path.insert(0, str(REPO / "src"))

from tensorsph.lake_at_rest import LakeConfig, evaluate_gates, run_case  # noqa: E402

SHIPPED = REPO / "tests" / "data" / "lake_at_rest"
MANIFEST = "MANIFEST.sha256"
CHANNEL = "channel.json"
REPORT = "report.json"

#: The two configurations of the anchor, both at the published resolution.
RUNS = {
    "periodic_dx020": dict(periodic_x=True),
    "wall_dx020": dict(periodic_x=False),
}

#: Quantities that a re-run must reproduce (they are not cancellation results:
#: they are one solve, one sum or one integer).  Compared with ``rtol``.
STABLE = [
    ("equilibrium", "iterations"),
    ("equilibrium", "converged"),
    ("equilibrium", "p_max"),
    ("equilibrium", "rho_over_rho0_min"),
    ("equilibrium", "rho_over_rho0_max"),
    ("profiles", "density_vs_eos_max_rel"),
    ("profiles", "pressure_vs_eos_max_rel"),
    ("profiles", "pressure_vs_rho0gh_max_rel"),
    ("rest", "dt"),
    ("rest", "steps"),
    ("rest", "t_end"),
    ("rest", "total_mass"),
]
STABLE_RTOL = 1.0e-8

#: Quantities that are *cancellation* results: the equilibrium residual and the
#: drift it produces over the run.  They are reproducible as a **scale**, not as
#: a number.  ``ratio`` is the allowed fresh/frozen band around 1.
ROUNDING = [
    ("equilibrium", "a_max_over_g"),
    ("equilibrium", "a_max_over_g_horizontal"),
    ("rest", "max_v_over_c0"),
    ("rest", "max_dy_over_H"),
    ("rest", "max_drho_over_rho0"),
]
ROUNDING_RATIO = (0.1, 10.0)
#: Above the tolerance the quantity is a truncation force, not a cancellation:
#: it is reproducible as a number.  This is also the guard that keeps the
#: wall-bounded honest FAIL from silently turning into a pass.
FAILING_RTOL = 1.0e-2
#: The anchor's own tolerance (machine precision on the analytic identity).
ANCHOR_TOLERANCE = 1.0e-9

#: Negative controls: they are one ``rhs`` evaluation on a fixed mass field, so
#: they must reproduce tightly *and* they must stay ``O(1)``/``O(g)`` -- that is
#: what makes the rounding-level equilibrium a balance of a live operator.
CONTROLS_RTOL = 1.0e-8


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _config(periodic_x: bool) -> LakeConfig:
    # Published configuration of the anchor (defaults of ``run.py``).
    return LakeConfig(dx=0.02, depth=0.30, width=0.40, t_end=0.50,
                      periodic_x=periodic_x)


def run_one(periodic_x: bool) -> dict:
    """One case run plus its gate record and the parameters that produced it."""
    cfg = _config(periodic_x)
    previous = torch.get_default_dtype()
    torch.set_default_dtype(cfg.torch_dtype)
    try:
        result = run_case(cfg)
    finally:
        torch.set_default_dtype(previous)
    gates = evaluate_gates(result)
    result["gates"] = {"rows": gates["rows"], "verdicts": gates["verdicts"],
                       "all_passed": gates["all_passed"]}
    result["parameters"] = {"periodic_x": bool(cfg.periodic_x), "dx": cfg.dx,
                            "depth": cfg.depth, "width": cfg.width,
                            "t_end": cfg.t_end, "max_courant": cfg.max_courant,
                            "h_factor": cfg.h_factor,
                            "prior_weight": cfg.prior_weight,
                            "solver_tol": cfg.solver_tol}
    return result


def _get(record: dict, path: tuple):
    node = record
    for key in path:
        node = node[key]
    return node


def _report(runs: dict) -> dict:
    """Summary of both runs: verdicts, anchor tolerance, reproducibility map."""
    periodic = runs["periodic_dx020"]
    wall = runs["wall_dx020"]
    return {
        "case": "CASE-SPH-LAKE-AT-REST",
        "title": "still lake: zero velocity, zero surface oscillation",
        "reference": periodic["reference"],
        "anchor": {
            "tolerance": 1.0e-9,
            "quantities": ["equilibrium residual max|a|/g",
                           "rest max|v|/c0", "surface max|dy|/H",
                           "density max|drho|/rho0"],
            "acceptance_gates": ["G1", "G2", "G3", "G4", "G5"],
            "note": "analytic identity: there is no experimental tolerance and "
                    "no fitted constant in this case",
        },
        "runs": {
            name: {
                "geometry": record["geometry"],
                "parameters": record["parameters"],
                "equilibrium": {k: v for k, v in
                                record["equilibrium"].items()
                                if k != "residual_history"},
                "rest": record["rest"],
                "controls": record["controls"],
                "profiles": {k: v for k, v in record["profiles"].items()
                             if k != "row_profiles"},
                "gates": record["gates"],
            }
            for name, record in runs.items()
        },
        "verdict": {
            "periodic_dx020": {
                "status": ("PASS" if periodic["gates"]["all_passed"]
                           else "FAIL"),
                "detail": "the anchor tolerance is met where the operator is "
                          "translation invariant in x",
            },
            "wall_dx020": {
                "status": ("PASS" if wall["gates"]["all_passed"] else "FAIL"),
                "detail": "honest FAIL: side-wall truncation leaves an odd "
                          "horizontal residual that no mirror-symmetric mass "
                          "field can cancel; see docs/verification_lake_at_rest.md",
            },
        },
        "reproducibility": {
            "stable": [".".join(path) for path in STABLE],
            "stable_rtol": STABLE_RTOL,
            "rounding_level": [".".join(path) for path in ROUNDING],
            "rounding_ratio_band": list(ROUNDING_RATIO),
            "note": "the rounding-level entries are cancellations of double "
                    "precision pair sums: identical code, identical inputs and "
                    "identical iteration counts still move them by ~30 % "
                    "between runs. The reproducible statement is the scale "
                    "(below the 1e-9 anchor tolerance), not the digits.",
        },
    }


#: Records produced by the *other* runners of this case and shipped next to the
#: two case runs: the rest-horizon study (``stability.py``) and its band-fit
#: re-reduction (``band_fit.py``).  ``freeze`` does not regenerate them -- they
#: cost minutes of cluster time -- but it must hash them, otherwise they land in
#: ``tests/data/lake_at_rest`` unmanifested and the evidence channel's own gate
#: (``tests/test_lake_at_rest_evidence.py``) fails on a fresh checkout.  Only the
#: files that exist are shipped, and ``--reship`` re-hashes the frozen payload
#: without re-running the physics so the published numbers keep their bytes.
EXTRA_RECORDS = ("stability.json", "stability_band_fit.json", "hydrostatic_closure.json", "stability_viscous.json", "wall_gap.json", "wellbalanced.json", "wb_operator.json")


def _extra_payload(ship: Path) -> list:
    return [ship / name for name in EXTRA_RECORDS if (ship / name).is_file()]


def reship(ship: Path = SHIPPED) -> dict:
    """Re-hash every shipped ``*.json`` without re-running the case."""
    payload = [ship / f"{name}.json" for name in RUNS] + [ship / REPORT]
    payload += _extra_payload(ship)
    missing = [str(path.name) for path in payload if not path.is_file()]
    if missing:
        raise SystemExit(f"cannot reship: missing {missing}")
    lines = [f"{_sha256(path)}  {path.stat().st_size}  {path.name}"
             for path in payload]
    (ship / MANIFEST).write_text("\n".join(lines) + "\n")
    channel = json.loads((ship / CHANNEL).read_text())
    channel["shipped"] = [path.name for path in payload] + [MANIFEST, CHANNEL]
    channel["extra_records_from"] = {
        "stability.json": "benchmarks/lake_at_rest/stability.py",
        "stability_band_fit.json": "benchmarks/lake_at_rest/band_fit.py",
        "stability_viscous.json": "benchmarks/lake_at_rest/stability.py "
                                  "--viscosity 1e-6",
        "wall_gap.json": "benchmarks/lake_at_rest/wall_gap.py",
    }
    (ship / CHANNEL).write_text(json.dumps(channel, indent=2, sort_keys=True))
    print(f"reship: {len(payload)} record(s) manifested into "
          f"{ship.relative_to(REPO)} (physics not re-run)")
    return channel


def freeze(ship: Path = SHIPPED) -> dict:
    """Run both configurations and write the text channel."""
    ship.mkdir(parents=True, exist_ok=True)
    runs = {}
    for name, kwargs in RUNS.items():
        record = run_one(**kwargs)
        (ship / f"{name}.json").write_text(
            json.dumps(record, indent=2, sort_keys=True))
        runs[name] = record
        print(f"{name}: all_passed={record['gates']['all_passed']} "
              f"max|a|/g={record['equilibrium']['a_max_over_g']:.3e} "
              f"max|v|/c0={record['rest']['max_v_over_c0']:.3e}")
    report = _report(runs)
    (ship / REPORT).write_text(json.dumps(report, indent=2, sort_keys=True))

    payload = (sorted(ship / f"{name}.json" for name in RUNS)
               + [ship / REPORT] + _extra_payload(ship))
    lines = [f"{_sha256(path)}  {path.stat().st_size}  {path.name}"
             for path in payload]
    (ship / MANIFEST).write_text("\n".join(lines) + "\n")
    channel = {
        "case": "CASE-SPH-LAKE-AT-REST",
        "shipped": [path.name for path in payload] + [MANIFEST, CHANNEL],
        "not_shipped": {
            "png": ["docs/lake_at_rest_periodic_*.png"],
            "artifacts": ["artifacts/lake_at_rest/*.json (gitignored)"],
            "why": "figures are artifact-class in the delivery rule and are "
                   "re-creatable from the shipped numbers; artifacts/ is not "
                   "pushed, so evidence kept only there is absent where the "
                   "case is read",
        },
        "frozen_by": "benchmarks/lake_at_rest/freeze.py",
        "regenerate": "PYTHONPATH=src python3 benchmarks/lake_at_rest/run.py "
                      "--periodic-x --json artifacts/lake_at_rest/"
                      "rest_dx020_periodic.json",
        "read_by": "tests/test_lake_at_rest_evidence.py",
        "manifest": MANIFEST,
    }
    (ship / CHANNEL).write_text(json.dumps(channel, indent=2, sort_keys=True))
    print(f"froze {len(payload)} record(s) into {ship.relative_to(REPO)}")
    return report


def verify(ship: Path = SHIPPED) -> list:
    """Re-run both configurations; return a list of failure strings."""
    frozen_report = json.loads((ship / REPORT).read_text())
    failures = []
    for name, kwargs in RUNS.items():
        frozen = json.loads((ship / f"{name}.json").read_text())
        fresh = run_one(**kwargs)
        for path in STABLE:
            a, b = _get(frozen, path), _get(fresh, path)
            if a == b:
                continue
            scale = max(abs(a), abs(b), 1e-300)
            if abs(a - b) / scale > STABLE_RTOL:
                failures.append(f"{name}:{'.'.join(path)} stable {a!r} -> {b!r}")
        for path in ROUNDING:
            a, b = _get(frozen, path), _get(fresh, path)
            below = (a < ANCHOR_TOLERANCE, b < ANCHOR_TOLERANCE)
            label = f"{name}:{'.'.join(path)}"
            if below[0] != below[1]:
                # A quantity that met the anchor tolerance must keep meeting
                # it; one that fails it may only fail (the wall-bounded tank's
                # O(g) side-wall residual is the honest FAIL and must not be
                # "fixed" by accident).
                failures.append(f"{label} crossed the anchor tolerance "
                                f"({a:.3e} -> {b:.3e})")
            elif below[0]:
                ratio = b / a if a else float("inf")
                low, high = ROUNDING_RATIO
                if not low <= ratio <= high:
                    failures.append(f"{label} outside the rounding band "
                                    f"({a:.3e} -> {b:.3e}, ratio {ratio:.2f})")
            else:
                scale = max(abs(a), abs(b), 1e-300)
                if abs(a - b) / scale > FAILING_RTOL:
                    failures.append(f"{label} moved on the failing side "
                                    f"({a:.3e} -> {b:.3e})")
        for key, control in frozen["controls"].items():
            for field, value in control.items():
                other = fresh["controls"][key][field]
                scale = max(abs(value), abs(other), 1e-300)
                if abs(value - other) / scale > CONTROLS_RTOL:
                    failures.append(f"{name}:controls.{key}.{field} "
                                    f"{value!r} -> {other!r}")
        regraded = evaluate_gates(frozen)
        shipped_gates = frozen_report["runs"][name]["gates"]
        if (shipped_gates["verdicts"] != regraded["verdicts"]
                or shipped_gates["all_passed"] != regraded["all_passed"]):
            failures.append(f"{name}: report gates disagree with the record "
                            f"they claim to summarise")
        if frozen["gates"]["verdicts"] != fresh["gates"]["verdicts"]:
            failures.append(f"{name}: gate verdicts changed "
                            f"{frozen['gates']['verdicts']} -> "
                            f"{fresh['gates']['verdicts']}")
        if frozen_report["verdict"][name]["status"] != (
                "PASS" if fresh["gates"]["all_passed"] else "FAIL"):
            failures.append(f"{name}: shipped verdict disagrees with the re-run")
    for line in (ship / MANIFEST).read_text().splitlines():
        digest, _size, name = line.split()
        if _sha256(ship / name) != digest:
            failures.append(f"{name}: sha256 mismatch against {MANIFEST}")
    return failures


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--verify", action="store_true",
                        help="re-run the case and compare with the frozen "
                             "channel instead of writing it")
    parser.add_argument("--out", type=Path, default=SHIPPED)
    parser.add_argument("--reship", action="store_true",
                        help="re-hash the frozen payload without re-running the "
                             "case (use after a sibling runner shipped a record)")
    args = parser.parse_args(argv)
    if args.reship:
        reship(args.out)
        return 0
    if args.verify:
        failures = verify(args.out)
        if failures:
            print(f"VERIFY FAILED ({len(failures)}):")
            for failure in failures:
                print(f"  {failure}")
            return 1
        print("VERIFY OK: frozen channel reproduced (stable quantities by "
              "rtol, rounding-level quantities by scale)")
        return 0
    freeze(args.out)
    return 0


if __name__ == "__main__":
    sys.exit(main())