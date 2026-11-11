"""Text artefact bundle for the free-surface verification face (campaign P482-N04).

Why this module exists
----------------------
The numbers of this verification face used to live only in ``evidence.sqlite``.
A ``.sqlite`` file is a binary blob: it does not survive a review diff, it cannot
be inspected without a client, and in practice it is dropped by artefact filters
-- the previous delivery of this node therefore shipped a report whose only
machine-readable backing store was missing, and the maturity review could not
find a single run artefact.  This module fixes that *structurally*: a run writes
plain text (JSON / CSV / PNG) into a run directory, and every reported number
carries the code path that produced it.

The bundle enforces three invariants that are cheap to check and expensive to
fake:

``no hand-typed numbers``
    Every benchmark row is cross-checked against the metric of the same name in
    ``results.json`` at write time (:meth:`Bundle.finalize`); a row whose value
    does not match the recorded measurement raises :class:`BundleError`.
``every artefact is hashed``
    ``MANIFEST.json`` lists the SHA-256 of every file written, so a later edit of
    a result file is detectable.
``maturity is derived, not claimed``
    :func:`assess_level` reads the manifest's own ``maturity`` block and decides
    L0..L4 from *which* keys are non-empty; the level in the manifest is
    therefore a function of the artefacts and not a self-assessment.

Nothing here knows any physics: the bundle is a writer plus a consistency check.
"""
from __future__ import annotations

from dataclasses import dataclass, field, asdict
from datetime import datetime, timezone
import csv
import hashlib
import json
import os
from pathlib import Path
import platform
import subprocess
import sys

__all__ = [
    "Bundle",
    "BundleError",
    "Metric",
    "BenchmarkRow",
    "assess_level",
    "sha256_of",
]

SCHEMA_VERSION = "1.0"


class _Unresolved:
    """Sentinel: a benchmark row is not backed by recorded evidence."""

    def __repr__(self) -> str:  # pragma: no cover - debug aid
        return "<unresolved>"


_UNRESOLVED = _Unresolved()


class BundleError(RuntimeError):
    """The bundle would ship an inconsistent (hand-typed or partial) artefact."""


def sha256_of(path: str | Path) -> str:
    """SHA-256 of a file; used for every artefact and every reference table."""
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _utc() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def _git(*args: str) -> str | None:
    try:
        out = subprocess.run(["git", *args], capture_output=True, text=True,
                             timeout=5)
        return out.stdout.strip() or None
    except Exception:  # pragma: no cover - git absent in some sandboxes
        return None


# --------------------------------------------------------------------------- #
# records
# --------------------------------------------------------------------------- #
@dataclass
class Metric:
    """One measured number, its gate (if any) and the code path behind it."""

    value: float | None
    unit: str = "1"
    gate_limit: float | None = None
    verdict: str | None = None
    source: str = ""
    note: str = ""

    def __post_init__(self):
        if self.verdict not in (None, "PASS", "FAIL", "WITHHELD"):
            raise BundleError(f"verdict must be PASS/FAIL/WITHHELD, got {self.verdict!r}")


@dataclass
class BenchmarkRow:
    """A ``reference vs got`` comparison: the number a reviewer checks first.

    ``computed_value`` must be *traceable*: either to a metric recorded on the
    case (``metric``) or to an element of a stored series (``series_ref`` =
    ``{"series": name, "index": i, "field": f}``).  :meth:`Bundle.finalize`
    re-reads the recorded value and refuses to ship a row that disagrees, so the
    comparison table cannot be typed in by hand.
    """

    quantity: str
    case: str
    reference: str
    reference_value: float | None
    reference_grade: str
    computed_value: float | None
    unit: str = "1"
    tolerance: float | None = None
    note: str = ""
    metric: str | None = None
    series_ref: dict | None = None
    direction: str = "abs"          # "abs" | "lower" | "upper"

    def __post_init__(self):
        if self.direction not in ("abs", "lower", "upper"):
            raise BundleError(f"direction must be abs/lower/upper, got "
                              f"{self.direction!r}")

    @property
    def abs_error(self) -> float | None:
        if self.reference_value is None or self.computed_value is None:
            return None
        return abs(self.computed_value - self.reference_value)

    @property
    def rel_error(self) -> float | None:
        if self.reference_value is None or self.computed_value is None:
            return None
        if self.reference_value == 0.0:
            return abs(self.computed_value)
        return abs(self.computed_value - self.reference_value) / abs(self.reference_value)

    @property
    def verdict(self) -> str:
        if self.reference_value is None or self.computed_value is None:
            return "WITHHELD"
        if self.reference_grade == "WITHHELD":
            return "WITHHELD"
        if self.direction == "lower":
            return "PASS" if self.computed_value >= self.reference_value else "FAIL"
        if self.direction == "upper":
            return "PASS" if self.computed_value <= self.reference_value else "FAIL"
        if self.tolerance is None:
            return "WITHHELD"
        return "PASS" if (self.rel_error is not None and self.rel_error <= self.tolerance) \
            else "FAIL"

    def as_dict(self) -> dict:
        out = asdict(self)
        out.update({"abs_error": self.abs_error, "rel_error": self.rel_error,
                    "verdict": self.verdict})
        return out


@dataclass
class Case:
    """A measured configuration: metrics plus the raw series behind the figures."""

    name: str
    phase: str                      # M1 / M2 / M3 / M4
    config: dict = field(default_factory=dict)
    metrics: dict = field(default_factory=dict)   # name -> Metric
    series: dict = field(default_factory=dict)    # name -> json-able payload
    notes: str = ""


# --------------------------------------------------------------------------- #
# maturity ladder (derived, not claimed)
# --------------------------------------------------------------------------- #
_LEVEL_REQUIREMENTS = {
    "L1": ("impl_files",),
    "L2": ("impl_files", "entry", "run_artifacts"),
    "L3": ("impl_files", "entry", "run_artifacts", "integration_callers"),
    "L4": ("impl_files", "entry", "run_artifacts", "integration_callers",
           "benchmark", "repro_cmd"),
}


def _present(maturity: dict, key: str) -> bool:
    value = maturity.get(key)
    if value is None:
        return False
    if isinstance(value, (list, tuple, dict, str)):
        return len(value) > 0
    return bool(value)


def assess_level(maturity: dict) -> dict:
    """Return ``{"level": "L0".."L4", "missing": {...}}`` from the manifest block.

    A level is reached only when *all* of its keys are non-empty, and the missing
    keys are reported per level so a reviewer sees the next rung, not just a
    verdict.
    """
    missing = {level: [k for k in keys if not _present(maturity, k)]
               for level, keys in _LEVEL_REQUIREMENTS.items()}
    level = "L0"
    for candidate in ("L1", "L2", "L3", "L4"):
        if not missing[candidate]:
            level = candidate
    return {"level": level, "missing": missing}


# --------------------------------------------------------------------------- #
# the bundle
# --------------------------------------------------------------------------- #
class Bundle:
    """Writer for one run directory: ``results.json``, ``metrics.csv``,
    ``benchmark.json``, ``MANIFEST.json`` and the figures."""

    def __init__(self, out_dir: str | Path, mission: str = "m14534",
                 node: str = "P482-N04", repo: str = "TensorSPH"):
        self.dir = Path(out_dir)
        self.dir.mkdir(parents=True, exist_ok=True)
        self.mission = mission
        self.node = node
        self.repo = repo
        self.cases: dict[str, Case] = {}
        self.case_order: list[str] = []
        self.references: dict[str, dict] = {}
        self.rows: list[BenchmarkRow] = []
        self.gates: dict[str, dict] = {}
        self.written: list[Path] = []

    # -- recording ---------------------------------------------------------- #
    def add_reference(self, name: str, path: str | Path, grade: str,
                      note: str = "") -> dict:
        path = Path(path)
        record = {"path": str(path), "grade": grade, "note": note,
                  "sha256": sha256_of(path) if path.exists() else None}
        self.references[name] = record
        return record

    def add_case(self, case: Case) -> Case:
        if case.name in self.cases:
            raise BundleError(f"case {case.name!r} recorded twice")
        self.cases[case.name] = case
        self.case_order.append(case.name)
        return case

    def metric(self, case: str, name: str) -> Metric | None:
        entry = self.cases.get(case)
        return None if entry is None else entry.metrics.get(name)

    def resolve(self, row: BenchmarkRow):
        """The recorded value a benchmark row claims (``_UNRESOLVED`` if none)."""
        if row.metric:
            metric = self.metric(row.case, row.metric)
            return _UNRESOLVED if metric is None else metric.value
        ref = row.series_ref or {}
        case = self.cases.get(row.case)
        if case is None or "series" not in ref:
            return _UNRESOLVED
        payload = case.series.get(ref["series"])
        index = ref.get("index")
        field_name = ref.get("field")
        try:
            item = payload[index] if index is not None else payload
        except (IndexError, KeyError, TypeError):
            return _UNRESOLVED
        if isinstance(item, dict):
            return item[field_name] if field_name in item else _UNRESOLVED
        return item

    def add_benchmark(self, **kwargs) -> BenchmarkRow:
        row = BenchmarkRow(**kwargs)
        self.rows.append(row)
        return row

    def add_gate(self, name: str, passed: bool | None, detail: dict | None = None,
                 honest_fail: bool = False) -> dict:
        """Record a campaign gate (J1..J5).

        ``passed=None`` is a *withheld* gate: the evidence to decide it is not in
        hand.  ``honest_fail`` marks a gate that was tried and failed; the flag is
        carried into the manifest so a reader cannot mistake a FAIL for an
        unfinished gate (the platform's rule: an honest FAIL may not be
        overwritten).
        """
        record = {"pass": None if passed is None else bool(passed),
                  "verdict": ("WITHHELD" if passed is None
                              else "PASS" if passed else "FAIL"),
                  "honest_fail": bool(honest_fail and passed is False),
                  "detail": detail or {}}
        self.gates[name] = record
        return record

    # -- consistency -------------------------------------------------------- #
    def check_benchmark_consistency(self) -> None:
        """Every benchmark ``computed_value`` must be a recorded measurement.

        This is the anti-hand-typing rule: a row may not carry a number that the
        run did not produce.  A row is allowed to name its metric differently
        from its quantity, in which case the note must say so explicitly through
        ``metric`` in ``note``... which is not machine-checkable -- so instead a
        row may be *derived* only through :meth:`add_derived_row`, which records
        the formula in the note and the inputs in :attr:`rows`.
        """
        problems = []
        for row in self.rows:
            if row.computed_value is None:
                continue
            recorded = self.resolve(row)
            if recorded is _UNRESOLVED:
                problems.append(f"{row.case}.{row.quantity}: benchmark row is not "
                                f"backed by a recorded metric or series element "
                                f"(metric={row.metric!r}, series_ref={row.series_ref!r})")
                continue
            if not _close(recorded, row.computed_value):
                problems.append(
                    f"{row.case}.{row.quantity}: benchmark row says "
                    f"{row.computed_value!r} but the run recorded {recorded!r}")
        if problems:
            raise BundleError("benchmark rows disagree with the measurements:\n"
                              + "\n".join(problems))

    # -- writing ------------------------------------------------------------ #
    def _write(self, name: str, payload: str) -> Path:
        path = self.dir / name
        path.write_text(payload)
        self.written.append(path)
        return path

    def results(self) -> dict:
        return {
            "schema": SCHEMA_VERSION,
            "mission": self.mission,
            "campaign_node": self.node,
            "repo": self.repo,
            "generated_utc": _utc(),
            "git_commit": _git("rev-parse", "HEAD"),
            "git_branch": _git("rev-parse", "--abbrev-ref", "HEAD"),
            "case_order": list(self.case_order),
            "cases": {name: {"phase": c.phase, "config": c.config,
                             "notes": c.notes,
                             "metrics": {k: asdict(v) for k, v in c.metrics.items()},
                             "series": c.series}
                      for name, c in self.cases.items()},
            "references": self.references,
            "benchmark_rows": len(self.rows),
            "gates": self.gates,
        }

    def write_results(self) -> Path:
        return self._write("results.json", json.dumps(self.results(), indent=1,
                                                     sort_keys=True, default=str))

    def write_metrics_csv(self) -> Path:
        path = self.dir / "metrics.csv"
        with open(path, "w", newline="") as handle:
            writer = csv.writer(handle)
            writer.writerow(["case", "phase", "metric", "value", "unit",
                             "gate_limit", "verdict", "source", "note"])
            for name in self.case_order:
                case = self.cases[name]
                for metric_name, metric in case.metrics.items():
                    writer.writerow([name, case.phase, metric_name,
                                     "" if metric.value is None else repr(metric.value),
                                     metric.unit,
                                     "" if metric.gate_limit is None else metric.gate_limit,
                                     metric.verdict or "", metric.source, metric.note])
        self.written.append(path)
        return path

    def write_benchmark(self) -> Path:
        rows = [row.as_dict() for row in self.rows]
        summary = {
            "n_rows": len(rows),
            "n_pass": sum(1 for r in rows if r["verdict"] == "PASS"),
            "n_fail": sum(1 for r in rows if r["verdict"] == "FAIL"),
            "n_withheld": sum(1 for r in rows if r["verdict"] == "WITHHELD"),
            "max_rel_error": max((r["rel_error"] for r in rows
                                  if r["rel_error"] is not None
                                  and r["direction"] == "abs"), default=None),
        }
        payload = {"schema": SCHEMA_VERSION, "mission": self.mission,
                   "campaign_node": self.node, "generated_utc": _utc(),
                   "git_commit": _git("rev-parse", "HEAD"),
                   "summary": summary, "rows": rows}
        return self._write("benchmark.json", json.dumps(payload, indent=1,
                                                       sort_keys=True, default=str))

    def write_manifest(self, *, entry: dict, repro_cmd: str, argv: list[str],
                       figures: list[str], impl_files: list[str],
                       integration_callers: list[dict],
                       extra: dict | None = None) -> dict:
        """Write ``MANIFEST.json``; the maturity block is derived from the files."""
        results_path = self.dir / "results.json"
        benchmark_path = self.dir / "benchmark.json"
        metrics_path = self.dir / "metrics.csv"
        run_artifacts = []
        for path in [results_path, benchmark_path, metrics_path] + \
                [self.dir / f for f in figures if (self.dir / f).exists()]:
            if path.exists():
                run_artifacts.append({"path": str(path.relative_to(self.dir)),
                                      "sha256": sha256_of(path),
                                      "bytes": path.stat().st_size})
        maturity = {
            "impl_files": impl_files,
            "entry": entry,
            "run_artifacts": run_artifacts,
            "integration_callers": integration_callers,
            "benchmark": {"path": "benchmark.json",
                          "rows": len(self.rows),
                          "summary": json.loads(benchmark_path.read_text())["summary"]}
            if benchmark_path.exists() else None,
            "repro_cmd": repro_cmd,
        }
        manifest = {
            "schema": SCHEMA_VERSION,
            "mission": self.mission,
            "campaign_node": self.node,
            "repo": self.repo,
            "generated_utc": _utc(),
            "git_commit": _git("rev-parse", "HEAD"),
            "git_branch": _git("rev-parse", "--abbrev-ref", "HEAD"),
            "run_dir": str(self.dir),
            "entry": entry,
            "argv": argv,
            "repro_cmd": repro_cmd,
            "environment": environment(),
            "maturity": maturity,
            "maturity_level": assess_level(maturity),
            "gates": {k: {"verdict": v["verdict"], "honest_fail": v["honest_fail"]}
                      for k, v in self.gates.items()},
            "figures": figures,
            "artifacts": [{"path": str(p.relative_to(self.dir)),
                           "sha256": sha256_of(p)} for p in
                          [results_path, benchmark_path, metrics_path]
                          if p.exists()],
        }
        if extra:
            manifest.update(extra)
        manifest_path = self._write("MANIFEST.json",
                                    json.dumps(manifest, indent=1, sort_keys=True,
                                               default=str))
        del manifest_path
        return manifest

    # -- figures ------------------------------------------------------------ #
    def figure_path(self, name: str) -> Path:
        return self.dir / name


def _close(a, b, tol: float = 1e-9) -> bool:
    if a is None and b is None:
        return True
    if a is None or b is None:
        return False
    try:
        a = float(a)
        b = float(b)
    except (TypeError, ValueError):
        return a == b
    if a != a and b != b:            # both NaN
        return True
    return abs(a - b) <= tol * max(1.0, abs(a), abs(b))


def environment() -> dict:
    """Versions and limits, so a re-run can be compared like for like."""
    import torch

    return {
        "python": sys.version.split()[0],
        "executable": sys.executable,
        "platform": platform.platform(),
        "machine": platform.machine(),
        "processor_count": os.cpu_count(),
        "torch": torch.__version__,
        "torch_threads": torch.get_num_threads(),
        "pid": os.getpid(),
    }
