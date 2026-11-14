"""Cross-validation tests for the Couette viscous-time evidence leg.

Anchor ``XVAL-SPH-METHOD-SELECTION-2026-11-12`` (TensorSPH method selection).

Context these tests lock down
-----------------------------
``wcsph_wall_no_slip`` is banked in ``docs/reports/method_selection_registry.json``
with a *steady-state precondition* that promised the reading would be bounded by a
viscous-time sweep (``benchmarks/couette/out_design/viscous_time_convergence.json``).
The sweep driver crashed on the compute node before that file existed
(``ModuleNotFoundError: No module named 'matplotlib'`` at import time, cluster job
cj2857), so for one mission the registry cited a bounding artefact that did not
exist.  Each test below fails on its own if that class of gap comes back:

* the sweep driver must import and compute without matplotlib (plotting is
  reporting, never a precondition for numbers),
* every repository path a registry record *cites* must exist (a precondition is a
  machine-checkable promise, not prose),
* the banked converged reading and its mobility must be reproducible from the raw
  sweep points, independently of the registry builder,
* admissibility must mean what the artefact says (drift gate), no worse.
"""
from __future__ import annotations

import json
import math
import re
import subprocess
import sys
import textwrap
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[1]
REGISTRY_JSON = REPO / "docs" / "reports" / "method_selection_registry.json"
SWEEP_JSON = REPO / "benchmarks" / "couette" / "out_design" / "viscous_time_convergence.json"
DRIVER = REPO / "benchmarks" / "couette" / "viscous_time_convergence.py"

#: Paths cited inside a registry record (precondition / quantity prose).
PATH_RE = re.compile(r"[A-Za-z0-9_][A-Za-z0-9_./-]*\.(?:json|md|py|png)")


def _registry() -> dict:
    return json.loads(REGISTRY_JSON.read_text())


def _sweep() -> dict:
    if not SWEEP_JSON.exists():
        pytest.fail(f"{SWEEP_JSON.relative_to(REPO)} is missing: the registry cites it as "
                    f"the bound on the Couette steady-state precondition")
    return json.loads(SWEEP_JSON.read_text())


# --------------------------------------------------------------------------- #
# 1. headless capability of the sweep driver
# --------------------------------------------------------------------------- #
def test_sweep_driver_imports_without_matplotlib():
    """The numbers must not depend on the plotting library being installed."""
    code = textwrap.dedent(
        """
        import importlib.abc, sys
        class Block(importlib.abc.MetaPathFinder):
            def find_spec(self, name, path=None, target=None):
                if name == "matplotlib" or name.startswith("matplotlib."):
                    raise ImportError("blocked: compute nodes ship no matplotlib")
                return None
        sys.meta_path.insert(0, Block())
        sys.path.insert(0, "benchmarks/couette")
        sys.path.insert(0, "src")
        import viscous_time_convergence as v
        try:
            v._pyplot()
        except RuntimeError as exc:
            assert "matplotlib" in str(exc)
        else:
            raise SystemExit("_pyplot() must raise RuntimeError without matplotlib")
        print("headless-ok")
        """
    )
    proc = subprocess.run([sys.executable, "-c", code], cwd=REPO,
                          capture_output=True, text=True, timeout=180)
    assert proc.returncode == 0, proc.stderr
    assert "headless-ok" in proc.stdout


def test_sweep_driver_has_no_module_level_matplotlib_import():
    """A regression guard that does not need a subprocess."""
    source = DRIVER.read_text()
    for line in source.splitlines():
        if line.startswith(" "):
            continue  # indented imports are the lazy ones we want
        stripped = line.strip()
        if stripped.startswith("#") or stripped.startswith('"""'):
            continue
        assert not stripped.startswith("import matplotlib"), \
            "matplotlib must stay a lazy import (see cj2857)"
        assert "matplotlib.use(" not in stripped or "def " in stripped, \
            "matplotlib.use() belongs inside _pyplot(), not at import time"


# --------------------------------------------------------------------------- #
# 2. cited evidence must exist
# --------------------------------------------------------------------------- #
def test_every_cited_repository_path_exists():
    """A precondition that cites an artefact is a promise; the file must be there."""
    cited: dict[str, list[str]] = {}
    for record in _registry()["records"]:
        prose = " ".join(str(record.get(key, "")) for key in ("precondition", "quantity"))
        for match in PATH_RE.findall(prose):
            if match.startswith(("http", "tensorsph.")):
                continue
            cited.setdefault(match, []).append(record["method"])

    def resolves(rel: str) -> bool:
        return (REPO / rel).exists() or next(REPO.rglob(Path(rel).name), None) is not None

    missing = sorted(p for p in cited if not resolves(p))
    assert not missing, (
        "registry records cite paths that do not exist in the repository: "
        f"{[(p, cited[p]) for p in missing]}"
    )


# --------------------------------------------------------------------------- #
# 3. the banked numbers are reproducible from the raw sweep
# --------------------------------------------------------------------------- #
def _admissible(rows: list[dict]) -> list[dict]:
    return [r for r in rows if r["admissible"]]


def _converged_pct(rows: list[dict]) -> float:
    """Reading at the *longest* admissible run (the steady end of the sweep)."""
    pool = _admissible(rows) or rows
    return max(pool, key=lambda r: r["viscous_times"])["wall_shear_error_pct_bottom"]


def _mobility_pp(rows: list[dict]) -> float:
    """How far the reading moves across admissible runs (percentage points)."""
    pool = _admissible(rows) or rows
    values = [r["wall_shear_error_pct_bottom"] for r in pool]
    return max(values) - min(values)


def _banked_record(extractor: str) -> dict:
    for record in _registry()["records"]:
        if record.get("extractor") == extractor:
            return record
    raise AssertionError(f"no registry record from extractor {extractor!r}")


@pytest.mark.parametrize("resolution", [8, 16])
def test_banked_converged_reading_matches_raw_sweep(resolution):
    doc = _sweep()
    rows = [p for p in doc["points"] if p["resolution_H_over_dx"] == resolution]
    assert rows, f"sweep has no points at H/dx={resolution}"
    extractor = f"x_couette_viscous_time_h{resolution}"
    record = _banked_record(extractor)
    expected = _converged_pct(rows)
    assert math.isclose(record["measured_pct"], expected, rel_tol=1e-9), (
        f"{extractor}: banked {record['measured_pct']:.6f} % but the raw sweep "
        f"recomputes {expected:.6f} %")


@pytest.mark.parametrize("resolution", [8, 16])
def test_banked_mobility_matches_raw_sweep(resolution):
    doc = _sweep()
    rows = [p for p in doc["points"] if p["resolution_H_over_dx"] == resolution]
    extractor = f"x_couette_viscous_time_mobility_h{resolution}"
    record = _banked_record(extractor)
    expected = _mobility_pp(rows)
    assert math.isclose(record["measured_pct"], expected, rel_tol=1e-9, abs_tol=1e-12), (
        f"{extractor}: banked {record['measured_pct']:.6f} pp but the raw sweep "
        f"recomputes {expected:.6f} pp")


# --------------------------------------------------------------------------- #
# 4. admissibility means what the artefact says
# --------------------------------------------------------------------------- #
def test_admissible_points_pass_the_artefact_steady_gate():
    doc = _sweep()
    drift_gate = float(doc["gates"]["steady_drift"])
    for point in doc["points"]:
        if point["admissible"]:
            assert float(point["steady_drift"]) < drift_gate, (
                f"{point['label']}: marked admissible although steady_drift "
                f"{point['steady_drift']:.3e} >= gate {drift_gate:.3e}")
        assert point["admissible"] == (
            float(point["steady_drift"]) < drift_gate), \
            f"{point['label']}: admissibility flag disagrees with the steady-drift gate"


def test_summary_block_agrees_with_points():
    """The artefact's summary may not drift away from its own raw points."""
    doc = _sweep()
    for resolution in doc["resolutions"]:
        rows = [p for p in doc["points"] if p["resolution_H_over_dx"] == resolution]
        block = doc["summary"][str(resolution)]
        assert block["n_points"] == len(rows)
        assert math.isclose(block["converged_error_pct"], _converged_pct(rows),
                            rel_tol=1e-9)
        adm = [r["viscous_times"] for r in _admissible(rows)]
        assert [float(x) for x in block["admissible_viscous_times"]] == pytest.approx(adm)


def test_sweep_covers_the_design_viscous_time():
    """The bound is only useful if the sweep contains, and passes, the design point."""
    doc = _sweep()
    design = float(doc["design_viscous_times"])
    swept = [float(v) for v in doc["viscous_times_swept"]]
    assert any(abs(design - v) < 1e-9 for v in swept), \
        f"design viscous time {design} not in the swept set {swept}"
    assert max(swept) > design, \
        "a sweep that stops at the design time cannot show the reading has settled"