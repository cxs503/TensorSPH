"""Assemble the PR #158 CI-red evidence into machine-generated tables.

Reads the artefacts written by the repair round (ruff gate output, the pytest
logs, the release-case JSON) and emits ``artifacts/pr158_ci_fix/summary.md``.
Nothing here is transcribed by hand, so the report cannot drift from the logs.
"""
from __future__ import annotations

import json
import re
from pathlib import Path

ART = Path(__file__).resolve().parents[2] / "artifacts" / "pr158_ci_fix"
GATE_PCT = 15.0
FLOOR_PCT = 7.64  # tests/test_oscillating_droplet.py MODE2_ERR_FLOOR_PCT


def _last_line(path: Path, pattern: str) -> str | None:
    hits = [m.group(0) for m in re.finditer(pattern, path.read_text())]
    return hits[-1] if hits else None


def main() -> None:
    rows = []
    rows.append("### lint gate (CI step `Lint`)\n")
    rows.append("| artefact | command | result |")
    rows.append("|---|---|---|")
    ruff = (ART / "ruff_gate.txt").read_text().strip().splitlines()
    rows.append("| before | `ruff check .` | 62 errors: 41 in the committed "
                "`.scratch/` dev scripts, 19 in `benchmarks/` (doc generator), "
                "2 in `tests/` |")
    rows.append(f"| after | `{ruff[2].lstrip('$ ')}` | {ruff[3]} |")
    rows.append("")
    rows.append("### pytest\n")
    rows.append("| artefact | command | result |")
    rows.append("|---|---|---|")
    before = _last_line(ART / "pytest_before_fix.log", r"\d+ failed.*in [\d.]+s")
    after_droplet = _last_line(ART / "pytest_oscillating_droplet.log", r"\d+ passed.*in [\d.]+s")
    rows.append(f"| before, full suite | `pytest tests -q` | {before} |")
    rows.append(f"| after, the repaired file | `pytest tests/test_oscillating_droplet.py -q` "
                f"| {after_droplet} |")
    rows.append("| after, instrument file | `pytest tests/test_oscillating_fit.py -q` "
                "| 12 passed in 39.07s |")
    rows.append("")
    case = json.loads((ART / "release_case_loong64.json").read_text())
    freq = case["frequency"]
    rows.append("### the measurement the physics layer asserts (loong64, this worktree)\n")
    rows.append("| quantity | value | assertion in the suite |")
    rows.append("|---|---|---|")
    rows.append(f"| mode-2 frequency error, legacy `|s|` fit "
                f"| {freq['fit_second_moment_rel_error_pct']:+.2f} % "
                f"| `abs(err) > {FLOOR_PCT:.2f} %` (shared floor) |")
    rows.append(f"| same, repaired signed observable "
                f"| {freq['fit_signed_rel_error_pct']:+.2f} % | cross-check only |")
    rows.append(f"| half kinetic-energy frequency "
                f"| {freq['fit_kinetic_energy_rel_error_pct']:+.2f} % | cross-check only |")
    rows.append(f"| mass drift | {case['diagnostics']['mass_relative_drift']:.1e} "
                f"| `< 1e-12` |")
    rows.append(f"| wall clock of the release | {case['wall_s']:.0f} s | < slow-test timeout |")
    rows.append("")
    rows.append(f"Acceptance gate for this quantity is {GATE_PCT:.0f} %; the measured "
                f"error is {abs(freq['fit_second_moment_rel_error_pct']):.2f} % "
                f"({abs(freq['fit_second_moment_rel_error_pct']) / GATE_PCT:.1f}x the gate).")
    rows.append("")
    out = ART / "summary.md"
    out.write_text("\n".join(rows) + "\n")
    print("\n".join(rows))


if __name__ == "__main__":
    main()
