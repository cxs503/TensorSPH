"""Render the two-panel evidence figure for the PR #185 CI heal.

Panel (a): the CI gate ladder.  "before" is *measured* by extracting the
committed (red) blob of ``docs/merge_pr110_evidence/post_report.py`` with
``git show HEAD:`` and running the very same ``ruff check`` the workflow runs;
"after" is read from ``ci_replay.json`` (produced by ``replay_ci.py``).
Panel (b): the mode-2 frequency pin behind the two locally-red tests, from
``mode2_arch_semantics.json`` -- the 15 % gate, the two architecture
measurements, and why strict xfail makes one of them a job failure.

Run:  PYTHONPATH=src python docs/pr185_ci_heal/make_figure.py
"""
from __future__ import annotations

import json
import pathlib
import subprocess
import sys
import tempfile

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.patches import Rectangle  # noqa: E402

ROOT = pathlib.Path(__file__).resolve().parents[2]
HERE = ROOT / "docs" / "pr185_ci_heal"
TARGET = "docs/merge_pr110_evidence/post_report.py"


def ruff_rc(path: pathlib.Path) -> tuple[int, str]:
    proc = subprocess.run([sys.executable, "-m", "ruff", "check", str(path)],
                          capture_output=True, text=True)
    return proc.returncode, (proc.stdout or "") + (proc.stderr or "")


def measure_lint_before_after() -> dict:
    """Return the ruff verdict on the committed blob and on the worktree file."""
    blob = subprocess.run(["git", "show", f"HEAD:{TARGET}"], cwd=ROOT,
                          capture_output=True, text=True)
    with tempfile.TemporaryDirectory() as td:
        p = pathlib.Path(td) / "post_report.py"
        p.write_bytes(blob.stdout.encode())
        before_rc, before_out = ruff_rc(p)
    after_rc, after_out = ruff_rc(ROOT / TARGET)
    rule = ""
    for line in before_out.splitlines():
        if line.strip().startswith(("W", "E", "F")) and " " in line.strip():
            rule = line.strip().split()[0]
            break
    return {"before_rc": before_rc, "after_rc": after_rc, "rule": rule,
            "before_tail": before_out.strip().splitlines()[-3:] if before_out else [],
            "after_tail": after_out.strip().splitlines()[-1:] if after_out else []}


def main() -> int:
    lint = measure_lint_before_after()
    replay = json.loads((HERE / "ci_replay.json").read_text())
    arch = json.loads((HERE / "mode2_arch_semantics.json").read_text())

    steps = [s for s in replay["steps"] if not s.get("skipped")]
    labels = [s["id"] for s in steps]
    n = len(labels)

    fig, (ax_a, ax_b) = plt.subplots(
        1, 2, figsize=(14.5, 6.2), gridspec_kw={"width_ratios": [1.25, 1.0]})

    # ---------------- panel (a): CI ladder, before vs after ----------------
    col_before, col_after = 0, 1
    ax_a.set_title("(a) CI gate ladder: lint short-circuit vs full replay\n"
                   f"arch={replay['arch']}, python={replay['python']}", fontsize=11)
    for row, step in enumerate(steps):
        y = n - 1 - row
        assert step["rc"] == 0, step
        ax_a.add_patch(Rectangle((col_after - 0.45, y - 0.38), 0.9, 0.76,
                                 facecolor="#cdead0", edgecolor="#2f7d32"))
        ax_a.text(col_after, y, "pass", ha="center", va="center", fontsize=9,
                  color="#1b5e20")
        if row == 0:
            ax_a.add_patch(Rectangle((col_before - 0.45, y - 0.38), 0.9, 0.76,
                                     facecolor="#f5c6c2", edgecolor="#b3261e"))
            ax_a.text(col_before, y, "FAIL", ha="center", va="center",
                      fontsize=9, color="#8c1d18", fontweight="bold")
        else:
            ax_a.add_patch(Rectangle((col_before - 0.45, y - 0.38), 0.9, 0.76,
                                     facecolor="#eeeeee", edgecolor="#9e9e9e",
                                     hatch="///"))
            ax_a.text(col_before, y, "not run", ha="center", va="center",
                      fontsize=7.5, color="#616161")
    ax_a.set_xlim(-0.7, 1.7)
    ax_a.set_ylim(-0.9, n - 0.1)
    ax_a.set_xticks([col_before, col_after])
    ax_a.set_xticklabels([
        f"before fix\nruff rc={lint['before_rc']} ({lint['rule'] or 'lint'})",
        f"after fix\nruff rc={lint['after_rc']}, {len(steps)} steps replayed"])
    ax_a.set_yticks(range(n))
    ax_a.set_yticklabels(list(reversed(labels)), fontsize=8.5)
    ax_a.grid(axis="y", color="#dddddd", lw=0.5)
    ax_a.set_axisbelow(True)
    note = ("the three red matrix jobs (3.10/3.11/3.12) are one lint step:\n"
            "`python -m ruff check .` precedes pytest in the same job, so a W292\n"
            "in a docs/*.py file blocks every Python version identically.")
    ax_a.text(-0.62, -0.75, note, fontsize=8, va="bottom", color="#333333")

    # ---------------- panel (b): mode-2 pin vs the 15 % gate ----------------
    gate = arch["gate_pct"]
    ax_b.axhspan(-gate, gate, color="#cdead0", alpha=0.55)
    ax_b.axhline(0.0, color="#555555", lw=0.8)
    for lvl in (-gate, gate):
        ax_b.axhline(lvl, color="#2f7d32", ls="--", lw=1.2)
    xs, ys, cs = [], [], []
    for row in arch["rows"]:
        x86 = row["architecture"].startswith("x86_64")
        xs.append(0 if x86 else 1)
        ys.append(row["error_pct"])
        cs.append("#1f77b4" if x86 else "#d62728")
    seen = set()
    for row, x, y, c in zip(arch["rows"], xs, ys, cs):
        label = row["test"].replace("test_", "")
        if label not in seen:
            seen.add(label)
            ax_b.plot(x, y, "o", ms=13, color=c, label=label)
        ax_b.annotate(f"{y:+.2f} %\n{row['verdict']}", (x, y),
                      textcoords="offset points", xytext=(14, 0),
                      fontsize=8.5, va="center", color=c)
    ax_b.set_xticks([0, 1])
    ax_b.set_xticklabels(["x86_64\n(GitHub runner)\njob GREEN",
                          "loongarch64\n(this container)\njob RED (local only)"])
    ax_b.set_ylabel("mode-2 frequency relative error  [%]")
    ax_b.set_title("(b) same code, same case: the pinned error is architecture-fragile\n"
                   "strict xfail => the gate-meeting side is reported as a failure",
                   fontsize=11)
    ax_b.set_xlim(-0.55, 1.75)
    ax_b.set_ylim(-42, 24)
    ax_b.axhline(gate, color="#2f7d32", ls="--", lw=1.2,
                 label=f"±{gate:.0f} % acceptance gate")
    ax_b.legend(fontsize=8, loc="lower left")
    ax_b.grid(axis="y", color="#dddddd", lw=0.5)
    ax_b.text(1.42, 18.5, "gate met\n(physics still a\ndocumented FAIL:\n-43.5% at R=2 cm)",
              fontsize=7.5, color="#2f7d32", ha="left", va="top")

    fig.suptitle("TensorSPH PR #185 (agent/m3912) CI heal — measured evidence", fontsize=13)
    fig.tight_layout(rect=(0, 0, 1, 0.96))
    out = HERE / "pr185_ci_heal_evidence.png"
    fig.savefig(out, dpi=125)
    print("wrote", out.relative_to(ROOT), lint)
    print("steps:", len(steps), "all rc==0:", all(s["rc"] == 0 for s in steps))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
