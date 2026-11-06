"""Regenerate the CI-repair evidence for m4814 (TensorSPH `main` red at d61b6c26).

Everything in ``evidence.json`` / ``ci_repair_matrix.png`` is produced here, so no
number in the write-up is pasted by hand.  The "before" side is measured on a
pristine snapshot of ``HEAD`` extracted with ``git archive`` (no worktree, no
``.git`` write), the "after" side on the current tree.

Run::

    PYTHONPATH=src python docs/ci_repair_m4814/make_evidence.py
"""
from __future__ import annotations

import importlib.util
import json
import os
import re
import shutil
import subprocess
import sys
import tarfile
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
OUT = Path(__file__).resolve().parent
SNAP = Path(tempfile.gettempdir()) / "m4814_prefix_snapshot"
ORDERS = {
    "kernel first (CI order)": ("tests/test_kernel_sum_consistency.py",
                                "tests/test_oscillating_droplet_gauges.py"),
    "gauges first (reversed)": ("tests/test_oscillating_droplet_gauges.py",
                                "tests/test_kernel_sum_consistency.py"),
}
SUMMARY = re.compile(r"(\d+) (passed|failed|error|xfailed)")


def snapshot_head() -> Path:
    """Extract ``HEAD`` into a scratch dir -- read-only w.r.t. this worktree."""
    if SNAP.exists():
        shutil.rmtree(SNAP)
    SNAP.mkdir(parents=True)
    tar = subprocess.run(["git", "archive", "HEAD"], cwd=ROOT,
                         capture_output=True, check=True)
    import io
    with tarfile.open(fileobj=io.BytesIO(tar.stdout)) as tf:
        tf.extractall(SNAP)
    return SNAP


def ruff_errors(root: Path) -> dict:
    proc = subprocess.run([sys.executable, "-m", "ruff", "check", ".",
                           "--output-format=concise"],
                          cwd=root, capture_output=True, text=True)
    counts: dict[str, int] = {}
    for line in proc.stdout.splitlines():
        m = re.match(r"^([^:]+):\d+:\d+: ([EWF]\d+)", line)
        if m:
            counts[m.group(1)] = counts.get(m.group(1), 0) + 1
    return {"total": sum(counts.values()), "by_file": counts,
            "clean": proc.returncode == 0}


def pytest_outcome(root: Path, files: tuple[str, str]) -> dict:
    proc = subprocess.run([sys.executable, "-m", "pytest", "-q", *files],
                          cwd=root, capture_output=True, text=True,
                          env={**os.environ, "PYTHONPATH": str(ROOT / "src")})
    tail = proc.stdout.strip().splitlines()[-1] if proc.stdout else ""
    counts: dict[str, int] = {}
    for n, word in SUMMARY.findall(tail):
        counts[word] = counts.get(word, 0) + int(n)
    return {"passed": counts.get("passed", 0), "failed": counts.get("failed", 0),
            "error": counts.get("error", 0), "exit": proc.returncode,
            "summary_line": tail}


def render_doc(bench_dir: Path, tag: str) -> tuple[str, int]:
    """sha256 of the markdown ``gauge_audit.make_doc`` renders for this tree."""
    import hashlib
    sys.path.insert(0, str(bench_dir))
    spec = importlib.util.spec_from_file_location(f"gauge_audit_{tag}",
                                                  bench_dir / "gauge_audit.py")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    audit = json.loads((bench_dir / "gauge_audit.json").read_text())
    out = Path(tempfile.mkdtemp()) / "doc.md"
    module.make_doc(audit, "x.png", out)
    data = out.read_bytes()
    return hashlib.sha256(data).hexdigest(), len(data)


def figure(ev: dict, path: Path) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(1, 2, figsize=(12.0, 4.4), dpi=140)

    ax = axes[0]
    orders = list(ORDERS)
    labels = ["before fix (HEAD d61b6c26)", "after fix (this branch)"]
    data = [[[ev["before"][o]["passed"], ev["before"][o]["failed"],
              ev["before"][o]["error"]],
             [ev["after"][o]["passed"], ev["after"][o]["failed"],
              ev["after"][o]["error"]]] for o in orders]
    width = 0.35
    xs = range(len(orders))
    for k, lab in enumerate(labels):
        vals = [data[i][k][0] for i in xs]
        ax.bar([x + (k - 0.5) * width for x in xs], vals, width,
               label=f"{lab}: passed", color=["tab:red", "tab:green"][k],
               alpha=0.85)
        bad = [data[i][k][1] + data[i][k][2] for i in xs]
        ax.bar([x + (k - 0.5) * width for x in xs], bad, width, bottom=vals,
               color="k", hatch="//", alpha=0.6)
        for x, v, b in zip(xs, vals, bad):
            ax.text(x + (k - 0.5) * width, v + b + 0.4,
                    f"{v}✓/{b}✗", ha="center", fontsize=8)
    ax.set_xticks(list(xs))
    ax.set_xticklabels(orders, fontsize=8)
    ax.set_ylabel("tests in the two files")
    ax.set_title("sys.modules['run'] collision: outcome vs collection order\n"
                 "black hatching = failed/errored", fontsize=9)
    ax.legend(fontsize=7)
    ax.grid(alpha=0.3, axis="y")

    ax = axes[1]
    files = sorted(ev["before_ruff"]["by_file"])
    before = [ev["before_ruff"]["by_file"].get(f, 0) for f in files]
    ax.barh(files, before, color="tab:red", alpha=0.85,
            label=f"HEAD: {ev['before_ruff']['total']} errors")
    ax.set_xlabel("ruff errors (E/F/W)")
    ax.set_title("lint step (runs before pytest) on HEAD vs this branch\n"
                 f"now: {ev['after_ruff']['total']} errors, "
                 f"clean={ev['after_ruff']['clean']}", fontsize=9)
    ax.legend(fontsize=7)
    ax.tick_params(labelsize=7)
    ax.grid(alpha=0.3, axis="x")

    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)


def main() -> int:
    ev: dict = {"repo": str(ROOT), "head": subprocess.run(
        ["git", "rev-parse", "--short", "HEAD"], cwd=ROOT,
        capture_output=True, text=True).stdout.strip()}

    snap = snapshot_head()
    ev["before_ruff"] = ruff_errors(snap)
    ev["after_ruff"] = ruff_errors(ROOT)
    ev["before"] = {o: pytest_outcome(snap, f) for o, f in ORDERS.items()}
    ev["after"] = {o: pytest_outcome(ROOT, f) for o, f in ORDERS.items()}
    before_doc = render_doc(snap / "benchmarks" / "oscillating_droplet", "snap")
    after_doc = render_doc(ROOT / "benchmarks" / "oscillating_droplet", "cur")
    ev["doc_render"] = {
        "sha256_before_reflow": before_doc[0], "bytes_before_reflow": before_doc[1],
        "sha256_after_fix": after_doc[0], "bytes_after_fix": after_doc[1],
        "identical": before_doc[0] == after_doc[0],
    }
    ev["full_suite"] = {}
    for name, path in (("after", OUT / "pytest_full.log"),):
        if Path(path).exists():
            lines = [ln for ln in Path(path).read_text().splitlines()
                     if re.search(r"\d+ passed", ln)]
            ev["full_suite"][name] = lines[-1].strip() if lines else ""

    (OUT / "evidence.json").write_text(json.dumps(ev, indent=1,
                                                  ensure_ascii=False))
    figure(ev, OUT / "ci_repair_matrix.png")
    print(json.dumps({k: ev[k] for k in ("head", "before_ruff", "after_ruff")},
                     indent=1))
    print(json.dumps({"before": ev["before"], "after": ev["after"]}, indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
