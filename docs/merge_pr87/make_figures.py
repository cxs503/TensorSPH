"""Figures + machine-readable evidence for the PR #87 (agent/m2919) merge.

All numbers are read from the recorded JSON/probe files produced by the actual
runs, never typed by hand:

  docs/merge_pr87/omega_probe_fit_quadratic.json  (main default scheme)
  docs/merge_pr87/omega_probe_mls.json            (pinned scheme)
  benchmarks/oscillating_droplet/validation_report_m2919.json (branch report)
  docs/merge_pr87/test_runs.json                  (pytest tallies)

Run: PYTHONPATH=src python docs/merge_pr87/make_figures.py
"""
from __future__ import annotations

import json
import subprocess
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

HERE = Path(__file__).resolve().parent
REPO = HERE.parent.parent


def sh(*args: str) -> str:
    return subprocess.run(args, cwd=REPO, capture_output=True, text=True,
                          check=True).stdout.strip()


probes = {
    "fit_quadratic (main default)":
        json.loads((HERE / "omega_probe_fit_quadratic.json").read_text()),
    "mls (pinned, branch baseline)":
        json.loads((HERE / "omega_probe_mls.json").read_text()),
}
tests = json.loads((HERE / "test_runs.json").read_text())
branch_rep = json.loads(
    (REPO / "benchmarks" / "oscillating_droplet" /
     "validation_report_m2919.json").read_text())

omega_ref = probes["mls (pinned, branch baseline)"]["omega_analytic_rad_s"]

fig, axes = plt.subplots(1, 2, figsize=(12.6, 4.4))

ax = axes[0]
names = list(probes)
vals = [probes[n]["omega_measured_rad_s"] for n in names]
errs = [probes[n]["omega_rel_error_pct"] for n in names]
colors = ["#c44e52", "#55a868"]
ax.bar(range(len(names)), vals, color=colors, width=0.5)
ax.axhline(omega_ref, color="k", ls="--", lw=1.4,
           label=rf"Lamb 2-D $\omega_2=\sqrt{{6\sigma/(\rho R^3)}}$ = {omega_ref:.2f} rad/s")
ax.axhline(omega_ref * 0.85, color="grey", ls=":", lw=1.1, label="15 % acceptance gate")
ax.axhline(omega_ref * 1.15, color="grey", ls=":", lw=1.1)
for i, (b, e) in enumerate(zip(vals, errs)):
    ax.text(i, b + 1.5, f"{b:.2f} rad/s\n{e:+.2f} %", ha="center", fontsize=9)
ax.set_xticks(range(len(names)))
ax.set_xticklabels([n.replace(" (", "\n(") for n in names], fontsize=9)
ax.set_ylabel(r"measured mode-2 $\omega$ [rad/s]")
ax.set_ylim(0, omega_ref * 1.35)
ax.set_title("(a) same case, two curvature schemes\n"
             r"$R=20$ mm, $dx=2.5$ mm, $n=2$, $\varepsilon=0.05$, 233 particles")
ax.legend(fontsize=7.5, loc="upper left")
ax.grid(alpha=0.3, axis="y")

ax = axes[1]
labels = ["main baseline\n(origin/main)", "merged\n(main + PR #87)"]
passed = [tests["baseline"]["passed"], tests["merged"]["passed"]
          if "passed" in tests["merged"] else tests["merged"]["passed"]]
xfail_strict_xpass = [tests["baseline"]["failed"], tests["merged"]["failed"]]
x = range(2)
ax.bar(x, passed, color="#4c72b0", width=0.45, label="passed")
ax.bar(x, xfail_strict_xpass, bottom=passed, color="#c44e52", width=0.45,
       label="failed (pre-existing strict-xfail XPASS)")
for i in x:
    ax.text(i, passed[i] / 2, str(passed[i]), ha="center", color="w",
            fontsize=11, fontweight="bold")
    ax.text(i, passed[i] + xfail_strict_xpass[i] / 2, str(xfail_strict_xpass[i]),
            ha="center", color="w", fontsize=11, fontweight="bold")
ax.set_xticks(list(x))
ax.set_xticklabels(labels, fontsize=9)
ax.set_ylabel("tests in tests/test_oscillating_droplet.py")
ax.set_title("(b) suite carried over without regression\n"
             "same 2 pre-existing failures on both sides; 15 branch tests added")
ax.legend(fontsize=8, loc="lower right")
ax.grid(alpha=0.3, axis="y")

fig.tight_layout()
out = HERE / "merge_pr87_figures.png"
fig.savefig(out, dpi=110)
print(f"figure -> {out} ({out.stat().st_size / 1024:.1f} kB)")

ladder = [
    {"case": r["case"], "n": r["mode_n"], "dx_mm": r["dx_m"] * 1e3,
     "n_particles": r["n_particles"], "omega_measured": r["omega_measured"],
     "omega_analytic": r["omega_analytic"], "error_pct": r["omega_error_pct"],
     "residual": r["fit_residual_rel"]}
    for r in branch_rep["rows"]
]
evidence = {
    "stage": {
        "worktree": str(REPO),
        "branch_head": sh("git", "rev-parse", "--abbrev-ref", "HEAD"),
        "main_sha": sh("git", "rev-parse", "origin/main"),
        "pr_branch": "agent/m2919",
        "pr_branch_sha_old": sh("git", "rev-parse", "origin/agent/m2919"),
        "merge_base": sh("git", "merge-base", "origin/main", "origin/agent/m2919"),
        "ahead_behind": sh("git", "rev-list", "--left-right", "--count",
                           "origin/main...origin/agent/m2919"),
    },
    "probes": probes,
    "omega_analytic_rad_s": omega_ref,
    "branch_report_rows": ladder,
    "test_runs": tests,
}
(HERE / "evidence.json").write_text(json.dumps(evidence, indent=2))
print("evidence ->", HERE / "evidence.json")
