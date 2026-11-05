"""PR #110 (agent/m3265) three-way merge evidence builder.

Regenerates, from the *merged* work tree:

* ``evidence.json``  -- machine-readable reconciliation table
  (capillary-relaxation convergence records + surface-tension amplitude for the
  three formulations that collide in the ``tests/test_laplace_pressure.py``
  conflict);
* ``pr110_merge_evidence.png`` -- two-panel figure used by the merge report.

All numbers are produced by running the repo code; nothing is hand-copied.

Run:  PYTHONPATH=$PWD/src python3 docs/merge_pr110_evidence/build_evidence.py
"""
from __future__ import annotations

import json
import pathlib
import sys

HERE = pathlib.Path(__file__).resolve().parent
ROOT = HERE.parent.parent
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "tests"))

import matplotlib  # noqa: E402  (after sys.path bootstrap)

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

import test_laplace_pressure as tlp  # noqa: E402

CONV = ROOT / "benchmarks/laplace_law/convergence"
FILES = ["relax_dx0.0015.json", "relax_dx0.0020.json", "relax_dx0.0025.json"]

# t_over_t_sigma floor required by tests/test_laplace_law.py::
# test_committed_verdicts_are_self_consistent
RELAX_FLOOR = 0.5


def convergence_records() -> list[dict]:
    rows = []
    for name in FILES:
        d = json.loads((CONV / name).read_text())
        s, lap = d["setup"], d["laplace"]
        rows.append(
            {
                "file": name,
                "dx_m": s["dx_m"],
                "steps": s["steps"],
                "t_over_t_sigma": s["t_over_t_sigma"],
                "lattice": s["lattice"],
                "relaxed_enough": s["t_over_t_sigma"] > RELAX_FLOOR,
                "delta_p_measured_Pa": lap["delta_p_measured_Pa"],
                "delta_p_theory_Pa": lap["delta_p_theory_Pa"],
                "rel_error_pct": lap["rel_error_pct"],
                "within_15pct": lap["within_15pct"],
            }
        )
    return rows


def amplitude_records() -> dict:
    return {
        "consistent_fit_quadratic_default": round(tlp._half_force_ratio(), 4),
        "consistent_mls_curvature": round(
            tlp._half_force_ratio(csf_curvature="mls"), 4
        ),
        "legacy_neighbour_count": round(tlp._half_force_ratio(csf_scheme="legacy"), 4),
        "nominal": 1.0,
    }


def plot(conv: list[dict], amp: dict) -> pathlib.Path:
    fig, (ax0, ax1) = plt.subplots(1, 2, figsize=(11.0, 4.2))

    # -- panel A: relaxation coverage of the committed convergence records ----
    labels = [f"dx={r['dx_m'] * 1e3:.1f} mm" for r in conv]
    cover = [r["t_over_t_sigma"] for r in conv]
    bars = ax0.bar(labels, cover, color=["#2b6cb0", "#3182ce", "#63b3ed"], width=0.55)
    ax0.axhline(RELAX_FLOOR, color="#c53030", ls="--", lw=1.6,
                label=f"gate t/t_sigma > {RELAX_FLOOR}")
    for b, r in zip(bars, conv):
        ax0.text(b.get_x() + b.get_width() / 2, b.get_height() + 0.02,
                 f"{r['steps']} steps\n{r['t_over_t_sigma']:.3f}",
                 ha="center", va="bottom", fontsize=8)
    ax0.set_ylabel(r"$t / t_\sigma$")
    ax0.set_title("(a) capillary relaxation coverage of committed records\n"
                  "(merged tree; both conflict sides regenerated these)", fontsize=9)
    ax0.set_ylim(0, max(cover) * 1.45)
    ax0.legend(fontsize=8, loc="upper left")
    ax0.grid(axis="y", alpha=0.3)

    # -- panel B: |F_cut| / (2 sigma) for the three formulations -------------
    names = ["legacy\n(neighbour count)", "consistent +\nquadratic fit\n(default)",
             "consistent +\nMLS curvature"]
    vals = [amp["legacy_neighbour_count"],
            amp["consistent_fit_quadratic_default"],
            amp["consistent_mls_curvature"]]
    colors = ["#a0aec0", "#2f855a", "#d69e2e"]
    bars = ax1.bar(names, vals, color=colors, width=0.55)
    ax1.axhline(1.0, color="#2b6cb0", ls="-", lw=1.2, label="nominal $\\sigma$")
    ax1.axhspan(0.85, 1.15, color="#2b6cb0", alpha=0.10,
                label="15 % acceptance window")
    for b, v in zip(bars, vals):
        ax1.text(b.get_x() + b.get_width() / 2, v + 0.02, f"{v:.4f}",
                 ha="center", va="bottom", fontsize=9)
    ax1.set_ylabel(r"$|F_{cut}| / (2\sigma)$")
    ax1.set_ylim(0, 1.25)
    ax1.set_title("(b) surface-tension amplitude at $R/h = 6.15$, square lattice\n"
                  "(the quantity the conflict pinned; measured, not assumed)",
                  fontsize=9)
    ax1.legend(fontsize=8, loc="upper left")
    ax1.grid(axis="y", alpha=0.3)

    fig.suptitle("TensorSPH PR #110 merge (m3912): reconciliation evidence", fontsize=11)
    fig.tight_layout(rect=(0, 0, 1, 0.94))
    out = HERE / "pr110_merge_evidence.png"
    fig.savefig(out, dpi=150)
    plt.close(fig)
    return out


def main() -> None:
    conv, amp = convergence_records(), amplitude_records()
    payload = {
        "mission": "TensorSPH#110 (m3912)",
        "pr": 110,
        "branch_merged": "agent/m3265",
        "convergence_records": conv,
        "surface_tension_amplitude": amp,
        "relaxation_gate": RELAX_FLOOR,
    }
    (HERE / "evidence.json").write_text(json.dumps(payload, indent=2) + "\n")
    png = plot(conv, amp)
    print(json.dumps(payload, indent=2))
    print("figure:", png)


if __name__ == "__main__":
    main()
