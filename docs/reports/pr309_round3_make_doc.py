"""Generate ``docs/mission_m4817_pr309_round3.md`` from the probe JSON.

The prose is a template; every measured number in the document is injected from
``docs/reports/pr309_round3_probe.json``, so the text cannot drift from the code
and no figure is hand-transcribed.

    PYTHONPATH=$PWD/src python docs/reports/pr309_round3_make_doc.py
"""
from __future__ import annotations

import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
PROBE = ROOT / "docs" / "reports" / "pr309_round3_probe.json"
TEMPLATE_MD = ROOT / "docs" / "reports" / "pr309_round3_template.md"
OUT = ROOT / "docs" / "mission_m4817_pr309_round3.md"

def _e(v: float) -> str:
    """``2e-04`` -> ``2e-4``: exponent without the zero padding."""
    return f"{v:.0e}".replace("e-0", "e-").replace("e+0", "e")


def main() -> int:
    p = json.loads(PROBE.read_text())
    a, b, c = (p["control_a_kernel_sum"], p["control_b_wall_pressure"],
               p["control_c_initial_density"])

    rows_a = "\n".join(
        f"| `{k}` | {v:+.6e} |" for k, v in a["bias_by_variant"].items())
    rows_b = "\n".join(
        f"| {r['mass_scale']} | {r['acc_rhs']:.6f} | {r['acc_closed_form']:.6f} "
        f"| {r['relerr']:.3e} | {'green' if r['passes_old_gate'] else '**red**'} "
        f"| {'green' if r['passes_shipped_gate'] else 'red'} |"
        for r in b["rows"])
    c_rows = []
    for label, d in (("square", c["square"]), ("hex", c["hex_same_h"])):
        c_rows.append(
            f"| {label} | {d['bias_raw']:+.3e} | "
            f"{'yes' if d['clamped_to_zero'] else 'no'} | "
            f"{d['p_ratio_raw']:.3e} | "
            + (f"{c['square']['p_ratio_calibrated']:.2e} |" if label == "square"
               else "not applicable (hex mass differs) |"))
    rows_c = "\n".join(c_rows)

    rows_b_vals = b["rows"]
    text = TEMPLATE_MD.read_text().format(
        main=p["main"],
        dx=a["dx"], hdx=a["h_over_dx"], n_pool=a["n_pool"],
        n_interior=a["n_interior"],
        rows_a=rows_a,
        spread=a["spread_max_minus_min"], spread_gate=a["spread_over_gate"],
        inside="yes" if a["all_inside_gate"] else "no",
        sign="yes" if a["all_same_sign"] else "no",
        old_pin=abs(a["old_pin"]), over_pin=a["measured_over_old_pin"],
        b32=a["bias_by_variant"]["fp32_as_shipped"],
        rtol=_e(b["shipped_gate"]["rel_tol"]), atol=_e(b["shipped_gate"]["abs_tol"]),
        or_tol=_e(b["old_gate"]["rel_tol"]), or_atol=_e(b["old_gate"]["abs_tol"]),
        rows_b=rows_b, head=rows_b_vals[0]["head_rho_g_dx_Pa"],
        margin=b["shipped_gate_margin"],
        rows_c=rows_c, hex_pa=c["hex_same_h"]["abs_p_false_Pa"],
        suite=SUITE_SUMMARY, before=BEFORE_SUMMARY,
        a0=rows_b_vals[0]["acc_rhs"], e0=rows_b_vals[0]["acc_closed_form"],
        r0=rows_b_vals[0]["relerr"],
        a3=rows_b_vals[1]["acc_rhs"], e3=rows_b_vals[1]["acc_closed_form"],
        r3=rows_b_vals[1]["relerr"],
        cal=c["square"]["bias_calibrated"],
    )
    OUT.write_text(text)
    print(f"wrote {OUT} ({len(text.splitlines())} lines)")
    return 0


def _summary(name: str) -> str:
    """Last ``pytest -q`` summary line of a logged run, verbatim."""
    log = ROOT / "docs" / "reports" / name
    if not log.exists():
        return "not run in this worktree"
    lines = [ln.strip() for ln in log.read_text().splitlines() if ln.strip()]
    tail = [ln for ln in lines if (" passed" in ln or " failed" in ln)]
    return tail[-1] if tail else "log has no summary line"


SUITE_SUMMARY = _summary("m4817_full_suite_after.log")
BEFORE_SUMMARY = _summary("m4817_full_suite_before.log")

if __name__ == "__main__":
    raise SystemExit(main())
