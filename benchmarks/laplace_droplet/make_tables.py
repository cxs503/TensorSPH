"""Render the Laplace-verification results.json into markdown tables.

Every number in docs/benchmarks/laplace-droplet.md comes from this script, so
nothing is transcribed by hand.

    PYTHONPATH=src python benchmarks/laplace_droplet/make_tables.py \
        --results benchmarks/laplace_droplet/results.json \
        --doc docs/benchmarks/laplace-droplet.md
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path


def fmt(value, spec="{:.3f}"):
    if isinstance(value, float):
        return spec.format(value)
    return str(value)


def render(results: dict) -> str:
    out = []
    cases = results["cases"]

    out.append("### Case table (auto-generated from `benchmarks/laplace_droplet/results.json`)\n")
    out.append("`R/dx` is the only governing resolution parameter; `sigma` enters linearly "
               "(verified: the ratio is identical at sigma = 0.072 / 0.72 / 7.2 N/m).\n")
    out.append("| R/dx | lattice | jitter | N | N_delta = int|grad c|dr | "
               "kappa_delta R (model) | kappa>0 frac. | <n.rhat>_delta | "
               "dp_model/dp_theory | error | within 15 % |")
    out.append("|---|---|---|---|---|---|---|---|---|---|---|")
    for c in cases:
        out.append(
            f"| {c['resolution_R_over_dx']:.0f} | {c['lattice']} | {c['jitter']:g} | "
            f"{c['n_particles']} | {c['N_delta']:.3f} | "
            f"{c['kappa_delta_weighted_times_R']:+.3f} | "
            f"{c['kappa_model_positive_fraction']:.2f} | "
            f"{c['normal_alignment_delta_weighted']:+.3f} | "
            f"{c['dp_ratio']:+.3f} | {c['relative_error']:.1%} | "
            f"{'yes' if c['within_15pct'] else '**no**'} |")

    out.append("\n### Error decomposition (auto-generated)\n")
    out.append("The same CSF form evaluated with pieces replaced by exact values. "
               "`B` and `C` are instrument checks: they must not depend on the model at all.\n")
    out.append("| R/dx | lattice | A: exact kappa, model normal | "
               "B: exact kappa + exact normal | C: exact operators + unit delta_s | "
               "model as merged |")
    out.append("|---|---|---|---|---|---|")
    for c in cases:
        out.append(
            f"| {c['resolution_R_over_dx']:.0f} | {c['lattice']} jitter={c['jitter']:g} | "
            f"{c['instrument_A_exact_kappa_model_normal_ratio']:+.3f} | "
            f"{c['instrument_B_exact_kappa_exact_normal_ratio']:+.3f} "
            f"(N_delta={c['N_delta']:.3f}) | "
            f"{c['instrument_C_exact_operators_ratio']:+.3f} | "
            f"{c['dp_ratio']:+.3f} |")

    v = results["verdict"]
    out.append("\n### Verdict (auto-generated)\n")
    out.append(f"- criterion: {v['criterion']}")
    out.append(f"- status: **{v['status']}**")
    out.append(f"- worst relative error over the {v['n_cases']} reference cases: "
               f"{v['max_relative_error']:.1%}")
    out.append(f"- cases inside tolerance: {v['n_within_tolerance']}/{v['n_cases']}")
    out.append(f"- monotone convergence under refinement: "
               f"{'yes' if v['converges_under_refinement'] else '**no**'}")
    ic = v["instrument_check"]
    out.append(f"- instrument B error against N_delta: {ic['max_deviation_B_vs_N_delta']:.2e}")
    out.append(f"- instrument C error against the Laplace law: {ic['max_deviation_C_vs_1']:.2e}")

    r = results["resolvability_audit"]
    out.append("\n### Why the jump is not read from the density field (auto-generated)\n")
    out.append("| quantity | value |")
    out.append("|---|---|")
    out.append(f"| dp = sigma/R [Pa] | {r['dp_theory_Pa']:.4f} |")
    out.append(f"| bulk modulus B = rho0 c0^2/gamma [Pa] | {r['bulk_modulus_Pa']:.1f} |")
    out.append(f"| dp/B | {r['dp_over_bulk']:.3e} |")
    out.append(f"| density signal (rho/rho0 - 1) | {r['density_ratio_signal']:.3e} |")
    out.append(f"| summation-density noise (reference) | {r['density_noise_reference']:.1e} |")
    out.append(f"| signal/noise | {r['signal_to_noise']:.3f} |")
    out.append(f"| verdict | {r['verdict']} |")
    return "\n".join(out) + "\n"


def render_acceptance(results: dict) -> str:
    """Per-quantity acceptance table: reference, measured, error, 15 %/3 % verdict."""
    out = []
    out.append("| quantity | case | reference | measured | relative error | < 15 % | < 3 % |")
    out.append("|---|---|---|---|---|---|---|")
    for c in results["cases"]:
        if c["jitter"] != 0.0 or c["sigma_N_per_m"] != 0.072:
            continue
        if c["radius_m"] != 0.1:
            continue
        out.append(f"| `dp` | {c['lattice']}, R/dx={c['resolution_R_over_dx']:.0f} | "
                   f"{c['dp_theory_Pa']:.4f} Pa | {c['dp_model_Pa']:.4f} Pa | "
                   f"{c['dp_ratio'] - 1.0:+.1%} | "
                   f"{'yes' if c['relative_error'] <= 0.15 else 'no'} | "
                   f"{'yes' if c['relative_error'] <= 0.03 else 'no'} |")
    deltas = [c["N_delta"] for c in results["cases"]]
    out.append(f"| `N_delta` = int abs(grad c) dr | all cases | 1.000 | "
               f"{min(deltas):.3f}-{max(deltas):.3f} | "
               f"{min(deltas) - 1.0:+.1%} to {max(deltas) - 1.0:+.1%} | no | no |")
    kaps = [c["kappa_delta_weighted_times_R"] for c in results["cases"]]
    out.append(f"| `<kappa>_delta R` | all cases | 1.000 | "
               f"{min(kaps):+.2f} to {max(kaps):+.2f} | sign and magnitude wrong | no | no |")
    return "\n".join(out) + "\n"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--results", default="benchmarks/laplace_droplet/results.json")
    ap.add_argument("--doc", default="docs/benchmarks/laplace-droplet.md")
    ap.add_argument("--out", default="benchmarks/laplace_droplet/results_tables.md")
    args = ap.parse_args()
    results = json.loads(Path(args.results).read_text())
    table = render(results)
    accept = render_acceptance(results)
    Path(args.out).write_text(table + "\n" + accept)

    doc = Path(args.doc)
    if doc.exists():
        text = doc.read_text()
        for begin, end, block in (("<!-- RESULTS:BEGIN -->", "<!-- RESULTS:END -->", table),
                                  ("<!-- ACCEPT:BEGIN -->", "<!-- ACCEPT:END -->", accept)):
            if begin in text and end in text:
                head, rest = text.split(begin, 1)
                _, tail = rest.split(end, 1)
                text = head + begin + "\n\n" + block + "\n" + end + tail
        doc.write_text(text)
        print(f"injected tables into {doc}")
    print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
