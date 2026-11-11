"""Acceptance criteria of the WCSPH dam-break / sloshing chain.

Everything here is a pure function of measured numbers: the benchmark script
measures, this module decides.  Keeping the decision out of the measurement is
what makes the verdicts auditable -- a criterion cannot be re-worded to fit the
number it is looking at, and the tests can feed synthetic values to check that
each gate flips exactly where it says it does.

Criteria (from the campaign node, verbatim numbers):

=================  ============================================================
id                 requirement
=================  ============================================================
``a-momentum``     single particle pair: momentum error <= 1e-10
``a-energy``       single particle pair: energy error <= 1 %
``b-2d-front``     2-D dam break vs Martin & Moyce: front error <= 10 %
                   at t = 0.1 / 0.2 / 0.3 s
``c-3d-front``     3-D dam break vs experiment/reference: <= 15 %
``d-sloshing``     2-D sloshing: peak wall pressure vs experiment <= 15 %
                   at 0.5 / 1.0 / 1.5 Hz
``e-convergence``  particle spacing halved: L2 order >= 1.0
``f-tests``        pytest >= 20 cases, coverage >= 85 %
=================  ============================================================

Verdict vocabulary: ``pass`` / ``fail`` / ``withheld``.  ``withheld`` is used
when the *reference* is not trustworthy enough to arbitrate the limit (the
repository's own audit of the bundled Martin & Moyce table is the precedent:
``benchmarks/dam_break/reference_audit.json`` grades it WITHHELD).  A withheld
gate is never reported as a pass.
"""
from __future__ import annotations

import math

import torch

MOMENTUM_LIMIT = 1.0e-10
ENERGY_LIMIT = 0.01
FRONT_2D_LIMIT = 0.10
FRONT_3D_LIMIT = 0.15
SLOSHING_LIMIT = 0.15
CONVERGENCE_MIN_ORDER = 1.0
MIN_TESTS = 20
MIN_COVERAGE = 0.85


def gate(criterion: str, measured, limit, reference: str | None = None,
         rule: str = "<=", note: str = "") -> dict:
    """One machine-readable acceptance gate."""
    entry = {"criterion": criterion, "measured": measured, "limit": limit,
             "rule": rule, "reference": reference, "note": note}
    if measured is None:
        entry["verdict"] = "withheld"
        return entry
    value = float(measured)
    if not math.isfinite(value):
        entry["verdict"] = "withheld"
        return entry
    ok = value <= limit if rule == "<=" else value >= limit
    entry["verdict"] = "pass" if ok else "fail"
    return entry


def l2_profile_error(pos_a: torch.Tensor, mass_a: torch.Tensor, rho_a: torch.Tensor,
                     pos_b: torch.Tensor, mass_b: torch.Tensor, rho_b: torch.Tensor,
                     x_edges: torch.Tensor, dim: int = 2) -> float:
    """Relative L2 distance between the depth profiles of two resolutions.

    The profile is the volume per unit length accumulated in each ``x`` bin,
    ``h(x) = sum_{i in bin} (m_i / rho_i) / dx_bin`` -- a quantity that both
    resolutions estimate without requiring a particle-to-particle match.  For
    the 3-D case the volume is per unit area, i.e. a mean depth.
    """
    slope = float(x_edges[1] - x_edges[0])

    def profile(pos, mass, rho):
        vol = (mass / rho)
        idx = torch.bucketize(pos[:, 0].contiguous(), x_edges) - 1
        idx = idx.clamp(0, len(x_edges) - 2)
        acc = torch.zeros(len(x_edges) - 1, dtype=vol.dtype)
        acc = acc.index_add(0, idx, vol)
        return acc / slope

    ha = profile(pos_a, mass_a, rho_a)
    hb = profile(pos_b, mass_b, rho_b)
    num = float(((ha - hb) ** 2).sum())
    den = float((hb ** 2).sum())
    del dim
    return math.sqrt(num / den) if den > 0 else float("inf")


def convergence_order(error_coarse: float, error_fine: float,
                      dx_coarse: float, dx_fine: float) -> float:
    """Observed order in ``error ~ dx**p`` (``nan`` if it cannot be formed)."""
    if error_fine <= 0.0 or dx_coarse <= 0.0 or dx_fine <= 0.0:
        return float("nan")
    if error_coarse <= 0.0:
        return float("nan")
    return math.log(error_coarse / error_fine) / math.log(dx_coarse / dx_fine)


def relative_error(sim: float, ref: float) -> float | None:
    """Signed relative error; ``None`` when the reference is zero."""
    if ref is None or sim is None or ref == 0.0:
        return None
    return (sim - ref) / ref


def summarise(gates: list[dict]) -> dict:
    counts = {"pass": 0, "fail": 0, "withheld": 0}
    for g in gates:
        counts[g["verdict"]] = counts.get(g["verdict"], 0) + 1
    if counts["fail"]:
        verdict = "fail"
    elif counts["withheld"]:
        verdict = "withheld"
    else:
        verdict = "pass"
    return {"verdict": verdict, **counts}


def evaluate(payload: dict) -> dict:
    """Turn a measurement payload into the criterion table + overall verdict.

    ``payload`` keys are optional; a missing key yields a ``withheld`` row
    rather than an error, so a partially finished run still produces a fully
    auditable table.
    """
    gates: list[dict] = []
    pair = payload.get("pair") or {}
    gates.append(gate("a-momentum", pair.get("momentum_drift"), MOMENTUM_LIMIT,
                      "single particle pair, free space", "<=",
                      "max |p(t) - p(0)| / |p(0)|, alpha_av = 0"))
    gates.append(gate("a-energy", pair.get("energy_drift"), ENERGY_LIMIT,
                      "single particle pair, free space", "<=",
                      "max |E(t) - E(0)| / |E(0)|, alpha_av = 0"))

    front2d = payload.get("front_2d") or {}
    gates.append(gate("b-2d-front", front2d.get("max_relative_error"),
                      FRONT_2D_LIMIT, front2d.get("reference"),
                      "<=", "t = 0.1/0.2/0.3 s, Z/a measured from the column edge"))
    gates.append(gate("b-2d-front-mae", front2d.get("mae_over_curve"),
                      FRONT_2D_LIMIT, front2d.get("reference"), "<=",
                      "mean |Z/a error| over every reference sample inside the run"))

    front3d = payload.get("front_3d") or {}
    gates.append(gate("c-3d-front", front3d.get("max_relative_error"),
                      FRONT_3D_LIMIT, front3d.get("reference"), "<=",
                      "front at the same non-dimensional times as the 2-D rung"))

    slosh = payload.get("sloshing") or {}
    for row in slosh.get("rows", []):
        gates.append(gate(f"d-sloshing-{row['frequency']:.1f}Hz",
                          row.get("relative_error"), SLOSHING_LIMIT,
                          row.get("reference"), "<=",
                          "peak wall pressure at the still-water level"))

    conv = payload.get("convergence") or {}
    for pair_row in conv.get("pairs", []):
        gates.append(gate(f"e-convergence-{pair_row['label']}",
                          pair_row.get("order"), CONVERGENCE_MIN_ORDER,
                          "L2 depth-profile error, self-convergence", ">=",
                          "particle spacing halved"))

    tests = payload.get("tests") or {}
    gates.append(gate("f-test-count", tests.get("collected"), MIN_TESTS,
                      "pytest --collect-only of tests/test_wcsph_*.py", ">=", ""))
    gates.append(gate("f-coverage", tests.get("coverage"), MIN_COVERAGE,
                      "coverage of tensorsph.wcsph", ">=", ""))

    return {"gates": gates, "summary": summarise(gates)}


__all__ = ["ENERGY_LIMIT", "FRONT_2D_LIMIT", "FRONT_3D_LIMIT", "MOMENTUM_LIMIT",
           "SLOSHING_LIMIT", "convergence_order", "evaluate", "gate",
           "l2_profile_error", "relative_error", "summarise"]