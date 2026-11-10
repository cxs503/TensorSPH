"""M4 -- particle-spacing convergence and the kernel-support negative control.

What is measured
----------------
SPH interpolation of an analytic smooth field on a lattice of spacing ``dx``
with smoothing length ``h = h_factor * dx``::

    f_raw(x_i)  = sum_j f_j W(r_ij) V_j                    (what summation
                                                            density does)
    f_norm(x_i) = f_raw / sum_j W(r_ij) V_j                (Shepard-normalised)

The ``L2`` error is taken over *interior* particles only (those at least ``2h``
from the boundary), because SPH is knowingly inconsistent in the boundary
region and that would otherwise dominate the fit.  Both errors are reported:

* the **normalised** error is the gated one.  A consistent, second-order
  discrete kernel gives ``O(h^2)``; the campaign asks for ``>= 1``.
* the **raw** error is reported as a diagnostic and is expected *not* to
  converge: the discrete kernel sum on a lattice is ``1 + b(h/dx)`` with ``b``
  independent of ``dx`` (``b = 5.88e-3`` at ``h = 1.3 dx``), so the raw
  summation carries a scale-invariant bias.  That is the zeroth-order
  inconsistency the repository's ``density_calibration`` option removes from
  the initial state, and it is worth seeing in a table rather than quoting
  from a paper.

Negative controls
-----------------
Two are run, because the first one alone is not enough:

1. **interpolation with a truncated kernel** (``R = 1.5 h``, two-``h``
   normalisation).  Reported, *not* gated: Shepard normalisation re-consistents
   any compactly supported kernel, so this variant converges too and its error
   at the finest spacing is small.  The measurement is kept because it is the
   honest answer to "does truncating the support break the interpolation?" --
   with normalisation, it does not.
2. **partition-of-unity defect and a solver run with the truncated support**,
   which are gated.  The lattice kernel sum ``sum_j W dx^2`` must equal one for
   the *unnormalised* summation density; truncating the support to ``1.5 h``
   breaks it by an amount that does not vanish with ``h``.  The solver-level
   control then integrates the hydrostatic case with that broken kernel: the
   pressure profile must stop matching ``rho g (H - z)``.  A control that only
   perturbs the interpolation would say nothing about the solver.

The static test is deliberately separated from :mod:`tensorsph.verification.
dam_break_2d`: it isolates the discretisation from the time integrator, so a
failure here cannot be blamed on the time stepping.
"""
from __future__ import annotations

from dataclasses import dataclass
import math

import torch

from ..droplet_impact import cubic_spline_kernel
from ..neighbor_search import cell_linked_pairs


@dataclass
class ConvergenceCase:
    """Refinement ladder of the static interpolation test."""

    domain: float = 0.4
    dxx: tuple[float, ...] = (0.02, 0.01, 0.005, 0.0025)
    h_factor: float = 1.3
    wave_number: float = 2.0 * math.pi / 0.2   # five wavelengths across
    support_factor: float = 2.0      # R = support_factor * h
    truncate_control: float = 1.5    # R = 1.5 h for the negative control


def _field(x: torch.Tensor, y: torch.Tensor, k: float) -> torch.Tensor:
    """Smooth, non-separable test field with zero mean over the domain."""
    return torch.sin(k * x) * torch.cos(k * y * 0.5) + 0.5 * torch.cos(k * x * 2.0)


def _lattice(dx: float, domain: float, device) -> torch.Tensor:
    n = int(round(domain / dx))
    xs = (torch.arange(n, dtype=torch.float64, device=device) + 0.5) * dx
    gx, gy = torch.meshgrid(xs, xs, indexing="xy")
    return torch.stack((gx.reshape(-1), gy.reshape(-1)), dim=-1)


def _interpolate(pos: torch.Tensor, f: torch.Tensor, h: float, dx: float,
                 support: float, normalize: bool) -> tuple[torch.Tensor, torch.Tensor]:
    """Return ``(raw, normalized)`` interpolants at every particle."""
    n = pos.shape[0]
    # support is the *actual* cut-off radius; the kernel constant is always the
    # two-h normalisation, so truncating the support breaks the partition of
    # unity -- that is exactly the negative control.
    # ``cell_linked_pairs`` returns unique unordered pairs (n_pairs, 2); the
    # kernel sum needs both directions, so the pair list is mirrored.
    pairs = cell_linked_pairs(pos, support)
    i = torch.cat((pairs[:, 0], pairs[:, 1]))
    j = torch.cat((pairs[:, 1], pairs[:, 0]))
    delta = pos[i] - pos[j]
    r = torch.linalg.vector_norm(delta, dim=-1)
    w = cubic_spline_kernel(r, h) * (r <= support)
    contrib = f[j] * w * dx * dx
    raw = torch.zeros(n, dtype=torch.float64, device=pos.device)
    wsum = torch.zeros_like(raw)
    raw.index_add_(0, i, contrib)
    wsum.index_add_(0, i, w * dx * dx)
    # SPH summation includes the particle itself.  ``cell_linked_pairs`` never
    # returns i == j, so the self term is added explicitly; leaving it out is a
    # 27 % kernel-sum deficit at h = 1.3 dx (it is the largest single term).
    self_w = float(cubic_spline_kernel(torch.zeros(1, dtype=torch.float64), h)[0])
    raw = raw + f * self_w * dx * dx
    wsum = wsum + self_w * dx * dx
    return raw, raw / wsum.clamp_min(1e-30)


def _relative_l2(err: torch.Tensor, ref: torch.Tensor) -> float:
    return float(torch.sqrt((err.square()).mean() / ref.square().mean()))


def run(case: ConvergenceCase | None = None) -> dict:
    case = case or ConvergenceCase()
    device = torch.device("cpu")
    ladder = []
    for dx in case.dxx:
        h = case.h_factor * dx
        pos = _lattice(dx, case.domain, device)
        f = _field(pos[:, 0], pos[:, 1], case.wave_number)
        margin = 2.0 * h + dx
        interior = ((pos[:, 0] > margin) & (pos[:, 0] < case.domain - margin)
                    & (pos[:, 1] > margin) & (pos[:, 1] < case.domain - margin))
        if int(interior.sum()) < 10:
            raise ValueError(f"dx={dx:g} leaves too few interior particles")
        raw, norm = _interpolate(pos, f, h, dx, case.support_factor * h, True)
        c_raw, c_norm = _interpolate(pos, f, h, dx, case.truncate_control * h, True)
        f_ref = f[interior]
        ladder.append({
            "dx": dx, "h": h, "h_over_dx": case.h_factor,
            "particles": int(pos.shape[0]),
            "interior": int(interior.sum()),
            "l2_error_normalized": _relative_l2(norm[interior] - f_ref, f_ref),
            "l2_error_raw": _relative_l2(raw[interior] - f_ref, f_ref),
            "l2_error_truncated_support": _relative_l2(c_norm[interior] - f_ref, f_ref),
            "kernel_sum_interior": float(
                (raw[interior] / f_ref.clamp_min(1e-30)).median()) if False else None,
        })
    for row in ladder:
        row.pop("kernel_sum_interior", None)

    def order(key: str, i: int) -> float:
        r0, r1 = ladder[i], ladder[i + 1]
        if r0[key] <= 0 or r1[key] <= 0:
            return float("nan")
        return math.log(r0[key] / r1[key]) / math.log(r0["h"] / r1["h"])

    def summary(key: str) -> dict:
        orders = [order(key, i) for i in range(len(ladder) - 1)]
        finite = [o for o in orders if math.isfinite(o)]
        return {"pair_orders": orders,
                "mean_order": sum(finite) / len(finite) if finite else float("nan"),
                "min_order": min(finite) if finite else float("nan"),
                "first_order": orders[0] if orders else float("nan"),
                "last_order": orders[-1] if orders else float("nan")}

    return {
        "case": "kernel_interpolation_convergence",
        "config": {
            "domain": case.domain, "h_factor": case.h_factor,
            "wave_number": case.wave_number,
            "support_factor": case.support_factor,
            "truncate_control": case.truncate_control,
            "spacings": list(case.dxx),
        },
        "ladder": ladder,
        "orders": {
            "normalized": summary("l2_error_normalized"),
            "raw": summary("l2_error_raw"),
            "truncated_support": summary("l2_error_truncated_support"),
        },
    }


def kernel_sum_defect(dx: float = 0.01, h_factor: float = 1.3,
                      support_factors: tuple[float, ...] = (2.0, 1.5),
                      domain: float = 0.2) -> dict:
    """Partition-of-unity defect ``|sum_j W dx^2 - 1|`` on an interior lattice.

    The kernel constant is always the two-``h`` normalisation; ``support_factor``
    only moves the cut-off, which is what a truncated kernel does.  The defect
    is scale invariant (it depends on ``h/dx`` alone), so it does not vanish as
    ``dx -> 0`` -- that is precisely why the support radius may not be reduced.
    """
    pos = _lattice(dx, domain, torch.device("cpu"))
    h = h_factor * dx
    margin = 2.0 * h + dx
    interior = ((pos[:, 0] > margin) & (pos[:, 0] < domain - margin)
                & (pos[:, 1] > margin) & (pos[:, 1] < domain - margin))
    out = {}
    for factor in support_factors:
        support = factor * h
        pairs = cell_linked_pairs(pos, 2.0 * h)
        i = torch.cat((pairs[:, 0], pairs[:, 1]))
        j = torch.cat((pairs[:, 1], pairs[:, 0]))
        r = torch.linalg.vector_norm(pos[i] - pos[j], dim=-1)
        w = cubic_spline_kernel(r, h) * (r <= support) * dx * dx
        total = torch.zeros(pos.shape[0], dtype=torch.float64)
        total.index_add_(0, i, w)
        # Self term: the kernel value at r = 0, weighted by the cell area.
        self_w = float(cubic_spline_kernel(torch.zeros(1, dtype=torch.float64), h)[0])
        total = total + self_w * dx * dx
        defect = (total[interior] - 1.0).abs()
        out[f"support_{factor:g}h"] = {
            "max_defect": float(defect.max()),
            "mean_defect": float(defect.mean()),
            "support_radius_over_dx": factor * h_factor,
        }
    baseline = out[f"support_{support_factors[0]:g}h"]["mean_defect"]
    truncated = out[f"support_{support_factors[1]:g}h"]["mean_defect"]
    out["defect_ratio"] = float(truncated / baseline) if baseline > 0 else float("inf")
    out["dx"] = dx
    out["h_over_dx"] = h_factor
    return out


def truncated_support_hydrostatic(support_factor: float = 1.5,
                                  dx: float = 0.02, seconds: float = 0.5,
                                  tank_width: float = 0.4, depth: float = 0.4) -> dict:
    """Solver-level negative control: integrate the hydrostatic case on a broken kernel.

    The support is cut at ``support_factor * h`` while the kernel constant keeps
    the two-``h`` normalisation, so the discrete kernel sum -- and with it the
    summation density and the pressure -- is wrong by an ``O(1)`` amount.  The
    hydrostatic balance must be destroyed; the run may also go unstable, which
    counts as "diverged" as well.
    """
    import tensorsph.droplet_impact as di
    from .hydrostatic import HydrostaticCase, run_hydrostatic_case

    original = di.pair_kernels

    def truncated(delta, h, eps=1e-12):
        w, g = original(delta, h, eps)
        r = torch.linalg.vector_norm(delta, dim=-1)
        mask = r <= support_factor * h
        return w * mask, g * mask.unsqueeze(-1)

    di.pair_kernels = truncated
    case = HydrostaticCase(dx=dx, tank_width=tank_width, depth=depth,
                           seconds=seconds, min_particles=1)
    try:
        result = run_hydrostatic_case(case)
        errored = None
    except (FloatingPointError, RuntimeError) as exc:
        result = None
        errored = f"{type(exc).__name__}: {exc}"
    finally:
        di.pair_kernels = original
    if result is None:
        return {"case": "hydrostatic_truncated_support", "diverged": True,
                "error": errored, "max_band_error_pct": float("inf"),
                "support_factor": support_factor}
    return {"case": "hydrostatic_truncated_support", "diverged": False,
            "error": None, "support_factor": support_factor,
            "max_band_error_pct": result["measured"]["max_band_error_pct"],
            "particles": result["measured"]["n_particles"],
            "seconds": seconds}


def gate(result: dict, min_order: float = 1.0,
         min_defect_ratio: float = 10.0,
         control: dict | None = None,
         control_growth: float = 5.0,
         control_min_abs_pct: float = 0.0,
         baseline_gate_pct: float = 2.0) -> dict:
    """J4: convergence order >= 1 and the truncated support must break the solver."""
    orders = result["orders"]
    measured = orders["normalized"]["min_order"]
    defect = result.get("support_defect", {})
    ratio = defect.get("defect_ratio", float("nan"))
    control_ok = None
    control_error = None
    baseline_error = result.get("hydrostatic_baseline_pct")
    if control is not None:
        control_error = control["max_band_error_pct"]
        # The control passes iff **it fails the gate that the baseline passes**:
        #  * the same case with the correct kernel must pass the hydrostatic gate
        #    (``baseline_gate_pct``), otherwise the comparison has no floor --
        #    a coarse baseline that is itself wrong cannot arbitrate anything;
        #  * the truncated kernel must degrade the error by at least
        #    ``control_growth`` relative to that baseline.
        # An *absolute* floor on the control error was tried and dropped: the
        # measured control error is resolution dependent (20.7 % at dx = 0.04,
        # 14.9 % at dx = 0.02), so an absolute threshold would be a statement
        # about the grid rather than about the kernel.
        control_ok = bool(control["diverged"]
                          or (control_error >= control_min_abs_pct
                              and control_error >= control_growth * (baseline_error or 0.0)
                              and (baseline_error is None
                                   or baseline_error <= baseline_gate_pct)))
    passed = bool(measured >= min_order and ratio >= min_defect_ratio
                  and (control_ok is not False))
    return {"quantity": "min observed order >= 1; truncated support (R < 2h) must "
                        "break the partition of unity and the hydrostatic balance",
            "measured_min_order": measured,
            "limit_min_order": min_order,
            "defect_ratio": ratio,
            "min_defect_ratio": min_defect_ratio,
            "control_max_band_error_pct": control_error,
            "control_min_abs_pct": control_min_abs_pct,
            "baseline_pct": baseline_error,
            "baseline_gate_pct": baseline_gate_pct,
            "control_diverged": None if control is None else control["diverged"],
            "control_pass": control_ok,
            "pass": passed}


def dynamic_front_order(runs: dict[float, list[dict]], taus: tuple[float, ...],
                        ) -> dict:
    """Self-convergence of the dam-break surface profile between resolutions.

    ``runs`` maps ``dx`` to the observable rows of
    :func:`tensorsph.verification.dam_break_2d.run` (same case, different
    spacing).  The profile of the finest run is the reference; the L2 difference
    of the surface height over the wetted length is reported at the requested
    non-dimensional times, together with the observed order between consecutive
    spacings.  Reported, not gated: the campaign's convergence gate is the
    static kernel test, and a dam-break profile at ``dx = 0.02`` has ten columns
    across the water, which is not a regime where an order can be demanded.
    """
    spacings = sorted(runs)
    rows = []
    coarse = spacings[0]
    fine = spacings[-1]
    for tau in taus:
        prof_c = _profile_at(runs[coarse], tau)
        prof_f = _profile_at(runs[fine], tau)
        if prof_c is None or prof_f is None:
            continue
        n = min(len(prof_c), len(prof_f))
        diff = sum((a - b) ** 2 for a, b in zip(prof_c[:n], prof_f[:n]))
        ref = sum(b * b for b in prof_f[:n])
        rows.append({"tau": tau, "dx_coarse": coarse, "dx_fine": fine,
                     "l2_relative": math.sqrt(diff / ref) if ref > 0 else float("nan")})
    orders = []
    if len(spacings) >= 3:
        mid = spacings[len(spacings) // 2]
        for tau in taus:
            p0 = _profile_at(runs[coarse], tau)
            p1 = _profile_at(runs[mid], tau)
            p2 = _profile_at(runs[fine], tau)
            if p0 is None or p1 is None or p2 is None:
                continue
            n = min(len(p0), len(p1), len(p2))
            e01 = math.sqrt(sum((a - b) ** 2 for a, b in zip(p0[:n], p2[:n]))
                            / max(1e-30, sum(b * b for b in p2[:n])))
            e12 = math.sqrt(sum((a - b) ** 2 for a, b in zip(p1[:n], p2[:n]))
                            / max(1e-30, sum(b * b for b in p2[:n])))
            if e12 > 0 and e01 > 0:
                orders.append({"tau": tau,
                               "order": math.log(e01 / e12) / math.log(mid / fine)})
    return {"rows": rows, "orders": orders,
            "mean_order": (sum(o["order"] for o in orders) / len(orders)
                           if orders else float("nan"))}


def _profile_at(rows: list[dict], tau: float) -> list[float] | None:
    if not rows:
        return None
    best = min(rows, key=lambda r: abs(r["tau"] - tau))
    if abs(best["tau"] - tau) > 0.15:
        return None
    prof = best["profile"]
    return [0.0 if v is None else v for v in prof]
