"""Experiment-free distribution metrics for the TVC cross-validation.

Why this module exists
----------------------
Round 1-3 of ``XVAL-SPH-TRANSPORT-VELOCITY-CORRECTION`` graded the ported
transport-velocity correction (TVC) on the Martin & Moyce dam-break *front
position*.  That metric is one particle wide and only exists where an
experimental reference exists, so the rounds themselves recorded two open
debts (``docs/xval_sphinx_tvc/RESOLUTION_LADDER.md`` §7.2):

1. "one case, one metric" -- the bulk metrics in the payloads were never
   turned into a ladder of their own;
2. "a scheme can be front-equivalent and distribution-different".

This module is the *distribution leg*: it scores the particle set alone, with
**analytic anchors** instead of an experiment, so the owner's 15 % standard can
be applied to every resolution without any reference data:

===============================  =================  =====================
metric                           analytic anchor    meaning
===============================  =================  =====================
``nn_median_over_dx``            ``1`` exactly      lattice spacing is dx
``nn_cv``                        ``0`` exactly      no spread in r_nn
``close_pair_fraction``          ``0`` exactly      share of pairs < 0.7 dx
``bulk_grad1_mean_h``            ``0`` exactly      ``h |sum_j grad_i W_ij V_j|``
``bulk_concentration_error``     ``0`` exactly      ``|C_i / C_lattice - 1|``
===============================  =================  =====================

The anchors are *exact*, not fitted: on a perfect square lattice the neighbour
pairs are symmetric, so ``sum_j grad_i W_ij V_j`` cancels term by term for a
full-support particle, and ``C_i = sum_j V_j W_ij`` equals the same lattice sum
that :func:`lattice_reference` evaluates.  A metric therefore reports a
*relative error* against a value known analytically, which is what makes
"< 15 %" meaningful here.  ``bulk_grad1_mean_h`` and ``close_pair_fraction``
have anchor 0, so they are reported as absolutes (a relative error against zero
is undefined) and compared against the pre-declared noise floor in
``docs/xval_sphinx_tvc/DISTRIBUTION_LADDER.md``.

Two subtleties that the anchors force on the implementation:

* **full-support subset.**  ``C`` and the ``grad-1`` residual are only
  anchor-zero for particles whose kernel support is completely inside the
  cloud.  The subset is selected geometrically -- the neighbour count within
  ``2 h`` must equal the count of a perfect lattice at the same ``h/dx`` -- and
  *not* by a concentration threshold, because a concentration threshold is the
  quantity under test and would make the metric self-fulfilling.
* **float64.**  The anchors are round-off level (1e-14), so the accumulation is
  done in double precision regardless of the solver's dtype; the solver itself
  is not touched.

References
    Adami, S., Hu, X.Y. & Adams, N.A. (2013), J. Comput. Phys. 241:292-307.
    Zhang, C., Rezavand, M. & Hu, X. (2020), J. Comput. Phys. 404:109135.
"""
from __future__ import annotations

import math

import torch

__all__ = [
    "ANCHORS",
    "lattice_reference",
    "lattice_metrics",
    "distribution_error",
]

#: Analytic anchor of every metric on a perfect square lattice.
ANCHORS = {
    "nn_median_over_dx": 1.0,
    "nn_cv": 0.0,
    "close_pair_fraction": 0.0,
    "bulk_grad1_mean_h": 0.0,
    "bulk_concentration_error": 0.0,
}


def _kernel_ops():
    from tensorsph.droplet_impact import (cubic_spline_gradient,
                                         cubic_spline_kernel)
    return cubic_spline_kernel, cubic_spline_gradient


def _pairs(pos: torch.Tensor, radius: float) -> torch.Tensor:
    from tensorsph.neighbor_search import cell_linked_pairs
    return cell_linked_pairs(pos, radius)


# ---------------------------------------------------------------------------
# Analytic reference: the perfect square lattice
# ---------------------------------------------------------------------------

def lattice_reference(dx: float, h: float, dim: int = 2,
                      reach: int | None = None) -> dict:
    """Lattice sums of a perfect (infinite) square lattice, evaluated once.

    Returns
    -------
    dict with
        ``neighbour_count`` -- number of lattice sites with ``0 < r <= 2 h``
            (the full-support neighbour count, self excluded);
        ``concentration`` -- ``sum_j V_j W_ij`` including the self term, i.e.
            the value ``C_i`` takes for a full-support particle on that
            lattice.  For ``h = 1.3 dx`` it is 1 to the quadrature error, but
            the metric compares against the *discrete* value so no quadrature
            bias enters the anchor;
        ``grad1`` -- ``|sum_j grad_i W_ij V_j|``; exactly 0 by pair symmetry,
            returned so a caller can pin the symmetry numerically.
    """
    if not math.isfinite(dx) or dx <= 0:
        raise ValueError("dx must be finite and positive")
    if not math.isfinite(h) or h <= 0:
        raise ValueError("h must be finite and positive")
    if reach is None:
        reach = int(math.ceil(2.0 * h / dx)) + 1
    kernel, gradient = _kernel_ops()
    idx = torch.arange(-reach, reach + 1, dtype=torch.float64)
    gx, gy = torch.meshgrid(idx * dx, idx * dx, indexing="ij")
    rel = torch.stack([gx.reshape(-1), gy.reshape(-1)], dim=1)
    volume = dx ** dim
    r = torch.linalg.vector_norm(rel, dim=1)
    neighbour = (r > 0.0) & (r <= 2.0 * h)       # the self site is r == 0
    # self term
    conc = float(kernel(torch.zeros(1, dtype=torch.float64), h)) * volume
    conc += float((kernel(r[neighbour], h) * volume).sum())
    neighbour_count = int(neighbour.sum())
    g = gradient(rel[neighbour], h) * volume
    grad1 = float(torch.linalg.vector_norm(g.sum(dim=0)))
    return {"dx": float(dx), "h": float(h), "dim": int(dim),
            "neighbour_count": neighbour_count,
            "concentration": conc, "grad1": grad1,
            "volume": volume}


# ---------------------------------------------------------------------------
# Metrics on an arbitrary particle set
# ---------------------------------------------------------------------------

def lattice_metrics(pos: torch.Tensor, dx: float, h: float,
                    volume: torch.Tensor | None = None,
                    lattice: dict | None = None) -> dict:
    """Distribution metrics of ``pos`` with the analytic anchors above.

    Parameters
    ----------
    pos : (N, D) positions.  Conventions of the solver: the physical particle
        set, no wall particles (walls are static and would break the lattice
        anchors).
    dx : lattice spacing the case started from (the unit of every length).
    h : smoothing length of the run.
    volume : (N,) particle volumes.  ``None`` uses the lattice volume
        ``dx**D`` for every particle, which is the quadrature the SPH sums are
        compared against; the solver's own ``m/rho`` volumes are *not* used so
        that the metric measures the *geometry* and cannot be biased by a
        density error the correction is meant to influence.
    lattice : pre-computed :func:`lattice_reference` (recomputed if None).

    Returns
    -------
    dict of metrics; ``particles`` and ``bulk_particles`` report the sample
    sizes, every metric is ``None`` when its sample is empty rather than being
    silently zero.
    """
    if pos.ndim != 2:
        raise ValueError("pos must be (N, D)")
    dim = pos.shape[1]
    x = pos.detach().to(torch.float64)
    n = x.shape[0]
    if lattice is None:
        lattice = lattice_reference(dx, h, dim)
    if volume is None:
        volume = torch.full((n,), float(lattice["volume"]), dtype=torch.float64)
    else:
        volume = volume.detach().to(torch.float64)
        if volume.shape != (n,):
            raise ValueError("volume must be (N,)")

    out: dict = {"particles": int(n), "dx": float(dx), "h": float(h),
                 "dim": int(dim),
                 # stable schema: a metric that has no sample is None, never
                 # silently absent or zero
                 "nn_median_over_dx": None, "nn_mean_over_dx": None,
                 "nn_cv": None, "nn_p10_over_dx": None, "nn_p90_over_dx": None,
                 "close_pair_fraction": None, "close_pair_sample": 0,
                 "nn_sample": 0}

    # ---- nearest-neighbour statistics (lattice: 1.0, spread 0) ------------
    if n >= 2:
        near = _pairs(x, 1.6 * dx)
        if near.numel():
            i, j = near.unbind(dim=1)
            r = torch.linalg.vector_norm(x[i] - x[j], dim=1) / dx
            out["close_pair_fraction"] = float((r < 0.7).to(torch.float64).mean())
            nn = torch.full((n,), float("inf"), dtype=torch.float64)
            nn = nn.scatter_reduce(0, i, r, reduce="amin", include_self=True)
            nn = nn.scatter_reduce(0, j, r, reduce="amin", include_self=True)
            finite = torch.isfinite(nn)
            if int(finite.sum()) >= 2:
                vals = nn[finite]
                out["nn_median_over_dx"] = float(vals.median())
                mean = float(vals.mean())
                out["nn_mean_over_dx"] = mean
                out["nn_cv"] = (float(vals.std(unbiased=True)) / mean
                                if mean > 0 else None)
                out["nn_p10_over_dx"] = float(torch.quantile(vals, 0.10))
                out["nn_p90_over_dx"] = float(torch.quantile(vals, 0.90))
                out["nn_sample"] = int(vals.numel())
        out["close_pair_sample"] = int(near.shape[0]) if near.numel() else 0

    # ---- full-support subset, selected geometrically ----------------------
    pairs = _pairs(x, 2.0 * h)
    support_count = torch.zeros(n, dtype=torch.int64)
    if pairs.numel():
        i, j = pairs.unbind(dim=1)
        support_count = support_count.index_add(
            0, i, torch.ones_like(i)).index_add(0, j, torch.ones_like(j))
    # Full support means *at least* as many neighbours as a perfect lattice:
    # a compressed region has more, and excluding it would hide the very
    # failure mode the correction targets.
    bulk = support_count >= int(lattice["neighbour_count"])
    out["bulk_particles"] = int(bulk.sum())

    kernel, gradient = _kernel_ops()
    conc = torch.zeros(n, dtype=torch.float64)
    g = torch.zeros_like(x)
    if pairs.numel():
        i, j = pairs.unbind(dim=1)
        r = torch.linalg.vector_norm(x[i] - x[j], dim=1)
        w = kernel(r, h)
        # C_i = V_i W(0) + sum_j V_j W_ij  (symmetric pair accumulation)
        conc = conc.index_add(0, i, w * volume[j])
        conc = conc.index_add(0, j, w * volume[i])
        gw = gradient(x[i] - x[j], h)                       # grad_i W_ij
        # grad_i W_ji = -grad_i W_ij ; pair accumulation, both directions
        g = g.index_add(0, i, gw * volume[j][:, None])
        g = g.index_add(0, j, -gw * volume[i][:, None])
    self_term = float(kernel(torch.zeros(1, dtype=torch.float64), h))
    conc = conc + self_term * volume
    if int(bulk.sum()):
        c_err = (conc[bulk] / float(lattice["concentration"]) - 1.0).abs()
        out["bulk_concentration_error"] = float(c_err.median())
        out["bulk_concentration_error_max"] = float(c_err.max())
        out["bulk_grad1_mean_h"] = float(
            torch.linalg.vector_norm(g[bulk], dim=1).mean() * h)
        out["bulk_grad1_max_h"] = float(
            torch.linalg.vector_norm(g[bulk], dim=1).max() * h)
    else:
        out["bulk_concentration_error"] = None
        out["bulk_concentration_error_max"] = None
        out["bulk_grad1_mean_h"] = None
        out["bulk_grad1_max_h"] = None
    out["grad1_mean_h_all"] = float(
        torch.linalg.vector_norm(g, dim=1).mean() * h) if n else None
    out["bulk_fraction"] = float(bulk.to(torch.float64).mean()) if n else None
    return out


# ---------------------------------------------------------------------------
# Error against the analytic anchor
# ---------------------------------------------------------------------------

def distribution_error(metrics: dict) -> dict:
    """Relative / absolute error of a metric set against :data:`ANCHORS`.

    ``relative`` entries exist only for metrics whose anchor is non-zero
    (``nn_median_over_dx``); the anchor-zero metrics report ``absolute``.
    ``worst_relative`` is the quantity the owner's 15 % standard is applied to;
    it is ``None`` when no relative metric could be formed.
    """
    out: dict = {"anchor": dict(ANCHORS), "relative": {}, "absolute": {},
                 "worst_relative": None}
    for key, anchor in ANCHORS.items():
        value = metrics.get(key)
        if value is None:
            continue
        out["absolute"][key] = float(value) - anchor
        if anchor != 0.0:
            out["relative"][key] = abs(float(value) - anchor) / abs(anchor)
    if out["relative"]:
        out["worst_relative"] = max(out["relative"].values())
    return out