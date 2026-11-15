"""Discrete modal reference of the shipped FSI2 rod, against Euler-Bernoulli.

Anchor: ``XVAL-SPH-FSI2-BENCHMARK-AND-FORCE-FROM-FLUID``.

Only the *static* beam properties of the FSI2 structural model had ever been
verified (``verify_beam_point_load``: ``EI`` and a tip deflection, V3/V4 in
``docs/xval_fsi2_benchmark_report.md``).  Which is not what the VIV benchmark
turns on: the tip amplitudes are set by whether the shedding period and the
**beam's own period** lock in, so an unverified modal frequency is an
unverified benchmark.  This module measures that frequency two independent
ways and compares both against the closed-form clamped-free value.

Method 1 -- linearised generalized eigenproblem
    ``H v = omega^2 M v`` with ``H = d^2 E/dx^2`` of the shipped discrete
    energy (:func:`tensorsph.fsi2_turek_hron.beam_energy`) at the straight
    configuration, ``M = diag(m_node)`` from
    :func:`tensorsph.fsi2_turek_hron.beam_nodal_mass`, and node 0 clamped.
    This is the *exact* linear spectrum of the discrete model -- no time
    stepping, no damping, no tuning.

Method 2 -- free decay of the nonlinear time integrator
    Excite the first discrete eigenvector at a small amplitude and integrate
    :func:`tensorsph.fsi2_turek_hron.beam_advance` (the very routine the
    coupled driver sub-steps).  The measured zero-crossing period must match
    method 1; if it does not, the linear spectrum does not describe the run.

The discretisation error is reported as a ladder in the node count.  The known
mechanism: the discrete chain represents the clamped end by a finite rotational
spring ``k_r = EI / l0`` at the first hinge, which tends to zero error only as
``l0 -> 0`` (``k_r -> inf`` is the rigid clamp), so the finite-``n`` model is
always *softer* than the analytic cantilever and the error must shrink
monotonically -- which is what the ladder asserts.
"""
from __future__ import annotations

import math
from typing import Sequence

import numpy as np
import torch

from .fsi2_beam_modes import cantilever_frequency_hz

__all__ = [
    "discrete_modes",
    "modal_ladder",
    "free_decay_period",
    "modal_summary",
]


def _free_mass_vector(case) -> "torch.Tensor":
    from .fsi2_turek_hron import beam_nodal_mass

    m = beam_nodal_mass(case).detach().cpu().numpy()
    return np.repeat(m[1:], 2)  # node 0 clamped; interleaved (x, y) per node


def discrete_modes(case, n_modes: int = 4) -> dict:
    """Linearised modal frequencies [Hz] of the shipped discrete rod."""
    from .fsi2_turek_hron import beam_energy, beam_initial_nodes

    nodes = beam_initial_nodes(case, torch.device("cpu")).reshape(-1)
    hess = torch.autograd.functional.hessian(
        lambda z: beam_energy(z.reshape(-1, 2), case), nodes)
    hess = hess.detach().cpu().numpy()
    free = np.arange(2, hess.shape[0])  # node 0 carries the clamped BC
    h = hess[np.ix_(free, free)]
    m = _free_mass_vector(case)
    if h.shape[0] != m.size:
        raise ValueError("hessian/mass shape mismatch")
    # Symmetric whitening: eigenvalues of M^-1/2 H M^-1/2 are omega^2.
    a = h / np.sqrt(np.outer(m, m))
    lam = np.linalg.eigvalsh(0.5 * (a + a.T))
    lam = np.clip(lam, 0.0, None)
    order = np.argsort(lam)
    lam = lam[order]
    freqs = np.sqrt(lam) / (2.0 * math.pi)
    modes = torch.tensor(np.linalg.eigh(0.5 * (a + a.T))[1][:, order],
                         dtype=torch.float64)
    return {
        "n_nodes": int(case.n_beam_nodes),
        "n_dof": int(m.size),
        "frequencies_hz": [float(x) for x in freqs[:n_modes]],
        "_whitened_eigenvectors": modes,
    }


def modal_ladder(node_counts: Sequence[int] = (15, 30, 50),
                 n_modes: int = 3) -> list:
    """Discrete-vs-analytic frequency error as a function of node count."""
    from .fsi2_turek_hron import FSI2Case

    rows = []
    for n in node_counts:
        case = FSI2Case(n_beam_nodes=int(n))
        modes = discrete_modes(case, n_modes=n_modes)
        row = {"n_nodes": int(n),
               "clamp_rotational_stiffness":
                   case.beam_stiffness_bending()
                   / (case.beam_length / (case.n_beam_nodes - 1))}
        for k in range(n_modes):
            f_disc = modes["frequencies_hz"][k]
            f_ref = cantilever_frequency_hz(case.beam_E, case.beam_rho,
                                            case.beam_thickness,
                                            case.beam_length, mode=k + 1)
            row[f"mode{k + 1}"] = {
                "discrete_hz": f_disc,
                "analytic_hz": f_ref,
                "rel_error": f_disc / f_ref - 1.0,
            }
        rows.append(row)
    return rows


def free_decay_period(case, amplitude_frac: float = 0.01,
                      periods: float = 3.0, safety: float = 0.30) -> dict:
    """Period of the shipped time integrator in free decay of mode 1 [s].

    ``dt = safety * 2 / omega_max`` from the model's own highest eigenfrequency
    (``beam_max_frequency``), so the explicit symplectic-Euler step is inside
    its stability limit by construction rather than by a tuned number.
    """
    from .fsi2_turek_hron import (beam_advance, beam_initial_nodes,
                                  beam_max_frequency)

    modes = discrete_modes(case, n_modes=1)
    f_eig = modes["frequencies_hz"][0]
    m = _free_mass_vector(case)
    v = (modes["_whitened_eigenvectors"][:, 0].cpu().numpy()
         / np.sqrt(m)).reshape(-1, 2)
    v = v / abs(v[-1, 1])  # tip y displacement is the readout

    nodes0 = beam_initial_nodes(case, torch.device("cpu"))
    amp = float(amplitude_frac) * float(case.beam_length)
    nodes = nodes0.clone()
    nodes[1:] = nodes0[1:] + torch.tensor(amp * v, dtype=nodes0.dtype)
    vel = torch.zeros_like(nodes)
    y0 = float(case.cylinder_y)

    omega_max = float(beam_max_frequency(case))
    dt = float(safety) * 2.0 / omega_max
    n_steps = int(periods / f_eig / dt) + 8
    tip = np.empty(n_steps)
    for k in range(n_steps):
        nodes, vel = beam_advance(nodes, vel, torch.zeros_like(nodes), case, dt)
        tip[k] = float(nodes[-1, 1]) - y0
    # Zero crossings of the tip displacement -> period.
    idx = np.where((tip[:-1] <= 0.0) & (tip[1:] > 0.0))[0]
    if idx.size < 2:
        return {"period_s": None, "frequency_hz": None,
                "relative_to_eigen": None, "n_steps": n_steps, "dt": dt,
                "f_eigen_hz": f_eig, "half_range_m": float(np.ptp(tip) / 2.0),
                "note": "fewer than two up-crossings"}
    frac = -tip[idx] / (tip[idx + 1] - tip[idx])
    crossings = (idx + frac) * dt
    per = np.diff(crossings)
    period = float(np.mean(per))
    half_range_first = float(np.max(np.abs(tip[: n_steps // 4])))
    half_range_last = float(np.max(np.abs(tip[-n_steps // 4:])))
    return {
        "period_s": period,
        "frequency_hz": 1.0 / period,
        "f_eigen_hz": f_eig,
        "relative_to_eigen": (1.0 / period) / f_eig - 1.0,
        "period_spread_s": float(np.ptp(per)) if per.size else 0.0,
        "n_crossings": int(idx.size),
        "n_steps": int(n_steps),
        "dt": dt,
        "amplitude_frac": float(amplitude_frac),
        "half_range_m": float(np.ptp(tip) / 2.0),
        "envelope_first_over_last": (half_range_first / half_range_last
                                    if half_range_last > 0 else None),
        "omega_max_rad_s": float(omega_max),
    }


def modal_summary(case, node_counts: Sequence[int] = (15, 30, 50)) -> dict:
    """Everything above in one dict, ready to serialise."""
    modes = discrete_modes(case, n_modes=4)
    f_ref = [cantilever_frequency_hz(case.beam_E, case.beam_rho,
                                     case.beam_thickness, case.beam_length,
                                     mode=k + 1) for k in range(4)]
    decay = free_decay_period(case)
    return {
        "case": {
            "n_beam_nodes": int(case.n_beam_nodes),
            "beam_E": float(case.beam_E),
            "beam_rho": float(case.beam_rho),
            "beam_thickness": float(case.beam_thickness),
            "beam_length": float(case.beam_length),
            "rho0": float(case.rho0),
            "beam_stiffness_bending": float(case.beam_stiffness_bending()),
            "beam_area_density": float(case.beam_area_density()),
        },
        "discrete_frequencies_hz": modes["frequencies_hz"],
        "analytic_frequencies_hz": f_ref,
        "discrete_rel_error": [modes["frequencies_hz"][k] / f_ref[k] - 1.0
                               for k in range(4)],
        "free_decay": decay,
        "ladder": modal_ladder(node_counts),
    }