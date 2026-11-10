"""Free-surface measurement operators shared by the three benchmarks.

These are *measurements*, not physics: everything here turns a particle cloud
into the field quantity an analytic solution is stated in, and nothing here
touches the solver. Keeping them in one place is what makes the three problems
comparable -- a depth profile, a front position, an L2 norm and a modal
amplitude are measured the same way whichever case produced the particles.

Conventions
-----------
* ``slab`` is the width of the vertical strip a particle is counted into. The
  default is two particle spacings, so a strip holds ~2 columns of a regular
  lattice and the count is smoothed by one particle worth of noise on each side.
* Depth is reconstructed from **mass**, not from the particle count:
  ``h(x) = sum_i (m_i / rho0) / slab`` over the strip. Using the count assumes a
  uniform lattice; the dam-break initialisation is hydrostatically compressed
  (``hydrostatic_density``), so with the count the initial depth of a 10 cm
  column would read a few percent high. Mass also makes the reconstruction
  exact for mass-conserving flow whatever the particle spacing.
* ``relative_l2`` is ``||sim - ref|| / ||ref||`` over the samples handed to it,
  so its value is dimensionless and comparable between resolutions.
"""
from __future__ import annotations

import math

import torch

__all__ = [
    "binned_depth",
    "binned_velocity",
    "first_mode_amplitude",
    "front_from_profile",
    "loglog_slope",
    "oscillation_frequency",
    "peak_decay_rate",
    "relative_l2",
    "relative_l2_norm",
    "slab_centres",
]


def slab_centres(x_min: float, x_max: float, slab: float) -> torch.Tensor:
    """Mid-points of the strips ``[x_min + k slab, x_min + (k+1) slab)``."""
    if slab <= 0.0:
        raise ValueError("slab width must be positive")
    if x_max <= x_min:
        raise ValueError("x_max must exceed x_min")
    count = max(1, int(math.ceil((x_max - x_min) / slab)))
    return x_min + (torch.arange(count, dtype=torch.float64) + 0.5) * slab


def _slab_index(x: torch.Tensor, x_min: float, slab: float, count: int) -> torch.Tensor:
    idx = torch.floor((x - x_min) / slab).long()
    return idx.clamp(0, count - 1)


def binned_depth(positions: torch.Tensor, mass: torch.Tensor, rho0: float,
                 slab: float, x_min: float = 0.0,
                 x_max: float | None = None) -> tuple[torch.Tensor, torch.Tensor]:
    """Free-surface profile ``h(x)`` from a particle cloud.

    Returns ``(centres, depth)``. ``depth`` has the dimension of length: the
    volume (per unit width) of the particles in a strip divided by the strip
    width, so ``sum(depth) * slab`` is the total 2-D volume of the water.
    """
    if positions.numel() == 0:
        raise ValueError("no particles to bin")
    x = positions[:, 0].double()
    upper = float(x.max()) if x_max is None else float(x_max)
    centres = slab_centres(x_min, max(upper, x_min + slab), slab)
    idx = _slab_index(x, x_min, slab, centres.numel())
    volume = (mass.double() / float(rho0))
    depth = torch.zeros(centres.numel(), dtype=torch.float64)
    depth.index_add_(0, idx, volume)
    return centres, depth / slab


def binned_velocity(positions: torch.Tensor, velocity: torch.Tensor,
                    mass: torch.Tensor, slab: float, x_min: float = 0.0,
                    x_max: float | None = None) -> tuple[torch.Tensor, torch.Tensor]:
    """Mass-weighted horizontal velocity per strip ``u(x)``.

    Empty strips get ``nan``: there is no water there, and filling them with
    zero would bias an L2 norm against a reference that is also zero only by
    accident.
    """
    if positions.numel() == 0:
        raise ValueError("no particles to bin")
    x = positions[:, 0].double()
    upper = float(x.max()) if x_max is None else float(x_max)
    centres = slab_centres(x_min, max(upper, x_min + slab), slab)
    idx = _slab_index(x, x_min, slab, centres.numel())
    count = centres.numel()
    weights = torch.zeros(count, dtype=torch.float64).index_add(0, idx, mass.double())
    momentum = torch.zeros(count, dtype=torch.float64).index_add(
        0, idx, mass.double() * velocity[:, 0].double())
    out = momentum / weights.clamp_min(1.0e-30)
    out[weights <= 0.0] = float("nan")
    return centres, out


def front_from_profile(centres: torch.Tensor, depth: torch.Tensor,
                       threshold: float, slab: float) -> float:
    """Rightmost position where the profile is still wet.

    The bare ``max(x)`` of a particle cloud is a single splash particle and is
    not a front; the front is defined here as the crossing of ``depth =
    threshold``, linearly interpolated between the last wet and first dry strip.
    Returns ``centres[0] - slab / 2`` if nothing is wet.
    """
    wet = depth > threshold
    if not bool(wet.any()):
        return float(centres[0] - 0.5 * slab)
    last = int(torch.nonzero(wet).max())
    if last + 1 >= centres.numel():
        return float(centres[last] + 0.5 * slab)
    h0 = float(depth[last])
    h1 = float(depth[last + 1])
    span = h0 - h1
    frac = 0.0 if span <= 0.0 else (h0 - threshold) / span
    frac = min(max(frac, 0.0), 1.0)
    # interpolate between the wet and dry strip *centres*, which is where the
    # two depth samples used above live
    return float(centres[last] + frac * slab)


def relative_l2_norm(sim: torch.Tensor, ref: torch.Tensor) -> float:
    """``||sim - ref||_2 / ||ref||_2`` on a common sample set."""
    sim = torch.as_tensor(sim, dtype=torch.float64)
    ref = torch.as_tensor(ref, dtype=torch.float64)
    if sim.shape != ref.shape:
        raise ValueError(f"shape mismatch: {tuple(sim.shape)} vs {tuple(ref.shape)}")
    denominator = float(torch.linalg.vector_norm(ref))
    if denominator <= 0.0:
        raise ValueError("reference has zero norm")
    return float(torch.linalg.vector_norm(sim - ref)) / denominator


def relative_l2(samples: torch.Tensor, reference: torch.Tensor, scale: float) -> float:
    """RMS deviation normalised by ``scale`` (both are differences, not vectors).

    Used where the reference vanishes somewhere in the window and a vector norm
    would be dominated by its largest samples: ``sqrt(mean((a-b)^2)) / scale``.
    """
    samples = torch.as_tensor(samples, dtype=torch.float64)
    reference = torch.as_tensor(reference, dtype=torch.float64)
    if samples.shape != reference.shape:
        raise ValueError("shape mismatch")
    if scale <= 0.0:
        raise ValueError("scale must be positive")
    return float(torch.sqrt(torch.mean((samples - reference) ** 2))) / float(scale)


def loglog_slope(spacings, errors) -> float:
    """Least-squares slope of ``log(error)`` against ``log(spacing)``.

    Order of convergence ``p`` in ``error = C h^p``. At least two points are
    required; the caller decides how much a two-point slope is worth.
    """
    h = torch.log(torch.as_tensor(list(spacings), dtype=torch.float64))
    e = torch.log(torch.as_tensor(list(errors), dtype=torch.float64))
    if h.numel() < 2:
        raise ValueError("need at least two spacings")
    if h.numel() != e.numel():
        raise ValueError("spacings and errors differ in length")
    mean_h = float(h.mean())
    mean_e = float(e.mean())
    covariance = float(((h - mean_h) * (e - mean_e)).sum())
    variance = float(((h - mean_h) ** 2).sum())
    if variance <= 0.0:
        raise ValueError("degenerate spacings")
    return covariance / variance


def oscillation_frequency(times, signal) -> float:
    """Angular frequency from upward mean crossings of ``signal``.

    A zero-crossing count is used rather than an FFT peak because the free
    decay records here are only a few periods long: a 3-period record gives a
    4-bin FFT, while the crossings still resolve the period to a fraction of a
    sample.
    """
    t = torch.as_tensor(list(times), dtype=torch.float64)
    y = torch.as_tensor(list(signal), dtype=torch.float64)
    if t.numel() != y.numel() or t.numel() < 5:
        raise ValueError("need at least five matching samples")
    if t.numel() > 1 and not bool(torch.all(t[1:] > t[:-1])):
        raise ValueError("times must be strictly increasing")
    y = y - y.mean()
    crossings = []
    for i in range(1, y.numel()):
        if y[i - 1] <= 0.0 < y[i]:
            span = float(y[i] - y[i - 1])
            frac = 0.0 if span <= 0.0 else float(-y[i - 1] / span)
            crossings.append(float(t[i - 1] + frac * (t[i] - t[i - 1])))
    if len(crossings) < 2:
        raise ValueError("signal does not cross its mean twice")
    periods = [crossings[i + 1] - crossings[i] for i in range(len(crossings) - 1)]
    period = sum(periods) / len(periods)
    if period <= 0.0:
        raise ValueError("non-positive period")
    return 2.0 * math.pi / period


def peak_decay_rate(times, amplitude, skip: int = 1) -> float:
    """Exponential decay rate of a decaying oscillation's peak envelope.

    ``|a(t)| = a0 exp(-rate t)`` is fitted over the interior peaks (the first
    ``skip`` peaks are dropped: the released-from-rest start is not yet on the
    envelope). Returns the rate in 1/s, or ``nan`` if fewer than two peaks are
    available.
    """
    t = torch.as_tensor(list(times), dtype=torch.float64)
    a = torch.as_tensor(list(amplitude), dtype=torch.float64).abs()
    if t.numel() != a.numel() or a.numel() < 3:
        return float("nan")
    peaks_t: list[float] = []
    peaks_a: list[float] = []
    for i in range(1, a.numel() - 1):
        if a[i] >= a[i - 1] and a[i] >= a[i + 1] and float(a[i]) > 0.0:
            peaks_t.append(float(t[i]))
            peaks_a.append(float(a[i]))
    if len(peaks_t) <= skip + 1:
        return float("nan")
    peaks_t = peaks_t[skip:]
    peaks_a = peaks_a[skip:]
    tt = torch.tensor(peaks_t, dtype=torch.float64)
    la = torch.log(torch.tensor(peaks_a, dtype=torch.float64))
    tt = tt - tt.mean()
    la = la - la.mean()
    variance = float((tt * tt).sum())
    if variance <= 0.0:
        return float("nan")
    return -float((tt * la).sum()) / variance


def first_mode_amplitude(centres: torch.Tensor, elevation: torch.Tensor,
                         length: float, slab: float) -> float:
    """Projection of a surface elevation onto ``cos(pi x / L)``.

    ``a1 = (2/L) * integral eta(x) cos(pi x / L) dx`` is the modal amplitude of
    the first sloshing mode: it returns ``A`` for ``eta = A cos(pi x / L)`` and
    zero for the second mode. Projecting instead of sampling a single gauge
    averages out the particle-scale noise of an SPH free surface, which at a
    single probe is of the order of the wave amplitude itself.
    """
    centres = torch.as_tensor(centres, dtype=torch.float64)
    elevation = torch.as_tensor(elevation, dtype=torch.float64)
    if centres.shape != elevation.shape:
        raise ValueError("centres and elevation differ in shape")
    if length <= 0.0:
        raise ValueError("length must be positive")
    # Dry strips carry no measurement (NaN); they are left out of the Riemann
    # sum rather than filled with zero, which would bias the projection down by
    # the fraction of dry width.
    usable = torch.isfinite(elevation) & torch.isfinite(centres)
    if not bool(usable.any()):
        return float("nan")
    weight = torch.cos(math.pi * centres[usable] / float(length))
    integral = float((elevation[usable] * weight).sum()) * float(slab)
    return 2.0 * integral / float(length)
