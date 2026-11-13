"""Turn a TensorSPH case run into an observation series for the protocol.

The reference practice observes a small set of integral quantities at a fixed
number of snapshots during the run (e.g. the dam-break front position at every
recorded iteration) and feeds the resulting series to the DTW regression test.
This module does the same for the TensorSPH dam-break case: it runs the case,
samples it at fixed physical times and returns a ``snapshot x quantity`` array.

Only position-derived quantities are used, so the series does not depend on the
lattice mass of the generation (which the calibration changes slightly):
``front_x`` (leading edge), ``top_y`` (free-surface maximum) and ``centroid_x``.
"""

from __future__ import annotations

import time
from dataclasses import dataclass

import numpy as np

from ..dam_break import DamBreakConfig, run_case

__all__ = ["QUANTITIES", "ObservationSeries", "dam_break_observations"]

QUANTITIES = ("front_x", "top_y", "centroid_x")


@dataclass
class ObservationSeries:
    """Observed series plus the run metadata needed to report it."""

    times: list[float]
    values: np.ndarray          # snapshot x quantity
    runtime: float              # wall-clock seconds for the run
    steps: int
    dt: float
    escaped: int
    mass_drift: float

    def as_rows(self) -> list[list[float]]:
        return [[float(value) for value in row] for row in self.values]


def _quantities_from_positions(positions) -> tuple[float, float, float]:
    array = positions.detach().cpu().numpy()
    return (
        float(array[:, 0].max()),
        float(array[:, 1].max()),
        float(array[:, 0].mean()),
    )


def dam_break_observations(cfg: DamBreakConfig, snapshot_times,
                           **run_kwargs) -> ObservationSeries:
    """Run the dam-break case and sample it at ``snapshot_times``.

    The snapshot times must be increasing and inside the simulated window
    ``cfg.steps * cfg.dt``; a shortfall is an error rather than a silently
    shortened series, because the protocol compares series starting from the
    first snapshot (see :mod:`tensorsph.regression_test.protocol`).
    """
    times = [float(t) for t in snapshot_times]
    if not times or any(b <= a for a, b in zip(times, times[1:])):
        raise ValueError("snapshot_times must be non-empty and strictly increasing")

    start = time.perf_counter()
    run = run_case(cfg, snapshot_times=times, **run_kwargs)
    runtime = time.perf_counter() - start

    recorded = [snapshot[0] for snapshot in run.snapshots]
    if len(run.snapshots) != len(times):
        raise RuntimeError(
            f"only {len(run.snapshots)} of {len(times)} snapshots were recorded "
            f"(simulated {run.simulated_time or times[-1]:.4f} s, last request "
            f"{times[-1]:.4f} s); increase cfg.steps or cfg.dt")
    values = np.asarray(
        [_quantities_from_positions(snapshot[1]) for snapshot in run.snapshots],
        dtype=float)
    return ObservationSeries(
        times=recorded,
        values=values,
        runtime=runtime,
        steps=run.steps_taken,
        dt=cfg.dt,
        escaped=run.escaped,
        mass_drift=run.mass_drift,
    )