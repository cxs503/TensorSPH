"""Dynamic-time-warping distance between observation series.

Reference practice
------------------
``Xiangyu-Hu/SPHinXsys``, ``src/shared/regression_test/dynamic_time_warping_method.hpp``
(``calculateDTWDistance``, lines 16-55) and ``dynamic_time_warping_method.h``
(``calculatePNorm``, lines 54-63).  The algorithm as it is written there is:

* local cost = ``|a_i - b_j|`` for scalars, Euclidean norm for vector-valued
  observations (``calculatePNorm`` overloads);
* half-band width ``window_size_ = 5``, widened to ``max(5, |len_a - len_b|)``;
* a *length guard*: ``b_len > 1.2 * a_len`` or ``b_len < 0.8 * a_len`` is an
  error (a big change in the time step, i.e. in the number of snapshots, is not
  a regression but a protocol violation);
* the DP table is initialised to **zero** (not infinity) and only the band
  ``max(1, i - w) <= j < min(b_len, i + w)`` is written, so cells outside the
  band keep the value ``0``.  The three-way minimum can therefore pick a
  zero-valued out-of-band neighbour; see :func:`band_edge_shortcuts`.

This module offers three functions:

``dtw_distance``
    Faithful port of the above, including the zero-initialised band.  Used by
    the protocol so that distances are comparable with the reference practice.
``dtw_distance_strict``
    Textbook variant (``inf`` initialised, optional full band).  Used as the
    "what the number should be" reference and as the opt-in strict mode.
``dtw_distance_shortest_path``
    Independent implementation lineage: the same constrained alignment posed as
    a shortest path on a DAG and solved with ``scipy.sparse.csgraph.dijkstra``.
    It shares no code or recurrence with the DP above, so agreement between it
    and :func:`dtw_distance_strict` is evidence that the DP is right rather than
    just self-consistent.

All functions accept a sequence of scalars or a sequence of vectors (arrays of
equal length) and return a finite float.  Timings are O(len_a * len_b) for the
DP variants and O(E log V) for the graph variant.
"""

from __future__ import annotations

import math

import numpy as np

__all__ = [
    "DTWLengthError",
    "band_edge_shortcuts",
    "dtw_distance",
    "dtw_distance_shortest_path",
    "dtw_distance_strict",
]

DEFAULT_WINDOW = 5
DEFAULT_LENGTH_GUARD = 0.2  # accepted relative length mismatch, per side


class DTWLengthError(ValueError):
    """Raised when the two series differ in length by more than the guard."""


def _as_points(series) -> list[tuple[float, ...]]:
    """Normalise a series into a list of float tuples (scalars become 1-tuples)."""
    array = np.asarray(series, dtype=float)
    if array.ndim == 1:
        return [(float(value),) for value in array]
    if array.ndim == 2:
        return [tuple(float(component) for component in row) for row in array]
    raise ValueError(f"series must be 1-D or 2-D, got shape {array.shape}")


def _cost(point_a: tuple[float, ...], point_b: tuple[float, ...]) -> float:
    """``calculatePNorm``: ``|a - b|`` for scalars, Euclidean norm for vectors."""
    if len(point_a) == 1 and len(point_b) == 1:
        return abs(point_a[0] - point_b[0])
    return math.dist(point_a, point_b)


def _check_lengths(len_a: int, len_b: int, guard: float | None) -> None:
    if guard is None:
        return
    if len_b > (1.0 + guard) * len_a or len_b < (1.0 - guard) * len_a:
        raise DTWLengthError(
            f"series lengths {len_a} and {len_b} differ by more than the "
            f"+-{100 * guard:g}% guard; the snapshot count changed too much to be "
            "a regression comparison (SPHinXsys exits here as well)")


def _effective_window(len_a: int, len_b: int, window: int | None) -> int | None:
    """Reference widening rule: ``window -> max(window, |len_a - len_b|)``."""
    if window is None:
        return None
    return max(window, abs(len_a - len_b))


def _sphinxsys_dp(points_a, points_b, window: int, track_shortcuts: bool):
    """DP exactly as ``calculateDTWDistance`` writes it (zero-initialised band)."""
    len_a, len_b = len(points_a), len(points_b)
    table = [[0.0] * len_b for _ in range(len_a)]
    table[0][0] = _cost(points_a[0], points_b[0])
    for i in range(1, len_a):
        table[i][0] = table[i - 1][0] + _cost(points_a[i], points_b[0])
    for j in range(1, len_b):
        table[0][j] = table[0][j - 1] + _cost(points_a[0], points_b[j])

    width = max(window, abs(len_a - len_b))
    shortcuts = 0
    for i in range(1, len_a):
        row = table[i]
        prev = table[i - 1]
        point_a = points_a[i]
        for j in range(max(1, i - width), min(len_b, i + width)):
            up = prev[j]
            left = row[j - 1]
            diagonal = prev[j - 1]
            best = up
            from_band_edge = abs((i - 1) - j) > width      # ``up`` is unwritten
            if left < best:
                best = left
                from_band_edge = abs(i - (j - 1)) > width  # ``left`` is unwritten
            if diagonal < best:
                best = diagonal
                from_band_edge = abs((i - 1) - (j - 1)) > width
            if track_shortcuts and from_band_edge:
                # the chosen predecessor lies outside the band, so it was never
                # written and still holds its zero initialisation: the alignment
                # path teleported out of the band and back in.
                shortcuts += 1
            row[j] = _cost(point_a, points_b[j]) + best

    distance = table[len_a - 1][len_b - 1]
    return (distance, shortcuts) if track_shortcuts else distance


def dtw_distance(series_a, series_b, window: int = DEFAULT_WINDOW,
                 length_guard: float | None = DEFAULT_LENGTH_GUARD) -> float:
    """DTW distance with the reference implementation's semantics.

    ``window`` is the half-band width (the reference hard-codes 5); it is
    widened internally to ``max(window, |len_a - len_b|)``.  ``length_guard``
    is the relative length mismatch accepted per side (the reference hard-codes
    0.2); pass ``None`` to disable the check.

    The DP table is zero-initialised and only band cells are written, which is
    what the reference does.  That makes the distance an *under*-estimate when
    the optimal alignment leaves the band; :func:`dtw_distance_strict` gives the
    constrained-optimum value instead.
    """
    points_a, points_b = _as_points(series_a), _as_points(series_b)
    if not points_a or not points_b:
        raise ValueError("both series must be non-empty")
    _check_lengths(len(points_a), len(points_b), length_guard)
    return _sphinxsys_dp(points_a, points_b, window, track_shortcuts=False)


def band_edge_shortcuts(series_a, series_b, window: int = DEFAULT_WINDOW) -> dict:
    """Diagnose the zero-initialised band: how often and how much it matters.

    Returns ``{"distance": faithful, "strict": constrained optimum,
    "shortcut_cells": n, "relative_underestimate": (strict - faithful)/strict}``.
    A non-zero ``shortcut_cells`` means the reference recurrence allowed the
    alignment path to jump across the band through never-written zero cells.
    """
    points_a, points_b = _as_points(series_a), _as_points(series_b)
    _check_lengths(len(points_a), len(points_b), DEFAULT_LENGTH_GUARD)
    faithful, shortcuts = _sphinxsys_dp(points_a, points_b, window, track_shortcuts=True)
    strict = _strict_dp(points_a, points_b, max(window, abs(len(points_a) - len(points_b))))
    relative = 0.0 if strict == 0.0 else (strict - faithful) / strict
    return {
        "distance": faithful,
        "strict": strict,
        "shortcut_cells": shortcuts,
        "relative_underestimate": relative,
    }


def _strict_dp(points_a, points_b, window: int | None) -> float:
    """Canonical DTW: inf-initialised table, band (half-width ``window``) or none."""
    len_a, len_b = len(points_a), len(points_b)
    inf = math.inf
    table = [[inf] * len_b for _ in range(len_a)]
    for i in range(len_a):
        if window is not None:
            j_lo, j_hi = max(0, i - window), min(len_b - 1, i + window)
        else:
            j_lo, j_hi = 0, len_b - 1
        row = table[i]
        prev = table[i - 1] if i else None
        point_a = points_a[i]
        for j in range(j_lo, j_hi + 1):
            if i == 0 and j == 0:
                row[0] = _cost(point_a, points_b[0])
                continue
            best = inf
            if prev is not None and prev[j] < best:
                best = prev[j]
            if j > 0 and row[j - 1] < best:
                best = row[j - 1]
            if prev is not None and j > 0 and prev[j - 1] < best:
                best = prev[j - 1]
            if best < inf:
                row[j] = _cost(point_a, points_b[j]) + best
    if table[len_a - 1][len_b - 1] == inf:
        raise ValueError(
            "the band does not connect the two ends of the series; widen the window")
    return table[len_a - 1][len_b - 1]


def dtw_distance_strict(series_a, series_b, window: int | None = DEFAULT_WINDOW,
                        length_guard: float | None = DEFAULT_LENGTH_GUARD,
                        widen: bool = True) -> float:
    """Canonical constrained DTW distance (``window=None`` = unconstrained).

    ``widen=True`` applies the reference widening rule
    ``window -> max(window, |len_a - len_b|)`` so that a length mismatch inside
    the guard still leaves an admissible path; pass ``widen=False`` for the
    plain fixed-width band.
    """
    points_a, points_b = _as_points(series_a), _as_points(series_b)
    if not points_a or not points_b:
        raise ValueError("both series must be non-empty")
    _check_lengths(len(points_a), len(points_b), length_guard)
    effective = _effective_window(len(points_a), len(points_b), window) if widen else window
    return _strict_dp(points_a, points_b, effective)


def dtw_distance_shortest_path(series_a, series_b, window: int | None = DEFAULT_WINDOW,
                               length_guard: float | None = DEFAULT_LENGTH_GUARD,
                               widen: bool = True) -> float:
    """Same object as :func:`dtw_distance_strict`, computed as a shortest path.

    Nodes are the admissible ``(i, j)`` cells of the alignment lattice; a step
    from ``(i-1, j)``, ``(i, j-1)`` or ``(i-1, j-1)`` to ``(i, j)`` costs the
    local distance of the destination cell.  ``scipy.sparse.csgraph.dijkstra``
    then returns the minimum total cost, which is the constrained DTW distance.
    Deliberately independent of the DP recurrences above (different algorithm,
    different code path) so it can serve as a cross-validation reference.
    """
    from scipy.sparse import csr_matrix
    from scipy.sparse.csgraph import dijkstra

    points_a, points_b = _as_points(series_a), _as_points(series_b)
    _check_lengths(len(points_a), len(points_b), length_guard)
    len_a, len_b = len(points_a), len(points_b)
    if widen:
        window = _effective_window(len_a, len_b, window)

    def node(i: int, j: int) -> int:
        return i * len_b + j

    def admissible(i: int, j: int) -> bool:
        if window is None:
            return True
        return abs(i - j) <= window

    rows, cols, weights = [], [], []
    for i in range(len_a):
        for j in range(len_b):
            if not admissible(i, j):
                continue
            target = node(i, j)
            for di, dj in ((1, 0), (0, 1), (1, 1)):
                source_i, source_j = i - di, j - dj
                if source_i < 0 or source_j < 0:
                    continue
                if not admissible(source_i, source_j):
                    continue
                rows.append(node(source_i, source_j))
                cols.append(target)
                weights.append(_cost(points_a[i], points_b[j]))

    graph = csr_matrix((weights, (rows, cols)), shape=(len_a * len_b, len_a * len_b))
    start, finish = node(0, 0), node(len_a - 1, len_b - 1)
    distances = dijkstra(graph, directed=True, indices=start)
    total = distances[finish]
    if math.isinf(total):
        raise ValueError("no admissible path; widen the window")
    # the source cell's own local cost is not on any edge
    return float(total + _cost(points_a[0], points_b[0]))