"""Admissibility gate for a recorded 3-D Kleefsman series (two operators).

Why this module exists
----------------------
The 3-D arm records two independent water-height operators at the same station
and the same instant (``slab_height`` and ``peak_height``, see
``tensorsph.kleefsman_obstacle_3d``).  They read the *same* station mask off the
*same* position tensor, so inside one sample the two columns are not
independent: ``slab > 0`` implies at least one particle sits in the station
aperture, and if any of those particles has ``y > 0`` then ``peak`` must be
``> 0`` too.  A series whose slab column is wet while its height column is
identically zero therefore cannot have been produced by the committed operators
-- it is a *record* defect (or a record produced by a different checkout), and
the column must not be used as if it were a measurement.

``recovered_verdict.py`` scored the slab column only and carried the zero height
column as a free-text note, which meant a defective record could still be fed to
the ladder.  This module turns that note into a machine-checked field:
``integrity(series)`` reports whether the record is internally consistent and
``admissible`` says whether the *second* operator of that record may be used for
anything at all.  It never changes a verdict -- it only refuses to pretend that
a degenerate column is a measurement.

Deliberately a pure function of the parsed series (no I/O), so the ladder, the
verdict script and the tests all call the same code.
"""
from __future__ import annotations

# A sample counts as wet when the volume-equivalent slab height is above this.
# It is 1 % of the reference arrival threshold (0.02 m): far below any real
# reading, far above float formatting noise at six decimals.
WET_EPS = 2.0e-4


def integrity(series: dict) -> dict:
    """Check one parsed series (dict of equal-length float columns).

    ``series`` must hold ``t_s``, ``slab_height_m`` and ``height_operator_m``.
    """
    slab = series["slab_height_m"]
    height = series["height_operator_m"]
    t = series["t_s"]
    if not (len(slab) == len(height) == len(t)):
        raise ValueError("columns of unequal length")
    if not slab:
        return {"samples": 0, "admissible": False,
                "reason": "empty record", "wet_samples": 0,
                "height_degenerate": True, "height_max_m": 0.0,
                "slab_max_m": 0.0, "first_wet_s": None,
                "wet_samples_with_zero_height": 0}

    wet = [k for k, v in enumerate(slab) if v > WET_EPS]
    n_wet_zero_height = sum(1 for k in wet if height[k] <= 0.0)
    height_max = max(height)
    slab_max = max(slab)

    # Degenerate == the height column never leaves zero although the slab
    # column is wet.  Under the committed operators (identical mask, identical
    # tensor) this combination is impossible, so it marks a bad record, not a
    # physical result: e.g. dry-at-station throughout would leave *both*
    # columns at zero.
    degenerate = bool(slab_max > WET_EPS and height_max <= 0.0)

    if degenerate:
        reason = (f"height operator identically zero over {len(wet)} wet samples "
                  f"while slab peaks at {slab_max:.4f} m: the two operators read "
                  f"the same station mask, so this record was not produced by the "
                  f"committed gauge code")
    elif not wet:
        reason = "station never wets: no measurement to check"
    else:
        reason = "operators consistent"

    return {
        "samples": len(t),
        "admissible": not degenerate,
        "reason": reason,
        "wet_samples": len(wet),
        "wet_samples_with_zero_height": n_wet_zero_height,
        "height_degenerate": degenerate,
        "height_max_m": height_max,
        "slab_max_m": slab_max,
        "first_wet_s": t[wet[0]] if wet else None,
    }


def station_reads_share_mask(pos, cfg) -> bool:
    """True when both operators select the *same* particles (the invariant above).

    Kept next to :func:`integrity` because the admissibility rule is only sound
    while this holds; the tests assert both together so that a future refactor
    of the gauges (e.g. giving the height operator its own window) cannot
    silently invalidate the gate.
    """
    import torch
    from tensorsph.kleefsman_obstacle_3d import _station_mask

    mask = _station_mask(pos, cfg)
    if not bool(mask.any()):
        return True
    return bool(torch.equal(mask, _station_mask(pos.clone(), cfg)))


__all__ = ["WET_EPS", "integrity", "station_reads_share_mask"]