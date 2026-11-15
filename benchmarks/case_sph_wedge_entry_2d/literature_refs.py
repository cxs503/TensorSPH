"""Literature reference set for CASE-SPH-WEDGE-ENTRY-2D (reference-side upgrade).

Why this module exists
----------------------
The case has so far been graded against *one* analytic number, the Wagner
asymptotic peak ``Cp_max = C^2 + 1`` (``C = pi / (2 tan beta)``), because the
mission-designated reference (Zhao & Faltinsen 1993) was closed access and
therefore ``WITHHELD``.  Two consequences went unnoticed:

* the Wagner *asymptotic* peak is not the peak of the real (regularised) flow --
  its profile is singular at the spray root, so "the maximum" of that family sits
  21 % above the exact self-similar solution at ``beta = 30 deg``;
* the case therefore carried a -20 % style residual that was a property of the
  denominator, not of the SPH solver.

``refs.json`` now carries a peer-reviewed, open-access table (Jain, Novakovic,
Bogaert & van der Meer 2022, JFM 934 A26, Table 1) that lists ``Cp_max`` at
``beta = 30 deg`` for six independent models, including the Zhao & Faltinsen
(1993) composite solution and the exact similarity solution of Dobrovol'skaya
(1969).  This module is the single access point to those numbers: nothing is
hard-coded here, everything is read out of ``refs.json`` and unit-converted in
one place (:func:`cp_q_units`), so the factor-2 normalisation trap stays closed.

Normalisation
-------------
The table defines ``Cp_max = P_max / (rho V^2)``.  The case defines
``Cp = p / q`` with ``q = rho V^2 / 2``.  Every table value is multiplied by 2 on
the way in.  ``refs.json`` documents this, and ``tests/test_*literature_regrade``
re-derives the factor from the pressure-profile equation (eq. 10 of the source)
instead of trusting it.
"""

from __future__ import annotations

import json
import math
from pathlib import Path

REF_PATH = Path(__file__).with_name("refs.json")

#: models of Table 1 that are *point* pressure maxima (the sensor columns are
#: space-averaged and therefore not comparable with a point Cp_max)
POINT_MODELS = ("OLM", "MLM", "GWM", "EIM", "ASM", "COM")

#: the mission-designated metric owner (Zhao & Faltinsen 1993 composite solution)
DESIGNATED_MODEL = "COM"
#: independent exact solution (Dobrovol'skaya 1969, numerics by Wang & Faltinsen 2017)
EXACT_MODEL = "EIM"


def load_refs(path: Path | str = REF_PATH) -> dict:
    """Parsed ``refs.json`` (provenance for every number in the case)."""
    return json.loads(Path(path).read_text())


def table_ref(refs: dict | None = None) -> dict:
    """The open-access Table-1 record."""
    refs = refs or load_refs()
    for r in refs["references"]:
        if r["id"].startswith("jain_"):
            return r
    raise KeyError("open-access table reference missing from refs.json")


def cp_q_units(cp_rho_v2: float) -> float:
    """``P_max/(rho V^2)`` -> ``P_max/(0.5 rho V^2)`` (the case's Cp)."""
    return 2.0 * float(cp_rho_v2)


def beta_key(beta_deg: float, refs: dict | None = None) -> str:
    """Table row key for an integer deadrise angle.

    Raises for a non-integer angle *and* for an integer angle that has no row, so
    a caller can never silently interpolate a literature table.
    """
    b = float(beta_deg)
    key = str(int(round(b)))
    rows = sorted_rows(refs)
    if abs(b - round(b)) > 1e-9 or key not in rows:
        raise KeyError(
            f"Table 1 has no row for beta = {b} deg (rows: {rows}); "
            "do not interpolate a literature table -- pick a tabulated angle")
    return key


def c_wagner(beta_rad: float) -> float:
    """Wagner wetting coefficient ``C = pi / (2 tan beta)`` (eq. 9 of the source)."""
    return math.pi / (2.0 * math.tan(beta_rad))


def sorted_rows(refs: dict | None = None) -> list[str]:
    return sorted(table_ref(refs)["cp_max_rho_v2"], key=int)


def cp_max_table(beta_deg: float, refs: dict | None = None) -> dict:
    """``{model: Cp_max in q units}`` for a tabulated deadrise angle."""
    refs = refs or load_refs()
    row = table_ref(refs)["cp_max_rho_v2"][beta_key(beta_deg)]
    return {m: cp_q_units(row[m]) for m in POINT_MODELS}


def cp_max_model(model: str, beta_deg: float, refs: dict | None = None) -> float:
    return cp_max_table(beta_deg, refs)[model]


def designated_cp_max(beta_deg: float, refs: dict | None = None) -> float:
    """Zhao & Faltinsen (1993) composite value in q units (mission metric owner)."""
    return cp_max_model(DESIGNATED_MODEL, beta_deg, refs)


def band(beta_deg: float, refs: dict | None = None) -> dict:
    """Model-form spread of the reference at one angle (systematic uncertainty).

    Returns the six point-pressure models plus the min/max/half-spread.  The
    spread is *not* a numerical tolerance: it is the disagreement between
    published models of the same idealisation, and it is reported next to the
    acceptance number so that a residual inside the spread is never presented as
    a solver success.
    """
    tab = cp_max_table(beta_deg, refs)
    lo, hi = min(tab.values()), max(tab.values())
    return {"models": tab, "min": lo, "max": hi,
            "half_spread": 0.5 * (hi - lo),
            "rel_half_spread": 0.5 * (hi - lo) / tab[DESIGNATED_MODEL]}


def wagner_eq10_over_q(tau, beta_rad: float):
    """Independent re-implementation of the source's pressure profile, eq. (10).

    ``p/(rho V^2) = C*s - (1/2) * tau^2/(1-tau^2)`` with ``s = 1/sqrt(1-tau^2)``;
    multiplied by 2 (``q = rho V^2/2``) and using ``tau^2/(1-tau^2) = s^2 - 1`` this
    is ``p/q = 2*C*s - s^2 + 1``, i.e. the in-repo profile.  Returns p/q.  Written from the published equation, *not* by calling
    ``tensorsph.wedge_entry.wagner_p_over_q`` -- that is the whole point: the
    test compares the two.
    """
    C = c_wagner(beta_rad)
    if isinstance(tau, (int, float)):
        tau = abs(float(tau))
        if tau >= 1.0:
            return float("inf")
        s = 1.0 / math.sqrt(1.0 - tau * tau)
        return 2.0 * (C * s - 0.5 * tau * tau / (1.0 - tau * tau))
    t = abs(tau)
    s = 1.0 / (1.0 - t * t).clamp_min(1e-30).sqrt()
    return 2.0 * (C * s - 0.5 * t * t / (1.0 - t * t).clamp_min(1e-30))


def wagner_peak_eq10(beta_rad: float) -> float:
    """Peak of eq. (10) in q units, analytically: at ``s = C`` -> ``C^2 + 1``."""
    C = c_wagner(beta_rad)
    return C * C + 1.0 if C >= 1.0 else 2.0 * C


def asymptotic_leading_order(beta_rad: float) -> float:
    """``C^2/2`` in ``rho V^2`` units -- the Wang & Faltinsen leading order (ASM).

    Used as a *unit* check on the digitised table: a wrong factor in the
    normalisation cannot reproduce this column for four angles at once.
    """
    C = c_wagner(beta_rad)
    return 0.5 * C * C