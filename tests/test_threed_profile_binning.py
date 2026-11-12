"""Is the 3-D hydrostatic per-layer error a property of the field or of the binning?

The column is created on the lattice planes ``z = walls + k dx``, i.e. exactly on
the edges of the Eulerian bins ``floor((z - walls) / dx)`` that
:func:`tensorsph.threed.solver.build_pressure_profile` used to compare against
``rho0 g (H - z)`` evaluated at the bin *centre*.  Those two choices do not match:
the label sits half a cell away from the mass it describes, and a row that is a
rounding error below a bin edge is counted in the bin underneath it.  The
resulting per-layer error is an artefact of the estimator, not of the SPH field,
and it grows like ``(dx / 2) / depth`` -- which is why the reported worst-layer
error lived near the free surface and did not converge with ``nz``.

These tests are written so that the *measurement* is falsifiable on its own:

* :func:`test_exact_field_is_recovered_only_by_the_lattice_row_estimator` feeds a
  field that is exactly ``rho0 g (H - z)`` on the lattice planes.  An estimator
  that is fit for purpose has to return zero error; the Eulerian one does not.
* :func:`test_committed_artifact_registers_the_binning_diagnostic` binds the claim
  to the committed artifact, so the artifact cannot silently lose the diagnostic
  or the Lagrangian numbers the ledger now quotes.
"""

import json
from pathlib import Path

import pytest
import torch

from tensorsph.threed.solver import build_pressure_profile

REPO = Path(__file__).resolve().parents[1]
HYDRO = REPO / "benchmarks" / "threed_hydrostatic" / "report.json"

DX = 0.01
WALLS = 0.02
NZ = 16
NX = NY = 4
RHO0 = 1000.0
G = 9.81
HEIGHT = NZ * DX
H = 1.3 * DX


def _exact_column():
    """Buckets of an exact hydrostatic field sampled on the lattice planes."""
    z_rows = [WALLS + k * DX for k in range(NZ)]
    z = torch.tensor([v for v in z_rows for _ in range(NX * NY)],
                     dtype=torch.float64)
    surface = WALLS + HEIGHT
    p = RHO0 * G * (surface - z)
    rho = torch.full_like(z, RHO0)
    buckets = [{"z": z, "p": p, "rho": rho}]
    row_index = torch.tensor([k for k in range(NZ) for _ in range(NX * NY)],
                             dtype=torch.int64)
    return buckets, row_index, surface


def _profile(*, with_rows: bool = True):
    buckets, rows, _ = _exact_column()
    kwargs = {"row_index": rows} if with_rows else {}
    return build_pressure_profile(
        buckets, nz=NZ, dx=DX, walls=WALLS, height=HEIGHT, rho0=RHO0,
        gravity=G, h=H, sound_speed=20.0, gamma=7.0, **kwargs)


def test_exact_field_is_recovered_only_by_the_lattice_row_estimator():
    profile = _profile()

    # Lagrangian rows: the field is the reference, so the error must vanish.
    assert profile["rows_by_lattice"], "the lattice-row table must be built"
    assert len(profile["rows_by_lattice"]) == NZ
    assert profile["max_rel_error_core_lattice"] == pytest.approx(0.0, abs=1e-12)
    assert profile["max_abs_error_over_scale_lattice"] == pytest.approx(
        0.0, abs=1e-12)

    # Eulerian bins: same field, non-zero error.  The error of a bin is
    # (bin centre - mean z of the particles in it) / local depth, so it is
    # bounded by one bin width over the depth and it is largest where the depth
    # is smallest -- i.e. towards the free surface, independently of the field.
    assert profile["max_rel_error_core"] > 0.05, (
        "an exact field must not be reported as an exact profile by the "
        f"Eulerian estimator: max_rel_error_core="
        f"{profile['max_rel_error_core']}")
    core = [r for r in profile["rows"] if r["support_complete"]]
    for row in core:
        bound = 1.5 * DX / row["depth_below_surface"]
        assert abs(row["p_rel_error"]) <= bound + 1e-12, row
    by_depth = sorted(core, key=lambda r: -r["depth_below_surface"])
    assert abs(by_depth[-1]["p_rel_error"]) > abs(by_depth[0]["p_rel_error"])


def test_binning_diagnostic_quantifies_the_degeneracy():
    profile = _profile()
    diag = profile["binning"]
    assert diag is not None
    # The particles sit on the bin edges, so the label (bin centre) is half a
    # cell away from the mass: the *RMS* offset is half a cell.  Whether the
    # mean offset is -1/2 or +1/2 depends on which side of the edge the floating
    # point arithmetic lands, which is exactly why the Eulerian partition is not
    # a usable estimator.
    assert diag["rms_offset_over_dx"] == pytest.approx(0.5, abs=0.02)
    assert abs(diag["mean_offset_over_dx"]) > 0.2
    assert diag["rows"] == NZ
    # Rows whose (z - walls) / dx lands just below an integer are counted in the
    # bin underneath their own, i.e. the estimator mixes neighbours.
    assert diag["misplaced_fraction"] > 0.0


def test_without_row_index_only_the_eulerian_table_is_built():
    """Backwards compatibility: callers that pass no lattice label still work."""
    profile = _profile(with_rows=False)
    # ``None`` means "no label supplied", not "row 0": the helper must fall back
    # to the Eulerian-only payload instead of raising or inventing rows.
    assert profile["rows"], profile
    assert profile["rows_by_lattice"] == []


def test_committed_artifact_registers_the_binning_diagnostic():
    if not HYDRO.is_file():
        pytest.fail(f"missing 3-D hydrostatic artifact {HYDRO}; run "
                    f"`PYTHONPATH=src python benchmarks/threed_hydrostatic/run.py "
                    f"--merge`")
    report = json.loads(HYDRO.read_text())
    assert report["runs"]
    for run in report["runs"]:
        pressure = run["pressure"]
        diag = pressure["binning"]
        assert diag is not None, run["case"]
        assert 0.0 <= diag["misplaced_fraction"] <= 1.0
        # The Lagrangian numbers are part of the record, and they must be the
        # ones quoted by the ledger: a profile that is exact on the lattice can
        # not be reported as a FAIL by a binning artefact alone.
        assert pressure["max_rel_error_core_lattice"] is not None
        assert pressure["mean_rel_error_core_lattice"] is not None
        assert len(run["profile_lattice"]) == run["nz"]
        assert sum(r["particles"] for r in run["profile_lattice"]) % (
            report["params"]["nx"] * report["params"]["ny"] * run["nz"]) == 0
    # The two estimators must disagree somewhere; otherwise the diagnostic is
    # dead code and the binning story is wrong.  Per run they can coincide (a bin
    # that holds two rows straddling the label has the right mean), so the test
    # is on the set, not on every element.
    spreads = [abs(r["pressure"]["max_rel_error_core_lattice"]
                   - r["pressure"]["max_rel_error_core"])
               for r in report["runs"]]
    assert max(spreads) > 0.01, spreads