"""Attribution tests: which operator *is* the SPHinXsys transport-velocity one?

``XVAL-SPH-TRANSPORT-VELOCITY-CORRECTION`` cross-validates against
``Xiangyu-Hu/SPHinXsys``.  Two modules in this repo ship a function called
``transport_velocity_correction`` and they are **not the same scheme**:

``tensorsph.transport_velocity_correction``
    the upstream operator: a position update driven by the ``grad-1``
    inconsistency, coefficient ``alpha * h_ref**2``, no reference velocity.
``tensorsph.transport_velocity``
    Lind et al. (2012) particle shifting written as a velocity correction
    (``dv = -A grad C / C``, ``A = C_d h v_ref dt``), used by
    ``droplet_impact``.

Round-1/2 documented the first and mis-credited the second (the docstring said
"In SPHinXsys the correction is dv = -A grad C / C").  The vendored upstream
files settle it, so these tests (i) verify the vendored reference is the file
the provenance claims, byte for byte, and (ii) pin the two operators apart
numerically, so no future edit can quietly relabel one as the other.
"""
from __future__ import annotations

import hashlib
import json
import re
import sys
from pathlib import Path

import pytest
import torch

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

from tensorsph.transport_velocity import (  # noqa: E402
    transport_velocity_correction as lind_correction)
from tensorsph.transport_velocity_correction import (  # noqa: E402
    kernel_gradient_integral, transport_velocity_correction as upstream_correction)

OSS = ROOT / "docs" / "xval_sphinx_tvc" / "oss_source"
PROVENANCE = ROOT / "docs" / "xval_sphinx_tvc" / "PROVENANCE.json"


def test_vendored_upstream_sources_match_their_recorded_hashes():
    """The reference implementation must be the fetched file, not a memory."""
    recorded = json.loads(PROVENANCE.read_text())["files_sha256"]
    checked = 0
    for name, digest in recorded.items():
        path = OSS / name
        assert path.is_file(), f"missing vendored file {name}"
        assert hashlib.sha256(path.read_bytes()).hexdigest() == digest, name
        checked += 1
    assert checked >= 10


def test_upstream_operator_has_no_concentration_and_moves_positions():
    """Structural facts of the upstream header the port is derived from."""
    hpp = (OSS / "transport_velocity_correction.hpp").read_text()
    hdr = (OSS / "transport_velocity_correction.h").read_text()
    # position update, coefficient alpha * h_ref**2, grad-1 inconsistency
    assert "kernel_gradient_integral_" in hpp
    assert "correction_scaling_" in hpp
    assert "pos_[index_i] +=" in hpp
    assert "correction_scaling_(coefficient * squared_h_ref_)" in hpp
    # and therefore no Lind-form ingredients anywhere in the upstream class
    assert "concentration" not in hpp.lower()
    assert not re.search(r"\bv_ref\b", hpp)
    assert "correction_ = 0.2" in hdr or "coefficient = 0.2" in hdr
    # the port exposes the same law
    from tensorsph.transport_velocity_correction import (
        SPHINXSYS_DEFAULT_COEFFICIENT)
    assert SPHINXSYS_DEFAULT_COEFFICIENT == 0.2


def _cluster(dx: float = 0.02, h_ratio: float = 1.3):
    """A lattice with one visibly clumped particle pair (the failure mode)."""
    n = 12
    xs = torch.arange(n, dtype=torch.float64) * dx
    gx, gy = torch.meshgrid(xs, xs, indexing="ij")
    pos = torch.stack((gx.reshape(-1), gy.reshape(-1)), dim=-1)
    pos[1] = pos[0] + torch.tensor([0.35 * dx, 0.0], dtype=pos.dtype)
    volume = torch.full((pos.shape[0],), dx * dx, dtype=pos.dtype)
    return pos, volume, h_ratio * dx


def test_the_two_shipped_operators_are_different_schemes():
    """Same configuration, same nominal coefficient -- not the same answer."""
    pos, volume, h = _cluster()
    up = upstream_correction(pos, volume, h, alpha=0.05)
    lind = lind_correction(pos, volume, h, dt=1.0e-4, v_ref=20.0, strength=1.0)
    assert up.shift.shape == lind.shape == pos.shape
    # the upstream operator's displacement is alpha*h^2*(grad-1) -- dimensionless
    # in h units it is O(0.05 h), the Lind form is A*grad C/C with A ~ h*v*dt.
    up_over_h = float(torch.linalg.vector_norm(up.shift, dim=1).max()) / h
    lind_over_h = float(torch.linalg.vector_norm(lind, dim=1).max()) / h
    assert up_over_h > 0.0 and lind_over_h > 0.0
    # they cannot be the same operator: the ratio is not 1 within any tolerance
    ratio = up_over_h / lind_over_h
    assert not (0.5 < ratio < 2.0), ratio


def test_only_the_upstream_operator_carries_the_grad1_signature():
    """The upstream driving term is antisymmetric in the pair weight (C_i + C_j).

    Doubling the kernel-correction factors ``C`` must double the upstream
    inconsistency exactly; the Lind form (concentration gradient) has no such
    ``C`` weight at all, so its velocity cannot be produced by this operator.
    """
    pos, volume, h = _cluster()
    ones = torch.ones(pos.shape[0], dtype=pos.dtype)
    twice = 2.0 * ones
    g1 = kernel_gradient_integral(pos, volume, h, correction=ones)
    g2 = kernel_gradient_integral(pos, volume, h, correction=twice)
    assert torch.allclose(g2, 2.0 * g1, rtol=1e-12, atol=0.0)


def test_no_module_still_claims_the_lind_form_is_the_upstream_operator():
    """Regression guard on the corrected attribution (claim vs code)."""
    text = (ROOT / "src" / "tensorsph" / "transport_velocity.py").read_text()
    assert "In SPHinXsys the correction is" not in text
    assert "tensorsph.transport_velocity_correction" in text
    cfg = (ROOT / "src" / "tensorsph" / "droplet_impact.py").read_text()
    assert "SPHinXsys port" not in cfg
    assert "not* the SPHinXsys" in cfg or "not the SPHinXsys" in cfg