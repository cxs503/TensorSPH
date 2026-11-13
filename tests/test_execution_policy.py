"""Execution-policy port: OSS invariants and launch-geometry invariance.

Anchor: ``XVAL-SPH-SYCL-EXECUTION-POLICY-NOT-TRANSFERABLE``.  The properties
asserted here are the ones the port claims to reproduce from
``Xiangyu-Hu/SPHinXsys`` (files and sha256 in
``artifacts/xval_execution_policy/provenance.json``):

1. one process-wide execution instance, not copyable / picklable;
2. the main policy is fixed before any kernel runs (compile-time analogue);
3. padded launch geometry equals the OSS conditional round-up, and every
   logical index is visited exactly once with the tail masked;
4. the padded geometry does not change the numbers (this is the invariant the
   mask exists to protect);
5. dropping the mask is measurably wrong -- the negative control.
"""
from __future__ import annotations

import copy
import math
import pickle

import pytest
import torch

from tensorsph import execution_policy as ep
from tensorsph.execution_policy import ExecutionInstance, execution_instance
from tensorsph.neighbor_search import PairListCache
from tensorsph.wcsph import kernels, swept
from tensorsph.wcsph.cases import lattice_box


@pytest.fixture(scope="module")
def density_case():
    """A compact 2-D lattice with its pair list -- the sweep's work load."""
    dx = 0.02
    h = 1.3 * dx
    pos = lattice_box([0.0, 0.0], [0.4, 0.2], dx, 2)
    cache = PairListCache(kernels.SUPPORT_FACTOR["quintic"] * h, 0.0,
                          fluid_count=pos.shape[0])
    pairs, _ = cache.get(pos)
    n = pos.shape[0]
    mass = torch.full((n,), 1000.0 * dx * dx, dtype=torch.float64)
    h_t = torch.full((n,), h, dtype=torch.float64)
    return {"pos": pos, "mass": mass, "h": h_t, "pairs": pairs, "n": n, "h0": h}


# --------------------------------------------------------------------------- #
# 1. non-transferability
# --------------------------------------------------------------------------- #


def test_instance_is_a_singleton():
    assert ExecutionInstance.get_instance() is ExecutionInstance.get_instance()
    assert execution_instance is ExecutionInstance.get_instance()


@pytest.mark.parametrize("copier", [copy.copy, copy.deepcopy])
def test_instance_refuses_copying(copier):
    with pytest.raises(TypeError, match="not copyable|not deep-copyable"):
        copier(execution_instance)


def test_instance_refuses_pickling():
    with pytest.raises(TypeError, match="not picklable"):
        pickle.dumps(execution_instance)


def test_main_policy_is_frozen_at_import():
    with pytest.raises(RuntimeError, match="MAIN_EXECUTION_POLICY is fixed"):
        ep.set_main_execution_policy(ep.par_device)
    # par_ck is the OSS alias for "whatever the main policy is"
    assert ep.par_ck is ep.MAIN_EXECUTION_POLICY
    if ep.USE_DEVICE_BACKEND:
        assert isinstance(ep.MAIN_EXECUTION_POLICY, ep.ParallelDevicePolicy)
    else:
        assert isinstance(ep.MAIN_EXECUTION_POLICY, ep.ParallelPolicy)


# --------------------------------------------------------------------------- #
# 2. padded launch geometry == OSS getUniformNdRange
# --------------------------------------------------------------------------- #


def _ceil_formula(global_size: int, local_size: int) -> int:
    """Independent re-derivation: ceil(n/l)*l, valid for ``n > 0``."""
    return -(-global_size // local_size) * local_size


@pytest.mark.parametrize("n", [1, 2, 7, 8, 63, 64, 65, 199, 200, 1000, 4097])
@pytest.mark.parametrize("local", [1, 2, 3, 8, 16, 64, 128, 1024])
def test_uniform_nd_range_matches_re_derivation(n, local):
    padded, local_out = ep.get_uniform_nd_range(n, local)
    assert local_out == local
    assert padded == _ceil_formula(n, local)
    assert padded % local == 0
    assert padded >= n
    assert padded - n < local


def test_exact_multiple_is_returned_unchanged():
    """The OSS expression is conditional, not a bare ceil: ``200/8 -> 200``.

    A naive ``(n//l + 1) * l`` would give 208 here, so this asserts the branch
    was ported rather than approximated.
    """
    assert ep.get_uniform_nd_range(200, 8) == (200, 8)
    assert ep.get_uniform_nd_range(0, 8) == (0, 8)


def test_bad_work_group_size_raises():
    for bad in (0, -4):
        with pytest.raises(ValueError):
            ep.get_uniform_nd_range(100, bad)
    with pytest.raises(ValueError):
        ep.get_uniform_nd_range(-1, 8)
    with pytest.raises(ValueError):
        execution_instance.set_work_group_size(0)


@pytest.mark.parametrize("n,local", [(10, 4), (200, 64), (200, 7), (1, 64)])
def test_blocks_cover_every_index_exactly_once(n, local):
    seen = torch.zeros(n, dtype=torch.long)
    padded = 0
    idle = 0
    for block in ep.iterate_blocks(n, local):
        padded += block.size
        idle += block.idle
        for index, live in zip(block.indices.tolist(), block.mask.tolist()):
            if live:
                assert 0 <= index < n
                seen[index] += 1
            else:
                assert index >= n
    assert padded == _ceil_formula(n, local)
    assert idle == padded - n
    assert torch.equal(seen, torch.ones(n, dtype=torch.long))


def test_work_item_counts_agree_with_blocks():
    for n, local in [(200, 64), (1000, 128), (7, 8)]:
        stats = ep.work_item_counts(n, local)
        assert stats["padded_global_size"] == _ceil_formula(n, local)
        assert stats["active_work_items"] == n
        assert stats["idle_work_items"] == stats["padded_global_size"] - n
        assert math.isclose(stats["idle_fraction"],
                            (stats["padded_global_size"] - n) / n)
        assert stats["work_groups"] * local == stats["padded_global_size"]


def test_device_clamp_sets_oss_default_work_group(density_case):
    """``work_group_size_`` is 128 until the device is resolved, then <= 64."""
    device = execution_instance.get_device()
    assert device.type in ("cpu", "cuda")
    assert execution_instance.work_group_size <= 64
    execution_instance.set_work_group_size(7)
    assert execution_instance.work_group_size == 7
    execution_instance.set_work_group_size(64)


# --------------------------------------------------------------------------- #
# 3. the invariant: launch geometry does not change the numbers
# --------------------------------------------------------------------------- #


@pytest.mark.parametrize("local", [1, 2, 3, 7, 8, 16, 31, 64, 127, 256, 1024])
def test_swept_matches_vectorised_reference(density_case, local):
    case = density_case
    reference = swept.summation_density_vectorised(
        case["pos"], case["mass"], case["h"], 2, case["pairs"])
    got = swept.summation_density_swept(
        case["pos"], case["mass"], case["h"], 2, case["pairs"], local_size=local)
    relative = ((got - reference).abs() / reference.abs()).max().item()
    assert relative < 1.0e-12, f"local_size={local} relative error {relative:.3e}"


def test_policy_invariance_spread_is_zero(density_case):
    """Spread across work-group sizes: 0 by construction, asserted numerically."""
    case = density_case
    runs = [swept.summation_density_swept(
        case["pos"], case["mass"], case["h"], 2, case["pairs"], local_size=local)
        for local in (1, 8, 64, 1000)]
    spread = max((run - runs[0]).abs().max().item() for run in runs)
    assert spread < 1.0e-12


def test_default_work_group_size_is_used(density_case):
    """``local_size=None`` takes the instance's work-group size."""
    case = density_case
    execution_instance.set_work_group_size(13)
    try:
        got = swept.summation_density_swept(
            case["pos"], case["mass"], case["h"], 2, case["pairs"])
        explicit = swept.summation_density_swept(
            case["pos"], case["mass"], case["h"], 2, case["pairs"], local_size=13)
        assert torch.equal(got, explicit)
    finally:
        execution_instance.set_work_group_size(64)


# --------------------------------------------------------------------------- #
# 4. negative control: the mask is load bearing
# --------------------------------------------------------------------------- #


def test_unmasked_tail_breaks_the_result(density_case):
    """Without the tail mask the padded work items double-count the last particle.

    The error is the *measured* consequence of the antipattern, not an
    assertion about intent: with ``n = 200`` and ``local_size = 64`` there are
    56 padded work items, all clamped onto particle 199.
    """
    case = density_case
    reference = swept.summation_density_vectorised(
        case["pos"], case["mass"], case["h"], 2, case["pairs"])
    bad = swept.summation_density_swept(
        case["pos"], case["mass"], case["h"], 2, case["pairs"],
        local_size=64, mask_tail=False)
    relative = ((bad - reference).abs() / reference.abs()).max().item()
    assert relative > 0.15, f"negative control should fail the 15% gate, got {relative:.3e}"
    # the damage is localised on the clamped (last) particle
    assert torch.argmax((bad - reference).abs()).item() == case["n"] - 1


def test_unmasked_tail_is_exact_when_n_is_a_multiple(density_case):
    """No idle work items -> nothing to mask -> the control reproduces the reference."""
    case = density_case
    n = case["n"]
    reference = swept.summation_density_vectorised(
        case["pos"], case["mass"], case["h"], 2, case["pairs"])
    bad = swept.summation_density_swept(
        case["pos"], case["mass"], case["h"], 2, case["pairs"],
        local_size=n, mask_tail=False)
    assert ((bad - reference).abs() / reference.abs()).max().item() < 1.0e-12


def test_sweep_diagnostics_report_launch_cost(density_case):
    case = density_case
    stats = swept.sweep_diagnostics(case["n"], 64)
    assert stats["padded_global_size"] == 256
    assert stats["idle_work_items"] == 56
    assert stats["work_groups"] == 4
