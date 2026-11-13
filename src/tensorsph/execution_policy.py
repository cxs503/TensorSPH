"""Execution policy: a port of SPHinXsys' device-execution policy layer.

Why this module exists
----------------------
TensorSPH had no notion of an *execution policy*: every kernel is a plain
vectorised tensor expression, so there is nothing that a caller could pass
around, share or (mis-)transfer.  SPHinXsys makes the opposite choice, and the
choice is deliberate -- its SYCL backend keeps the policy and the resources it
resolves to **non-transferable**:

* the policy is a *tag type* picked at compile time
  (``SPHINXSYS_USE_SYCL`` selects ``MainExecutionPolicy`` between
  ``ParallelPolicy`` and ``ParallelDevicePolicy``), so a policy value cannot be
  chosen, handed over or serialised at run time;
* the state the tag resolves to (the queue and the work-group size) lives in
  ``ExecutionInstance``, whose copy constructor and assignment operator are
  ``= delete``, and which is reachable only through ``getInstance()`` -- a
  process-wide singleton;
* device memory is allocated and copied through that one queue
  (``sycl::malloc_device<T>(size, execution_instance.getQueue())``,
  ``queue.memcpy(...)``), so a device pointer is bound to the instance that
  created it and cannot be moved to another queue/device.

Provenance (all facts above read from source, nothing transcribed by hand;
sha256 of each file is recorded by the benchmark in
``artifacts/xval_execution_policy/provenance.json``):

* ``src/shared/particle_dynamics/execution/execution_policy.h`` --
  ``DeviceExecution`` / ``ParallelDevicePolicy`` / ``seq_device`` /
  ``par_ck`` / the ``#if SPHINXSYS_USE_SYCL`` alias of ``MainExecutionPolicy``.
* ``src/src_sycl/shared/particle_dynamics/implementation_sycl.h`` --
  ``ExecutionInstance`` (deleted copy ctor / assignment, ``getInstance()``,
  ``getQueue()`` with ``sycl::default_selector_v`` and
  ``work_group_size_ = SMIN(max_work_group_size, 64)``, the static
  ``getUniformNdRange``) and the ``allocateDeviceOnly``/``copyToDevice``
  family that binds device memory to that queue.
* ``src/src_sycl/shared/common/algorithm_primitive_sycl.hpp`` --
  ``generic_for``: the launch uses the *padded* nd-range and the kernel body
  carries the tail mask ``if (index.get_global_id(0) < loop_bound)``.

What transfers to a Python/torch code base, and what does not
-------------------------------------------------------------
Transfers (implemented here):

* one process-wide, non-copyable execution context holding the device and the
  work-group size (``ExecutionInstance``);
* a padded launch geometry (``get_uniform_nd_range`` / ``iterate_blocks``) with
  an *explicit* tail mask that the kernel body must consume, plus the
  bookkeeping needed to prove the mask is complete;
* the invariant that is actually load-bearing for numerics: **the padded launch
  geometry must not change the result** (see
  ``tensorsph.wcsph.swept.summation_density_swept``).

Does not transfer: the compile-time tag.  Python has no compile-time switch, so
the analogue is a module constant frozen at import (``MAIN_EXECUTION_POLICY``);
``set_main_execution_policy`` therefore raises instead of silently re-binding a
process-wide singleton half-way through a run.  SYCL's device-memory ownership
machinery also has no analogue: torch owns tensor storage and
``Tensor.to(device)`` already refuses cross-device arithmetic, so no
``malloc_device``/``memcpy`` port is attempted.
"""
from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Iterator

import torch

__all__ = [
    "SequencedPolicy",
    "UnsequencedPolicy",
    "ParallelPolicy",
    "ParallelUnsequencedPolicy",
    "DeviceExecution",
    "ParallelDevicePolicy",
    "SequencedDevicePolicy",
    "seq",
    "unseq",
    "par",
    "par_unseq",
    "par_device",
    "seq_device",
    "par_ck",
    "MAIN_EXECUTION_POLICY",
    "USE_DEVICE_BACKEND",
    "set_main_execution_policy",
    "ExecutionInstance",
    "execution_instance",
    "WorkBlock",
    "get_uniform_nd_range",
    "iterate_blocks",
    "work_item_counts",
]

# --------------------------------------------------------------------------- #
# Policy tags -- mirror of execution_policy.h
# --------------------------------------------------------------------------- #


class SequencedPolicy:
    """Sequential, ordered execution (``SequencePolicy`` in the OSS header)."""


class UnsequencedPolicy:
    """Vectorisable but not reorderable execution."""


class ParallelPolicy:
    """Host-parallel execution."""


class ParallelUnsequencedPolicy:
    """Host-parallel and vectorisable execution."""


class DeviceExecution:
    """Marker base of ``DeviceExecution<>`` (the empty template pack)."""


class ParallelDevicePolicy(DeviceExecution, ParallelPolicy):
    """``DeviceExecution<ParallelPolicy>``: parallel on an accelerator."""


class SequencedDevicePolicy(DeviceExecution, SequencedPolicy):
    """``DeviceExecution<SequencedPolicy>``: sequential on an accelerator."""


seq = SequencedPolicy()
unseq = UnsequencedPolicy()
par = ParallelPolicy()
par_unseq = ParallelUnsequencedPolicy()
par_device = ParallelDevicePolicy()
seq_device = SequencedDevicePolicy()

#: Analogue of ``SPHINXSYS_USE_SYCL``.  Read once, at import: a policy is a
#: build-time choice in the OSS design, so re-reading the environment later
#: would give a policy that changes under a running kernel.
USE_DEVICE_BACKEND = os.environ.get("TENSORSPH_USE_DEVICE", "").strip().lower() in {
    "1",
    "true",
    "yes",
    "on",
}

MAIN_EXECUTION_POLICY: DeviceExecution | ParallelPolicy = (
    par_device if USE_DEVICE_BACKEND else par
)
#: ``par_ck`` is the OSS name for "whatever the main policy is".
par_ck = MAIN_EXECUTION_POLICY


def set_main_execution_policy(policy) -> None:
    """Refuse to re-bind the main policy at run time.

    The OSS header resolves ``MainExecutionPolicy`` with ``#if
    SPHINXSYS_USE_SYCL``; accepting a run-time change would break the property
    this port is about (one policy per process, chosen before any kernel runs).
    """
    raise RuntimeError(
        "MAIN_EXECUTION_POLICY is fixed at import time (analogue of the "
        f"SPHINXSYS_USE_SYCL compile-time switch); it is {MAIN_EXECUTION_POLICY!r}"
    )


# --------------------------------------------------------------------------- #
# Execution instance -- non-copyable, process-wide
# --------------------------------------------------------------------------- #


class ExecutionInstance:
    """Process-wide device context: device + work-group size.

    Deliberately *not* copyable, picklable or assignable, matching the deleted
    copy constructor and assignment operator of the OSS class.  Use
    :meth:`get_instance`.
    """

    _instance: "ExecutionInstance | None" = None

    def __init__(self) -> None:  # pragma: no cover - guarded by get_instance
        # OSS private ctor: ``ExecutionInstance() : work_group_size_(128)``.
        # The lazy ``getQueue()`` then clamps to ``SMIN(max_work_group_size,
        # 64)`` on first use, so 128 is only visible before the first launch.
        self._work_group_size: int = 128
        self._device: torch.device | None = None

    # -- non-transferability ------------------------------------------------ #
    def __copy__(self):
        raise TypeError(
            "ExecutionInstance is not copyable (copy ctor deleted in the OSS "
            "class); use ExecutionInstance.get_instance()"
        )

    def __deepcopy__(self, memo):
        raise TypeError(
            "ExecutionInstance is not deep-copyable (copy ctor deleted in the "
            "OSS class); use ExecutionInstance.get_instance()"
        )

    def __reduce__(self):
        raise TypeError(
            "ExecutionInstance is not picklable (copy ctor deleted in the OSS "
            "class); the device context cannot be transferred between processes"
        )

    def __setstate__(self, state):  # pragma: no cover - defensive
        raise TypeError("ExecutionInstance state cannot be restored from a pickle")

    # -- singleton ---------------------------------------------------------- #
    @classmethod
    def get_instance(cls) -> "ExecutionInstance":
        if cls._instance is None:
            cls._instance = cls()
        return cls._instance

    @classmethod
    def _reset_for_tests(cls) -> None:  # pragma: no cover - tests only
        cls._instance = None

    # -- device (analogue of getQueue) -------------------------------------- #
    def get_device(self) -> torch.device:
        """Lazy device selection, analogue of ``sycl::default_selector_v``.

        OSS: the queue is created on first use and its device's maximum
        work-group size clamps the launch geometry to 64.  Torch has no
        work-group concept; the cap uses ``max_threads_per_block`` on CUDA and
        the OSS constant 64 on CPU (deviation recorded in the report).
        """
        if self._device is None:
            if torch.cuda.is_available():  # pragma: no cover - no GPU in CI
                self._device = torch.device("cuda")
                cap = int(torch.cuda.get_device_properties(0).max_threads_per_block)
            else:
                self._device = torch.device("cpu")
                cap = 64
            self._work_group_size = min(cap, 64)
        return self._device

    @property
    def device(self) -> torch.device:
        return self.get_device()

    @property
    def work_group_size(self) -> int:
        return self._work_group_size

    def set_work_group_size(self, work_group_size: int) -> None:
        """Mutate the launch geometry (public in OSS: ``setWorkGroupSize``)."""
        work_group_size = int(work_group_size)
        if work_group_size < 1:
            raise ValueError("work_group_size must be a positive integer")
        self._work_group_size = work_group_size

    # -- launch geometry ---------------------------------------------------- #
    @staticmethod
    def get_uniform_nd_range(global_size: int, local_size: int) -> tuple[int, int]:
        return get_uniform_nd_range(global_size, local_size)

    def get_uniform_nd_range_for(self, global_size: int) -> tuple[int, int]:
        """``getUniformNdRange(global_size)`` using the instance's work-group.

        The device is touched first on purpose: in the OSS class the work-group
        size is only clamped to the device maximum inside the lazy ``getQueue()``,
        so a launch that never resolved the queue would use the raw 128.
        """
        self.get_device()
        return get_uniform_nd_range(global_size, self._work_group_size)


execution_instance = ExecutionInstance.get_instance()


# --------------------------------------------------------------------------- #
# Padded launch geometry -- replica of getUniformNdRange + generic_for
# --------------------------------------------------------------------------- #


def get_uniform_nd_range(global_size: int, local_size: int) -> tuple[int, int]:
    """Round ``global_size`` up to a whole number of work-groups.

    Replica of the OSS static helper, including its behaviour for an exact
    multiple (the global size is returned unchanged, not recomputed).::

        global_size % local_size ? (global_size / local_size + 1) * local_size
                                 : global_size

    The OSS expression divides by ``local_size`` without a guard; a zero
    work-group size is undefined behaviour in SYCL and raises here.
    """
    global_size = int(global_size)
    local_size = int(local_size)
    if local_size < 1:
        raise ValueError("local_size must be a positive integer")
    if global_size < 0:
        raise ValueError("global_size must be non-negative")
    if global_size % local_size:
        return (global_size // local_size + 1) * local_size, local_size
    return global_size, local_size


@dataclass(frozen=True)
class WorkBlock:
    """One launched work-group: padded indices + the tail mask.

    ``indices`` may contain indices ``>= global_size`` -- those are the padded
    work items that the kernel body must drop (``mask`` is ``False``).  This is
    exactly the ``if (global_id < loop_bound)`` guard of ``generic_for``, moved
    out of the kernel so that a missing mask becomes visible to a test.
    """

    indices: torch.Tensor
    mask: torch.Tensor
    offset: int

    @property
    def size(self) -> int:
        return int(self.indices.numel())

    @property
    def active(self) -> int:
        return int(self.mask.sum().item())

    @property
    def idle(self) -> int:
        return self.size - self.active


def iterate_blocks(
    global_size: int, local_size: int | None = None, *, device: torch.device | None = None
) -> Iterator[WorkBlock]:
    """Yield the padded work-groups of a launch over ``range(global_size)``."""
    if local_size is None:
        local_size = execution_instance.work_group_size
    padded, local = get_uniform_nd_range(int(global_size), int(local_size))
    if padded == 0:
        return
    dev = torch.device("cpu") if device is None else device
    arange = torch.arange(padded, dtype=torch.long, device=dev)
    for offset in range(0, padded, local):
        idx = arange[offset:offset + local]
        yield WorkBlock(indices=idx, mask=idx < int(global_size), offset=offset)


def work_item_counts(global_size: int, local_size: int | None = None) -> dict:
    """Launch bookkeeping: padded / active / idle work items and the overhead."""
    n = int(global_size)
    if local_size is None:
        local_size = execution_instance.work_group_size
    padded, local = get_uniform_nd_range(n, local_size)
    idle = padded - n
    return {
        "global_size": n,
        "local_size": int(local),
        "padded_global_size": padded,
        "active_work_items": n,
        "idle_work_items": idle,
        "idle_fraction": (idle / n) if n else 0.0,
        "work_groups": (padded // int(local)) if int(local) else 0,
    }
