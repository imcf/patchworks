"""Tests for cluster-allocation-aware CPU and memory detection.

On a shared node ``os.cpu_count()`` and ``psutil.virtual_memory().available``
describe the *machine*, not the slice of it this job was granted. Sizing work
against the machine is how a job walks into an OOM kill, so these are the
checks that would have caught it.
"""

import numpy as np

from patchworks import (
    auto_tile_shape_cellpose,
    cpu_allocation,
    safe_worker_count,
)
from patchworks._chunks import _get_available_memory

GIB = 1024**3


def test_cpu_allocation_prefers_slurm(monkeypatch):
    """SLURM's grant wins over the machine's core count."""
    monkeypatch.setenv("SLURM_CPUS_PER_TASK", "4")
    assert cpu_allocation() == 4


def test_cpu_allocation_ignores_junk_and_falls_back(monkeypatch):
    """A malformed or absent value falls through to the affinity mask."""
    monkeypatch.setenv("SLURM_CPUS_PER_TASK", "not-a-number")
    monkeypatch.delenv("SLURM_CPUS_ON_NODE", raising=False)
    assert cpu_allocation() >= 1

    monkeypatch.delenv("SLURM_CPUS_PER_TASK", raising=False)
    assert cpu_allocation() >= 1


def test_available_memory_takes_the_smallest_limit(monkeypatch):
    """The node's free RAM must never override a smaller allocation."""
    monkeypatch.setenv("SLURM_MEM_PER_NODE", str(16 * 1024))  # 16 GiB, in MB
    monkeypatch.setattr(
        "patchworks._chunks._cgroup_memory_limit", lambda: 512 * GIB
    )
    assert _get_available_memory() == 16 * GIB


def test_available_memory_respects_the_cgroup(monkeypatch):
    """With no SLURM hint, the cgroup ceiling still bounds the answer."""
    monkeypatch.delenv("SLURM_MEM_PER_NODE", raising=False)
    monkeypatch.delenv("SLURM_MEM_PER_CPU", raising=False)
    monkeypatch.setattr(
        "patchworks._chunks._cgroup_memory_limit", lambda: 2 * GIB
    )
    assert _get_available_memory() <= 2 * GIB


def test_mem_per_cpu_scales_with_the_allocation(monkeypatch):
    """SLURM_MEM_PER_CPU is per core, so it multiplies by the core grant."""
    monkeypatch.delenv("SLURM_MEM_PER_NODE", raising=False)
    monkeypatch.setenv("SLURM_MEM_PER_CPU", str(2 * 1024))  # 2 GiB per cpu
    monkeypatch.setenv("SLURM_CPUS_PER_TASK", "4")
    monkeypatch.setattr(
        "patchworks._chunks._cgroup_memory_limit", lambda: 512 * GIB
    )
    assert _get_available_memory() == 8 * GIB


def test_worker_count_is_bounded_by_the_allocation(monkeypatch):
    """The merge-style sizing must fit the grant, not the machine.

    This is the concrete failure: a 32 GiB job on a 512 GiB node, with chunks
    big enough that only a couple fit the grant.
    """
    monkeypatch.setenv("SLURM_CPUS_PER_TASK", "32")
    monkeypatch.setenv("SLURM_MEM_PER_NODE", str(32 * 1024))  # 32 GiB
    monkeypatch.setattr(
        "patchworks._chunks._cgroup_memory_limit", lambda: 512 * GIB
    )

    chunk_nbytes = int(np.prod((16, 1024, 1024))) * 4  # int32 tile ~64 MB
    n = safe_worker_count(chunk_nbytes * 40, fn_overhead=3)
    assert n < 32, "must not size itself to the core count when RAM is tighter"
    assert n >= 1


def test_gpu_tile_sizing_is_bounded_by_the_host_allocation(monkeypatch):
    """A big GPU must not excuse a tile the job's own host RAM can't hold.

    This is the concrete failure a `do_3D` nuclei segmentation hit: an ample
    GPU (24 GiB) sized the tile against VRAM alone, and the job -- granted
    only 1 GiB of host RAM here -- was SIGKILLed loading it, unrelated to
    ``nuclei_channel``. The sizer must take whichever budget is tighter.
    """
    monkeypatch.delenv("SLURM_MEM_PER_CPU", raising=False)
    monkeypatch.setenv("SLURM_MEM_PER_NODE", str(1024))  # 1 GiB
    monkeypatch.setattr(
        "patchworks._chunks._cgroup_memory_limit", lambda: 512 * GIB
    )

    tile = auto_tile_shape_cellpose(
        (128, 2048, 2048),
        "uint16",
        diameter=30,
        do_3D=True,
        use_gpu=True,
        gpu_memory=24 * GIB,
    )
    generous = auto_tile_shape_cellpose(
        (128, 2048, 2048),
        "uint16",
        diameter=30,
        do_3D=True,
        use_gpu=True,
        gpu_memory=24 * GIB,
        available_memory=64 * GIB,
    )
    assert np.prod(tile) < np.prod(generous), (
        "the 1 GiB host grant must shrink the tile below what the same "
        "24 GiB GPU would otherwise allow"
    )
