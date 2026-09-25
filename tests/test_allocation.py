"""Tests for cluster-allocation-aware CPU and memory detection.

On a shared node ``os.cpu_count()`` and ``psutil.virtual_memory().available``
describe the *machine*, not the slice of it this job was granted. Sizing work
against the machine is how a job walks into an OOM kill, so these are the
checks that would have caught it.
"""

import sys
import types

import numpy as np
import pytest

from patchworks import (
    auto_tile_shape_cellpose,
    cpu_allocation,
    safe_worker_count,
)
from patchworks._chunks import _get_available_memory

GIB = 1024**3


@pytest.fixture(autouse=True)
def _plenty_of_free_ram(monkeypatch):
    """Report 1 TiB of free RAM, so only the limits under test can bind.

    Without this every expected value depends on the test machine: a runner
    with less free RAM than a test's SLURM grant gets the machine's figure.
    """
    fake = types.SimpleNamespace(
        virtual_memory=lambda: types.SimpleNamespace(available=1024 * GIB)
    )
    monkeypatch.setitem(sys.modules, "psutil", fake)


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


def test_auto_tile_shape_cellpose_budgets_for_the_anisotropy_resize():
    """Cellpose resizes a do_3D tile to z * anisotropy planes before the net

    runs, so budgeting against the unscaled z hands the GPU a tile that is
    `anisotropy` times bigger than the estimate. Auto-deriving anisotropy
    turned that from dormant into live, so the sizer has to know about it.
    """
    from patchworks import auto_tile_shape_cellpose

    kwargs = dict(
        shape=(64, 4096, 4096),
        dtype="uint16",
        do_3D=True,
        use_gpu=True,
        gpu_memory=8 * 1024**3,
        available_memory=64 * 1024**3,
    )
    isotropic = auto_tile_shape_cellpose(**kwargs)
    anisotropic = auto_tile_shape_cellpose(**kwargs, anisotropy=2.215)

    # z is pinned to the full extent either way; the cost is paid in y/x.
    assert anisotropic[0] == isotropic[0]
    assert anisotropic[1] < isotropic[1]
    assert anisotropic[2] < isotropic[2]
    # An anisotropy at or below 1 cannot *grow* the budget.
    assert auto_tile_shape_cellpose(**kwargs, anisotropy=1.0) == isotropic
    assert auto_tile_shape_cellpose(**kwargs, anisotropy=0.5) == isotropic


def test_auto_tile_shape_cellpose_budgets_for_the_diameter_rescale():
    """Cellpose rescales the tile by 30/diameter on *every* axis before the

    net runs, so a diameter below 30 upsamples cubically -- diameter 15 is
    8x the voxels. The sizer used diameter only as a minimum-tile floor, so
    it handed the GPU a tile 8x bigger than it had budgeted for.
    """
    from patchworks import auto_tile_shape_cellpose

    kwargs = dict(
        shape=(64, 4096, 4096),
        dtype="uint16",
        do_3D=True,
        use_gpu=True,
        gpu_memory=8 * 1024**3,
        available_memory=64 * 1024**3,
    )
    native = auto_tile_shape_cellpose(**kwargs, diameter=30)
    upsampled = auto_tile_shape_cellpose(**kwargs, diameter=15)

    # 30 -> rescale 1.0 (no resize); 15 -> rescale 2.0, so the tile must shrink.
    assert upsampled[1] < native[1]
    assert upsampled[2] < native[2]
    # A diameter above 30 predicts a downsample; that must not *grow* the
    # tile, since this factor is a safety margin, not a measurement.
    assert auto_tile_shape_cellpose(**kwargs, diameter=60)[1] <= native[1]


def _fake_cgroup(monkeypatch, tmp_path, proc_lines):
    from patchworks import _chunks

    root = tmp_path / "cgroup"
    root.mkdir()
    proc = tmp_path / "proc_cgroup"
    proc.write_text("\n".join(proc_lines) + "\n")
    monkeypatch.setattr(_chunks, "_CGROUP_ROOT", str(root))
    monkeypatch.setattr(_chunks, "_PROC_CGROUP", str(proc))
    return root


def test_cgroup_v2_limit_on_the_jobs_own_cgroup(monkeypatch, tmp_path):
    """SLURM on cgroup v2 caps the job's nested cgroup, not the mount root.

    Reading only /sys/fs/cgroup/memory.max found nothing there, so the job's
    ceiling was ignored and sizing fell back to the whole node's free RAM.
    """
    from patchworks._chunks import _cgroup_memory_limit

    job = "system.slice/slurmstepd.scope/job_42/step_0"
    root = _fake_cgroup(monkeypatch, tmp_path, [f"0::/{job}"])
    leaf = root / job
    leaf.mkdir(parents=True)
    (leaf / "memory.max").write_text(f"{32 * GIB}\n")
    # Page cache is reclaimable; only anon memory reduces the headroom.
    (leaf / "memory.stat").write_text(f"anon {4 * GIB}\nfile {27 * GIB}\n")
    (root / "system.slice" / "memory.max").write_text("max\n")

    assert _cgroup_memory_limit() == 28 * GIB


def test_cgroup_tightest_ancestor_wins(monkeypatch, tmp_path):
    from patchworks._chunks import _cgroup_memory_limit

    root = _fake_cgroup(monkeypatch, tmp_path, ["0::/a/b"])
    (root / "a" / "b").mkdir(parents=True)
    (root / "a" / "b" / "memory.max").write_text(f"{64 * GIB}\n")
    (root / "a" / "memory.max").write_text(f"{16 * GIB}\n")

    assert _cgroup_memory_limit() == 16 * GIB


def test_cgroup_v1_nested_memory_limit(monkeypatch, tmp_path):
    from patchworks._chunks import _cgroup_memory_limit

    root = _fake_cgroup(monkeypatch, tmp_path, ["4:memory:/slurm/job_7"])
    leaf = root / "memory" / "slurm" / "job_7"
    leaf.mkdir(parents=True)
    (leaf / "memory.limit_in_bytes").write_text(f"{8 * GIB}\n")
    (leaf / "memory.stat").write_text(f"cache 1\ntotal_rss {GIB}\n")
    (root / "memory" / "memory.limit_in_bytes").write_text(f"{2**63 - 4096}")

    assert _cgroup_memory_limit() == 7 * GIB


def test_no_cgroup_limit_is_none(monkeypatch, tmp_path):
    from patchworks._chunks import _cgroup_memory_limit

    root = _fake_cgroup(monkeypatch, tmp_path, ["0::/"])
    (root / "memory.max").write_text("max\n")
    assert _cgroup_memory_limit() is None


def test_cpu_quota_caps_the_allocation(monkeypatch, tmp_path):
    """docker --cpus=2.5 leaves every core in the affinity mask."""
    from patchworks import _chunks

    monkeypatch.delenv("SLURM_CPUS_PER_TASK", raising=False)
    monkeypatch.delenv("SLURM_CPUS_ON_NODE", raising=False)
    root = _fake_cgroup(monkeypatch, tmp_path, ["0::/ctr"])
    (root / "ctr").mkdir()
    (root / "ctr" / "cpu.max").write_text("250000 100000\n")
    monkeypatch.setattr(
        _chunks.os,
        "sched_getaffinity",
        lambda pid: set(range(64)),
        raising=False,
    )

    assert _chunks._cgroup_cpu_limit() == 3
    assert _chunks.cpu_allocation() == 3
