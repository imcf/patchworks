"""Timing benchmarks for patchworks' hot paths.

    python benchmarks/bench.py out.json          # run, write JSON
    python benchmarks/compare.py base.json head.json

Each case runs a few times on synthetic but realistic data and keeps the
fastest run (the least disturbed by a noisy machine). A case the installed
patchworks cannot run (e.g. a feature newer than a base branch) is recorded
as null, so a head/base comparison still works across such changes.
"""

from __future__ import annotations

import json
import shutil
import sys
import tempfile
import time
from pathlib import Path

import numpy as np
import zarr

REPEATS = 3


def _blob_labels(shape, n, seed=0, r=4):
    """Many small blobs, labelled per tile as a stage store holds them."""
    from scipy import ndimage as ndi

    rng = np.random.default_rng(seed)
    mask = np.zeros(shape, bool)
    radii = np.array([max(1, min(r, (d - 1) // 2)) for d in shape])
    centres = rng.integers(radii, np.array(shape) - radii, (n, len(shape)))
    for c in centres:
        mask[tuple(slice(x - q, x + q) for x, q in zip(c, radii))] = True
    return ndi.label(mask)[0].astype("int32")


def _staged(tmp, shape=(8, 1024, 1024), tile=(8, 256, 256), n=4000):
    """A stage store: dense 1..n ids per tile, plus each tile's count."""
    lab = _blob_labels(shape, n)
    path = str(tmp / "stage.zarr")
    arr = zarr.open_group(path, mode="w").create_array(
        "staged", shape=shape, chunks=tile, dtype="int32"
    )
    from patchworks._chunks import chunk_slices
    from patchworks._relabel import relabel_sequential_array

    counts = []
    for sl in chunk_slices(shape, tile):
        block = relabel_sequential_array(lab[sl]).astype("int32")
        arr[sl] = block
        counts.append(int(block.max()))
    return path, counts


def case_merge_touch(tmp):
    from patchworks._merge import zarr_native_merge

    path, counts = _staged(tmp)
    t = time.perf_counter()
    zarr_native_merge(
        path,
        "staged",
        str(tmp / "out.zarr"),
        "labels",
        n_workers=2,
        label_counts=counts,
    )
    return time.perf_counter() - t


def case_merge_renumber(tmp):
    """The merge without per-tile counts: includes the renumber pass."""
    from patchworks._merge import zarr_native_merge

    path, _ = _staged(tmp)
    t = time.perf_counter()
    zarr_native_merge(
        path, "staged", str(tmp / "out.zarr"), "labels", n_workers=2
    )
    return time.perf_counter() - t


def case_image_pyramid(tmp):
    from patchworks.plugins.ome_zarr import to_ome_zarr

    img = np.random.default_rng(0).integers(0, 4000, (16, 1024, 1024))
    img = img.astype("uint16")
    t = time.perf_counter()
    to_ome_zarr(img, tmp / "img.zarr", axes="zyx", n_levels=5, progress=False)
    return time.perf_counter() - t


def case_label_pyramid(tmp):
    from patchworks.plugins.ome_zarr import (
        add_pyramid,
        to_ome_zarr,
        write_labels,
    )

    lab = _blob_labels((16, 1024, 1024), 6000)
    store = to_ome_zarr(
        np.zeros((1, 8, 8), "uint16"),
        tmp / "i.zarr",
        axes="zyx",
        n_levels=1,
        progress=False,
    )
    write_labels(store, lab, name="l", n_levels=1, progress=False)
    t = time.perf_counter()
    add_pyramid(f"{store}/labels/l", n_levels=5, progress=False)
    return time.perf_counter() - t


def case_filter_by_size(tmp):
    from patchworks import filter_labels_by_size

    lab = _blob_labels((16, 1024, 1024), 6000)
    path = str(tmp / "f.zarr")
    z = zarr.open_group(path, mode="w").create_array(
        "l", shape=lab.shape, chunks=(16, 256, 256), dtype="int32"
    )
    z[:] = lab
    t = time.perf_counter()
    filter_labels_by_size(path, "l", min_voxels=50)
    return time.perf_counter() - t


def case_tile_process(tmp):
    import dask.array as da
    from scipy import ndimage as ndi

    from patchworks import tile_process

    img = (_blob_labels((8, 512, 512), 1500) > 0).astype("uint16") * 1000
    arr = da.from_array(img, chunks=(8, 128, 128))
    t = time.perf_counter()
    tile_process(
        arr,
        lambda b: ndi.label(b > 500)[0].astype("int32"),
        overlap=8,
        write_to=tmp / "o.zarr",
        progress=False,
    )
    return time.perf_counter() - t


CASES = {
    name[5:]: fn
    for name, fn in sorted(globals().items())
    if name.startswith("case_")
}


def run() -> dict[str, float | None]:
    results: dict[str, float | None] = {}
    for name, fn in CASES.items():
        times = []
        for _ in range(REPEATS):
            tmp = Path(tempfile.mkdtemp(prefix="pwbench_"))
            try:
                times.append(fn(tmp))
            except Exception as exc:  # a case this version cannot run
                print(f"{name}: skipped ({type(exc).__name__}: {exc})")
                times = []
                break
            finally:
                shutil.rmtree(tmp, ignore_errors=True)
        results[name] = min(times) if times else None
        if times:
            print(f"{name}: {results[name]:.3f}s")
    return results


if __name__ == "__main__":
    out = run()
    if len(sys.argv) > 1:
        Path(sys.argv[1]).write_text(json.dumps(out, indent=2))
