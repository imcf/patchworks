<p align="center">
  <img src="https://raw.githubusercontent.com/imcf/patchworks/main/docs/assets/logo.png" alt="patchworks logo" width="220">
</p>

# patchworks

[![PyPI](https://img.shields.io/pypi/v/patchworks.svg)](https://pypi.org/project/patchworks/)
[![Python versions](https://img.shields.io/pypi/pyversions/patchworks.svg)](https://pypi.org/project/patchworks/)
[![License: GPLv3](https://img.shields.io/badge/License-GPLv3-blue.svg)](https://www.gnu.org/licenses/gpl-3.0)
[![Docs](https://img.shields.io/badge/docs-imcf.one%2Fpatchworks-blue)](https://imcf.one/patchworks/)

> Tiled processing of arbitrarily large images — any image, any function.

```text
┌──────┬──────┬──────┐     fn(tile) → labels      ┌──────┬──────┬──────┐
│ tile │ tile │ tile │  ─────────────────────►    │  1   │  2   │  3   │
├──────┼──────┼──────┤                            ├──────┼──────┼──────┤
│ tile │ tile │ tile │                            │  4   │  5   │  6   │   globally
├──────┼──────┼──────┤                            ├──────┼──────┼──────┤   consistent
│ tile │ tile │ tile │                            │  7   │  8   │  9   │   labels
└──────┴──────┴──────┘                            └──────┴──────┴──────┘
```

patchworks splits a large image into tiles, runs **any callable** on each
tile in parallel, and merges the results into a globally consistent label array.
It handles terabyte-scale images without loading them into memory.

---

## Installation

```bash
pip install patchworks
```

Optional extras:

```bash
pip install "patchworks[gpu]"      # GPU VRAM querying (nvidia-ml-py)
pip install "patchworks[cellpose]" # Cellpose plugin
pip install "patchworks[bioio]"    # convert any image format to OME-ZARR
pip install "patchworks[imaris]"   # convert Imaris .ims files to OME-ZARR
pip install "patchworks[napari]"   # interactive napari viewer plugin
pip install "patchworks[all]"      # Everything, incl. the napari viewer
```

> `bioio` reads CZI/LIF/ND2/OME-TIFF/… The `[bioio]` extra bundles the common
> native readers (`bioio-nd2`, `bioio-ome-tiff`, `bioio-czi`, `bioio-tifffile`,
> `bioio-lif`) plus `bioio-bioformats`, the Bio-Formats catch-all reader (JVM).
> `[imaris]` adds native `.ims` support (HDF5, no JVM). Physical pixel
> calibration is read from the input and written into the OME-ZARR.

---

## Quick start — 5 lines

```python
from patchworks import tile_process


def my_fn(tile):
    from skimage.filters import threshold_otsu
    from skimage.measure import label

    return label(tile > threshold_otsu(tile)).astype("int32")


result = tile_process("image.zarr", my_fn)
```

Done. `result` is a **lazy dask array** of integer labels (call `.compute()`
for a NumPy array), same spatial shape as the input, with globally unique IDs
across all tiles. By default the labels are also written **into the input
store** at `image.zarr/labels/labels/` as a multi-scale pyramid, so the image
and its segmentation live in one OME-ZARR. Pass `write_to="labels.zarr"` to
write a separate store instead.

---

## With Cellpose

```python
from patchworks import tile_process
from patchworks.plugins.cellpose import cellpose_fn

fn = cellpose_fn("cyto3", gpu=True, diameter=30)

tile_process(
    "image.zarr",
    fn,
    tile_shape=(1, 2048, 2048),  # one z-slice per tile
    overlap=20,  # gives boundary cells enough context
    write_to="labels.zarr",  # stream directly to disk — no RAM accumulation
    progress=True,
)
```

---

## With StarDist

```python
from stardist.models import StarDist2D
from patchworks import tile_process

model = StarDist2D.from_pretrained("2D_versatile_fluo")


def stardist_fn(tile):
    img = tile[0] if tile.ndim == 3 and tile.shape[0] == 1 else tile
    norm = img.astype("float32") / (img.max() or 1)
    labels, _ = model.predict_instances(norm)
    return labels.astype("int32")[None] if tile.ndim == 3 else labels.astype("int32")


tile_process(
    "image.zarr",
    stardist_fn,
    tile_shape=(1, 1024, 1024),
    overlap=32,
    write_to="labels.zarr",
    progress=True,
)
```

---

## With any function

```python
import numpy as np
from scipy.ndimage import gaussian_filter
from skimage.measure import label
from patchworks import tile_process


def my_custom_fn(tile: np.ndarray) -> np.ndarray:
    smoothed = gaussian_filter(tile.astype("float32"), sigma=1.5)
    binary = smoothed > smoothed.mean()
    return label(binary).astype("int32")


tile_process("image.zarr", my_custom_fn, tile_shape=(1, 512, 512))
```

---

## Convert to OME-ZARR & view in napari

Optional plugins close the loop: convert any image (Imaris `.ims`, CZI, LIF,
ND2, OME-TIFF, … via bioio) to a pyramidal, **calibrated** OME-ZARR, then view
the image and its labels in napari.

```python
from patchworks.plugins.ome_zarr import to_ome_zarr
from patchworks.plugins.napari import view_in_napari

to_ome_zarr("scan.ims", "scan.zarr")          # lazy, OOM-safe, keeps µm calibration
view_in_napari("scan.zarr", labels="scan.zarr/labels/labels")
```

Pyramids downsample **X/Y only** (Z kept full-res) and are built level-by-level
from disk, so terabyte volumes convert in bounded RAM. See the
[OME-ZARR & napari guide](https://imcf.one/patchworks/guide/ome_zarr_napari/).

---

## Common patterns

### Auto-size tiles from available memory

```python
from patchworks import tile_process

tile_process("image.zarr", fn, tile_shape="auto", use_gpu=True)
```

### Skip empty tiles (sparse volumes)

```python
from patchworks import estimate_empty_tiles, tile_process

info = estimate_empty_tiles("image.zarr", tile_shape=(120, 697, 697))
print(f"{info['empty_fraction']:.0%} tiles are background — will be skipped")

tile_process(
    "image.zarr",
    fn,
    tile_shape=(120, 697, 697),
    skip_empty=True,
    empty_threshold=info["threshold"],
    write_to="labels.zarr",
)
```

### Distributed cluster for GPU

```python
from patchworks import make_local_cluster, tile_process

client, cluster = make_local_cluster(use_gpu=True)
try:
    tile_process("image.zarr", fn, write_to="labels.zarr", progress=True)
finally:
    client.close()
    cluster.close()
```

### Contiguous label numbering

```python
# Labels are globally unique by default, but may be gappy (block-encoded IDs).
# sequential_labels=True does a linear relabel O(voxels) — not O(n_tiles²).
tile_process("image.zarr", fn, write_to="labels.zarr", sequential_labels=True)
```

### Use only the merge step (bring your own tiling)

If you already have per-tile labels from your own pipeline, just call the
merge step directly:

```python
import dask.array as da
import numpy as np
from patchworks import merge_tile_labels

# Your own tiling + segmentation
image = da.from_zarr("image.zarr").rechunk((1, 1024, 1024))
labeled = image.map_blocks(
    my_segment_fn, dtype="int32", meta=np.empty((0,) * image.ndim, dtype="int32")
)

merged = merge_tile_labels(labeled, write_to="labels.zarr", progress=True)
```

Or merge from a zarr store your pipeline already wrote:

```python
from patchworks import merge_tile_labels

merged = merge_tile_labels(
    "my_staged_labels.zarr",
    input_component="raw_labels",
    write_to="merged.zarr",
    sequential_labels=True,
)
```

---

## How tiling and merging work

See the [Merging labels guide](https://imcf.one/patchworks/guide/merging/) for
a full explanation. Short version:

1. Image is split into tiles (with optional overlap for boundary context).
2. Your function is called independently on each tile. Dask handles parallelism
   and streaming — tiles are never all in memory at once.
3. Each tile's labels are written to a temp zarr exactly once (the staging
   step — this prevents your function being called 3-4× per tile during merge).
4. Thin slabs at each tile boundary are scanned for touching label pairs.
5. scipy connected components on the pairs → relabeling lookup table.
6. LUT applied to every tile in parallel → globally consistent labels.

The merge is **zarr-native** (no dask task graph), so it scales to thousands of
tiles where the dask-image approach stalls.

---

## Known pitfalls (and how patchworks avoids them)

| Pitfall                        | Symptom                                   | How patchworks handles it                                     |
| ------------------------------ | ----------------------------------------- | ------------------------------------------------------------- |
| In-process Dask client         | `FutureCancelledError: lost dependencies` | Detected at startup, raises immediately with fix instructions |
| 3-4× fn recompute during merge | Cellpose runs 3× per tile                 | Staging writes labels once, merge reads from disk             |
| O(n²) sequential relabelling   | Graph construction hangs at 1000+ tiles   | Linear post-pass O(voxels) via `np.unique` + LUT              |
| Wrong overlap boundary         | Output shape mismatch                     | Always uses `boundary="none"`                                 |
| Persisting large arrays        | Worker OOM                                | Never persists; keeps dask graph lazy and streams             |

---

## Documentation

Full docs, guides and tutorials: **<https://imcf.one/patchworks/>**

- [Getting Started](https://imcf.one/patchworks/getting_started/)
- [User Guide](https://imcf.one/patchworks/guide/tiling/) — tiling, merging,
  empty-tile skipping, GPU/distributed, OME-ZARR & napari, pitfalls
- [Examples](https://imcf.one/patchworks/examples/cellpose_2d/) — Cellpose,
  StarDist, custom functions, standalone merge
- [API Reference](https://imcf.one/patchworks/api/tile_process/) ·
  [pdoc API](https://imcf.one/apidocs/patchworks/)

---

## Requirements

- Python ≥ 3.9
- dask[array], numpy, zarr, scipy

Optional:

- `psutil` — accurate RAM sizing for `tile_shape="auto"`
- `nvidia-ml-py` — accurate GPU VRAM sizing
- `tqdm` — progress bars
- `cellpose` — Cellpose plugin (`patchworks[cellpose]`)
- `bioio` + readers — convert CZI/LIF/ND2/OME-TIFF/… to OME-ZARR
  (`patchworks[bioio]`)
- `imaris-ims-file-reader` — convert Imaris `.ims` (`patchworks[imaris]`)
- `napari` — interactive viewer plugin (`patchworks[napari]`)

---

## License

GNU General Public License v3.0 (GPL-3.0). See [LICENSE](LICENSE).
