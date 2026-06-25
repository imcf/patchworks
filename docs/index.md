# patchworks

<p align="center">
  <img src="assets/logo.png" alt="patchworks logo" width="200">
</p>

**Tiled processing of arbitrarily large images — any image, any function.**

```text
┌──────┬──────┬──────┐                    ┌──────┬──────┬──────┐
│      │      │      │   fn(tile) → IDs   │  1   │  2   │  3   │
│      │      │      │  ───────────────►  │      │      │      │
│      │ 20GB │      │                    │      │      │      │
├──────┼──────┼──────┤                    ├──────┼──────┼──────┤
│      │ IMG  │      │                    │  4   │  5   │  6   │
│      │      │      │                    │      │      │      │
├──────┼──────┼──────┤                    ├──────┼──────┼──────┤
│      │      │      │                    │  7   │  8   │  9   │
└──────┴──────┴──────┘                    └──────┴──────┴──────┘
         tiles                               globally consistent labels
```

patchworks splits a large image into tiles, runs **any callable** on each tile
in parallel, and merges the results into a globally consistent label array.
It handles terabyte-scale images without loading them into RAM.

## Why patchworks?

Modern fluorescence microscopy produces images in the hundreds of GB to several TB
range. Instance segmentation tools (Cellpose, StarDist, threshold methods, your
own model) all assume the image fits in memory. They don't scale.

The naive approach — split the image into tiles, segment each tile, stitch the
labels — creates **split objects**: any cell spanning a tile boundary gets two
different label IDs. patchworks solves this with a zarr-native boundary merge:

1. Tiles are segmented independently and streamed to disk
2. Thin slabs at each tile boundary are scanned for touching label pairs
3. scipy connected components on the pairs → globally consistent relabeling
4. No tile is ever loaded fully into RAM more than once

This approach scales to thousands of tiles and terabyte images. It is the same
strategy used by
[skeleplex](https://github.com/kevinyamauchi/skeleplex) and
[cellpose distributed](https://github.com/MouseLand/cellpose/tree/main/cellpose).

## Quick example

```python
from patchworks import tile_process


def my_fn(tile):
    from skimage.filters import threshold_otsu
    from skimage.measure import label

    return label(tile > threshold_otsu(tile)).astype("int32")


result = tile_process("image.zarr", my_fn)
```

Any function. Any image.

## Method agnostic

patchworks doesn't care what's inside `fn`:

```python
# Cellpose
from patchworks.plugins.cellpose import cellpose_fn

fn = cellpose_fn("cyto3", gpu=True, diameter=30)

# StarDist
from stardist.models import StarDist2D

model = StarDist2D.from_pretrained("2D_versatile_fluo")
fn = lambda tile: model.predict_instances(tile)[0].astype("int32")

# Your own PyTorch model
fn = lambda tile: my_model(torch.from_numpy(tile)).argmax(0).numpy()

# All work identically with tile_process
tile_process(
    "image.zarr",
    fn,
    tile_shape=(1, 1024, 1024),
    overlap=20,
    write_to="labels.zarr",
    progress=True,
)
```

## Installation

```bash
pip install patchworks
```

See [Getting Started](getting_started.md) for installation options and
your first run.
