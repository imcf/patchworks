# Skipping empty tiles

## Why it matters

Fluorescence microscopy images are often **sparse**: most of the image is
empty space (background), with signal concentrated in a small region. For a
250 GB light-sheet volume with 78% background tiles, segmenting all tiles
wastes 78% of runtime. With `skip_empty=True`, background tiles return
all-zero labels immediately instead of running your function.

For a Cellpose 3-D run with 3-minute tiles, skipping 78% of tiles reduces
wall time from ~110 hours to ~24 hours.

## Quick usage

```python
from patchworks import estimate_empty_tiles, tile_process
from patchworks.plugins.cellpose import cellpose_fn

fn = cellpose_fn("cyto3", gpu=True, diameter=30)
TILE = (120, 697, 697)

# Step 1: preview the empty fraction and pick a threshold
info = estimate_empty_tiles("image.zarr", tile_shape=TILE)
print(f"{info['empty_fraction']:.0%} of tiles are background")
print(f"Threshold: {info['threshold']:.1f}")

# Step 2: run with skip_empty
tile_process(
    "image.zarr",
    fn,
    tile_shape=TILE,
    skip_empty=True,
    empty_threshold=info["threshold"],  # or let patchworks auto-derive it
    write_to="labels.zarr",
    progress=True,
)
```

## How `estimate_empty_tiles` works

For each tile in the grid, only a small centred **sample window** is read
(default: 24×256×256 voxels). If the maximum value in that window exceeds
the threshold, the tile is marked as occupied.

This is **bounded I/O**: the total data read is `n_tiles × sample_window`,
not the full image. For a 2200-tile image with the default window, this reads
≈ 30 MB instead of 250 GB — and it runs in seconds.

!!! warning "Approximate"
    `estimate_empty_tiles` inspects only the tile centre. Signal confined to
    a tile's edge can be missed. The actual `tile_process` run always inspects
    the **full tile** max inline — so no objects are ever dropped in the real
    run, only in the preview.

## Threshold selection

```python
info = estimate_empty_tiles("image.zarr", tile_shape=(120, 697, 697))
```

When `threshold=None` (default), an Otsu threshold is derived from the
gathered samples. This works well when the image has a clear bimodal
distribution (background vs signal).

You can also set it explicitly:

```python
info = estimate_empty_tiles(
    "image.zarr", tile_shape=(120, 697, 697), threshold=200.0
)  # anything ≤ 200 → empty
```

Or let patchworks auto-derive it at runtime:

```python
tile_process(
    "image.zarr",
    fn,
    skip_empty=True,
    # empty_threshold=None → auto-derive from a bounded sample
    write_to="labels.zarr",
)
```

## Empty fraction report

After a `tile_process` run with `skip_empty=True`, the log reports exactly
how many tiles ran your function:

```text
INFO patchworks._core: skip_empty: 486/2200 tiles ran fn, 1714 skipped (max<=412.0)
```
