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

## Two ways to decide

### `estimate_empty_tiles` — a fast preview

For each tile in the grid, only a small centred **sample window** is read
(default: 24×256×256 voxels). If the maximum value in that window exceeds
the threshold, the tile is marked as occupied.

This is **bounded I/O**: the total data read is `n_tiles × sample_window`,
not the full image. For a 2200-tile image with the default window, this reads
≈ 30 MB instead of 250 GB — and it runs in seconds.

!!! warning "Approximate — a preview, not a skip list"
    Only the tile centre is inspected. On a `(16, 1024, 1024)` tile the
    default window covers **6.25% of the tile's area**, so an object in the
    outer ring is invisible. `tile_process` re-tests the **full tile** max
    inline, so nothing is dropped in a real run — but anything that uses this
    result *as* the skip list would drop those tiles for good.

### `build_occupancy_map` + `tile_occupancy` — exact

Reduces every `block`-sized brick of the image to its **maximum** and stores
that beside `image.zarr` (about 1/16384 of the image at the default 128 px
block). `tile_occupancy` then reduces the map over each tile's full footprint.

```python
from patchworks import (
    auto_empty_threshold, block_for_tile, build_occupancy_map, tile_occupancy
)

# Size the block from the tile, or every tile over-covers the same block and
# tests occupied — correct, but useless as a skip list.
build_occupancy_map("image.zarr", block=block_for_tile(TILE))
info = tile_occupancy(
    "image.zarr", TILE, channel=0, threshold=auto_empty_threshold(img, 0, 0)
)
```

The map is stored **beside** the image (`image.occupancy.zarr`), not inside
it: it is not an NGFF array, and a zarr hierarchy containing one cannot be
walked. It is rebuilt automatically if an existing map was built at a
different block.

This is **exact, not approximate**: `block_max > threshold` is true exactly
when some voxel in that block exceeds the threshold, so comparing pooled
maxima against a threshold derived from raw voxels answers the same question
as scanning every voxel. Max-pooling cannot lose a bright voxel.

The map is built once and shared by every segmentation reading that image, so
a three-config run pays for one pooling pass instead of three sampling passes.
This is what the Snakemake workflow uses, and it builds the map on first use
for stores converted before the map existed.

Skipping a tile pays off twice over: it is never segmented, **and** the merge
skips its chunk too — neither reading it nor writing zeros over it, so the
background never reaches disk at all.

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
