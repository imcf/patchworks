# Merging

Each tile is segmented on its own, so its labels are only locally unique and
an object crossing a seam gets one id on each side. The merge gives every
object one global id. `tile_process` does it for you; this page covers the
options.

```text
tile A │ tile B           merged
 2 1 1 │ 1 1 2             2 1 1 │ 1 1 3
 2 1 1 │ 1 1 2     →       2 1 1 │ 1 1 3
```

How it works, in short: ids are offset per tile, only the voxels on either
side of each seam are read, touching labels are joined (connected components)
and one lookup table relabels every chunk in parallel. It is linear in the
volume, never needs it in memory, and skips chunks holding no labels.

## Keeping touching cells apart: IoU stitching

By default any two labels touching across a seam are joined. That is right
for one object cut by the seam, wrong for two cells pressed together exactly
there. `stitch="iou"` joins them only when both tiles labelled the same
object in their shared halo: each is the other's best match and they overlap
on at least `iou_threshold` (default 0.5) of the smaller one, since a tile
sees an object crossing the halo cut off where its read region ends.

```python
tile_process("image.zarr", fn, tile_shape=(16, 1024, 1024), overlap=30, stitch="iou")
```

It needs `overlap > 0`. On the cluster: `stitch: "iou"` in the config.

## Checking the seams

[`seam_report`](../api/seams.md) compares how often objects end at a seam
with how often they end on planes inside the tiles, where nothing was
stitched:

```python
from patchworks import seam_report

report = seam_report("scan.zarr/labels/cells", tile_shape=(16, 1024, 1024))
report["axes"][2]  # {'seam_rate': 0.04, 'interior_rate': 0.05, 'ratio': 0.8, ...}
```

A ratio near 1 means the tiling does not show; well above it, raise `overlap`
or try `stitch="iou"`. The cluster workflow writes it to
`<work_dir>/<label_name>/seams.json` after every merge.

## Choosing the overlap from the data

[`suggest_overlap`](../api/seams.md) segments a crop once untiled, then tiled
at increasing overlaps, and returns the smallest overlap that gives the same
result:

```python
from patchworks import load_ome_zarr, suggest_overlap

image = load_ome_zarr("scan.zarr", channel=0)
suggest_overlap(image, fn, tile_shape=(16, 512, 512))
# {'overlap': 16, 'scores': {0: 0.91, 4: 0.95, 8: 0.97, 16: 0.995}, ...}
```

Pass `region=` to pick a crop with typical objects. Command line:
`patchworks segment ... --overlap auto`.

## Size filter and numbering

Objects can only be judged by size after the merge, once they are whole:

```python
from patchworks import filter_labels_by_size, min_voxels_for_volume
from patchworks.plugins.ome_zarr import read_pixel_size

min_voxels = min_voxels_for_volume(5.0, read_pixel_size("image.zarr"))  # µm³ → voxels
filter_labels_by_size("labels.zarr", "labels", min_voxels=min_voxels, max_voxels=50000)
```

It streams the array and renumbers the survivors `1..N`. On the cluster,
`min_volume:` / `max_volume:` (µm³) in the config do the same.

Merged ids are unique but may have gaps; `sequential_labels=True` numbers
them `1..N` at no extra cost.

## Resuming

`tile_process(..., resume=True)` keeps finished tiles if the run dies;
rerunning the same call continues from there. The cluster workflow does this
per job.

## Merging your own tiles

`merge_tile_labels` works on any tiled label array: a dask array or a zarr
your own pipeline wrote.

```python
from patchworks import merge_tile_labels

merge_tile_labels("my_tiles.zarr", input_component="raw_labels",
                  write_to="merged.zarr", sequential_labels=True)
```

See also the [standalone merge example](../examples/standalone_merge.md).
