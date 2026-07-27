# I/O helpers

::: patchworks.load_ome_zarr

## Deciding which tiles hold signal

`estimate_empty_tiles` is a fast **preview** — it samples a centred window per
tile, so it can miss signal at a tile's edge. `build_occupancy_map` +
`tile_occupancy` are **exact**: a brick maximum exceeds the threshold exactly
when some voxel in that brick does. Use the latter pair when the result is
used as a skip list. See [Skipping empty tiles](../guide/skip_empty.md).

::: patchworks.estimate_empty_tiles

::: patchworks.build_occupancy_map

::: patchworks.tile_occupancy

::: patchworks.auto_empty_threshold
