# OME-ZARR conversion & napari viewing

Two optional plugins close the loop around `tile_process`: convert any input
to a fast, pyramidal OME-ZARR, then inspect the image and its labels in napari.

## Why convert to a pyramidal OME-ZARR?

A single full-resolution array is slow to browse: every pan or zoom touches the
whole plane. A **pyramid** stores progressively downsampled copies, so a viewer
only reads the resolution it needs for the current zoom level. OME-ZARR is the
chunked, cloud-friendly NGFF standard that napari (and Fiji, validators, …)
read natively.

## Convert any image to OME-ZARR

`to_ome_zarr` accepts a dask/NumPy array, an existing `.zarr` store, or **any
file format** readable by [bioio](https://github.com/bioio-devs/bioio) (CZI,
LIF, ND2, OME-TIFF, …). File inputs are read **lazily** — pixels stream from
disk and are written level by level through dask, so terabyte images convert in
bounded RAM.

```python
from patchworks.plugins.ome_zarr import to_ome_zarr

# From a proprietary microscope file (lazy, via bioio):
to_ome_zarr("scan.czi", "scan.zarr", n_levels=5)

# From the labels written by tile_process:
import dask.array as da
to_ome_zarr(
    da.from_zarr("labels.zarr", component="labels"),
    "labels_pyramid.zarr",
    axes="zyx",
)
```

!!! note "Install the readers you need"
    `pip install "patchworks[bioio]"` pulls `bioio` plus the `bioio-bioformats`
    catch-all reader (needs a JVM). For speed, add native readers for your
    formats, e.g. `bioio-ome-tiff`, `bioio-czi`, `bioio-lif`, `bioio-nd2`.

Downsampling uses **strided, nearest-neighbour** subsampling. This is the
correct choice for label images: interpolating label values would invent
objects that never existed. Only the spatial axes (`z`/`y`/`x`) are
downsampled — channel and time axes pass through unchanged.

## View the result in napari

`view_in_napari` opens the image and overlays the labels as a proper *Labels*
layer in one call. OME-ZARR pyramids are handed to napari as a lazy multi-scale
list, so even huge stores open instantly and only on-screen data is fetched.

```python
from patchworks.plugins.napari import view_in_napari

# image as OME-ZARR, labels as the plain store from tile_process:
view_in_napari("scan.zarr", labels="labels.zarr")
```

The label store written by `tile_process` keeps its array under the
`output_component` name (default `"labels"`); `view_in_napari` reads that
component and casts it to `int32` for the Labels layer. Pass
`labels_component=...` if you changed it.

!!! note
    napari is a GUI-heavy extra and is **not** included in `patchworks[all]`.
    Install it explicitly: `pip install "patchworks[napari]"`.

## End-to-end

```python
from patchworks import tile_process
from patchworks.plugins.ome_zarr import to_ome_zarr
from patchworks.plugins.napari import view_in_napari

# 1. segment a large image, streaming labels to disk
tile_process("scan.zarr", fn, write_to="labels.zarr", progress=True)

# 2. (optional) make a pyramid of the raw image for snappy browsing
to_ome_zarr("scan.zarr", "scan_pyramid.zarr")

# 3. inspect image + labels together
view_in_napari("scan_pyramid.zarr", labels="labels.zarr")
```
