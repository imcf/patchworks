# OME-ZARR conversion & napari viewing

Two optional plugins close the loop around `tile_process`: convert any input
to a fast, pyramidal OME-ZARR, then inspect the image and its labels in napari.

## Everything in one OME-ZARR (the default)

When you call `tile_process` on a `.zarr` store **without** `write_to`, the
labels are written **back into that same store** under the NGFF
`labels/<name>/` group, as their own multi-scale pyramid:

```python
from patchworks import tile_process

# labels land in scan.zarr/labels/labels/ with a pyramid — nothing else needed
tile_process("scan.zarr", fn)
```

After this, `scan.zarr` holds both the image and its segmentation, each
pyramidal, in a single NGFF store that napari, Fiji and validators read
natively. Pass `write_to="other.zarr"` to instead write a separate
single-resolution label store, or `output_component="cells"` to name the label
image.

The label pyramid is built lazily (`da.to_zarr`, streamed chunk by chunk), so
it stays OOM-safe even for terabyte volumes. Control it with `pyramid_levels`
and `pyramid_downscale`.

## Why a pyramid?

A single full-resolution array is slow to browse: every pan or zoom touches the
whole plane. A **pyramid** stores progressively downsampled copies, so a viewer
only reads the resolution it needs. Pyramids here downsample **X and Y only** —
`Z` (and channel/time) stay at full resolution, matching anisotropic microscopy
stacks. Downsampling is **strided, nearest-neighbour**: the correct choice for
label images, since interpolating label values would invent objects that never
existed.

## Convert any image to OME-ZARR

`to_ome_zarr` accepts a dask/NumPy array, an existing `.zarr` store, an
**Imaris `.ims`** file, or **any format** readable by
[bioio](https://github.com/bioio-devs/bioio) (CZI, LIF, ND2, OME-TIFF, …). File
inputs are read **lazily**.

```python
from patchworks.plugins.ome_zarr import to_ome_zarr

to_ome_zarr("scan.czi", "scan.zarr", n_levels=5)   # via bioio
to_ome_zarr("scan.ims", "scan.zarr")               # Imaris, native HDF5
```

!!! note "Imaris pyramids: rebuild (default) or reuse"
    `.ims` files carry their own resolution pyramid. By default `to_ome_zarr`
    reads only the **full-resolution** level and **builds a fresh NGFF pyramid**
    (XY-only, nearest-neighbour, calibrated) for consistency. Pass
    `reuse_pyramid=True` to instead **copy the Imaris levels** as-is — faster,
    no recompute, keeping each level's native scale:

    ```python
    to_ome_zarr("scan.ims", "scan.zarr", reuse_pyramid=True)
    ```

### Pixel calibration

The physical voxel size is read from the input — bioio's `physical_pixel_sizes`,
the Imaris resolution metadata, or an existing OME-ZARR's scale — and written
into the NGFF `coordinateTransformations` (in micrometers), so calibration is
preserved regardless of input. Override or supply it for bare arrays with
`pixel_size={"z": 2.0, "y": 0.32, "x": 0.32}`.

### Won't OOM

Each pyramid level is built by reading the **previous level back from disk**
and streaming the downsampled result out through dask with bounded chunks. The
graph never chains level-on-level and no whole plane/volume is held in RAM, so
terabyte images convert in bounded memory.

!!! note "Install the readers you need"
    `pip install "patchworks[bioio]"` pulls `bioio` plus the `bioio-bioformats`
    catch-all reader (needs a JVM). For speed, add native readers for your
    formats, e.g. `bioio-ome-tiff`, `bioio-czi`, `bioio-lif`, `bioio-nd2`.

## Add a pyramid to an existing store

Already have a flat (single-resolution) zarr? `add_pyramid` writes the missing
levels in place, lazily:

```python
from patchworks.plugins.ome_zarr import add_pyramid

add_pyramid("flat.zarr", base="0", n_levels=5)
```

And `write_labels` stores any label array inside an existing OME-ZARR under the
`labels/` group (the same thing `tile_process` does by default):

```python
from patchworks.plugins.ome_zarr import write_labels

write_labels("scan.zarr", my_labels, name="nuclei")
```

## View image + labels in napari

`view_in_napari` opens the image and overlays the labels as a proper *Labels*
layer in one call. OME-ZARR pyramids are handed to napari as a lazy multi-scale
list, so even huge stores open instantly and only on-screen data is fetched.

Because `tile_process` writes labels **into** the store by default, you usually
need no `labels=` argument at all — `view_in_napari` auto-loads every label
image found under `scan.zarr/labels/`:

```python
from patchworks.plugins.napari import view_in_napari

# auto-loads scan.zarr/labels/* as Labels layers:
view_in_napari("scan.zarr")

# or point at a separate plain label store written with write_to=:
view_in_napari("scan.zarr", labels="labels.zarr")
```

!!! note
    napari is a GUI-heavy extra and is **not** included in `patchworks[all]`.
    Install it explicitly: `pip install "patchworks[napari]"`.

## End-to-end

```python
from patchworks import tile_process
from patchworks.plugins.napari import view_in_napari

# 1. segment — labels are written into scan.zarr with a pyramid, by default
tile_process("scan.zarr", fn, progress=True)

# 2. inspect image + labels together, straight from the one store
view_in_napari("scan.zarr")  # labels auto-loaded from scan.zarr/labels/
```

Plugging in a different segmentation method is just swapping `fn` — any
callable taking a tile and returning an integer label array works (see the
Cellpose and StarDist examples).
