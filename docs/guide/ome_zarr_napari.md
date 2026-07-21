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
**Imaris `.ims`** file, **any format** readable by
[bioio](https://github.com/bioio-devs/bioio) (CZI, LIF, ND2, OME-TIFF, …), or
a **folder of single-plane TIFFs** (see below). File inputs are read
**lazily**.

```python
from patchworks.plugins.ome_zarr import to_ome_zarr

to_ome_zarr("scan.czi", "scan.zarr", n_levels=5)   # via bioio
to_ome_zarr("scan.ims", "scan.zarr")               # Imaris, native HDF5
```

### A folder of single-plane TIFFs

Some acquisitions/stitching tools save **one TIFF per Z/C plane** instead of a
single multi-page file, e.g.:

```text
sample_T0_Z000_C0_V0.tif
sample_T0_Z000_C1_V0.tif
sample_T0_Z001_C0_V0.tif
sample_T0_Z001_C1_V0.tif
...
```

Pass `sequence_pattern=` and let `source` be a glob over the folder instead of
one file path. The pattern is a regex whose **named groups** map to axis
labels — axis order follows the order the groups appear in the pattern:

```python
to_ome_zarr(
    "sample/*.tif",
    "sample.zarr",
    sequence_pattern=r"_T(?P<T>\d+)_Z(?P<Z>\d+)_C(?P<C>\d+)_V\d+",
    shard=True,  # recommended for large sequences, see Sharding below
)
```

Each file becomes exactly **one dask chunk**, decoded lazily on access — no
data is duplicated or eagerly loaded, so this scales to huge (multi-TB)
sequences (built on `tifffile.TiffSequence`, the same mechanism Cellpose's
own distributed pipeline uses). Pixel calibration is read automatically from
the first file's own metadata — see [Pixel calibration](#pixel-calibration)
below.

Available on the cluster too: the Snakemake `convert` rule reads a
`sequence_pattern:` key from the config (`input:` then being the glob), so a
folder-of-TIFFs conversion runs through the same SLURM profile as any other
input — see [Cluster usage](snakemake.md).

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
the Imaris resolution metadata, an existing OME-ZARR's scale, or (for a TIFF
sequence) the first file's own ImageJ metadata (`spacing`/`unit`) or
`XResolution`/`YResolution` tags — and written into the NGFF
`coordinateTransformations` (in micrometers), so calibration is preserved
regardless of input. Override or supply it for bare arrays with
`pixel_size={"z": 2.0, "y": 0.32, "x": 0.32}`.

### Won't OOM

Each pyramid level is built by reading the **previous level back from disk**
and streaming the downsampled result out through dask with bounded chunks. The
graph never chains level-on-level and no whole plane/volume is held in RAM, so
terabyte images convert in bounded memory.

### Sharding (fewer files)

A big array becomes tens of thousands of tiny chunk files, which strain
filesystems and object stores. Sharding packs many chunks into one **shard**
file (zarr v3), cutting the file count ~100×:

```python
to_ome_zarr("scan.ims", "scan.zarr", shard=True)        # auto ~512 MB shards
to_ome_zarr("scan.ims", "scan.zarr", shard=(1, 16, 2048, 2048))  # explicit
```

Default is `shard=False` for maximum reader compatibility — sharding is
zarr-v3-only, so older tools may not read it (your zarr/napari stack does).
A sharded write holds ~one shard per worker in RAM, so very large shards cost
memory.

### Progress

All write steps show a dask progress bar **by default** (`progress=True`), so
you can see how long a conversion will take. Pass `progress=False` to silence
it.

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

# auto-loads scan.zarr/labels/* as Labels layers, all image channels shown:
view_in_napari("scan.zarr")

# or point at a separate plain label store written with write_to=:
view_in_napari("scan.zarr", labels="labels.zarr")
```

By default every channel of the image is shown (`channel=None`), independent
of which one you segmented on — Cellpose might run on channel 0 while you
still want to see the whole multi-channel acquisition. Pass an int to view
just one channel instead: `view_in_napari("scan.zarr", channel=0)`.

!!! note
    napari ships in `patchworks[all]`, or install just it with
    `pip install "patchworks[napari]"`.

!!! tip "Measuring all the objects"
    Once labels are loaded, use
    [napari-chunked-regionprops](https://github.com/imcf/napari-chunked-regionprops)'s
    "Measure" dock widget for area/centroid/intensity stats — it works
    out-of-core straight off the Labels layer's backing dask/zarr array, so
    it scales to the same huge label images `tile_process` writes, unlike
    plain `skimage.measure.regionprops`. Bundled in `patchworks[napari]`. See
    [Measurements](measurements.md)
    for the non-interactive/headless equivalent.

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
