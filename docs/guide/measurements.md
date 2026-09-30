# Measurements (fast, whole-volume regionprops)

`skimage.measure.regionprops` needs the full labelled + intensity array in
RAM — fine for one tile, not for a hundred-thousand-object OME-ZARR.

## Already measured: the object tables

The workflow measures every object once, right after the merge: size
(`area_voxels`, `area_um3`), centroid and bounding box, plus mean/std
intensity for the channels listed in `table_channels`. The table is stored
with the labels (`image.zarr/labels/<name>/table`):

```python
from patchworks import read_table

cells = read_table("results/image.zarr/labels/cyto_labels")  # pandas
```

For a store without one, `patchworks tables results/image.zarr --channels 0,1`
adds them. With review corrections applied, and as csv files that
napari-chunked-regionprops loads directly: `patchworks review
results/image.zarr --export tables/` (see [Reviewing](review.md)).

## Interactively, in napari

[napari-chunked-regionprops](https://github.com/imcf/napari-chunked-regionprops)
is built for this — its "Measure" dock widget computes area/centroid/intensity
stats directly off a Labels layer's dask/zarr-backed array, out-of-core, and
scales with chunk count rather than object count. It's the best fit for
measuring *every* object in a store this size, not just a cropped region —
see [View image + labels in napari](ome_zarr_napari.md#view-image-labels-in-napari).
Bundled in `patchworks[napari]`.

For interactively inspecting individual cells by clicking in the viewer (not
all objects at once), the
[napari-skimage-regionprops](https://github.com/haesleinhuepf/napari-skimage-regionprops)
plugin's table widget also works well — point it at a cropped region rather
than the full volume, since it loads its input fully into memory.

## Headless / scripted

Use [`dask-image`](https://image.dask.org)'s `ndmeasure`, which computes
directly on the dask/zarr-backed arrays, chunk-parallel, without
materializing the volume:

```bash
pip install dask-image
```

```python
import dask.array as da
from dask_image.ndmeasure import area, center_of_mass, mean, standard_deviation

labels = da.from_zarr("results/image.zarr", component="labels/cyto_labels/0")
image = da.from_zarr("results/image.zarr", component="0")[0]  # channel 0, level 0

ids = da.unique(labels[labels > 0]).compute()
areas = area(image, labels, ids).compute()               # voxel counts
means = mean(image, labels, ids).compute()                # mean intensity
stds = standard_deviation(image, labels, ids).compute()
centroids = center_of_mass(image, labels, ids).compute()  # voxel coords (z, y, x)
```

Multiply `areas` by the voxel's physical volume and `centroids` by the pixel
size (both read straight from the OME-ZARR's own `multiscales` metadata) to
get µm-scale measurements.
