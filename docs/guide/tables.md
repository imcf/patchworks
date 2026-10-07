# Tables and relations

Every label image carries a table, one row per object, stored with the
labels (`image.zarr/labels/<name>/table`): size (`area_voxels`, `area_um3`),
centroid, bounding box, shape (length, elongation, main axis), and mean and
standard deviation of intensity for the channels you ask for. The cluster
workflow writes it during the merge, from what the segment jobs measured per
tile, without reading the labels again.

```python
from patchworks import read_table

cells = read_table("results/image.zarr/labels/cyto_labels")  # pandas DataFrame
```

For a store without tables, or to add intensities:

```bash
patchworks tables results/image.zarr --channels 0,1
```

## Relating two label images

Which cell is each nucleus in? `label_relations` maps every object of one
label image to the object of another it overlaps most, reading both chunk by
chunk:

```python
from patchworks import label_relations

table = label_relations(
    "results/image.zarr/labels/nuclei_labels/0",
    "results/image.zarr/labels/cyto_labels/0",
    a_component="",
    b_component="",
)
table[2]  # {'match': 3, 'overlap_voxels': 4821, 'overlap_fraction': 0.94}
```

The two must be segmentations of the same image at the same chunking. To
relate several children to one parent, reading it once,
`patchworks.label_relations_many({"nuclei": ..., "cilia": ...}, cells)`.

To keep the relation in the tables instead, as columns of the child's table
(`cyto_labels_id`, the overlap), where [`patchworks review`](review.md)
corrections apply to it:

```bash
patchworks tables results/image.zarr --relate nuclei_labels:cyto_labels
patchworks tables results/image.zarr --relate cilia_labels:cyto_labels --max-distance 1.0
```

`--max-distance` (µm) gives an object touching no parent the nearest one
within that distance. The [cluster workflow](snakemake.md) does all of this
from the `relations:` of a multi config, and writes a workbook per pair.

## Exporting

```bash
patchworks review results/image.zarr --export tables/ --format parquet  # or csv, xlsx
```

writes every table with the review corrections applied.
[napari-chunked-regionprops](https://github.com/imcf/napari-chunked-regionprops)
(in `patchworks[napari]`) reads them, and measures any label layer in napari
out of core.
