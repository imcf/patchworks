# Relating labels across segmentations

Once you have two segmentations of the same image (e.g. nuclei inside cells),
`label_relations()` maps each label in one to the label it overlaps most in
the other — by streaming both arrays chunk by chunk, so it scales to
hundreds of thousands of objects without loading anything fully into RAM.

Both label arrays must share the exact same chunk layout — same
`tile_shape`/pyramid `level` when they were produced.

```python
import dask.array as da
from patchworks import label_relations

nuclei = da.from_zarr("results/image.zarr", component="labels/nuclei_labels/0")
cells = da.from_zarr("results/image.zarr", component="labels/cyto_labels/0")

table = label_relations(nuclei, cells)
table[2]
# {'match': 3, 'overlap_voxels': 4821, 'overlap_fraction': 0.94}
# -> nucleus 2 belongs to cell 3, 94% of its voxels fall inside it
```

`table` only contains matched `a` labels (nuclei with at least one
overlapping voxel in `cells`) — unmatched labels and full per-`b` coverage
need a bit more bookkeeping (the [cluster workflow's `run_multi.py`
script](snakemake.md#one-command-multiple-segmentations--relations) does
this for you and writes it as a two-sheet workbook).

Save it as a table yourself:

```python
import csv

with open("nuclei_to_cell.csv", "w", newline="") as f:
    w = csv.writer(f)
    w.writerow(["nucleus_id", "cell_id", "overlap_voxels", "overlap_fraction"])
    for nucleus_id, m in table.items():
        w.writerow([nucleus_id, m["match"], m["overlap_voxels"], m["overlap_fraction"]])
```

On the cluster, producing the two label stores in the first place is a
matter of running the workflow twice against the same `work_dir` — see
[Running two segmentations](snakemake.md#running-two-segmentations-eg-nuclei--cytoplasm).
