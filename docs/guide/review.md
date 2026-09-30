# Reviewing and correcting results

A segmentation of a whole tissue has tens of thousands of objects, and some
of them are wrong. Scrolling through the volume hoping to spot them doesn't
work. `patchworks review` shows you the objects **most likely to be wrong**,
one at a time, and lets you fix each one with a single key. Your
corrections go straight into the result tables and workbooks, and no voxel
is rewritten unless you ask.

```bash
pip install "patchworks[napari]"      # on the machine you review on
patchworks review results/image.zarr
```

![The review panel](../assets/review_panel.png)

*A cilium only 25% inside its cell. The view jumps to it, shows it with its
cell outlined, and hides everything else. Press G if it is right, H to give
it the right cell, W if it isn't a cilium at all.*

## What gets flagged

Each object in the queue shows why it is there:

| Flag | Means | Typical cause |
| --- | --- | --- |
| *not inside any cell* | the object overlaps no parent object | debris, a missed cell, a cilium on the lumen side |
| *only 30% inside cell #88* | less than `min_overlap` of it is inside its parent | wrong cell picked at a boundary, or a merge of two objects |
| *0 nuclei (expected 1)* | a parent holds an unexpected number of children | a cell split in two, two cells merged, a missed nucleus |
| *meets #412 exactly at a tile seam (z)* | two objects touch face to face on a tile boundary | one object the tiling cut in two, or two neighbours; press J to join them |
| *unusually large (6.2x the median)* | the size is far from the rest (robust z-score > 3.5) | two objects merged, or a fragment |

The most suspicious objects come first. "Children" and "parents" come from
the relations of a multi run (e.g. `cilia_labels → cyto_labels`); the size
and seam flags apply to every label image.

**Expected counts** are yours to state, since only you know that a cell has
one nucleus and zero to two cilia. Put them in `multi.yaml`, and the
workflow checks them before it starts and stores them with the results:

```yaml
review:
  expect:
    cyto_labels:
      nuclei_labels: 1         # exactly one
      cilia_labels: [0, 2]     # zero to two
  min_overlap: 0.5             # or per pair: {cilia_labels: {cyto_labels: 0.3}}
```

Or pass them when you open the review:
`patchworks review image.zarr --expect cyto_labels:nuclei_labels=1 cyto_labels:cilia_labels=0-2`.

## Deciding

| Key | Decision | Effect on the results |
| --- | --- | --- |
| **G** | correct as it is | kept; marked `ok` |
| **W** | wrong: not a real object | dropped from every table and count |
| **H**, then click | belongs to another parent: click the right one (the background means "none") | its parent id changes; both parents' counts follow |
| **J** | join with the suggested object (a seam split) | the two become one object: sizes added, centroid and intensities weighted |
| **Shift+J**, then click | join with the object you click | same |
| **N** / **Shift+N** | skip / go back | nothing |
| **U** | undo the decision about this object | back to unreviewed |

Every decision is saved immediately, in the store, next to the object
table. Close napari whenever you like: the next `patchworks review`
continues where you stopped, and the queue leaves out what has been
decided. Corrections chain as you would expect. For example, joining two
halves of a cell moves the cilia of both halves to the joined cell.

Two options in the panel help with the view:

- **Show only this object and its relatives** (on by default) hides
  everything but the object, its parent (outlined) and its children.
  Untick it to see the neighbourhood.
- **Colour these objects by their parent** adds a layer where every child
  has its parent's colour: all cilia of one cell share a colour, so an
  assignment error stands out as a different colour.

Orange rings mark the flagged objects still open. They stay visible in 3D,
where napari's coarse 3D level hides small objects.

## How good is the segmentation?

Pick the queue **Random sample (error rate)** and review objects in the
order it gives. The panel reports the fraction found wrong, with a 95%
confidence interval, e.g. *Error rate 3.0% (95% CI 1.0–8.5%) from 100
random objects*. Stop when the interval is narrow enough for what you need.
The estimate is unbiased because the order is random and fixed. Objects you
already decided from the flagged queue count too: a decision about an
object is true however it came up.

## Using the corrections

The workbooks and tables are always read *with* the decisions applied:

- **The relation workbooks** of a multi run:
  `patchworks review image.zarr --workbooks results/` writes every one
  (`<child>_to_<parent>.xlsx`) from the corrected tables, with a `qc`
  column (`ok`, `fixed`, or blank if not reviewed). Re-running the same
  `run_multi` command does it too: a workbook older than your decisions is
  rewritten, straight from the tables, without re-reading any labels.
- **Export** the corrected tables, one file per label image, from the
  panel or with `patchworks review image.zarr --export results/tables
  --format xlsx` (or `csv`, `parquet`). A csv loads straight into
  [napari-chunked-regionprops](measurements.md) ("Reload previous
  results").
- **Write corrected labels** (panel button, or `--write-labels
  nuclei_labels`) writes `labels/nuclei_labels_reviewed`, the label image
  with the decisions applied to the voxels. You only need this for figures,
  or for tools that read label images only.

`patchworks review image.zarr --summary` prints the counts and the error
estimate without opening napari.

## Where the tables come from

Every label image the workflow writes gets an **object table**: one row per
object with its size (`area_voxels`, `area_um3`), centroid, bounding box
and, for a multi run, the parent it sits in (`cyto_labels_id`,
`cyto_labels_overlap`). The table lives inside the label group:

```text
image.zarr/labels/cilia_labels/
  0/ 1/ 2/ …     the label pyramid
  table/         the object table (one zarr array per column)
```

So it travels with the labels, including in a zip bundle, and it is
replaced whenever the labels are. A table computed from older labels is
recognised and ignored, never shown against the wrong segmentation.
Measuring costs one read of the labels, after the merge. Add intensity
columns with `table_channels: [0, 2]`, or turn tables off with
`object_table: false` (see the [workflow config](snakemake.md)).

For a store that has none, for example a run made before tables existed,
or labels from elsewhere:

```bash
patchworks tables image.zarr --relate nuclei_labels:cyto_labels cilia_labels:cyto_labels
```

In Python:

```python
from patchworks import Review, read_table

cells = read_table("image.zarr/labels/cyto_labels")      # pandas, as computed
rv = Review("image.zarr", expect={"cyto_labels": {"nuclei_labels": 1}})
rv.flags("cyto_labels")[:5]                               # the worst five
rv.decide("nuclei_labels", 17, "wrong")
rv.effective("cyto_labels")                               # with corrections
```

## Where to review

On any machine with a screen and napari, reading the store the cluster
wrote:

- **Copy the store** to your computer (tables and decisions live in it), or
  open it on a mounted cluster filesystem. napari reads only what is on
  screen, so a network mount works.
- The store must be **writable**, since that is where the decisions are
  saved. A `.zip` bundle is read-only, so unzip it first.
- The decisions are small. If you reviewed a copy, `--workbooks` and
  `--export` work on that copy directly; there is no need to copy anything
  back.
