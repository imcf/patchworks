# Cells from a membrane stain

Epithelia, organoids and tissues are often imaged with a **membrane** (or
cortex) marker plus a **nuclear** dye. Cellpose struggles there in a
recognisable way: Cellpose 4 (`cpsam`) reads every channel it is given
without a "cytoplasm" or "nucleus" role, so with a bright nuclear channel
next to a faint membrane it segments the **nuclei**; and on the membrane
alone, a wall that is faint in places merges two cells. Patchworks ships
three plugins for this case, plus an optional denoising step in front of any
method.

| | Needs | Cells come from | Best when |
| --- | --- | --- | --- |
| [Nuclei-seeded watershed](#nuclei-seeded-watershed) | scikit-image | the membrane, flooded from the nuclei | every cell has one nucleus; start here |
| [PlantSeg](#plantseg) | `plant-seg` (conda-forge) | a boundary U-Net, then GASP / multicut | the membrane is uneven or noisy |
| [PlantSeg + nuclei](#plantseg) | same | the U-Net's boundaries and the nuclei | both of the above |
| Cellpose, membrane only | cellpose | Cellpose | `nuclei_channel: null`, `do_3D: true` |
| [Denoising first](#denoising-first-careamics) | `careamics` | any of these, on denoised tiles | noisy acquisitions |

All of them are ordinary [custom functions](custom_segmentation.md): one
config block, the same tiling, merge, tables and review as any other run.

## Nuclei-seeded watershed

The nuclei are the easy part of the image: bright, compact, separated. They
say how many cells there are and where; flooding the membrane image from
them grows each nucleus out to its cell's walls. A cell can then not be
split (one seed each) or merged with its neighbour (two seeds never join),
even across a gap in the wall.

```yaml
channel: 0              # membrane
nuclei_channel: 1       # stacked onto each tile as [membrane, nuclei]
method: "custom"
label_name: "cyto_labels"
stitch: "iou"           # required: see below
custom:
  module: "patchworks.plugins.watershed"
  kwargs:
    nuclei_min_size: 200    # voxels: specks smaller than a nucleus
    foreground: "otsu"      # stop at the tissue edge
    # max_radius_um: 15     # or: no further than this from the nucleus
    # nuclei_threshold: 800 # intensity, if Otsu misses dim nuclei
```

`stitch: "iou"` is required with this plugin and with PlantSeg, and
`prepare` refuses the config without it. They give every voxel to some
cell, so neighbouring cells touch at every tile seam, and the default
`"touch"` stitching would join each such pair: on a test grid of 12 cells
cut by six tiles, 5 came out. `"iou"` joins two pieces only where both tiles
agree on their overlap, and gives the 12. (From the command line, pass
`--stitch iou` yourself: nothing checks it there.)

Two nuclei touching each other give one seed, so one cell for two: raise
`nuclei_threshold` if that happens -- or grow the cells from nuclei you have
already segmented, with `seed_labels` instead of `nuclei_channel` (see
[the example below](#how-seed_labels-works)). Each tile's halo (`overlap`) must hold a
whole cell, as for any method.

## PlantSeg

[PlantSeg](https://github.com/kreshuklab/plant-seg) predicts cell
boundaries with a 3-D U-Net trained on membrane stains, then partitions the
boundary map into cells (supervoxels by a distance-transform watershed,
merged by GASP, mutex watershed or multicut). A wall the U-Net sees at all
gets closed by the partitioning, where Cellpose would merge across it.

PlantSeg is on conda-forge only, so it has its own pixi environment:

```bash
pixi install -e plantseg
pixi run -e plantseg plantseg-fetch generic_confocal_3D_unet   # once, with internet
pixi run -e plantseg multi-slurm
```

```yaml
method: "custom"
stitch: "iou"                           # required, as for the watershed
custom:
  module: "patchworks.plugins.plantseg"
  kwargs:
    model: "generic_confocal_3D_unet"   # or generic_light_sheet_3D_unet, ...
    segmentation: "gasp"                # gasp | mutex_ws | multicut | dt_watershed
    beta: 0.6                           # lower merges more, higher splits more
    foreground: "otsu"                  # a boundary U-Net puts cells everywhere
```

With `nuclei_channel` set, two more `segmentation` modes use the nuclei:

- `"nuclei_watershed"` -- the U-Net's boundary map flooded from the nuclei:
  the watershed above, on a much cleaner boundary image.
- `"lifted_multicut"` -- PlantSeg's lifted multicut: supervoxels in one
  nucleus pulled together, in different nuclei pushed apart.

Each tile is resampled to the voxel size the model was trained at (from the
image's own calibration, `rescale: true`), which matters more than any other
setting for a pretrained U-Net; the prediction is resampled back.

## Example: Cellpose nuclei + PlantSeg cells in one run

The workflow ships this pairing ready to edit: Cellpose segments the nuclei,
then PlantSeg grows the cells from **exactly those nuclei** -- one cell per
nucleus Cellpose found -- and the two are related. Three files, next to
`config/multi.yaml`:

```yaml
# config/multi_plantseg.yaml
common: config/common.yaml           # input, work_dir, tile_shape, level: shared

segmentations:
- config/config_nuclei.yaml          # Cellpose "nuclei" model on channel 1
- config/config_cyto_plantseg.yaml   # PlantSeg on channel 0, seeded by nuclei_labels

relations:
- a: nuclei_labels
  b: cyto_labels
  output: nuclei_to_cyto.xlsx

review:                              # a sanity check: one nucleus per cell
  expect:
    cyto_labels:
      nuclei_labels: 1
```

```yaml
# config/config_nuclei.yaml (unchanged)
channel: 1
overlap: [4, 30, 30]
method: "cellpose"
label_name: "nuclei_labels"
cellpose:
  model: "nuclei"
  diameter: 15
  do_3D: true
  gpu: true
```

```yaml
# config/config_cyto_plantseg.yaml
channel: 0                  # membrane
seed_labels: "nuclei_labels" # each tile becomes [membrane, nuclei labels]
overlap: [4, 40, 40]        # the halo must hold a whole cell
stitch: "iou"               # cells touch at every seam: join on agreement only
method: "custom"
label_name: "cyto_labels"
custom:
  module: "patchworks.plugins.plantseg"
  function: "segment"
  kwargs:
    model: "generic_confocal_3D_unet"   # generic_light_sheet_3D_unet for light-sheet
    segmentation: "nuclei_watershed"    # U-Net boundaries flooded from the nuclei
    foreground: "otsu"                  # keep the tissue only
    # max_radius_um: 15
```

Set `input` and `work_dir` in `config/common.yaml`, then, from `workflow/`:

```bash
pixi install -e plantseg
pixi run -e plantseg plantseg-fetch generic_confocal_3D_unet   # once, with internet
pixi run -e plantseg multi-plantseg-dry                        # check the plan
pixi run -e plantseg multi-plantseg-slurm                      # submit
```

The `plantseg` environment is the default one plus PlantSeg, so the
Cellpose run comes from it too. Afterwards, `nuclei_to_cyto.xlsx` gives
each nucleus its cell, and `pixi run -e viewer review <work_dir>/image.zarr`
lists any cell not holding exactly one nucleus.

### How `seed_labels` works

- **Order.** `run_multi` starts the cells' config only once the config
  producing `nuclei_labels` has finished; everything else listed (cilia,
  say) runs alongside. If the nuclei fail, the cells are skipped, not run
  without seeds. Two configs seeding each other are refused up front. A
  dry run (`-n`) waits for nothing.
- **Tiles.** The nuclei label image is stacked onto the membrane as each
  tile's second channel, halo included, so neighbouring tiles see the same
  nuclei and a cell crossing a seam is grown from the same nucleus on both
  sides. Both runs must use the same `level`, which `run_multi` already
  enforces; the plugin is told `seeds: "labels"` automatically.
- **Seeds as given.** Two touching nuclei that Cellpose split stay two
  cells; a dim nucleus Cellpose found still gets its cell. Cellpose's
  mistakes carry over the same way: a nucleus split in two makes two
  cells. Correcting the nuclei first (`patchworks review`, then
  `--write-labels nuclei_labels` and `seed_labels: nuclei_labels_reviewed`)
  gives the cells the corrected nuclei.
- **On its own**, outside `run_multi`, the cells' config needs
  `labels/nuclei_labels` already in `image.zarr`: `prepare` checks, and
  stops with a message rather than segmenting without seeds.
- **Re-segmenting the nuclei** does not re-run the cells by itself: delete
  `<work_dir>/cyto_labels` and `image.zarr/labels/cyto_labels` to grow them
  again from the new nuclei.

`nuclei_channel: 1` instead of `seed_labels` makes the plugin find the
nuclei itself, in the nuclear stain (Otsu per tile, `nuclei_*` options),
independently of Cellpose -- both segmentations then run at the same time.

Without PlantSeg, the same works with the plain
[nuclei-seeded watershed](#nuclei-seeded-watershed): set `module:
"patchworks.plugins.watershed"` with only the `foreground` and
`max_radius_um` keys, and use the default environment (`pixi run
multi-slurm` with this pair listed in `config/multi.yaml`).

## Cellpose: membrane only, in 3-D

If you stay with Cellpose on such images:

- Give it **only the membrane** (`nuclei_channel: null`): `cpsam` then has
  nothing brighter to lock onto.
- `do_3D: true` combines the three orthogonal views, so a wall faint in one
  plane is found in the others. `stitch_threshold` instead joins
  independent 2-D masks across z and fixes nothing in the masks
  themselves. A GPU (`gpu: true`) only changes the speed.

## Denoising first (CAREamics)

Noise breaks segmentations: a wall lost in the noise merges two cells.
[Noise2Void](https://careamics.github.io) learns to remove the noise from
the image itself (no clean ground truth, no annotation), so a model trained
once on a few crops of the store denoises every tile before any method sees
it.

```bash
pixi install -e careamics
# on a GPU node: the brightest crops of the channel, ~30 epochs
pixi run -e careamics denoise-train <work_dir>/image.zarr --channel 0 --out n2v_membrane.ckpt
pixi run -e careamics denoise-train <work_dir>/image.zarr --channel 1 --out n2v_nuclei.ckpt
```

then, in the segmentation config, with any `method`:

```yaml
denoise:
  model: "/path/to/n2v_membrane.ckpt"      # .ckpt, or a BioImage.IO .zip
  nuclei_model: "/path/to/n2v_nuclei.ckpt" # optional, for nuclei_channel
  # tile_size: [16, 256, 256]              # CAREamics' tiling inside a tile (VRAM)
```

and run from the `careamics` environment (`plantseg-careamics` for both).
The model paths are checked during `prepare`; the denoised image is not
stored, only used for segmenting. From the command line, `patchworks segment
... --denoise n2v_membrane.ckpt` does the same.

Look at a denoised tile before a full run:

```python
from patchworks import load_ome_zarr
from patchworks.plugins.careamics import denoise

tile = load_ome_zarr("image.zarr", channel=0)[20:44, 1000:1512, 1000:1512]
clean = denoise(tile.compute(), model="n2v_membrane.ckpt")
```
