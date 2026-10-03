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
custom:
  module: "patchworks.plugins.watershed"
  kwargs:
    nuclei_min_size: 200    # voxels: specks smaller than a nucleus
    foreground: "otsu"      # stop at the tissue edge
    # max_radius_um: 15     # or: no further than this from the nucleus
    # nuclei_threshold: 800 # intensity, if Otsu misses dim nuclei
```

Two nuclei touching each other give one seed, so one cell for two: raise
`nuclei_threshold` if that happens. Each tile's halo (`overlap`) must hold a
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
PlantSeg the cells, flooded from the nuclei, and the two are related and
checked against each other. Three files, next to `config/multi.yaml`:

```yaml
# config/multi_plantseg.yaml
common: config/common.yaml           # input, work_dir, tile_shape: shared

segmentations:
- config/config_nuclei.yaml          # Cellpose "nuclei" model on channel 1
- config/config_cyto_plantseg.yaml   # PlantSeg + nuclei seeds on channel 0

relations:
- a: nuclei_labels
  b: cyto_labels
  output: nuclei_to_cyto.xlsx

review:                              # flag cells with no nucleus, or two
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
nuclei_channel: 1           # each tile becomes [membrane, nuclei]
overlap: [4, 40, 40]        # the halo must hold a whole cell
method: "custom"
label_name: "cyto_labels"
custom:
  module: "patchworks.plugins.plantseg"
  function: "segment"
  kwargs:
    model: "generic_confocal_3D_unet"   # generic_light_sheet_3D_unet for light-sheet
    segmentation: "nuclei_watershed"    # U-Net boundaries flooded from the nuclei
    nuclei_min_size: 200                # voxels
    # nuclei_threshold: 800             # if Otsu misses dim nuclei / joins touching ones
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
Cellpose run comes from it too; both segmentations run at the same time, one
GPU job per batch of tiles each. Afterwards, `nuclei_to_cyto.xlsx` gives
each nucleus its cell, and `pixi run -e viewer review <work_dir>/image.zarr`
lists the cells that hold no nucleus or two.

!!! note "Two looks at the nuclei"

    The cells' seeds are found in the nuclear channel by the plugin itself
    (Otsu per tile), independently of the Cellpose nuclei. Where the two
    disagree -- touching nuclei that Cellpose splits but the threshold
    joins, a dim nucleus only Cellpose finds -- the review's `expect` rule
    flags the cell. Many such flags mean `nuclei_threshold` or
    `nuclei_min_size` want adjusting.

Without PlantSeg, the same run works with the plain
[nuclei-seeded watershed](#nuclei-seeded-watershed): set `module:
"patchworks.plugins.watershed"` with only the `nuclei_*`, `foreground` and
`max_radius_um` keys, and use the default environment (`pixi run multi-slurm`
with this pair listed in `config/multi.yaml`).

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
