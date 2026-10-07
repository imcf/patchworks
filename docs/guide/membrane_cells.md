# Cells from a membrane stain

With a membrane marker and a nuclear dye, Cellpose 4 (`cpsam`) tends to
segment the bright nuclei, and on the membrane alone a faint wall merges two
cells. patchworks has plugins for this, all ordinary
[custom functions](custom_segmentation.md):

| Method | Cells come from | Best when |
| --- | --- | --- |
| [Nuclei-seeded watershed](#nuclei-seeded-watershed) | the membrane, flooded from the nuclei | every cell has one nucleus: start here |
| [PlantSeg](#plantseg) | a boundary U-Net, then GASP / multicut, optionally seeded by nuclei | the membrane is uneven or noisy |
| [Cellpose, membrane only](#cellpose-membrane-only) | Cellpose in 3-D | you want to stay with Cellpose |

Any of them can run on [denoised tiles](#denoising-first).

Watershed and PlantSeg give every voxel to a cell, so neighbours touch at
every tile seam: they need `stitch: "iou"` (the workflow refuses the config
without it; on the command line, pass `--stitch iou`).

## Nuclei-seeded watershed

Nuclei are easy to find; flooding the membrane from them grows each nucleus
to its cell's walls, one cell per nucleus, even across a gap in a wall.

```yaml
channel: 0              # membrane
nuclei_channel: 1       # each tile becomes [membrane, nuclei]
method: "custom"
label_name: "cyto_labels"
stitch: "iou"
custom:
  module: "patchworks.plugins.watershed"
  kwargs:
    nuclei_min_size: 200    # voxels
    foreground: "otsu"      # stop at the tissue edge
    # max_radius_um: 15     # or: at most this far from the nucleus
    # nuclei_threshold: 800 # if Otsu misses dim nuclei, or two touching nuclei give one seed
```

The halo (`overlap`) must hold a whole cell.

## PlantSeg

[PlantSeg](https://github.com/kreshuklab/plant-seg) predicts cell boundaries
with a U-Net trained on membrane stains and partitions them into cells. It is
on conda-forge only, so it has its own environment (on a login node without
a GPU, set `CONDA_OVERRIDE_CUDA` first, see [Set up](snakemake.md#1-set-up)):

```bash
pixi install -e plantseg
pixi run -e plantseg plantseg-fetch generic_confocal_3D_unet   # once, with internet
```

```yaml
method: "custom"
stitch: "iou"
custom:
  module: "patchworks.plugins.plantseg"
  kwargs:
    model: "generic_confocal_3D_unet"   # or generic_light_sheet_3D_unet, ...
    segmentation: "gasp"                # gasp | mutex_ws | multicut | dt_watershed
    beta: 0.6                           # lower merges more, higher splits more
    foreground: "otsu"
```

With nuclei, `segmentation: "nuclei_watershed"` floods the U-Net's boundaries
from them, and `"lifted_multicut"` keeps supervoxels of one nucleus together.
Tiles are resampled to the model's training voxel size automatically.

## Cellpose nuclei, then PlantSeg cells

The workflow ships this pairing: Cellpose segments the nuclei, PlantSeg grows
one cell from each, and the two are related. Set `input` and `work_dir` in
`config/common.yaml`, then:

```bash
pixi run -e plantseg multi-plantseg-dry
pixi run -e plantseg multi-plantseg-slurm
```

`config/multi_plantseg.yaml` lists `config_nuclei.yaml` and
`config_cyto_plantseg.yaml`; the second one seeds the cells with the first's
labels:

```yaml
seed_labels: "nuclei_labels"        # each tile becomes [membrane, nuclei labels]
custom:
  module: "patchworks.plugins.plantseg"
  kwargs:
    segmentation: "nuclei_watershed"
```

- The cells start once the nuclei are done; other segmentations run
  alongside. Run your own multi config with PlantSeg from the same
  environment: `pixi run -e plantseg multi-slurm --config my_multi.yaml`.
- Mistakes in the nuclei carry over to the cells. Correct them first with
  [`patchworks review`](review.md), write them with `--write-labels
  nuclei_labels`, and seed from `nuclei_labels_reviewed`.
- Re-segmenting the nuclei does not re-run the cells: delete
  `<work_dir>/cyto_labels` and `image.zarr/labels/cyto_labels` first.
- Without PlantSeg, the same pairing works with
  `module: "patchworks.plugins.watershed"` in the default environment.

## Cellpose, membrane only

Give Cellpose only the membrane (`nuclei_channel: null`) and set
`do_3D: true`: it combines three orthogonal views, so a wall faint in one
plane is found in the others.

## Denoising first

[Noise2Void](https://careamics.github.io) learns to remove noise from the
image itself, without clean ground truth. Train once per channel (on a GPU
node):

```bash
pixi install -e careamics
pixi run -e careamics denoise-train <work_dir>/image.zarr --channel 0 --out n2v_membrane.ckpt
```

then add to any segmentation config, and run from the `careamics`
environment (`plantseg-careamics` for both):

```yaml
denoise:
  model: "/path/to/n2v_membrane.ckpt"
  # nuclei_model: "/path/to/n2v_nuclei.ckpt"   # for nuclei_channel
```

Only the segmentation sees the denoised tiles; nothing extra is stored.
Command line: `patchworks segment ... --denoise n2v_membrane.ckpt`. Try it on
one tile with `patchworks.plugins.careamics.denoise(tile, model=...)`.
