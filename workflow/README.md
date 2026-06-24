# patchworks Snakemake workflow

A SLURM-ready pipeline that segments an arbitrarily large image and spreads the
expensive Cellpose step across **many GPUs** — one tile per SLURM job.

> **Full step-by-step guide:**
> <https://imcf.one/patchworks/guide/snakemake/> — install, configure every
> field, dry-run, local vs SLURM, monitoring, outputs and troubleshooting.

```text
convert ──▶ prepare (checkpoint) ──▶ segment {tile}  ──▶ merge
                                     one GPU job/tile
```

## Why

`tile_process` runs tiles serially on a single GPU. This workflow instead
submits **one GPU job per (non-empty) tile**, so N GPUs cut the wall-time ~N×.
Each job writes a disjoint chunk of a shared stage store; a final CPU job
stitches labels across tile boundaries and writes them back into the image as a
calibrated, multi-scale `labels/` group.

## Steps

1. **convert** — input (`.ims`, `.czi`, `.lif`, `.nd2`, OME-TIFF, `.zarr`) → a
   pyramidal OME-ZARR (`image.zarr`).
2. **prepare** (checkpoint) — plan tiles, skip background tiles, create the
   empty `stage.zarr`, and list the tiles to segment.
3. **segment {tile}** — one GPU job per tile: read tile + halo, run Cellpose,
   trim, write to the stage. Scattered across the cluster.
4. **merge** — zarr-native boundary stitch + renumber, written into
   `image.zarr/labels/<name>/`.

## Install

```bash
pip install "patchworks[workflow,cellpose,imaris,bioio]"
# workflow → snakemake + the SLURM executor plugin
```

Prefer pixi? No conda needed — a `pixi.toml` is included:

```bash
pixi install
pixi run dry    # dry-run
pixi run go     # run locally
pixi run slurm  # submit to SLURM
```

## Configure

Edit `config/config.yaml` (input, output dir, channel, tile shape, Cellpose
model/diameter/`do_3D`, …) and `profile/slurm/config.yaml` (partitions,
account, GPU request).

## Run

```bash
# locally (single machine) — mtime triggers => upgrades don't redo conversion
snakemake --cores 8 --configfile config/config.yaml --rerun-triggers mtime

# on SLURM — one GPU job per tile, up to `jobs:` in parallel
# (the profile already sets --rerun-triggers mtime)
snakemake --workflow-profile profile/slurm --configfile config/config.yaml
```

The GPU request lives in `profile/slurm/config.yaml` under
`set-resources: segment:` (`--gres=gpu:1`). Raise `jobs:` to use more GPUs at
once.

## Output

`<work_dir>/image.zarr` — the image plus `labels/<label_name>/` (multi-scale,
calibrated). Open it directly:

```python
from patchworks.plugins.napari import view_in_napari
view_in_napari("<work_dir>/image.zarr")   # auto-loads the labels
```

## Layout

```text
workflow/
  Snakefile            # includes the rule files below
  rules/               # convert.smk, segment.smk, merge.smk, common.smk
  scripts/             # thin wrappers over patchworks' public API
  config/config.yaml
  profile/slurm/config.yaml
```

The rule scripts are intentionally thin — the work is done by patchworks'
public API (`spatial_tiles`, `create_stage`, `stage_tile`, `merge_tile_labels`,
`write_labels`), so the same per-tile distribution is available from your own
code too.

## Notes

- Tiles overlap on read (halo) but write **disjoint** regions, so the per-tile
  jobs are safe to run concurrently.
- Background tiles are skipped (`skip_empty`), so only occupied tiles become
  jobs.
- `method:` selects `cellpose` (default) or a simple `threshold` (no GPU —
  handy for testing or quick masks).
- For very large stores, set `shard: true` in the config to cut the file count.
