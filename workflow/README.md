# patchworks cluster workflow

Segments an image too large for memory on a SLURM cluster: the image is
converted once, its tiles are segmented as parallel GPU jobs, and the
results are merged into the image's OME-Zarr store with a table of every
object. Several segmentations can run together and be related to each other.

Full guide: <https://imcf.one/patchworks/guide/snakemake/>

```bash
pixi install
# one segmentation: your copy of config/config.yaml in config/local/
# (git ignores it), and profile/slurm/config.yaml for your cluster
pixi run dry   --configfile config/local/my_run.yaml
pixi run slurm --configfile config/local/my_run.yaml
pixi run status config/local/my_run.yaml     # how far it got
# several, related: list them in a multi.yaml (see config/multi.yaml)
pixi run multi-dry   --config my_multi.yaml
pixi run multi-slurm --config my_multi.yaml
```

| Path | What |
| --- | --- |
| `config/` | example configs: one segmentation (`config.yaml`), shared settings (`common.yaml`), several (`multi.yaml`) |
| `profile/slurm/config.yaml` | partitions, GPU request, memory and time per step |
| `Snakefile`, `rules/` | the steps: convert, prepare, segment, merge |
| `scripts/` | what each step runs, plus `run_multi.py` (several configs) and `relate.py` |
| `viewer/` | a small pixi workspace for napari on any OS |
