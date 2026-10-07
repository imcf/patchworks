<p align="center">
  <img src="https://raw.githubusercontent.com/imcf/patchworks/main/docs/assets/logo.png" alt="patchworks logo" width="220">
</p>

# patchworks

[![PyPI](https://img.shields.io/pypi/v/patchworks.svg)](https://pypi.org/project/patchworks/)
[![Python versions](https://img.shields.io/pypi/pyversions/patchworks.svg)](https://pypi.org/project/patchworks/)
[![License: GPLv3](https://img.shields.io/badge/License-GPLv3-blue.svg)](https://www.gnu.org/licenses/gpl-3.0)
[![Docs](https://img.shields.io/badge/docs-imcf.one%2Fpatchworks-blue)](https://imcf.one/patchworks/)

> Tiled processing of arbitrarily large images — any image, any function.

```text
┌──────┬──────┬──────┐     fn(tile) → labels      ┌──────┬──────┬──────┐
│ tile │ tile │ tile │  ─────────────────────►    │  1   │  2   │  3   │
├──────┼──────┼──────┤                            ├──────┼──────┼──────┤
│ tile │ tile │ tile │                            │  4   │  5   │  6   │   globally
├──────┼──────┼──────┤                            ├──────┼──────┼──────┤   consistent
│ tile │ tile │ tile │                            │  7   │  8   │  9   │   labels
└──────┴──────┴──────┘                            └──────┴──────┴──────┘
```

patchworks splits an image too large for memory into tiles, runs **any
segmentation function** on each tile, and stitches the results into one
consistent label image — stored with the image in a single OME-Zarr, with a
table of every object.

> [!NOTE]
> **On how this was written.** Large parts of patchworks were vibe coded —
> written with heavy LLM assistance rather than line by line. It is covered by
> a test suite and has been run on real data, so it is not untested, but the
> usual caveats apply: read the code before you trust it with anything
> irreplaceable, and please open an issue if something looks off.

## Install

```bash
pip install patchworks                 # core
pip install "patchworks[cellpose]"     # + Cellpose
pip install "patchworks[bioio,imaris]" # + converting CZI/LIF/ND2/TIFF/.ims to OME-Zarr
pip install "patchworks[napari]"       # + viewing and reviewing in napari
pip install "patchworks[all]"          # everything above and more
```

GPU options of the DoG plugin need `cupy` matching your CUDA version
(`pip install cupy-cuda12x`). PlantSeg is on conda-forge only. See
[Getting started](https://imcf.one/patchworks/getting_started/) for every
extra.

## Three ways to use it

**Python** — any function from a tile to labels:

```python
from patchworks import tile_process
from patchworks.plugins.cellpose import cellpose_fn

tile_process("scan.zarr", cellpose_fn("cyto3", gpu=True, diameter=30))
```

The labels go into `scan.zarr/labels/labels`, as a multiscale pyramid next
to the image.

**Command line** — the same building blocks, no script:

```bash
patchworks convert scan.czi scan.zarr
patchworks segment scan.zarr --method cellpose --model cyto3 --diameter 30 --gpu
patchworks view scan.zarr
```

**On a cluster** — a Snakemake workflow with a pixi environment: convert,
segment tile batches as GPU jobs, merge, relate label images (which cell
each nucleus is in), and bundle, from one YAML config per segmentation:

```bash
cd workflow
pixi run multi-slurm --config my_multi.yaml
```

## What you get

- **One OME-Zarr** holding the image and every label image, readable by
  napari, Fiji and any OME-Zarr tool.
- **Object tables** — size, position, shape — and **relations** between label
  images (the cell of each nucleus, the position of each cilium in its cell),
  exported as spreadsheets.
- **Review in napari**: `patchworks review scan.zarr` steps through the
  objects most likely to be wrong; corrections flow into the tables.
- Plugins for **Cellpose**, **PlantSeg**, a **nuclei-seeded watershed**,
  **difference of Gaussians** (cilia, spots) and **Noise2Void** denoising —
  or bring your own function.

## Documentation

**<https://imcf.one/patchworks/>** — getting started, the cluster workflow,
guides, examples and the API reference.

## License

GNU General Public License v3.0 (GPL-3.0). See [LICENSE](LICENSE).
