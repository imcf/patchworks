# Cellpose 2-D

Segment every z-slice independently: each tile is one plane `(1, y, x)`.

```bash
pip install "patchworks[cellpose,gpu]"
```

```python
from functools import partial
from patchworks import auto_tile_shape_cellpose, tile_process
from patchworks.plugins.cellpose import cellpose_fn

tile_process(
    "image.zarr",
    cellpose_fn("cyto3", gpu=True, diameter=30),
    channel=0,
    tile_shape=partial(auto_tile_shape_cellpose, diameter=30, use_gpu=True),  # fits the VRAM
    overlap=20,        # about one cell diameter
    skip_empty=True,
)
```

Cellpose arguments not listed in `cellpose_fn`'s signature (`flow_threshold`,
`cellprob_threshold`, ...) are passed through. A cyto stain with a nuclear
channel: on the cluster, set `nuclei_channel:`
([Options worth knowing](../guide/snakemake.md#options-worth-knowing)).
