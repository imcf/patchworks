"""patchworks — tiled processing for any image, any function.

Process arbitrarily large images by splitting them into overlapping tiles,
running any callable on each tile, and stitching the results back into globally
consistent labels.

📖 **Full documentation, guides and tutorials:**
<https://imcf.one/patchworks/>

Quick start
-----------
>>> from patchworks import tile_process
>>>
>>> def my_fn(tile):
...     from skimage.filters import threshold_otsu
...     from skimage.measure import label
...     return label(tile > threshold_otsu(tile)).astype("int32")
>>>
>>> result = tile_process("image.zarr", my_fn, write_to="labels.zarr")

With Cellpose:

>>> from patchworks.plugins.cellpose import cellpose_fn
>>> fn = cellpose_fn("cyto3", gpu=True, diameter=30)
>>> tile_process("image.zarr", fn, tile_shape=(1, 2048, 2048),
...              overlap=20, write_to="labels.zarr", progress=True)
"""

from importlib.metadata import PackageNotFoundError
from importlib.metadata import version as _pkg_version

from ._chunks import (
    auto_overlap,
    auto_tile_shape,
    auto_tile_shape_cellpose,
    cpu_allocation,
    safe_worker_count,
)
from ._cluster import make_local_cluster
from ._core import tile_process
from ._distributed import (
    create_stage,
    normalize_overlap,
    spatial_tiles,
    stage_tile,
)
from ._io import auto_empty_threshold, estimate_empty_tiles, load_ome_zarr
from ._merge import capped_output_chunks, merge_tile_labels
from ._occupancy import build_occupancy_map, occupancy_path, tile_occupancy
from ._postprocess import dilate_labels
from ._relabel import relabel_sequential_array, relabel_sequential_zarr
from ._relations import label_relations

try:
    __version__ = _pkg_version("patchworks")
except PackageNotFoundError:  # not installed (e.g. running from a checkout)
    __version__ = "0+unknown"
__all__ = [
    "tile_process",
    "merge_tile_labels",
    "capped_output_chunks",
    "auto_overlap",
    "auto_tile_shape",
    "auto_tile_shape_cellpose",
    "cpu_allocation",
    "safe_worker_count",
    "load_ome_zarr",
    "estimate_empty_tiles",
    "auto_empty_threshold",
    "build_occupancy_map",
    "occupancy_path",
    "tile_occupancy",
    "make_local_cluster",
    "relabel_sequential_array",
    "relabel_sequential_zarr",
    "label_relations",
    "normalize_overlap",
    "spatial_tiles",
    "create_stage",
    "stage_tile",
    "dilate_labels",
]
