"""Snakemake script: convert the input to a pyramidal OME-ZARR.

The rule's output is a marker file inside the store (``image.zarr/zarr.json``),
so Snakemake skips this step entirely when the store already exists — the
conversion is not redone. To force a fresh conversion, delete ``image.zarr``
(or run ``snakemake --forcerun convert``).
"""

import dask
from patchworks import cpu_allocation
from patchworks.plugins.ome_zarr import to_ome_zarr

from _pw import start_log

start_log(snakemake.log[0])  # noqa: F821
cfg = snakemake.config  # noqa: F821  (injected by Snakemake)
chunks = cfg.get("convert_chunks")

# dask's threaded scheduler otherwise spawns one worker per *machine* core, not
# per allocated core: on a 128-core node a 32-core job would run 128 chunk
# reads at once and blow through its cgroup limit. This is the same class of
# bug that OOM-killed conversion before, previously worked around by raising
# mem_mb in the profile.
dask.config.set(scheduler="threads", num_workers=cpu_allocation())

to_ome_zarr(
    cfg["input"],
    str(snakemake.output[0]).removesuffix("/zarr.json"),  # noqa: F821
    sequence_pattern=cfg.get("sequence_pattern"),
    chunks=tuple(chunks) if chunks else None,
    shard=bool(cfg.get("shard", False)),
    reuse_pyramid=bool(cfg.get("reuse_pyramid", False)),
    progress=False,
    overwrite=True,
)
