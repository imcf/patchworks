"""Snakemake script: convert the input to a pyramidal OME-ZARR.

The rule's output is a marker file inside the store (``image.zarr/zarr.json``),
so Snakemake skips this step entirely when the store already exists — the
conversion is not redone. To force a fresh conversion, delete ``image.zarr``
(or run ``snakemake --forcerun convert``).
"""

from patchworks.plugins.ome_zarr import to_ome_zarr

from _pw import start_log

start_log(snakemake.log[0])  # noqa: F821
cfg = snakemake.config  # noqa: F821  (injected by Snakemake)
chunks = cfg.get("convert_chunks")
to_ome_zarr(
    cfg["input"],
    str(snakemake.output[0]).removesuffix("/zarr.json"),  # noqa: F821
    chunks=tuple(chunks) if chunks else None,
    shard=bool(cfg.get("shard", False)),
    reuse_pyramid=bool(cfg.get("reuse_pyramid", False)),
    progress=False,
    overwrite=True,
)
