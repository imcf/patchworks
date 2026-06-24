"""Snakemake script: convert the input to a pyramidal OME-ZARR."""

from patchworks.plugins.ome_zarr import to_ome_zarr

cfg = snakemake.config  # noqa: F821  (injected by Snakemake)
chunks = cfg.get("convert_chunks")
to_ome_zarr(
    cfg["input"],
    snakemake.output[0],  # noqa: F821
    chunks=tuple(chunks) if chunks else None,
    shard=bool(cfg.get("shard", False)),
    reuse_pyramid=bool(cfg.get("reuse_pyramid", False)),
    progress=False,
    overwrite=True,
)
