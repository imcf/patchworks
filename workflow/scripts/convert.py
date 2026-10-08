"""Snakemake script: convert the input to a pyramidal OME-ZARR.

The rule's output is a marker file inside the store (``image.zarr/zarr.json``,
or ``.zgroup`` under ``ngff_version: "0.4"``),
so Snakemake skips this step entirely when the store already exists — the
conversion is not redone. To force a fresh conversion, delete ``image.zarr``
(or run ``snakemake --forcerun convert``).
"""

import dask
from patchworks import cpu_allocation, set_compression
from patchworks.plugins.ome_zarr import to_ome_zarr

from _pw import start_log

start_log(snakemake.log[0])  # noqa: F821
# Codec for every array this step creates (config `compression:`).
set_compression(snakemake.config.get("compression", "zstd"))  # noqa: F821
cfg = snakemake.config  # noqa: F821  (injected by Snakemake)
chunks = cfg.get("convert_chunks")

# dask's threads bounded by the job's CPUs, not the machine's.
dask.config.set(scheduler="threads", num_workers=cpu_allocation())

to_ome_zarr(
    cfg["input"],
    # The marker's name depends on the zarr format (see common.smk); strip
    # whichever one this run uses rather than assuming v3's.
    str(snakemake.output[0])  # noqa: F821
    .removesuffix("/zarr.json")
    .removesuffix("/.zgroup"),
    sequence_pattern=cfg.get("sequence_pattern"),
    chunks=tuple(chunks) if chunks else None,
    shard=bool(cfg.get("shard", False)),
    ngff_version=cfg.get("ngff_version", "auto"),
    reuse_pyramid=bool(cfg.get("reuse_pyramid", False)),
    # Progress goes to the log as periodic lines, not a redrawing bar:
    # this runs for hours in a batch job where silence is indistinguishable
    # from a hang.
    progress=True,
    overwrite=True,
)
