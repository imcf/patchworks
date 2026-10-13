"""Snakemake (local) script: cache the segmentation model before segmenting.

Runs on the submit host, so the offline GPU nodes never try to download
weights at run time. When the submit host is itself a compute node without
internet (the launcher's controller job), fetch first on the login node:
``pixi run fetch <config>`` (scripts/fetch_models.py).
"""

from _pw import fetch_models, start_log

start_log(snakemake.log[0])  # noqa: F821
print(f"[patchworks] {fetch_models(snakemake.config)}")  # noqa: F821
