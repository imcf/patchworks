"""Snakemake (local) script: cache the segmentation model before segmenting.

Runs on the submit host, which has network access, so the offline GPU nodes
never try to download Cellpose weights at run time (they read the shared
``$HOME/.cellpose`` cache instead).
"""

from _pw import start_log

start_log(snakemake.log[0])  # noqa: F821
cfg = snakemake.config  # noqa: F821

if cfg.get("method", "cellpose") == "cellpose":
    # _get_model downloads + caches the weights keyed by (model, gpu).
    from patchworks.plugins.cellpose import _get_model

    model = cfg["cellpose"]["model"]
    _get_model({"model": model, "gpu": False})
    print(f"[patchworks] cached segmentation model: {model}")
else:
    print(f"[patchworks] method={cfg.get('method')!r}; no model to prefetch")
