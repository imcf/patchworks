"""Config glue for the patchworks Snakemake workflow.

The heavy lifting lives in patchworks' public API
(``spatial_tiles``/``create_stage``/``stage_tile``/``merge_tile_labels``);
these helpers only turn the Snakemake config into the right arguments.
"""

from __future__ import annotations

import json
import logging
import sys
from functools import partial
from pathlib import Path

from patchworks import load_ome_zarr


class _Tee:
    """Write to several streams at once (e.g. the SLURM log and a file)."""

    def __init__(self, *streams):
        self._streams = streams

    def write(self, data):
        for stream in self._streams:
            stream.write(data)

    def flush(self):
        for stream in self._streams:
            stream.flush()


def start_log(path, *, append=True):
    """Tee stdout/stderr (and logging) into ``path``.

    Captures prints, tracebacks and library logging into a file in the work
    directory, independent of the (often empty) SLURM job log. Line-buffered,
    so output up to a crash or OOM kill is preserved.

    Parameters
    ----------
    path : str or Path
        Log file to write. Parent directories are created.
    append : bool, optional
        Append to an existing log (keep retry history) instead of truncating.
        Default True.

    Returns
    -------
    TextIO
        The open log file (kept open for the lifetime of the process).
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    handle = open(path, "a" if append else "w", buffering=1)
    sys.stdout = _Tee(sys.__stdout__, handle)
    sys.stderr = _Tee(sys.__stderr__, handle)
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
        handlers=[logging.StreamHandler(sys.stdout)],
        force=True,
    )
    return handle


def open_image(work_dir, channel, level):
    """Open the converted image for segmentation.

    Parameters
    ----------
    work_dir : str or Path
        Workflow output directory containing ``image.zarr``.
    channel : int or None
        Channel to select.
    level : int
        Pyramid level to read.

    Returns
    -------
    da.Array
        The (lazy) image array.
    """
    return load_ome_zarr(
        str(Path(work_dir) / "image.zarr"), channel=channel, level=level
    )


def stage_path(work_dir, label_name):
    """Path of the staged-labels store for one segmentation run.

    Namespaced under ``<work_dir>/<label_name>/`` so two segmentations
    (different ``label_name``) can target the same ``work_dir`` without
    colliding — see docs/guide/snakemake.md "Running two segmentations".

    Parameters
    ----------
    work_dir : str or Path
        Workflow output directory.
    label_name : str
        This run's ``label_name`` (from config).

    Returns
    -------
    str
        ``<work_dir>/<label_name>/stage.zarr``.
    """
    return str(Path(work_dir) / label_name / "stage.zarr")


def load_tiles_json(path):
    """Load the tile manifest written by ``prepare_tiles.py``.

    Parameters
    ----------
    path : str or Path
        Path to ``tiles.json``.

    Returns
    -------
    dict
        The manifest (``tile_shape``, ``overlap``, ``occupied`` indices, …).
    """
    return json.loads(Path(path).read_text())


def build_fn(cfg):
    """Build the per-tile segmentation function from the config.

    Parameters
    ----------
    cfg : dict
        Snakemake config. ``method`` selects ``"cellpose"`` (default), a simple
        ``"threshold"`` (testing / no-GPU), or ``"custom"`` to import your own
        function (``cfg["custom"] = {module, function, kwargs}``).

    Returns
    -------
    callable
        ``(ndarray) -> ndarray`` returning integer labels.
    """
    method = cfg.get("method", "cellpose")
    if method == "custom":
        # Import a user-provided function, e.g.
        #   custom: {module: my_seg, function: segment, kwargs: {...}}
        # The module must be importable on the cluster (a file in
        # workflow/scripts/, on PYTHONPATH, or an installed package).
        import importlib

        spec = cfg["custom"]
        fn = getattr(
            importlib.import_module(spec["module"]),
            spec.get("function", "segment"),
        )
        kwargs = spec.get("kwargs") or {}
        return partial(fn, **kwargs) if kwargs else fn

    if method == "threshold":

        def fn(tile):
            from skimage.filters import threshold_otsu
            from skimage.measure import label

            thr = threshold_otsu(tile) if tile.max() > tile.min() else 0
            return label(tile > thr).astype("int32")

        return fn

    if method == "cellpose":
        from patchworks.plugins.cellpose import cellpose_fn

        cp = cfg["cellpose"]
        extra = {
            k: v
            for k, v in cp.items()
            if k not in ("model", "diameter", "do_3D", "gpu")
        }
        return cellpose_fn(
            cp.get("model", "cyto3"),
            gpu=cp.get("gpu", True),
            diameter=cp.get("diameter"),
            do_3D=cp.get("do_3D", False),
            **extra,
        )

    raise ValueError(f"unknown segmentation method: {method!r}")
