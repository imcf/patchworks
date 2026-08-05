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

# Segmentation methods `_build_method_fn` can dispatch. Adding one means
# teaching that function to build it and adding the name here; the validator
# reads this list so the two cannot drift apart. Most new methods need
# neither: `method: "custom"` already imports any `(tile) -> labels` callable.
KNOWN_METHODS = ("cellpose", "threshold", "custom")


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


def open_image(work_dir, channel, level, nuclei_channel=None):
    """Open the converted image for segmentation.

    Parameters
    ----------
    work_dir : str or Path
        Workflow output directory containing ``image.zarr``.
    channel : int or None
        Channel to select.
    level : int
        Pyramid level to read.
    nuclei_channel : int or None, optional
        Second channel to stack in front of *channel*, giving a
        ``(2, z, y, x)`` array whose leading axis is ``[channel,
        nuclei_channel]``. This is Cellpose's cyto+nucleus pair (see
        ``config/config_cyto.yaml``). ``None`` (default) returns the plain
        ``(z, y, x)`` single-channel array.

    Returns
    -------
    da.Array
        The (lazy) image array.
    """
    path = str(Path(work_dir) / "image.zarr")
    if nuclei_channel is None:
        return load_ome_zarr(path, channel=channel, level=level)

    import dask.array as da

    if nuclei_channel == channel:
        raise ValueError(
            f"nuclei_channel ({nuclei_channel}) is the same as channel "
            f"({channel}); Cellpose would get the same image twice. Channel "
            "indices are 0-based."
        )
    # Two lazy reads of the same spatial box, stacked. The tiling stays
    # spatial -- stage_tile carries this axis through rather than tiling it.
    return da.stack(
        [
            load_ome_zarr(path, channel=c, level=level)
            for c in (channel, nuclei_channel)
        ],
        axis=0,
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


def _with_voxel_size(fn, kwargs, cfg):
    """Pass the image's calibration to a function that asks for it.

    A method that needs physical units -- a deconvolution's voxel sizes, a
    diameter in micrometers -- should not have them retyped into the config
    where they can drift away from the image. Any custom function that
    declares a ``voxel_size`` parameter gets ``{"z": .., "y": .., "x": ..}``
    from the store's own NGFF calibration, unless the config sets it.

    Returns *kwargs* unchanged when the function does not take it, the config
    already supplied it, or the store carries no calibration.
    """
    import inspect

    if "voxel_size" in kwargs:
        return kwargs
    # A **kwargs passthrough names its real target (see dog.segment).
    target = getattr(fn, "patchworks_kwargs_target", fn)
    try:
        params = inspect.signature(target).parameters
    except (TypeError, ValueError):
        return kwargs
    if "voxel_size" not in params:
        return kwargs

    from patchworks.plugins.ome_zarr import read_pixel_size

    store = str(Path(cfg["work_dir"]) / "image.zarr")
    calibration = read_pixel_size(store)
    if not calibration:
        logging.getLogger(__name__).warning(
            "%s takes voxel_size but %s carries no calibration; set the "
            "physical sizes in the config instead.",
            getattr(target, "__name__", target),
            store,
        )
        return kwargs
    return {**kwargs, "voxel_size": calibration}


def _unknown_kwargs(fn, provided, *, skip=()) -> list[str]:
    """Keys in *provided* that *fn* would reject, ignoring *skip*.

    Returns an empty list when the signature can't be introspected (C
    extensions, ``**kwargs``-only wrappers) rather than guessing.
    """
    import inspect

    try:
        params = inspect.signature(fn).parameters
    except (TypeError, ValueError):
        return []
    if any(p.kind is p.VAR_KEYWORD for p in params.values()):
        return []
    return sorted(set(provided) - set(params) - set(skip))


def validate_config(cfg) -> None:
    """Reject config mistakes before any expensive step runs.

    Everything here used to surface much later and much more expensively: a
    misspelled ``cellpose:`` key was swept into ``extra`` and forwarded blind
    to ``model.eval()``, so it failed in the first GPU job -- after convert
    and prepare had already run. ``gpu_memory_gb: "auto"`` was truthy, so
    ``"auto" * 1024**3`` tried to build a ~4 GB string.

    Raises
    ------
    ValueError
        With every problem found, not just the first.
    """
    problems = []

    gpu_gb = cfg.get("gpu_memory_gb")
    if gpu_gb is not None and (
        isinstance(gpu_gb, bool)
        or not isinstance(gpu_gb, (int, float))
        or gpu_gb <= 0
    ):
        problems.append(
            f"gpu_memory_gb must be null or a positive number of GB (e.g. 24 "
            f'for an RTX 4090); got {gpu_gb!r}. Only tile_shape accepts "auto".'
        )

    ts = cfg.get("tile_shape", "auto")
    if isinstance(ts, str):
        if ts != "auto":
            problems.append(
                f'tile_shape must be "auto" or a list like [16, 1024, 1024]; '
                f"got {ts!r} (a bare string would be read character by "
                "character)"
            )
    elif ts is not None and not all(isinstance(v, int) and v > 0 for v in ts):
        problems.append(f"tile_shape must be positive integers; got {ts!r}")

    nuc = cfg.get("nuclei_channel")
    if nuc is not None:
        if isinstance(nuc, bool) or not isinstance(nuc, int):
            problems.append(
                "nuclei_channel must be null or a 0-based channel index "
                f"(e.g. 1); got {nuc!r}"
            )
        elif nuc == cfg.get("channel"):
            problems.append(
                f"nuclei_channel ({nuc}) is the same as channel; Cellpose "
                "would get the same image twice. Indices are 0-based."
            )
        if cfg.get("method", "cellpose") == "threshold":
            problems.append(
                'nuclei_channel has no meaning for method: "threshold", '
                "which segments a single channel"
            )

    method = cfg.get("method", "cellpose")
    if method not in KNOWN_METHODS:
        listed = ", ".join(f'"{m}"' for m in KNOWN_METHODS)
        problems.append(f"method must be one of {listed}; got {method!r}")

    if method == "cellpose":
        cp = cfg.get("cellpose")
        if not isinstance(cp, dict):
            problems.append('method: "cellpose" needs a cellpose: block')
        else:
            problems.extend(_cellpose_problems(cp))
    elif method == "custom":
        problems.extend(_custom_problems(cfg.get("custom")))

    if problems:
        raise ValueError("invalid config:\n  - " + "\n  - ".join(problems))


def _cellpose_problems(cp: dict) -> list[str]:
    """Unknown keys in the ``cellpose:`` block, checked against model.eval."""
    known = ("model", "diameter", "do_3D", "gpu")
    extra = {k: v for k, v in cp.items() if k not in known}
    if not extra:
        return []
    try:
        from patchworks.plugins.cellpose import _get_model
    except ImportError:
        return []
    try:
        model = _get_model({"model": cp.get("model", "cyto3"), "gpu": False})
    except Exception:
        # Weights unavailable here (fetch_model runs separately); skip rather
        # than fail a config that may be perfectly fine.
        return []
    unknown = _unknown_kwargs(
        model.eval, extra, skip=("channels", "channel_axis")
    )
    if unknown:
        return [
            f"unknown cellpose: key(s) {unknown} -- these are forwarded to "
            "model.eval(), which does not accept them"
        ]
    return []


def _custom_problems(spec) -> list[str]:
    """Check a ``custom:`` spec resolves and its kwargs match the target."""
    if not isinstance(spec, dict) or "module" not in spec:
        return ['method: "custom" needs a custom: block with a module']
    import importlib

    try:
        module = importlib.import_module(spec["module"])
    except ImportError as exc:
        return [f"custom.module {spec['module']!r} is not importable: {exc}"]
    name = spec.get("function", "segment")
    fn = getattr(module, name, None)
    if fn is None:
        return [f"{spec['module']}.{name} does not exist"]
    # A `**kwargs` passthrough accepts anything, so check the function it
    # actually forwards to when the plugin names one.
    target = getattr(fn, "patchworks_kwargs_target", fn)
    unknown = _unknown_kwargs(target, spec.get("kwargs") or {})
    if unknown:
        return [f"unknown custom.kwargs {unknown} for {spec['module']}.{name}"]
    return []


def build_fn(cfg):
    """Build the per-tile segmentation function from the config.

    Parameters
    ----------
    cfg : dict
        Snakemake config. ``method`` selects ``"cellpose"`` (default), a simple
        ``"threshold"`` (testing / no-GPU), or ``"custom"`` to import your own
        function (``cfg["custom"] = {module, function, kwargs}``). Optional
        ``cfg["dilate"]``: int, pixels to grow labels by after segmentation
        (via ``patchworks.dilate_labels``), applied regardless of ``method``.
        Omitted/0 disables dilation. ``cfg["dilate_gpu"]``: bool, dilate via
        cupyx instead of scipy (default ``False``); only takes effect when
        ``dilate`` is set, and needs a GPU allocated for the segment job
        (independent of whether ``method`` itself uses one).

    Returns
    -------
    callable
        ``(ndarray) -> ndarray`` returning integer labels.
    """
    fn = _build_method_fn(cfg)

    dilate = cfg.get("dilate")
    if dilate:
        from patchworks import dilate_labels

        fn = dilate_labels(
            fn, iterations=dilate, use_gpu=cfg.get("dilate_gpu", False)
        )

    return fn


def _build_method_fn(cfg):
    """Build the per-tile segmentation function for ``cfg["method"]``.

    Parameters
    ----------
    cfg : dict
        Snakemake config, see :func:`build_fn`.

    Returns
    -------
    callable
        ``(ndarray) -> ndarray`` returning integer labels.
    """
    method = cfg.get("method", "cellpose")  # see KNOWN_METHODS
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
        kwargs = dict(spec.get("kwargs") or {})
        kwargs = _with_voxel_size(fn, kwargs, cfg)
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
        if cfg.get("nuclei_channel") is not None:
            # open_image stacks [channel, nuclei_channel] on axis 0; tell
            # Cellpose where they are. setdefault so an explicit
            # cellpose.channel_axis in the config still wins.
            extra.setdefault("channel_axis", 0)
        return cellpose_fn(
            cp.get("model", "cyto3"),
            gpu=cp.get("gpu", True),
            diameter=cp.get("diameter"),
            do_3D=cp.get("do_3D", False),
            **extra,
        )

    raise ValueError(f"unknown segmentation method: {method!r}")
