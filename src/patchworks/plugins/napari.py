"""napari viewer plugin for patchworks.

Convenience helpers to open an OME-ZARR image and overlay the label array
produced by :func:`patchworks.tile_process` as a proper napari *Labels* layer
— in one call.

Everything is loaded lazily: OME-ZARR pyramids are handed to napari as a
multi-scale list (napari fetches only the resolution/region on screen), so even
terabyte stores open instantly.

napari is an optional, GUI-heavy dependency. Install it with
``pip install "patchworks[napari]"``.

Usage
-----
>>> from patchworks import tile_process  # doctest: +SKIP
>>> from patchworks.plugins.napari import view_in_napari  # doctest: +SKIP
>>>
>>> # labels are written into scan.zarr/labels/ by default …
>>> tile_process("scan.zarr", fn)  # doctest: +SKIP
>>> # … so the viewer finds and overlays them with no labels= argument:
>>> view_in_napari("scan.zarr")  # doctest: +SKIP
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any, Mapping, Union

import dask.array as da
import numpy as np

from .._io import from_zarr_any, load_ome_zarr, open_group_any
from .ome_zarr import read_ngff_attr

logger = logging.getLogger(__name__)


def short_numba_cache() -> None:
    """Keep numba's compiled-code cache out of the install directory.

    napari's labels colouring is numba-compiled, and numba caches the
    result next to the source, in site-packages. When the environment sits
    deep on a network share (a pixi environment inside a project on a
    mapped drive, say), that cache path passes Windows' 260-character limit
    and every Labels layer fails to draw with a FileNotFoundError on a
    ``.nbc.tmp`` file. A short local directory avoids it, and costs nothing
    elsewhere. An explicit ``NUMBA_CACHE_DIR`` is kept; numba reads it once,
    at import, so this must run before napari (or numba) is imported.
    """
    import os
    import tempfile

    os.environ.setdefault(
        "NUMBA_CACHE_DIR",
        os.path.join(tempfile.gettempdir(), "patchworks-numba-cache"),
    )


def _require_napari():
    """Import and return napari, or raise an actionable ImportError.

    Returns
    -------
    module
        The imported ``napari`` module.
    """
    short_numba_cache()
    try:
        import napari
    except ImportError as exc:
        raise ImportError(
            "napari is not installed. Install it with:\n"
            "    pip install 'patchworks[napari]'"
        ) from exc
    return napari


def _is_zarr(src: Any) -> bool:
    """Whether *src* names a zarr store, directory or ``.zip`` bundle.

    The ``.zip`` case matters: everything downstream reads a bundle fine,
    but this is the gate that decides whether to go there at all. Matching
    only ``.zarr`` sent a bundle to bioio, which reported a missing
    optional dependency -- a confusing way to say "this is not an image
    file", about a store patchworks had just written.

    Parameters
    ----------
    src : Any
        Candidate source.

    Returns
    -------
    bool
        True for a str/Path ending in ``.zarr``, or naming a ``.zip``
        bundle (optionally with a group path after it).
    """
    if not isinstance(src, (str, Path)):
        return False
    from .._io import split_zip_path

    text = str(src).rstrip("/\\")
    return text.endswith(".zarr") or split_zip_path(text) is not None


def _has_multiscales(path: Union[str, Path]) -> bool:
    """Whether a zarr group carries NGFF ``multiscales`` metadata.

    Parameters
    ----------
    path : str or Path
        Zarr group path.

    Returns
    -------
    bool
        True if the group has a ``multiscales`` attribute.
    """
    root = open_group_any(path)
    return read_ngff_attr(root.attrs, "multiscales") is not None


def _multiscale_levels(
    path: Union[str, Path], channel: int | None
) -> list[da.Array]:
    """Return every pyramid level as a lazy dask array (napari multi-scale).

    Parameters
    ----------
    path : str or Path
        OME-ZARR group path.
    channel : int or None
        Channel to select, or ``None`` to keep all channels.

    Returns
    -------
    list of da.Array
        One lazy array per resolution level.
    """
    root = open_group_any(path)
    datasets = read_ngff_attr(root.attrs, "multiscales")[0]["datasets"]
    return [
        load_ome_zarr(path, channel=channel, level=i)
        for i in range(len(datasets))
    ]


def _resolve_image(
    source: Union[da.Array, str, Path], channel: int | None
) -> Union[da.Array, list[da.Array]]:
    """Resolve *source* into data napari can display (lazily).

    Parameters
    ----------
    source : da.Array, str or Path
        OME-ZARR store, other image file, or an in-memory array.
    channel : int or None
        Channel to display, or ``None`` to keep all channels.

    Returns
    -------
    da.Array or list of da.Array
        A single array, or a multi-scale list for an OME-ZARR pyramid.
    """
    if _is_zarr(source):
        if _has_multiscales(source):
            return _multiscale_levels(source, channel)
        return load_ome_zarr(source, channel=channel)
    if isinstance(source, (str, Path)):
        # Any other file format → bioio (reuse the conversion plugin's reader).
        from .ome_zarr import _open_bioio

        # _open_bioio returns (array, axes, pixel size); unpacking two
        # crashed every non-zarr file, and channel= was never applied.
        arr, axes, _ = _open_bioio(str(source), 0)
        if channel is not None and "c" in axes:
            arr = arr[(slice(None),) * axes.index("c") + (channel,)]
        return arr
    return source


def _pyramid_translate(
    path: Union[str, Path], ndim: int
) -> "list[float] | None":
    """Level-0 physical offset, e.g. of labels segmented at a coarser level.

    Returns None when there is none, leaving napari's default of 0.
    """
    from .ome_zarr import read_translation

    offset = read_translation(str(path))
    if not offset:
        return None
    return [float(offset.get(a, 0.0)) for a in _store_axes(path, ndim)]


def _store_axes(path: Union[str, Path], ndim: int) -> str:
    """The store's own axis names (its multiscales metadata), or the usual
    trailing ones when it records none."""
    from .ome_zarr import _DEFAULT_ORDER, _multiscale_meta

    meta = _multiscale_meta(str(path)) or {}
    names = "".join(
        str(a.get("name", "") if isinstance(a, dict) else a)[:1]
        for a in meta.get("axes") or []
    )
    if len(names) == ndim:
        return names
    return _DEFAULT_ORDER[len(_DEFAULT_ORDER) - ndim :]


def _pyramid_calibration(
    path: Union[str, Path], ndim: int
) -> tuple[list[float], list[str]] | tuple[None, None]:
    """Read the level-0 physical scale and units from an OME-ZARR's metadata.

    Without this, napari shows every axis with an implicit scale of 1 and
    "pixel" units, so a volume with anisotropic voxels (e.g. z coarser than
    x/y) renders with the wrong aspect ratio and no real-world units.

    Parameters
    ----------
    path : str or Path
        OME-ZARR store path (image or label group) — each carries its own
        ``multiscales`` metadata, so image and label stores are read
        independently.
    ndim : int
        Number of spatial dimensions of the loaded array (2 -> "yx", 3 ->
        "zyx"), used to align the calibration to the right axes.

    Returns
    -------
    tuple
        ``(scale, units)`` — physical size and unit name per axis — or
        ``(None, None)`` if the store carries no calibration (napari then
        falls back to its uncalibrated pixel default).
    """
    from .ome_zarr import _base_scale, _read_zarr_calibration

    axes = _store_axes(path, ndim)
    pixel_size = _read_zarr_calibration(str(path), axes)
    if not pixel_size:
        return None, None
    scale = _base_scale(axes, pixel_size)
    units = ["micrometer"] * len(scale)
    return scale, units


def _label_hint(path: Union[str, Path]) -> dict[str, Any]:
    """Read the known-object-count hint from a label group's zarr attrs.

    ``write_labels(..., n_objects=...)`` persists this when the labels were
    renumbered to a contiguous ``1..N`` range (``sequential_labels=True``
    during the merge) — the exact id set is then ``range(1, n_objects +
    1)`` by construction, with no scan needed. Passed through as a Labels
    layer's ``metadata`` so a downstream consumer (e.g.
    napari-chunked-regionprops, https://github.com/imcf/napari-chunked-regionprops)
    can use it instead of re-deriving the id set from the array itself.

    Parameters
    ----------
    path : str or Path
        Label group path (e.g. ``f"{image}/labels/{name}"``).

    Returns
    -------
    dict
        ``{"patchworks_labels": path}`` (the label group, where its object
        table lives), plus ``{"n_objects": int, "sequential_labels": True}``
        if the group has the hint; ``{}`` if the group cannot be opened —
        safe to splat straight into ``metadata=``/merge into a bigger dict
        either way.
    """
    try:
        attrs = open_group_any(path).attrs
    except Exception:
        return {}
    # Where the labels live, so a plugin can find their object table
    # (napari-chunked-regionprops loads it instead of re-measuring).
    hint: dict[str, Any] = {"patchworks_labels": str(path).rstrip("/")}
    if "n_objects" in attrs:
        hint["n_objects"] = attrs["n_objects"]
        hint["sequential_labels"] = attrs.get("sequential_labels", False)
    return hint


def _inner_label_names(store: Union[str, Path]) -> list[str]:
    """List label images registered under an OME-ZARR's ``labels/`` group.

    Parameters
    ----------
    store : str or Path
        OME-ZARR store path.

    Returns
    -------
    list of str
        Registered label-image names (empty if there are none).
    """
    try:
        grp = open_group_any(f"{store}/labels")
    except Exception:
        return []
    return list(read_ngff_attr(grp.attrs, "labels", []) or [])


def _resolve_labels(
    source: Union[da.Array, str, Path], component: str
) -> Union[da.Array, list[da.Array]]:
    """Resolve a label *source* into integer data for a Labels layer.

    Parameters
    ----------
    source : da.Array, str or Path
        Label store (plain or multi-scale) or an in-memory array.
    component : str
        Array name inside a plain-zarr label store.

    Returns
    -------
    da.Array or list of da.Array
        Integer (``int32``) labels; a list for a multi-scale store.
    """
    if _is_zarr(source):
        if _has_multiscales(source):
            levels = _multiscale_levels(source, None)
            return [lvl.astype("int32") for lvl in levels]
        arr = from_zarr_any(source, component=component)
    elif isinstance(source, (str, Path)):
        arr = from_zarr_any(source)
    else:
        arr = da.asarray(source)
    return arr.astype("int32")


class _ChannelView:
    """One channel of a zarr array, read lazily, without dask.

    napari reads a zarr pyramid as it is, but splitting channels (its
    ``channel_axis``) wraps each array in dask, and slicing a dask array of
    a million chunks -- a whole-slide image in 1024-pixel chunks -- cost
    ~1.3 s per view and channel against 0.01 s straight from zarr.
    """

    def __init__(self, array: Any, axis: int, index: int):
        self.array, self.axis, self.index = array, axis, index
        shape = tuple(array.shape)
        self.shape = shape[:axis] + shape[axis + 1 :]
        self.dtype = array.dtype
        self.ndim = len(self.shape)
        self.size = int(np.prod(self.shape))

    def __getitem__(self, key: Any) -> np.ndarray:
        key = key if isinstance(key, tuple) else (key,)
        if any(k is Ellipsis for k in key):
            at = next(i for i, k in enumerate(key) if k is Ellipsis)
            fill = (slice(None),) * (self.ndim - len(key) + 1)
            key = key[:at] + fill + key[at + 1 :]
        key = key + (slice(None),) * (self.ndim - len(key))
        full = key[: self.axis] + (self.index,) + key[self.axis :]
        return np.asarray(self.array[full])

    def __array__(self, dtype=None, copy=None) -> np.ndarray:
        out = self[...]
        return out if dtype is None else out.astype(dtype)

    def __len__(self) -> int:
        return self.shape[0]


def _zarr_levels(path: Union[str, Path]) -> list:
    """Every pyramid level as the zarr array it is (no dask)."""
    root = open_group_any(path)
    datasets = read_ngff_attr(root.attrs, "multiscales")[0]["datasets"]
    return [root[d["path"]] for d in datasets]


def _label_levels(path: Union[str, Path]) -> list:
    """A label pyramid for napari: the zarr arrays themselves when they hold
    integers, else cast to int32 (through dask)."""
    levels = _zarr_levels(path)
    if all(np.dtype(lvl.dtype).kind in "iu" for lvl in levels):
        return levels
    return [lvl.astype("int32") for lvl in _multiscale_levels(path, None)]


# Each channel's colour when all are shown together.
CHANNEL_COLORMAPS = ("green", "magenta", "cyan", "yellow", "gray", "red")


def image_layers(
    image: Any,
    channel: int | None,
    image_name: str = "image",
    extra: Mapping[str, Any] | None = None,
) -> list[tuple[Any, Any, dict]]:
    """The image layers for napari: ``(data, name, add_image kwargs)``.

    An OME-Zarr pyramid is handed over as zarr arrays, each channel its own
    layer -- shown together, in its own colour, switched on and off at once
    (on one layer with a channel slider, every change of channel read a new
    plane and cached none).
    """
    extra = dict(extra or {})
    if (
        not (_is_zarr(image) and _has_multiscales(image))
        or "channel_axis" in extra
    ):
        img = _resolve_image(image, channel)
        ndim = img[0].ndim if isinstance(img, list) else img.ndim
        scale, units = (
            _pyramid_calibration(image, ndim)
            if _is_zarr(image)
            else (None, None)
        )
        kwargs = {"multiscale": isinstance(img, list), "scale": scale}
        return [(img, image_name, {**kwargs, "units": units, **extra})]
    levels = _zarr_levels(image)
    ndim = levels[0].ndim
    axes = _store_axes(image, ndim)
    scale, units = _pyramid_calibration(image, ndim)
    if "c" not in axes:
        kwargs = {"multiscale": True, "scale": scale, "units": units}
        return [(levels, image_name, {**kwargs, **extra})]
    c = axes.index("c")
    if scale is not None:
        scale = [v for i, v in enumerate(scale) if i != c]
        units = [v for i, v in enumerate(units) if i != c]
    n = levels[0].shape[c]
    together = channel is None and n > 1
    out = []
    for i in range(n) if channel is None else [channel]:
        kwargs: dict[str, Any] = {
            "multiscale": True,
            "scale": scale,
            "units": units,
        }
        if together:
            kwargs["colormap"] = CHANNEL_COLORMAPS[i % len(CHANNEL_COLORMAPS)]
            kwargs["blending"] = "additive"
        out.append(
            (
                [_ChannelView(lvl, c, i) for lvl in levels],
                f"{image_name} [{i}]" if together else image_name,
                {**kwargs, **extra},
            )
        )
    return out


def view_in_napari(
    image: Union[da.Array, str, Path],
    labels: Union[da.Array, str, Path, None] = None,
    *,
    channel: int | None = None,
    labels_component: str = "labels",
    image_name: str = "image",
    labels_name: str = "labels",
    glasbey: bool = True,
    show: bool = True,
    **add_image_kwargs: Any,
):
    """Open *image* in napari and overlay *labels* as a Labels layer.

    Parameters
    ----------
    image : da.Array, str or Path
        OME-ZARR store (multi-scale aware), any bioio-readable file, or an
        in-memory array.
    labels : da.Array, str, Path or None
        Label array to overlay. A plain ``.zarr`` store written by
        ``tile_process`` is read from its ``labels_component``; an OME-ZARR
        pyramid is shown multi-scale. ``None`` (default) **auto-loads** every
        label image stored inside the OME-ZARR under ``labels/<name>/`` — the
        place ``tile_process`` writes them by default — each as its own Labels
        layer. (Falls back to image-only if there are none.)
    channel : int or None, optional
        Channel to display from the image. ``None`` (default) shows every
        channel — the segmentation channel doesn't need to match what you
        view. Pass an int to show just that one.
    labels_component : str, optional
        Array name inside a plain-zarr label store (default ``"labels"``,
        matching ``tile_process``'s ``output_component``).
    image_name, labels_name : str, optional
        Layer names shown in napari.
    glasbey : bool, optional
        Colour the labels with a glasbey palette (many distinct, high-contrast
        colours, tuned to read on the dark canvas) instead of napari's default.
        Default ``True``. Needs the ``glasbey`` package (ships with
        ``patchworks[napari]``).
    show : bool, optional
        Start the napari event loop (blocking). Set ``False`` in scripts/tests
        that manage the loop themselves.
    **add_image_kwargs
        Extra keyword arguments forwarded to ``viewer.add_image``
        (e.g. ``colormap``, ``contrast_limits``).

    Returns
    -------
    napari.Viewer
        The viewer instance (useful when ``show=False``).

    Notes
    -----
    In 3-D view (the cube icon, or ``viewer.dims.ndisplay = 3``), napari
    always shows the **coarsest** pyramid level — there is no automatic
    zoom-based switching in 3-D, only in 2-D. To pin a specific resolution
    (needs ``napari>=0.7.1``)::

        viewer.layers["labels"].locked_data_level = 0   # full resolution
        viewer.layers["labels"].locked_data_level = None  # back to coarsest

    A widget for this is also in the layer controls panel in napari>=0.7.1.

    Examples
    --------
    >>> view_in_napari("scan.zarr")  # auto-loads scan.zarr/labels/*  # doctest: +SKIP
    """
    napari = _require_napari()

    viewer = napari.Viewer()
    for data, name, kwargs in image_layers(
        image, channel, image_name, add_image_kwargs
    ):
        viewer.add_image(data, name=name, **kwargs)

    label_kwargs: dict[str, Any] = {}
    if glasbey:
        try:
            import glasbey as _glasbey
            import numpy as np
            from napari.utils.colormaps import CyclicLabelColormap

            # Glasbey colours, lightened for the dark canvas, cycled per label.
            palette = _glasbey.create_palette(256, lightness_bounds=(40, 100))
            colors = np.array(
                [
                    [
                        int(h[1:3], 16) / 255,
                        int(h[3:5], 16) / 255,
                        int(h[5:7], 16) / 255,
                        1.0,
                    ]
                    for h in palette
                ]
            )
            label_kwargs["colormap"] = CyclicLabelColormap(colors=colors)
        except ImportError:
            logger.warning(
                "glasbey not installed; using napari's default label colours "
                "(pip install glasbey, or it ships with patchworks[napari])."
            )

    if labels is not None:
        lab = _resolve_labels(labels, labels_component)
        lab_ndim = lab[0].ndim if isinstance(lab, list) else lab.ndim
        lab_scale, lab_units = (
            _pyramid_calibration(labels, lab_ndim)
            if _is_zarr(labels)
            else (None, None)
        )
        metadata = _label_hint(labels) if _is_zarr(labels) else {}
        viewer.add_labels(
            lab,
            name=labels_name,
            multiscale=isinstance(lab, list),
            scale=lab_scale,
            translate=(
                _pyramid_translate(labels, lab_ndim)
                if _is_zarr(labels)
                else None
            ),
            units=lab_units,
            metadata=metadata,
            **label_kwargs,
        )
    elif _is_zarr(image):
        # No labels given: every label image in the store, each as a multiscale
        # layer (also for one level, for 3-D resolution switching).
        names = _inner_label_names(image)
        if not names:
            logger.warning(
                "%s has no label images under labels/, so nothing was "
                "overlaid. Pass labels=<path> explicitly if they live "
                "somewhere else.",
                image,
            )
        else:
            logger.info(
                "auto-loading %d label image(s) from %s/labels: %s",
                len(names),
                image,
                ", ".join(names),
            )
        loaded = 0
        for name in names:
            store = f"{image}/labels/{name}"
            # One bad label image must not hide the others.
            try:
                lab = _label_levels(store)
                lab_scale, lab_units = _pyramid_calibration(store, lab[0].ndim)
                viewer.add_labels(
                    lab,
                    name=name,
                    multiscale=True,
                    scale=lab_scale,
                    translate=_pyramid_translate(store, lab[0].ndim),
                    units=lab_units,
                    metadata=_label_hint(store),
                    **label_kwargs,
                )
            except Exception:
                logger.exception(
                    "could not add labels/%s as a layer; skipping it and "
                    "continuing with the rest.",
                    name,
                )
                continue
            loaded += 1
            logger.info("auto-loaded labels/%s from %s", name, image)
        if loaded:
            _show_relations(viewer, image)
        if names and not loaded:
            logger.error(
                "found %d label image(s) in %s/labels but none could be "
                "added -- see the errors above.",
                len(names),
                image,
            )

    if show:
        napari.run()
    return viewer


def _show_relations(viewer: Any, store: Any) -> None:
    """Hovering an object shows its parent and children in the status bar,
    from the store's object tables (when it has them)."""
    try:
        from .._review import Review

        rv = Review(str(store))
    except Exception:
        logger.info("no object tables in %s: no relations on hover", store)
        return
    for name in rv.names:
        if name in viewer.layers:
            try:
                viewer.layers[name].features = rv.hover_features(name)
            except Exception:  # pragma: no cover - depends on napari
                logger.debug(
                    "could not set features on %s", name, exc_info=True
                )
