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
import time
from contextlib import contextmanager
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


# glasbey.create_palette(256, lightness_bounds=(40, 100)): many distinct
# colours that read on napari's dark canvas. Stored, not computed: the
# glasbey package took ~15 s to import and ~3 s per palette, every opening.
GLASBEY_256 = (
    "db202d 009a00 8259ff aefb00 00b6ca ff86e7 f79a00 8a7569 0cffc2 babeff "
    "9eba71 c210d7 dbaeaa ba5900 ffef92 657d9e 358a71 f3149a 9eefff 04d220 "
    "a682b2 8a8210 ff7971 869ea2 55a6ff ffd2ff b25971 00c28a d7c600 bee7c2 "
    "ba9259 b686ff f35100 cad2df 8aeb82 7d9269 d782a2 7daa00 2875ff 86cabe "
    "6dc6ff cea6d7 dbb675 db61ff 008a9a 969ac6 ffdbc6 9e5dce d259b2 c27155 "
    "ba9e10 ffef00 ae8a92 ffaace babeaa ffa275 718282 ff3969 3da25d 5da696 "
    "7582d7 0092ca 00ebe3 9ece00 d7ff8a 9a713d fb39ef e37d31 719eb6 82798e "
    "d2fff7 ff51a6 efffdb c6db8a df0061 c24d2d ffca51 86d29e ce08aa aebac6 "
    "b27d00 929eff 65c261 e392ff 698635 8271ae 4dff4d 9a7182 a69e86 dbc2ca "
    "beaebe df9286 00b228 aa5992 db6171 00beae 8effeb 1cdbf7 ffaeff 8e9e45 "
    "b6dbff 00e38a 92b69a dfd7ff ff8ea6 9affbe ce86c6 00baef b60cff 009e96 "
    "eb5149 7dc2d2 dbdbb2 00dfb6 aaba35 b28675 c6458a 00a67d b6dbd7 a2695d "
    "dfae39 cee300 c6a6fb 82aadf ba7ddb ffebfb d78e10 baaa5d d7a27d 9292a2 "
    "a269ff e3beff ce9eb6 7d7d45 b67196 fbbe9e d7efff ba757d ba59ca c64169 "
    "659200 618269 7192c2 d2ffba ffbec2 ffd79a ff713d 208e55 9a8661 9a82ce "
    "8e9e92 0cff8a 619296 a2d265 db3100 009eba ae9aba e320f7 71b279 ff658a "
    "ba793d ffb661 92e7c6 a6beba cac25d ef20c2 a28e3d aa653d db6196 e79a51 "
    "dbebdf 619a75 aece9e ff86c2 ebdf59 be55ff d76531 b2aed7 4d7dca 31e75d "
    "49cece 69aeb2 e76ddf 9aff6d 598aff 9e41ff eb8661 dbc2a6 e7b6db 9aa26d "
    "ffa69e 656dff 797d65 9282ff 8286aa 71c228 96beff 8e6d96 6dbe9e bebe8a "
    "ba45ae e79ace ffff75 8ee3df 71aa4d aacedf ba96d7 65e318 c29a92 e375ba "
    "4586a6 aae7a2 8665ce 00ba61 ba8aaa ebebff baffdf 96bedf 71ebaa 9ea200 "
    "bac6e7 a6d2ba e7457d c2ae8e ca8a65 5dffff ff8610 798e9e e79eaa aeff9e "
    "e7d279 a2a6ba dfff4d d70082 6d9286 ff5dd2 0c92f3 ffefba ffb200 b65d51 "
    "8ad7f3 a69ae7 ca7500 96710c ffd200 ffcae3 9a869a 419239 c6e361 df3d55 "
    "b66db2 e77d8a e36d59 d2c2df 4da2d2 8eaebe"
).split()


@contextmanager
def _timed(what: str):
    """Log how long *what* took: opening a large store, the slow part shows."""
    start = time.perf_counter()
    yield
    logger.info("%s: %.1f s", what, time.perf_counter() - start)


def _label_levels(path: Union[str, Path]) -> list:
    """A label pyramid for napari, as int32 dask arrays."""
    return [lvl.astype("int32") for lvl in _multiscale_levels(path, None)]


def image_layers(
    image: Any,
    channel: int | None,
    image_name: str = "image",
    extra: Mapping[str, Any] | None = None,
    split_channels: bool = False,
) -> list[tuple[Any, Any, dict]]:
    """The image layers for napari: ``(data, name, add_image kwargs)``.

    Lazy dask arrays: napari cuts the region on screen out of every axis
    not displayed before it picks the plane, which is free on dask but
    reads every channel and plane of it from a zarr array (8 minutes for a
    whole slide). All channels are one layer with a channel slider;
    *split_channels* makes each its own layer, shown together -- every zoom
    and pan then reads every channel.
    """
    extra = dict(extra or {})
    img = _resolve_image(image, channel)
    ndim = img[0].ndim if isinstance(img, list) else img.ndim
    scale, units = (
        _pyramid_calibration(image, ndim) if _is_zarr(image) else (None, None)
    )
    name: Any = image_name
    if split_channels and channel is None and _is_zarr(image):
        axes = _store_axes(image, ndim)
        shape = img[0].shape if isinstance(img, list) else img.shape
        if "c" in axes and shape[axes.index("c")] > 1:
            c = axes.index("c")
            extra.setdefault("channel_axis", c)
            name = [f"{image_name} [{i}]" for i in range(shape[c])]
            if scale is not None:
                scale = [v for i, v in enumerate(scale) if i != c]
                units = [v for i, v in enumerate(units) if i != c]
    kwargs = {"multiscale": isinstance(img, list), "scale": scale}
    return [(img, name, {**kwargs, "units": units, **extra})]


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
    split_channels: bool = False,
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
        Default ``True``.
    show : bool, optional
        Start the napari event loop (blocking). Set ``False`` in scripts/tests
        that manage the loop themselves.
    split_channels : bool, optional
        Each channel its own layer, shown together in its own colour,
        instead of one layer with a channel slider. Every zoom and pan then
        reads every channel.
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

    with _timed("napari window"):
        viewer = napari.Viewer()
    with _timed("image: opening its arrays"):
        layers = image_layers(
            image, channel, image_name, add_image_kwargs, split_channels
        )
    with _timed("image: napari adding the layer"):
        for data, name, kwargs in layers:
            viewer.add_image(data, name=name, **kwargs)

    label_kwargs: dict[str, Any] = {}
    if glasbey:
        from napari.utils.colormaps import CyclicLabelColormap

        # Glasbey colours, cycled per label.
        colors = np.array(
            [
                [int(h[i : i + 2], 16) / 255 for i in (0, 2, 4)] + [1.0]
                for h in GLASBEY_256
            ]
        )
        label_kwargs["colormap"] = CyclicLabelColormap(colors=colors)

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
                with _timed(f"labels/{name}"):
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
        if loaded and show:
            # Minutes for a whole slide's tables: read after the window
            # is up, not before.
            _show_relations_later(viewer, image)
        elif loaded:
            with _timed("object tables, for hovering"):
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


def _hover_tables(store: Any) -> dict:
    """Each label image's hover features (its parents and children), from
    the store's object tables; {} without tables."""
    try:
        from .._review import Review

        rv = Review(str(store))
    except Exception:
        logger.info("no object tables in %s: no relations on hover", store)
        return {}
    out = {}
    for name in rv.names:
        try:
            out[name] = rv.hover_features(name)
        except Exception:  # pragma: no cover - a table it cannot read
            logger.debug("no hover features for %s", name, exc_info=True)
    return out


def _apply_hover_tables(viewer: Any, tables: dict) -> None:
    for name, features in tables.items():
        if name in viewer.layers:
            try:
                viewer.layers[name].features = features
            except Exception:  # pragma: no cover - depends on napari
                logger.debug(
                    "could not set features on %s", name, exc_info=True
                )


def _show_relations(viewer: Any, store: Any) -> None:
    """Hovering an object shows its parent and children in the status bar,
    from the store's object tables (when it has them)."""
    _apply_hover_tables(viewer, _hover_tables(store))


def _show_relations_later(viewer: Any, store: Any) -> None:
    """:func:`_show_relations`, the tables read in a background thread and
    handed to the layers once read (on napari's own thread)."""
    from napari.qt.threading import thread_worker

    start = time.perf_counter()
    logger.info("reading the object tables for hovering, in the background")

    def done(tables):
        _apply_hover_tables(viewer, tables)
        logger.info(
            "object tables, for hovering: %.1f s", time.perf_counter() - start
        )

    worker = thread_worker(_hover_tables)(store)
    worker.returned.connect(done)
    worker.start()
