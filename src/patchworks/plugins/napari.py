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
>>> from patchworks import tile_process
>>> from patchworks.plugins.napari import view_in_napari
>>>
>>> # labels are written into scan.zarr/labels/ by default …
>>> tile_process("scan.zarr", fn)
>>> # … so the viewer finds and overlays them with no labels= argument:
>>> view_in_napari("scan.zarr")
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any, Union

import dask.array as da
import zarr

from .._io import load_ome_zarr

logger = logging.getLogger(__name__)


def _require_napari():
    """Import and return napari, or raise an actionable ImportError.

    Returns
    -------
    module
        The imported ``napari`` module.
    """
    try:
        import napari
    except ImportError as exc:
        raise ImportError(
            "napari is not installed. Install it with:\n"
            "    pip install 'patchworks[napari]'"
        ) from exc
    return napari


def _is_zarr(src: Any) -> bool:
    """Whether *src* is a path ending in ``.zarr``.

    Parameters
    ----------
    src : Any
        Candidate source.

    Returns
    -------
    bool
        True for a str/Path ending in ``.zarr``.
    """
    return isinstance(src, (str, Path)) and str(src).endswith(".zarr")


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
    root = zarr.open_group(str(path), mode="r")
    return "multiscales" in root.attrs


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
    root = zarr.open_group(str(path), mode="r")
    datasets = root.attrs["multiscales"][0]["datasets"]
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

        arr, _ = _open_bioio(str(source), 0)
        return arr
    return source


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
        grp = zarr.open_group(f"{store}/labels", mode="r")
    except Exception:
        return []
    return list(grp.attrs.get("labels", []))


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
        arr = da.from_zarr(str(source), component=component)
    elif isinstance(source, (str, Path)):
        arr = da.from_zarr(str(source))
    else:
        arr = da.asarray(source)
    return arr.astype("int32")


def _label_colormap(name: str | None):
    """Build a cyclic napari label colormap from a colorcet glasbey palette.

    Parameters
    ----------
    name : str or None
        A ``colorcet`` palette attribute, e.g. ``"glasbey_dark"`` (glasbey on a
        dark background). ``None`` falls back to napari's default label colours.

    Returns
    -------
    napari.utils.colormaps.CyclicLabelColormap or None
        The colormap to pass to ``add_labels``, or ``None`` for the default.
    """
    if not name:
        return None
    try:
        import colorcet
    except ImportError:
        logger.warning(
            "colorcet not installed; using napari's default label colours "
            "(pip install colorcet, or it ships with patchworks[napari])."
        )
        return None

    palette = getattr(colorcet, name, None)
    if palette is None:
        logger.warning("colorcet has no palette %r; using default colours.", name)
        return None

    import numpy as np
    from napari.utils.colormaps import CyclicLabelColormap

    def _hex_to_rgba(h: str):
        h = h.lstrip("#")
        return (int(h[0:2], 16) / 255, int(h[2:4], 16) / 255,
                int(h[4:6], 16) / 255, 1.0)

    colors = np.array([_hex_to_rgba(c) for c in palette], dtype=float)
    return CyclicLabelColormap(colors=colors)


def view_in_napari(
    image: Union[da.Array, str, Path],
    labels: Union[da.Array, str, Path, None] = None,
    *,
    channel: int | None = 0,
    labels_component: str = "labels",
    image_name: str = "image",
    labels_name: str = "labels",
    label_colormap: str | None = "glasbey_dark",
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
        Channel to display from the image (``None`` keeps all channels).
    labels_component : str, optional
        Array name inside a plain-zarr label store (default ``"labels"``,
        matching ``tile_process``'s ``output_component``).
    image_name, labels_name : str, optional
        Layer names shown in napari.
    label_colormap : str or None, optional
        ``colorcet`` palette for the label LUT; default ``"glasbey_dark"``
        (glasbey on a dark background — many distinct, high-contrast colours).
        Any colorcet name works (e.g. ``"glasbey_light"``); ``None`` uses
        napari's default label colours. Needs ``colorcet`` (ships with
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

    Examples
    --------
    >>> view_in_napari("scan.zarr")  # auto-loads scan.zarr/labels/*  # doctest: +SKIP
    """
    napari = _require_napari()

    img = _resolve_image(image, channel)
    viewer = napari.Viewer()
    viewer.add_image(
        img,
        name=image_name,
        multiscale=isinstance(img, list),
        **add_image_kwargs,
    )

    cmap = _label_colormap(label_colormap)
    label_kwargs = {"colormap": cmap} if cmap is not None else {}

    if labels is not None:
        lab = _resolve_labels(labels, labels_component)
        viewer.add_labels(lab, name=labels_name, **label_kwargs)
    elif _is_zarr(image):
        # No labels given → auto-overlay every label image stored inside the
        # OME-ZARR under labels/<name>/ (the default place tile_process writes
        # them), each as its own multi-scale Labels layer.
        for name in _inner_label_names(image):
            levels = _multiscale_levels(f"{image}/labels/{name}", None)
            lab = [lvl.astype("int32") for lvl in levels]
            viewer.add_labels(
                lab if len(lab) > 1 else lab[0], name=name, **label_kwargs
            )
            logger.info("auto-loaded labels/%s from %s", name, image)

    if show:
        napari.run()
    return viewer
