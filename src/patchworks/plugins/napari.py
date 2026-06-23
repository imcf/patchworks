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
    try:
        import napari
    except ImportError as exc:
        raise ImportError(
            "napari is not installed. Install it with:\n"
            "    pip install 'patchworks[napari]'"
        ) from exc
    return napari


def _is_zarr(src: Any) -> bool:
    return isinstance(src, (str, Path)) and str(src).endswith(".zarr")


def _has_multiscales(path: Union[str, Path]) -> bool:
    root = zarr.open_group(str(path), mode="r")
    return "multiscales" in root.attrs


def _multiscale_levels(
    path: Union[str, Path], channel: int | None
) -> list[da.Array]:
    """Return every pyramid level as a lazy dask array (napari multi-scale)."""
    root = zarr.open_group(str(path), mode="r")
    datasets = root.attrs["multiscales"][0]["datasets"]
    return [
        load_ome_zarr(path, channel=channel, level=i)
        for i in range(len(datasets))
    ]


def _resolve_image(
    source: Union[da.Array, str, Path], channel: int | None
) -> Union[da.Array, list[da.Array]]:
    """Resolve *source* into data napari can display (lazily)."""
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
    """Names registered under an OME-ZARR's NGFF ``labels/`` group, if any."""
    try:
        grp = zarr.open_group(f"{store}/labels", mode="r")
    except Exception:
        return []
    return list(grp.attrs.get("labels", []))


def _resolve_labels(
    source: Union[da.Array, str, Path], component: str
) -> Union[da.Array, list[da.Array]]:
    """Resolve a label *source* into integer data for an Labels layer."""
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


def view_in_napari(
    image: Union[da.Array, str, Path],
    labels: Union[da.Array, str, Path, None] = None,
    *,
    channel: int | None = 0,
    labels_component: str = "labels",
    image_name: str = "image",
    labels_name: str = "labels",
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

    if labels is not None:
        lab = _resolve_labels(labels, labels_component)
        viewer.add_labels(lab, name=labels_name)
    elif _is_zarr(image):
        # No labels given → auto-overlay every label image stored inside the
        # OME-ZARR under labels/<name>/ (the default place tile_process writes
        # them), each as its own multi-scale Labels layer.
        for name in _inner_label_names(image):
            levels = _multiscale_levels(f"{image}/labels/{name}", None)
            lab = [lvl.astype("int32") for lvl in levels]
            viewer.add_labels(lab if len(lab) > 1 else lab[0], name=name)
            logger.info("auto-loaded labels/%s from %s", name, image)

    if show:
        napari.run()
    return viewer
