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
    from .ome_zarr import _base_scale, _default_axes, _read_zarr_calibration

    axes = _default_axes(ndim)
    pixel_size = _read_zarr_calibration(str(path), axes)
    if not pixel_size:
        return None, None
    scale = _base_scale(axes, pixel_size)
    units = ["micrometer"] * len(scale)
    return scale, units


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


def view_in_napari(
    image: Union[da.Array, str, Path],
    labels: Union[da.Array, str, Path, None] = None,
    *,
    channel: int | None = 0,
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
        Channel to display from the image (``None`` keeps all channels).
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

    img = _resolve_image(image, channel)
    img_ndim = img[0].ndim if isinstance(img, list) else img.ndim
    img_scale, img_units = (
        _pyramid_calibration(image, img_ndim) if _is_zarr(image) else (None, None)
    )
    viewer = napari.Viewer()
    viewer.add_image(
        img,
        name=image_name,
        multiscale=isinstance(img, list),
        scale=img_scale,
        units=img_units,
        **add_image_kwargs,
    )

    label_kwargs: dict[str, Any] = {}
    if glasbey:
        try:
            import glasbey as _glasbey
            import numpy as np
            from napari.utils.colormaps import CyclicLabelColormap

            # glasbey palette (biased lighter so colours read on the dark
            # canvas), wrapped in a CyclicLabelColormap so each label value
            # cycles through a distinct colour. Passing the raw palette list
            # makes napari map large label IDs past the end -> one flat colour.
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
        viewer.add_labels(
            lab,
            name=labels_name,
            multiscale=isinstance(lab, list),
            scale=lab_scale,
            units=lab_units,
            **label_kwargs,
        )
    elif _is_zarr(image):
        # No labels given → auto-overlay every label image stored inside the
        # OME-ZARR under labels/<name>/ (the default place tile_process writes
        # them), each as its own multi-scale Labels layer. Kept as a list (not
        # unwrapped to a single array) even for one level, so napari always
        # treats it as multiscale — required for 3D resolution switching, see
        # https://napari.org/stable/gallery/add_multiscale_volume.html
        for name in _inner_label_names(image):
            store = f"{image}/labels/{name}"
            levels = _multiscale_levels(store, None)
            lab = [lvl.astype("int32") for lvl in levels]
            lab_scale, lab_units = _pyramid_calibration(store, lab[0].ndim)
            viewer.add_labels(
                lab,
                name=name,
                multiscale=True,
                scale=lab_scale,
                units=lab_units,
                **label_kwargs,
            )
            logger.info("auto-loaded labels/%s from %s", name, image)

    if show:
        napari.run()
    return viewer
