"""Optional denoising before segmentation, with CAREamics (Noise2Void).

A noisy membrane is where segmentations break: a wall lost in the noise
merges two cells, noise inside a cell splits it. Noise2Void learns to remove
the noise from the image itself -- no clean ground truth, no annotation --
so a model trained once on a few crops of the store denoises every tile
before the segmentation sees it, whatever the method (Cellpose, PlantSeg,
the nuclei-seeded watershed, a custom function).

Two steps:

1. Train, once, on a GPU node::

       patchworks denoise-train image.zarr --channel 0 --out n2v_membrane.ckpt

   (:func:`train_n2v`; with ``nuclei_channel``, train a second model on
   that channel.)

2. Denoise before segmenting -- in the workflow config::

       denoise:
         model: "/path/to/n2v_membrane.ckpt"
         nuclei_model: "/path/to/n2v_nuclei.ckpt"   # optional

   or from the API, :func:`denoise_fn` wraps any per-tile function.

Needs ``careamics>=0.3`` (``pip install patchworks[careamics]``), which
brings PyTorch. Any CAREamics model works: a checkpoint (``.ckpt``) or a
BioImage.IO archive (``.zip``), Noise2Void, CARE or Noise2Noise.
"""

from __future__ import annotations

import logging
import math
import tempfile
from functools import partial
from pathlib import Path
from typing import Any, Callable, Sequence

import numpy as np

from .._gpu import free_gpu_caches, retry_on_oom
from .watershed import split_channels

logger = logging.getLogger(__name__)

#: Spatial tile CAREamics predicts at a time, per tile dimensionality: a
#: whole 3-D patchworks tile through a U-Net at once does not fit in VRAM.
DEFAULT_TILE = {3: (16, 256, 256), 2: (512, 512)}


def default_overlap(tile: Sequence[int]) -> tuple[int, ...]:
    """A quarter of each tile side, even, at least 2.

    >>> default_overlap((16, 256, 256))
    (4, 64, 64)
    """
    return tuple(max(2, (int(t) // 4) // 2 * 2) for t in tile)


# Per process: one CAREamist per model file, loaded on first use.
_models: dict[str, Any] = {}


def _require_careamics():
    try:
        import careamics  # noqa: F401
    except ImportError as exc:
        raise ImportError(
            "CAREamics is not installed. Install it with:\n"
            "    pip install patchworks[careamics]\n"
            "or use the workflow's careamics environment:\n"
            "    pixi install -e careamics"
        ) from exc


def _load(path: str) -> Any:
    """The CAREamist for *path*, cached per process."""
    if path not in _models:
        from careamics import CAREamist

        p = Path(path)
        if not p.exists():
            raise FileNotFoundError(f"denoising model not found: {path}")
        # Lightning logs and checkpoints go to work_dir: keep them out of
        # the workflow directory.
        work_dir = tempfile.mkdtemp(prefix="pw_careamics_")
        source = (
            {"bmz_path": p} if p.suffix == ".zip" else {"checkpoint_path": p}
        )
        _models[path] = CAREamist(
            **source, work_dir=work_dir, enable_progress_bar=False
        )
    return _models[path]


def _release() -> None:
    """Forget the loaded models (before an OOM backoff)."""
    _models.clear()
    free_gpu_caches()


def denoise_image(
    img: np.ndarray,
    model: str,
    *,
    tile_size: Sequence[int] | None = None,
    tile_overlap: Sequence[int] | None = None,
    batch_size: int = 1,
) -> np.ndarray:
    """Denoise one single-channel ``(z, y, x)`` or ``(y, x)`` image.

    Returns a float32 array of the same shape.
    """
    img = np.asarray(img, dtype="float32")
    ndim = img.ndim
    axes = "ZYX" if ndim == 3 else "YX"
    tile = tuple(tile_size) if tile_size else DEFAULT_TILE.get(ndim)
    overlap = (
        tuple(tile_overlap)
        if tile_overlap
        else (default_overlap(tile) if tile else None)
    )
    if tile is not None and all(s <= t for s, t in zip(img.shape, tile)):
        tile = overlap = None  # fits in one piece

    def _predict():
        preds, _ = _load(str(model)).predict(
            pred_data=img,
            data_type="array",
            axes=axes,
            tile_size=tile,
            tile_overlap=overlap if tile else None,
            batch_size=batch_size,
        )
        return preds[0]

    out = retry_on_oom(_predict, enabled=True, on_release=_release)
    out = np.asarray(out, dtype="float32")
    if out.size != img.size:
        raise ValueError(
            f"denoising returned {out.shape} for an image of {img.shape}; "
            "is the model 2-D while the tiles are 3-D (or the reverse)?"
        )
    return out.reshape(img.shape)


def denoise_tile(tile: np.ndarray, cfg: dict[str, Any]) -> np.ndarray:
    """Denoise a tile: ``[membrane, nuclei]`` each with its own model.

    The first (segmented) channel uses ``model``; a nuclear channel, when
    the tile carries one, uses ``nuclei_model`` or stays as it is.
    """
    kw = dict(
        tile_size=cfg.get("tile_size"),
        tile_overlap=cfg.get("tile_overlap"),
        batch_size=cfg.get("batch_size", 1),
    )
    first, nuclei = split_channels(tile)
    first = denoise_image(first, cfg["model"], **kw)
    if nuclei is None:
        return first
    if cfg.get("nuclei_model"):
        nuclei = denoise_image(nuclei, cfg["nuclei_model"], **kw)
    return np.stack([first, np.asarray(nuclei, "float32")])


def denoise_fn(
    fn: Callable[[np.ndarray], np.ndarray],
    model: str,
    *,
    nuclei_model: str | None = None,
    tile_size: Sequence[int] | None = None,
    tile_overlap: Sequence[int] | None = None,
    batch_size: int = 1,
) -> Callable[[np.ndarray], np.ndarray]:
    """Wrap *fn* so every tile is denoised before *fn* segments it.

    Parameters
    ----------
    fn :
        Any per-tile segmentation function.
    model :
        CAREamics model for the segmented channel: a checkpoint (``.ckpt``,
        e.g. from :func:`train_n2v`) or a BioImage.IO ``.zip``.
    nuclei_model :
        Model for the nuclear channel (``nuclei_channel``), which keeps its
        raw intensities without one.
    tile_size, tile_overlap :
        CAREamics' own tiling inside a patchworks tile (VRAM). Defaults:
        :data:`DEFAULT_TILE`, overlapping by :func:`default_overlap`. Each tile side must
        be divisible by 2**depth of the U-Net (4 for the default N2V).
    batch_size :
        CAREamics tiles per forward pass.

    Returns
    -------
    Callable[[ndarray], ndarray]
        Picklable ``tile -> labels``.
    """
    _require_careamics()
    for path in (model, nuclei_model):
        if path and not Path(path).exists():
            raise FileNotFoundError(f"denoising model not found: {path}")
    cfg = dict(
        model=str(model),
        nuclei_model=str(nuclei_model) if nuclei_model else None,
        tile_size=tuple(tile_size) if tile_size else None,
        tile_overlap=tuple(tile_overlap) if tile_overlap else None,
        batch_size=int(batch_size),
    )
    return partial(_denoised, fn=fn, cfg=cfg)


def _denoised(
    tile: np.ndarray, fn: Callable, cfg: dict[str, Any]
) -> np.ndarray:
    return fn(denoise_tile(tile, cfg))


def denoise(tile: np.ndarray, **kwargs: Any) -> np.ndarray:
    """Just the denoised tile (float32), e.g. to look at it before
    segmenting with it. Takes :func:`denoise_fn`'s keyword arguments."""
    _require_careamics()
    model = kwargs.pop("model")
    cfg = dict(model=str(model), **kwargs)
    return denoise_tile(tile, cfg)


# --------------------------------------------------------------- training


def training_crops(
    store: str | Path,
    *,
    channel: int = 0,
    level: int = 0,
    crop_shape: Sequence[int] = (32, 512, 512),
    n_crops: int = 4,
) -> list[np.ndarray]:
    """The *n_crops* brightest crops of a channel, to train a denoiser on.

    Ranked on the store's coarsest pyramid level, so picking them reads
    almost nothing; empty background (nothing to learn the noise of
    *structure* from) is skipped that way too.
    """
    from .._io import load_ome_zarr
    from .ome_zarr import _multiscale_meta

    full = load_ome_zarr(str(store), channel=channel, level=level)
    meta = _multiscale_meta(str(store)) or {}
    n_levels = len(meta.get("datasets") or [None])
    coarse = np.asarray(
        load_ome_zarr(str(store), channel=channel, level=n_levels - 1)
    ).astype("float32")
    shape = tuple(full.shape)
    crop = tuple(
        min(int(c), s) for c, s in zip(tuple(crop_shape)[-len(shape) :], shape)
    )
    starts = [
        np.unique(np.linspace(0, s - c, max(1, math.ceil(s / c))).astype(int))
        for s, c in zip(shape, crop)
    ]
    scale = [cs / s for cs, s in zip(coarse.shape, shape)]
    scored = []
    for corner in np.stack(np.meshgrid(*starts, indexing="ij"), -1).reshape(
        -1, len(shape)
    ):
        box = tuple(
            slice(int(c * f), max(int(c * f) + 1, int(math.ceil((c + w) * f))))
            for c, w, f in zip(corner, crop, scale)
        )
        scored.append(
            (float(coarse[box].mean()), tuple(int(c) for c in corner))
        )
    scored.sort(reverse=True)
    out = []
    for _, corner in scored[: max(1, int(n_crops))]:
        sl = tuple(slice(c, c + w) for c, w in zip(corner, crop))
        out.append(np.asarray(full[sl], dtype="float32"))
    return out


def train_n2v(
    store: str | Path,
    out: str | Path,
    *,
    channel: int = 0,
    level: int = 0,
    crop_shape: Sequence[int] = (32, 512, 512),
    n_crops: int = 4,
    patch_size: Sequence[int] | None = None,
    batch_size: int = 8,
    epochs: int = 30,
    n2v2: bool = False,
    work_dir: str | Path | None = None,
) -> Path:
    """Train a Noise2Void model on crops of one channel of *store*.

    Parameters
    ----------
    store :
        OME-Zarr image (the workflow's ``<work_dir>/image.zarr``).
    out :
        Where to write the trained checkpoint (``.ckpt``), which
        ``denoise: {model: ...}`` then loads.
    channel, level :
        Channel and pyramid level -- the ones the segmentation reads.
    crop_shape, n_crops :
        What to train on: the *n_crops* brightest crops of that shape
        (:func:`training_crops`). A few hundred million voxels is plenty;
        Noise2Void learns the noise, not the objects.
    patch_size :
        Training patch, ``(z, y, x)`` or ``(y, x)``; each side a multiple of
        8 and at least 8. Default ``(16, 64, 64)`` / ``(64, 64)``.
    batch_size, epochs :
        Training length. 30 epochs takes minutes to an hour on one GPU.
    n2v2 :
        Noise2Void2 (fewer checkerboard artefacts on structured noise).
    work_dir :
        Where CAREamics keeps its logs and intermediate checkpoints.

    Returns
    -------
    Path
        *out*.
    """
    _require_careamics()
    from careamics import CAREamist
    from careamics.config import create_n2v_config

    crops = training_crops(
        store,
        channel=channel,
        level=level,
        crop_shape=crop_shape,
        n_crops=n_crops,
    )
    ndim = crops[0].ndim
    axes = "ZYX" if ndim == 3 else "YX"
    if patch_size is None:
        patch_size = (16, 64, 64) if ndim == 3 else (64, 64)
    patch_size = tuple(
        min(int(p), _floor8(s)) for p, s in zip(patch_size, crops[0].shape)
    )
    logger.info(
        "training Noise2Void on %d crop(s) of %s, patch %s",
        len(crops),
        crops[0].shape,
        patch_size,
    )
    # CAREamics holds back n_val_patches of the training patches for
    # validation (8 by default) and refuses when there are not more than
    # that: keep about a fifth for a small training set.
    n_patches = sum(
        math.prod(s // p for s, p in zip(c.shape, patch_size)) for c in crops
    )
    config = create_n2v_config(
        experiment_name="patchworks_n2v",
        data_type="array",
        axes=axes,
        patch_size=list(patch_size),
        batch_size=int(batch_size),
        num_epochs=int(epochs),
        use_n2v2=bool(n2v2),
        n_val_patches=max(1, min(8, n_patches // 5)),
    )
    work_dir = Path(work_dir or tempfile.mkdtemp(prefix="pw_n2v_"))
    careamist = CAREamist(config, work_dir=work_dir)
    careamist.train(train_data=crops if len(crops) > 1 else crops[0])
    out = Path(out)
    out.parent.mkdir(parents=True, exist_ok=True)
    careamist.trainer.save_checkpoint(out)
    logger.info("Noise2Void model written to %s", out)
    return out


def _floor8(n: int) -> int:
    """Largest multiple of 8 <= n (at least 8)."""
    return max(8, (int(n) // 8) * 8)
