# /// script
# requires-python = ">=3.11"
# dependencies = ["numpy", "h5py", "tifffile"]
# ///
"""Convert a folder of single-plane TIFFs into a BigDataViewer H5/XML pair.

The input is one 2D TIFF per plane, named like::

    A1--W00001--P00082--Z00009--T00000--405nm.tif
    |   |       |       |       |       '-- channel (wavelength)
    |   |       |       |       '---------- timepoint
    |   |       |       '------------------ z plane in the stack
    |   |       '-------------------------- position (one tile)
    |   '---------------------------------- field / well index
    '-------------------------------------- well

Each (position, channel) becomes one BDV view setup -- a tile in one
channel -- so the result opens directly in BigDataViewer and BigStitcher.
Every T found becomes a timepoint; a position/channel missing at some T is
written as a MissingView rather than as a black stack. Well and field are
expected to be constant; if they are not, they become part of the tile
name so no two stacks collide.

Each stack gets a multiresolution pyramid (2x mean downsampling, z only
once the voxels are close to isotropic), which BigDataViewer needs to be
responsive on large data and BigStitcher uses for pairwise shifts.

Voxel size is read from the TIFF tags (ImageJ or resolution tags) when it
is there; pass ``--voxel-size X Y Z`` (in µm) to set or override it -- the
z step is rarely stored in single-plane files.

Tile positions come from the stage coordinates in the TIFF metadata of
each position's first plane: OME-XML, Micro-Manager, key/value text such
as ``XPosition = 1234.5`` in the description tags, or the baseline TIFF
XPosition/YPosition tags. If the tiles land mirrored or rotated, the
stage axes differ from the camera's: try ``--flip-x``, ``--flip-y`` and
``--swap-xy``. ``--dump-metadata`` prints every tag of one file, to see
what is there. Without positions, ``--grid-columns`` / ``--overlap`` /
``--snake`` lay the tiles out on a regular grid instead.

Usage
-----
    uv run scripts/tifs_to_bdv.py /data/scan --dump-metadata
    uv run scripts/tifs_to_bdv.py /data/scan --dry-run
    uv run scripts/tifs_to_bdv.py /data/scan --voxel-size 0.325 0.325 2
    uv run scripts/tifs_to_bdv.py /data/scan -o /out/scan.xml \\
        --voxel-size 0.325 0.325 2 --grid-columns 14 --overlap 10 --snake

Without uv: ``pip install numpy h5py tifffile`` and run it with python.
Only uint8/uint16 images are supported (BDV HDF5 stores uint16).
"""

from __future__ import annotations

import argparse
import re
import sys
import time
import xml.etree.ElementTree as ET
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from pathlib import Path

import h5py
import numpy as np
import tifffile

NAME_RE = re.compile(
    r"^(?P<well>.+?)--W(?P<field>\d+)--P(?P<pos>\d+)--Z(?P<z>\d+)"
    r"--T(?P<t>\d+)--(?P<chan>.+?)\.tiff?$",
    re.IGNORECASE,
)

# Largest chunk edge in x/y; z chunks are scaled by the anisotropy.
CHUNK_XY = 64
# Stop adding pyramid levels once every dimension is at most this.
MIN_LEVEL_SIZE = 64
MAX_LEVELS = 8


@dataclass
class Stack:
    """All planes of one (tile, channel, timepoint).

    Attributes
    ----------
    planes : dict of int to Path
        TIFF file of each z index, as numbered in the file names.
    """

    planes: dict[int, Path] = field(default_factory=dict)  # z -> file


@dataclass
class Setup:
    """One BDV view setup: a tile in one channel.

    Attributes
    ----------
    id : int
        Setup id, also its ``s##`` group in the HDF5 file.
    tile : int
        Index of the tile attribute.
    channel : int
        Index of the channel attribute.
    name : str
        Display name, e.g. ``P00082_405nm``.
    size_xyz : tuple of int
        Full-resolution size in pixels, x, y, z.
    z_min : int
        Lowest z index in the file names; plane 0 of the stack.
    factors : list of tuple of int
        Downsampling factors of each pyramid level, x, y, z.
    chunks : list of tuple of int
        HDF5 chunk shape of each pyramid level, x, y, z.
    """

    id: int
    tile: int
    channel: int
    name: str
    size_xyz: tuple[int, int, int] = (0, 0, 0)
    z_min: int = 0
    factors: list[tuple[int, int, int]] = field(default_factory=list)
    chunks: list[tuple[int, int, int]] = field(default_factory=list)


# --------------------------------------------------------------------------
# Scanning
# --------------------------------------------------------------------------


def scan(folder: Path):
    """Group the TIFFs of a folder by tile, channel and timepoint.

    Parameters
    ----------
    folder : Path
        Folder holding the ``A1--W#--P#--Z#--T#--<channel>.tif`` files.
        Files that do not match the pattern are skipped with a note.

    Returns
    -------
    dict
        Maps ``((well, field, position), channel, t)`` to a `Stack`.
    """
    stacks: dict[tuple, Stack] = defaultdict(Stack)
    skipped = []
    for path in sorted(folder.iterdir()):
        if not path.is_file() or path.suffix.lower() not in (".tif", ".tiff"):
            continue
        m = NAME_RE.match(path.name)
        if not m:
            skipped.append(path.name)
            continue
        tile = (m["well"], int(m["field"]), int(m["pos"]))
        key = (tile, m["chan"], int(m["t"]))
        z = int(m["z"])
        if z in stacks[key].planes:
            sys.exit(f"Duplicate plane: {path.name}")
        stacks[key].planes[z] = path
    if skipped:
        print(
            f"Skipped {len(skipped)} TIFF(s) not matching the pattern, "
            f"e.g. {skipped[0]}"
        )
    if not stacks:
        sys.exit(f"No matching TIFFs in {folder}")
    return stacks


def channel_sort_key(name: str):
    """Sort key putting channels in wavelength order.

    Parameters
    ----------
    name : str
        Channel name from the file name, e.g. ``488nm``.

    Returns
    -------
    tuple
        Names with a number sort by that number, the others after them.
    """
    m = re.search(r"\d+(\.\d+)?", name)
    return (0, float(m.group()), name) if m else (1, 0.0, name)


def tile_names(tiles):
    """Short, unique display name of each tile.

    Parameters
    ----------
    tiles : list of tuple
        ``(well, field, position)`` of every tile.

    Returns
    -------
    dict of tuple to str
        ``P00082``, prefixed with the well and field only where those
        vary, e.g. ``A1-W00001-P00082``.
    """
    wells = {t[0] for t in tiles}
    fields = {t[1] for t in tiles}
    names = {}
    for well, fld, pos in tiles:
        parts = []
        if len(wells) > 1:
            parts.append(well)
        if len(fields) > 1:
            parts.append(f"W{fld:05d}")
        parts.append(f"P{pos:05d}")
        names[(well, fld, pos)] = "-".join(parts)
    return names


# --------------------------------------------------------------------------
# Metadata
# --------------------------------------------------------------------------


def read_plane_info(path: Path):
    """Read the shape, type and pixel size of one TIFF plane.

    Parameters
    ----------
    path : Path
        A single-plane TIFF.

    Returns
    -------
    shape : tuple of int
        Plane shape, (y, x).
    dtype : numpy.dtype
        Pixel type.
    pixel : tuple of float or None
        Pixel size (x, y) in µm, if the resolution tags give one.
    z_step : float or None
        Z spacing in µm from the ImageJ metadata, if present.
    """
    with tifffile.TiffFile(path) as tif:
        page = tif.pages[0]
        shape, dtype = page.shape, page.dtype
        pixel = None
        unit_scale = {1: None, 2: 25400.0, 3: 10000.0}
        res_unit = page.tags.get("ResolutionUnit")
        res_unit = int(res_unit.value) if res_unit else 1
        ij = tif.imagej_metadata or {}
        ij_unit = str(ij.get("unit", "")).lower()
        xres = page.tags.get("XResolution")
        yres = page.tags.get("YResolution")
        if xres and yres:
            rx = xres.value[0] / xres.value[1] if xres.value[1] else 0
            ry = yres.value[0] / yres.value[1] if yres.value[1] else 0
            if rx > 0 and ry > 0:
                scale = unit_scale.get(res_unit)
                if scale is None and ij_unit in ("micron", "um", "µm"):
                    scale = 1.0
                if scale is not None and not (rx == 1 and ry == 1):
                    pixel = (scale / rx, scale / ry)
        z_step = ij.get("spacing")
    if len(shape) != 2:
        sys.exit(f"{path.name}: expected a single 2D plane, got {shape}")
    return shape, dtype, pixel, z_step


NUM = r"(-?\d+(?:\.\d*)?(?:[eE][-+]?\d+)?)"
# "XPositionUm": 12.3, X position = 12.3, <PosX>12.3</PosX>, posX="12.3"...
KV_RE = {
    axis: re.compile(
        rf"(?:\b|_)(?:stage[ _]?)?(?:{axis}[ _-]?pos(?:ition)?|"
        rf"pos(?:ition)?[ _-]?{axis})(?:[ _]?\(?(?:um|µm|micron)\)?)?"
        rf"[\"']?\s*[=:>]\s*[\"']?{NUM}",
        re.IGNORECASE,
    )
    for axis in "xyz"
}
OME_TO_UM = {"µm": 1.0, "um": 1.0, "nm": 1e-3, "mm": 1e3, "cm": 1e4, "m": 1e6}


def _ome_position(xml: str):
    """Stage position from an OME-XML document.

    Parameters
    ----------
    xml : str
        The OME-XML of the file.

    Returns
    -------
    list of float or None
        x, y, z in µm (z may be None) from the first ``Plane`` with a
        position, else the first ``StageLabel``; None if neither exists.
    """
    root = ET.fromstring(xml)
    for el in root.iter():
        tag = el.tag.rsplit("}", 1)[-1]
        if tag == "Plane" and "PositionX" in el.attrib:
            keys = ("PositionX", "PositionY", "PositionZ")
        elif tag == "StageLabel" and "X" in el.attrib:
            keys = ("X", "Y", "Z")
        else:
            continue
        out = []
        for k in keys:
            v = el.attrib.get(k)
            unit = el.attrib.get(f"{k}Unit", "µm")
            out.append(
                None if v is None else float(v) * OME_TO_UM.get(unit, 1.0)
            )
        return out
    return None


def read_stage_position(path: Path):
    """Read the stage position stored in one TIFF.

    Tries, in order: OME-XML, Micro-Manager metadata, key/value text in
    any string tag (``XPosition = 1.5``, ``"XPositionUm": 1.5``,
    ``<PosX>1.5</PosX>``...), and the baseline XPosition/YPosition tags.

    Parameters
    ----------
    path : Path
        A TIFF file.

    Returns
    -------
    position : tuple of float or None
        Stage x, y, z in µm; z may be None. None if nothing was found.
    source : str or None
        Which kind of metadata the position came from.
    """
    with tifffile.TiffFile(path) as tif:
        page = tif.pages[0]
        tags = page.tags

        if tif.ome_metadata:
            try:
                pos = _ome_position(tif.ome_metadata)
            except ET.ParseError:
                pos = None
            if pos and pos[0] is not None and pos[1] is not None:
                return tuple(pos), "OME-XML"

        mm = tags.get("MicroManagerMetadata")
        if mm and isinstance(mm.value, dict):
            v = mm.value
            if "XPositionUm" in v and "YPositionUm" in v:
                z = v.get("ZPositionUm")
                return (
                    float(v["XPositionUm"]),
                    float(v["YPositionUm"]),
                    None if z is None else float(z),
                ), "Micro-Manager"

        # Free text: ImageDescription, ImageJ Info, vendor XML/JSON tags.
        texts = []
        for tag in tags:
            if isinstance(tag.value, str):
                texts.append(tag.value)
            elif isinstance(tag.value, bytes):
                texts.append(tag.value.decode("latin-1", "replace"))
        texts.append(str((tif.imagej_metadata or {}).get("Info", "")))
        for text in texts:
            found = {a: r.search(text) for a, r in KV_RE.items()}
            if found["x"] and found["y"]:
                z = found["z"]
                return (
                    float(found["x"].group(1)),
                    float(found["y"].group(1)),
                    None if z is None else float(z.group(1)),
                ), "key/value text in the TIFF tags (assumed µm)"

        # Baseline TIFF tags, in ResolutionUnit (inch or cm).
        xp, yp = tags.get("XPosition"), tags.get("YPosition")
        if xp and yp:
            res_unit = tags.get("ResolutionUnit")
            scale = {2: 25400.0, 3: 10000.0}.get(
                int(res_unit.value) if res_unit else 2, 25400.0
            )
            x = xp.value[0] / xp.value[1] * scale
            y = yp.value[0] / yp.value[1] * scale
            return (x, y, None), "TIFF XPosition/YPosition tags"
    return None, None


def dump_metadata(path: Path, limit: int = 4000):
    """Print every tag of one TIFF and the position read from it.

    Parameters
    ----------
    path : Path
        A TIFF file.
    limit : int, optional
        Truncate each tag value to this many characters.
    """
    print(f"Metadata of {path}")
    with tifffile.TiffFile(path) as tif:
        for tag in tif.pages[0].tags:
            value = tag.value
            if isinstance(value, bytes):
                value = value.decode("latin-1", "replace")
            text = str(value)
            if len(text) > limit:
                text = text[:limit] + f" ... [{len(text) - limit} more chars]"
            print(f"--- {tag.code} {tag.name}\n{text}")
        if tif.imagej_metadata:
            print(f"--- ImageJ metadata\n{tif.imagej_metadata}")
    pos, source = read_stage_position(path)
    print("===")
    if pos:
        print(f"Stage position found ({source}): {pos}")
    else:
        print("No stage position recognised.")


# --------------------------------------------------------------------------
# Pyramid layout
# --------------------------------------------------------------------------


def plan_pyramid(size_xyz, voxel_xyz):
    """Plan the multiresolution pyramid of one stack.

    Each level halves x and y; z is halved only once its voxel edge is
    within a factor of 2 of the finest axis, so levels tend to isotropy.
    Levels stop once every dimension is at most ``MIN_LEVEL_SIZE``.

    Parameters
    ----------
    size_xyz : tuple of int
        Full-resolution size in pixels, x, y, z.
    voxel_xyz : tuple of float
        Voxel size, x, y, z.

    Returns
    -------
    factors : list of tuple of int
        Downsampling factors of each level, x, y, z.
    chunks : list of tuple of int
        HDF5 chunk shape of each level, x, y, z.
    """
    factors = [(1, 1, 1)]
    while len(factors) < MAX_LEVELS:
        prev = factors[-1]
        dims = [max(s // f, 1) for s, f in zip(size_xyz, prev, strict=True)]
        if max(dims) <= MIN_LEVEL_SIZE:
            break
        phys = [v * f for v, f in zip(voxel_xyz, prev, strict=True)]
        smallest = min(phys)
        # Double a dimension only if it is not already much coarser than
        # the finest one and there is something left to halve.
        new = tuple(
            f * 2 if (p < 2 * smallest and d >= 2) else f
            for f, p, d in zip(prev, phys, dims, strict=True)
        )
        if new == prev:
            break
        factors.append(new)

    chunks = []
    for f in factors:
        dims = [max(s // fi, 1) for s, fi in zip(size_xyz, f, strict=True)]
        phys = [v * fi for v, fi in zip(voxel_xyz, f, strict=True)]
        cz = round(CHUNK_XY * min(phys[0], phys[1]) / phys[2])
        cz = max(1, min(CHUNK_XY, cz))
        chunks.append(
            (min(CHUNK_XY, dims[0]), min(CHUNK_XY, dims[1]), min(cz, dims[2]))
        )
    return factors, chunks


# --------------------------------------------------------------------------
# Writing
# --------------------------------------------------------------------------


def to_uint16(plane: np.ndarray) -> np.ndarray:
    """Convert a plane to uint16, the type BDV HDF5 stores.

    Parameters
    ----------
    plane : numpy.ndarray
        A uint8 or uint16 plane; other types stop the script.

    Returns
    -------
    numpy.ndarray
        The plane as uint16.
    """
    if plane.dtype == np.uint16:
        return plane
    if plane.dtype == np.uint8:
        return plane.astype(np.uint16)
    sys.exit(f"Unsupported pixel type {plane.dtype}; only uint8/uint16.")


def write_level0(ds, stack: Stack, setup: Setup, pool, chunk_z: int):
    """Write the full-resolution level of one stack.

    Planes are read in parallel, one z chunk at a time, so memory holds
    a single chunk rather than the whole stack. A z index without a file
    is written as zeros.

    Parameters
    ----------
    ds : h5py.Dataset
        The level-0 ``cells`` dataset, (z, y, x) int16.
    stack : Stack
        The planes to write.
    setup : Setup
        The setup the stack belongs to.
    pool : concurrent.futures.Executor
        Pool reading the TIFFs.
    chunk_z : int
        Planes per batch, the dataset's z chunk.
    """
    nx, ny, nz = setup.size_xyz
    zs = list(range(nz))
    blank = np.zeros((ny, nx), np.uint16)

    def load(z):
        path = stack.planes.get(z + setup.z_min)
        if path is None:
            return blank
        plane = tifffile.imread(path)
        if plane.shape != (ny, nx):
            sys.exit(f"{path.name}: shape {plane.shape} != {(ny, nx)}")
        return to_uint16(plane)

    for start in range(0, nz, chunk_z):
        batch = zs[start : start + chunk_z]
        planes = list(pool.map(load, batch))
        ds[start : start + len(batch)] = np.stack(planes).view(np.int16)


def downsample(src, dst, rel_zyx):
    """Mean-downsample one pyramid level into the next.

    Works a z slab at a time to bound memory.

    Parameters
    ----------
    src : h5py.Dataset
        The finer level, (z, y, x) int16 holding uint16 data.
    dst : h5py.Dataset
        The coarser level to fill.
    rel_zyx : tuple of int
        Downsampling factor from `src` to `dst`, z, y, x (1 or 2 each).
    """
    rz, ry, rx = rel_zyx
    nz, ny, nx = dst.shape
    slab = max(1, dst.chunks[0])
    for z0 in range(0, nz, slab):
        z1 = min(nz, z0 + slab)
        block = src[z0 * rz : z1 * rz, : ny * ry, : nx * rx]
        block = block.view(np.uint16).astype(np.float32)
        block = block.reshape(z1 - z0, rz, ny, ry, nx, rx).mean(axis=(1, 3, 5))
        dst[z0:z1] = np.round(block).astype(np.uint16).view(np.int16)


def write_setup_meta(h5, setup: Setup):
    """Write the pyramid description of one setup.

    Parameters
    ----------
    h5 : h5py.File
        The output file.
    setup : Setup
        Setup whose ``resolutions`` and ``subdivisions`` to write.
    """
    grp = h5.require_group(f"s{setup.id:02d}")
    for name in ("resolutions", "subdivisions"):
        if name in grp:
            del grp[name]
    grp.create_dataset(
        "resolutions", data=np.array(setup.factors, dtype=np.float64)
    )
    grp.create_dataset(
        "subdivisions", data=np.array(setup.chunks, dtype=np.int32)
    )


def write_stack(h5, stack, setup, tp, pool, compression):
    """Write every pyramid level of one stack at one timepoint.

    Parameters
    ----------
    h5 : h5py.File
        The output file.
    stack : Stack
        The planes to write.
    setup : Setup
        The setup the stack belongs to.
    tp : int
        Timepoint.
    pool : concurrent.futures.Executor
        Pool reading the TIFFs.
    compression : str or None
        HDF5 compression filter, e.g. ``"gzip"``.
    """
    grp = h5.require_group(f"t{tp:05d}/s{setup.id:02d}")
    prev = None
    prev_f = None
    for level, (f, c) in enumerate(
        zip(setup.factors, setup.chunks, strict=True)
    ):
        shape = tuple(
            max(s // fi, 1)
            for s, fi in zip(setup.size_xyz[::-1], f[::-1], strict=True)
        )
        ds = grp.create_dataset(
            f"{level}/cells",
            shape=shape,
            dtype=np.int16,
            chunks=tuple(c[::-1]),
            compression=compression,
        )
        if level == 0:
            write_level0(ds, stack, setup, pool, c[2])
        else:
            rel = tuple(
                a // b for a, b in zip(f[::-1], prev_f[::-1], strict=True)
            )
            downsample(prev, ds, rel)
        prev, prev_f = ds, f


# --------------------------------------------------------------------------
# XML
# --------------------------------------------------------------------------


def sub(parent, tag, text=None, **attrib):
    """Append a child element.

    Parameters
    ----------
    parent : xml.etree.ElementTree.Element
        Element to append to.
    tag : str
        Tag of the new element.
    text : object, optional
        Text content, converted with `str`.
    **attrib
        XML attributes.

    Returns
    -------
    xml.etree.ElementTree.Element
        The new element.
    """
    el = ET.SubElement(parent, tag, attrib)
    if text is not None:
        el.text = str(text)
    return el


def fmt(v: float) -> str:
    """Format a number for the XML.

    Parameters
    ----------
    v : float
        The number.

    Returns
    -------
    str
        Its shortest round-tripping representation.
    """
    return repr(float(v))


def write_xml(
    xml_path,
    h5_path,
    setups,
    tiles,
    channels,
    timepoints,
    missing,
    voxel,
    tile_offsets,
):
    """Write the BDV/BigStitcher XML describing the HDF5 file.

    Parameters
    ----------
    xml_path : Path
        Output XML.
    h5_path : Path
        The HDF5 file, referenced relative to the XML.
    setups : list of Setup
        All view setups.
    tiles : list of str
        Tile names, by tile index.
    channels : list of str
        Channel names, by channel index.
    timepoints : list of int
        All timepoints.
    missing : list of tuple of int
        ``(timepoint, setup id)`` of views without data.
    voxel : tuple of float
        Voxel size in µm, x, y, z.
    tile_offsets : list of tuple of float or None
        Location in µm (x, y, z) of each tile, or None for all at origin.
    """
    root = ET.Element("SpimData", version="0.2")
    sub(root, "BasePath", ".", type="relative")
    seq = sub(root, "SequenceDescription")
    loader = sub(seq, "ImageLoader", format="bdv.hdf5")
    rel = Path(h5_path).resolve().relative_to(Path(xml_path).resolve().parent)
    sub(loader, "hdf5", rel.as_posix(), type="relative")

    vs = sub(seq, "ViewSetups")
    for s in setups:
        el = sub(vs, "ViewSetup")
        sub(el, "id", s.id)
        sub(el, "name", s.name)
        sub(el, "size", " ".join(map(str, s.size_xyz)))
        vox = sub(el, "voxelSize")
        sub(vox, "unit", "µm")
        sub(vox, "size", " ".join(fmt(v) for v in voxel))
        attrs = sub(el, "attributes")
        sub(attrs, "illumination", 0)
        sub(attrs, "channel", s.channel)
        sub(attrs, "tile", s.tile)
        sub(attrs, "angle", 0)

    att = sub(vs, "Attributes", name="illumination")
    ill = sub(att, "Illumination")
    sub(ill, "id", 0)
    sub(ill, "name", 0)
    att = sub(vs, "Attributes", name="channel")
    for i, name in enumerate(channels):
        ch = sub(att, "Channel")
        sub(ch, "id", i)
        sub(ch, "name", name)
    att = sub(vs, "Attributes", name="tile")
    for i, name in enumerate(tiles):
        t = sub(att, "Tile")
        sub(t, "id", i)
        sub(t, "name", name)
        if tile_offsets is not None:
            loc = tile_offsets[i]
            sub(t, "location", " ".join(fmt(v) for v in loc))
    att = sub(vs, "Attributes", name="angle")
    ang = sub(att, "Angle")
    sub(ang, "id", 0)
    sub(ang, "name", 0)

    if timepoints == list(range(timepoints[0], timepoints[-1] + 1)):
        tps = sub(seq, "Timepoints", type="range")
        sub(tps, "first", timepoints[0])
        sub(tps, "last", timepoints[-1])
    else:
        tps = sub(seq, "Timepoints", type="pattern")
        sub(tps, "integerpattern", ", ".join(map(str, timepoints)))
    if missing:
        mv = sub(seq, "MissingViews")
        for tp, sid in missing:
            sub(mv, "MissingView", timepoint=str(tp), setup=str(sid))

    # Calibration is relative to the finest axis, as BigStitcher does it,
    # so world coordinates are in units of the smallest voxel edge.
    smallest = min(voxel)
    cal = [v / smallest for v in voxel]
    regs = sub(root, "ViewRegistrations")
    missing_set = set(missing)
    for tp in timepoints:
        for s in setups:
            if (tp, s.id) in missing_set:
                continue
            reg = sub(
                regs, "ViewRegistration", timepoint=str(tp), setup=str(s.id)
            )
            tx, ty = (0.0, 0.0)
            if tile_offsets is not None:
                tx, ty = (o / smallest for o in tile_offsets[s.tile][:2])
            tz = s.z_min * cal[2]
            if tx or ty or tz:
                tr = sub(reg, "ViewTransform", type="affine")
                sub(tr, "Name", "Translation to Regular Grid")
                sub(
                    tr,
                    "affine",
                    " ".join(
                        fmt(v) for v in (1, 0, 0, tx, 0, 1, 0, ty, 0, 0, 1, tz)
                    ),
                )
            tr = sub(reg, "ViewTransform", type="affine")
            sub(tr, "Name", "calibration")
            sub(
                tr,
                "affine",
                " ".join(
                    fmt(v)
                    for v in (cal[0], 0, 0, 0, 0, cal[1], 0, 0, 0, 0, cal[2], 0)
                ),
            )

    for tag in (
        "ViewInterestPoints",
        "BoundingBoxes",
        "PointSpreadFunctions",
        "StitchingResults",
        "IntensityAdjustments",
    ):
        sub(root, tag)

    ET.indent(root, space="  ")
    ET.ElementTree(root).write(xml_path, encoding="utf-8", xml_declaration=True)


def metadata_offsets(stacks, tiles, channels, flip_x, flip_y, swap_xy):
    """Tile locations from the stage positions in the TIFF metadata.

    The position of each tile is read from its first plane (lowest T,
    first channel, lowest Z) and shifted so the smallest is 0.

    Parameters
    ----------
    stacks : dict
        Output of `scan`.
    tiles : list of tuple
        ``(well, field, position)`` of every tile, in tile order.
    channels : list of str
        Channel names, in channel order.
    flip_x, flip_y : bool
        Negate the stage x or y axis.
    swap_xy : bool
        Exchange the stage x and y axes, before any flip.

    Returns
    -------
    offsets : list of tuple of float or None
        Location in µm (x, y, z=0) of each tile; None if no file has a
        position. Stops the script if only some tiles have one.
    source : str or None
        Which kind of metadata the positions came from.
    """
    first_planes = []
    for tile in tiles:
        keys = sorted(
            (k for k in stacks if k[0] == tile),
            key=lambda k: (k[2], channels.index(k[1])),
        )
        planes = stacks[keys[0]].planes
        first_planes.append(planes[min(planes)])

    positions, sources = [], set()
    for path in first_planes:
        pos, source = read_stage_position(path)
        positions.append(pos)
        sources.add(source)
    found = sum(p is not None for p in positions)
    if found == 0:
        print(
            "WARNING: no stage position in the TIFF metadata; all tiles "
            "are at the origin. Run with --dump-metadata to see what the "
            "files contain, or give --grid-columns."
        )
        return None, None
    if found < len(positions):
        missing = [p.name for p, q in zip(first_planes, positions) if q is None]
        sys.exit(
            f"Stage position missing in {len(missing)} of {len(positions)} "
            f"tiles, e.g. {missing[0]}. Use --grid-columns or "
            "--ignore-positions."
        )

    xy = np.array([p[:2] for p in positions], dtype=float)
    if swap_xy:
        xy = xy[:, ::-1]
    if flip_x:
        xy[:, 0] *= -1
    if flip_y:
        xy[:, 1] *= -1
    xy -= xy.min(axis=0)
    if len(xy) > 1 and np.ptp(xy, axis=0).max() == 0:
        print("WARNING: every tile has the same stage position.")
    offsets = [(float(x), float(y), 0.0) for x, y in xy]
    return offsets, ", ".join(sorted(s for s in sources if s))


def grid_offsets(n_tiles, cols, overlap, snake, column_major, tile_xy, voxel):
    """Tile locations on a regular grid, in position order.

    Parameters
    ----------
    n_tiles : int
        Number of tiles.
    cols : int
        Grid columns.
    overlap : float
        Overlap between neighbouring tiles, in percent.
    snake : bool
        Reverse every other row (or column, if `column_major`).
    column_major : bool
        Fill columns first.
    tile_xy : tuple of int
        Tile size in pixels, x, y.
    voxel : tuple of float
        Voxel size in µm, x, y, z.

    Returns
    -------
    list of tuple of float
        Location in µm (x, y, z=0) of each tile.
    """
    step_x = tile_xy[0] * voxel[0] * (1 - overlap / 100)
    step_y = tile_xy[1] * voxel[1] * (1 - overlap / 100)
    rows = -(-n_tiles // cols)
    out = []
    for i in range(n_tiles):
        if column_major:
            c, r = divmod(i, rows)
            if snake and c % 2:
                r = rows - 1 - r
        else:
            r, c = divmod(i, cols)
            if snake and r % 2:
                c = cols - 1 - c
        out.append((c * step_x, r * step_y, 0.0))
    return out


# --------------------------------------------------------------------------
# Main
# --------------------------------------------------------------------------


def main(argv=None):
    """Run the conversion from the command line.

    Parameters
    ----------
    argv : list of str, optional
        Arguments; defaults to ``sys.argv[1:]``.
    """
    p = argparse.ArgumentParser(
        description=__doc__.split("\n\n")[0],
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    p.add_argument("folder", type=Path, help="folder with the TIFFs")
    p.add_argument(
        "-o",
        "--output",
        type=Path,
        help="output .xml (default: <folder>/dataset.xml); the "
        ".h5 is written next to it with the same name",
    )
    p.add_argument(
        "--voxel-size",
        nargs=3,
        type=float,
        metavar=("X", "Y", "Z"),
        help="voxel size in µm (default: from the TIFF tags, "
        "z step 1 if absent)",
    )
    p.add_argument(
        "--compression",
        choices=("none", "gzip"),
        default="none",
        help="HDF5 compression (gzip: smaller, slower)",
    )
    p.add_argument(
        "--workers",
        type=int,
        default=8,
        help="threads reading TIFFs (default 8)",
    )
    p.add_argument(
        "--ignore-positions",
        action="store_true",
        help="do not read stage positions from the TIFF metadata",
    )
    p.add_argument(
        "--flip-x",
        action="store_true",
        help="stage x runs opposite to image x",
    )
    p.add_argument(
        "--flip-y",
        action="store_true",
        help="stage y runs opposite to image y",
    )
    p.add_argument(
        "--swap-xy",
        action="store_true",
        help="stage x is image y and vice versa (applied before flips)",
    )
    p.add_argument(
        "--dump-metadata",
        action="store_true",
        help="print every tag of the first TIFF and exit",
    )
    p.add_argument(
        "--grid-columns",
        type=int,
        help="place the tiles on a grid this many columns wide, in "
        "position order, instead of using the stage positions",
    )
    p.add_argument(
        "--grid-rows", type=int, help="rows of the grid, with --column-major"
    )
    p.add_argument(
        "--overlap",
        type=float,
        default=10.0,
        help="tile overlap in %% for --grid-columns (default 10)",
    )
    p.add_argument(
        "--snake",
        action="store_true",
        help="serpentine grid: every other row runs backwards",
    )
    p.add_argument(
        "--column-major",
        action="store_true",
        help="positions fill columns first (needs --grid-rows)",
    )
    p.add_argument(
        "--dry-run",
        action="store_true",
        help="report what was found, write nothing",
    )
    p.add_argument(
        "--overwrite", action="store_true", help="replace an existing output"
    )
    args = p.parse_args(argv)

    folder = args.folder
    if not folder.is_dir():
        sys.exit(f"Not a folder: {folder}")
    xml_path = args.output or folder / "dataset.xml"
    if xml_path.suffix.lower() != ".xml":
        xml_path = xml_path.with_suffix(".xml")
    h5_path = xml_path.with_suffix(".h5")

    stacks = scan(folder)
    if args.dump_metadata:
        planes = stacks[min(stacks)].planes
        dump_metadata(planes[min(planes)])
        return
    tiles = sorted({k[0] for k in stacks})
    channels = sorted({k[1] for k in stacks}, key=channel_sort_key)
    timepoints = sorted({k[2] for k in stacks})
    names = tile_names(tiles)
    tile_idx = {t: i for i, t in enumerate(tiles)}
    chan_idx = {c: i for i, c in enumerate(channels)}

    first = next(iter(stacks.values())).planes
    shape, dtype, pixel, z_step = read_plane_info(first[min(first)])
    if args.voxel_size:
        voxel = tuple(args.voxel_size)
    else:
        if pixel is None:
            print(
                "WARNING: no pixel size in the TIFF tags; using 1 µm. "
                "Pass --voxel-size X Y Z."
            )
            pixel = (1.0, 1.0)
        if z_step is None:
            print(
                "WARNING: no z step in the TIFF tags; using 1 µm. "
                "Pass --voxel-size X Y Z."
            )
            z_step = 1.0
        voxel = (float(pixel[0]), float(pixel[1]), float(z_step))

    # One setup per (tile, channel). The z range is the union over
    # timepoints, so a setup has one size as BDV requires.
    setups: list[Setup] = []
    setup_of = {}
    for tile in tiles:
        for chan in channels:
            zs = [
                z
                for (t, c, _), st in stacks.items()
                if t == tile and c == chan
                for z in st.planes
            ]
            if not zs:
                continue
            z_min, z_max = min(zs), max(zs)
            s = Setup(
                id=len(setups),
                tile=tile_idx[tile],
                channel=chan_idx[chan],
                name=f"{names[tile]}_{chan}",
                size_xyz=(shape[1], shape[0], z_max - z_min + 1),
                z_min=z_min,
            )
            s.factors, s.chunks = plan_pyramid(s.size_xyz, voxel)
            setups.append(s)
            setup_of[(tile, chan)] = s

    missing = []
    gaps = 0
    for tp in timepoints:
        for (tile, chan), s in setup_of.items():
            st = stacks.get((tile, chan, tp))
            if st is None:
                missing.append((tp, s.id))
            else:
                gaps += s.size_xyz[2] - len(st.planes)

    tile_offsets = None
    if not (args.grid_columns or args.grid_rows or args.ignore_positions):
        tile_offsets, source = metadata_offsets(
            stacks, tiles, channels, args.flip_x, args.flip_y, args.swap_xy
        )
        if tile_offsets is not None:
            xs = [o[0] for o in tile_offsets]
            ys = [o[1] for o in tile_offsets]
            print(
                f"Tile positions from {source}: "
                f"x 0..{max(xs):.1f} µm, y 0..{max(ys):.1f} µm "
                f"(one tile is {shape[1] * voxel[0]:.1f} x "
                f"{shape[0] * voxel[1]:.1f} µm)"
            )
    if args.grid_columns or args.grid_rows:
        if args.column_major:
            if not args.grid_rows:
                sys.exit("--column-major needs --grid-rows")
            cols = -(-len(tiles) // args.grid_rows)
        else:
            if not args.grid_columns:
                sys.exit(
                    "--grid-rows needs --column-major; otherwise give "
                    "--grid-columns"
                )
            cols = args.grid_columns
        tile_offsets = grid_offsets(
            len(tiles),
            cols,
            args.overlap,
            args.snake,
            args.column_major,
            (shape[1], shape[0]),
            voxel,
        )

    n_files = sum(len(st.planes) for st in stacks.values())
    zsizes = sorted({s.size_xyz[2] for s in setups})
    print(f"Found {n_files} planes in {folder}")
    print(
        f"  tiles:      {len(tiles)} ({names[tiles[0]]} .. {names[tiles[-1]]})"
    )
    print(f"  channels:   {len(channels)} ({', '.join(channels)})")
    print(
        f"  timepoints: {len(timepoints)} ({', '.join(map(str, timepoints))})"
    )
    print(f"  plane:      {shape[1]} x {shape[0]} px, {dtype}")
    print(f"  z planes:   {zsizes[0]}..{zsizes[-1]} per stack")
    print(f"  voxel:      {voxel[0]:g} x {voxel[1]:g} x {voxel[2]:g} µm")
    print(
        f"  setups:     {len(setups)}, pyramid levels "
        f"{min(len(s.factors) for s in setups)}.."
        f"{max(len(s.factors) for s in setups)}"
    )
    if missing:
        print(
            f"  missing:    {len(missing)} view(s) absent at some "
            "timepoint (written as MissingViews)"
        )
    if gaps:
        print(
            f"  WARNING: {gaps} plane(s) missing inside stacks, "
            "filled with zeros"
        )
    size_gb = n_files * shape[0] * shape[1] * 2 / 1e9
    print(
        f"  output:     {xml_path} + {h5_path.name} "
        f"(~{size_gb * 1.15:.1f} GB uncompressed)"
    )
    if args.dry_run:
        return

    for path in (xml_path, h5_path):
        if path.exists() and not args.overwrite:
            sys.exit(f"{path} exists; pass --overwrite to replace it")
    xml_path.parent.mkdir(parents=True, exist_ok=True)

    compression = None if args.compression == "none" else "gzip"
    todo = [
        (tp, tile, chan)
        for tp in timepoints
        for (tile, chan) in setup_of
        if (tile, chan, tp) in stacks
    ]
    t0 = time.time()
    with (
        h5py.File(h5_path, "w") as h5,
        ThreadPoolExecutor(args.workers) as pool,
    ):
        for s in setups:
            write_setup_meta(h5, s)
        for i, (tp, tile, chan) in enumerate(todo, 1):
            s = setup_of[(tile, chan)]
            write_stack(h5, stacks[(tile, chan, tp)], s, tp, pool, compression)
            elapsed = time.time() - t0
            eta = elapsed / i * (len(todo) - i)
            print(
                f"[{i}/{len(todo)}] t{tp} {s.name}  "
                f"{elapsed / 60:.1f} min, ~{eta / 60:.1f} min left",
                flush=True,
            )

    write_xml(
        xml_path,
        h5_path,
        setups,
        [names[t] for t in tiles],
        channels,
        timepoints,
        missing,
        voxel,
        tile_offsets,
    )
    print(f"Done: {xml_path}")


if __name__ == "__main__":
    main()
