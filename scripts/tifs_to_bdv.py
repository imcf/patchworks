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

The channel suffix is optional: ``A1--W00001--P00082--Z00009--T00000.tif``
is read as a single-channel acquisition.

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

To add a pyramid to an existing BDV H5/XML that has none (e.g. a fused
BigStitcher export), pass the .xml instead of a folder. Level 0 is kept,
the lower levels are added to the same .h5::

    uv run scripts/tifs_to_bdv.py /data/Scan/fused_bdv.xml --dry-run  # report
    uv run scripts/tifs_to_bdv.py /data/Scan/fused_bdv.xml            # add

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
from functools import lru_cache
from pathlib import Path

import h5py
import numpy as np
import tifffile

NAME_RE = re.compile(
    r"^(?P<well>.+?)--W(?P<field>\d+)--P(?P<pos>\d+)--Z(?P<z>\d+)"
    r"--T(?P<t>\d+)(?:--(?P<chan>.+?))?\.tiff?$",
    re.IGNORECASE,
)

# Largest chunk edge in x/y; z chunks are scaled by the anisotropy.
CHUNK_XY = 64
# Stop adding pyramid levels once every dimension is at most this.
MIN_LEVEL_SIZE = 64
MAX_LEVELS = 12
# Memory budget for one downsampling block, in bytes of float32.
BLOCK_BYTES = 256 * 2**20


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
        # Single-channel acquisitions have no channel suffix.
        key = (tile, m["chan"] or "", int(m["t"]))
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
        Pixel size (x, y) in µm from the resolution tags, else from a
        companion OME-XML; None if neither has it.
    z_step : float or None
        Z spacing in µm from the ImageJ metadata or a companion OME-XML.
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
        comp = companion_of(tif, path)
        if comp is not None and comp.voxel is not None:
            if pixel is None and None not in comp.voxel[:2]:
                pixel = comp.voxel[:2]
            if z_step is None:
                z_step = comp.voxel[2]
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


def _um(attrib, key):
    """Read an OME length attribute in µm.

    Parameters
    ----------
    attrib : dict
        Attributes of an OME element.
    key : str
        Attribute name, e.g. ``PositionX``; its unit is ``<key>Unit``.

    Returns
    -------
    float or None
        The value in µm (OME's default unit), None if absent.
    """
    v = attrib.get(key)
    if v is None:
        return None
    return float(v) * OME_TO_UM.get(attrib.get(f"{key}Unit", "µm"), 1.0)


def _local(el):
    """Tag of an XML element without its namespace."""
    return el.tag.rsplit("}", 1)[-1]


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
        tag = _local(el)
        if tag == "Plane" and "PositionX" in el.attrib:
            keys = ("PositionX", "PositionY", "PositionZ")
        elif tag == "StageLabel" and "X" in el.attrib:
            keys = ("X", "Y", "Z")
        else:
            continue
        return [_um(el.attrib, k) for k in keys]
    return None


@dataclass
class CompanionOME:
    """Metadata from a companion ``.ome.xml`` file (e.g. ScanR's).

    Attributes
    ----------
    path : Path
        The companion file.
    positions : dict of str to tuple
        Stage position (x, y, z) in µm of each TIFF, by file name.
    voxel : tuple of float or None
        Voxel size (x, y, z) in µm; an entry is None if not stored.
    """

    path: Path
    positions: dict[str, tuple]
    voxel: tuple | None


_COMPANIONS: dict[Path, CompanionOME] = {}


def find_companion(tif_path: Path, description: str):
    """Locate the companion OME-XML a ``BinaryOnly`` TIFF points to.

    Parameters
    ----------
    tif_path : Path
        The TIFF.
    description : str
        Its ImageDescription.

    Returns
    -------
    Path or None
        The companion file, looked for next to the TIFF and up to two
        folders above it; None if the TIFF names none or it is missing.
    """
    m = re.search(r"<BinaryOnly[^>]*MetadataFile=\"([^\"]+)\"", description)
    if not m:
        return None
    return _locate(tif_path.resolve().parent, m.group(1))


@lru_cache
def _locate(folder: Path, name: str):
    """Find `name` in `folder` or up to two folders above, warn once."""
    for d in (folder, *list(folder.parents)[:2]):
        if (d / name).is_file():
            return d / name
    print(f"WARNING: the TIFFs point to {name}, not found near {folder}.")
    return None


def load_companion(path: Path) -> CompanionOME:
    """Parse a companion OME-XML once and cache it.

    Each ``TiffData`` names its file in ``UUID/@FileName`` and its plane
    with ``FirstZ``/``FirstC``/``FirstT``; that plane's ``Plane`` element
    gives the stage position. Without a matching ``Plane`` the first
    plane of the image is used.

    Parameters
    ----------
    path : Path
        The companion ``.ome.xml``.

    Returns
    -------
    CompanionOME
        Positions by file name, and the voxel size.
    """
    if path in _COMPANIONS:
        return _COMPANIONS[path]
    print(f"Reading {path} ...", flush=True)
    root = ET.parse(path).getroot()
    positions = {}
    voxel = None
    for pixels in root.iter():
        if _local(pixels) != "Pixels":
            continue
        a = pixels.attrib
        planes = {}
        for el in pixels:
            if _local(el) == "Plane" and "PositionX" in el.attrib:
                key = tuple(
                    int(el.attrib.get(k, 0)) for k in ("TheZ", "TheC", "TheT")
                )
                planes[key] = tuple(
                    _um(el.attrib, k)
                    for k in ("PositionX", "PositionY", "PositionZ")
                )
        if voxel is None and "PhysicalSizeX" in a:
            z = _um(a, "PhysicalSizeZ")
            if z is None:
                # No PhysicalSizeZ: take the step between planes 0 and 1.
                z0, z1 = planes.get((0, 0, 0)), planes.get((1, 0, 0))
                if z0 and z1 and z0[2] is not None and z1[2] is not None:
                    z = abs(z1[2] - z0[2]) or None
            voxel = (_um(a, "PhysicalSizeX"), _um(a, "PhysicalSizeY"), z)
        first = planes[min(planes)] if planes else None
        for td in pixels:
            if _local(td) != "TiffData":
                continue
            uuid = next((u for u in td if _local(u) == "UUID"), None)
            if uuid is None or "FileName" not in uuid.attrib:
                continue
            key = tuple(
                int(td.attrib.get(k, 0)) for k in ("FirstZ", "FirstC", "FirstT")
            )
            pos = planes.get(key, first)
            if pos is not None:
                positions[Path(uuid.attrib["FileName"]).name] = pos
    comp = CompanionOME(path=path, positions=positions, voxel=voxel)
    _COMPANIONS[path] = comp
    return comp


def companion_of(tif, path: Path):
    """Companion metadata of an open TIFF, if it has one.

    Parameters
    ----------
    tif : tifffile.TiffFile
        The open TIFF.
    path : Path
        Its path.

    Returns
    -------
    CompanionOME or None
    """
    desc = tif.pages[0].tags.get("ImageDescription")
    if desc is None or not isinstance(desc.value, str):
        return None
    comp_path = find_companion(path, desc.value)
    return load_companion(comp_path) if comp_path else None


def read_stage_position(path: Path):
    """Read the stage position stored in one TIFF.

    Tries, in order: OME-XML, a companion OME-XML file the TIFF points
    to (``BinaryOnly``, as ScanR writes), Micro-Manager metadata, key/value text in
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

        comp = companion_of(tif, path)
        if comp is not None and path.name in comp.positions:
            return comp.positions[path.name], f"OME companion {comp.path.name}"

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
        comp = companion_of(tif, path)
    if comp is not None:
        print(f"--- Companion {comp.path}")
        text = comp.path.read_text(errors="replace")
        print(text[:limit] + (" ..." if len(text) > limit else ""))
        print(
            f"=== {len(comp.positions)} files with a position, "
            f"voxel {comp.voxel}"
        )
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

    Works block by block -- a z slab of chunk height and as many rows as
    fit in ``BLOCK_BYTES`` -- so memory stays bounded even for a single
    huge 2D plane.

    Parameters
    ----------
    src : h5py.Dataset
        The finer level, (z, y, x). int16 is read as the uint16 BDV
        stores in it; other types are averaged as they are.
    dst : h5py.Dataset
        The coarser level to fill, same type as `src`.
    rel_zyx : tuple of int
        Downsampling factor from `src` to `dst`, z, y, x (1 or 2 each).
    """
    rz, ry, rx = rel_zyx
    nz, ny, nx = dst.shape
    as_u16 = src.dtype == np.int16
    out_type = np.uint16 if as_u16 else dst.dtype
    cz = max(1, (dst.chunks or dst.shape)[0])
    cy = max(1, (dst.chunks or dst.shape)[1])
    # Rows per block: a multiple of the chunk height within the budget.
    row_bytes = cz * rz * ry * nx * rx * 4
    rows = max(cy, BLOCK_BYTES // max(row_bytes, 1) // cy * cy)
    for z0 in range(0, nz, cz):
        z1 = min(nz, z0 + cz)
        for y0 in range(0, ny, rows):
            y1 = min(ny, y0 + rows)
            block = src[z0 * rz : z1 * rz, y0 * ry : y1 * ry, : nx * rx]
            if as_u16:
                block = block.view(np.uint16)
            block = block.astype(np.float32).reshape(
                z1 - z0, rz, y1 - y0, ry, nx, rx
            )
            block = block.mean(axis=(1, 3, 5))
            if np.issubdtype(out_type, np.integer):
                block = np.round(block)
            block = block.astype(out_type)
            dst[z0:z1, y0:y1] = block.view(np.int16) if as_u16 else block


def add_pyramid(xml_path: Path, overwrite: bool = False, dry_run: bool = False):
    """Add a multiresolution pyramid to an existing BDV H5/XML.

    For a file written without one (e.g. a fused BigStitcher export):
    every setup gets the same levels this script writes for new data,
    computed from the level-0 data already in the HDF5 file. Only new
    datasets are added and the setup's ``resolutions`` and
    ``subdivisions`` updated; level 0 is never touched.

    Parameters
    ----------
    xml_path : Path
        The BDV XML; its ``<hdf5>`` names the file to extend.
    overwrite : bool, optional
        Rebuild the levels of setups that already have a pyramid.
        Without it, those setups are left alone.
    dry_run : bool, optional
        Only report the levels each setup has; change nothing.
    """
    root = ET.parse(xml_path).getroot()
    el = root.find("SequenceDescription/ImageLoader/hdf5")
    if el is None:
        sys.exit(f"{xml_path} is not a BDV HDF5 dataset.")
    h5_path = Path(el.text.strip().replace("\\", "/"))
    if el.get("type", "relative") == "relative":
        h5_path = xml_path.parent / h5_path
    if not h5_path.is_file():
        sys.exit(f"{xml_path.name} points to {h5_path}, which does not exist.")

    setups = {}
    for vs in root.iter("ViewSetup"):
        size = tuple(int(v) for v in vs.findtext("size").split())
        vox = vs.findtext("voxelSize/size")
        voxel = tuple(float(v) for v in vox.split()) if vox else (1, 1, 1)
        setups[int(vs.findtext("id"))] = (size, voxel)

    t0 = time.time()
    with h5py.File(h5_path, "r" if dry_run else "r+") as h5:
        for sid, (size, voxel) in sorted(setups.items()):
            sname = f"s{sid:02d}"
            if sname not in h5:
                continue
            levels = len(h5[f"{sname}/resolutions"])
            if dry_run:
                factors = h5[f"{sname}/resolutions"][()].astype(int).tolist()
                print(
                    f"  {sname}: {levels} level(s), downsampling "
                    + ", ".join("x".join(map(str, f)) for f in factors)
                )
                continue
            if levels > 1 and not overwrite:
                print(f"  {sname}: already {levels} levels, skipped")
                continue
            factors, chunks = plan_pyramid(size, voxel)
            tps = sorted(
                k for k in h5 if k.startswith("t") and f"{k}/{sname}" in h5
            )
            for tp in tps:
                grp = h5[f"{tp}/{sname}"]
                base = grp["0/cells"]
                for level in list(grp):
                    if level != "0":
                        del grp[level]
                prev, prev_f = base, factors[0]
                for level in range(1, len(factors)):
                    f, c = factors[level], chunks[level]
                    shape = tuple(
                        max(n // fi, 1)
                        for n, fi in zip(size[::-1], f[::-1], strict=True)
                    )
                    ds = grp.create_dataset(
                        f"{level}/cells",
                        shape=shape,
                        dtype=base.dtype,
                        chunks=tuple(c[::-1]),
                        compression=base.compression,
                        compression_opts=base.compression_opts,
                    )
                    rel = tuple(
                        a // b
                        for a, b in zip(f[::-1], prev_f[::-1], strict=True)
                    )
                    downsample(prev, ds, rel)
                    prev, prev_f = ds, f
            for name, data, dtype in (
                ("resolutions", factors, np.float64),
                ("subdivisions", chunks, np.int32),
            ):
                del h5[f"{sname}/{name}"]
                h5[f"{sname}/{name}"] = np.array(data, dtype=dtype)
            top = [
                max(n // fi, 1) for n, fi in zip(size, factors[-1], strict=True)
            ]
            print(
                f"  {sname}: {len(factors)} levels, coarsest "
                f"{' x '.join(map(str, top))} "
                f"({(time.time() - t0) / 60:.1f} min)",
                flush=True,
            )
    if not dry_run:
        print(f"Done: {h5_path}")


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
    tile_locations=None,
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
        Position in µm (x, y, z) of each tile relative to the first, which
        the registrations use; None for all at the origin.
    tile_locations : list of tuple of float, optional
        Absolute stage position in µm written as each tile's
        ``<location>``, so other data in stage coordinates (a prescan,
        detections) can be placed against these tiles. Defaults to
        `tile_offsets`.
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
        sub(ch, "name", name or "0")
    att = sub(vs, "Attributes", name="tile")
    for i, name in enumerate(tiles):
        t = sub(att, "Tile")
        sub(t, "id", i)
        sub(t, "name", name)
        locations = tile_locations or tile_offsets
        if locations is not None:
            sub(t, "location", " ".join(fmt(v) for v in locations[i]))
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
        Location in µm (x, y, z=0) of each tile, relative to the lowest;
        None if no file has a position. Stops the script if only some
        tiles have one.
    stage : list of tuple of float or None
        The absolute stage positions (x, y, z) in µm, as read.
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
        return None, None, None
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
    stage = [(float(p[0]), float(p[1]), float(p[2] or 0.0)) for p in positions]
    return offsets, stage, ", ".join(sorted(s for s in sources if s))


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
    p.add_argument(
        "folder",
        type=Path,
        help="folder with the TIFFs -- or an existing BDV .xml, to add a "
        "pyramid to its .h5 (see --overwrite)",
    )
    p.add_argument(
        "-o",
        "--output",
        type=Path,
        help="output .xml (default: dataset.xml next to the folder, "
        "outside the TIFFs); the "
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
        "--xml-only",
        action="store_true",
        help="rewrite only the .xml next to an existing .h5 (e.g. to "
        "update tile positions); the image data is not touched",
    )
    p.add_argument(
        "--dry-run",
        action="store_true",
        help="report what was found, write nothing",
    )
    p.add_argument(
        "--overwrite",
        action="store_true",
        help="replace an existing output; with an .xml input, rebuild "
        "pyramids that already exist",
    )
    args = p.parse_args(argv)

    folder = args.folder
    if folder.is_file() and folder.suffix.lower() == ".xml":
        add_pyramid(folder, overwrite=args.overwrite, dry_run=args.dry_run)
        return
    if not folder.is_dir():
        sys.exit(f"Not a folder: {folder}")
    # Next to the folder, not in it: among thousands of TIFFs the output
    # sorts to the bottom of a file browser and is easily missed.
    xml_path = args.output or folder.resolve().parent / "dataset.xml"
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
                "WARNING: no pixel size in the metadata; using 1 µm. "
                "Pass --voxel-size X Y Z."
            )
            pixel = (1.0, 1.0)
        if z_step is None:
            print(
                "WARNING: no z step in the metadata; using 1 µm. "
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
                name=f"{names[tile]}_{chan}" if chan else names[tile],
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

    tile_offsets = tile_locations = None
    if not (args.grid_columns or args.grid_rows or args.ignore_positions):
        tile_offsets, tile_locations, source = metadata_offsets(
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
    print(
        f"  channels:   {len(channels)} ({', '.join(c or 'none' for c in channels)})"
    )
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

    if args.xml_only:
        if not h5_path.is_file():
            sys.exit(f"--xml-only: {h5_path} does not exist")
    else:
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
    if args.xml_only:
        todo = []
    t0 = time.time()
    with (
        h5py.File(h5_path, "r" if args.xml_only else "w") as h5,
        ThreadPoolExecutor(args.workers) as pool,
    ):
        for s in setups if not args.xml_only else ():
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
        tile_locations,
    )
    for path in (xml_path, h5_path):
        if not path.is_file():
            sys.exit(f"ERROR: {path.resolve()} was not written.")
        print(f"Wrote {path.resolve()} ({path.stat().st_size / 1e9:.2f} GB)")
    print(
        "Done. Open the .xml in Fiji: Plugins > BigDataViewer or BigStitcher."
    )


if __name__ == "__main__":
    main()
