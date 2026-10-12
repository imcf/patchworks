# /// script
# requires-python = ">=3.11"
# dependencies = []
# ///
"""Make a MoBIE collection table from the BDV H5/XML files below a folder.

Every BigDataViewer ``.xml`` found (recursively) becomes a dataset; its
rows are named after the folders above it (``--name-level``), its view
after its own folder. Selecting the view in MoBIE shows all its tiles and
channels together.

MoBIE loads one single-channel image per table row, and for a BDV XML its
``channel`` column picks a *view setup* (a tile in one channel). So the
table has one row per setup: with 168 tiles and 4 channels, 672 rows per
XML. Tiles of the same channel share one ``display``, so a single contrast
and color setting controls the whole channel.

**Stage coordinates.** Everything is placed in the microscope's stage
coordinates, in µm, so all acquisitions of a sample overlay:

- The tiled 3D dataset (the reference, see below) is put at its stage
  position, from a ``positions.xml`` whose spots match its tiles, or
  from absolute tile ``<location>``s (``tifs_to_bdv.py`` writes them;
  ``--xml-only`` updates an older XML without touching its .h5).
- A stitched 2D dataset with a ScanR ``metadata.ome.xml`` in its own
  folder or elsewhere in the sample (e.g. the prescan ``Scan``) is put
  on that ScanR field grid: centred on it, scaled so the fields land on
  their stage positions. ``Scan`` is mirrored in x by default
  (``--flip-x``/``--flip-y`` NAME change that).
- Datasets with other calibrations are scaled from their pixels to µm.
- ``--offset NAME=DX,DY`` fine-tunes a dataset (by its folder name).

**Views.** The view of every dataset is the name of its folder
(``Acquisition``, ``Scan``, ``Detection``, ...). All views share the
stage coordinates, so showing several overlays them.

**2D in 3D.** Datasets are grouped by *sample*, the folder above their
own (``ScanR/Acquisition`` and ``ScanR/Scan`` are both in sample ``ScanR``).
Within a sample, every 2D dataset (one z plane) is shifted along z to the
middle of the sample's 3D dataset, through the ``affine`` column, so it
sits inside the stack instead of at its top or bottom. If a sample has
several 3D datasets, the one with "scan" in its name is the reference
(``--reference`` picks another).

**Spots.** A ScanR ``metadata.ome.xml`` in a folder without images or
BDV XML (e.g. ``Detection``) becomes a spots dataset: the distinct
``PositionX``/``PositionY`` of its planes, in acquisition order, are
written to ``detection_positions.tsv`` next to it (``spot_id``, ``x``,
``y``, ``z``; tab-separated, as MoBIE reads ``.tsv``). A ``positions.xml``
alone in a folder likewise becomes ``positions_spots.tsv``. Spots are in
stage µm and drawn at mid-stack with a radius of 10 (``--spot-radius``).

Columns written (see https://mobie.github.io/tutorials/mobie_collection_table.html):

- ``uri``: the XML or spots table, relative to the collection table (or
  absolute, ``--absolute``)
- ``type``: ``intensities`` or ``spots``
- ``channel``: setup index in the XML
- ``name``: ``<view>_<setup name>``, unique per row
- ``view``: the dataset's folder name
- ``display``: ``<view>_<channel>``, shared by all tiles of a channel
- ``color``: from the channel's wavelength (405 Blue, 488 Green, ...)
- ``contrast_limits``: empty (MoBIE's default) unless ``--contrast-limits``
- ``spot_radius``: for spots
- ``affine``: from the XML's coordinates to stage µm (scaling, stage
  position, flips, ``--offset``, z shift of 2D datasets)
- ``channel_name``, ``pixel_unit``, ``pixel_size_x/y/z``,
  ``num_pixels_x/y/z``, ``sample``, ``tile``: annotations for browsing,
  the first ones as Fiji's own "Create MoBIE Collection Table" writes them

Usage
-----
    uv run scripts/bdv_to_mobie_table.py /data/project --dry-run
    uv run scripts/bdv_to_mobie_table.py /data/project
    uv run scripts/bdv_to_mobie_table.py /data/project -o /data/project/mobie.tsv

Then in Fiji: Plugins > MoBIE > Open > Open Collection Table..., pick the
table, ``Data Root``: ``UseTableFolder`` (``PathsInTableAreAbsolute`` with
``--absolute``), ``Viewing mode``: ``ThreeDimensional``.
"""

from __future__ import annotations

import argparse
import csv
import re
import sys
import xml.etree.ElementTree as ET
from dataclasses import dataclass, field
from pathlib import Path

# Order and annotation columns as Fiji's MoBIE > Create > Create MoBIE
# Collection Table writes them.
COLUMNS = [
    "uri",
    "type",
    "name",
    "channel_name",
    "pixel_unit",
    "pixel_size_x",
    "pixel_size_y",
    "pixel_size_z",
    "num_pixels_x",
    "num_pixels_y",
    "num_pixels_z",
    "channel",
    "view",
    "display",
    "color",
    "contrast_limits",
    "spot_radius",
    "affine",
    "sample",
    "tile",
]

# (upper wavelength bound in nm, MoBIE color)
WAVELENGTH_COLORS = [
    (460, "Blue"),
    (530, "Green"),
    (600, "Red"),
    (10_000, "Magenta"),
]

IDENTITY = (1.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0)

# Spot coordinate keys, compared lowercased without separators:
# x, posx, positionx, xpos, centerx, stagex, xum, positionxum, ...
COORD_RE = {
    axis: re.compile(
        rf"^(?:pos(?:ition)?|center|centre|stage)?{axis}"
        rf"(?:pos(?:ition)?)?(?:um|µm|micron|px|pixel)?$"
    )
    for axis in "xyz"
}
ID_KEYS = ("id", "spotid", "position", "name", "label", "index")


@dataclass
class SetupInfo:
    """One view setup of a BDV XML.

    Attributes
    ----------
    index : int
        Position in the id-sorted setup list; MoBIE's ``channel`` value.
    name : str
        Setup name, e.g. ``P00082_405nm``.
    tile : str
        Name of its tile, empty if the XML has no tile attribute.
    channel : str
        Name of its channel, empty if the XML has no channel attribute.
    unit : str
        Unit of the voxel size, e.g. ``µm``.
    voxel : tuple of str
        Voxel size x, y, z, as written in the XML.
    size : tuple of str
        Image size in pixels x, y, z, as written in the XML.
    z_range : tuple of float
        Lowest and highest z the setup covers in world coordinates,
        through its registration at the first timepoint.
    um_per_world : float
        Size of one world unit in µm: XMLs written by ``tifs_to_bdv.py``
        or BigStitcher count world units in pixels, not µm.
    """

    index: int
    name: str
    tile: str
    channel: str
    unit: str = ""
    voxel: tuple[str, str, str] = ("", "", "")
    size: tuple[str, str, str] = ("", "", "")
    z_range: tuple[float, float] = (0.0, 0.0)
    um_per_world: float = 1.0
    reg: tuple = IDENTITY


@dataclass
class Source:
    """One dataset of the table: a BDV XML or a spots file.

    Attributes
    ----------
    path : Path
        The BDV XML, or the ``positions.xml`` of a spots dataset.
    kind : str
        ``intensities`` or ``spots``.
    view : str
        View name in MoBIE.
    sample : Path
        Folder that groups datasets for the 2D-in-3D placement.
    setups : list of SetupInfo
        Setups of an XML; empty for spots.
    spots : list of dict
        ``spot_id``, ``x``, ``y``, ``z`` (None if absent) of a spots set.
    dz : float
        z shift (µm) that puts a 2D dataset in the middle of the
        sample's 3D reference.
    offset : tuple of float
        Extra x, y, z shift in µm, from ``--offset``.
    """

    path: Path
    kind: str
    view: str = ""
    sample: Path = Path()
    setups: list[SetupInfo] = field(default_factory=list)
    spots: list[dict] = field(default_factory=list)
    dz: float = 0.0
    offset: tuple[float, float, float] = (0.0, 0.0, 0.0)
    flip: tuple[bool, bool] = (False, False)
    meta: dict | None = None
    matrix: tuple | None = None
    table: Path | None = None

    @property
    def is_3d(self):
        """Whether any setup has more than one z plane."""
        return any(s.size[2] not in ("", "1") for s in self.setups)

    @property
    def z_range(self):
        """World z range covered by all setups together."""
        return (
            min(s.z_range[0] for s in self.setups),
            max(s.z_range[1] for s in self.setups),
        )

    @property
    def scale(self):
        """µm per world unit of this XML."""
        return self.setups[0].um_per_world if self.setups else 1.0

    @property
    def z_range_um(self):
        """z range in µm, offset included."""
        lo, hi = self.z_range
        return lo * self.scale + self.offset[2], hi * self.scale + self.offset[
            2
        ]

    def xy_range(self):
        """World x and y ranges covered by all setups together."""
        xs, ys = [], []
        for st in self.setups:
            try:
                dims = [max(int(v) - 1, 0) for v in st.size]
            except ValueError:
                dims = [0, 0, 0]
            m = st.reg
            for x in (0, dims[0]):
                for y in (0, dims[1]):
                    xs.append(m[0] * x + m[1] * y + m[3])
                    ys.append(m[4] * x + m[5] * y + m[7])
        return (min(xs), max(xs)), (min(ys), max(ys))

    def affine(self):
        """MoBIE ``affine`` cell, from this XML's world units to µm.

        Either the placement computed from stage metadata (`matrix`), or
        a scaling to µm, mirrored about the dataset's centre if asked;
        then ``--offset`` and the z shift.
        """
        ox, oy, oz = self.offset
        if self.matrix is not None:
            m = list(self.matrix)
        else:
            sc = self.scale
            m = [sc, 0, 0, 0, 0, sc, 0, 0, 0, 0, sc, 0]
            (x0, x1), (y0, y1) = self.xy_range()
            if self.flip[0]:
                m[0], m[3] = -sc, sc * (x0 + x1)
            if self.flip[1]:
                m[5], m[7] = -sc, sc * (y0 + y1)
        m[3] += ox
        m[7] += oy
        m[11] += oz + self.dz
        if tuple(m) == IDENTITY:
            return ""
        return "(" + ",".join(fmt(v) for v in m) + ")"


# --------------------------------------------------------------------------
# BDV XML
# --------------------------------------------------------------------------


def compose(a, b):
    """Compose two row-packed 3x4 affines, ``a`` after ``b``.

    Parameters
    ----------
    a, b : sequence of float
        12 values each, row-packed.

    Returns
    -------
    tuple of float
        The 12 values of ``a @ b``.
    """
    out = []
    for r in range(3):
        row = a[4 * r : 4 * r + 4]
        for c in range(4):
            v = sum(row[k] * b[4 * k + c] for k in range(3))
            out.append(v + (row[3] if c == 3 else 0.0))
    return tuple(out)


def invert(m):
    """Inverse of a row-packed 3x4 affine.

    Parameters
    ----------
    m : sequence of float

    Returns
    -------
    tuple of float
    """
    a, b, c, d, e, f, g, h, i = (
        m[0],
        m[1],
        m[2],
        m[4],
        m[5],
        m[6],
        m[8],
        m[9],
        m[10],
    )
    det = a * (e * i - f * h) - b * (d * i - f * g) + c * (d * h - e * g)
    if not det:
        sys.exit("A registration in the XML is not invertible.")
    inv = [
        (e * i - f * h) / det,
        (c * h - b * i) / det,
        (b * f - c * e) / det,
        (f * g - d * i) / det,
        (a * i - c * g) / det,
        (c * d - a * f) / det,
        (d * h - e * g) / det,
        (b * g - a * h) / det,
        (a * e - b * d) / det,
    ]
    t = (m[3], m[7], m[11])
    out = []
    for r in range(3):
        row = inv[3 * r : 3 * r + 3]
        out += [*row, -sum(row[k] * t[k] for k in range(3))]
    return tuple(out)


def read_scanr_meta(path: Path):
    """Field positions of a ScanR acquisition, from its metadata.ome.xml.

    Parameters
    ----------
    path : Path
        A ``metadata.ome.xml`` as ScanR writes it.

    Returns
    -------
    dict or None
        ``positions``: the distinct stage (x, y) in µm of the fields, in
        acquisition order (the planes repeat them per channel and z);
        ``field_px``: field size (x, y) in pixels; ``pixel``: nominal
        pixel size (x, y) in µm. None if the file has no plane positions.
    """
    try:
        root = ET.parse(path).getroot()
    except ET.ParseError:
        return None
    positions, field_px, pixel = {}, None, None
    for el in root.iter():
        tag = el.tag.rsplit("}", 1)[-1]
        if tag == "Pixels" and field_px is None:
            try:
                field_px = (int(el.get("SizeX")), int(el.get("SizeY")))
                pixel = (
                    float(el.get("PhysicalSizeX")),
                    float(el.get("PhysicalSizeY")),
                )
            except (TypeError, ValueError):
                pass
        elif tag == "Plane" and el.get("PositionX") is not None:
            try:
                xy = (float(el.get("PositionX")), float(el.get("PositionY")))
            except (TypeError, ValueError):
                continue
            positions.setdefault(xy, None)
    if not positions:
        return None
    return {"positions": list(positions), "field_px": field_px, "pixel": pixel}


def place_on_grid(src, meta, flip):
    """Put a stitched image of a ScanR grid at its stage position.

    The stitched image spans the field grid, so its centre is the grid's
    centre, and the spacing of the fields in the image sets its scale:
    ``(grid extent in µm) / (image size - one field, in pixels)``. That
    is more reliable for overlaying stage-positioned data than the
    nominal pixel size, which the stitching need not reproduce.

    Parameters
    ----------
    src : Source
        The stitched dataset (one setup is used for the geometry).
    meta : dict
        Output of `read_scanr_meta`.
    flip : tuple of bool
        Mirror the image in x and/or y.

    Returns
    -------
    matrix : tuple of float
        Row-packed affine from the XML's world coordinates to stage µm.
    note : str
        What was done, for the report.
    """
    st = src.setups[0]
    w, h = (int(v) for v in st.size[:2])
    xs = sorted({p[0] for p in meta["positions"]})
    ys = sorted({p[1] for p in meta["positions"]})
    nominal = meta["pixel"] or (st.um_per_world, st.um_per_world)
    fw, fh = meta["field_px"] or (0, 0)
    scale = []
    for extent, size, one_field, nom in (
        (xs[-1] - xs[0], w, fw, nominal[0]),
        (ys[-1] - ys[0], h, fh, nominal[1]),
    ):
        fitted = (
            extent / (size - one_field) if size > one_field and extent else 0
        )
        # Fall back to the nominal size if the fit is implausible.
        scale.append(fitted if 0.85 < fitted / nom < 1.15 else nom)
    sx, sy = scale
    fx, fy = (-1 if f else 1 for f in flip)
    cx, cy = (xs[0] + xs[-1]) / 2, (ys[0] + ys[-1]) / 2
    uc, vc = (w - 1) / 2, (h - 1) / 2
    try:
        sz = float(st.voxel[2])
    except ValueError:
        sz = 1.0
    pixel_to_stage = (
        fx * sx,
        0,
        0,
        cx - fx * sx * uc,
        0,
        fy * sy,
        0,
        cy - fy * sy * vc,
        0,
        0,
        sz,
        0,
    )
    matrix = compose(pixel_to_stage, invert(st.reg))
    note = (
        f"{src.view}: placed on its ScanR grid ({len(xs)} x {len(ys)} fields), "
        f"centred at stage ({fmt(cx)}, {fmt(cy)}) µm, {fmt(sx)} x {fmt(sy)} "
        f"µm/px (nominal {fmt(nominal[0])})"
        + (
            ", flipped in " + "/".join(a for a, f in zip("XY", flip) if f)
            if any(flip)
            else ""
        )
    )
    return matrix, note


def stage_offset(ref, spots):
    """Shift that puts a tiled dataset at its absolute stage position.

    A ScanR stage position is the centre of a field, and
    ``tifs_to_bdv.py`` writes tile positions relative to the first; the
    absolute ones come from matching stage-coordinate spots (a
    ``positions.xml``) to the tiles, or from the tiles' ``<location>``
    when it holds absolute positions (newer ``tifs_to_bdv.py``).

    Parameters
    ----------
    ref : Source
        The tiled dataset.
    spots : list of dict or None
        Spots in stage µm, e.g. from ``positions.xml``.

    Returns
    -------
    tuple of float or None
        x, y shift in µm, None if no absolute position is known.
    source : str
    """
    tiles = tile_geometry(ref.path)
    if not tiles:
        return None, ""
    sc = ref.scale
    pairs = []
    if spots:
        mapped = map_spots_to_tiles(spots, tiles)
        if mapped:
            centre = {t["name"]: t["center"] for t in tiles}
            for p, m in zip(spots, mapped, strict=True):
                if m["tile"]:
                    c = centre[m["tile"]]
                    pairs.append(((p["x"], p["y"]), (c[0] * sc, c[1] * sc)))
            source = "positions.xml"
    # Relative locations start at 0; absolute (stage) ones do not.
    absolute = any(abs(min(t["loc"][i] for t in tiles)) > 1e-6 for i in (0, 1))
    if not pairs and absolute:
        for t in tiles:
            pairs.append((t["loc"], (t["center"][0] * sc, t["center"][1] * sc)))
        source = "tile locations"
    if not pairs:
        return None, ""
    dx = sum(a[0] - b[0] for a, b in pairs) / len(pairs)
    dy = sum(a[1] - b[1] for a, b in pairs) / len(pairs)
    return (dx, dy), source


def registrations(root):
    """World transform of each setup at its first timepoint.

    BDV lists a view's transforms outermost first, so the world
    transform is their product in document order.

    Parameters
    ----------
    root : xml.etree.ElementTree.Element
        The ``SpimData`` element.

    Returns
    -------
    dict of int to tuple of float
        Setup id to its row-packed 3x4 affine.
    """
    found: dict[int, tuple[int, tuple]] = {}
    for reg in root.iter("ViewRegistration"):
        sid, tp = int(reg.get("setup")), int(reg.get("timepoint"))
        if sid in found and found[sid][0] <= tp:
            continue
        m = IDENTITY
        for vt in reg.findall("ViewTransform"):
            values = [float(v) for v in (vt.findtext("affine") or "").split()]
            if len(values) == 12:
                m = compose(m, values)
        found[sid] = (tp, m)
    return {sid: m for sid, (_, m) in found.items()}


def world_z_range(m, size):
    """Lowest and highest world z of an image box under an affine.

    Parameters
    ----------
    m : sequence of float
        Row-packed 3x4 affine.
    size : tuple of str
        Image size x, y, z.

    Returns
    -------
    tuple of float
    """
    try:
        dims = [max(int(v) - 1, 0) for v in size]
    except ValueError:
        dims = [0, 0, 0]
    zs = [
        m[8] * x + m[9] * y + m[10] * z + m[11]
        for x in (0, dims[0])
        for y in (0, dims[1])
        for z in (0, dims[2])
    ]
    return min(zs), max(zs)


UNIT_TO_UM = {
    "µm": 1.0,
    "μm": 1.0,
    "um": 1.0,
    "micron": 1.0,
    "nm": 1e-3,
    "mm": 1e3,
}


def um_per_world(m, voxel, unit):
    """Size of one world unit in µm, for a setup.

    A pixel is ``voxel[0]`` µm wide and the registration maps it to
    ``|m[:, 0]|`` world units, so a world unit is their ratio.

    Parameters
    ----------
    m : sequence of float
        The setup's row-packed 3x4 registration.
    voxel : list of str
        Voxel size x, y, z from the XML.
    unit : str
        Unit of the voxel size.

    Returns
    -------
    float
        µm per world unit; 1 if the voxel size is unknown.
    """
    try:
        vx = float(voxel[0]) * UNIT_TO_UM.get(unit, 1.0)
    except (IndexError, ValueError):
        return 1.0
    norm = (m[0] ** 2 + m[4] ** 2 + m[8] ** 2) ** 0.5
    return vx / norm if norm else 1.0


def read_setups(xml_path: Path):
    """Read the view setups of a BDV XML.

    Parameters
    ----------
    xml_path : Path
        A BigDataViewer / BigStitcher XML.

    Returns
    -------
    list of SetupInfo or None
        The setups in id order, as MoBIE indexes them; None if the file
        is not a BDV XML.
    """
    try:
        root = ET.parse(xml_path).getroot()
    except ET.ParseError:
        return None
    if root.tag != "SpimData":
        return None
    vs = root.find("SequenceDescription/ViewSetups")
    if vs is None:
        return None
    regs = registrations(root)

    # Attribute id -> name, per attribute kind (tile, channel...).
    attr_names: dict[str, dict[str, str]] = {}
    for attrs in vs.findall("Attributes"):
        kind = attrs.get("name", "")
        attr_names[kind] = {
            (el.findtext("id") or "").strip(): (
                el.findtext("name") or el.findtext("id") or ""
            ).strip()
            for el in attrs
        }

    setups = []
    for el in vs.findall("ViewSetup"):
        sid = int(el.findtext("id"))
        attrs = el.find("attributes")

        def attr(kind, attrs=attrs):
            if attrs is None:
                return ""
            value = (attrs.findtext(kind) or "").strip()
            return attr_names.get(kind, {}).get(value, value)

        size = (el.findtext("size") or "").split()
        voxel = (el.findtext("voxelSize/size") or "").split()
        size = tuple(size) if len(size) == 3 else ("", "", "")
        setups.append(
            (
                sid,
                SetupInfo(
                    index=0,
                    name=(el.findtext("name") or "").strip() or f"setup{sid}",
                    tile=attr("tile"),
                    channel=attr("channel"),
                    unit=(el.findtext("voxelSize/unit") or "").strip(),
                    voxel=tuple(voxel) if len(voxel) == 3 else ("", "", ""),
                    size=size,
                    z_range=world_z_range(regs.get(sid, IDENTITY), size),
                    reg=regs.get(sid, IDENTITY),
                    um_per_world=um_per_world(
                        regs.get(sid, IDENTITY),
                        voxel,
                        (el.findtext("voxelSize/unit") or "").strip(),
                    ),
                ),
            )
        )
    setups.sort(key=lambda t: t[0])
    for i, (_, s) in enumerate(setups):
        s.index = i
    return [s for _, s in setups]


def check_image_file(xml_path: Path):
    """Check that the data file a BDV XML points to exists.

    MoBIE opens the file named in the XML's ``ImageLoader`` (``<hdf5>``
    or ``<n5>``), not the one that happens to sit next to the XML, so a
    data file renamed after the XML was written fails only in MoBIE.

    Parameters
    ----------
    xml_path : Path
        A BDV XML.

    Returns
    -------
    str or None
        A description of the problem, None if the file exists or the
        loader is not file based.
    """
    root = ET.parse(xml_path).getroot()
    loader = root.find("SequenceDescription/ImageLoader")
    if loader is None:
        return None
    el = next((e for e in loader if e.tag in ("hdf5", "n5", "zarr")), None)
    if el is None or not (el.text or "").strip():
        return None
    ref = Path(el.text.strip().replace("\\", "/"))
    if el.get("type", "relative") == "relative":
        base = xml_path.parent
        bp = root.find("BasePath")
        if bp is not None and bp.get("type", "relative") == "relative":
            base = base / (bp.text or ".").strip()
        target = base / ref
    else:
        target = ref
    if target.exists():
        return None
    others = sorted(p.name for p in xml_path.parent.glob(f"*{ref.suffix}"))
    msg = (
        f"{xml_path.name} points to <{el.tag}> {el.text.strip()}, which "
        f"does not exist ({target})."
    )
    if others:
        msg += (
            f" {ref.suffix} files next to it: {', '.join(others)}. "
            f"Edit the <{el.tag}> line of the XML to the right name."
        )
    return msg


# --------------------------------------------------------------------------
# Spots
# --------------------------------------------------------------------------


def _key(name: str) -> str:
    return re.sub(r"[\s_\-.()\[\]]", "", name.rsplit("}", 1)[-1].lower())


def read_positions(path: Path):
    """Read spot coordinates from a ``positions.xml``.

    The format is not fixed, so every element is checked for x and y
    values, as attributes (``<Spot x="1" y="2"/>``), as child elements
    (``<Position><X>1</X><Y>2</Y></Position>``) or as LabVIEW name/value
    pairs, as ScanR writes them (``<Cluster><DBL><Name>X</Name>
    <Val>1</Val></DBL>...``), under the usual names (``x``, ``PosX``,
    ``PositionX``, ``X_um``, ...). The element type with the most such
    matches holds the spots.

    Parameters
    ----------
    path : Path
        The ``positions.xml``.

    Returns
    -------
    list of dict
        ``spot_id``, ``x``, ``y`` and ``z`` (None if absent) per spot,
        and ``extra``: the element's other values (e.g. ``well``), kept
        as extra columns of the spots table.
    """
    try:
        root = ET.parse(path).getroot()
    except ET.ParseError as e:
        sys.exit(f"{path}: not readable as XML ({e})")

    def local(el):
        return el.tag.rsplit("}", 1)[-1]

    def values(el):
        found = {"extra": {}}
        ids = {}
        items = [(k, v) for k, v in el.attrib.items()]
        for c in el:
            if len(c) == 0:
                items.append((c.tag, c.text or ""))
                continue
            # LabVIEW clusters: <DBL><Name>X</Name><Val>1.5</Val></DBL>
            sub = {local(g): (g.text or "") for g in c}
            if "Name" in sub and "Val" in sub:
                items.append((sub["Name"], sub["Val"]))
        for k, v in items:
            raw, k, v = k, _key(k), v.strip()
            is_coord = False
            for axis, rx in COORD_RE.items():
                if axis not in found and rx.match(k):
                    try:
                        found[axis] = float(v.replace(",", "."))
                        is_coord = True
                    except ValueError:
                        pass
            if is_coord:
                continue
            if k in ID_KEYS:
                ids[k] = v
            if v and len(v) < 64:
                found["extra"][_key(raw)] = v
        # Most specific id name first: "id" before "position" before "name".
        for k in ID_KEYS:
            if ids.get(k):
                found["spot_id"] = ids[k]
                break
        return found

    by_tag: dict[str, list[dict]] = {}
    for el in root.iter():
        v = values(el)
        if "x" in v and "y" in v:
            by_tag.setdefault(el.tag, []).append(v)
    if not by_tag:
        return []
    spots = max(by_tag.values(), key=len)
    # Drop values shared by all spots (element names, counts...).
    if len(spots) > 1:
        keys = {k for p in spots for k in p["extra"]}
        for k in keys:
            if len({p["extra"].get(k) for p in spots}) == 1:
                for p in spots:
                    p["extra"].pop(k, None)
    for i, s in enumerate(spots, 1):
        s.setdefault("z", None)
        # MoBIE needs numeric, unique spot ids.
        sid = s.get("spot_id", "")
        s["spot_id"] = sid if sid.isdigit() else str(i)
    if len({s["spot_id"] for s in spots}) < len(spots):
        for i, s in enumerate(spots, 1):
            s["spot_id"] = str(i)
    return spots


# --------------------------------------------------------------------------
# Table
# --------------------------------------------------------------------------


def channel_color(channel: str) -> str:
    """MoBIE color for a channel, from the wavelength in its name.

    Parameters
    ----------
    channel : str
        Channel name, e.g. ``488nm``.

    Returns
    -------
    str
        ``Blue``, ``Green``, ``Red`` or ``Magenta`` by wavelength;
        ``White`` if the name holds no plausible wavelength.
    """
    m = re.search(r"\d{3}", channel)
    if not m or not 300 <= int(m.group()) <= 1000:
        return "White"
    nm = int(m.group())
    return next(color for limit, color in WAVELENGTH_COLORS if nm < limit)


def view_names(paths: list[Path], level: int = 1):
    """Name each dataset after a folder above it, adding detail if needed.

    Parameters
    ----------
    paths : list of Path
        The BDV XMLs and positions files.
    level : int, optional
        Which folder names the view: 1 is the folder holding the file,
        2 its parent, and so on.

    Returns
    -------
    dict of Path to str
        The folder name. Where two views would share it, the folders
        from there down to the file (``SampleA_Acquisition``), and the
        file name only if that still clashes.
    """

    def folder(x, lvl):
        parents = x.parents
        return parents[min(lvl, len(parents)) - 1].name

    def dedupe(names, longer):
        counts: dict[str, int] = {}
        for n in names.values():
            counts[n] = counts.get(n, 0) + 1
        return {x: longer(x) if counts[n] > 1 else n for x, n in names.items()}

    names = {x: folder(x, level) for x in paths}
    # Clashes: spell out the folders from the naming one down to the file
    # (SampleA_Acquisition), then, if still needed, add the file name.
    below = {
        x: "_".join(x.parent.parts[len(x.parent.parts) - level :])
        for x in paths
    }
    names = dedupe(names, lambda x: below[x])
    return dedupe(names, lambda x: f"{below[x]}_{x.stem}")


def tile_geometry(xml_path: Path):
    """Where each tile of a BDV XML sits, in µm and in world coordinates.

    Parameters
    ----------
    xml_path : Path
        A BDV XML whose tiles carry a ``<location>``, as the ones
        ``tifs_to_bdv.py`` writes from the stage positions.

    Returns
    -------
    list of dict
        Per tile: ``name``, ``loc`` (x, y of its ``<location>``),
        ``corner`` and ``center`` (x, y in world coordinates, through its
        registration). Empty if the tiles have no location.
    """
    root = ET.parse(xml_path).getroot()
    regs = registrations(root)
    locs, names = {}, {}
    for t in root.iter("Tile"):
        tid = (t.findtext("id") or "").strip()
        loc = (t.findtext("location") or "").split()
        names[tid] = (t.findtext("name") or tid).strip()
        if len(loc) >= 2:
            locs[tid] = (float(loc[0]), float(loc[1]))
    tiles, seen = [], set()
    for vs in root.iter("ViewSetup"):
        tid = (vs.findtext("attributes/tile") or "").strip()
        if tid not in locs or tid in seen:
            continue
        seen.add(tid)
        m = regs.get(int(vs.findtext("id")), IDENTITY)
        sx, sy = (int(v) for v in vs.findtext("size").split()[:2])

        def world(x, y, m=m):
            return (m[0] * x + m[1] * y + m[3], m[4] * x + m[5] * y + m[7])

        tiles.append(
            {
                "name": names[tid],
                "loc": locs[tid],
                "corner": world(0, 0),
                "center": world(sx / 2, sy / 2),
            }
        )
    return tiles


def map_spots_to_tiles(spots, tiles):
    """Map stage coordinates onto the tiles they were acquired at.

    The tile locations ``tifs_to_bdv.py`` writes are the stage positions
    minus their minimum, possibly with axes flipped or swapped. Every
    orientation is tried; for each, the offset that makes the most spots
    land exactly on a tile location wins. The spots then go to the
    centre of their tile in world coordinates -- a ScanR stage position
    is the centre of the field, while the tile is placed by its corner.

    Parameters
    ----------
    spots : list of dict
        Spots with stage ``x`` and ``y`` (µm).
    tiles : list of dict
        Output of `tile_geometry`.

    Returns
    -------
    list of dict or None
        Per spot: ``x``, ``y`` (world) and ``tile``; None if fewer than
        half the spots match a tile, i.e. the spots are not tile
        positions.
    """
    if len(tiles) < 2:
        return None
    by_loc = {(round(t["loc"][0], 1), round(t["loc"][1], 1)): t for t in tiles}
    best = (0, None, None)
    for swap in (False, True):
        for fx in (1, -1):
            for fy in (1, -1):

                def orient(p, swap=swap, fx=fx, fy=fy):
                    a, b = (p["y"], p["x"]) if swap else (p["x"], p["y"])
                    return fx * a, fy * b

                votes: dict[tuple, int] = {}
                for p in spots:
                    a, b = orient(p)
                    for lx, ly in by_loc:
                        key = (round(lx - a, 1), round(ly - b, 1))
                        votes[key] = votes.get(key, 0) + 1
                offset, n = max(votes.items(), key=lambda kv: kv[1])
                if n > best[0]:
                    best = (n, orient, offset)
    n, orient, (ox, oy) = best
    if n < max(2, len(spots) / 2):
        return None

    # Spots on a tile go to its centre; others (if any) are placed with
    # the location -> world scale fitted on the tiles.
    t0, t1 = (
        tiles[0],
        max(tiles, key=lambda t: abs(t["loc"][0] - tiles[0]["loc"][0])),
    )
    span = t1["loc"][0] - t0["loc"][0]
    scale = (t1["corner"][0] - t0["corner"][0]) / span if span else 1.0
    out = []
    for p in spots:
        a, b = orient(p)
        lx, ly = round(a + ox, 1), round(b + oy, 1)
        t = by_loc.get((lx, ly))
        if t is not None:
            out.append(
                {"x": t["center"][0], "y": t["center"][1], "tile": t["name"]}
            )
        else:
            half_x = t0["center"][0] - t0["corner"][0]
            half_y = t0["center"][1] - t0["corner"][1]
            out.append(
                {
                    "x": t0["corner"][0] + (lx - t0["loc"][0]) * scale + half_x,
                    "y": t0["corner"][1] + (ly - t0["loc"][1]) * scale + half_y,
                    "tile": "",
                }
            )
    return out


def match_sources(sources, key: str):
    """Datasets named by ``key``.

    The dataset's own folder name is compared first, exactly (``Scan``
    is the ``Scan`` folder, not ``ScanR/Acquisition``); only if none
    matches, any view name containing ``key``.

    Parameters
    ----------
    sources : list of Source
    key : str
        Case-insensitive name.

    Returns
    -------
    list of Source
    """
    key = key.lower()
    exact = [s for s in sources if s.path.parent.name.lower() == key]
    return exact or [s for s in sources if key in s.view.lower()]


def pick_reference(sources, wanted: str | None):
    """The 3D dataset 2D data of a sample is centred on.

    Parameters
    ----------
    sources : list of Source
        Datasets of one sample.
    wanted : str or None
        Case-insensitive part of the reference's view name; by default
        "scan", and the only 3D dataset if none matches.

    Returns
    -------
    Source or None
        None if the sample has no 3D dataset.
    """
    stacks = [s for s in sources if s.kind == "intensities" and s.is_3d]
    if not stacks:
        return None
    named = match_sources(stacks, wanted or "scan")
    if named:
        return named[0]
    if wanted:
        print(
            f"WARNING: no 3D dataset matching '{wanted}' in {stacks[0].sample}"
        )
    return stacks[0]


def fmt(v: float) -> str:
    """Compact number for the table."""
    return f"{v:.6g}"


def main(argv=None):
    """Run from the command line.

    Parameters
    ----------
    argv : list of str, optional
        Arguments; defaults to ``sys.argv[1:]``.
    """
    p = argparse.ArgumentParser(
        description=__doc__.split("\n\n")[0],
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    p.add_argument("root", type=Path, help="folder to search for BDV XMLs")
    p.add_argument(
        "-o",
        "--output",
        type=Path,
        help="table to write, .tsv or .csv (default: <root>/mobie_collection.tsv)",
    )
    p.add_argument(
        "--name-level",
        type=int,
        default=1,
        help="folder that names a view: 1 = the file's folder (default), "
        "2 = its parent, ...",
    )
    p.add_argument(
        "--absolute",
        action="store_true",
        help="write absolute paths instead of paths relative to the table",
    )
    p.add_argument(
        "--contrast-limits",
        metavar="MIN,MAX",
        help="initial contrast for every channel, e.g. 100,5000, or "
        "'auto' (default: empty, MoBIE's own default)",
    )
    p.add_argument(
        "--reference",
        metavar="NAME",
        help="3D dataset that 2D datasets are centred on, by part of its "
        "view name (default: the one with 'scan' in it, else the only one)",
    )
    p.add_argument(
        "--offset",
        action="append",
        default=[],
        metavar="NAME=DX,DY[,DZ]",
        help="shift the datasets whose view name contains NAME by DX, DY "
        "(and DZ) µm, e.g. Scan=120,-35; repeatable",
    )
    p.add_argument(
        "--flip-x",
        action="append",
        metavar="NAME",
        help="mirror the dataset(s) in folder NAME in x; repeatable "
        "(default: Scan)",
    )
    p.add_argument(
        "--flip-y",
        action="append",
        default=[],
        metavar="NAME",
        help="mirror the dataset(s) in folder NAME in y; repeatable",
    )
    p.add_argument(
        "--spot-radius",
        type=float,
        default=10.0,
        help="radius of spots from positions.xml (default 10)",
    )
    p.add_argument(
        "--spot-scale",
        type=float,
        default=1.0,
        help="multiply spot coordinates by this, to match the images' "
        "space (default 1)",
    )
    p.add_argument(
        "--dry-run",
        action="store_true",
        help="list the views and preview the table, write nothing",
    )
    args = p.parse_args(argv)

    root = args.root.resolve()
    if not root.is_dir():
        sys.exit(f"Not a folder: {root}")
    out = (args.output or root / "mobie_collection.tsv").resolve()
    delimiter = "," if out.suffix.lower() == ".csv" else "\t"

    # Collect the BDV XMLs; then positions.xml and ScanR metadata.ome.xml
    # files, which become spots when alone in a folder.
    sources: list[Source] = []
    for x in sorted(root.rglob("*.xml")):
        setups = read_setups(x)
        if setups:
            sources.append(Source(path=x, kind="intensities", setups=setups))
    bdv_folders = {s.path.parent for s in sources}
    metas: dict[Path, dict] = {}
    for x in sorted(root.rglob("*")):
        name = x.name.lower()
        if name == "metadata.ome.xml":
            # A TIFF folder's own metadata is not a detection list.
            if any(x.parent.glob("*.tif")):
                continue
            meta = read_scanr_meta(x)
            if meta is None:
                continue
            metas[x.parent] = meta
            if x.parent not in bdv_folders:
                spots = [
                    {
                        "spot_id": str(i),
                        "x": px,
                        "y": py,
                        "z": None,
                        "extra": {},
                    }
                    for i, (px, py) in enumerate(meta["positions"], 1)
                ]
                src = Source(path=x, kind="spots", spots=spots, meta=meta)
                src.table = x.with_name("detection_positions.tsv")
                sources.append(src)
        elif name == "positions.xml" and x.parent not in bdv_folders:
            spots = read_positions(x)
            if not spots:
                print(f"WARNING: no x/y coordinates recognised in {x}")
                continue
            src = Source(path=x, kind="spots", spots=spots)
            src.table = x.with_name("positions_spots.tsv")
            sources.append(src)
    if not sources:
        sys.exit(f"No BDV XML, positions.xml or metadata.ome.xml below {root}")

    names = view_names([s.path for s in sources], args.name_level)
    for s in sources:
        s.view = names[s.path]
        s.sample = s.path.parent.parent
        if s.kind == "intensities":
            s.meta = metas.get(s.path.parent)
    for spec in args.offset:
        key, _, values = spec.partition("=")
        try:
            vals = [float(v) for v in values.split(",")]
        except ValueError:
            sys.exit(f"--offset {spec}: expected NAME=DX,DY[,DZ]")
        vals = (vals + [0.0, 0.0, 0.0])[:3]
        hits = match_sources(sources, key)
        if not hits:
            sys.exit(f"--offset {spec}: no dataset named like '{key}'")
        for s in hits:
            s.offset = tuple(vals)
    flip_x = args.flip_x if args.flip_x is not None else ["Scan"]
    for key, axis in [(k, 0) for k in flip_x] + [(k, 1) for k in args.flip_y]:
        for s in match_sources(sources, key):
            f = list(s.flip)
            f[axis] = True
            s.flip = tuple(f)

    # Per sample, everything goes into stage coordinates (µm): the tiled
    # 3D reference at its stage position, stitched ScanR grids on their
    # grid, spots as they are. 2D data and spots go to mid-stack.
    by_sample: dict[Path, list[Source]] = {}
    for s in sources:
        by_sample.setdefault(s.sample, []).append(s)
    notes = []
    for members in by_sample.values():
        ref = pick_reference(members, args.reference)
        stage_spots = [
            p
            for s in members
            if s.kind == "spots" and s.meta is None
            for p in s.spots
        ]
        sample_metas = [s.meta for s in members if s.meta is not None]
        in_stage = False
        if ref is not None:
            shift, how = stage_offset(ref, stage_spots)
            if shift is not None:
                o = ref.offset
                ref.offset = (o[0] + shift[0], o[1] + shift[1], o[2])
                in_stage = True
                notes.append(
                    f"{ref.view}: at its stage position (from {how}), "
                    f"shifted by ({fmt(shift[0])}, {fmt(shift[1])}) µm"
                )
        mid = None
        if ref is not None:
            lo, hi = ref.z_range_um
            mid = (lo + hi) / 2
        for s in members:
            if s.kind != "intensities" or s.is_3d or s is ref:
                continue
            meta = s.meta or (
                sample_metas[0] if len(sample_metas) == 1 else None
            )
            if meta is not None:
                s.matrix, note = place_on_grid(s, meta, s.flip)
                notes.append(note)
                placed_z = 0.0
                if not in_stage and ref is not None:
                    notes.append(
                        f"WARNING: {ref.view} has no absolute stage position "
                        f"(no positions.xml, relative tile locations), so "
                        f"{s.view} will not line up with it. Rewrite its XML "
                        "with tifs_to_bdv.py --xml-only."
                    )
            else:
                placed_z = s.z_range_um[0]
            if mid is not None:
                s.dz = mid - placed_z
                notes.append(
                    f"{s.view}: 2D, at z {fmt(mid)} µm, the middle of {ref.view}"
                )
        for s in members:
            if s.kind != "spots":
                continue
            if not (in_stage or sample_metas) and ref is not None:
                # No stage frame: map stage positions onto the tiles.
                mapped = map_spots_to_tiles(s.spots, tile_geometry(ref.path))
                if mapped is not None:
                    sc, (ox, oy, _) = ref.scale, ref.offset
                    for p, m in zip(s.spots, mapped, strict=True):
                        p["x"], p["y"] = m["x"] * sc + ox, m["y"] * sc + oy
                        p["extra"]["tile"] = m["tile"]
            elif args.spot_scale != 1:
                for p in s.spots:
                    p["x"] *= args.spot_scale
                    p["y"] *= args.spot_scale
            if mid is not None:
                # Shown in the middle of the stack, whatever z they had.
                for p in s.spots:
                    p["z"] = mid
                notes.append(
                    f"{s.view}: {len(s.spots)} spots at z {fmt(mid)} µm, the "
                    f"middle of {ref.view}"
                )

    contrast = ""
    if args.contrast_limits == "auto":
        contrast = "auto"
    elif args.contrast_limits:
        lo, hi = (float(v) for v in args.contrast_limits.split(","))
        contrast = f"({lo:g},{hi:g})"

    def rel(path):
        if args.absolute:
            return path.as_posix()
        try:
            return path.relative_to(out.parent).as_posix()
        except ValueError:
            sys.exit(
                f"{path} is not below the table's folder {out.parent}; "
                "pass --absolute or write the table higher up."
            )

    def view_of(src):
        # The dataset's own folder (Acquisition, Scan, Detection...); all
        # share stage coordinates, so views can be shown together.
        return src.path.parent.name

    rows = []
    problems = []
    spot_tables = {}
    for src in sources:
        print(f"  {src.view}  <- {src.path.relative_to(root)}")
        if src.kind == "spots":
            table = src.table
            spot_tables[table] = src.spots
            xs = [p["x"] for p in src.spots]
            ys = [p["y"] for p in src.spots]
            print(
                f"      {len(src.spots)} spots, x {fmt(min(xs))}..{fmt(max(xs))}"
                f", y {fmt(min(ys))}..{fmt(max(ys))}, radius "
                f"{fmt(args.spot_radius)} -> {table.name}"
            )
            rows.append(
                {
                    "uri": rel(table),
                    "type": "spots",
                    "name": src.view,
                    "view": view_of(src),
                    "display": src.view,
                    "spot_radius": fmt(args.spot_radius),
                    "sample": src.sample.name,
                }
            )
            continue

        problem = check_image_file(src.path)
        if problem:
            problems.append(problem)
        affine = src.affine()
        for s in src.setups:
            rows.append(
                {
                    "uri": rel(src.path),
                    "type": "intensities",
                    "name": f"{src.view}_{s.name}",
                    "channel_name": s.channel,
                    # MoBIE reads "um"; BDV XMLs often say "µm".
                    "pixel_unit": "um" if s.unit in ("µm", "μm") else s.unit,
                    "pixel_size_x": s.voxel[0],
                    "pixel_size_y": s.voxel[1],
                    "pixel_size_z": s.voxel[2],
                    "num_pixels_x": s.size[0],
                    "num_pixels_y": s.size[1],
                    "num_pixels_z": s.size[2],
                    "channel": s.index,
                    "view": view_of(src),
                    "display": f"{src.view}_{s.channel}"
                    if s.channel
                    else src.view,
                    "color": channel_color(s.channel),
                    "contrast_limits": contrast,
                    "affine": affine,
                    "sample": src.sample.name,
                    "tile": s.tile,
                }
            )
        tiles = {st.tile for st in src.setups}
        chans = dict.fromkeys(st.channel for st in src.setups)
        lo, hi = src.z_range_um
        print(
            f"      {len(src.setups)} rows: {len(tiles)} tile(s) x "
            f"{len(chans)} channel(s): "
            + ", ".join(f"{c or 'unnamed'} ({channel_color(c)})" for c in chans)
            + f"; {'3D' if src.is_3d else '2D'}, z {fmt(lo)}..{fmt(hi)} µm, "
            f"{fmt(src.scale)} µm per world unit"
        )

    if notes:
        print()
    for note in notes:
        print(f"  {note}")
    for problem in problems:
        print(f"\nWARNING: {problem}")
    if problems:
        print("MoBIE will fail to open these views until that is fixed.")

    n_views = len(sources)
    if args.dry_run:
        print(
            f"\nDry run: would write {out} ({n_views} views, {len(rows)} rows)"
            + (", replacing the existing file" if out.exists() else "")
            + (
                f", and {len(spot_tables)} spots table(s)"
                if spot_tables
                else ""
            )
            + "."
        )
        preview = 5
        print(f"First {min(preview, len(rows))} rows:")
        writer = csv.DictWriter(sys.stdout, COLUMNS, delimiter=delimiter)
        writer.writeheader()
        writer.writerows(rows[:preview])
        return

    for table, spots in spot_tables.items():
        extras = list(
            dict.fromkeys(
                k
                for p in spots
                for k in p["extra"]
                if k not in ("spotid", "x", "y", "z")
            )
        )
        with open(table, "w", newline="", encoding="utf-8") as f:
            writer = csv.writer(f, delimiter="\t")
            writer.writerow(["spot_id", "x", "y", "z", *extras])
            for p in spots:
                writer.writerow(
                    [p["spot_id"], fmt(p["x"]), fmt(p["y"]), fmt(p["z"] or 0)]
                    + [p["extra"].get(k, "") for k in extras]
                )
        print(f"Wrote {table} ({len(spots)} spots)")

    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, COLUMNS, delimiter=delimiter)
        writer.writeheader()
        writer.writerows(rows)
    data_root = "PathsInTableAreAbsolute" if args.absolute else "UseTableFolder"
    print(
        f"Wrote {out} ({n_views} views, {len(rows)} rows).\n"
        "Open in Fiji: Plugins > MoBIE > Open > Open Collection Table..., "
        f"Data Root: {data_root}, Viewing mode: ThreeDimensional."
    )


if __name__ == "__main__":
    main()
