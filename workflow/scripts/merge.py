"""Snakemake script: stitch the segmented tiles into one label image.

Joins objects across tile seams, renumbers them, applies the size filter,
and writes ``image.zarr/labels/<name>/``: a calibrated pyramid with its
object table, provenance and seam report.
"""

import json
import shutil
from pathlib import Path

import numpy as np
import zarr
from patchworks import (
    capped_output_chunks,
    cpu_allocation,
    merge_tile_labels,
    provenance,
    safe_worker_count,
    set_compression,
)
from patchworks._chunks import _get_available_memory
from patchworks._provenance import write_provenance
from patchworks._volume_filter import (
    filter_labels_by_size,
    max_voxels_for_volume,
    min_voxels_for_volume,
)
from patchworks.plugins.ome_zarr import (
    read_pixel_size,
    register_labels,
    reshard_level,
)

from _pw import (
    halo_path,
    load_tiles_json,
    merge_connectivity,
    parts_path,
    stage_path,
    start_log,
)

start_log(snakemake.log[0])  # noqa: F821
set_compression(snakemake.config.get("compression", "zstd"))  # noqa: F821
cfg = snakemake.config  # noqa: F821
work_dir = cfg["work_dir"]
label_name = cfg.get("label_name", "labels")
# Level 0 chunks stay viewer-friendly however large the tiles are.
LABEL_CHUNK_CAP = (16, 1024, 1024)
image_store = str(Path(work_dir) / "image.zarr")
label_group = f"{image_store}/labels/{label_name}"

# Where the segment jobs wrote: labels/<name>/0 itself, or a scratch store.
manifest = load_tiles_json(snakemake.input.tiles)  # noqa: F821
target_path = manifest["target_path"]
target_component = manifest.get("target_component", "staged")
in_place = bool(manifest.get("in_place", False))


def _finished(group: str) -> bool:
    """Whether the label group is a finished label image (its pyramid
    registered: the merge got that far)."""
    try:
        attrs = zarr.open_group(group, mode="r").attrs.asdict()
    except Exception:  # noqa: BLE001 - absent or unreadable: not finished
        return False
    return bool(
        attrs.get("multiscales") or (attrs.get("ome") or {}).get("multiscales")
    )


def write_seam_report(group, record=None) -> None:
    # Did the tiling leave marks? Objects ending on seams vs inside tiles.
    if cfg.get("seam_report", True):
        from patchworks import seam_report

        report = seam_report(
            group,
            manifest["tile_shape"],
            max_faces=int(cfg.get("seam_report_faces", 64)),
        )
        Path(work_dir, label_name, "seams.json").write_text(
            json.dumps(report, indent=2)
        )
        seams = {
            ax: {k: row[k] for k in ("seam_rate", "interior_rate", "ratio")}
            for ax, row in report["axes"].items()
        }
        if record is not None:
            record["seams"] = seams
            write_provenance(zarr.open_group(str(group), mode="r+"), record)
        for ax, row in report["axes"].items():
            interior = row["interior_rate"]
            print(
                f"[patchworks] seams axis {ax}: {100 * row['seam_rate']:.1f}% of "
                f"labels end on a seam vs "
                f"{'n/a' if interior is None else f'{100 * interior:.1f}%'} "
                "inside tiles"
            )


def write_object_table(group, objects: dict) -> None:
    # The object table, written after the labels are registered so relate
    # finds it current. Intensities need the image, so they are measured.
    if cfg.get("object_table", True):
        from patchworks._tables import compute_table, write_table_from_sums

        if objects and not cfg.get("table_channels"):
            cols = write_table_from_sums(image_store, label_name, objects)
            how = "from the tiles' sums"
        else:
            cols = compute_table(
                image_store,
                label_name,
                channels=cfg.get("table_channels") or None,
            )
            how = "measured from the labels"
        print(
            f"[patchworks] object table: {len(cols['label']):,} objects in "
            f"{group}/table ({how})"
        )


def finish(group, record=None, objects=None) -> None:
    """Everything after the labels are registered; the scratch store is
    removed last, so a merge stopped before the end can simply be rerun."""
    write_seam_report(group, record)
    write_object_table(group, objects or {})
    shutil.rmtree(parts_path(work_dir, label_name), ignore_errors=True)
    if not in_place:
        # The scratch store and its marker go together, or a rerun would
        # segment into a store that no longer exists.
        shutil.rmtree(stage_path(work_dir, label_name), ignore_errors=True)
        Path(f"{stage_path(work_dir, label_name)}.done").unlink(missing_ok=True)
    print(f"[patchworks] labels written to {group}")
    open(snakemake.output[0], "w").close()  # noqa: F821


# A merge stopped after removing the scratch store (before this order was
# fixed): its labels are written if their pyramid is; finish the rest. If not,
# the segmented tiles are gone and only redoing the segmentation helps.
if not in_place and not Path(target_path).exists():
    if _finished(label_group):
        print(
            "[patchworks] merge: an earlier attempt wrote the labels and "
            "removed its stage store, then stopped before the end; "
            "finishing (seam report, object table)"
        )
        finish(label_group)
        raise SystemExit(0)
    raise RuntimeError(
        f"{target_path} (the segmented tiles) is gone and {label_group} is "
        "not finished: segment again -- delete "
        f"{Path(work_dir) / label_name} and {label_group}, then launch again"
    )

staged = zarr.open_group(target_path, mode="r")[target_component]

# Workers bounded by this job's CPUs and memory (each holds a few chunks).
chunk_nbytes = int(np.prod(staged.chunks)) * staged.dtype.itemsize
default_workers = min(
    cpu_allocation(), safe_worker_count(chunk_nbytes, fn_overhead=3)
)
print(
    f"[patchworks] merge: {cpu_allocation()} cpu(s), "
    f"{_get_available_memory() / 1024**3:.0f} GiB budget, "
    f"{default_workers} worker(s) for {chunk_nbytes / 1024**2:.0f} MB chunks"
)
# Labels per tile, from the segment jobs' markers: global ids follow by a
# cumulative sum. The paths are built here rather than taken from
# snakemake.input, which is a placeholder once the stage store is deleted.
label_counts = {}
seg_dir = Path(work_dir) / label_name / "seg"
for batch in range(len(manifest["batches"])):
    marker = seg_dir / f"{batch}.done"
    for index, n in json.loads(marker.read_text())["counts"].items():
        label_counts[int(index)] = int(n)

if in_place:
    out_chunks = None
else:
    root = zarr.open_group(image_store, mode="a")
    parent = root.require_group("labels")
    if label_name in parent:
        del parent[label_name]
    parent.require_group(label_name)
    out_chunks = capped_output_chunks(staged.chunks, LABEL_CHUNK_CAP)

# The size filter, in voxels at the segmented level's resolution.
min_volume = cfg.get("min_volume")
max_volume = cfg.get("max_volume")
min_voxels = max_voxels = None
if min_volume or max_volume:
    voxel_size = read_pixel_size(image_store, level=int(cfg.get("level", 0)))
    if not voxel_size:
        raise RuntimeError(
            f"min_volume/max_volume filtering needs calibration in "
            f"{image_store}, which has none -- set both to null, or make "
            "sure the source carries a pixel size at conversion time"
        )
    min_voxels = (
        min_voxels_for_volume(min_volume, voxel_size) if min_volume else None
    )
    max_voxels = (
        max_voxels_for_volume(max_volume, voxel_size) if max_volume else None
    )

# The segment jobs measured each tile's objects. With every tile's sums
# there, the merge builds the object table and applies the size filter in its
# own relabel pass; otherwise (a stage from an older version) both are done
# from the merged labels.
parts_dir = parts_path(work_dir, label_name)
fused = bool(label_counts) and all(
    (Path(parts_dir) / f"{index}.npz").exists()
    for index, n in label_counts.items()
    if n > 0
)
print(
    "[patchworks] object table: "
    + (
        "from the tiles' sums, with the merge"
        if fused
        else "measured after the merge (tiles without sums)"
    )
)
objects: dict = {}

# `shard_labels: true` shards level 0 one tile per shard as the merge writes
# it, instead of rewriting the whole level afterwards (8 min and up). Not
# when the volume filter rewrites level 0 after the merge (a stage without
# the tiles' sums): its workers write single chunks, which in a shared
# shard would be lost. A list asks for a shard shape of its own, which
# takes that second pass too.
shard_labels = cfg.get("shard_labels", False)
shard_in_merge = (
    shard_labels is True
    and not in_place
    and (fused or not (min_voxels or max_voxels))
)

_, n_objects = merge_tile_labels(
    target_path,
    write_to=label_group if not in_place else target_path,
    input_component=target_component,
    output_component="0" if not in_place else target_component,
    output_chunks=out_chunks,
    # One shard per tile, written whole by the worker that relabels it.
    shard_output=shard_in_merge,
    sequential_labels=cfg.get("sequential_labels", True),
    n_workers=cfg.get("merge_workers") or default_workers,
    progress=True,
    return_count=True,
    label_counts=label_counts,
    halo_dir=(
        halo_path(work_dir, label_name)
        if cfg.get("stitch", "touch") == "iou"
        else None
    ),
    iou_threshold=float(cfg.get("iou_threshold", 0.5)),
    connectivity=merge_connectivity(cfg),
    parts_dir=parts_dir if fused else None,
    min_voxels=min_voxels if fused else None,
    max_voxels=max_voxels if fused else None,
    objects_out=objects,
)
shutil.rmtree(halo_path(work_dir, label_name), ignore_errors=True)

# The size filter judges whole objects, never one tile's fragment, and runs
# before the pyramid so every level reflects it.
if (min_voxels or max_voxels) and objects:
    print(
        f"[patchworks] volume filter: [{min_volume or 0}, "
        f"{max_volume or 'inf'}] µm³ ([{min_voxels or 0}, "
        f"{max_voxels or 'inf'}] voxels) applied with the merge, "
        f"{len(objects['label'])} object(s) remain"
    )
elif min_voxels or max_voxels:
    n_objects, n_removed = filter_labels_by_size(
        label_group,
        "0",
        min_voxels,
        max_voxels,
        relabel=cfg.get("sequential_labels", True),
    )
    print(
        f"[patchworks] volume filter: dropped {n_removed} object(s) outside "
        f"[{min_volume or 0}, {max_volume or 'inf'}] µm³ "
        f"([{min_voxels or 0}, {max_voxels or 'inf'}] voxels), "
        f"{n_objects} remain"
    )

level0 = zarr.open_array(f"{label_group}/0", mode="r")
n_chunks = int(
    np.prod([-(-s // c) for s, c in zip(level0.shape, level0.chunks)])
)
print(
    f"[patchworks] {label_name} level 0: shape={level0.shape} "
    f"chunks={level0.chunks} -> {n_chunks:,} chunks"
)

print(
    f"[patchworks] sharding: shard={cfg.get('shard', False)!r} "
    f"shard_labels={shard_labels!r}"
)
if (not shard_labels or not cfg.get("shard")) and n_chunks > 2000:
    missing = " and ".join(
        k
        for k, on in (
            ("shard", cfg.get("shard")),
            ("shard_labels", shard_labels),
        )
        if not on
    )
    print(
        f"[patchworks] NOTE: level 0 is {n_chunks:,} chunks, and typically "
        "the great majority of this label group's files -- the pyramid "
        "levels above it shrink fourfold each. Set `" + missing + ": true` "
        "to pack them into shards: `shard` covers levels 1..N, "
        "`shard_labels` covers level 0, which is the one that counts."
    )
if shard_labels and not shard_in_merge:
    # A shard shape of its own (a list), or relabeled in place: level 0 is
    # written by many processes, so it is sharded only now, in one pass.
    spec = (cfg.get("shard") or True) if shard_labels is True else shard_labels
    reshard_level(label_group, "0", shard=spec, progress=True)

# How these labels were made (read_provenance): the whole config and what
# was decided at run time.
_PRIVATE = ("notify_email", "notify_events")
settings = {k: v for k, v in cfg.items() if k not in _PRIVATE}
settings.update(
    label_name=label_name,
    tile_shape=manifest["tile_shape"],
    overlap=manifest["overlap"],
    resolved={
        "n_tiles": manifest.get("n_tiles"),
        "tiles_segmented": len(manifest.get("occupied", [])),
        "empty_threshold": manifest.get("empty_threshold"),
        "intensity_range": manifest.get("intensity_range"),
        "thresholds": manifest.get("kwargs_overrides"),
        "min_voxels": min_voxels,
        "max_voxels": max_voxels,
        "n_objects": n_objects,
    },
)
if cfg.get("method", "cellpose") == "cellpose":
    try:
        # Not plugins.cellpose: that imports Cellpose, and torch with it.
        from patchworks.plugins._cellpose_defaults import applied_defaults

        cp = cfg.get("cellpose") or {}
        settings["resolved"]["cellpose"] = {
            "normalize": cfg.get("normalize", "image"),
            **applied_defaults(bool(cp.get("do_3D", False)), cp),
        }
    except ImportError:
        pass
record = provenance(**settings)

group = register_labels(
    image_store,
    label_name,
    provenance=record,
    n_levels=int(cfg.get("pyramid_levels", 5)),
    downscale=int(cfg.get("pyramid_downscale", 2)),
    progress=True,
    n_objects=n_objects,
    shard=cfg.get("shard", False),
    ngff_version=cfg.get("ngff_version", "auto"),
    level=int(cfg.get("level", 0)),
)

finish(group, record, objects)
