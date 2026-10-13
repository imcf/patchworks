"""How far a run has got: converted, tiles segmented, merged, related.

    pixi run status config/config.yaml        # one segmentation
    pixi run status my_multi.yaml             # several (a multi config)
    pixi run status my_multi.yaml --json      # for the web launcher

Reads only the markers the workflow leaves in work_dir (no Snakemake, no
image data), so it answers at once, while a run is going and after it.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from run_multi import _config_bases, _load_yaml, _resolve  # noqa: E402


def _configs(path: Path) -> tuple[list[dict], dict | None]:
    """The segmentation configs of *path*, and the multi config if it is one."""
    cfg = _load_yaml(path)
    if "segmentations" not in cfg:
        return [cfg], None
    inner = (path.parent, Path(__file__).resolve().parent.parent)
    common = cfg.get("common")
    common_cfg = _load_yaml(_resolve(common, *inner)) if common else {}
    return [
        {**common_cfg, **_load_yaml(_resolve(p, *inner))}
        for p in cfg["segmentations"]
    ], cfg


def _attrs(path: Path) -> dict:
    """Attributes of a zarr group or array (v3 zarr.json, or v2 .zattrs)."""
    for name, inner in (("zarr.json", "attributes"), (".zattrs", None)):
        try:
            data = json.loads((path / name).read_text())
        except (OSError, ValueError):
            continue
        return data.get(inner, {}) if inner else data
    return {}


def segmentation_status(cfg: dict) -> dict:
    """Where one segmentation stands, from its markers in work_dir."""
    work = Path(cfg["work_dir"])
    label = cfg.get("label_name", "labels")
    run = work / label
    store = work / "image.zarr"
    out: dict = {
        "label_name": label,
        "converted": (store / "zarr.json").exists()
        or (store / ".zgroup").exists(),
        "batches": None,
        "batches_done": 0,
        "merged": (run / "labels.done").exists(),
        "objects": None,
        "seams": None,
    }
    try:
        manifest = json.loads((run / "tiles.json").read_text())
        out["batches"] = len(manifest["batches"])
    except (OSError, ValueError, KeyError):
        pass
    seg = run / "seg"
    if seg.is_dir():
        out["batches_done"] = sum(1 for _ in seg.glob("*.done"))
    if out["merged"]:
        level0 = _attrs(store / "labels" / label / "0")
        out["objects"] = level0.get("patchworks_n_objects")
        try:
            report = json.loads((run / "seams.json").read_text())
            # The report keys axes by index into the (z, y, x) labels.
            out["seams"] = {
                "zyx"[int(ax)] if str(ax) in "012" else str(ax): {
                    k: row.get(k) for k in ("seam_rate", "interior_rate")
                }
                for ax, row in report["axes"].items()
            }
        except (OSError, ValueError, KeyError):
            pass
    if out["merged"]:
        out["stage"] = "done"
    elif out["batches"] is not None:
        out["stage"] = "segmenting"
    elif out["converted"]:
        out["stage"] = "converted"
    else:
        out["stage"] = "not started"
    return out


def run_status(cfgs: list[dict], multi: dict | None) -> dict:
    """Every segmentation, and for a multi run its relations and bundle."""
    status = {
        "work_dir": str(cfgs[0]["work_dir"]),
        "segmentations": [segmentation_status(c) for c in cfgs],
    }
    if multi is None:
        return status
    work = Path(cfgs[0]["work_dir"])
    relations = []
    for rel in multi.get("relations") or []:
        out = work / rel.get("output", f"{rel['a']}_to_{rel['b']}.xlsx")
        # A sheet too long for Excel is written as two csv files instead.
        csvs = [
            out.with_name(f"{out.stem}_{n}.csv") for n in (rel["a"], rel["b"])
        ]
        files = [out] if out.exists() else [p for p in csvs if p.exists()]
        relations.append(
            {"a": rel["a"], "b": rel["b"], "files": [str(p) for p in files]}
        )
    status["relations"] = relations
    bundle = multi.get("bundle", {})
    if bundle is not False:
        fmt = (bundle or {}).get("format") or "zip"
        path = Path((bundle or {}).get("output") or f"{work}/image.zarr.{fmt}")
        status["bundle"] = str(path) if path.exists() else None
    return status


def format_status(status: dict) -> str:
    lines = [f"work_dir: {status['work_dir']}"]
    for s in status["segmentations"]:
        if s["batches"] is not None:
            tiles = f"{s['batches_done']}/{s['batches']} batches"
        else:
            tiles = "not prepared"
        line = f"  {s['label_name']:<24} {s['stage']:<12} {tiles}"
        if s["objects"] is not None:
            line += f", {s['objects']:,} objects"
        lines.append(line)
        for ax, row in (s["seams"] or {}).items():
            inner = row["interior_rate"]
            lines.append(
                f"    seams {ax}: {100 * row['seam_rate']:.1f}% of objects end "
                "on a seam vs "
                f"{'n/a' if inner is None else f'{100 * inner:.1f}%'} inside"
            )
    for rel in status.get("relations", []):
        files = ", ".join(rel["files"]) or "not yet"
        lines.append(f"  relation {rel['a']} -> {rel['b']}: {files}")
    if "bundle" in status:
        lines.append(f"  bundle: {status['bundle'] or 'not yet'}")
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("config", help="a run config, or a multi config")
    parser.add_argument("--json", action="store_true", help="print JSON")
    args = parser.parse_args()
    workflow_dir = Path(__file__).resolve().parent.parent
    path = _resolve(args.config, *_config_bases(workflow_dir))
    if not path.is_file():
        sys.exit(f"[status] ERROR: {args.config} not found")
    status = run_status(*_configs(path))
    print(json.dumps(status) if args.json else format_status(status))


if __name__ == "__main__":
    main()
