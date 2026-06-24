# Shared paths and helpers for the patchworks workflow.

WORK = config["work_dir"]
IMAGE = f"{WORK}/image.zarr"
# A single file inside the store, used as the convert rule's output and as the
# dependency marker for downstream rules. Tracking a leaf file (not the
# directory) lets Snakemake skip conversion when the store already exists and
# avoids wiping the whole store on a re-run (same trick as imcf/sopa).
IMAGE_OK = f"{IMAGE}/zarr.json"
TILES = f"{WORK}/tiles.json"
STAGE = f"{WORK}/stage.zarr"
# Completion sentinel for the stage store. Tracking a touch()ed marker instead
# of directory(STAGE) keeps Snakemake from deleting/recreating the store on a
# re-run and avoids directory-mtime quirks (same touch() discipline as sopa).
STAGE_OK = f"{STAGE}.done"


def occupied_done(wildcards):
    """Per-tile markers for the occupied tiles (resolved after the checkpoint)."""
    tiles = checkpoints.prepare.get().output.tiles
    occupied = json.loads(Path(tiles).read_text())["occupied"]
    return [f"{WORK}/seg/{i}.done" for i in occupied]
