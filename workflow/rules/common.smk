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


# Logs: one shared file for the sequential CPU steps (convert/prepare/merge),
# one file per tile for the GPU segment jobs.
LOGS = f"{WORK}/logs"
STEPLOG = f"{LOGS}/steps.log"

# Marker that the segmentation model is cached locally. Produced by a local
# rule (runs on the networked submit host) so offline GPU nodes never download.
MODEL_OK = f"{WORK}/model.ready"


def occupied_done(wildcards):
    """Per-tile markers for the occupied tiles (resolved after the checkpoint)."""
    tiles = checkpoints.prepare.get().output.tiles
    occupied = json.loads(Path(tiles).read_text())["occupied"]
    return [f"{WORK}/seg/{i}.done" for i in occupied]
