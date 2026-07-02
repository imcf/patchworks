# Shared paths and helpers for the patchworks workflow.

WORK = config["work_dir"]
IMAGE = f"{WORK}/image.zarr"
# A single file inside the store, used as the convert rule's output and as the
# dependency marker for downstream rules. Tracking a leaf file (not the
# directory) lets Snakemake skip conversion when the store already exists and
# avoids wiping the whole store on a re-run (same trick as imcf/sopa).
IMAGE_OK = f"{IMAGE}/zarr.json"

# Everything below is per-segmentation, namespaced under WORK/<label_name>/, so
# running the workflow twice with two configs (different label_name, e.g.
# "nuclei_labels" and "cell_labels") against the *same* work_dir never
# collides — each gets its own tiles/stage/seg/model/labels.done, and both
# read the *same* already-converted image.zarr. See docs/guide/snakemake.md
# "Running two segmentations" for the two-config recipe.
LABEL_NAME = config.get("label_name", "labels")
RUN = f"{WORK}/{LABEL_NAME}"
TILES = f"{RUN}/tiles.json"
STAGE = f"{RUN}/stage.zarr"
# Completion sentinel for the stage store. Tracking a touch()ed marker instead
# of directory(STAGE) keeps Snakemake from deleting/recreating the store on a
# re-run and avoids directory-mtime quirks (same touch() discipline as sopa).
STAGE_OK = f"{STAGE}.done"


# Logs: one shared file for the sequential CPU steps (convert/prepare/merge),
# one file per tile for the GPU segment jobs.
LOGS = f"{RUN}/logs"
STEPLOG = f"{LOGS}/steps.log"

# Marker that the segmentation model is cached locally. Produced by a local
# rule (runs on the networked submit host) so offline GPU nodes never download.
# Namespaced per-run too: two configs using different models must each fetch
# their own, rather than the second silently reusing the first's marker.
MODEL_OK = f"{RUN}/model.ready"


def occupied_done(wildcards):
    """Per-tile markers for the occupied tiles (resolved after the checkpoint)."""
    tiles = checkpoints.prepare.get().output.tiles
    occupied = json.loads(Path(tiles).read_text())["occupied"]
    return [f"{RUN}/seg/{i}.done" for i in occupied]
