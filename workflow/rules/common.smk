# Shared paths and helpers.

WORK = config["work_dir"]
IMAGE = f"{WORK}/image.zarr"
# The store's root file marks it done: zarr.json (v3), or .zgroup for
# ngff_version "0.4" (zarr v2).
ZARR_ROOT_FILE = (
    ".zgroup" if str(config.get("ngff_version", "auto")) == "0.4" else "zarr.json"
)
IMAGE_OK = f"{IMAGE}/{ZARR_ROOT_FILE}"

# Per-brick maxima of the image (for skip_empty), shared by every config.
OCCUPANCY = f"{WORK}/image.occupancy.zarr/{int(config.get('level', 0))}"
OCCUPANCY_OK = f"{OCCUPANCY}/zarr.json"
OCCUPANCYLOG = f"{WORK}/logs/occupancy.log"

# Everything else is per segmentation, under WORK/<label_name>/.
LABEL_NAME = config.get("label_name", "labels")
RUN = f"{WORK}/{LABEL_NAME}"
TILES = f"{RUN}/tiles.json"
STAGE = f"{RUN}/stage.zarr"
STAGE_OK = f"{STAGE}.done"

# One log per step (a shared log would be cleared by each rule).
LOGS = f"{RUN}/logs"
CONVERTLOG = f"{WORK}/logs/convert.log"
PREPARELOG = f"{LOGS}/prepare.log"
MERGELOG = f"{LOGS}/merge.log"

# The model is downloaded on the submit host: GPU nodes are often offline.
MODEL_OK = f"{RUN}/model.ready"


def batch_done(wildcards):
    """One marker per batch of tiles, known once prepare has run."""
    tiles = checkpoints.prepare.get().output.tiles
    manifest = json.loads(Path(tiles).read_text())
    return [f"{RUN}/seg/{i}.done" for i in range(len(manifest["batches"]))]


# SLURM's own mail for the long steps (not segment: one job per batch).
NOTIFY_EMAIL = config.get("notify_email") or ""
NOTIFY_EVENTS = config.get("notify_events") or ["finish", "error"]


def notify_extra(*_args, **_kwargs):
    """slurm_extra fragment requesting mail for this job."""
    from patchworks._notify import slurm_mail_extra

    return slurm_mail_extra(NOTIFY_EMAIL, NOTIFY_EVENTS)
