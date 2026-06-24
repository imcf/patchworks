# Shared paths and helpers for the patchworks workflow.

WORK = config["work_dir"]
IMAGE = f"{WORK}/image.zarr"
TILES = f"{WORK}/tiles.json"
STAGE = f"{WORK}/stage.zarr"


def occupied_done(wildcards):
    """Per-tile markers for the occupied tiles (resolved after the checkpoint)."""
    tiles = checkpoints.prepare.get().output.tiles
    occupied = json.loads(Path(tiles).read_text())["occupied"]
    return [f"{WORK}/seg/{i}.done" for i in occupied]
