# Plan tiles (checkpoint) and segment each tile on a GPU.


rule fetch_model:
    """Download the model on the submit host (GPU nodes are often offline)."""
    output:
        touch(MODEL_OK),
    log:
        f"{LOGS}/fetch_model.log",
    script:
        "../scripts/fetch_model.py"


checkpoint prepare:
    input:
        IMAGE_OK,
        OCCUPANCY_OK,
    output:
        tiles=TILES,
        stage=touch(STAGE_OK),
    log:
        PREPARELOG,
    script:
        "../scripts/prepare_tiles.py"


rule segment:
    """Segment one batch of `tiles_per_job` tiles on a GPU."""
    input:
        tiles=TILES,
        stage=STAGE_OK,
        image=IMAGE_OK,
        model=MODEL_OK,
    output:
        f"{RUN}/seg/{{batch}}.done",
    log:
        f"{LOGS}/segment/{{batch}}.log",
    script:
        "../scripts/segment_tile.py"
