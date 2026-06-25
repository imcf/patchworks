# Plan tiles (checkpoint) and segment each tile on a GPU.

checkpoint prepare:
    input:
        IMAGE_OK,
    output:
        tiles=TILES,
        stage=touch(STAGE_OK),
    log:
        STEPLOG,
    script:
        "../scripts/prepare_tiles.py"


rule segment:
    """Segment one tile on a GPU and write it into the stage store."""
    input:
        tiles=TILES,
        stage=STAGE_OK,
        image=IMAGE_OK,
    output:
        f"{WORK}/seg/{{index}}.done",
    log:
        f"{LOGS}/segment/{{index}}.log",
    script:
        "../scripts/segment_tile.py"
