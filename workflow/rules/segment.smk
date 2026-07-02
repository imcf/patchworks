# Plan tiles (checkpoint) and segment each tile on a GPU.


rule fetch_model:
    """Cache the segmentation model on the (networked) submit host.

    Declared local (see ``localrules`` in the Snakefile) so it never runs on an
    offline GPU node — Cellpose downloads its weights here, into shared $HOME.
    """
    output:
        touch(MODEL_OK),
    log:
        f"{LOGS}/fetch_model.log",
    script:
        "../scripts/fetch_model.py"


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
        model=MODEL_OK,
    output:
        f"{RUN}/seg/{{index}}.done",
    log:
        f"{LOGS}/segment/{{index}}.log",
    script:
        "../scripts/segment_tile.py"
