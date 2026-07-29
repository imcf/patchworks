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
        # Depend on the map rather than building it inline: it streams the
        # whole image, and several configs' prepare steps run concurrently, so
        # inline each would stream the volume and all but one discard it.
        OCCUPANCY_OK,
    output:
        tiles=TILES,
        stage=touch(STAGE_OK),
    log:
        PREPARELOG,
    script:
        "../scripts/prepare_tiles.py"


rule segment:
    """Segment one batch of tiles on a GPU and write them into the stage store.

    A batch is `tiles_per_job` tiles (see config), processed sequentially in
    one process so they share a single CUDA init and a single Cellpose model
    load. Batches write disjoint chunks, so any number of them run in
    parallel across GPUs.
    """
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
