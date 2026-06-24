# Plan tiles (checkpoint) and segment each tile on a GPU.

checkpoint prepare:
    input:
        IMAGE_OK,
    output:
        tiles=TILES,
        stage=directory(STAGE),
    script:
        "../scripts/prepare_tiles.py"


rule segment:
    """Segment one tile on a GPU and write it into the stage store."""
    input:
        tiles=TILES,
        stage=STAGE,
        image=IMAGE_OK,
    output:
        f"{WORK}/seg/{{index}}.done",
    script:
        "../scripts/segment_tile.py"
