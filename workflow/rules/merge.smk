# Stitch labels across tile boundaries and write them into the image.

rule merge:
    input:
        markers=batch_done,
        tiles=TILES,
    output:
        touch(f"{RUN}/labels.done"),
    log:
        MERGELOG,
    script:
        "../scripts/merge.py"
