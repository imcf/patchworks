# Stitch labels across tile boundaries and write them into the image.

rule merge:
    input:
        occupied_done,
    output:
        touch(f"{WORK}/labels.done"),
    log:
        STEPLOG,
    script:
        "../scripts/merge.py"
