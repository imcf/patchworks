# Stitch labels across tile boundaries and write them into the image.

rule merge:
    input:
        occupied_done,
    output:
        touch(f"{WORK}/labels.done"),
    script:
        "../scripts/merge.py"
