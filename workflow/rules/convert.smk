# Convert the input to a pyramidal OME-ZARR.

rule convert:
    output:
        # marker file inside the store; existence => skip re-conversion.
        IMAGE_OK,
    log:
        STEPLOG,
    script:
        "../scripts/convert.py"
