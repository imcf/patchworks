# Convert the input to a pyramidal OME-ZARR.

rule convert:
    output:
        # marker file inside the store; existence => skip re-conversion.
        IMAGE_OK,
    script:
        "../scripts/convert.py"
