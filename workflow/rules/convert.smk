# Convert the input to a pyramidal OME-ZARR.

rule convert:
    output:
        directory(IMAGE),
    script:
        "../scripts/convert.py"
