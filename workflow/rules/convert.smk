# Convert the input to a pyramidal OME-ZARR.

rule convert:
    output:
        # marker file inside the store; existence => skip re-conversion.
        IMAGE_OK,
    log:
        CONVERTLOG,
    script:
        "../scripts/convert.py"


# Build the occupancy map as a real job. It streams the whole image, so doing
# it in the run_multi driver ran it on the login node, where a multi-terabyte
# read is killed with no traceback. Built once and reused by every config.
rule occupancy:
    input:
        IMAGE_OK,
    output:
        OCCUPANCY_OK,
    log:
        OCCUPANCYLOG,
    script:
        "../scripts/build_occupancy.py"
