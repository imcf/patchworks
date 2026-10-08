# Convert the input to a pyramidal OME-Zarr, once.

rule convert:
    output:
        IMAGE_OK,
    resources:
        slurm_extra=notify_extra,
    log:
        CONVERTLOG,
    script:
        "../scripts/convert.py"


# The occupancy map for skip_empty: one pass over the image, shared by every
# config.
rule occupancy:
    input:
        IMAGE_OK,
    output:
        OCCUPANCY_OK,
    resources:
        slurm_extra=notify_extra,
    log:
        OCCUPANCYLOG,
    script:
        "../scripts/build_occupancy.py"
