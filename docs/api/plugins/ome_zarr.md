# OME-ZARR conversion plugin

Write any array or image file to a pyramidal OME-ZARR store, add resolution
levels to an existing store, or store a label image inside an OME-ZARR under
the NGFF `labels/` group. Uses only the core dependencies for arrays and
`.zarr` inputs; reading other file formats needs the optional `bioio` extra
(`pip install "patchworks[bioio]"`).

Pyramids downsample **X and Y only** — `Z` (and channel/time) are kept at full
resolution, matching anisotropic microscopy stacks.

## to_ome_zarr

::: patchworks.plugins.ome_zarr.to_ome_zarr

## add_pyramid

::: patchworks.plugins.ome_zarr.add_pyramid

## write_labels

::: patchworks.plugins.ome_zarr.write_labels

## register_labels

::: patchworks.plugins.ome_zarr.register_labels
