# OME-ZARR conversion plugin

Write any array or image file to a pyramidal OME-ZARR store. Uses only the
core dependencies for arrays and `.zarr` inputs; reading other file formats
needs the optional `bioio` extra (`pip install "patchworks[bioio]"`).

::: patchworks.plugins.ome_zarr.to_ome_zarr
