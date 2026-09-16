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

## reshard_level

::: patchworks.plugins.ome_zarr.reshard_level

## read_pixel_size

::: patchworks.plugins.ome_zarr.read_pixel_size

## NGFF metadata layout

NGFF 0.4 is defined over zarr v2 and puts its keys at the top level; 0.5 is
the zarr-v3 revision and nests them under `ome`. patchworks writes whichever
matches the store, and reads both.

::: patchworks.plugins.ome_zarr.ngff_version

Every writer takes an `ngff_version=` keyword: `"auto"` (default) follows the
installed zarr — 0.5 on v3, 0.4 on v2 — and `"0.4"` pins the older, zarr-v2
layout. Writing into an existing store always follows *that store's* format,
so a label pyramid added later can never disagree with the image it sits in.
NGFF 0.6 is released but not written: RFC-5 replaces `axes` with
`coordinateSystems` and requires `input`/`output` on every coordinate
transformation, and no reader supports it yet.

::: patchworks.plugins.ome_zarr.read_ngff_attr

::: patchworks.plugins.ome_zarr.write_ngff_attrs
