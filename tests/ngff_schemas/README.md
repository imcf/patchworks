# Vendored OME-NGFF JSON schemas

The official OME-NGFF (OME-ZARR) metadata schemas, copied verbatim from
[ome/ngff-spec](https://github.com/ome/ngff-spec) at tags `0.4` and `0.5`.
Licensed CC-BY-4.0 by the Open Microscopy Environment.

`test_ome_zarr.py::test_output_conforms_to_the_official_ngff_schemas`
validates the metadata patchworks actually writes against them, so a change
to how a store is described cannot silently stop being OME-ZARR. They are
vendored rather than fetched so the check runs offline and pins a known
revision.

NGFF **0.6** (released 2026-09-14) is deliberately absent: patchworks does
not write it. RFC-5 replaces a multiscale's `axes` with `coordinateSystems`
and requires `input`/`output` on every coordinate transformation, and no
reader supports it yet. Add the schemas here when that changes.
