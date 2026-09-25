# Provenance

Every label image patchworks writes records how it was made: the
segmentation function and its bound settings, tiling, overlap, stitching,
level, channel, codec, the input, the library versions and a UTC timestamp.
It sits in the label group's attrs under `"patchworks"` (for `write_to=`
stores, on the labels array), so it travels with the data.

```python
from patchworks import read_provenance

read_provenance("scan.zarr/labels/cilia_labels")["settings"]["custom"]
```

::: patchworks.read_provenance

::: patchworks.provenance
