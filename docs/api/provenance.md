# Provenance

Every label image patchworks writes records how it was made: the
segmentation function and its bound settings, tiling, overlap, stitching,
level, channel, codec, the input, the library versions and a UTC timestamp.
It sits in the label group's attrs under `"patchworks"` (for `write_to=`
stores, on the labels array), so it travels with the data.

The cluster workflow stores its whole effective config (everything but the
email settings), plus what was only decided at run time under `resolved`
(tiles segmented, the empty threshold used, the size filter in voxels, the
object count) and the [seam report](../guide/merging.md#checking-the-seams)
ratios under `seams`.

```python
from patchworks import read_provenance

rec = read_provenance("scan.zarr/labels/cilia_labels")
rec["settings"]["custom"]                # the method's settings
rec["settings"]["resolved"]              # tiles, empty threshold, object count
rec["seams"]                             # seam vs interior orphan rate per axis
```

::: patchworks.read_provenance

::: patchworks.provenance
