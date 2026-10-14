"""What each setting of the launcher means, with an example.

Shown by Streamlit's ``help=`` (the small question mark next to a field,
read on hover). Kept here, in one place, so the form stays readable and the
texts can be checked against the workflow's documentation
(docs/guide/snakemake.md, workflow/config/config.yaml).
"""

HELP = {
    # -- connecting ---------------------------------------------------------
    "preset": "A cluster from `clusters.yaml`: its host, host key, workflow "
    "folder and SLURM defaults. *Custom* to type a host in.",
    "host": "The cluster's login node, as for `ssh`.\n\n"
    "Example: `login.scicore.unibas.ch`",
    "port": "The SSH port: 22 unless your cluster says otherwise.",
    "username": "Your cluster account, the one you `ssh` with.",
    "password": "Your cluster password. It only opens the SSH connection: "
    "never stored, and cleared from the form once connected.",
    "allow_unverified": "Connect although this host's key is not pinned in "
    "`clusters.yaml`, so its identity is not checked. Only when you run "
    "this app yourself, on a network you trust.",
    # -- what to run --------------------------------------------------------
    "run_kind": "**One segmentation**: one label image (say the cells).\n\n"
    "**Several**: label images of the same image (nuclei, cells, cilia ...) "
    "segmented side by side, then related: which nucleus is in which cell.",
    "n_segs": "How many label images to make of this image, each with its "
    "own channel and method. They share the image, its conversion and the "
    "tiling.",
    # -- image ----------------------------------------------------------------
    "input": "The image to segment, on the cluster: `.ims`, `.czi`, `.lif`, "
    "`.nd2`, OME-TIFF or OME-Zarr. A folder of one TIFF per plane is picked "
    "with *Use this folder's TIFFs as one image* in the browser.\n\n"
    "Example: `/scicore/home/lab/me/data/embryo_01.czi`",
    "work_dir": "Where everything this run writes goes, on the cluster: the "
    "converted image (`image.zarr`), the label images inside it, logs, the "
    "relation workbooks. Use a fast, large filesystem (scratch); several "
    "runs on one image share it.\n\n"
    "Example: `/scratch/me/embryo_01`",
    "sequence_pattern": "For an image stored as one TIFF per plane: a "
    "regular expression saying which part of each file name is which "
    "axis. Each named group (`Z`, `C`, `T`) captures that axis's index.\n\n"
    "Example, for `scan_T0_Z012_C1.tif`: `_Z(?P<Z>\\d+).*?_C(?P<C>\\d+)`",
    "reuse_pyramid": "For an `.ims` file: copy its own lower-resolution "
    "levels instead of computing them again. Faster to convert; off when "
    "unsure.",
    "shard": "Pack the converted image's many small chunk files into fewer, "
    "larger ones (zarr sharding). For filesystems that dislike many small "
    "files, or a file quota. Choose before converting.",
    "ngff_version": "The OME-Zarr version written. **auto**: 0.5 (zarr v3). "
    "**0.4**: zarr v2, for older readers (some Fiji/Java tools); no "
    "sharding.",
    "compression": "How chunks are compressed. **zstd**: the best for label "
    "images. **zstd:3**: about 10% smaller raw images, slower. **blosc**: "
    "for older Java viewers without zstd. **none**: largest, fastest.",
    "convert_chunks": "The converted image's chunk shape, as `[z, y, x]`. "
    "Blank: chosen for you, which suits nearly every image.\n\n"
    "Example: `[8, 512, 512]`",
    # -- tiling ---------------------------------------------------------------
    "level": "Which resolution to segment: 0 is the full resolution, 1 half "
    "of it in y and x, 2 a quarter ... A coarser level is much faster, for "
    "objects many pixels wide.\n\n"
    "Example: 1 for nuclei 60 pixels across",
    "tile_shape": "The size of the pieces the image is cut into, one per "
    "GPU job step, as `[z, y, x]`. **auto**: sized to the GPU's memory "
    "(the usual).\n\n"
    "Example: `[16, 1024, 1024]`; smaller if a GPU runs out of memory",
    "tiles_per_job": "Tiles each GPU job segments one after the other, "
    "loading the model once. More: fewer jobs, each longer; keep a job "
    "under its runtime.\n\n"
    "Example: 4",
    "gpu_memory_gb": "The memory of the GPU the segment jobs get, in GB, to "
    "size `tile_shape: auto`. 0: taken from the segment partition's GPU "
    "(24 for an RTX 4090), or 8 when unknown.\n\n"
    "Example: 24",
    "skip_empty": "Skip tiles with no signal (background), found from a "
    "low-resolution map of the image. Saves GPU hours on sparse samples. "
    "If tiles inside the tissue come out empty, lower *empty_threshold* or "
    "turn this off.",
    "empty_threshold": "The intensity below which a tile counts as empty, "
    "in the image's own values. Blank: Otsu's threshold of the channel, "
    "computed for you.\n\n"
    "Example: `120` for a 16-bit image with background around 100",
    # -- one segmentation -----------------------------------------------------
    "label_name": "The name of this label image, inside the converted image "
    "(`image.zarr/labels/<name>`) and in the result tables. Letters, "
    "digits and `_`.\n\n"
    "Example: `nuclei_labels`",
    "channel": "Which channel to segment, counting from 0. *See the "
    "channels* (under Image) shows each, once converted.\n\n"
    "Example: 0 for the first channel",
    "nuclei_channel": "A second channel with the nuclei, from 0. Cellpose 3 "
    "uses it as its second input; the watershed and PlantSeg grow one cell "
    "from each nucleus. Blank: none.\n\n"
    "Example: 1",
    "overlap": "How far each tile reads into its neighbours (the halo), in "
    "pixels: about one object, so objects cut by a tile edge are seen "
    "whole. One number, or one per axis `[z, y, x]`.\n\n"
    "Example: `[4, 30, 30]` for cells about 30 pixels across",
    "method": "**cellpose**: cells or nuclei from a deep-learning model, "
    "the usual. **PlantSeg**: cells from a membrane stain. **watershed**: "
    "cells grown from nuclei on a membrane stain. **DoG**: small bright "
    "spots and thin structures (cilia, puncta), optionally deconvolved. "
    "**threshold**: bright objects on a dark background (quick, on the "
    "CPU). **custom**: your own function.",
    # cellpose
    "cp_model": "The Cellpose model. Cellpose 3: `cyto3` (cells), `nuclei`. "
    "Cellpose 4: `cpsam`. Only the cluster's installed version's names "
    "are offered; or type the path of your own trained model.",
    "cp_diameter": "The typical object diameter, in pixels (at the level "
    "segmented). 0: Cellpose estimates it, tile by tile -- slower and less "
    "even.\n\n"
    "Example: 30",
    "cp_do_3d": "Segment in 3-D: objects are followed through z. Off: each "
    "plane alone, then stitched. 3-D is better for round objects in a "
    "fine z-stack, and needs more GPU memory.",
    "cp_gpu": "Run Cellpose on the GPU (the segment jobs have one). Off "
    "only to test on CPU nodes: much slower.",
    "cp_extra_kwargs": "Any other Cellpose setting, as YAML `key: value` "
    "lines.\n\n"
    "Examples: `flow_threshold: 0.4` (2-D: lower keeps only well-shaped "
    "cells), `cellprob_threshold: -1` (more, larger cells), "
    "`anisotropy: 1`, `min_size: 50`",
    "normalize": "Cellpose rescales each input's contrast. **One range for "
    "the whole image** (measured once): the same contrast in every tile, "
    "so cells are not cut at tile seams -- the usual. **Each tile its "
    "own**: for very uneven staining.",
    # PlantSeg and the watershed
    "ps_model": "PlantSeg's U-Net predicting cell boundaries. "
    "`generic_confocal_3D_unet` for confocal, "
    "`generic_light_sheet_3D_unet` for light-sheet; others from PlantSeg's "
    "zoo can be typed. Fetched on the login node before the run.",
    "ps_segmentation": "How the boundaries become cells. **gasp** (the "
    "usual), **mutex_ws**, **multicut**: agglomerate supervoxels. "
    "**dt_watershed**: supervoxels only. **nuclei_watershed**: one cell per "
    "nucleus, flooded from them. **lifted_multicut**: keeps each nucleus's "
    "supervoxels together. The last two need nuclei.",
    "ps_beta": "Under- or over-segmentation: lower merges more (fewer, "
    "larger cells), higher splits more. PlantSeg's default 0.6.\n\n"
    "Example: 0.5 if cells come out split",
    "ps_post_minsize": "Cells smaller than this many voxels are merged into "
    "a neighbour.\n\n"
    "Example: 100",
    "ws_nuclei_min_size": "Nuclei found smaller than this many voxels are "
    "ignored as seeds (noise, debris).\n\n"
    "Example: 200",
    "seed_from": "Where the nuclei come from. **The nuclei channel**: found "
    "in the channel given above, tile by tile. **Another segmentation**: "
    "the objects of another label image of this run (say Cellpose's "
    "nuclei), one cell grown from each; it runs first.",
    "seed_labels": "The label image whose objects seed the cells: one cell "
    "grows from each.\n\n"
    "Example: `nuclei_labels`",
    "max_radius_um": "Stop each cell this far from its nucleus, in µm. "
    "Blank: cells grow up to their walls (or the tissue's edge).\n\n"
    "Example: 15",
    "foreground": "**otsu**: stop at the tissue's edge (Otsu's threshold of "
    "the stain), since a boundary U-Net puts cells everywhere, empty space "
    "included. **none**: no mask.",
    "extra_kwargs": "Any other setting of the plugin, as YAML `key: value` "
    "lines.\n\n"
    "Examples: `patch: [64, 128, 128]` (PlantSeg, less GPU memory), "
    "`nuclei_threshold: 800`, `compactness: 0.01`",
    # DoG
    "dog_low_sigma": "The narrow blur, in pixels (or µm): about the size of "
    "the structures to find. The DoG keeps what is between the two "
    "blurs.\n\n"
    "Example: 1 for spots 2-3 pixels wide",
    "dog_high_sigma": "The wide blur, larger than low_sigma: the background "
    "removed. About 2-5 times low_sigma.\n\n"
    "Example: 3",
    "dog_threshold": "What counts as signal in the DoG image, in the image's "
    "own intensity units (not 0-1). Too low: noise becomes objects; too "
    "high: dim ones are lost. Check with *Try it on a GPU*.\n\n"
    "Example: 50 for a 16-bit image with spots ~300 above background",
    "dog_sigma_units": "**px**: sigmas in pixels on every axis, so z (often "
    "a coarser step) is blurred further. **um**: in µm, the same physical "
    "blur along z as across, from the image's calibration.",
    "dog_use_gpu": "Blur and label on the GPU (cupy). Faster on large "
    "tiles; needs a CUDA environment (`cuda12`, or `plantseg`).",
    "dog_psf": "A point-spread-function image on the cluster, to deconvolve "
    "each tile first (sharper spots, on a GPU). Blank: no deconvolution.\n\n"
    "Example: `/scicore/home/lab/psf/488_63x.tif`",
    "dog_wavelength": "The emission wavelength, in nm, for the "
    "deconvolution.\n\n"
    "Example: 525 for GFP",
    "dog_na": "The objective's numerical aperture.\n\nExample: 1.4 (oil 63x)",
    "dog_nimm": "The immersion medium's refractive index.\n\n"
    "Example: 1.515 oil, 1.33 water, 1.47 glycerol",
    "dog_dup_rev_z": "Mirror each tile in z before deconvolving, against "
    "ghosts from the FFT wrapping around on thin tiles. **auto**: when the "
    "tile is thin.",
    # custom
    "custom_module": "The Python module with your function: a file in "
    "`workflow/scripts/` (without `.py`), or an installed package.\n\n"
    "Example: `my_seg`",
    "custom_function": "The function's name in that module: it gets a tile "
    "(numpy array) and returns its labels.\n\nExample: `segment`",
    "custom_kwargs": "Keyword arguments for the function, as YAML.\n\n"
    "Example: `threshold: 0.5`",
    # stitching and clean-up
    "stitch": "How objects cut by tile edges are joined. **touch**: pieces "
    "touching across the edge become one (nuclei, objects apart). "
    "**iou**: only pieces both tiles agree on, by overlap in the halo, so "
    "cells touching each other stay apart (cells, membranes). PlantSeg and "
    "the watershed need iou.",
    "iou_threshold": "With iou: how much two pieces must overlap, as a share "
    "of the smaller one, to be joined.\n\n"
    "Example: 0.5; lower if cells come out cut at seams",
    "fill_holes": "Fill holes inside each object. **3-D**: closed cavities. "
    "**per plane**: holes in each z-plane, for shells open at the top or "
    "bottom.",
    "open_radius": "Cut spurs thinner than about 2r+1 pixels off each "
    "object (morphological opening). 0: off. Not for thin objects such as "
    "cilia.\n\n"
    "Example: 1",
    "dilate": "Grow every object by this many pixels, after segmenting. "
    "Nuclei grown into a rough cell area, say.\n\nExample: 2",
    "dilate_gpu": "Grow the objects on the GPU (cupy): faster for large "
    "images; needs a CUDA environment.",
    "min_volume": "Drop objects smaller than this, in µm³, judged on whole "
    "objects after the merge. Blank: none.\n\n"
    "Example: 50 for debris under a small nucleus",
    "max_volume": "Drop objects larger than this, in µm³ (merged clumps, "
    "background). Blank: none.\n\nExample: 5000",
    # outputs
    "sequential_labels": "Number the objects 1, 2, 3 ... without gaps. Off "
    "keeps the numbers the tiles gave.",
    "pyramid_levels": "How many resolutions of each label image to write, "
    "for fast viewing zoomed out.\n\nExample: 5",
    "pyramid_downscale": "How much smaller each of those levels is than the "
    "one before, in y and x.\n\nExample: 2",
    "shard_labels": "Pack the label image's full resolution into fewer, "
    "larger files after the merge (one extra pass). As *shard*, for the "
    "labels.",
    "seam_report": "Write `seams.json`: how many objects end exactly on a "
    "tile seam, against inside the tiles. Far more at the seams means "
    "stitching cut objects (raise overlap, or use iou).",
    "merge_workers": "Parallel workers for the merge. Blank: from the "
    "merge job's CPUs and memory.\n\nExample: 16",
    "object_table": "Write one row per object next to the labels: size, "
    "centroid, bounding box, shape. `patchworks review` and the relation "
    "workbooks read it.",
    "table_channels": "Channels whose mean and standard deviation of "
    "intensity to measure per object, comma-separated. Blank: none.\n\n"
    "Example: `0,2`",
    "notify_email": "Mail this address when the long steps finish or fail, "
    "with the failing step's log.\n\nExample: `me@unibas.ch`",
    "notify_events": "When to mail: at the start, when it finishes, on an "
    "error.",
    # relations and review
    "relate_partition": "The SLURM partition for the relate jobs, which "
    "find each object's parent (each nucleus's cell). Blank: the default.",
    "relate_mem": "Memory for each relate job.\n\nExample: `64G`",
    "relate_cpus": "CPUs for each relate job. 0: the default.",
    "relate_time": "Time for each relate job, in minutes. 0: the default. "
    "Keep it within the QOS's limit.",
    "relate_qos": "The QOS for the relate jobs. Blank: the account's default.",
    "bundle_format": "Pack the finished `image.zarr` into one file to copy "
    "or archive. **zip**: read directly by zarr, napari and patchworks. "
    "**iso**: a read-only disk image. **none**: leave the folder.",
    "min_overlap": "A child (say a nucleus) less than this share inside its "
    "parent (its cell) is flagged by `patchworks review`.\n\n"
    "Example: 0.5",
    "pos_child": "Classify these objects (say cilia) as apical, basal, "
    "lateral or central in their parent.",
    "pos_parent": "The parent they are classified in (say the cells).",
    "pos_apical": "Which way is apical. **Away from** a label image: "
    "opposite its objects (away from the nuclei, for epithelia with basal "
    "nuclei). **+z / -z**: up or down the stack.",
    # templates and past runs
    "template": "Settings saved with *Save as template*: they fill the form, "
    "keeping its image and work_dir.",
    "past_launch": "A run launched from this workflow folder: its settings "
    "fill the form, image included.",
    "load_path": "Any run config or multi config on the cluster (`.yaml`).",
    "template_name": "A name for these settings.\n\n"
    "Example: `nuclei_cells_cilia_63x`",
    "template_with_image": "Off (the usual): the image and work_dir are "
    "left out, and loading the template keeps the form's own -- the same "
    "settings for the next image.",
    # launch
    "pixi_env": "The workflow's pixi environment the run starts from; its "
    "jobs use it too. Picked from what the config needs: `plantseg` for "
    "PlantSeg, `cuda12` for GPU steps, `default` otherwise.",
    "what_to_run": "**Everything**: convert, segment, merge (and relate). "
    "**Convert only**: just the conversion, so you can check and plan "
    "before segmenting; a later full run goes straight on.",
    "run_mode": "**Dry run**: list what would run, start nothing. "
    "**Controller as a SLURM job** (the usual): a small 1-CPU job submits "
    "and follows the GPU jobs; you can close the browser. **On the login "
    "node**: for short runs only, a reboot ends it.",
    "ctrl_time": "How long the controller job may run: the whole run, "
    "conversion to merge. It must fit its partition's and QOS's "
    "limits.\n\nExample: `3-00:00:00` (3 days)",
    "ctrl_mem": "Memory of the controller job: it only follows the other "
    "jobs.\n\nExample: `4G`",
    "ctrl_partition": "The partition for the controller job: one whose time "
    "limit covers the whole run. Blank: the cluster's default.",
    "ctrl_qos": "The QOS for the controller job: its MaxWall must cover the "
    "time above, or sbatch refuses the job (QOSMaxWallDurationPerJobLimit). "
    "Blank: the account's default QOS, often a short one (sciCORE: 6 hours); "
    "when that is too short, one that allows the time is chosen for you."
    "\n\nExample: `1week` for a 3-day run on sciCORE",
    "ctrl_account": "The SLURM account to charge. Blank: your default.",
}
