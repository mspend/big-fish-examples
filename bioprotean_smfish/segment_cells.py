# run this in an environment with cellpose, such as the merfish3d environment

# last time it ran for 3.4 hours

from pathlib import Path
import numpy as np
from tifffile import imread, imwrite

from cellpose import models, io

# Enable Cellpose progress/logging
io.logger_setup()

# -----------------------------------------------------------------------------
# Input/output
# -----------------------------------------------------------------------------

image_path = Path("/data/smfish/20260311_bartelle_smFISH_cryo_48hr_male/fused/fiducial_no_downsampling.ome.tiff")
output_path = Path("/data/smfish/20260311_bartelle_smFISH_cryo_48hr_male/big_fish/segmentation/")

# -----------------------------------------------------------------------------
# Load image
# -----------------------------------------------------------------------------

img = imread(image_path)

print(f"Image shape: {img.shape}")
print(f"Image dtype: {img.dtype}")
# print(img.min(), img.max())
# print(np.percentile(img, [0, 1, 50, 99, 99.9, 100]))

# test bright, dim, dense, and sparse regions

test = img[10000:12000, 8000:10000]

# test_image = output_path / "test_image.tif"

# imwrite(
#     test_image,
#     test,
#     compression="zlib",
# )

# -----------------------------------------------------------------------------
# Load CPSAM model
# -----------------------------------------------------------------------------

model = models.CellposeModel(gpu=True)

# -----------------------------------------------------------------------------
# Run segmentation
# -----------------------------------------------------------------------------

masks, flows, styles = model.eval(
    test,
    normalize={"tile_norm_blocksize": 512}, # test if tiled normalization is better than global normalization for images with high variability
    cellprob_threshold=-1.0, # test2 - lower cellprob threshold to find more cells. Lowering it allows lower-confidence pixels to participate in mask creation.
    # flow_threshold=0.6, # increase flow_threshold to get more cells. This is a QC filter: masks whose predicted flows don't agree sufficiently with the flows reconstructed from the proposed ROI are discarded.
    )

print(f"Found {masks.max()} cells")

del flows
del styles

# -----------------------------------------------------------------------------
# Save masks as TIFF
# -----------------------------------------------------------------------------

dtype = np.uint16 if masks.max() < 65535 else np.uint32

masks_path = output_path / "fiducial_no_downsampling_test5.ome.tiff"

imwrite(
    masks_path,
    masks.astype(dtype),
    compression="zlib",
)

print(f"Saved masks to {output_path}")