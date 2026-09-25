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
output_path = image_path.with_name(image_path.stem + "_masks.tiff")

# -----------------------------------------------------------------------------
# Load image
# -----------------------------------------------------------------------------

img = imread(image_path)

print(f"Image shape: {img.shape}")
print(f"Image dtype: {img.dtype}")

# -----------------------------------------------------------------------------
# Load CPSAM model
# -----------------------------------------------------------------------------

model = models.CellposeModel(gpu=True)

# -----------------------------------------------------------------------------
# Run segmentation
# -----------------------------------------------------------------------------

masks, flows, styles = model.eval(img, diameter=None,      # CPSAM ignores diameter
)

print(f"Found {masks.max()} cells")

del flows
del styles

# -----------------------------------------------------------------------------
# Save masks as TIFF
# -----------------------------------------------------------------------------

dtype = np.uint16 if masks.max() < 65535 else np.uint32

imwrite(
    output_path,
    masks.astype(dtype),
    compression="zlib",
)

print(f"Saved masks to {output_path}")