## this took FOREVER. Ran for 50 minutes

import os
import numpy as np
import bigfish
import bigfish.stack as stack
import bigfish.detection as detection
import bigfish.multistack as multistack
import bigfish.plot as plot
import pandas as pd
import matplotlib.pyplot as plt
from skimage import segmentation
from skimage.segmentation import find_boundaries
import matplotlib.patches as mpatches
from scipy import ndimage
from pathlib import Path
import psfmodels as psfm
from matplotlib.colors import PowerNorm
from sklearn.metrics import r2_score

# Change for your data
data = Path("/data/smfish/20260311_bartelle_smFISH_cryo_48hr_male")

# Load in image data 

# These tiffs are the registered, deconvolved, fused, 2D image
input_dir = data / "fused" / "all_tiles_2D"
path = os.path.join(input_dir, "fused_max_projected_bit005.ome.tiff")
rna = stack.read_image(path)
print("smfish channel")
print("\r shape: {0}".format(rna.shape))
print("\r dtype: {0}".format(rna.dtype))
print("\r nbytes: {0}".format(rna.nbytes))

# polyDT is our fiducial, or reference marker. This probe labels all polyadenylated RNA and is used for CellPose segmentation.
# We load in this data to visualize the cell boundaries.
polyDT_path = data / "fused"
path = os.path.join(polyDT_path, "fiducial_no_downsampling.ome.tiff")
polyDT_mip = stack.read_image(path)
print("polyDT channel")
print("\r shape: {0}".format(polyDT_mip.shape))
print("\r dtype: {0}".format(polyDT_mip.dtype))
print("\r nbytes: {0}".format(polyDT_mip.nbytes))

# Here you see we create the plots using the polyDT maxiumum intensity projection

# stretch the contrast otherwise the spots will be dim and hard to see
polyDT_image_contrasted = stack.rescale(polyDT_mip, channel_to_stretch=0)

image_contrasted = stack.rescale(rna, channel_to_stretch=0)

# segmented cells
segmentation = data / "big_fish" / "segmentation"
path = os.path.join(segmentation, "fiducial_no_downsampling_masks.ome.tiff")
cell_label = stack.read_image(path)
print("segmented cells")
print("\r shape: {0}".format(cell_label.shape))
print("\r dtype: {0}".format(cell_label.dtype), "\n")

# detected spots
output_dir = data / "big_fish" / "results" / "all_tiles_2D"
spots_path = os.path.join(output_dir, "spots_bit_5.csv")
spots = pd.read_csv(spots_path, index_col=0, usecols=["spot_id", "y", "x", "bit"])
print("detected spots")
print("\r shape: {0}".format(spots.shape))
print(spots.head())

# visuzalize the Cellpose segmentations on top of the polyDT channel

# Create figure and axes
fig, ax = plt.subplots(figsize=(12, 10))

# Display the polyDT_mip image
ax.imshow(polyDT_image_contrasted, cmap='gray')

# Overlay the cellpose segmentation masks as outlines using contour
# asks Matplotlib to calculate contours corresponding to potentially thousands of individual values across a 626-million-pixel image
ax.contour(cell_label, levels=np.unique(cell_label)[1:], colors='red', linewidths=0.5, alpha=0.7)

ax.set_title('PolyDT fiducial channel with Cellpose segmentation masks', fontsize=14)
ax.set_xlabel('X (pixels)')
ax.set_ylabel('Y (pixels)')

plt.tight_layout()

# Save as PDF
plt.savefig(
   segmentation / "cell_outlines_over_polyDT.pdf", 
   bbox_inches="tight")