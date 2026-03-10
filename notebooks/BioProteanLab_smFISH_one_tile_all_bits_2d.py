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
import matplotlib.patches as mpatches
from scipy import ndimage

# hard-code the paths of our input and output directories
path_input = "/data/smFISH/20251028_bartelle_smFISH_mm_microglia_newbuffers/qi2labdatastore/big_fish/tiffs"
path_output = "/data/smFISH/20251028_bartelle_smFISH_mm_microglia_newbuffers/qi2labdatastore/big_fish/results/one_tile_2D"

# Load in data 
# These tiffs are the registered, deconvolved image
path = os.path.join(path_input, "tile000bit005.ome.tiff")
rna = stack.read_image(path)
print("smfish channel")
print("\r shape: {0}".format(rna.shape))

# polyDT is our fiducial, or reference marker. This probe labels all polyadenylated RNA and is used for CellPose segmentation.
# We load in this data to visualize the cell boundaries.
path = os.path.join(path_input, "max_projected_tile000round000corrected_polyDT.ome.tiff")
polyDT_mip = stack.read_image(path)
print("polyDT channel")
print("\r shape: {0}".format(polyDT_mip.shape))

# Create a maxiumum intensity projection of the RNA channel
rna_mip = stack.maximum_projection(rna)
print("smfish channel (2D maximum projection)")
print("\r shape: {0}".format(rna_mip.shape))

# Detect spots in 2D
# Does not use the polyDT fiducial channel
spots, threshold = detection.detect_spots(
    images=rna_mip, 
    return_threshold=True, 
    voxel_size=(103, 103),  # in nanometer (one value per dimension yx)
    spot_radius=(150, 150))  # in nanometer (one value per dimension yx)

# The function detect_spots returns the coordinates (or list of coordinates) 
# of the spots with shape (nb_spots, 3) or (nb_spots, 2), for 3-d or 2-d images respectively.
print("detected spots")
print("\r shape: {0}".format(spots.shape))
print("\r threshold: {0}".format(threshold))

spots_post_decomposition, dense_regions, reference_spot = detection.decompose_dense(
    image=rna_mip, 
    spots=spots, 
    voxel_size=(103, 103), 
    spot_radius=(150, 150), 
    alpha=0.7,  # alpha impacts the number of spots per candidate region
    beta=1,  # beta impacts the number of candidate regions to decompose
    gamma=5)  # gamma the filtering step to denoise the image
print("detected spots before decomposition")
print("\r shape: {0}".format(spots.shape))
print("detected spots after decomposition")
print("\r shape: {0}".format(spots_post_decomposition.shape))

# save in csv files
# The header of the csv file is y, x, cluster identity #
spots_df = pd.DataFrame(spots_post_decomposition, columns=['y', 'x',])
path = os.path.join(path_output, "spots.csv")
stack.save_data_to_csv(spots_df, path, delimiter=',')