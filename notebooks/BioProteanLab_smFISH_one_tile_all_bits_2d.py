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

n_bits = 16
tile_idx = 0

# All_spots is a list to which we will append all the results of the spot detection
all_spots = []

# because range is exclusive of the stop
for bit in range(1, n_bits + 1):
    # Load in data 
    # These tiffs are the registered, deconvolved image
    path = os.path.join(path_input, "tile"+str(tile_idx).zfill(3)+"bit"+str(bit).zfill(3)+".ome.tiff")
    rna = stack.read_image(path)

    # Create a maxiumum intensity projection of the RNA channel
    rna_mip = stack.maximum_projection(rna)

    # Detect spots in 2D
    # Does not use the polyDT fiducial channel
    spots, threshold = detection.detect_spots(
        images=rna_mip, 
        return_threshold=True, 
        voxel_size=(103, 103),  # in nanometer (one value per dimension yx)
        spot_radius=(150, 150))  # in nanometer (one value per dimension yx)
    
    spots_df = pd.DataFrame(spots, columns=['y', 'x',])
    spots_df['bit'] = bit
    # Append the dataframe of the spots to the list all_spots
    all_spots.append(spots_df)
    print(f'Done with bit {bit}')


    # spots_post_decomposition, dense_regions, reference_spot = detection.decompose_dense(
    #     image=rna_mip, 
    #     spots=spots, 
    #     voxel_size=(103, 103), 
    #     spot_radius=(150, 150), 
    #     alpha=0.7,  # alpha impacts the number of spots per candidate region
    #     beta=1,  # beta impacts the number of candidate regions to decompose
    #     gamma=5)  # gamma the filtering step to denoise the image
    # print("detected spots before decomposition")
    # print("\r shape: {0}".format(spots.shape))
    # print("detected spots after decomposition")
    # print("\r shape: {0}".format(spots_post_decomposition.shape))

    # spots_df = pd.DataFrame(spots_post_decomposition, columns=['y', 'x',])

# Concatenate the spots from all bits
spots_df = pd.concat(all_spots, ignore_index=True)

# # save in csv files
# # The header of the csv file is y, x, bit # representing the coordinates of the spot and its identity.
path = os.path.join(path_output, "spots.csv")
stack.save_data_to_csv(spots_df, path, delimiter=',')

# Summarize results
# include all 16 bits even if one has zero spots
summary = (
    spots_df['bit']
    .value_counts()
    .reindex(range(1,17), fill_value=0)
    .rename_axis('bit')
    .reset_index(name='fish_spots')
)

path = os.path.join(path_output, "summary_spots.csv")
stack.save_data_to_csv(summary, path, delimiter=',')