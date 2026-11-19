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
path_input = "/data/smfish/20260311_bartelle_smFISH_cryo_48hr_male/fused/one_tile/tile0022/z_max_projected/"
path_output = "/data/smfish/20260311_bartelle_smFISH_cryo_48hr_male/big_fish/results/one_tile_2D/tile0022"

n_bits = 16
tile_idx = 22

# All_spots is a list to which we will append all the results of the spot detection
all_spots = []

# because range is exclusive of the stop
for bit in range(1, n_bits + 1):
    # Load in data 
    # These tiffs are the registered, deconvolved image
    path = os.path.join(path_input, "tile"+str(tile_idx).zfill(4)+"_max_projected_bit"+str(bit).zfill(3)+".ome.tiff")
    rna = stack.read_image(path)

    # # Create a maxiumum intensity projection of the RNA channel
    # rna_mip = stack.maximum_projection(rna)

    # Detect spots in 2D
    # Does not use the polyDT fiducial channel
    spots, threshold = detection.detect_spots(
        images=rna, 
        return_threshold=True, 
        voxel_size=(103, 103),  # in nanometer (one value per dimension yx)
        spot_radius=(150, 150))  # in nanometer (one value per dimension yx)

    spots_df = pd.DataFrame(spots, columns=['y', 'x',])
    spots_df['bit'] = bit
    # Append the dataframe of the spots to the list all_spots
    all_spots.append(spots_df)
    print(f'Done with bit {bit}')

# Concatenate the spots from all bits
spots_df = pd.concat(all_spots, ignore_index=True)

# Save all detected spots to one Numpy file while preserving the bit identity.
# A structured array keeps y/x coordinates separate from the categorical bit feature.
os.makedirs(path_output, exist_ok=True)
spots_npy_dtype = np.dtype([
    ('y', np.float64),
    ('x', np.float64),
    ('bit', np.int32),
])
spots_npy = np.empty(len(spots_df), dtype=spots_npy_dtype)
spots_npy['y'] = spots_df['y'].to_numpy()
spots_npy['x'] = spots_df['x'].to_numpy()
spots_npy['bit'] = spots_df['bit'].to_numpy(dtype=np.int32)

spots_npy_path = os.path.join(path_output, "spots_all_bits.npy")
np.save(spots_npy_path, spots_npy)
print(f"Saved {len(spots_npy)} spots with bit identities to {spots_npy_path}")


# # save in csv files
# # The header of the csv file is y, x, bit # representing the coordinates of the spot and its identity.
path = os.path.join(path_output, "spots.csv")
stack.save_data_to_csv(spots_df, path, delimiter=',')

# # Summarize results
# # include all 16 bits even if one has zero spots
# summary = (
#     spots_df['bit']
#     .value_counts()
#     .reindex(range(1,17), fill_value=0)
#     .rename_axis('bit')
#     .reset_index(name='fish_spots')
# )
# # Gene IDs, in order of bit number
# gene_names = ['Angpt1',
#     'Crmp1',
#     'Dcx',
#     'Hdac11',
#     'Itgam',
#     'Notch2',
#     'Ptch1',
#     'Serpine1',
#     'Shh',
#     'Slc1a3',
#     'Tcf12',
#     'Tek',
#     'Tgfb1',
#     'Tgfbi',
#     'Aif1',
#     'Nnat']


# summary['gene_name'] = gene_names

# path = os.path.join(path_output, "summary_spots.csv")
# stack.save_data_to_csv(summary, path, delimiter=',')