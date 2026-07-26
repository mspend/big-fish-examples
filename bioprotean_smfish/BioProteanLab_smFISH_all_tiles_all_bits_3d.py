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
from pathlib import Path
import argparse
import time

def parse_args():
    parser = argparse.ArgumentParser(
        description="Detect smFISH spots in 3D"
    )
    parser.add_argument(
        "root_path",
        type=Path,
        help="Root experiment folder (example: /data/smFISH/20251028_bartelle_smFISH_mm_microglia_newbuffers)",
    )
    return parser.parse_args()

def main(root_path: Path):

    root_path = Path(root_path).expanduser().resolve()

    input_path = root_path / "fused" / "tile0000"
    output_path = root_path / "big_fish" / "results" / "all_tiles_3D"

    # Create output directory if needed
    output_path.mkdir(parents=True, exist_ok=True)

    metadata_path = root_path / "scan_metadata.csv"
    metadata = pd.read_csv(metadata_path, index_col=0)

    # Obtain camera metadata
    # NA = numerical aperture
    # provide voxel size in nanometer
    na = metadata['na'][0]
    z_voxel = metadata['z_voxel_um'][0] * 1000 # in nanometer
    yx_voxel = metadata['yx_voxel_um'][0] * 1000 # in nanometer
    voxel_size = [z_voxel, yx_voxel, yx_voxel]

    # Wavelengths of the channels
    lambda_red = 670 # Alexa647
    lambda_yellow = 590 # Atto565

    n_bits = 16

    # All_spots is a list to which we will append all the results of the spot detection
    all_spots = []

    print("ready for spot detection")

    # because range is exclusive of the stop
    for bit in range(1, n_bits+1):

        # Load in data 
        # These tiffs are the globally registered, deconvolved image
        path = os.path.join(input_path, "tile000bit" +str(bit).zfill(3) + ".ome.tiff")
        rna = stack.read_image(path)
        # rna = rna.astype(np.uint16)
        print(f"Bit {bit} loaded")

        # Spot radius calculated using Abbe’s diffraction formula for lateral (XY) resolution is: d = λ/(2NA)
        # Abbe’s diffraction formula for axial (Z) resolution is: d = 2λ/(NA)2

        if bit % 2 == 1: 
            spot_radius_yx = (lambda_yellow / (2 * na))

            spot_radius_z = (2* lambda_yellow / (2 * na))
            spot_radius = [spot_radius_z, spot_radius_yx, spot_radius_yx]

        if bit % 2 == 0: 
            spot_radius_yx = (lambda_red  / (2 * na))
            spot_radius_z = (2* lambda_red  / (2 * na))
            spot_radius = [spot_radius_z, spot_radius_yx, spot_radius_yx]

        print(spot_radius)
  
        # # Detect spots in 3D 
        # spots, threshold = detection.detect_spots(
        #     images=rna, 
        #     return_threshold=True, 
        #     voxel_size=voxel_size,  # in nanometer (one value per dimension zyx)
        #     spot_radius=spot_radius)  # in nanometer (one value per dimension zyx)
        spots = [
            ["a", "a", "a"],
            ["b", "b", "b"],
            ["c", "c", "c"]]

        # The function detect_spots returns the coordinates (or list of coordinates) 
        # of the spots with shape (nb_spots, 3) for 3D images.
        print(f"Spot detection for bit {bit} complete")

        # print("\r shape: {0}".format(spots.shape))
        # print("\r dtype: {0}".format(spots.dtype))
        # print("\r threshold: {0}".format(threshold))

        spots_df = pd.DataFrame(spots, columns=['z', 'y', 'x'])
        # print(spots_df)
        spots_df['bit'] = bit
        print(spots_df)

        path = os.path.join(output_path, (f"spots_bit_{bit}.csv"))
        spots_df.to_csv(path)

        print(f'Done with bit {bit}')


if __name__ == "__main__":
    args = parse_args()
    main(args.root_path)