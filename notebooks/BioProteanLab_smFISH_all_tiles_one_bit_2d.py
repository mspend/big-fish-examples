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
print("Big-FISH version: {0}".format(bigfish.__version__))

def parse_args():
    parser = argparse.ArgumentParser(
        description="Fuse channels and export per-channel OME-TIFFs."
    )
    parser.add_argument(
        "root_path",
        type=Path,
        help="Root experiment folder (example: /data/smFISH/20251028_bartelle_smFISH_mm_microglia_newbuffers)",
    )
    return parser.parse_args()

def main(root_path: Path):

    root_path = Path(root_path).expanduser().resolve()

    input_dir = root_path / "qi2labdatastore" / "big_fish" / "tiffs"
    output_dir = root_path / "qi2labdatastore" / "big_fish" / "results" / "all_tiles_2D"
    segmentation = root_path / "qi2labdatastore" / "segmentation" / "cellpose"
    metadata_dir = root_path / "scan_metadata.csv"

    # Create output directory if needed
    output_dir.mkdir(parents=True, exist_ok=True)

    # Load in data 

    # These tiffs are the registered, deconvolved image
    path = os.path.join(input_dir, "fused_bit005.ome.tiff")
    rna = stack.read_image(path)
    # rna = rna.astype(np.uint16)

    # create z-maxiumum projection
    rna_mip = stack.maximum_projection(rna)
    print("Data loaded")

    # # polyDT is our fiducial, or reference marker. This probe labels all polyadenylated RNA.
    # # Segmentation is performed on the 3D polyDT data using Cellpose
    # # We load in Cellpose masks to visualize the cell boundaries.
    # path = os.path.join(segmentation, "polyDT_max_projection.ome.tiff")
    # polyDT_masks = stack.read_image(path)
    # print("polyDT channel")
    # print("\r shape: {0}".format(polyDT_masks.shape))
    # print("\r dtype: {0}".format(polyDT_masks.dtype), "\n")

    metadata = pd.read_csv(metadata_dir, index_col=0)

    # Obtain camera metadata
    # NA stands for numerical aperture
    # provide voxel size in nanometer
    na = metadata['na'][0]
    # z_voxel = metadata['z_voxel_um'][0] * 1000 # in nanometer
    yx_voxel = metadata['yx_voxel_um'][0] * 1000 # in nanometer

    # Wavelengths of the channels
    lambda_red = 670 # Alexa647
    lambda_yellow = 590 # Atto565

    print(yx_voxel)
    # print(z_voxel)

    voxel_size = [yx_voxel, yx_voxel]

    # Calculated using Abbe’s diffraction formula for lateral (XY) resolution is: d = λ/(2NA)
    # Abbe’s diffraction formula for axial (Z) resolution is: d = 2λ/(NA)2
    spot_radius_xy = (lambda_yellow / (2 * na))
    # spot_radius_z = (2* lambda_yellow / (2 * na))
    spot_radius = [spot_radius_xy, spot_radius_xy]

    print(voxel_size)
    print(spot_radius)

    start = time.perf_counter()
    # Detect spots in 3D 
    # Detection in 3D takes ~5 minutes
    # Does not use the polyDT channel
    spots, threshold = detection.detect_spots(
        images=rna_mip, 
        return_threshold=True, 
        voxel_size=voxel_size,  # in nanometer (one value per dimension zyx)
        spot_radius=spot_radius)  # in nanometer (one value per dimension zyx)
    end = time.perf_counter()

    # The function detect_spots returns the coordinates (or list of coordinates) 
    # of the spots with shape (nb_spots, 3) for 3D images.
    print("detected spots")
    print("\r shape: {0}".format(spots.shape))
    print("\r dtype: {0}".format(spots.dtype))
    print("\r threshold: {0}".format(threshold))

    spots_df = pd.DataFrame(spots, columns=["y", "x"])
    print(spots_df.head())

    print(f"spot detection time: {end - start:.6f} seconds")

    # save results
    # save in npy files
    path = os.path.join(output_dir, "bit5_spots.npy")
    stack.save_array(spots, path)

    # save in csv files
    # The header of the csv file is y, x, cluster identity #
    spots_df = pd.DataFrame(spots, columns=['y', 'x'])
    path = os.path.join(output_dir, "bit5_spots.csv")
    stack.save_data_to_csv(spots_df, path, delimiter=',')

if __name__ == "__main__":
    args = parse_args()
    main(args.root_path)