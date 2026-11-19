import os
import numpy as np
import bigfish
import bigfish.stack as stack
import bigfish.detection as detection
import bigfish.multistack as multistack
import pandas as pd
from pathlib import Path
import argparse

def parse_args():
    parser = argparse.ArgumentParser(
        description=""
    )
    parser.add_argument(
        "root_path",
        type=Path,
    )
    return parser.parse_args()

def main(root_path: Path):

    root_path = Path(root_path).expanduser().resolve()

    input_path = root_path / "fused" / "one_tile" / "tile0022" 
    output_path = root_path / "big_fish" / "results" / "one_tile_3D" / "tile0022"

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

    # could change this so it's accepted from user instead of being hard coded
    tile_idx = 22

    # All_spots is a list to which we will append all the results of the spot detection
    all_spots = []

    print("ready for spot detection")

    # because range is exclusive of the stop
    for bit in range(1, n_bits + 1):
        # Load in data 
        # These tiffs are the registered, deconvolved image
        path = os.path.join(input_path, "tile"+str(tile_idx).zfill(4)+"bit"+str(bit).zfill(3)+".ome.tiff")
        rna = stack.read_image(path)
        print(f"Bit {bit} loaded")

        ndim = rna.ndim

        if ndim == 2:
            is_3d = False
        elif ndim == 3:
            is_3d = True
        else:
            raise ValueError(
                f"Expected a 2D or 3D image, got shape {rna.shape}")

        if is_3d:
            voxel_size = [z_voxel, yx_voxel, yx_voxel]
        else:
            voxel_size = [yx_voxel, yx_voxel]

        if bit % 2 == 1: 
            spot_radius_yx = (lambda_yellow / (2 * na))

            spot_radius_z = (2* lambda_yellow / (2 * na))
            if is_3d:
                spot_radius = [spot_radius_z, spot_radius_yx, spot_radius_yx]
            else:
                spot_radius = [spot_radius_yx, spot_radius_yx]
            

        if bit % 2 == 0: 
            spot_radius_yx = (lambda_red  / (2 * na))
            spot_radius_z = (2* lambda_red  / (2 * na))
            if is_3d:
                spot_radius = [spot_radius_z, spot_radius_yx, spot_radius_yx]
            else:
                spot_radius = [spot_radius_yx, spot_radius_yx]

        # Detect spots
        # Does not use the polyDT fiducial channel
        spots, threshold = detection.detect_spots(
            images=rna, 
            return_threshold=True, 
            voxel_size=voxel_size,  # in nanometer (one value per dimension yx)
            spot_radius=spot_radius)  # in nanometer (one value per dimension yx)

        print(f"Spot detection for bit {bit} complete")

        if is_3d:
            spots_df = pd.DataFrame(
                spots,
                columns=["z", "y", "x"],
            )
        else:
            spots_df = pd.DataFrame(
                spots,
                columns=["y", "x"],
            )      

        spots_df['bit'] = bit

        # Append the dataframe of the spots to the list all_spots
        all_spots.append(spots_df)

        # save as csv, one per bit
        path = os.path.join(output_path, (f"spots_bit_{bit}.csv"))
        spots_df.to_csv(path)

        print(f'Done with bit {bit}')

    # Concatenate the spots from all bits
    spots_df = pd.concat(all_spots, ignore_index=True)

    # Save all detected spots to one Numpy file while preserving bit identity.
    # A structured array keeps y/x coordinates separate from the categorical bit feature.
    if is_3d:
        spots_npy_dtype = np.dtype([
            ("z", np.float64),
            ("y", np.float64),
            ("x", np.float64),
            ("bit", np.int32),
        ])
        spots_npy = np.empty(len(spots_df), dtype=spots_npy_dtype)
        spots_npy["z"] = spots_df["z"].to_numpy(dtype=np.float64)
        spots_npy["y"] = spots_df["y"].to_numpy(dtype=np.float64)
        spots_npy["x"] = spots_df["x"].to_numpy(dtype=np.float64)
    else:
        spots_npy_dtype = np.dtype([
            ("y", np.float64),
            ("x", np.float64),
            ("bit", np.int32),
        ])
        spots_npy = np.empty(len(spots_df), dtype=spots_npy_dtype)
        spots_npy["y"] = spots_df["y"].to_numpy(dtype=np.float64)
        spots_npy["x"] = spots_df["x"].to_numpy(dtype=np.float64)

    spots_npy["bit"] = spots_df["bit"].to_numpy(dtype=np.int32)

    spots_npy_path = output_path / "spots_all_bits.npy"
    np.save(spots_npy_path, spots_npy)
    print(f"Saved {len(spots_npy)} spots with bit identities to {spots_npy_path}")

    # # save in csv files
    # # The header of the csv file is y, x, bit # representing the coordinates of the spot and its identity.
    path = str(output_path / "spots.csv")
    stack.save_data_to_csv(spots_df, path, delimiter=',')



if __name__ == "__main__":
    args = parse_args()
    main(args.root_path)