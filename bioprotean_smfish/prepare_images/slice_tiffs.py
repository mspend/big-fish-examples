import os
import numpy as np
import bigfish
import bigfish.stack as stack
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
import tifffile
from tifffile import TiffWriter

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

    input_path = root_path / "fused"
    output_path = root_path / "fused" / "quadrants"

    # Create output directory if needed
    output_path.mkdir(parents=True, exist_ok=True)

    metadata_path = root_path / "scan_metadata.csv"
    scan_metadata = pd.read_csv(metadata_path, index_col=0)

    # Obtain camera metadata
    # NA = numerical aperture
    na = scan_metadata['na'][0]
    z_voxel = scan_metadata['z_voxel_um'][0] 
    yx_voxel = scan_metadata['yx_voxel_um'][0] 
    voxel_zyx_um = [z_voxel, yx_voxel, yx_voxel]

    n_bits = 16

    # # because range is exclusive of the stop
    # for bit in range(1, n_bits+1):

    bit = 1

    # Load in data 
    # These tiffs are the globally registered, deconvolved image
    path = os.path.join(input_path, "fused_bit" +str(bit).zfill(3) + ".ome.tiff")
    fused_readout = stack.read_image(path)
    print(f"Bit {bit} loaded")

    print(fused_readout.shape)
    shape = fused_readout.shape
    z_size = fused_readout.shape[0]
    y_size = fused_readout.shape[1]
    x_size = fused_readout.shape[2]

    # 5% TOTAL overlap between neighboring quadrants
    # Half of the overlap extends into each quadrant
    y_midpoint = round(y_size/2)
    y_overlap = round(y_size * 0.025)

    x_midpoint = round(x_size/2)
    x_overlap = round(x_size * 0.025)

    top_left = fused_readout[:, :y_midpoint + y_overlap, :x_midpoint + x_overlap]

    top_right = fused_readout[:, :y_midpoint + y_overlap, x_midpoint - x_overlap:]

    bottom_left = fused_readout[:, y_midpoint - y_overlap:, :x_midpoint + x_overlap]
    
    bottom_right = fused_readout[:, y_midpoint - y_overlap:, x_midpoint - x_overlap:]

    quadrants = {
        "top_left": top_left,
        "top_right": top_right,
        "bottom_left": bottom_left,
        "bottom_right": bottom_right,
    }

    for name, image in quadrants.items():

        filename = "fused_bit" +str(bit).zfill(3) + "_" + name + ".ome.tiff"
        print(filename)

        image_path = (
            output_path
            / Path(filename)
        )
        print(image_path)
        print(f"Saving {name}: {image.shape}")

        with TiffWriter(image_path, bigtiff=True) as tif:
            metadata = {
                "axes": "ZYX",
                "SignificantBits": 16,
                "PhysicalSizeX": float(voxel_zyx_um[2]),
                "PhysicalSizeXUnit": "µm",
                "PhysicalSizeY": float(voxel_zyx_um[1]),
                "PhysicalSizeYUnit": "µm",
                'PhysicalSizeZ': float(voxel_zyx_um[0]),
                'PhysicalSizeZUnit': 'µm',                
            }
            options = {
                "compression": "zlib",
                "compressionargs": {"level": 8},
                "predictor": True,
                "photometric": "minisblack",
                "resolutionunit": "CENTIMETER",
            }
            tif.write(
                image,
                resolution=(
                    1e4 / float(voxel_zyx_um[2]),
                    1e4 / float(voxel_zyx_um[1]),
                ),
                **options,
                metadata=metadata,
            )

    print(f'Done with bit {bit}')


if __name__ == "__main__":
    args = parse_args()
    main(args.root_path)