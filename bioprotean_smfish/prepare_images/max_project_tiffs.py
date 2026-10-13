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
    output_path = root_path / "fused" / "z_max_projected"

    # Create output directory if needed
    output_path.mkdir(parents=True, exist_ok=True)

    metadata_path = root_path / "scan_metadata.csv"
    metadata = pd.read_csv(metadata_path, index_col=0)

    # Obtain camera metadata
    # NA = numerical aperture
    na = metadata['na'][0]
    z_voxel = metadata['z_voxel_um'][0] 
    yx_voxel = metadata['yx_voxel_um'][0] 
    voxel_zyx_um = [z_voxel, yx_voxel, yx_voxel]

    n_bits = 16

    # because range is exclusive of the stop
    for bit in range(1, n_bits+1):

        # Load in data 
        # These tiffs are the globally registered, deconvolved image
        path = os.path.join(input_path, "fused_bit" +str(bit).zfill(3) + ".ome.tiff")
        fused_readout = stack.read_image(path)
        print(f"Bit {bit} loaded")

        print(fused_readout.shape)

        # create max projection
        max_projection = np.max(np.squeeze(fused_readout), axis=0)

        filename = "fused_max_projected_bit" +str(bit).zfill(3) + ".ome.tiff"

        filename_path = (
            output_path
            / Path(filename)
        )
        with TiffWriter(filename_path, bigtiff=True) as tif:
            metadata = {
                "axes": "YX",
                "SignificantBits": 16,
                "PhysicalSizeX": float(voxel_zyx_um[2]),
                "PhysicalSizeXUnit": "µm",
                "PhysicalSizeY": float(voxel_zyx_um[1]),
                "PhysicalSizeYUnit": "µm",
            }
            options = {
                "compression": "zlib",
                "compressionargs": {"level": 8},
                "predictor": True,
                "photometric": "minisblack",
                "resolutionunit": "CENTIMETER",
            }
            tif.write(
                max_projection,
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