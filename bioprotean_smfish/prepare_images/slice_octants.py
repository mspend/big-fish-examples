from pathlib import Path
import argparse
import pandas as pd
import numpy as np
import tifffile
from tifffile import TiffWriter

# import os
# import bigfish.stack as stack

def parse_args():
    parser = argparse.ArgumentParser(
        description="Load fused TIFFs, slice into octants and save them as individual TIFFs."
    )
    parser.add_argument(
        "root_path",
        type=Path,
        help="Root experiment folder",
    )
    return parser.parse_args()


def main(root_path: Path):

    root_path = Path(root_path).expanduser().resolve()

    input_path = root_path / "fused"
    output_path = root_path / "fused" / "octants"
    output_path.mkdir(parents=True, exist_ok=True)

    metadata_path = root_path / "scan_metadata.csv"
    scan_metadata = pd.read_csv(metadata_path, index_col=0)

    voxel_zyx_um = [
        scan_metadata["z_voxel_um"][0],
        scan_metadata["yx_voxel_um"][0],
        scan_metadata["yx_voxel_um"][0],
    ]

    coordinates = []

    n_bits = 16
    for bit in range(1, n_bits+1):

        # read in fused image
        filename = f"fused_bit{bit:03d}.ome.tiff"
        path = input_path / filename

        # read in using tiffile
        full_image = tifffile.imread(path)
        assert full_image.ndim == 3, f"Expected ZYX image, got shape {full_image.shape}"

        # optionally, read in using bigfish
        # image = stack.read_image(str(path))
        print(f"Bit {bit} loaded")
    
        z_size, y_size, x_size = full_image.shape
        print(full_image.shape)

        # Compute tile boundaries
        # Split into 4 rows × 2 columns
        y_edges = np.linspace(0, y_size, 5, dtype=int)
        x_edges = np.linspace(0, x_size, 3, dtype=int)

        # 5% TOTAL overlap
        # (half of the overlap extends into each neighboring tile)
        y_overlap = round(y_size * 0.025)
        x_overlap = round(x_size * 0.025)

        octants = {}

        tile = 1

        for row in range(4):

            y0 = y_edges[row]
            y1 = y_edges[row + 1]

            # Extend interior boundaries
            if row > 0:
                y0 -= y_overlap
            if row < 3:
                y1 += y_overlap

            for col in range(2):

                x0 = x_edges[col]
                x1 = x_edges[col + 1]

                if col > 0:
                    x0 -= x_overlap
                if col < 1:
                    x1 += x_overlap

                octants[tile] = full_image[:, y0:y1, x0:x1]

                coordinates.append({
                    "bit": bit,
                    "octant": tile,
                    "z0": 0,
                    "z1": z_size,
                    "y0": y0,
                    "y1": y1,
                    "x0": x0,
                    "x1": x1,
                })

                print(f"Octant {tile}: {octants[tile].shape}")

                tile += 1

        # save each as a tiff
        for number, image in octants.items():

            output_filename = f"fused_bit{bit:03d}_{number}.ome.tiff"
            output_file = (output_path / output_filename)

            print(f"{number}: {image.shape}")

            with TiffWriter(output_file, bigtiff=True) as tif:

                metadata = {
                    "axes": "ZYX",
                    "SignificantBits": 16,
                    "PhysicalSizeX": float(voxel_zyx_um[2]),
                    "PhysicalSizeXUnit": "µm",
                    "PhysicalSizeY": float(voxel_zyx_um[1]),
                    "PhysicalSizeYUnit": "µm",
                    "PhysicalSizeZ": float(voxel_zyx_um[0]),
                    "PhysicalSizeZUnit": "µm",
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

            print(f"\nSaved to:\n{output_file}")

    coordinate_df = pd.DataFrame(coordinates)

    coordinate_df.to_csv(
        output_path / "octant_coordinates.csv",
        index=False
    )

if __name__ == "__main__":
    args = parse_args()
    main(args.root_path)