import os
from pathlib import Path
import argparse
import pandas as pd
import bigfish.stack as stack
from tifffile import TiffWriter


def parse_args():
    parser = argparse.ArgumentParser(
        description="Load quadrant TIFFs and save one eighth of the image."
    )
    parser.add_argument(
        "root_path",
        type=Path,
        help="Root experiment folder",
    )
    return parser.parse_args()


def main(root_path: Path):

    root_path = Path(root_path).expanduser().resolve()

    input_path = root_path / "fused" / "quadrants"
    output_path = root_path / "fused" / "eighth_test"
    output_path.mkdir(parents=True, exist_ok=True)

    metadata_path = root_path / "scan_metadata.csv"
    scan_metadata = pd.read_csv(metadata_path, index_col=0)

    voxel_zyx_um = [
        scan_metadata["z_voxel_um"][0],
        scan_metadata["yx_voxel_um"][0],
        scan_metadata["yx_voxel_um"][0],
    ]

    bit = 1

    quadrant_names = [
        "top_left",
        "top_right",
        "bottom_left",
        "bottom_right",
    ]

    quadrants = {}

    # Load each quadrant
    for name in quadrant_names:

        filename = f"fused_bit{bit:03d}_{name}.ome.tiff"
        path = input_path / filename

        image = stack.read_image(str(path))
        quadrants[name] = image

        print(f"{name}: {image.shape}")

    # Find the smallest quadrant
    smallest_name = min(quadrants, key=lambda k: quadrants[k].size)
    smallest = quadrants[smallest_name]

    print(f"\nSmallest quadrant: {smallest_name}")
    print(f"Shape: {smallest.shape}")

    z, y, x = smallest.shape

    # Split along the smaller spatial axis
    if y <= x:
        print("Splitting along Y axis")
        midpoint = y // 2
        eighth = smallest[:, :midpoint, :]
    else:
        print("Splitting along X axis")
        midpoint = x // 2
        eighth = smallest[:, :, :midpoint]

    print(f"Eighth shape: {eighth.shape}")

    output_file = output_path / f"fused_bit{bit:03d}_{smallest_name}_eighth.ome.tiff"

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
            eighth,
            resolution=(
                1e4 / float(voxel_zyx_um[2]),
                1e4 / float(voxel_zyx_um[1]),
            ),
            **options,
            metadata=metadata,
        )

    print(f"\nSaved test TIFF to:\n{output_file}")


if __name__ == "__main__":
    args = parse_args()
    main(args.root_path)