"""


Shepherd 2025/03 - created script.
Spendlove 2026/04 - 
"""

import argparse
from pathlib import Path
from tifffile import TiffWriter
import zarr

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

    # input path
    root_path = Path(root_path).expanduser().resolve()

    # define input path
    input_path = root_path / "qi2labdatastore" / "fused" / "fused.ome.zarr"

    # define output path
    output_path = root_path / "big_fish" / "tiffs" /  Path("fused.tif")
    output_path.parent.mkdir(parents=True, exist_ok=True)

    z = zarr.open(input_path / "0", mode="r")
    print(z.shape)

    with TiffWriter(output_path, bigtiff=True) as tif:
        tif.write(
            z,
            shape=z.shape,
            dtype=z.dtype,
        )

    print("done")


if __name__ == "__main__":
    args = parse_args()
    main(args.root_path)