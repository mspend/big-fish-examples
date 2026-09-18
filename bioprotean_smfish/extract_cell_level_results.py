import argparse
from pathlib import Path
import numpy as np
import pandas as pd
import bigfish
import bigfish.stack as stack

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

    # segmented cells
    cellpose_path = root_path / "fused" / "fiducial_no_downsampling.ome_masks.tiff"
    cell_label = stack.read_image(str(cellpose_path))
    print("segmented cells")
    print("\r shape: {0}".format(cell_label.shape))
    print("\r dtype: {0}".format(cell_label.dtype), "\n")

    # detected spots
    spots_path = root_path / "big_fish" / "results" / "all_tiles_2D" / "spots_bit_1.csv"
    spots = pd.read_csv(spots_path,index_col=0)
    print("detected spots")
    print("\r shape: {0}".format(spots.shape))

    # Coordinates must be integer pixel indices
    y = spots["y"].to_numpy(dtype=np.int64)
    x = spots["x"].to_numpy(dtype=np.int64)

    # Assign each spot to the label at that pixel
    spots["cell_id"] = cell_label[y, x]

    output_path = root_path / "big_fish" / "results" / "all_tiles_2D" / "spots_with_cell_ids.csv"
    spots.to_csv(str(output_path))

if __name__ == "__main__":
    args = parse_args()
    main(args.root_path)