import numpy as np
import bigfish
import bigfish.stack as stack
import bigfish.multistack as multistack
import bigfish.plot as plot
import matplotlib.pyplot as plt
from skimage import segmentation
import matplotlib.patches as mpatches
from scipy import ndimage
import argparse
from pathlib import Path
import pandas as pd


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

    # maybe it would be better to load this csv using Pandas
    # header is spot_id, y, x, bit_#

    # detected spots
    spots_path = root_path / "big_fish" / "results" / "all_tiles_2D"
    spots_path = root_path / "big_fish" / "results" / "all_tiles_2D" / "spots_bit_1.csv"
    spots = stack.read_array_from_csv(spots_path, skiprows=1, delimiter=',', dtype=np.int64)
    print("detected spots")
    print("\r shape: {0}".format(spots.shape))
    print("\r dtype: {0}".format(spots.dtype), "\n")
    # Check that the coordinates for the spots are on the same scale as the cellpose image
    spot_id = spots[:, 0]
    y = spots[:, 1]
    x = spots[:, 2]
    bit = spots[:, 3]

    print(f"x: min={x.min()}, max={x.max()}")
    print(f"y: min={y.min()}, max={y.max()}")

    # # load in smFISH image
    # image_path = root_path / "fused" / "all_tiles_2D" / "fused_max_projected_bit001.ome.tiff"
    # image = stack.read_image(str(image_path))
    # print("smfish channel")
    # print("\r shape: {0}".format(image.shape))
    # print("\r dtype: {0}".format(image.dtype))

    fov_results = multistack.extract_cell(
        cell_label=cell_label, 
        ndim=3, 
        rna_coord=spots, 
        # image=image,
    )
    print("number of cells identified: {0}".format(len(fov_results)))

if __name__ == "__main__":
    args = parse_args()
    main(args.root_path)