import os
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
    cellpose_path = root_path / "qi2labdatastore" / "segmentation" / "cellpose"
    path = os.path.join(cellpose_path, "fiducial_max_projection_cp_masks.tif")
    cell_label = stack.read_image(path).astype(np.int64)
    print("segmented cells")
    print("\r shape: {0}".format(cell_label.shape))
    print("\r dtype: {0}".format(cell_label.dtype), "\n")
    print(cell_label.min(), cell_label.max())

    # maybe it would be better to load this csv using Pandas
    # header is spot_id, y, x, bit_#

    # detected spots
    spots_path = root_path / "big_fish" / "results" / "all_tiles_2D"
    path = os.path.join(spots_path, "spots_bit_1.csv")
    # ignore header using skiprows
    spots = stack.read_array_from_csv(path, delimiter=',', skiprows = 1, dtype=np.int64)
    print("detected spots")
    print("\r shape: {0}".format(spots.shape))
    print("\r dtype: {0}".format(spots.dtype), "\n")
    print("y:", spots[:, 1].min(), spots[:, 1].max())
    print("x:", spots[:, 2].min(), spots[:, 2].max())


    






    fov_results = multistack.extract_cell(
        cell_label=cell_label, 
        ndim=2, 
        rna_coord=spots,
        # image=image_contrasted
        )
    print("number of cells identified: {0}".format(len(fov_results)))






if __name__ == "__main__":
    args = parse_args()
    main(args.root_path)