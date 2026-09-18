import argparse
from pathlib import Path
import numpy as np
import pandas as pd
import bigfish.stack as stack


def parse_args():
    parser = argparse.ArgumentParser(
        description="Assign Cellpose cell IDs to detected smFISH spots"
    )
    parser.add_argument(
        "root_path",
        type=Path,
        help=(
            "Root experiment folder "
            "(example: /data/smFISH/20251028_bartelle_smFISH_mm_microglia_newbuffers)"
        ),
    )
    return parser.parse_args()


def main(root_path: Path):

    root_path = Path(root_path).expanduser().resolve()

    # -------------------------------------------------------------------------
    # Load Cellpose segmentation
    # -------------------------------------------------------------------------
    cellpose_path = root_path / "fused" / "fiducial_no_downsampling.ome_masks.tiff"

    print(f"Loading segmentation:\n  {cellpose_path}")

    cell_label = stack.read_image(str(cellpose_path))

    print("Segmented cells")
    print(f"  shape: {cell_label.shape}")
    print(f"  dtype: {cell_label.dtype}")
    print(f"  number of labels: {cell_label.max():,}")
    print()

    # Make sure segmentation is 2D
    if cell_label.ndim != 2:
        raise ValueError(
            f"Expected a 2D segmentation mask, but got shape {cell_label.shape}"
        )

    height, width = cell_label.shape

    # -------------------------------------------------------------------------
    # Directory containing spot detection results
    # -------------------------------------------------------------------------
    results_dir = (
        root_path
        / "big_fish"
        / "results"
        / "all_tiles_2D"
    )

    # -------------------------------------------------------------------------
    # Process bits 1-16
    # -------------------------------------------------------------------------
    for bit in range(1, 17):

        spots_path = results_dir / f"spots_bit_{bit}.csv"

        print("=" * 70)
        print(f"Processing bit {bit}")
        print(f"  {spots_path}")

        # Make sure file exists
        if not spots_path.exists():
            print("  WARNING: file does not exist. Skipping.")
            continue

        # Load spots
        # preserve spot_id as a column not the index for downstream analysis
        spots = pd.read_csv(spots_path, header=0, names = ["spot_id", "y", "x", "bit"])

        print(f"  spots: {len(spots):,}")

        # Check required columns
        required_columns = {"y", "x"}

        if not required_columns.issubset(spots.columns):
            raise ValueError(
                f"{spots_path} does not contain required columns "
                f"'y' and 'x'. Columns found: {list(spots.columns)}"
            )

        # ---------------------------------------------------------------------
        # Convert spot coordinates to integer pixel indices
        # ---------------------------------------------------------------------
        y = spots["y"].to_numpy(dtype=np.int64)
        x = spots["x"].to_numpy(dtype=np.int64)

        # ---------------------------------------------------------------------
        # Check that all coordinates are inside the segmentation image
        # ---------------------------------------------------------------------
        valid = (
            (y >= 0)
            & (y < height)
            & (x >= 0)
            & (x < width)
        )

        n_invalid = np.count_nonzero(~valid)

        if n_invalid > 0:
            raise ValueError(
                f"Bit {bit} contains {n_invalid:,} spot coordinates outside "
                f"the segmentation image bounds ({height}, {width}). "
                "Original CSV has NOT been overwritten."
            )

        # ---------------------------------------------------------------------
        # Assign Cellpose label to each spot
        # ---------------------------------------------------------------------
        spots["cell_id"] = cell_label[y, x]

        # Count spots inside/outside segmented cells
        n_inside = np.count_nonzero(spots["cell_id"].to_numpy() != 0)
        n_outside = len(spots) - n_inside

        print(f"  spots inside cells:  {n_inside:,}")
        print(f"  spots outside cells: {n_outside:,}")

        # ---------------------------------------------------------------------
        # Overwrite original CSV
        # ---------------------------------------------------------------------
        spots.to_csv(spots_path)

        print("  Added cell_id and saved.")

    print()
    print("=" * 70)
    print("Finished processing all 16 bits.")


if __name__ == "__main__":
    args = parse_args()
    main(args.root_path)












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