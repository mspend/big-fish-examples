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
    cellpose_path = root_path / "big_fish" / "segmentation" / "fiducial_no_downsampling_masks.ome.tiff"

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

        spots_path = results_dir / "9_18_26" / f"spots_bit_{bit}.csv"

        print("=" * 70)
        print(f"Processing bit {bit}")
        print(f"  {spots_path}")

        # Make sure file exists
        if not spots_path.exists():
            print("  WARNING: file does not exist. Skipping.")
            continue

        # Load spots
        # preserve spot_id as a column not the index for downstream analysis
        spots = pd.read_csv(spots_path, index_col=0, usecols=["spot_id", "y", "x", "bit"])
        print(spots.head())

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
        results_path = results_dir / f"spots_bit_{bit}.csv"
        spots.to_csv(results_path)

        print("  Added cell_id and saved.")

    print()
    print("=" * 70)
    print("Finished processing all 16 bits.")


if __name__ == "__main__":
    args = parse_args()
    main(args.root_path)