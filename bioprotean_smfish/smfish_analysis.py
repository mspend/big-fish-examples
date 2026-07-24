import argparse
from pathlib import Path
import pandas as pd




def parse_args():
    parser = argparse.ArgumentParser(
        description=""
    )
    parser.add_argument(
        "root_path",
        type=Path,
    )
    return parser.parse_args()

def main(root_path: Path):

    root_path = Path(root_path).expanduser().resolve()
    input_path = root_path / "big_fish" / "results" / "all_tiles_3D" / "tile0_spots_all_bits.csv"

    fish_spots = pd.read_csv(filepath_or_buffer=input_path)
    print("data loaded")

if __name__ == "__main__":
    args = parse_args()
    main(args.root_path)