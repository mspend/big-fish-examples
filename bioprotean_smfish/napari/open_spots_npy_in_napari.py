import argparse

import napari
import numpy as np


def load_spots(path: str):
    data = np.load(path, allow_pickle=False)

    if data.dtype.names:
        required = {"y", "x", "bit"}
        if not required.issubset(data.dtype.names):
            raise ValueError(
                f"Numpy file fields must include {sorted(required)}; got {data.dtype.names}"
            )
        coordinates = np.column_stack((data["y"], data["x"])).astype(np.float64)
        bit = data["bit"].astype(np.int32)
    else:
        if data.ndim != 2 or data.shape[1] != 3:
            raise ValueError(
                "Raw spot arrays must have shape (n_spots, 3) ordered as y, x, bit"
            )
        coordinates = data[:, :2].astype(np.float64)
        bit = data[:, 2].astype(np.int32)

    return coordinates, bit


def main():
    parser = argparse.ArgumentParser(
        description="Open a 2D smFISH spots Numpy file in napari"
    )
    parser.add_argument("path", help="Path to spots_all_bits.npy")
    parser.add_argument("--size", type=float, default=5.0, help="Point size")
    args = parser.parse_args()

    coordinates, bit = load_spots(args.path)

    viewer = napari.Viewer()

    bit_labels = np.array([f"bit {value}" for value in bit])

    bit_colors = [
        "#1f77b4", "#ff7f0e", "#2ca02c", "#d62728",
        "#9467bd", "#8c564b", "#e377c2", "#7f7f7f",
        "#bcbd22", "#17becf", "#393b79", "#637939",
        "#8c6d31", "#843c39", "#7b4173", "#3182bd",
    ]

    # print("Bit-to-color mapping:")
    # for bit_number, color in enumerate(bit_colors, start=1):
    #     print(f"  bit {bit_number}: {color}")

    bit_styles = {
        1: ("blue", "#1f77b4"),
        2: ("orange", "#ff7f0e"),
        3: ("green", "#2ca02c"),
        4: ("red", "#d62728"),
        5: ("purple", "#9467bd"),
        6: ("brown", "#8c564b"),
        7: ("pink", "#e377c2"),
        8: ("gray", "#7f7f7f"),
        9: ("olive", "#bcbd22"),
        10: ("cyan", "#17becf"),
        11: ("navy", "#393b79"),
        12: ("olive green", "#637939"),
        13: ("ochre", "#8c6d31"),
        14: ("dark red", "#843c39"),
        15: ("plum", "#7b4173"),
        16: ("steel blue", "#3182bd"),
    }

    for bit_number, (color_name, color_code) in bit_styles.items():
        mask = bit == bit_number
        bit_coordinates = coordinates[mask]

        viewer.add_points(
            bit_coordinates,
            name=f"bit {bit_number} ({color_name})",
            size=args.size,
            face_color=color_code,
            features={"bit": bit[mask]},
        )

    napari.run()


if __name__ == "__main__":
    main()
