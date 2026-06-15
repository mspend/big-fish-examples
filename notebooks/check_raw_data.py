from pathlib import Path
import bigfish.stack as stack


datasets = [
    "20250129_Bartelle_smFISH_control",
    "20250220_Bartelle_control_smFISH_TqIB",
    "20260311_bartelle_smFISH_cryo_48hr_male",
    "20260406_bartelle_smFISH_cryo_48hr_male",
    "20260413_bartelle_smFISH_control",
    "20260504_bartelle_smFISH_48hr_cryo",
]

for dataset in datasets:
    root_path = Path("/data/smfish") / dataset
    reference_shape = None
    reference_tiff = None

    for round_idx in range(1, 9):
        for tile_idx in range(154):
            tiff_path = (
                root_path
                / f"data_r{round_idx:04d}_tile{tile_idx:04d}_1"
                / f"data_r{round_idx:04d}_tile{tile_idx:04d}_NDTiffStack.tif"
            )

            if not tiff_path.exists():
                raise FileNotFoundError(
                    f"Missing TIFF for dataset='{dataset}', round={round_idx}, tile={tile_idx}:\n"
                    f"  {tiff_path}"
                )

            image = stack.read_image(str(tiff_path))
            current_shape = image.shape

            if reference_shape is None:
                reference_shape = current_shape
                reference_tiff = tiff_path
                # print(f"[REFERENCE] {reference_tiff} -> shape={reference_shape}")
            elif current_shape != reference_shape:
                raise ValueError(
                    f"Shape mismatch in dataset='{dataset}':\n"
                    f"  Reference TIFF: {reference_tiff}\n"
                    f"  Reference shape: {reference_shape}\n"
                    f"  Mismatched TIFF: {tiff_path}\n"
                    f"  Mismatched shape: {current_shape}"
                )

            # print(f"OK: {tiff_path} -> shape={current_shape}")