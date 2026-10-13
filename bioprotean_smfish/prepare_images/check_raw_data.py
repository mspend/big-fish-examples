from pathlib import Path
import logging
import bigfish.stack as stack


datasets = [
    "20250129_Bartelle_smFISH_control",
    "20250220_Bartelle_control_smFISH_TqIB",
    "20260311_bartelle_smFISH_cryo_48hr_male",
    "20260406_bartelle_smFISH_cryo_48hr_male",
    "20260413_bartelle_smFISH_control",
    "20260504_bartelle_smFISH_48hr_cryo",
]

# Configure logging
log_file = "/data/smfish/smfish_validation.log"

logging.basicConfig(
    filename=log_file,
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
)

logging.info("Starting validation")

n_missing = 0
n_shape_mismatch = 0
n_read_errors = 0

for dataset in datasets:
    print(f"Checking {dataset}")
    logging.info(f"Checking dataset {dataset}")

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

            # Check existence
            if not tiff_path.exists():
                n_missing += 1
                logging.error(
                    f"MISSING FILE | dataset={dataset} "
                    f"| round={round_idx} | tile={tile_idx} "
                    f"| path={tiff_path}"
                )
                continue

            # Read image
            try:
                image = stack.read_image(str(tiff_path))
            except Exception as e:
                n_read_errors += 1
                logging.error(
                    f"READ ERROR | dataset={dataset} "
                    f"| round={round_idx} | tile={tile_idx} "
                    f"| path={tiff_path} | error={e}"
                )
                continue

            current_shape = image.shape

            # Save first shape as reference
            if reference_shape is None:
                reference_shape = current_shape
                reference_tiff = tiff_path

                logging.info(
                    f"REFERENCE SHAPE | dataset={dataset} "
                    f"| shape={reference_shape} "
                    f"| file={reference_tiff}"
                )

            # Compare shapes
            elif current_shape != reference_shape:
                n_shape_mismatch += 1

                logging.error(
                    f"SHAPE MISMATCH | dataset={dataset} "
                    f"| round={round_idx} | tile={tile_idx} "
                    f"| reference_shape={reference_shape} "
                    f"| current_shape={current_shape} "
                    f"| current_file={tiff_path}"
                )

summary = (
    f"Validation complete. "
    f"Missing files: {n_missing}, "
    f"Shape mismatches: {n_shape_mismatch}, "
    f"Read errors: {n_read_errors}"
)

print(summary)
logging.info(summary)

print(f"Detailed log written to: {log_file}")