# packages availabe in merfish3d environment
# this makes one tiff file for each bit in the tile.
# takes ~2 minutes

from merfish3danalysis.qi2labDataStore import qi2labDataStore
from tifffile import TiffWriter
from pathlib import Path
import argparse

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

    # initialize datastore
    print("\nInitializing datastore...")
    datastore_path = root_path / Path(r"qi2labdatastore")
    datastore = qi2labDataStore(datastore_path)

    # define output path   
    output_path = root_path / "big_fish" / "tiffs"
    output_path.mkdir(parents=True, exist_ok=True)

    spacing_zyx_um = datastore.voxel_size_zyx_um

    tile_idx = 0
    n_bits = 16

    for bit_idx in range(n_bits):
        bit_data = datastore.load_local_registered_image(tile=tile_idx,bit=bit_idx,return_future=False)
        filename = "tile"+str(tile_idx).zfill(3)+"bit"+str(bit_idx+1).zfill(3)+".ome.tiff"
        filename_path = output_path / Path(filename)
        with TiffWriter(filename_path, bigtiff=True) as tif:
            metadata={
                'axes': 'ZYX',
                'SignificantBits': 16,
                'PhysicalSizeX': float(spacing_zyx_um[2]),
                'PhysicalSizeXUnit': 'µm',
                'PhysicalSizeY': float(spacing_zyx_um[1]),
                'PhysicalSizeYUnit': 'µm',
                'PhysicalSizeZ': float(spacing_zyx_um[0]),
                'PhysicalSizeZUnit': 'µm',
            }
            options = dict(
                compression='zlib',
                compressionargs={'level': 8},
                predictor=True,
                photometric='minisblack',
                resolutionunit='CENTIMETER',
            )
            tif.write(
                bit_data,
                resolution=(
                    1e4 / float(spacing_zyx_um[2]),
                    1e4 / float(spacing_zyx_um[1])
                ),
                **options,
                metadata=metadata
            )

if __name__ == "__main__":
    args = parse_args()
    main(args.root_path)
