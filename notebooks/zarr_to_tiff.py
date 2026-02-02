# packages availabe in merfish3d environment
# this makes one tiff file for each bit in the tile.

from merfish3danalysis.qi2labDataStore import qi2labDataStore
from tifffile import TiffWriter
from pathlib import Path

def main():
    datastore_path = Path(r"/data/smFISH/20251028_bartelle_smFISH_mm_microglia_newbuffers/qi2labdatastore")
    datastore = qi2labDataStore(datastore_path)
    output_path = Path(r"/data/smFISH/20251028_bartelle_smFISH_mm_microglia_newbuffers/qi2labdatastore/big_fish")
    output_path.mkdir(parents=True, exist_ok=True)
    spacing_zyx_um = datastore.voxel_size_zyx_um

    # if I load the rounds instead of bits, the function load_local_registered_image will automatically detect that you are looking for the fiducial channel (polyDT) instead of the bit data (spots)

    tile_idx = 0
    # round_id indexes from 0
    # for the 1-28 data, only the first round (round 0) has a registered_decon_data for the polyDT channel in tile 0. None of the other rounds have the registered_decon_data, so the following function was returning none
    # path_to_data = "/data/smFISH/20251028_bartelle_smFISH_mm_microglia_newbuffers/qi2labdatastore/polyDT/tile0000/round001.zarr/registered_decon_data"
    round_idx = 0

    round_data = datastore.load_local_registered_image(tile=tile_idx,round=round_idx,return_future=False)

    # n_rounds = 8

    # # If I put range(1, n_rounds) it only goes 1=7. But if I put range(1, n_rounds+1), it says list index out of range.
    # for round_idx in range(1, n_rounds+1):
    #     print(round_idx)
    #     round_data = datastore.load_local_registered_image(tile=tile_idx,round=round_idx,return_future=False)
    #     # print(round_data.dtype)

    #     # If loader returns a lazy/dask array or future-like, try to materialize it.
    #     if round_data is not None and hasattr(round_data, "compute"):
    #         try:
    #             round_data = round_data.compute()
    #         except Exception:
    #             pass

    #     # If there's no data, skip this round and warn the user.
    #     if round_data is None:
    #         print(f"Warning: no image data for tile {tile_idx} round {round_idx}; skipping")
    #         continue

    filename = "tile"+str(tile_idx).zfill(3)+"round"+str(round_idx).zfill(3)+"corrected_polyDT.ome.tiff"
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
            round_data,
            shape=round_data.shape,
            dtype=round_data.dtype,
            resolution=(
                1e4 / float(spacing_zyx_um[2]),
                1e4 / float(spacing_zyx_um[1])
            ),
            **options,
            metadata=metadata
        )

if __name__ == "__main__":
    main()
