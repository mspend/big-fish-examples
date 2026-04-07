# packages availabe in merfish3d environment
# this makes one tiff file for each bit in the tile.

from merfish3danalysis.qi2labDataStore import qi2labDataStore
from tifffile import TiffWriter
from pathlib import Path
import numpy as np

def main():
    datastore_path = Path(r"/data/smFISH/20251028_bartelle_smFISH_mm_microglia_newbuffers/qi2labdatastore")
    datastore = qi2labDataStore(datastore_path)
    output_path = Path(r"/data/smFISH/20251028_bartelle_smFISH_mm_microglia_newbuffers/qi2labdatastore/big_fish/tiffs")
    output_path.mkdir(parents=True, exist_ok=True)
    spacing_zyx_um = datastore.voxel_size_zyx_um

    n_tiles = 25
    n_bits = 16
    
    # Assume datastore has tile_positions_zyx_um as a list of [z, y, x] positions in microns for each tile
    tile_positions_zyx_um = datastore.tile_positions_zyx_um  # Replace with actual attribute/method if different
    
    for bit_idx in range(n_bits):
        # Collect data and positions for all tiles for this bit
        tiles_data = []
        positions_pix = []
        for tile_idx in range(n_tiles):
            bit_data = datastore.load_local_registered_image(tile=tile_idx, bit=bit_idx, return_future=False)
            pos_um = tile_positions_zyx_um[tile_idx]  # [z, y, x] in microns
            pos_pix = [int(p / s) for p, s in zip(pos_um, spacing_zyx_um)]  # Convert to pixels
            tiles_data.append(bit_data)
            positions_pix.append(pos_pix)
        
        # Determine canvas size
        shapes = [data.shape for data in tiles_data]
        max_z = max(pos[0] + shape[0] for pos, shape in zip(positions_pix, shapes))
        max_y = max(pos[1] + shape[1] for pos, shape in zip(positions_pix, shapes))
        max_x = max(pos[2] + shape[2] for pos, shape in zip(positions_pix, shapes))
        
        # Create empty canvas
        stitched_data = np.zeros((max_z, max_y, max_x), dtype=tiles_data[0].dtype)
        
        # Place each tile on the canvas (assuming no overlap or overwriting as needed)
        for data, pos in zip(tiles_data, positions_pix):
            z_start, y_start, x_start = pos
            stitched_data[z_start:z_start + data.shape[0], y_start:y_start + data.shape[1], x_start:x_start + data.shape[2]] = data
        
        # Save the stitched image
        filename = f"bit{bit_idx+1:03d}_stitched.ome.tiff"
        filename_path = output_path / Path(filename)
        with TiffWriter(filename_path, bigtiff=True) as tif:
            metadata = {
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
                stitched_data,
                resolution=(
                    1e4 / float(spacing_zyx_um[2]),
                    1e4 / float(spacing_zyx_um[1])
                ),
                **options,
                metadata=metadata
            )

if __name__ == "__main__":
    main()
