# packages available in merfish3d environment
# This makes one tiff file for each bit in the globally stitched image

from merfish3danalysis.qi2labDataStore import qi2labDataStore
from tifffile import TiffWriter
from pathlib import Path
import numpy as np

def stitch_global_image_for_bit(datastore, bit_idx):
    """
    Load tile data for a specific bit and stitch into a global image using global coordinates.
    
    Parameters
    ----------
    datastore : qi2labDataStore
        The datastore instance
    bit_idx : int
        Bit index to load
        
    Returns
    -------
    global_image : np.ndarray
        Stitched global image for the specified bit
    global_origin_zyx_um : np.ndarray
        Origin of the global coordinate system
    global_spacing_zyx_um : np.ndarray
        Spacing of the global coordinate system
    """
    
    # n_tiles = len(datastore.tile_ids)
    n_tiles= 10
    tile_data_list = []
    tile_transforms = []
    
    # Load data and transforms for all tiles
    for tile_idx in range(n_tiles):
        # Load local registered image for this tile and bit
        bit_data = datastore.load_local_registered_image(
            tile=tile_idx, 
            bit=bit_idx, 
            return_future=False
        )
        
        if bit_data is None:
            print(f"Warning: no image data for tile {tile_idx} bit {bit_idx}; skipping")
            continue
        
        # Load global coordinate transforms for this tile
        affine_zyx_um, origin_zyx_um, spacing_zyx_um = datastore.load_global_coord_xforms_um(
            tile=tile_idx
        )
        
        if affine_zyx_um is None:
            print(f"Warning: no global transforms for tile {tile_idx}; skipping")
            continue
        
        tile_data_list.append(bit_data)
        tile_transforms.append({
            'affine': affine_zyx_um,
            'origin': origin_zyx_um,
            'spacing': spacing_zyx_um
        })
    
    if not tile_data_list:
        print(f"No valid data found for bit {bit_idx}")
        return None, None, None
    
    # Use the first tile's spacing and origin as reference (they should all be the same)
    global_spacing_zyx_um = tile_transforms[0]['spacing']
    global_origin_zyx_um = tile_transforms[0]['origin']
    
    # Calculate global canvas size by finding bounds of all tiles
    # The affine transform and origin define where each tile sits in global space
    min_zyx = np.array([np.inf, np.inf, np.inf])
    max_zyx = np.array([-np.inf, -np.inf, -np.inf])
    
    for data, transform in zip(tile_data_list, tile_transforms):
        origin = transform['origin']
        shape = np.array(data.shape)
        spacing = transform['spacing']
        
        # Calculate tile bounds in global coordinates
        tile_max = origin + (shape * spacing)
        
        min_zyx = np.minimum(min_zyx, origin)
        max_zyx = np.maximum(max_zyx, tile_max)
    
    # Create canvas for global image
    global_shape_um = max_zyx - min_zyx
    global_shape_pix = np.ceil(global_shape_um / global_spacing_zyx_um).astype(int)
    
    # Initialize global image
    global_image = np.zeros(global_shape_pix, dtype=tile_data_list[0].dtype)
    
    # Place each tile on the global canvas
    for data, transform in zip(tile_data_list, tile_transforms):
        origin = transform['origin']
        
        # Calculate offset in pixels from the global image origin
        offset_um = origin - min_zyx
        offset_pix = (offset_um / global_spacing_zyx_um).astype(int)
        
        # Calculate bounds where this tile will be placed
        shape = data.shape
        z_start, y_start, x_start = offset_pix
        z_end = min(z_start + shape[0], global_shape_pix[0])
        y_end = min(y_start + shape[1], global_shape_pix[1])
        x_end = min(x_start + shape[2], global_shape_pix[2])
        
        # Calculate how much of the tile actually fits
        z_tile_end = z_end - z_start
        y_tile_end = y_end - y_start
        x_tile_end = x_end - x_start
        
        # Place tile data
        if z_tile_end > 0 and y_tile_end > 0 and x_tile_end > 0:
            global_image[z_start:z_end, y_start:y_end, x_start:x_end] = \
                data[:z_tile_end, :y_tile_end, :x_tile_end]
    
    # Update origin to reflect the actual global image origin
    actual_origin_zyx_um = min_zyx
    
    return global_image, actual_origin_zyx_um, global_spacing_zyx_um


def main():
    datastore_path = Path(r"/data/smFISH/20251028_bartelle_smFISH_mm_microglia_newbuffers/qi2labdatastore")
    datastore = qi2labDataStore(datastore_path)
    output_path = Path(r"/data/smFISH/20251028_bartelle_smFISH_mm_microglia_newbuffers/qi2labdatastore/big_fish/tiffs")
    output_path.mkdir(parents=True, exist_ok=True)
    spacing_zyx_um = datastore.voxel_size_zyx_um

    n_bits = 16

    for bit_idx in range(n_bits):
        print(f"Processing bit {bit_idx + 1}/{n_bits}...")
        
        # Load and stitch global image for this bit
        global_image, origin_zyx_um, spacing_zyx_um = stitch_global_image_for_bit(
            datastore, bit_idx
        )
        
        if global_image is None:
            print(f"Skipping bit {bit_idx} due to missing data")
            continue
        
        # Save to TIFF
        filename = f"global_bit{bit_idx+1:03d}.ome.tiff"
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
                global_image,
                resolution=(
                    1e4 / float(spacing_zyx_um[2]),
                    1e4 / float(spacing_zyx_um[1])
                ),
                **options,
                metadata=metadata
            )
        
        print(f"Saved: {filename_path}")


if __name__ == "__main__":
    main()