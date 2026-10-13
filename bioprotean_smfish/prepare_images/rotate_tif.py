# take 3D tiffs and rotate them
# the camera acquisition is slanted and I want to rotate it for when I slice it up into octants

import math
import tifffile
from tifffile import TiffWriter
from scipy.ndimage import rotate
from pathlib import Path
import pandas as pd

root_path = Path("/data/smfish/20260311_bartelle_smFISH_cryo_48hr_male")
output_path = root_path / "fused" / "rotated" 

# Create output directory if needed
output_path.mkdir(parents=True, exist_ok=True)

metadata_path = root_path / "scan_metadata.csv"
scan_metadata = pd.read_csv(metadata_path, index_col=0)

# Obtain camera metadata
# NA = numerical aperture
na = scan_metadata['na'][0]
z_voxel = scan_metadata['z_voxel_um'][0] 
yx_voxel = scan_metadata['yx_voxel_um'][0] 
voxel_zyx_um = [z_voxel, yx_voxel, yx_voxel]

n_bits = 16

# because range is exclusive of the stop
for bit in range(1, n_bits+1):

# bit = 2

    filename = f"fused_bit{bit:03d}.ome.tiff"
    path = root_path / "fused" / filename

    # read in using tiffile
    image = tifffile.imread(path)
    # read in using bigfish
    # image = stack.read_image(str(path))

    print(f"bit {bit} loaded")

    # set to the desired angle
    angle = math.degrees(math.atan(274/3761))

    rotated = rotate(
        image,
        angle=angle,
        reshape=False,
        order=3,          # cubic interpolation
        mode='constant',
        cval=0
    )

    output_filename = f"fused_bit{bit:03d}_rotated.ome.tiff"
    image_path = (output_path / Path(output_filename))

    with TiffWriter(image_path, bigtiff=True) as tif:
        metadata = {
            "axes": "ZYX",
            "SignificantBits": 16,
            "PhysicalSizeX": float(voxel_zyx_um[2]),
            "PhysicalSizeXUnit": "µm",
            "PhysicalSizeY": float(voxel_zyx_um[1]),
            "PhysicalSizeYUnit": "µm",
            'PhysicalSizeZ': float(voxel_zyx_um[0]),
            'PhysicalSizeZUnit': 'µm',                
        }
        options = {
            "compression": "zlib",
            "compressionargs": {"level": 8},
            "predictor": True,
            "photometric": "minisblack",
            "resolutionunit": "CENTIMETER",
        }
        tif.write(
            rotated,
            resolution=(
                1e4 / float(voxel_zyx_um[2]),
                1e4 / float(voxel_zyx_um[1]),
            ),
            **options,
            metadata=metadata,
        )
        
        print(f"done with bit {bit}")
