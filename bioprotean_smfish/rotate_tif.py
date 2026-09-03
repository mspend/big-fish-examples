import math
import tifffile
from scipy.ndimage import rotate
from pathlib import Path
import numpy as np

root_path = "/data/smfish/20260311_bartelle_smFISH_cryo_48hr_male/fused/z_max_projected/"
# output_path = root_path / "rotated"

file_path = Path(root_path) / "fused_max_projected_bit001.ome.tiff"

img = tifffile.imread(file_path)

print(img.shape)

angle = math.degrees(math.atan(274/3761))

rotated = rotate(
    img,
    angle=angle,
    reshape=False,
    order=3,          # cubic interpolation
    mode='constant',
    cval=0
)

tifffile.imwrite("fused_max_projected_bit001_rotated.tif", rotated)

