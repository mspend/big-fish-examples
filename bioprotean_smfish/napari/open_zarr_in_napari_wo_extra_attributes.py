# run this file in the napari-env
# For zarrs created with v0.10.0 of the merfish3d-analysis repo, there is an incompatibility with the ome-zarr and zarr requirements
# The zarr.json contains an unexpected argument "extra_attributes" 
# This is preventing opening zarrs in napari

import argparse
import napari
from zarr.core.group import GroupMetadata

original_zarr_data = GroupMetadata.from_dict.__func__

def remove_extra_attributes(cls, data):
    """Read Zarr metadata while ignoring the invalid 'extra_attributes' field."""
    data.pop("extra_attributes", None)
    return original_zarr_data(cls, data)

# Temporarily patch Zarr's metadata reader.
GroupMetadata.from_dict = classmethod(remove_extra_attributes)

# Get the Zarr path from the command line.
parser = argparse.ArgumentParser(
    description="Open an OME-Zarr file in napari."
)
parser.add_argument(
    "path",
    help="Path to the .ome.zarr directory",
)
args = parser.parse_args()

viewer = napari.Viewer()
viewer.open(args.path, plugin="napari-ome-zarr")
napari.run()