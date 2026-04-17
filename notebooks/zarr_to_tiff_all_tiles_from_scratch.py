"""
Fuse all channels into individual ome-ngff v0.4 for viewing.

Shepherd 2025/03 - created script.
"""

import numpy as np
import zarr
import dask
import dask.array as da
import dask.diagnostics
from multiview_stitcher import fusion, msi_utils, ngff_utils, registration
from multiview_stitcher import spatial_image_utils as si_utils
from pathlib import Path
from tqdm import tqdm
import gc
import multiprocessing as mp
from merfish3danalysis.qi2labDataStore import qi2labDataStore

def main():

    # input path
    root_path = "/data/smFISH/20251028_bartelle_smFISH_mm_microglia_newbuffers"


    # load datastore
    # initialize datastore
    print("\nInitializing datastore...")
    datastore_path = root_path / Path(r"qi2labdatastore")
    datastore = qi2labDataStore(datastore_path)
    # gene_ids = list(datastore.codebook["gene_id"])
    # channel_ids = ["polyDT", *gene_ids]
    # num_bits = datastore.num_bits

    # define shape of registered image using round 0 and a temporary variable
    im_data = datastore.load_local_registered_image(tile=0, round=0, return_future=False)

    im_shape = im_data.shape
    del im_data

    # convert local tiles from first round to multiscale spatial images
    print("\nLazy loading fiducial channel...")

    # msims is a list that holds the MultiscaleSpatialImages
    # one for each tile
    # contains image data and metadata like coodinate system, voxel spacing, and transformation information

    msims = []
    for _, tile_id in enumerate(tqdm(datastore.tile_ids, desc="tile")):
        # load voxel size
        voxel_zyx_um = datastore.voxel_size_zyx_um

        # format voxel size for multiview-stitcher
        scale = {"z": voxel_zyx_um[0], "y": voxel_zyx_um[1], "x": voxel_zyx_um[2]}

        # load stage positions and camera <-> stage mapping from first round of imaging
        # all tiles are already mapped to round 0, so we use this as the coordinate system
        tile_position_zyx_um, affine_zyx_px = (
            datastore.load_local_stage_position_zyx_um(tile_id, datastore.round_ids[0])
        )

        # format tile positions for multiview-stitcher
        tile_grid_positions = {
            "z": np.round(tile_position_zyx_um[0], 2),
            "y": np.round(tile_position_zyx_um[1], 2),
            "x": np.round(tile_position_zyx_um[2], 2),
        }

        # create empty array to hold all channels for this tile
        # im_data has shape ("c", "z", "y", "x")
        im_data = da.zeros((1, im_shape[0], im_shape[1], im_shape[2]), dtype=np.uint16)

        #  Find the Zarr file for this tile's fiducial channel, load the deconvolved data from it, and put it into the first channel of your multi-channel image array.

        # construct the file path
        input_path = (datastore_path / Path("polyDT") / Path(tile_id) / Path("round001.zarr"))

        # load image data from zarr file
        im_data[0, :] = da.from_zarr(input_path, component="registered_decon_data").astype(
            "uint16"
        )

        # create spatial image for all channels in current tile
        sim = si_utils.get_sim_from_array(
                    im_data,
                    dims=("c", "z", "y", "x"),
                    scale=scale,
                    translation=tile_grid_positions,
                    affine=affine_zyx_px,
                    transform_key="stage_metadata",
                )
        # convert to multiscale spatial image object and append to list for registration
        msim = msi_utils.get_msim_from_sim(sim, scale_factors=[])
        msims.append(msim)
        del im_data
        gc.collect()

    # print(len(msims))
    # print(msims)

    # perform registration
    print("\nPerforming registration...")
    with dask.diagnostics.ProgressBar():
        _ = registration.register(
            msims,
            reg_channel_index=0,
            transform_key="stage_metadata",
            new_transform_key="affine_registered",
            registration_binning={"z": 3, "y": 6, "x": 6},
            post_registration_do_quality_filter=True,
            n_parallel_pairwise_regs=20
        )



    print("done")


# This thing is necessary for code with multiprocessing when it spawns child processes
if __name__ == "__main__":
    main()





# # load all bits with the fiducial channel into im_data
# for bit_id in range(num_bits):
#     for _, tile_id in enumerate(tqdm(datastore.tile_ids, desc="tile")):
#         bit_data = datastore.load_local_registered_image(tile=tile_id, bit=bit_id, return_future=False)
#         print(bit_id, tile_id)




# im_data =  datastore.load_local_registered_image(
#         tile=0, round=0, return_future=False
#     )
# print(type(im_data))
# print(im_data.shape)


            # # you can pass in either round number or bit number but not both
            # im_data = datastore.load_local_registered_image(
            #     tile=tile_id, bit=bit_idx, return_future=False
            # )


# construct the spatial image
# sim = si_utils.get_sim_from_array(
#         im_data,
#         dims=("c", "z", "y", "x"),
#         scale=scale,
#         translation=tile_grid_positions,
#         affine=affine_zyx_px,
#         transform_key="stage_metadata",
#     )


# globally register
# fuse all channels


# directly fuse the images to disk as ome-zarr.
#  write it out at full resolution instead of the coarsed-array that we normally use for the fidicual array.


#  you can load the fused ome-zarr back into memory, split into individual TIFFs, and write them each out.