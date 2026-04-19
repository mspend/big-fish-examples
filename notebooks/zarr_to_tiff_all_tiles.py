"""
Fuse all channels into individual ome-ngff v0.4 for viewing.

Shepherd 2025/03 - created script.
Spendlove 2026/04 - 
"""

import numpy as np
import zarr
import dask
import dask.array as da
import dask.diagnostics
from multiview_stitcher import fusion, msi_utils, ngff_utils, registration
from multiview_stitcher import spatial_image_utils as si_utils
from pathlib import Path
import argparse
from tqdm import tqdm
import gc
import multiprocessing as mp
from tifffile import TiffWriter
from merfish3danalysis.qi2labDataStore import qi2labDataStore

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
    output_path = datastore_path / "big_fish" / "tiffs"
    output_path.mkdir(parents=True, exist_ok=True)

    # grab the list of genes from the datastore
    gene_ids = list(datastore.codebook["gene_id"])
    channel_ids = ["polyDT", *gene_ids]
    print(gene_ids)
    print(channel_ids)

    # define shape of registered image using round 0 and a temporary variable im_data
    im_data = datastore.load_local_registered_image(tile=0, round=0, return_future=False)

    im_shape = im_data.shape
    del im_data

    # convert local tiles from first round (polyDT) to multiscale spatial images
    print("\nLazy loading fiducial channel...")

    # msims is a list that holds the MultiscaleSpatialImages
    # there is one multiscale spatial image for each tile
    # it contains image data and metadata like coodinate system, voxel spacing, and transformation information

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

        # find the Zarr file for this tile's fiducial channel, load the deconvolved data from it, and put it into the first channel of your multi-channel image array.

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

    # perform global registration
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

    # load the channel data and prepare for writing out to disk

    print("\nLazy loading and fusing full-resolution polyDT and readouts...")
    tile_ids = datastore.tile_ids

    for ch_id in tqdm(range(len(channel_ids)), desc="channel"):
        msims_full = []
        for tile_id, msim in enumerate(tqdm(msims, desc="tile")):
            # parse the registered fidicual channel to get the registration metadata
            affine = msi_utils.get_transform_from_msim(
                msim, transform_key="affine_registered"
            ).data.squeeze()
            affine = np.round(affine, 2)
            origin = si_utils.get_origin_from_sim(
                msi_utils.get_sim_from_msim(msim), asarray=False
            )
            scale = si_utils.get_spacing_from_sim(
                msi_utils.get_sim_from_msim(msim), asarray=False
            )

            # temporary variable for channel data
            im_data = da.zeros(
                (1, im_shape[0], im_shape[1], im_shape[2]), dtype=np.uint16
            )

            # lazy load tile data
            tile_id = tile_ids[tile_id]

            # lazy load deconvolved polyDT
            if ch_id == 0:
                input_path = (
                    datastore_path
                    / Path("polyDT")
                    / Path(tile_id)
                    / Path("round001.zarr")
                )
                im_data[0, :] = da.from_zarr(
                    input_path, component="registered_decon_data"
                ).astype(np.uint16)
            # lazy load deconvolved * (u-fish prediction>0.25) readout bits
            else:
                input_path = (
                    datastore_path
                    / Path("readouts")
                    / Path(tile_id)
                    / Path("bit" + str(ch_id).zfill(3) + ".zarr")
                )
                im_data[0, :] = (
                    da.from_zarr(input_path, component="registered_decon_data").astype(
                        np.float32
                    )
                    * da.from_zarr(input_path, component="registered_ufish_data")
                    .astype(np.float32)
                    .clip(0.25, 1)
                ).astype(np.uint16)

            # create spatial image for all channels in current tile using registration metadata instead of stage metadata
            # this uses different parameters than above - Why?
            sim_full = si_utils.get_sim_from_array(
                im_data,
                dims=("c", "z", "y", "x"),
                scale=scale,
                translation=origin,
                affine=affine,
                transform_key="affine_registered",
                c_coords=channel_ids[ch_id],
            )

            # convert to multiscale spatial image object and append to list for fusion
            msim_full = msi_utils.get_msim_from_sim(sim_full, scale_factors=[])
            msims_full.append(msim_full)
            del im_data
            gc.collect()

        # fuse all tiles together for one channel
        # create fused image object using previously calculated registration metadata and all channels
        print("Constructing fusion...")
        with dask.diagnostics.ProgressBar():
            fused = fusion.fuse(
                [msi_utils.get_sim_from_msim(msim_full) for msim_full in msims_full],
                transform_key="affine_registered",
                output_chunksize=512,
                overlap_in_pixels=64,
            )

        #  write out each channel as a tiff file
        if ch_id == 0:
            filename = "fused_"+"polyDT"+".ome.tiff"
        else:
            filename = "fused_"+"bit"+str(ch_id).zfill(3)+".ome.tiff"
        filename_path = output_path / Path(filename)

        with TiffWriter(filename_path, bigtiff=True) as tif:
            metadata={
                'axes': 'ZYX',
                'SignificantBits': 16,
                'PhysicalSizeX': float(voxel_zyx_um[2]),
                'PhysicalSizeXUnit': 'µm',
                'PhysicalSizeY': float(voxel_zyx_um[1]),
                'PhysicalSizeYUnit': 'µm',
                'PhysicalSizeZ': float(voxel_zyx_um[0]),
                'PhysicalSizeZUnit': 'µm',
            }
            options = dict(
                compression='zlib',
                compressionargs={'level': 8},
                predictor=True,
                photometric='minisblack',
                resolutionunit='CENTIMETER',
            )

            # convert the dask array to numpy BEFORE writing
            fused_computed = fused.compute()

            # the image data is stored in the last three columns of the array, zyx
            fused_zyx = fused_computed[0, 0, :, :, :]

            tif.write(
                fused_zyx,
                resolution=(
                    1e4 / float(voxel_zyx_um[2]),
                    1e4 / float(voxel_zyx_um[1])
                ),
                **options,
                metadata=metadata
            )
            print(f"Done with bit {ch_id}")

    print("done")


# This thing is necessary for code with multiprocessing when it spawns child processes
if __name__ == "__main__":
    args = parse_args()
    main(args.root_path)