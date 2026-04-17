"""
Fuse all channels into individual ome-ngff v0.4 for viewing.

Shepherd 2025/03 - created script.
Refactored 2026/04 - load all channels upfront, register once, fuse to disk directly.
"""

import warnings
warnings.filterwarnings("ignore", category=UserWarning)
warnings.simplefilter("ignore", category=FutureWarning)

from merfish3danalysis.qi2labDataStore import qi2labDataStore
from pathlib import Path
import gc
from tqdm import tqdm
from multiview_stitcher import spatial_image_utils as si_utils
from multiview_stitcher import msi_utils, registration, fusion
import dask.diagnostics
import dask.array as da
import dask
import numpy as np
import zarr
import multiprocessing as mp
from skimage import io

mp.set_start_method('spawn', force=True)


def fuse_all_channels(root_path: Path) -> None:
    """Register and fuse all channels across all tiles into OME-Zarr at full resolution.
    
    All channels (fiducial + readouts) are loaded upfront, registered together,
    and fused directly to disk. Output OME-Zarr is then split into individual
    channel TIFFs.

    Parameters
    ----------
    root_path: Path
        path to experiment
    """

    # initialize datastore
    print("\nInitializing datastore...")
    datastore_path = root_path / Path(r"qi2labdatastore")
    datastore = qi2labDataStore(datastore_path)
    gene_ids = list(datastore.codebook['gene_id'])
    channel_ids = ["polyDT"] + gene_ids
    num_channels = len(channel_ids)

    im_data = datastore.load_local_registered_image(
        tile=0, round=0, return_future=False
    )
    
    im_shape = im_data.shape
    del im_data

    # Load all channels for all tiles upfront
    print("\nLazy loading all channels for all tiles...")
    msims = []
    for tile_idx, tile_id in enumerate(tqdm(datastore.tile_ids, desc="tile")):

        # load voxel size
        voxel_zyx_um = datastore.voxel_size_zyx_um

        # format voxel size for multiview-stitcher
        scale = {
            "z": voxel_zyx_um[0], 
            "y": voxel_zyx_um[1], 
            "x": voxel_zyx_um[2]
        }

        # load stage positions and camera <-> stage mapping from first round of imaging
        tile_position_zyx_um, affine_zyx_px = datastore.load_local_stage_position_zyx_um(
            tile_id, datastore.round_ids[0]
        )

        # format tile positions for multiview-stitcher
        tile_grid_positions = {
            "z": np.round(tile_position_zyx_um[0], 2),
            "y": np.round(tile_position_zyx_um[1], 2),
            "x": np.round(tile_position_zyx_um[2], 2),
        }

        # Create array to hold ALL channels for this tile (fiducial + readouts)
        im_data = da.zeros(
            (num_channels, im_shape[0], im_shape[1], im_shape[2]),
            dtype=np.uint16
        )
        
        # Load fiducial channel (polyDT) first at index 0
        input_path = datastore_path / Path("polyDT") / Path(tile_id) / Path("round001.zarr")
        im_data[0, :, :, :] = da.from_zarr(input_path, component="registered_decon_data").astype("uint16")
        
        # Load all readout bits (channels 1 onwards)
        for ch_idx in range(1, num_channels):
            input_path = datastore_path / Path("readouts") / Path(tile_id) / Path(
                "bit" + str(ch_idx).zfill(3) + ".zarr"
            )
            # Load deconvolved * (u-fish prediction > 0.25) readout bits
            im_data[ch_idx, :, :, :] = (
                da.from_zarr(input_path, component="registered_decon_data").astype(np.float32) 
                * da.from_zarr(input_path, component="registered_ufish_data")
                  .astype(np.float32).clip(0.25, 1)
            ).astype(np.uint16)

        # Create spatial image with ALL channels at once
        sim = si_utils.get_sim_from_array(
            im_data,
            dims=("c", "z", "y", "x"),
            scale=scale,
            translation=tile_grid_positions,
            affine=affine_zyx_px,
            transform_key="stage_metadata",
            c_coords=channel_ids,
        )

        # convert to multiscale spatial image object and append to list for registration
        msim = msi_utils.get_msim_from_sim(sim, scale_factors=[])
        msims.append(msim)
        del im_data
        gc.collect()
    
    # Perform registration on all channels together
    print("\nPerforming registration on all channels...")
    with dask.diagnostics.ProgressBar():
        _ = registration.register(
            msims,
            reg_channel_index=0,  # Register based on fiducial channel
            transform_key="stage_metadata",
            new_transform_key="affine_registered",
            pre_registration_pruning_method="keep_axis_aligned",
            registration_binning={"z": 3, "y": 6, "x": 6},
            post_registration_do_quality_filter=True,
        )
 
    # Fuse all channels directly to OME-Zarr at full resolution
    print("\nFusing all channels to OME-Zarr at full resolution...")
    fused_path = root_path / Path("fused")
    fused_path.mkdir(exist_ok=True)
    ome_output_path = fused_path / Path("fused_all_channels.ome.zarr")
    
    print(f"Fusing views and saving output to {ome_output_path!s}...")
    with dask.diagnostics.ProgressBar():
        fusion.fuse(
            [msi_utils.get_sim_from_msim(msim) for msim in msims],
            transform_key="affine_registered",
            output_chunksize=512,
            overlap_in_pixels=64,
            output_zarr_url=str(ome_output_path),
        )
    
    # Load fused OME-Zarr back into memory and split into individual channel TIFFs
    print("\nLoading fused OME-Zarr and splitting into individual channel TIFFs...")
    fused_data = da.from_zarr(str(ome_output_path), component="0")  # Load multiscale level 0 (full resolution)
    fused_data_computed = fused_data.compute()
    
    tiff_output_path = fused_path / Path("individual_channels")
    tiff_output_path.mkdir(exist_ok=True)
    
    for ch_idx, channel_name in enumerate(tqdm(channel_ids, desc="writing TIFF")):
        output_file = tiff_output_path / Path(f"{channel_name}.tif")
        io.imsave(str(output_file), fused_data_computed[ch_idx, :, :, :], check_contrast=False)
        print(f"Saved {channel_name} to {output_file}")

          
if __name__ == "__main__":
    root_path = Path(r"/data/smFISH/20251028_bartelle_smFISH_mm_microglia_newbuffers")
    fuse_all_channels(root_path)