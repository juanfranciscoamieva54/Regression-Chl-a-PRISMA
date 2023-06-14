#!/usr/bin/env python
# coding: utf-8

import rasterio
from rasterio.transform import from_gcps
from rasterio.control import GroundControlPoint as GCP
from rasterio.crs import CRS
from shapely.geometry import box
from rasterio.windows import from_bounds
from rasterio.enums import Resampling
import rasterio.warp
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import datetime
from datetime import datetime
import os

list_ids_images=[1,2,4,6,10,13,17,18,19,21,23,24]

for idx in list_ids_images:

    ### Initial setup
    df=pd.read_excel(r"./data/dataset_config.xls", index_col=0)
    df.sort_values("acquisition_ID")

    ## EXECUTION CODE

    acquisition_id=idx
    row=df.loc[df.index==acquisition_id].iloc[0]

    print("starting_acquisition_Nº: ", acquisition_id)

    ##INPUTS PATHS:
    PRISMA_PATH='./data/inputs/prisma_imgs/'
    S2_PATH='./data/inputs/s2_reference_coreg/green_reflectance_s2_fill.tif'
    S3_PATH='./data/inputs/s3_reflectances/'+row.s3_fname+"/"+row.s3_fname

    #INT OUTPUTS:
    INTERM_PATH='./data/intermediate_files/'+str(acquisition_id)+"/"

    #CHL_GT_DATA_PATH:
    CHL_PATH='./data/inputs/intersected_inputs/'

    #### Create dirs
    if os.path.exists(INTERM_PATH):
        pass
    else:
        os.mkdir(INTERM_PATH)
    if os.path.exists(INTERM_PATH+"final_indices/"):
        pass
    else:
        os.mkdir(INTERM_PATH+"final_indices/")
    
    ###################################################################################
    # function def:
    ###################################################################################
    def intersect_rasters(path_rs1,path_rs2):
        with rasterio.open(path_rs1) as ras1, rasterio.open(path_rs2) as ras2:
            ext1=box(*ras1.bounds)
            meta1=ras1.meta
            ext2=box(*ras2.bounds)
            meta2=ras2.meta
            intersection=ext1.intersection(ext2)
            #print(ext1,ext2)
            #print(intersection)
            win1=rasterio.windows.from_bounds(*intersection.bounds, ras1.transform)
            win2=rasterio.windows.from_bounds(*intersection.bounds, ras2.transform)
            np1=ras1.read(window=win1)
            np2=ras2.read(window=win2)
        # cálculo de transformaciones respetando resoluciones de c/dataset
        tr_np1_overlap= rasterio.transform.from_bounds(*intersection.bounds, np1.shape[2], np1.shape[1])
        tr_np2_overlap= rasterio.transform.from_bounds(*intersection.bounds, np2.shape[2], np2.shape[1])

        meta1["width"]=np1.shape[2]
        meta1["height"]=np1.shape[1]
        meta1["transform"]=tr_np1_overlap
        meta2["width"]=np2.shape[2]
        meta2["height"]=np2.shape[1]
        meta2["transform"]=tr_np2_overlap

        return intersection, np1, np2, meta1, meta2, tr_np1_overlap, tr_np2_overlap  
    
    with rasterio.open(CHL_PATH +"GT_300/"+ str(idx)+".tif") as dst:
        np_chl = dst.read()

    print("Open S3 reflectances and preparing indices")
    # ### Open S3 reflectances and preparing indices----------------------------

    with rasterio.open(S3_PATH+"B03_(Raw).tiff") as dst:
        b03=dst.read()
        metas3=dst.meta.copy()

    with rasterio.open(S3_PATH+"B04_(Raw).tiff") as dst:
        b04=dst.read()

    with rasterio.open(S3_PATH+"B06_(Raw).tiff") as dst:
        b06=dst.read()

    with rasterio.open(S3_PATH+"B08_(Raw).tiff") as dst:
        b08=dst.read()

    with rasterio.open(S3_PATH+"B11_(Raw).tiff") as dst:
        b11=dst.read()

    with rasterio.open(S3_PATH+"B12_(Raw).tiff") as dst:
        b12=dst.read()

    index1=(b11/b08) # presence of chlorophyll-a
    index2=(b12/b11) # anomalies to remove
    index3a=(b06/b03)
    index3b=(b06/b04) # presence of phytoplankton
            
    with rasterio.open(INTERM_PATH+f"index1.tif" , mode="w" , **metas3) as dst:
            dst.write(index1)

    with rasterio.open(INTERM_PATH+f"index2.tif" , mode="w" , **metas3) as dst:
            dst.write(index2)

    with rasterio.open(INTERM_PATH+f"index3a.tif" , mode="w" , **metas3) as dst:
            dst.write(index3a)

    with rasterio.open(INTERM_PATH+f"index3b.tif" , mode="w" , **metas3) as dst:
            dst.write(index3b)

    list_indices = ["index1", "index2", "index3a", "index3b"]

    for index_id in list_indices:

        ### INTERSECTION OF CHL-a MAP WRT INDICES:
        path_rs1 = INTERM_PATH+f"{index_id}.tif"
        path_rs2 = CHL_PATH +"GT_300/"+ str(idx)+".tif"

        intersection, np1, np2, meta1, meta2, tr_np1_overlap, tr_np2_overlap = \
                intersect_rasters(path_rs1,path_rs2)

        with rasterio.open(INTERM_PATH+f"{index_id}_inters1_{str(idx)}.tif", mode="w", **meta1) as dst:
            dst.write(np1)
            
        ### RESIZING THE INDICES TO CORRECT DIFFERENCES IN GSD:
        ref_path = CHL_PATH +"GT_300/"+ str(idx)+".tif" 
        resize_path = INTERM_PATH+f"{index_id}_inters1_{str(idx)}.tif"
        resized_path = INTERM_PATH+f"final_indices/{index_id}.tif"

        # Open reference raster
        with rasterio.open(ref_path) as src:
            ref_arr = src.read(1)  # Read the first band of the raster
            ref_profile = src.profile  # Get the metadata of the raster

        # Open raster to resize
        with rasterio.open(resize_path) as src:
            resize_arr = src.read(1)  # Read the first band of the raster
            resize_profile = src.profile  # Get the metadata of the raster

        # Resize the raster to match the reference raster
        resized_arr = resize_arr.copy()  # Make a copy of the array to resize
        resampled, _ = rasterio.warp.reproject(
            source=resized_arr,
            destination=np.empty(ref_arr.shape, dtype=resized_arr.dtype),
            src_transform=src.transform,
            src_crs=src.crs,
            dst_transform=ref_profile['transform'],
            dst_crs=ref_profile['crs'],
            resampling=Resampling.bilinear
        )

        # Update the metadata of the resized raster
        resize_profile.update({
            'transform': ref_profile['transform'],
            'width': ref_profile['width'],
            'height': ref_profile['height']
        })

        # Write the resized raster to disk
        with rasterio.open(resized_path, 'w', **resize_profile) as dst:
            dst.write(resampled, 1)  # Write the resized array to the first band of the raster

    with rasterio.open(CHL_PATH +"GT_300/"+ str(idx)+".tif") as dst:
        meta_chl_out = dst.meta.copy()
        chl_map = dst.read()

    with rasterio.open(INTERM_PATH+"final_indices/index1.tif") as dst:
        index1 = dst.read()

    with rasterio.open(INTERM_PATH+"final_indices/index2.tif") as dst:
        index2 = dst.read()

    with rasterio.open(INTERM_PATH+"final_indices/index3a.tif") as dst:
        index3a = dst.read()

    with rasterio.open(INTERM_PATH+"final_indices/index3b.tif") as dst:
        index3b = dst.read()

    print("first condition chl_map > 10 and index1 < 1   ---> anomalous pixel")
    # first condition chl_map > 10 and index1 < 1   ---> anomalous pixel

    condition1 = (chl_map[0]>10)*(index1[0]<1)

    print("second condition: index2>1 anomalous pixels..")
    # second condition: index2>1 anomalous pixels..

    condition2 = index2[0]>1

    print("third condition: index3a>1 but index3b<1 or vice verza")
    # third condition: index3a>1 but index3b<1 or vice verza

    condition3_1=((index3a>1)[0] * (index3b<1)[0])
    condition3_2=((index3a<1)[0] * (index3b>1)[0])

    total_contitions = condition1 + condition2 + condition3_1 + condition3_2

    chl_map_corrected = chl_map[0]*(total_contitions!=1)

    chl_map_corrected = np.expand_dims(chl_map_corrected,0)

    with rasterio.open(CHL_PATH +"GT_300_no_anomalies/"+ str(idx)+".tif",mode="w",**meta_chl_out) as dst:
        dst.write(chl_map_corrected)



