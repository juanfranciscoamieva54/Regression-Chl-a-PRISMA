#!/usr/bin/env python
# coding: utf-8

import argparse
import os
import shutil
import numpy as np
import matplotlib.pyplot as plt
plt.style.use('bmh')

import datetime
from datetime import datetime
import time
import pandas as pd
import string
from scipy import stats
import cv2
import rasterio as rio
from rasterio.transform import from_gcps
from rasterio.control import GroundControlPoint as GCP
from rasterio.crs import CRS
from shapely.geometry import box
from rasterio.windows import from_bounds
from rasterio.enums import Resampling

from pyproj import Proj, transform

from read_funcs import read_prs_l2d
from proc_funcs import find_nearest, reflectance_norm_single_vnir, reflectance_norm_single_swir

import sys
sys.path.append('gefolki/python')
from algorithm import EFolki
from tools import wrapData

### PARAMS definition: (from command line)

####################################################################################################
argParser = argparse.ArgumentParser()

argParser.add_argument("-id_acquisition", "--idx",  type=int,  help="Acquisition Identifier")

args = argParser.parse_args()

idx = args.idx
####################################################################################################


### UTILS:

###INTERSECTION FUNCTION
def intersect_rasters(path_rs1,path_rs2):
    with rio.open(path_rs1) as ras1, rio.open(path_rs2) as ras2:
        ext1=box(*ras1.bounds)
        meta1=ras1.meta
        ext2=box(*ras2.bounds)
        meta2=ras2.meta
        intersection=ext1.intersection(ext2)
        win1=rio.windows.from_bounds(*intersection.bounds, ras1.transform)
        win2=rio.windows.from_bounds(*intersection.bounds, ras2.transform)
        np1=ras1.read(window=win1)
        np2=ras2.read(window=win2)
 
    tr_np1_overlap= rio.transform.from_bounds(*intersection.bounds, np1.shape[2], np1.shape[1])
    tr_np2_overlap= rio.transform.from_bounds(*intersection.bounds, np2.shape[2], np2.shape[1])

    meta1["width"]=np1.shape[2]
    meta1["height"]=np1.shape[1]
    meta1["transform"]=tr_np1_overlap
    meta2["width"]=np2.shape[2]
    meta2["height"]=np2.shape[1]
    meta2["transform"]=tr_np2_overlap

    return intersection, np1, np2, meta1, meta2, tr_np1_overlap, tr_np2_overlap

#### RESIZING FUNCTION
def resizing(upscale_factor,orig_path, dest_path, transform2=None):

            with rio.open(orig_path) as dataset:

                # resample data to target shape
                data = dataset.read(
                    out_shape=(
                        dataset.count,
                        int(dataset.height * upscale_factor),
                        int(dataset.width * upscale_factor)
                    ),
                    resampling=Resampling.nearest
                )

                # scale image transform
                transform = dataset.transform * dataset.transform.scale(
                    (dataset.width / data.shape[-1]),
                    (dataset.height / data.shape[-2])
                )

                out_meta = dataset.meta.copy()
                out_meta["count"], out_meta["height"], out_meta["width"] = data.shape

                if transform2!=None:
                    out_meta["transform"] = transform2
                else:
                    out_meta["transform"] = transform

                with rio.open(dest_path , mode="w" , **out_meta) as dst:
                        dst.write(data)

### APPLY COREGISTRATION FUNCTION:
def apply_coreg(u, v, orig_path, dest_path):

    with rio.open(orig_path) as dst:
        img=dst.read()
        out_meta=dst.meta.copy()

    img_coreg=np.zeros([img.shape[0],img.shape[1],img.shape[2]])
    print(img.shape)
    for idx,bands in enumerate (range(img.shape[0])):
        img_coreg[idx,:,:] = wrapData(img[idx,:,:], u, v)

    with rio.open(dest_path, mode="w", **out_meta) as new_dataset:
        new_dataset.write(img_coreg)

### Initial setup

df=pd.read_excel(r"./data/dataset_config.xls", index_col=0)
df.sort_values("acquisition_ID")

## EXECUTION CODE

for idx,row in df.iterrows():

    acquisition_id=idx
    row=df.loc[df.index==acquisition_id].iloc[0]

    print("starting_acquisition_Nº: ", acquisition_id)

    ##INPUTS PATHS:
    PRISMA_PATH='./data/inputs/prisma_imgs/'
    S2_PATH='./data/inputs/s2_reference_coreg/green_reflectance_s2_fill.tif'
    S3_PATH='./data/inputs/s3_reflectances/'+row.s3_fname+"/"+row.s3_fname

    date_wqp=row.date_wqp
    FILTER_DATE=datetime.strptime(date_wqp, '%d/%m/%y')
    FILTER_D=str(FILTER_DATE.day)
    FILTER_M=str(FILTER_DATE.month)
    FILTER_Y=str(FILTER_DATE.year)

    #TEMP FILES:
    TEMP_PATH='./data/temp/'+str(acquisition_id)+"/"

    #INT OUTPUTS:
    INTERM_PATH='./data/intermediate_files/'+str(acquisition_id)+"/"

    #OUTPUTS:
    OUT_PATH='./data/inputs/intersected_inputs/'

    if os.path.exists(TEMP_PATH):
        pass
    else:
        os.mkdir(TEMP_PATH)

    if os.path.exists(INTERM_PATH):
        pass
    else:
        os.mkdir(INTERM_PATH)
        
    if os.path.exists(OUT_PATH):
        pass
    else:
        os.mkdir(OUT_PATH)

    print("Intersection of PRISMA - S2 and resizing PRISMA to S2")
    # ### Intersection of PRISMA - S2 and resizing PRISMA to S2

    with rio.open(S2_PATH) as dst:
        meta_copy=dst.meta.copy()

    # #### Transforming PRISMA to Geotiff

    tstart=str(row["tstart"])
    tend=str(row["tend"])

    y1, x1, vwl1, swl1, vrf1, srf1, info1 = read_prs_l2d(PRISMA_PATH, tstart, tend)

    img_prsm=np.dstack((vrf1,srf1))

    wavelengths=np.hstack((vwl1,swl1))
    np.save(TEMP_PATH+"wavelengths_prs.npy",wavelengths)

    inProj = Proj(init='epsg:4326')
    outProj = Proj(init='epsg:32632')
    x2,y2 = transform(inProj,outProj,x1[0,0],y1[0,0])

    meta_prisma=meta_copy
    meta_prisma['driver'] = 'GTiff'
    meta_prisma["width"]=img_prsm.shape[1]
    meta_prisma["height"]=img_prsm.shape[0]
    meta_prisma["count"]= img_prsm.shape[2]
    meta_prisma["dtype"]='uint16'
    meta_prisma ['nodata']= 0
    meta_prisma["transform"]= rio.transform.Affine(30, 0.0, x2, 0.0, -30, y2)
    meta_prisma['crs'] = CRS.from_epsg(32632)

    prsm_reshaped=np.moveaxis(img_prsm, -1, 0)

    with rio.open(
            TEMP_PATH+"PRS.tif",
            mode="w",
            **meta_prisma

    ) as new_dataset:
            new_dataset.write(prsm_reshaped)

    print("Intersection of PRS wrt S2 - Coregistration")
    # #### Intersection of PRS wrt S2 - Coregistration///////////////////////////////////////

    intersect_PRS_s2, PRS_ovl, s2_ovl, meta_PRS_ovl, meta_s2_ovl, tr_PRS_ovl, tr_s2_ovl = \
        intersect_rasters(TEMP_PATH+"PRS.tif", S2_PATH)

    with rio.open(TEMP_PATH+"PRS_intersected.tif", mode="w",**meta_PRS_ovl) as dst:
        dst.write(PRS_ovl)

    resizing(3,TEMP_PATH+"PRS_intersected.tif", TEMP_PATH+"resized_PRS_intersected.tif", tr_s2_ovl)
    with rio.open(TEMP_PATH+"resized_PRS_intersected.tif") as dst:
        resized_PRS=dst.read()

    intersect_PRS2_s2, PRS2_ovl, s2_ovl, meta_PRS2_ovl, meta_s2_ovl, tr_PRS2_ovl, tr_s2_ovl = \
        intersect_rasters(TEMP_PATH+"resized_PRS_intersected.tif", S2_PATH)

    vnir=PRS2_ovl[:63,:,:]
    swir=PRS2_ovl[63:,:,:]

    vnir_refl_temp = np.memmap("./data/bin_files/vnir_refl.dat", dtype='float32', mode='w+', shape=(vnir.shape[0],vnir.shape[1],vnir.shape[2]))
    swir_refl_temp = np.memmap("./data/bin_files/swir_refl.dat", dtype='float32', mode='w+', shape=(swir.shape[0],swir.shape[1],swir.shape[2]))

    for idx,bands in enumerate (range(vnir.shape[0])):
        vnir_refl_temp[idx,:,:]= reflectance_norm_single_vnir(PRISMA_PATH,vnir[idx,:,:],tstart,tend)

    for idx,bands in enumerate (range(swir.shape[0])):
        swir_refl_temp[idx,:,:] = reflectance_norm_single_swir(PRISMA_PATH,swir[idx,:,:],tstart,tend)

    vnir_refl_temp.flush()
    swir_refl_temp.flush()

    img1_v = vnir_refl_temp
    img1_r = swir_refl_temp


    img_prsm=np.vstack((img1_v,img1_r))

    meta_PRS2_ovl["dtype"]="float32"

    with rio.open(TEMP_PATH+"PRS_intersected_resized_reflectance.tif", mode="w",**meta_PRS2_ovl) as dst:
        dst.write(img_prsm)

    gr_s2=s2_ovl[0,:,:]
    gr_s2=np.clip(gr_s2,0,1)

    G_idx=find_nearest(wavelengths,560)
    gr_PRS=img_prsm[G_idx,:,:]

    starttime=time.time()
    u, v = EFolki(gr_s2, gr_PRS, iteration=8, radius=[30,30], rank=4, levels=6)
    endtime=time.time()
    print("coreg_delayed: ",(endtime-starttime)/60)

    with rio.open(TEMP_PATH+"PRS_intersected_resized_reflectance.tif") as dst:
            img=dst.read()
            out_meta=dst.meta.copy()

    np.save(TEMP_PATH+"u_prs.npy",u)
    np.save(TEMP_PATH+"v_prs.npy",v)

    u=np.load(TEMP_PATH+"u_prs.npy")
    v=np.load(TEMP_PATH+"v_prs.npy")


    fp_img = np.memmap("./data/bin_files/img_uint16.dat", dtype='uint16', mode='w+', shape=(img.shape[0],img.shape[1],img.shape[2]))
    fp_coreg = np.memmap("./data/bin_files/img_coreg_float32.dat", dtype='float32', mode='w+', shape=(img.shape[0],img.shape[1],img.shape[2]))

    img_coreg=np.zeros([img.shape[0],img.shape[1],img.shape[2]],dtype="float32")

    for idx,bands in enumerate (range(img.shape[0])):
        fp_img[idx,:,:]=img[idx,:,:]*2**16
        fp_coreg[idx,:,:] = wrapData(fp_img[idx,:,:], u, v)

    fp_img.flush()
    fp_coreg.flush()


    img_coreg=fp_coreg

    out_meta["dtype"]="float32"
    with rio.open(TEMP_PATH+"PRS_int_resized_coregistered.tif", mode="w", **out_meta) as new_dataset:
        new_dataset.write(img_coreg)

    with rio.open(TEMP_PATH+"PRS_int_resized_coregistered.tif") as dataset:
       # resample data to target shape
        data = dataset.read(
            out_shape=(
                dataset.count,
                int(dataset.height * (1/3)),
                int(dataset.width * (1/3))
            ),
            resampling=Resampling.nearest
        )

        # scale image transform
        transform = dataset.transform * dataset.transform.scale(
            (dataset.width / data.shape[-1]),
            (dataset.height / data.shape[-2])
        )

        out_meta["count"], out_meta["height"], out_meta["width"] = data.shape

        out_meta["transform"] = transform

        with rio.open(INTERM_PATH+"PRS_int_resized_coregistered_30m.tif" , mode="w" , **out_meta) as dst:
                dst.write(data)


    print("Intersection of PRISMA wrt Chl-a map")
    # ### Intersection of PRISMA wrt Chl-a map

    df_files_times=pd.read_csv('./data/SIMILE_chl_a/chl_files_times.csv')

    wqp_raster= rio.open('./data/SIMILE_chl_a/chl_stack.tif', 'r') #stack of the dataset requested

    meta_wqp = wqp_raster.meta # collecting the metadata of the imported stack
    np_wqp=wqp_raster.read()

    filter_date=np.where((df_files_times["day_of_year"]==int(FILTER_DATE.strftime('%j')))&(df_files_times["year"]==FILTER_DATE.year),1,0)

    #filter datacube just to selected dates
    np_wqp=np_wqp[np.nonzero(filter_date)[0],:,:]
    np_wqp=np.nan_to_num(np_wqp,0)
    meta_wqp["count"]=1

    with rio.open(INTERM_PATH+"chl_wqp.tif" , mode="w" , **meta_wqp) as dst:
            dst.write(np_wqp)

    intersect_PRS_chl, PRS_ovl, chl_ovl, meta_PRS_ovl, meta_chl_ovl, tr_PRS_ovl, tr_chl_ovl = \
        intersect_rasters(INTERM_PATH+"PRS_int_resized_coregistered_30m.tif", INTERM_PATH+"chl_wqp.tif")

    with rio.open(INTERM_PATH+"chl_wqp_intersected.tif", mode="w", **meta_chl_ovl) as dst:
        dst.write(chl_ovl)

    with rio.open(INTERM_PATH+"PRS_intersected_w_chl.tif", mode="w", **meta_PRS_ovl) as dst:
        dst.write(PRS_ovl)

    resizing(10,INTERM_PATH+"chl_wqp_intersected.tif", INTERM_PATH+"chl_wqp_inters_resized.tif", tr_PRS_ovl)
    with rio.open(INTERM_PATH+"chl_wqp_inters_resized.tif") as dst:
        resized_chl_resized=dst.read()

    intersect_PRS_chl, PRS_ovl, chl_ovl2, meta_PRS_ovl, meta_chl_ovl2, tr_PRS_ovl, tr_chl_ovl2 = \
        intersect_rasters(INTERM_PATH+"PRS_int_resized_coregistered_30m.tif", INTERM_PATH+"chl_wqp_inters_resized.tif")

    prisma_masked=PRS_ovl*(chl_ovl2[0]!=0)
    chl_masked=chl_ovl2*(PRS_ovl[0]!=0)

    OUT_PRS = OUT_PATH + "PRS_imgs/"
    OUT_CHL = OUT_PATH + "GT/"
    
    if os.path.exists(OUT_PRS):
        pass
    else:
        os.mkdir(OUT_PRS)
        
    if os.path.exists(OUT_CHL):
        pass
    else:
        os.mkdir(OUT_CHL)
    
    with rio.open(OUT_PRS+f"{str(acquisition_id)}.tif", mode="w", **meta_PRS_ovl) as dst:
        dst.write(prisma_masked)

    with rio.open(OUT_CHL+f"{str(acquisition_id)}.tif", mode="w", **meta_chl_ovl2) as dst:
        dst.write(chl_masked)

    print("Removing temporary and intermediate files")
    shutil.rmtree(TEMP_PATH)
    shutil.rmtree(INTERM_PATH)

