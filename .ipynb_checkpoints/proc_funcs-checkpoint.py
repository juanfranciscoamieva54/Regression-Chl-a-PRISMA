import numpy as np
import h5py

def find_nearest(array, value):
    idx = (np.abs(array - value)).argmin()
    return idx

def reflectance_norm_single_vnir(path_l2d,t1_0,tstart1,tend1):
    
    pf1 = h5py.File(path_l2d+'PRS_L2D_STD_'+tstart1+'_'+tend1+'_0001.he5','r')
    attrs1 = pf1.attrs
    L2ScaleVnirMax1 = attrs1['L2ScaleVnirMax']
    L2ScaleVnirMin1 = attrs1['L2ScaleVnirMin']
    t1=L2ScaleVnirMin1 + t1_0*(L2ScaleVnirMax1-L2ScaleVnirMin1)/65535 # scaling to get reflectance
    return t1

def reflectance_norm_single_swir(path_l2d,t1_0,tstart1,tend1):
    
    pf1 = h5py.File(path_l2d+'PRS_L2D_STD_'+tstart1+'_'+tend1+'_0001.he5','r')
    attrs = pf1.attrs
    L2ScaleSwirMax = attrs['L2ScaleSwirMax']
    L2ScaleSwirMin = attrs['L2ScaleSwirMin']
    t1=L2ScaleSwirMin + t1_0*(L2ScaleSwirMax-L2ScaleSwirMin)/65535 # scaling to get reflectance
    return t1
