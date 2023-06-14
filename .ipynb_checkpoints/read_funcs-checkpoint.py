import numpy as np
import h5py

def read_prs_l2d(datapath, tstart, tend):
    """
    Open PRISMA L2D file and return data

    Parameters:
    ----------
    datapath: str, folder with raw L2D files
    tstart: str, acquisition start time
    tend: str, acquisition end time

    Returns:
    -------
    lat: ndarray, latitude matrix
    lon: ndarray, longitude matrix
    cwv: ndarray, central wavelength vector
    vrf: ndarray, visible reflectance cube
    srf: ndarray, infrared reflectance cube
    """
    pf = h5py.File(datapath+'PRS_L2D_STD_'+tstart+'_'+tend+'_0001.he5','r')

    # Read wavelengths, drop zero ones and overlap
    attrs = pf.attrs
    img_id = attrs['Image_ID']
    vn_wvl = np.array([ wvl for wvl in attrs['List_Cw_Vnir'] ])
    sw_wvl = np.array([ wvl for wvl in attrs['List_Cw_Swir'] ])


    info = {}
    info['img_id'] = img_id

    # Read geometry information
    geom = pf['HDFEOS']['SWATHS']['PRS_L2D_HCO']['Geometric Fields']
    relang = np.array(geom['Rel_Azimuth_Angle'][:])
    obsang = np.array(geom['Observing_Angle'][:])
    szeang = np.array(geom['Solar_Zenith_Angle'][:])

    data = pf['HDFEOS']['SWATHS']['PRS_L2D_HCO']

    # Read geographical information
    lat = np.array(data['Geolocation Fields']['Latitude'][:])
    lon = np.array(data['Geolocation Fields']['Longitude'][:])

    vrf = np.array(data['Data Fields']['VNIR_Cube'][:]) 
    srf = np.array(data['Data Fields']['SWIR_Cube'][:]) 

    # Adjust visible bands and data
    vn_wvl = vn_wvl[3::]
    vrf = vrf[:,3::,:]
    vn_wvl = vn_wvl[::-1]
    vrf = vrf[:,::-1,:]

    # Adjust infrared bands and data
    srf = srf[:,::-1,:]
    sw_wvl = sw_wvl[::-1]
    srf = srf[:,6::,:]
    sw_wvl = sw_wvl[6::]

    # Reorder dimensions to space-bands
    vrf = np.moveaxis(vrf, 1, 2)
    srf = np.moveaxis(srf, 1, 2)

    pf.close()

    return lat, lon, vn_wvl, sw_wvl, vrf, srf, info


