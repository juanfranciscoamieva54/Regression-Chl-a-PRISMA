## In this folder must be placed the S3 reflectance bands used in the script "2_Removing_anomalous_pixels.py". For each use case, the file "dataset_config.xls" (present in the parent folder "data") specifies the name of the associated S3 product.

## Names of the bands used in the mentioned script:

    prefix = s3_fname column in the dataset_config.xls file

    prefix + B03_(Raw).tiff
    prefix + B04_(Raw).tiff
    prefix + B06_(Raw).tiff
    prefix + B08_(Raw).tiff
    prefix + B11_(Raw).tiff
    prefix + B12_(Raw).tiff

## The bounding box to be considered for downloading the images is available also in the parent folder "data".