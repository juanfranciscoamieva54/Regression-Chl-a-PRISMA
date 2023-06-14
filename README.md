# Regression-Chl-a-PRISMA
Regression of Chlorophyll-a values in Insubric Lakes with PRISMA hyperspectral imagery

## Main objectives and introduction to the project

The main objective of this work was to determine the concentration of chlorophyll-a in a selection of Insubric Lakes, namely Lake Maggiore, Lake Como, and Lake Lugano, located in the cross-basin between Italy and Switzerland. These lakes were previously analyzed in the "Informative System for the Integrated Monitoring of Insubric Lakes and their Ecosystems" (SIMILE) project [1], which provided the baseline for the production of ground truth information used in our study.

Considering the importance of freshwater for communities and the role of lakes in ecosystems, it becomes evident that monitoring and managing water quality is a critical requirement for society. Specifically, this study aimed to develop a model to predict chlorophyll-a concentrations using hyperspectral imagery, with a set of previously computed chlorophyll-a (Chl-a) concentration maps from the SIMILE project serving as the ground truth data.

Chlorophyll-a is a relevant parameter in the monitoring of the lakes because it can be used as an estimator of their biomass concentration. This parameter is particularly important because it allows understanding the trophic state of the lakes, which is associated with their biodiversity: the presence of nutrients in a lake can produce an overgrowth of algae and other aquatic plants that when they die are decomposed causing consumption of the oxygen in the water and affecting the biodiversity, process known as eutrophication [2].

The decision of using hyperspectral imagery was mainly based on the advantage provided by this kind of information which offers a higher spectral resolution with respect to multispectral imagery. Given that the hyperspectral images provide a high spectral resolution, they form a datacube in which the number of channels adds a third dimension and for each pixel,  it is provided the spectral response of each of the measurement bands [4]. This high spectral resolution can be exploited to understand the physicochemical composition of the Earth [4]. In our specific case, we used this information to train machine and deep learning models in order to predict chlorophyll-a concentrations on the mentioned insubric lakes.

Our study selected the Italian hyperspectral imager PRecursore IperSpettrale della Missione Applicativa (PRISMA) as the instrument to get the input data used for the proposed objective. The PRISMA mission was led by the Italian Space Agency (ASI), launched in 2019, and it is composed of VNIR and SWIR spectrometers and a panchromatic camera [3][4].

This repository was prepared to carry out the experimental development of the project. Specifically it covered the complete set of manipulations applied on the input data: coregistration of input images, the intersection between hyperspectral PRISMA images and the associated chlorophyll-a maps, pre-processing steps related to the removal of anomalous pixels, among others. Also, there was implemented the dimensionality reduction technique Principal Component Analysis to reduce the dimensionality of the hyperspectral input images and different normalization which were applied on the input data.

Regarding the models taken into account for this project, a total of four model typologies were taken into account: Random Forest Regressor, Support Vector Regressor, Long-Short Term Memory networks and Gated Recurrent Unit networks.

## Technical considerations for using this repository

- The following repository must be cloned in the same hierarchy of this one because it was considering for performing the coregistration operations applied in the script "1_Pre_processing_inputs.py": https://github.com/aplyer/gefolki 
- The ground truth maps (chlorophyll-a maps) are present in the folder "SIMILE_chl_a" which is inside the folder "data" of this repository.
- The number which is placed as a prefix of the different files present in this repository indicates the order of execution that must be followed for the proper use. 
- The notebook "4_inference" is optional because it can be used in case that it is just desired to infer using a model which was previously trained and saved.
- The script files without number prefix are general utils which are used in the "1_Pre_processing_inputs.py" script.
- For executing the script "1_Pre_processing_inputs.py" for each of the use cases which are detailed in the file "dataset_config.xls" (present in the folder "data"), the following instruction should be provided from command line: "python 1_Pre_processing_inputs.py -idx". "idx" must be replaced by the identifier present in the "dataset_config.xls" file.

### Downloading images

- The file "dataset_config.xls" present in the folder "data" specifies the filenames and relevant information associated with the different products to be download for each use case.

- The AOI to be considered for all the imagery to be downloaded, is available in the folder "data" and the filename of this bounding box is "bounding_box_AOI.geojson".

#### The products which must be downloaded are:

+ PRISMA images: the processing level used was L2D. This information must be downloaded from the official website of the PRISMA mission, using the appropiate credentials. The mentioned file "dataset_config.xls" specifies the name corresponding to each acquisition.

+ Sentinel 2 - green band: S2 green band in reflectance units corresponding to the mosaic produced taking into account the period  "11/09/2022" - "18/09/2022". This file must be placed in the path "/data/inputs/s2_reference_coreg/". The file containing the mentioned S2 green band must be named: "green_reflectance_s2_fill.tif".

+ Sentinel 3 bands: Different bands of S3 satellite are used in the script "2_Removing_anomalous_pixels.py" and for each use case, the file "dataset_config.xls" (present in the parent folder "data") specifies the name of the associated S3 product. The files to download must be placed in the folder "/data/inputs/s3_reflectances/" and they must take into account the naming convention specified next.
  
 #### Names of the S3 bands used in the mentioned script:

   - prefix = s3_fname column in the dataset_config.xls file
   - prefix + B03_(Raw).tiff
   - prefix + B04_(Raw).tiff
   - prefix + B06_(Raw).tiff
   - prefix + B08_(Raw).tiff
   - prefix + B11_(Raw).tiff
   - prefix + B12_(Raw).tiff


## References

1) M. A. Brovelli, M. Cannata, and M. Rogora. Simile, a geospatial enabler of the monitoring of sustainable development goal 6 (ensure availability and sustainability of water for all). The International Archives of the Photogrammetry, Remote Sensing and Spatial Information Sciences, XLII-4/W20:3–10, 2019. doi: 10.5194/isprs-archives-XLII-4-W20-3-2019.
2) R. Vollenweider and J. Kerekes. Eutrophication of waters: monitoring, assessment and control. OECD, 1982.
3) S. Cogliati, F. Sarti, L. Chiarantini, M. Cosi, R. Lorusso, E. Lopinto, F. Miglietta, L. Genesio, L. Guanter, A. Damm, S. Pérez-López, D. Scheffler, G. Tagliabue, C. Panigada, U. Rascher, T. Dowling, C. Giardino, and R. Colombo. The prisma imaging spectroscopy mission: overview and first performance analysis. Remote Sensing of Environment, 262:112499, 2021. ISSN 0034-4257. doi: https://doi.org/10.1016/j.rse.2021.112499. URL: https://www.sciencedirect.com/science/article/pii/S0034425721002170.
4) S.-E. Qian. Hyperspectral satellites, evolution, and development history. IEEE Journal of Selected Topics in Applied Earth Observations and Remote Sensing, 14:7032– 7056, 2021. doi: https://doi.org/10.1109/JSTARS.2021.3090256.


