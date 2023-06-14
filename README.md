# Regression-Chl-a-PRISMA
Regression of Chlorophyll-a values in Insubric Lakes with PRISMA hyperspectral imagery

## Main objectives and introduction to the project

The main objective of this work was to determine the concentration of chlorophyll-a in a selection of Insubric Lakes, namely Lake Maggiore, Lake Como, and Lake Lugano, located in the cross-basin between Italy and Switzerland. These lakes were previously analyzed in the "Informative System for the Integrated Monitoring of Insubric Lakes and their Ecosystems" (SIMILE) project \cite{brovelli-2019}, which provided the baseline for the production of ground truth information used in our study. Further details about SIMILE will be provided in Section \ref{section:simile_project}.\\

Considering the importance of freshwater for communities and the role of lakes in ecosystems, it becomes evident that monitoring and managing water quality is a critical requirement for society. Specifically, this study aimed to develop a model to predict chlorophyll-a concentrations using hyperspectral imagery, with a set of previously computed chlorophyll-a (Chl-a) concentration maps from the SIMILE project serving as the ground truth data.\\

Chlorophyll-a is a relevant parameter in the monitoring of the lakes because it can be used as an estimator of their biomass concentration. This parameter is particularly important because it allows understanding the trophic state of the lakes, which is associated with their biodiversity: the presence of nutrients in a lake can produce an overgrowth of algae and other aquatic plants that when they die are decomposed causing consumption of the oxygen in the water and affecting the biodiversity, process known as eutrophication \cite{vollenweider1982eutrophication}.\\

The decision of using hyperspectral imagery was mainly based on the advantage provided by this kind of information which offers a higher spectral resolution with respect to multispectral imagery. Given that the hyperspectral images provide a high spectral resolution, they form a datacube in which the number of channels adds a third dimension and for each pixel,  it is provided the spectral response of each of the measurement bands \cite{evolution_hsi}. This high spectral resolution can be exploited to understand the physicochemical composition of the Earth \cite{evolution_hsi}. In our specific case, we used this information to train machine and deep learning models in order to predict chlorophyll-a concentrations on the mentioned insubric lakes.

Our study selected the Italian hyperspectral imager PRecursore IperSpettrale della Missione Applicativa (PRISMA) as the instrument to get the input data used for the proposed objective. The PRISMA mission was led by the Italian Space Agency (ASI), launched in 2019, and it is composed of VNIR and SWIR spectrometers and a panchromatic camera \cite{evolution_hsi}\cite{PRISMA}.

This repository was prepared to carry out the experimental development of the project. Specifically it covered the complete set of manipulations applied on the input data: coregistration of input images, the intersection between hyperspectral PRISMA images and the associated chlorophyll-a maps, pre-processing steps related to the removal of anomalous pixels, among others. Also, there was implemented the dimensionality reduction technique Principal Component Analysis to reduce the dimensionality of the hyperspectral input images and different normalization which were applied on the input data.

Regarding the models taken into account for this project, a total of four model typologies were taken into account: Random Forest Regressor, Support Vector Regressor, Long-Short Term Memory networks and Gated Recurrent Unit networks.

## Technical considerations for using this repository

### Downloading images

- The file "dataset_config.xls" present in the folder "data" specifies the filenames and relevant information associated with the different products to be download for each use case.
- The AOI to be considered for all the imagery to be downloaded, is available in the folder "data" and the filename of this bounding box is "
- The ground truth maps (chlorophyll-a maps) are present in the folder "SIMILE_chl_a" which is inside the folder "data" of this repository.
- On the other hand, the products which must be downloaded are:
+ PRISMA images: the processing level used was L2D. This information must be downloaded from the official website of the PRISMA mission, using the appropiate credentials. The mentioned file "dataset_config.xls" specifies the name corresponding to each acquisition. The AOI is present in the folder 

## References
