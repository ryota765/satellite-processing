import numpy as np
from osgeo import gdal, gdalconst


'''
Convert
tif image -> npy array
'''

city = 'otofuke'
date = '1108'

tif_path = '/Users/ryotanomura/沖研/satellite/sar_2_ndvi/tsumagoi/land_cover/land_cover.tif' # f'/Users/ryotanomura/沖研/satellite/sar_2_ndvi/sentinel/{city}/output/collocate_crop_nn_cloud_{date}.tif'
output_path = '/Users/ryotanomura/沖研/satellite/sar_2_ndvi/tsumagoi/land_cover/land_cover.npy' # f'/Users/ryotanomura/沖研/satellite/sar_2_ndvi/sentinel/{city}/bin/raw/collocate_crop_nn_cloud_{date}.npy'
band_num = 1

src = gdal.Open(tif_path,gdalconst.GA_ReadOnly)

# check band number
# if src is not None:
#     print ("band count: " + str(src.RasterCount))

band_array = []
for i in range(band_num):
    band = src.GetRasterBand(i+1).ReadAsArray()
    band_array.append(band)

band_array = np.array(band_array)

print(band_array.shape)
np.save(output_path, band_array)

