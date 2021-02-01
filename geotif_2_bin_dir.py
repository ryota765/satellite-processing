import glob

import numpy as np
from osgeo import gdal, gdalconst


'''
Convert
tif image -> npy array
'''

city = 'biei'

path_list = glob.glob(f'../sar_2_ndvi/{city}/modis/output/*.tif')

for path in path_list:
    tif_path = path
    output_path = tif_path.replace('/output/', '/bin/raw/').replace('.tif', '.npy')

    print(tif_path, '->', output_path)

    src = gdal.Open(tif_path,gdalconst.GA_ReadOnly)

    # check band number
    if src is not None:
        print ("band count: " + str(src.RasterCount))
        band_num = src.RasterCount

    band_array = []
    for i in range(band_num):
        band = src.GetRasterBand(i+1).ReadAsArray()
        band_array.append(band)

    band_array = np.array(band_array)

    print(band_array.shape)
    np.save(output_path, band_array)

