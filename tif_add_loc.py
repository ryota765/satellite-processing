import numpy as np
import gdal, gdalconst, osr


# Set parameters
"""
(UL_X,UL_Y) => transform image lat, lon with (https://epsg.io/transform)
"""
# archive
loc_dict_frame = {
    'PSR070701-1': [227911.74, 4038591.19, 6300, 5500],
    'PSR070701-2': [220898.38, 4093028.53, 6200, 5400],
    'PSR070816-1': [228184.17, 4038693.69, 6300, 5500],
    'PSR070816-2': [220482.24, 4093931.02, 6200, 5500],
    'PSR071001-1': [228625.42, 4038457.65, 6300, 5500],
    'PSR071001-2': [220185.73, 4093052.01, 6200, 5400],
    'PSR071018': [181330.06, 4043630.34, 6500, 5700],
    'PSR080518': [222594.54, 4093083.98, 6200, 5400],
    'PSR080703': [219372.99, 4092745.56, 6200, 5400],
    'PSR081003': [217450.09, 4093920.42, 6200, 5500],
}
DTYPE = gdal.GDT_Float32
folder_name = 'PSR071001-2'
input_path = f'../tsumagoi/ALOS/{folder_name}/hhhv071001.img'
output_path = f'../tsumagoi/ALOS/{folder_name}/{folder_name}_hhhv.img'
XSIZE = loc_dict_frame[folder_name][2]
YSIZE = loc_dict_frame[folder_name][3]
UL_X = loc_dict_frame[folder_name][0]
UL_Y = loc_dict_frame[folder_name][1]
ESPG = 32654
H_RES = 12.5
BAND_NUM = 2


# input bands from source
input_src = gdal.Open(input_path,gdalconst.GA_ReadOnly)

band_array = []
for i in range(BAND_NUM):
    band = input_src.GetRasterBand(i+1).ReadAsArray()
    band_array.append(band)
band_array = np.array(band_array)

# ALOS PALSAR image has 96 header bytes
print(band_array.shape)
channels, height, width = band_array.shape
band_array = band_array[:,:,(width-XSIZE):]
print(band_array.shape)

# Write bands with geo data
output_src = gdal.GetDriverByName('GTiff').Create(output_path, XSIZE, YSIZE, BAND_NUM, DTYPE)
output_src.SetGeoTransform((UL_X, H_RES, 0, UL_Y, 0, -H_RES))
srs = osr.SpatialReference()
srs.ImportFromEPSG(ESPG)
output_src.SetProjection(srs.ExportToWkt())

for i in range(BAND_NUM):
    output_src.GetRasterBand(i+1).WriteArray(band_array[i])
output_src.FlushCache() 
output_src = None
