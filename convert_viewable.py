import numpy as np
import gdal, gdalconst, osr

input_path = '../tsumagoi/ALOS/AV2070812/ALAV2A082412890-O1B2G_U.tif'
output_path = '../tsumagoi/ALOS/AV2070812/ALAV2A082412890-O1B2G_U_viewable.img'
BAND_NUM = 4
DTYPE = gdal.GDT_Float32

input_src = gdal.Open(input_path,gdalconst.GA_ReadOnly)
geo_transform = input_src.GetGeoTransform()
proj = input_src.GetProjection()

print(geo_transform)
print(proj)

band_array = []
for i in range(BAND_NUM):
    band = input_src.GetRasterBand(i+1).ReadAsArray()
    band_array.append(band)
band_array = np.array(band_array)

YSIZE, XSIZE = band_array.shape[1:]

# ALOS PALSAR image has 96 header bytes
# print(band_array.shape)
# channels, height, width = band_array.shape
# band_array = band_array[:,:,(width-XSIZE):]
# print(band_array.shape)

# Write bands with geo data
output_src = gdal.GetDriverByName('GTiff').Create(output_path, XSIZE, YSIZE, BAND_NUM, DTYPE)
output_src.SetGeoTransform(geo_transform)
# srs = osr.SpatialReference()
# srs.ImportFromEPSG(ESPG)
output_src.SetProjection(proj)

print(XSIZE, YSIZE, band_array.shape)

for i in range(BAND_NUM):
    output_src.GetRasterBand(i+1).WriteArray(band_array[i])
output_src.FlushCache() 
output_src = None