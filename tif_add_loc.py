import numpy as np
import gdal, gdalconst, osr


# Set parameters
"""
(UL_X,UL_Y) => transform image lat, lon with (https://epsg.io/transform)
(EPSG: 4326, WGS 84) -> (EPSG: 32654, WGS 84 / UTM zone 54N)
"""
# archive
loc_dict_frame = {
    # DIR_NAME: [ulx, uly, brx, bry, x_num, y_num]
    # ALOS Palsar
    'PSR070701-1': [227911.74, 4038591.19, 6300, 5500],
    'PSR070701-2': [220898.38, 4093028.53, 298374.11, 4025446.72, 6200, 5400],
    'PSR070816-1': [228184.17, 4038693.69, 6300, 5500],
    'PSR070816-2': [220482.24, 4093931.02, 297920.21, 4025235.27, 6200, 5500],
    # 'PSR071001-1': [228625.42, 4038457.65, 6300, 5500],
    # 'PSR071001-2': [220185.73, 4093052.01, 6200, 5400],
    # 'PSR071018': [181330.06, 4043630.34, 6500, 5700],
    # 'PSR080518': [222594.54, 4093083.98, 6200, 5400],
    # 'PSR080703': [219372.99, 4092745.56, 6200, 5400],
    # 'PSR081003': [217450.09, 4093920.42, 6200, 5500],
    # ALOS Avnir (size in summary text is different from image size?)
    'AV2070606-1': [234030.23962515697, 4113601.7531219465, 317349.64689508255, 4026913.0147153023, 8556, 8450], # [234030.24, 4113601.75, 317349.65, 4026913.01, 8556, 8450, 138.006, 37.131], # res_12.5: [5425.2 5408]
    'AV2070606-2': [219699.05, 4059183.58, 302928.79, 3972395.92, 8552, 8446], # res_12.5: [5473.28 5405.44]
    # 'AV2070623': [174100.15, 4061240.03, 252805.80, 3973325.05, 8399, 8288], # res_12.5: [5375.36 5304.32]
    'AV2070812': [245220.14, 4077490.85, 399928.60, 3982077.85, 15662, 9221] # res_12.5: [10023.68 5901.44]
}
DTYPE = gdal.GDT_Float32
folder_name = 'AV2070606-1'
input_path = f'../tsumagoi/ALOS/{folder_name}/ALAV2A072642860-O1B2G_U.tif'
# output_path = f'{folder_name}_pre_.tif' # for single image
output_path =  f'../tsumagoi/ALOS/{folder_name}/{folder_name}_loc.img' # for merging image
XSIZE = loc_dict_frame[folder_name][4]
YSIZE = loc_dict_frame[folder_name][5]
UL_X = loc_dict_frame[folder_name][0]
UL_Y = loc_dict_frame[folder_name][1]
BR_X = loc_dict_frame[folder_name][2]
BR_Y = loc_dict_frame[folder_name][3]
ESPG = 32654
BAND_NUM = 4
# Adjust resolution dynamically
H_RES = abs(BR_X-UL_X)/XSIZE
V_RES = abs(UL_Y-BR_Y)/YSIZE
print(abs(BR_X-UL_X), abs(UL_Y-BR_Y))
print(H_RES, V_RES)

# input bands from source
input_src = gdal.Open(input_path,gdalconst.GA_ReadOnly)

band_array = []
for i in range(BAND_NUM):
    band = input_src.GetRasterBand(i+1).ReadAsArray()
    band_array.append(band)
band_array = np.array(band_array)

# ALOS PALSAR image has 96 header bytes
# print(band_array.shape)
# channels, height, width = band_array.shape
# band_array = band_array[:,:,(width-XSIZE):]
# print(band_array.shape)

# Write bands with geo data
output_src = gdal.GetDriverByName('GTiff').Create(output_path, XSIZE, YSIZE, BAND_NUM, DTYPE)
output_src.SetGeoTransform((UL_X, H_RES, 0, UL_Y, 0, -V_RES))
srs = osr.SpatialReference()
srs.ImportFromEPSG(ESPG)
output_src.SetProjection(srs.ExportToWkt())

print(XSIZE, YSIZE, band_array.shape)

for i in range(BAND_NUM):
    output_src.GetRasterBand(i+1).WriteArray(band_array[i])
output_src.FlushCache() 
output_src = None
