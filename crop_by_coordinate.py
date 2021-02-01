import numpy as np
import gdal, gdalconst, osr



# (EPSG: 4326, WGS 84) -> (EPSG: 32654, WGS 84 / UTM zone 54N)
ESPG = 32654

# Insert QuickBird image coordinates
# QB0613 maybe numbers in gdalinfo is right
ulx_qb = 269606.400 # 269815.96 # 138.4260861
uly_qb = 4047600.000 # 4047599.85 # 36.5460611
brx_qb = 277814.400 # 277608.75 # 138.5200667
bry_qb = 4039389.600 # 4039390.02 # 36.4740556
# QB0813
# ulx_qb  = 269821.36
# uly_qb  = 4047600.82
# brx_qb  = 277614.15
# bry_qb  = 4039391.00

# Insert target image info
archive_dict = {
    'AV2070606-1': [234024.30, 4113570.16, 317338.46, 4026855.29, 8556, 8450], #[234030.24, 4113601.75, 317349.65, 4026913.01, 8556, 8450],
    'AV2070606-2': [219699.05, 4059183.58, 302928.79, 3972395.92, 8552, 8446],
    'AV2070812': [245220.14, 4077490.85, 399928.60, 3982077.85, 15662, 9221],
    'PSR070701-2': [220898.38, 4093028.53, 298374.11, 4025446.72, 6200, 5400],
    'PSR070816-2': [220482.24, 4093931.02, 297920.21, 4025235.27, 6200, 5500],
}
path = 'AV2070606-1'
ulx_target = archive_dict[path][0]
uly_target = archive_dict[path][1]
brx_target = archive_dict[path][2]
bry_target = archive_dict[path][3]
x_num = archive_dict[path][4]
y_num = archive_dict[path][5]
band_num = 4
# res = 12.5
h_res = (brx_target - ulx_target)/x_num
v_res = (bry_target - uly_target)/y_num
tif_path = f'/Users/ryotanomura/沖研/satellite/tsumagoi/fixed/avnir/ALAV2A072642860-O1B2G_U_viewable.img'
output_path = 'test1.tif'

# print(f'hres: {h_res}, vres: {v_res}')


def nearest_index(sorted_list, value):
    sorted_array = np.array(sorted_list)
    return np.argmin(np.abs(sorted_array - value))

x_array = np.linspace(ulx_target, brx_target, x_num)
y_array = np.linspace(bry_target, uly_target, y_num)

index_ulx = nearest_index(x_array, ulx_qb)
index_uly = y_num - nearest_index(y_array, uly_qb) - 1 # Convert to image axis
index_brx = nearest_index(x_array, brx_qb)
index_bry = y_num - nearest_index(y_array, bry_qb) - 1 # Convert to image axis
print(index_ulx, index_uly, index_brx, index_bry)

x_num_out = int(index_brx - index_ulx + 1)
y_num_out = int(index_bry - index_uly  + 1)

# input target image
src = gdal.Open(tif_path,gdalconst.GA_ReadOnly)
geo_transform = src.GetGeoTransform()
proj = src.GetProjection()

band_array = []
for i in range(band_num):
    band = src.GetRasterBand(i+1).ReadAsArray()
    band_array.append(band)

band_array = np.array(band_array)
print(band_array.shape)

# crop array
band_array = band_array[:, index_uly:index_bry+1, index_ulx:index_brx+1]
print(band_array.shape)

# save array
dtype = gdal.GDT_Float32
output_src = gdal.GetDriverByName('GTiff').Create(output_path, x_num_out, y_num_out, band_num, dtype)
# output_src.SetGeoTransform((ulx_qb, h_res, 0, uly_qb, 0, v_res))
# srs = osr.SpatialReference()
# srs.ImportFromEPSG(ESPG)
# output_src.SetProjection(srs.ExportToWkt())
output_src.SetGeoTransform(geo_transform)
output_src.SetProjection(proj)

for i in range(band_num):
    output_src.GetRasterBand(i+1).WriteArray(band_array[i])
output_src.FlushCache()
output_src = None