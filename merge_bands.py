import numpy as np
import gdal, gdalconst, osr

num_bands = 2
target_dir = '../tsumagoi/ALOS/PSR071001-1'
img_paths = [f'{target_dir}/img-hh-alpsrp089780710-h1.img', f'{target_dir}/img-hv-alpsrp089780710-h1.img']
out_path = f'{target_dir}/hhhv.img'
assert len(img_paths) == num_bands

# Input
band_array = []

for i in range(num_bands):
    src = gdal.Open(img_paths[i],gdalconst.GA_ReadOnly)
    band = src.GetRasterBand(1).ReadAsArray()
    band_array.append(band)
band_array = np.array(band_array)

# Output
dtype = gdal.GDT_Float32
xsize = band_array.shape[2]
ysize = band_array.shape[1]

output = gdal.GetDriverByName('GTiff').Create(out_path, xsize, ysize, num_bands, dtype)
for i in range(num_bands):
    output.GetRasterBand(1).WriteArray(band_array[i])

output.FlushCache() 
output = None