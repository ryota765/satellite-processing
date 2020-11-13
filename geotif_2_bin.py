import numpy as np
import gdal, gdalconst


'''
Convert
tif image -> npy array
'''

tif_path = 'out.tif' # 'output/ALAV2A072642860-O1B2G_U.tif' # '/Users/ryotanomura/沖研/satellite/tsumagoi/ALOS/Palsardata(090412)_Tadono/200904210224/l1data/output/ALOS-P1_5GUA-ORBIT__ALPSRP171320720_Cal_Spk.tif' 
output_path = 'out.npy'
band_num = 2

src = gdal.Open(tif_path,gdalconst.GA_ReadOnly)

band_array = []
for i in range(band_num):
    band = src.GetRasterBand(i+1).ReadAsArray()
    band_array.append(band)

band_array = np.array(band_array)
np.save(output_path, band_array)

