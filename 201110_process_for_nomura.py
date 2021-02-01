import numpy as np

ywidth = 4998
xwidth = 5001
import pyproj

srcproj = pyproj.Proj(init="EPSG:32619")
outproj = pyproj.Proj(init="EPSG:4326")

#pyproj.transform(outproj,srcproj,118.802266,-4.52113836)

size = 20

ULY = 600000
ULX = 499980
ULY2 = 600000 - 109800 - ywidth*20*2
ULX2 = 499980 + 109800 + xwidth*20*1

#pyproj.transform(srcproj,outproj,ULX,ULY)

ynum = int((ULY-ULY2)/size)
xnum = int((ULX2-ULX)/size)

out_lat = np.zeros((ynum,xnum))
out_lon = np.zeros((ynum,xnum))

for y in range(0,ynum):
    for x in range(0,xnum):
        src_y = ULY - size*(y+0.5)
        src_x = ULX + size*(x+0.5)
        out_lon[y,x], out_lat[y,x] = pyproj.transform(srcproj,outproj,src_x,src_y)

    if y%int(ynum/100) == 0:
        print(str(y/int(ynum/100)) + "/100 finished")

out_lon.astype("f").tofile("/data28/rarai/201028_colombia/sen2_data/x_projection_sentinel2.bin")
out_lat.astype("f").tofile("/data28/rarai/201028_colombia/sen2_data/y_projection_sentinel2.bin")

