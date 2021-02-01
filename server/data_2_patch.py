import os
import glob

import numpy as np
import matplotlib.pyplot as plt


PATCH_SIZE = 33
PATCH_MARGIN = PATCH_SIZE//2
OUTPUT_SIZE = 17 # Change by model
OUTPUT_MARGIN = OUTPUT_SIZE//2
CHANNEL_NUM = 3

h = 6400
w = 9000
SPACING=20

num_rows = (h - (PATCH_SIZE//2)*2) // SPACING
num_cols = (w - (PATCH_SIZE//2)*2) // SPACING

val_h_threshold = test_h_threshold = h
val_w_threshold = test_w_threshold = w

save_dir = 'output'


def cal_ndvi(img_r, img_ir):
    return (img_ir - img_r) / (img_ir + img_r)

def standarlization(img_2d):
    return (img_2d - np.min(img_2d)) / (np.max(img_2d) - np.min(img_2d))

def feature_extraction(img):
    img_ndvi = cal_ndvi(img[4], img[5])
    img_vh = standarlization(img[0])
    img_vv = standarlization(img[1])
    img_elev = standarlization(img[6])
    img_std = np.array([img_ndvi, img_vh, img_vv, img_elev])
    return img_std

def generate_patch(img, SPACING=SPACING):
    
    img_std = feature_extraction(img)
    
    train_patch = []
    val_patch = []
    test_patch = [] # For temporary test (Actual test uses patch with spacing=1)

    for i in range(num_rows):
        for j in range(num_cols):

            start_h = i*SPACING
            end_h = PATCH_SIZE+i*SPACING
            start_w = j*SPACING
            end_w = PATCH_SIZE+j*SPACING
            img_extracted = img_std[:,start_h:end_h, start_w: end_w]
            
            if end_h <= val_h_threshold or end_w <= val_w_threshold:
                train_patch.append(img_extracted)
            elif end_h <= test_h_threshold or end_w <= test_w_threshold:
                # np.save(os.path.join(save_dir, f'train_{i}_{j}.npy'), img_extracted)
                val_patch.append(img_extracted)
            else:
                # np.save(os.path.join(save_dir, f'test_{i}_{j}.npy'), img_extracted)
                test_patch.append(img_extracted)

    train_patch = np.array(train_patch)
    val_patch = np.array(val_patch)
    test_patch = np.array(test_patch)

    print(train_patch.shape)
    print(val_patch.shape)
    print(test_patch.shape)
    
    return train_patch, val_patch, test_patch

img_list = glob.glob('data/*.npy')

for path in img_list:
    idx = path[-8:-4]
    img = np.load(path)
    img = img[:7]
    
    train_patch, val_patch, test_patch = generate_patch(img)
    np.save(os.path.join(save_dir, f'train_{idx}.npy'), train_patch)
