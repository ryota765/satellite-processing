import os
import glob

import numpy as np
import matplotlib.pyplot as plt

from keras import optimizers
from keras.models import Sequential
from keras.layers import Input, Activation, Conv2D, Flatten, Dense, Dropout
from keras.callbacks import EarlyStopping, ModelCheckpoint


'''Hyper Parameters'''
PATCH_SIZE = 33
PATCH_MARGIN = PATCH_SIZE//2
OUTPUT_SIZE = 17
OUTPUT_MARGIN = OUTPUT_SIZE//2
CHANNEL_NUM = 3
n_epochs = 500
batch_size = 128
train_path = 'data/all.npy'
model_save_path = 'model/cnn_prev_pt_niigata.h5'


# Methods for preprocessing
def create_data_map(train_patch, dist=OUTPUT_MARGIN):
    X_train = train_patch[:, 1:, :, :].transpose(0,2,3,1)
    y_train = train_patch[:, :1, (PATCH_MARGIN-dist):(PATCH_MARGIN+dist+1), (PATCH_MARGIN-dist):(PATCH_MARGIN+dist+1)].transpose(0,2,3,1)
    return X_train, y_train


def unison_shuffled_copies(a, b):
    assert len(a) == len(b)
    p = np.random.permutation(len(a))
    return a[p], b[p]


# Method for model
def create_prev_map_model():
    model = Sequential()

    model.add(Conv2D(48, (9, 9), padding='valid',input_shape=(PATCH_SIZE, PATCH_SIZE, CHANNEL_NUM)))
    model.add(Activation('relu'))

    model.add(Conv2D(32, (5, 5), padding='valid'))
    model.add(Activation('relu'))

    model.add(Conv2D(1, (5, 5), padding='valid'))

    sgd = optimizers.SGD(lr=0.0005, momentum=0.9)
    model.compile(loss='mae', optimizer='sgd')
    
    return model


if __name__ == "__main__":
    train_patch = np.load(train_path)
    print(train_patch.shape)

    X_train, y_train = create_data_map(train_patch)
    X_train, y_train = unison_shuffled_copies(X_train, y_train)

    model = create_prev_map_model()
    model.fit(X_train, y_train, epochs=n_epochs, batch_size=batch_size, verbose=1) 

    model.save_weights(model_save_path)

