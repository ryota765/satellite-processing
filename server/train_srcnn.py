# import sys

# sys.path.remove('/usr/local/lib/python3.9/site-packages')
# sys.path.append('/usr/local/lib/python3.7/site-packages')

import glob

import numpy as np

import keras
from keras import optimizers
from keras.models import Sequential
from keras.layers import Input, Activation, Conv2D, Flatten, Dense, Dropout
from keras.callbacks import EarlyStopping, ModelCheckpoint
from keras import backend as K

'''Hyper Parameters'''
CHANNEL_NUM = 4
n_epochs = 250
batch_size = 128

X_train_path = 'data/X_train.npy'
y_train_path = 'data/y_train.npy'
X_val_path = 'data/X_val.npy'
y_val_path = 'data/y_val.npy'
model_save_path = 'model/srcnn_4ch.h5'


def create_srcnn(channel_num=CHANNEL_NUM):
    model = Sequential()

    model.add(Conv2D(filters=64, kernel_size= 9, activation='relu', padding='same', input_shape=(None,None,channel_num)))
    model.add(Conv2D(filters=32, kernel_size= 1, activation='relu', padding='same'))
    model.add(Conv2D(filters=1, kernel_size= 5, padding='same'))

    return model


def psnr(y_true,y_pred):
    return -10*K.log(K.mean(K.flatten((y_true-y_pred))**2)
    )/np.log(10)


class DataGenerator(keras.utils.Sequence):
    'Generates data for Keras'
    def __init__(self, X_IDs, batch_size=128, dim=(125,125), n_input_channels=4, n_output_channels=1, shuffle=True):
        'Initialization'
        self.dim = dim
        self.batch_size = batch_size
        self.X_IDs = X_IDs
        # self.y_IDs = y_IDs
        self.n_input_channels = n_input_channels
        self.n_output_channels = n_output_channels
        self.shuffle = shuffle
        self.on_epoch_end()

    def __len__(self):
        'Denotes the number of batches per epoch'
        return int(np.floor(len(self.X_IDs) / self.batch_size))

    def __getitem__(self, index):
        'Generate one batch of data'
        # Generate indexes of the batch
        indexes = self.indexes[index*self.batch_size:(index+1)*self.batch_size]

        # Find list of IDs
        X_IDs_tmp = [self.X_IDs[k] for k in indexes]

        # Generate data
        X, y = self.__data_generation(X_IDs_tmp)

        return X, y

    def on_epoch_end(self):
        'Updates indexes after each epoch'
        self.indexes = np.arange(len(self.X_IDs))
        if self.shuffle == True:
            np.random.shuffle(self.indexes)

    def __data_generation(self, X_IDs_tmp):
        'Generates data containing batch_size samples' # X : (n_samples, *dim, n_channels)
        # Initialization
        X = np.empty((self.batch_size, *self.dim, self.n_input_channels), dtype=np.float32)
        y = np.empty((self.batch_size, *self.dim, self.n_output_channels), dtype=np.float32)

        # Generate data
        for i, X_ID in enumerate(X_IDs_tmp):
            # Store sample
            X[i,] = np.load(X_ID)

            # Store class
            y_ID = X_ID.replace('X', 'y')
            y[i,] = np.load(y_ID)

        return X, y


if __name__ == "__main__":
    # X_train = np.load(X_train_path) # [:,:,:,0][:, :, :, np.newaxis]
    # y_train = np.load(y_train_path)
    # X_val = np.load(X_val_path) # [:,:,:,0][:, :, :, np.newaxis]
    # y_val = np.load(y_val_path)

    X_IDs = glob.glob('data/all/X*.npy')

    model = create_srcnn()
    model.compile(loss='mean_absolute_error', optimizer= 'adam', metrics=[psnr])

    # model.fit(X_train, y_train, epochs=n_epochs, batch_size=batch_size, validation_data=(X_val, y_val))
    training_generator = DataGenerator(X_IDs)
    model.fit_generator(generator=training_generator, epochs=n_epochs, verbose=1)


    model.save_weights(model_save_path)