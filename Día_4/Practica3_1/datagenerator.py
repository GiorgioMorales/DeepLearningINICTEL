# -*- coding: utf-8 -*-
"""
GENERADOR DE DATA EN KERAS

"""
import numpy as np
import keras
import cv2


class DataGenerator(keras.utils.Sequence):
    'Inicializa variables'
    def __init__(self, list_IDs, batch_size=64, dim=(513, 513), n_channels=3, path='',
                 shuffle=True):
        'Initialization'
        self.dim = dim
        self.batch_size = batch_size
        self.list_IDs = list_IDs
        self.n_channels = n_channels
        self.path = path
        self.shuffle = shuffle
        self.on_epoch_end()

    def __len__(self):
        'Calcula el número de batches por época'
        return int(np.floor(len(self.list_IDs) / self.batch_size))

    def __getitem__(self, index):
        'Genera un batch'
        # Genera los índices del batch
        indexes = self.indexes[index*self.batch_size:(index+1)*self.batch_size]

        # Crea una lista de IDs correspondientes a indexes
        list_IDs_temp = [self.list_IDs[k] for k in indexes]

        # Genera la data
        X, y = self.__data_generation(list_IDs_temp)

        return X, y

    def on_epoch_end(self):
        'Actualiza indexes'
        self.indexes = np.arange(len(self.list_IDs))
        if self.shuffle == True:
            np.random.shuffle(self.indexes)

    def __data_generation(self, list_IDs_temp):
        'Genera data' # X : (n_samples, *dim, n_channels)
        # Inicializa input y output
        X = np.empty((self.batch_size, *self.dim, self.n_channels))
        y = np.empty((self.batch_size, *self.dim, 1))

        # Generate data
        for i, ID in enumerate(list_IDs_temp): #[(0,'C://dataset//im423.jpg'),(1,'C://dataset//im672.jpg'),...]
            
            addr = self.path + ID[len(self.path):]
            addrm = addr[0:-4]+"_mask.tif"  # Lee dirección de la máscara / label

            # Lee imagen con OpenCV
            img = cv2.imread(addr)
            b, g, r = cv2.split(img)
            img2 = cv2.merge([r, g, b])
            
            # Guarda muestra
            X[i, ] = img2.astype(np.uint8)[0:512, 0:512, :]
            # Guarda máscara / label
            y[i, ] = np.reshape((cv2.imread(addrm, 0)/255)[0:512, 0:512], (512, 512, 1))

        return X, y
