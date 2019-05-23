# -*- coding: utf-8 -*-
"""
CREA BASE DE DATOS H5 DE PERROS Y GATOS
"""

from random import shuffle
import glob
import numpy as np
import h5py
import cv2

#####################################################################
#   LISTA IMÁGENES Y ETIQUETAS
#####################################################################

shuffle_data = True  # Flag para reordenar todos los archivos antes de guardar

train_path = 'dataset/train/*.jpg'
test_path = 'dataset/test/*.jpg'

# Lee direcciones de archivos en el folder de entrenamiento
train_addrs = glob.glob(train_path)
train_labels = [1 if 'cat' in addr else 0 for addr in train_addrs]  # 0 = Dog, 1 = Cat

# Reordena aleatoriamente las direcciones junto con sus respectivos labels
if shuffle_data:
    c = list(zip(train_addrs, train_labels))
    shuffle(c)
    train_addrs, train_labels = zip(*c)

train_addrs = train_addrs[0:10000]
train_labels = train_labels[0:10000]

# Lee direcciones de archivos en el folder de test
test_addrs = glob.glob(test_path)
test_addrs = test_addrs[0:100]
#####################################################################
#   CREA ARCHIVO HDF5
#####################################################################

# Establece tamaños (Imagen 256x256x3)
train_shape = (len(train_addrs), 256, 256, 3)   #10000x256x256x3
test_shape = (len(test_addrs), 256, 256, 3)     #100x256x256x3

# Crea archivo .hdf5
hdf5_path = 'dataset/datasetdogvscat.hdf5'  # address to where you want to save the hdf5 file
hdf5_file = h5py.File(hdf5_path, mode='w')

# Crea campos "train" y "test" con sus tamaños respectivos
hdf5_file.create_dataset("train_img", train_shape, np.uint8)
hdf5_file.create_dataset("test_img", test_shape, np.uint8)

# Asigna tamaños y guarda etiquetas
hdf5_file.create_dataset("train_labels", (len(train_labels),), np.int8)
hdf5_file["train_labels"][...] = train_labels

#####################################################################
#   CARGA Y GUARDA IMÁGENES
#####################################################################

for i in range(len(train_addrs)):

    # Imprime cuántas imágenes han sido grabadas cada 1000 imágenes
    if i % 1000 == 0 and i > 1:
        print('Train data: {}/{}'.format(i, len(train_addrs)))

    # cv2 carga imágenes como BGR, convertirlas a RGB
    addr = train_addrs[i]
    bgr_img = cv2.imread(addr)
    b, g, r = cv2.split(bgr_img)  # separa los canales
    rgb_img = cv2.merge([r, g, b])  # los vuelve a unir con el orden invertido

    hdf5_file["train_img"][i, ...] = rgb_img


for i in range(len(test_addrs)):

    # Imprime cuántas imágenes han sido grabadas cada 1000 imágenes
    if i % 1000 == 0 and i > 1:
        print('Test data: {}/{}'.format(i, len(train_addrs)))

    # cv2 carga imágenes como BGR, convertirlas a RGB
    addr = test_addrs[i]
    bgr_img = cv2.imread(addr)
    b, g, r = cv2.split(bgr_img)  # separa los canales
    rgb_img = cv2.merge([r, g, b])  # los vuelve a unir con el orden invertido

    hdf5_file["test_img"][i, ...] = rgb_img

hdf5_file.close()
