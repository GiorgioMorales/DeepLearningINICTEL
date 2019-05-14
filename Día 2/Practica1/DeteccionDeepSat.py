# -*- coding: utf-8 -*-
"""
Verificación de entrenamiento DeepSat en RGB

Giorgio Morales
"""
import matplotlib.pyplot as plt
import scipy.io as sio
import numpy as np
from keras.models import load_model

import keras.backend as K
K.set_image_data_format('channels_last')
K.set_learning_phase(0)


"""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""
""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""
Predicción de una muestra
""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""
"""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""   


def Predictinput(X, name_model):    
 
    # Load model
    model = load_model(name_model)  
    
    # Normalize image vectors
    X = X/255.
    mean = 0.446080688697
    std = 0.18487011389
    X -= mean  # zero-center
    X /= std # normalize

    # Evaluate
    Y = model.predict(X)
    
    return Y


def decode(x):
    
    ind = np.argmax(x)
    if ind == 0:
        str = 'building'
    elif ind == 1:
        str = 'barren'
    elif ind == 2:
        str = 'trees'
    elif ind == 3:
        str = 'grassland'
    elif ind == 4:
        str = 'road'
    elif ind == 5:
        str = 'water'
    
    return str
    
"""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""
""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""
EJECUTA
""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""
"""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""

# LECTURA DE BASE DE DATOS
mat_contents = sio.loadmat('Datasets/sat-6-full.mat')
train_x = mat_contents['train_x']
train_y = mat_contents['train_y']
test_x = mat_contents['test_x']
test_y = mat_contents['test_y']
mat_contents = None

# Split sets
test_x = test_x[:, :, :, 60750:]
test_y = test_y[:, 60750:]

# Reorder
test_x = test_x.transpose(3,0,1,2)
test_y = test_y.transpose(1,0)

# Cut RGB
test_x = test_x[:, :, :, 0:3]

# PREDICT INPUTS
Y = Predictinput(test_x[0:75, :, :, :], 'Redes/CNNsimpleSAT6.h5')

# DRAW RESULTS
fig = plt.figure()
for i, index in enumerate(range(0, 25)):
    a = fig.add_subplot(5, 5, i+1)
    a.set_title("Label: " + decode(test_y[index, :]) + "  Result: " + decode(Y[index, :]), fontsize=10)
    plt.imshow(test_x[index, :, :, :])
    plt.axis('off')

# Plot learning curves
acc = np.load('Resultados/acc.npy')
val_acc = np.load('Resultados/val_acc.npy')
loss = np.load('Resultados/loss.npy')
val_loss = np.load('Resultados/val_loss.npy')

plt.figure()
plt.plot(acc)
plt.plot(val_acc)
plt.legend(['train', 'validation'], loc='lower right')
plt.title('Model Accuracy')
plt.ylabel('accuracy')
plt.xlabel('epoch')
plt.show()
plt.savefig('accuracy.png', dpi=1200)
plt.figure()
plt.plot(loss)
plt.plot(val_loss)
plt.legend(['train', 'validation'], loc='lower right')
plt.title('Model Loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.show()
plt.savefig('loss2.png', dpi=1200)
