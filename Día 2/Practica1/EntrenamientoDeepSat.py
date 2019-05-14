# -*- coding: utf-8 -*-
"""
Entrenamiento DeepSat en RGB

Giorgio Morales - INICTEL-UNI
"""

import matplotlib.pyplot as plt
import scipy.io as sio
import numpy as np
from keras.optimizers import Adam
from keras.layers import Input, Dense, BatchNormalization, Flatten, Conv2D, MaxPooling2D
from keras.models import Model
from keras.initializers import glorot_uniform
from keras.callbacks import ModelCheckpoint

import keras.backend as K
K.set_image_data_format('channels_last')
K.set_learning_phase(1)


##############################################################################
##############################################################################
#                            Build the model
##############################################################################
##############################################################################

def CNNsimple(input_shape=(28, 28, 4), classes=6):
    """
    CONV1(28×28×48)→BN1→MAxP1→
    CONV2(14×14×64)→BN2→MAxP2→
    CONV3(7×7x128)→
    CONV4(3×3×256)→
    FC5(128)→Softmax.
    
    Arguments:
    input_shape -- shape of the images of the dataset
    classes -- integer, number of classes

    Returns:
    model -- a Model() instance in Keras
    """

    # Define the input as a tensor with shape input_shape
    x_input = Input(input_shape)
    
    # Stage 1 Conv2D + BN + MaxPooling
    x = Conv2D(48, (5, 5), strides=(1, 1), padding='same', name='conv1', kernel_initializer=glorot_uniform(seed=0))(x_input)
      # BN name='bn_conv1'
      # MaxPool
    
    # Stage 2
      # Conv2D name='conv2'
      # BN name='bn_conv2'
      # MaxPool

    # Stage 3
      # Conv2D name='conv3'

    # Stage 4
      # Conv2D name='conv4'

    # output layer
      # Flatten
      # Dense

      # Final Dense

    # Create model (states inputs and outputs)


    return model    


##############################################################################
##############################################################################
#                             Read Data
##############################################################################
##############################################################################

# LECTURA DE BASE DE DATOS
mat_contents = sio.loadmat('Datasets/sat-6-full.mat')
train_x = mat_contents['train_x']
train_y = mat_contents['train_y']       # Train = 324000 samples (80%)
test_x = mat_contents['test_x']
test_y = mat_contents['test_y']         # Test = 81000 samples (20%)
mat_contents = None

# Split sets
val_x = test_x[:, :, :, 0:40500]
val_y = test_y[:, 0:40500]
test_x = test_x[:, :, :, 40500:]
test_y = test_y[:, 405000:]

# Reorder
train_x = train_x.transpose(3, 0, 1, 2)
val_x = val_x.transpose(3, 0, 1, 2)
test_x = test_x.transpose(3, 0, 1, 2)

train_y = train_y.transpose(1, 0)
val_y = val_y.transpose(1, 0)
test_y = test_y.transpose(1, 0)

# Cut RGB
train_x = np.concatenate((train_x, val_x), axis=0)
train_y = np.concatenate((train_y, val_y), axis=0)
train_x = train_x[:, :, :, 0:3]
test_x = test_x[:, :, :, 0:3]
val_x = None
val_y = None


# Normalize image vectors
train_x = train_x / 255.
test_x = test_x / 255.
mean = np.mean(train_x)
std = np.std(train_x)
train_x -= mean  # zero-center
train_x /= std  # normalize
print("Mean: " + str(mean))
print("Std: " + str(std))
test_x -= mean  # zero-center
test_x /= std  # normalize


print("number of training examples = " + str(train_x.shape[0]))
print("number of test examples = " + str(test_x.shape[0]))
print("x_train shape: " + str(train_x.shape))
print("Y_train shape: " + str(train_y.shape))
print("x_test shape: " + str(test_x.shape))
print("Y_test shape: " + str(test_y.shape))


##############################################################################
##############################################################################
#                                Training
##############################################################################
##############################################################################

optimizer = Adam(lr=0.0001, beta_1=0.9, beta_2=0.999, epsilon=1e-08)

# Crea el modelo con una entrada de (28,28,3) y 3 clases  (1 línea)

# Compila el modelo usando "optimizier" como optimizador, función de costo categorical y como métrica accuracy (1 línea)


model.summary()

# checkpoint
filepath = "weights-{epoch:03d}-{val_acc:.3f}.h5"
checkpoint = ModelCheckpoint(filepath, verbose=1, save_best_only=True, mode='min')
callbacks_list = [checkpoint]

# Train!
print("Comienza entrenamiento...................................................................")
history = model.fit(train_x, train_y,
                  epochs=200, batch_size=128,
                  validation_split=0.11111,
                  #shuffle=True,
                  callbacks=callbacks_list)

# Guarda historial
np.save('acc.npy', history.history['acc'])
np.save('val_acc.npy', history.history['val_acc'])
np.save('loss.npy', history.history['loss'])
np.save('val_loss.npy', history.history['val_loss'])

# Print loss and accuracy history
plt.figure()
plt.plot(history.history['acc'])
plt.plot(history.history['val_acc'])
plt.legend(['train', 'validation'], loc='upper left')
plt.title('Model Accuracy')
plt.ylabel('accuracy')
plt.xlabel('epoch')
plt.show()
plt.figure()
plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.legend(['train', 'validation'], loc='upper left')
plt.title('Model Loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.show()

preds = model.evaluate(test_x, test_y)
print("Loss = " + str(preds[0]))
print("Test Accuracy = " + str(preds[1]))

model.save('Redes/CNNsimpleSAT6.h5')




