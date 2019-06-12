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
    
    # Stage 1
    x = Conv2D(48, (5, 5), strides=(1, 1), padding='same', name='conv1', kernel_initializer=glorot_uniform(seed=0))(x_input)
    x = BatchNormalization(axis=3, name='bn_conv1')(x)
    x = MaxPooling2D(pool_size=(2, 2), strides=(2, 2), padding='valid')(x)
    
    # Stage 2
    x = Conv2D(64, (3, 3), strides=(1, 1), padding='same', name='conv2', kernel_initializer=glorot_uniform(seed=0))(x)
    x = BatchNormalization(axis=3, name='bn_conv2')(x)
    x = MaxPooling2D(pool_size=(2, 2), strides=(2, 2), padding='valid')(x)

    # Stage 3
    x = Conv2D(128, (3, 3), strides=(1, 1), padding='same', name='conv3', kernel_initializer=glorot_uniform(seed=0))(x)

    # Stage 4
    x = Conv2D(256, (3, 3), strides=(2, 2), padding='valid', name='conv4', kernel_initializer=glorot_uniform(seed=0))(x)

    # output layer
    x = Flatten()(x)
    x = Dense(128, activation='relu', name='fc' + str(128), kernel_initializer=glorot_uniform(seed=0))(x)

    x = Dense(classes, activation='softmax', name='fc' + str(classes), kernel_initializer=glorot_uniform(seed=0))(x)

    # Create model
    model = Model(inputs=x_input, outputs=x, name='CNNsimple')

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

model = CNNsimple(input_shape=(28, 28, 3), classes=6)
optimizer = Adam(lr=0.0001, beta_1=0.9, beta_2=0.999, epsilon=1e-08)
model.compile(optimizer=optimizer, loss='categorical_crossentropy', metrics=['accuracy'])

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




