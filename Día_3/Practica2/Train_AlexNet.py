"""
Entrenamiento AlexNet Cats vs Dogs

Giorgio Morales - INICTEL-UNI
"""

from keras.layers import Input, Dense, BatchNormalization, Flatten, Conv2D, MaxPooling2D, Dropout
from keras.models import Model
from keras.optimizers import Adam
import h5py
import numpy as np
##############################################################################
##############################################################################
#                             Define Model
##############################################################################
##############################################################################


def alex_net(input_shape=(227, 227, 3), classes=2):

    # Define the input as a tensor with shape input_shape
    x_input = Input(input_shape)

    # Stage 1
    #x = Conv2D..
    #x = MaxPooling2D..
    #x = BatchNormalization..

    # Stage 2
    # x = Conv2D..
    # x = MaxPooling2D..
    # x = BatchNormalization..

    # Stage 3
    # x = Conv2D..
    # x = Conv2D..
    # x = Conv2D..
    # x = MaxPooling2D..
    # x = BatchNormalization..

    # Stage 4
    #x = Flatten
    #x = Dense
    #x = Dense
    
    if classes == 2:
        x = Dense(1, activation='sigmoid', name='FC_3')(x)
    else:
        x = Dense(classes, activation='softmax', name='FC_3')(x)

    # Create model
    model = Model(input=x_input, output=x)

    return model


##############################################################################
##############################################################################
#                             Read Data
##############################################################################
##############################################################################

# LEE ARCHIVO HDF5
hdf5_file = h5py.File('dataset/datasetdogvscat.hdf5', "r")

# CREA VARIABLES DE ENTRENAMIENTO
train_x = np.array(hdf5_file["train_img"][...])
train_y = np.array(hdf5_file["train_labels"][...])
# CREA VARIABLES DE TEST
test_x = np.array(hdf5_file["test_img"][...])

train_x = train_x/255.

##############################################################################
##############################################################################
#                                Training
##############################################################################
##############################################################################

model = alex_net(input_shape=(256, 256, 3), classes=2)
optimizer = Adam(lr=0.0003, beta_1=0.9, beta_2=0.999, epsilon=1e-08)
model.compile(optimizer=optimizer, loss='binary_crossentropy', metrics=['accuracy'])

model.summary()

# Train!
print("Comienza entrenamiento...................................................................")
history = model.fit(train_x, train_y,
                  epochs=200, batch_size=16,
                  validation_split=0.11111,
                  shuffle=True)

# Guarda historial
np.save('Alex_acc.npy', history.history['acc'])
np.save('Alex_val_acc.npy', history.history['val_acc'])
np.save('Alex_loss.npy', history.history['loss'])
np.save('Alex_val_loss.npy', history.history['val_loss'])
model.save('Redes/AlexCatvsDog.h5')
