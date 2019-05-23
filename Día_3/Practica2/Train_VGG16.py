"""
Entrenamiento VGG16 Cats vs Dogs con Transfer Learning

Giorgio Morales - INICTEL-UNI
"""

from keras.applications.vgg16 import VGG16
from keras.models import Model, Sequential
from keras.layers import Dense, Flatten
from keras.optimizers import Adam
import h5py
import numpy as np

##############################################################################
##############################################################################
#                             Read Data
##############################################################################
##############################################################################

# LEE ARCHIVO HDF5
hdf5_file = h5py.File('dataset/datasetdogvscat2.hdf5', "r")

# CREA VARIABLES DE ENTRENAMIENTO
train_x = np.array(hdf5_file["train_img"][...])
train_y = np.array(hdf5_file["train_labels"][...])
# CREA VARIABLES DE TEST
test_x = np.array(hdf5_file["test_img"][...])

train_x = train_x/255.

##############################################################################
##############################################################################
#                    Define Model with random weights
##############################################################################
##############################################################################

vgg = VGG16(weights=None, include_top=True, input_shape=(224, 224, 3))

vgg.summary()

# Crear nuevo modelo desde la capa de entrada hasta la 2da capa fully-connected

model1 = Model(input=vgg.input, output=vgg.get_layer('fc2').output)

# Crear nuevo modelo reemplazando las 3 últimas capas fully-connected
# por FC1 (1024 unidades), FC2 (256 unidades), PREDICTION (una unidad)

model2 = Sequential()
for i in range(0, 19):
    model2.add(vgg.layers[i])
model2.add(Flatten())
model2.add(Dense(1024, activation='relu', name='fc1'))
model2.add(Dense(256, activation='relu', name='fc2'))
model2.add(Dense(1, activation='sigmoid', name='Prediction'))

model2.summary()

##############################################################################
##############################################################################
#                                Training
##############################################################################
##############################################################################

optimizer = Adam(lr=0.0003, beta_1=0.9, beta_2=0.999, epsilon=1e-08)
model2.compile(optimizer=optimizer, loss='binary_crossentropy', metrics=['accuracy'])

# Train!
print("Comienza entrenamiento...................................................................")
history = model2.fit(train_x, train_y,
                     epochs=200, batch_size=16,
                     validation_split=0.11111,
                     shuffle=True)

# Guarda historial
np.save('acc.npy', history.history['acc'])
np.save('val_acc.npy', history.history['val_acc'])
np.save('loss.npy', history.history['loss'])
np.save('val_loss.npy', history.history['val_loss'])
model2.save('Redes/VGGCatvsDog.h5')
##############################################################################
##############################################################################
#                   Define Model with Imagenet weights
##############################################################################
##############################################################################

vgg = VGG16(weights='imagenet', include_top=False, input_shape=(224, 224, 3))
vgg.summary()

# Crear nuevo modelo reemplazando la última capa fully-connected de 1000 clases
# (predictions) por una binaria

model3 = Sequential()
for i in range(0, 19):
    model3.add(vgg.layers[i])
model3.add(Flatten())
model3.add(Dense(1024, activation='relu', name='fc1'))
model3.add(Dense(256, activation='relu', name='fc2'))
model3.add(Dense(1, activation='sigmoid', name='Prediction'))

model3.summary()

# Bloquea el entrenamiento desde la primera capa hasta la capa "block5_pool"
for layer in model3.layers:
    layer.trainable = False
    if layer.name == 'block5_pool':
        break

# Verificar que el número de parámetros no entrenables ha aumentado
model3.summary()
##############################################################################
##############################################################################
#                                Training
##############################################################################
##############################################################################

optimizer = Adam(lr=0.0003, beta_1=0.9, beta_2=0.999, epsilon=1e-08)
model3.compile(optimizer=optimizer, loss='binary_crossentropy', metrics=['accuracy'])

# Train!
print("Comienza entrenamiento...................................................................")
history = model3.fit(train_x, train_y,
                     epochs=50, batch_size=16,
                     validation_split=0.11111,
                     shuffle=True)

# Guarda historial
np.save('acc2.npy', history.history['acc'])
np.save('val_acc2.npy', history.history['val_acc'])
np.save('loss2.npy', history.history['loss'])
np.save('val_loss2.npy', history.history['val_loss'])
model3.save('Redes/VGGCatvsDog2.h5')



