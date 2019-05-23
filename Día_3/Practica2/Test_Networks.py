"""
Verificación de entrenamiento de los clasificadores de perros y gatos

Giorgio Morales
"""
from keras.applications.vgg16 import VGG16
from keras.models import Sequential
from keras.layers import Dense, Flatten
from keras.models import load_model
import matplotlib.pyplot as plt
import numpy as np
import h5py

##############################################################################
##############################################################################
#                                Dibuja curvas
##############################################################################
##############################################################################

acc = np.load('Resultados/Alex_acc.npy')
val_acc = np.load('Resultados/Alex_val_acc.npy')
loss = np.load('Resultados/Alex_loss.npy')
val_loss = np.load('Resultados/Alex_val_loss.npy')
acc2 = np.load('Resultados/VGG_acc.npy')
val_acc2 = np.load('Resultados/VGG_val_acc.npy')
loss2 = np.load('Resultados/VGG_loss.npy')
val_loss2 = np.load('Resultados/VGG_val_loss.npy')

# Print loss and accuracy history
plt.figure()
plt.plot(acc)
plt.plot(val_acc)
plt.plot(acc2)
plt.plot(val_acc2)
plt.legend(['trainAlex', 'validationAlex', 'trainVGG', 'validationVGG'], loc='lower right')
plt.title('Model Accuracy')
plt.ylabel('accuracy')
plt.xlabel('epoch')
plt.show()
plt.figure()
plt.plot(loss)
plt.plot(val_loss)
plt.plot(loss2)
plt.plot(val_loss2)
plt.legend(['trainAlex', 'validationAlex', 'trainVGG', 'validationVGG'], loc='upper right')
plt.title('Model Loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.show()

##############################################################################
##############################################################################
#                             Read Data 1
##############################################################################
##############################################################################

# LEE ARCHIVO HDF5
hdf5_file = h5py.File('dataset/datasetdogvscat.hdf5', "r")

# CREA VARIABLES DE TEST
test_x = np.array(hdf5_file["test_img"][...])

test_x = test_x/255.

##############################################################################
##############################################################################
#                     Clasifica sobre el test set (AlexNet)
##############################################################################
##############################################################################

# Load model
model = load_model('Redes/AlexCatvsDog.h5')

# Predict
test_y = model.predict(test_x)
model = None

# DRAW RESULTS
fig = plt.figure()
for i in range(0, 100):
    a = fig.add_subplot(10, 10, i+1)
    if test_y[i]>=0.5:
        clase = "Cat"
    else:
        clase = "Dog"
    a.set_title(clase, fontsize=10, va='top')
    plt.imshow(test_x[i, :, :, :])
    plt.axis('off')
plt.savefig('Alex_results.png', dpi=1200)
##############################################################################
##############################################################################
#                             Read Data 2
##############################################################################
##############################################################################

# LEE ARCHIVO HDF5
hdf5_file = h5py.File('dataset/datasetdogvscat2.hdf5', "r")

# CREA VARIABLES DE TEST
test_x = np.array(hdf5_file["test_img"][...])

test_x = test_x/255.

##############################################################################
##############################################################################
#                     Clasifica sobre el test set (AlexNet)
##############################################################################
##############################################################################
vgg = VGG16(weights='imagenet', include_top=False, input_shape=(224, 224, 3))
vgg.summary()

# Crear nuevo modelo reemplazando la última capa fully-connected de 1000 clases
# (predictions) por una binaria

model2 = Sequential()
for i in range(0, 19):
    model2.add(vgg.layers[i])
model2.add(Flatten())
model2.add(Dense(1024, activation='relu', name='fc1'))
model2.add(Dense(256, activation='relu', name='fc2'))
model2.add(Dense(1, activation='sigmoid', name='Prediction'))

# Load model
model2.load_weights('Redes/VGGCatvsDogPesos.h5')

# Predict
test_y2 = model2.predict(test_x)

fig = plt.figure()
for i in range(0, 100):
    a = fig.add_subplot(10, 10, i+1)
    if test_y2[i] >= 0.5:
        clase = "Cat"
    else:
        clase = "Dog"
    a.set_title(clase, fontsize=10, va='top')
    plt.imshow(test_x[i, :, :, :])
    plt.axis('off')
plt.savefig('VGG_results.png', dpi=1200)
