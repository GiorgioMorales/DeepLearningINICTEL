"""
Prueba la clase Imagenerator
"""

import matplotlib.pyplot as plt
from keras.preprocessing.image import ImageDataGenerator

##############################################################################
##############################################################################
#                      Setea parámetros del generador
##############################################################################
##############################################################################

datagen = ImageDataGenerator(rescale=1./255,
                             shear_range=0.2,
                             rotation_range=90,
                             zoom_range=0.2,
                             horizontal_flip=True,
                             validation_split=0.1111)

##############################################################################
##############################################################################
#              Crea generadores de entrenamiento y validación
##############################################################################
##############################################################################

img_width  = 256    # Ancho deseado
img_height = 256    # Alto deseado
batch_size = 9
train_data_dir = 'dataset//train'   # Se deben crear las subcarpetas Dog y Cat

train_generator = datagen.flow_from_directory(
    train_data_dir,
    target_size=(img_width, img_height),
    batch_size=batch_size,
    class_mode='binary',
    subset='training')

validation_generator = datagen.flow_from_directory(
    train_data_dir,
    target_size=(img_width, img_height),
    batch_size=batch_size,
    class_mode='binary',
    subset='validation')

##############################################################################
##############################################################################
#                                   Dibujar
##############################################################################
##############################################################################

for X_batch, y_batch in train_generator:

    for i in range(0, 9):
        plt.subplot(3, 3, i+1)
        plt.imshow(X_batch[i])
    plt.show()
    break
