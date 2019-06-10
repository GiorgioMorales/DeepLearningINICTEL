"""
@author: Giorgio Morales

Testea funcionamiento de la Red Entrenada
"""
import matplotlib.pyplot as plt
import numpy as np
import cv2
import pickle
from keras.models import load_model
import keras.backend as K
K.set_image_data_format('channels_last')
K.set_learning_phase(0)


def relu6(x):
    return K.relu(x, max_value=6)


def iou_coef(y_true, y_pred, smooth=1):
    """
    IoU = (|X & Y|)/ (|X or Y|)
    """
    intersection = K.sum(K.abs(y_true * y_pred), axis=-1)
    union = K.sum(y_true, -1) + K.sum(y_pred, -1) - intersection

    return (intersection + smooth) / (union + smooth)


def iou_coef_metric(y_true, y_pred):
    return iou_coef(y_true, y_pred)

##############################################################################
##############################################################################
#                            Carga Generadores
##############################################################################
##############################################################################

with open('Generators/Train', 'rb') as f:
    training_generator = pickle.load(f)
with open('Generators/Validation', 'rb') as f:
    validation_generator = pickle.load(f)
with open('Generators/Test', 'rb') as f:
    test_generator = pickle.load(f)

##############################################################################
##############################################################################
#                            Carga Red Entrenada
##############################################################################
##############################################################################

# model = load_model('Redes/DeeplabG.h5') # Causará error porque no conoce "Relu6". Comentar

model = load_model('Redes/DeeplabG.h5', custom_objects={'relu6': relu6, 'iou_coef_metric': iou_coef_metric})

##############################################################################
##############################################################################
#                   Prueba la red en imágenes del test set
##############################################################################
##############################################################################
from random import randint
vec = np.zeros(10)
vec = [randint(0, 100) for i in range(10)]

fig, axs = plt.subplots(3, 10)
for cnt, n in enumerate(vec):

    # Obtiene dirección de imagen con índice n
    addr = test_generator.list_IDs[n]

    # Lee la imagen con OpenCV
    img = cv2.imread(addr)
    b, g, r = cv2.split(img)
    x = cv2.merge([r, g, b])
    x = np.reshape(x[0:512, 0:512, :], (1, 512, 512, 3))

    # Obtiene dirección de la máscara
    IDm = addr[0:-4] + "_mask.tif"

    # Lee máscara
    y = cv2.imread(IDm, 0)[0:512, 0:512]

    # Predice salida
    y_test = np.reshape(model.predict(x), (512, 512))

    # plt
    titles = ['Original', 'Segmentation', 'Ground truth']
    axs[1,cnt].imshow(y_test)
    axs[1,cnt].set_title(titles[1])
    axs[1,cnt].axis('off')
    axs[0,cnt].imshow(x[0,:,:,:])
    axs[0,cnt].set_title(titles[0])
    axs[0,cnt].axis('off')
    axs[2,cnt].imshow(y)
    axs[2,cnt].set_title(titles[2])
    axs[2,cnt].axis('off')

##############################################################################
##############################################################################
#                                Dibuja curvas
##############################################################################
##############################################################################

acc = np.load('Resultados/DeeplabG_acc.npy')
val_acc = np.load('Resultados/DeeplabG_valacc.npy')
loss = np.load('Resultados/DeeplabG_loss.npy')
val_loss = np.load('Resultados/DeeplabG_valloss.npy')

# Print loss and accuracy history
plt.figure()
plt.plot(acc)
plt.plot(val_acc)
plt.legend(['trainDeeplabG', 'validationDeeplabG'], loc='lower right')
plt.title('Model Accuracy')
plt.ylabel('accuracy')
plt.xlabel('epoch')
plt.show()
plt.figure()
plt.plot(loss)
plt.plot(val_loss)
plt.legend(['trainDeeplabG', 'validationDeeplabG'], loc='upper right')
plt.title('Model Loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.show()

