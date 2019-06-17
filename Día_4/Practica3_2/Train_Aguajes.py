# -*- coding: utf-8 -*-
"""
Created on Tue Aug 28 16:13:09 2018

@author: Giorgio Morales

Entrenamiento de segmentación de aguajes con Deeplab3+G atrous Depthwise separable convolution
"""

from random import shuffle
import glob
from keras.layers import Input, Dropout, Concatenate
from keras.layers import BatchNormalization, Activation
from keras.layers.convolutional import UpSampling2D, Conv2D, DepthwiseConv2D, SeparableConv2D
from keras.models import Model
from keras.optimizers import Adam
from keras.layers import Add

from datagenerator import DataGenerator

import keras.backend as K
K.set_image_data_format('channels_last')
K.set_learning_phase(1)

import pickle
"""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""
""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""
Modelo
""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""
"""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""


def build_generator(img_shape=(512, 512, 3)):

    def relu6(x):
        return K.relu(x, max_value=6)

    def _inverted_res_block(inputs, expansion, stride, pointwise_filters, block_id, skip_connection, rate=1):

        in_channels = inputs._keras_shape[-1]

        x = inputs
        prefix = 'expanded_conv_{}_'.format(block_id)

        if block_id:
            # Expand

            x = Conv2D(expansion * in_channels, kernel_size=1, padding='same',
                       use_bias=False, activation=None,
                       name=prefix + 'expand')(x)
            x = BatchNormalization(epsilon=1e-3, momentum=0.999,
                                   name=prefix + 'expand_BN')(x)
            x = Activation(relu6, name=prefix + 'expand_relu')(x)
        else:
            prefix = 'expanded_conv_'
        # Depthwise
        x = DepthwiseConv2D(kernel_size=3, strides=stride, activation=None,
                            use_bias=False, padding='same', dilation_rate=(rate, rate),
                            name=prefix + 'depthwise')(x)
        x = BatchNormalization(epsilon=1e-3, momentum=0.999,
                               name=prefix + 'depthwise_BN')(x)

        x = Activation(relu6, name=prefix + 'depthwise_relu')(x)

        # Project
        x = Conv2D(pointwise_filters,
                   kernel_size=1, padding='same', use_bias=False, activation=None,
                   name=prefix + 'project')(x)
        x = BatchNormalization(epsilon=1e-3, momentum=0.999,
                               name=prefix + 'project_BN')(x)

        if skip_connection:
            return Add(name=prefix + 'add')([inputs, x])

        return x

    # Image input
    d0 = Input(shape=img_shape)

    """""""""""""""""""""""""""
    """"""""""""""""""""""""""
    Feature extraction
    """"""""""""""""""""""""""
    """""""""""""""""""""""""""

    first_block_filters = 32
    x = Conv2D(first_block_filters,
               kernel_size=3,
               strides=(2, 2), padding='same',
               use_bias=False, name='Conv')(d0)
    x = BatchNormalization(epsilon=1e-3, momentum=0.999, name='Conv_BN')(x)
    x = Activation(relu6, name='Conv_Relu6')(x)

    x = _inverted_res_block(x, pointwise_filters=16, stride=1,
                            expansion=1, block_id=0, skip_connection=False)

    skip1 = x

    x = _inverted_res_block(x, pointwise_filters=24, stride=2,
                            expansion=6, block_id=1, skip_connection=False)
    x = _inverted_res_block(x, pointwise_filters=24, stride=1,
                            expansion=6, block_id=2, skip_connection=True)

    x = _inverted_res_block(x, pointwise_filters=32, stride=2,
                            expansion=6, block_id=3, skip_connection=False)
    x = _inverted_res_block(x, pointwise_filters=32, stride=1,
                            expansion=6, block_id=4, skip_connection=True)
    x = _inverted_res_block(x, pointwise_filters=32, stride=1,
                            expansion=6, block_id=5, skip_connection=True)

    """""""""""""""""""""""""""
    """"""""""""""""""""""""""
    ASPP
    """"""""""""""""""""""""""
    """""""""""""""""""""""""""
    atrous_rates = (6, 12, 18)

    # simple 1x1
    b0 = Conv2D(256, (1, 1), padding='same', use_bias=False, name='aspp0')(x)
    b0 = BatchNormalization(name='aspp0_BN', epsilon=1e-5)(b0)
    b0 = Activation('relu', name='aspp0_activation')(b0)

    # rate = 6 (12)
    b1 = SeparableConv2D(256, kernel_size=3, strides=(1, 1), padding='same',
                         dilation_rate=atrous_rates[0], name='app1')(x)
    # rate = 12 (24)
    b2 = SeparableConv2D(256, kernel_size=3, strides=(1, 1), padding='same',
                         dilation_rate=atrous_rates[1], name='app2')(x)
    # rate = 18 (36)
    b3 = SeparableConv2D(256, kernel_size=3, strides=(1, 1), padding='same',
                         dilation_rate=atrous_rates[2], name='app3')(x)

    # concatenate ASPP branches and project
    x = Concatenate()([b0, b1, b2, b3])
    b0 = None
    b1 = None
    b2 = None
    b3 = None

    x = Conv2D(256, (1, 1), padding='same',
               use_bias=False, name='concat_projection')(x)
    x = BatchNormalization(name='concat_projection_BN', epsilon=1e-5)(x)
    x = Activation('relu')(x)
    x = Dropout(0.1)(x)

    """""""""""""""""""""""""""
    """"""""""""""""""""""""""
    Decoder
    """"""""""""""""""""""""""
    """""""""""""""""""""""""""

    x = UpSampling2D(size=4)(x)
    dec_skip1 = Conv2D(48, (1, 1), padding='same',
                       use_bias=False, name='feature_projection0')(skip1)
    dec_skip1 = BatchNormalization(
        name='feature_projection0_BN', epsilon=1e-5)(dec_skip1)
    dec_skip1 = Activation('relu')(dec_skip1)
    x = Concatenate()([x, dec_skip1])

    dec_skip1 = None

    x = SeparableConv2D(256, kernel_size=3, strides=(1, 1), padding='same',
                        dilation_rate=1, name='decoder_conv0')(x)
    x = SeparableConv2D(256, kernel_size=3, strides=(1, 1), padding='same',
                        dilation_rate=1, name='decoder_conv1')(x)

    x = Conv2D(1, (1, 1), padding='same', name="last_layer", activation='sigmoid')(x)
    x = UpSampling2D(size=2)(x)

    return Model(d0, x)


"""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""
""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""
Listas
""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""
"""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""

# Obtiene listas de imágenes de entrenamiento, valdiación y test
shuffle_data = True
orig_path = 'dataset//*.jpg'

# Obtiene una lista de las direcciones de las imágenes y sus máscaras
addri = sorted(glob.glob(orig_path))
addrm = list(addri)
for i in range(len(addri)):
    addrm[i] = addri[i][0:-4] + "_mask.tif"

# Reordena aleatoriamente las direcciones por pares
if shuffle_data:
    c = list(zip(addri, addrm))
    shuffle(c)
    addri, addrm = zip(*c)

# Divide 90% train, 5% validation, and 5% test
train_origin = addri[0:int(0.9 * len(addri))]
train_labels = addrm[0:int(0.9 * len(addri))]
val_origin = addri[int(0.9 * len(addri)):int(0.95 * len(addri))]
val_labels = addrm[int(0.9 * len(addri)):int(0.95 * len(addri))]
test_origin = addri[int(0.95 * len(addri)):]
test_labels = addrm[int(0.95 * len(addri)):]

# Parametros para la generación de data
path = 'dataset'
n_channels = 3
dim = 512
params = {'dim': (dim, dim),
          'batch_size': 64,
          'n_channels': n_channels,
          'path': path,
          'shuffle': True}

# Crea diccionarios
data_dict = {}
data_dict["train"] = train_origin
data_dict["validation"] = val_origin
data_dict["test"] = test_origin

# Generadores
training_generator = DataGenerator(data_dict['train'], **params)
validation_generator = DataGenerator(data_dict['validation'], **params)
test_generator = DataGenerator(data_dict['test'], **params)

# Guarda generadores
# with open('Generators/Train', 'wb') as f:
#     pickle.dump(training_generator, f)
#
# with open('Generators/Validation', 'wb') as f:
#     pickle.dump(validation_generator, f)
#
# with open('Generators/Test', 'wb') as f:
#     pickle.dump(test_generator, f)

# Si los generadores ya han sido creados con anterioridad, sólo se cargan
# with open('Generators/Train', 'rb') as f:
#     training_generator = pickle.load(f)
# with open('Generators/Validation', 'rb') as f:
#     validation_generator = pickle.load(f)
# with open('Generators/Test', 'rb') as f:
#     test_generator = pickle.load(f)

"""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""
""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""
Entrenamiento
""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""
"""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""
# Carga modelo
print("Cargando modelo...")
model = build_generator(img_shape=(dim, dim, n_channels))
model.summary()

# Compila modelol
optimizer = Adam(lr=0.003, beta_1=0.9, beta_2=0.999, epsilon=1e-08)
model.compile(optimizer=optimizer, loss='binary_crossentropy', metrics=['acc'])

# Train model
print("Empieza entrenamiento...")
history = model.fit_generator(generator=training_generator,
                              validation_data=validation_generator,
                              use_multiprocessing=False,
                              shuffle=True,
                              epochs=100,
                              )
