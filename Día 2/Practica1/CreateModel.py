"""
Crear modelo en Keras

Giorgio Morales - INICTEL-UNI
"""

# Importar capas de convolución, max pooling, input, BN y fully connected
from keras.layers import Input, Conv2D, MaxPooling2D, BatchNormalization, Flatten, Dense

# Importa la clase "Sequential" y "Model"
from keras.models import Model, Sequential

# Importa el optimizador
from keras.optimizers import Adam

# Opcionales
from keras.initializers import glorot_uniform
from keras.callbacks import ModelCheckpoint

# Opcional
import keras.backend as K
K.set_image_data_format('channels_last') # Setea el format0 (alto, ancho, canales)
K.set_learning_phase(1) # Determina si está en fase de entrenamiento (1) o test (0)

##############################################################################
##############################################################################
#                          Crear modelo secuencial
##############################################################################
##############################################################################

model = Sequential()
# Agrega convolución 1:
#   Tamaño de entrada = 258 x 258 x 3
#   Filtros = 32
#   Tamaño de filtro = 3 x 3
#   Stride = 1 x 1
#   Padding = Valid
#   Activation = Relu
#   Tamaño de salida esperado = 256 x 256 x 32  ( (N-f+1) x (N-f+1) )
model.add(Conv2D(32, (3, 3), strides=(1, 1), padding='valid', activation='relu', input_shape=(258, 258, 3)))

model.add(BatchNormalization())
#   Tamaño de salida esperado = 128 x 128 x 32
model.add(MaxPooling2D(pool_size=(2, 2)))


# Agrega convolución 2:
#   Filtros = 64
#   Tamaño de filtro = 3 x 3
#   Stride = 2 x 2
#   Padding = Same
#   Activation = Relu
#   Tamaño de salida esperado = 64 x 64 x 64
model.add(Conv2D(64, (3, 3), strides=(2, 2), padding='same', activation='sigmoid'))
model.add(BatchNormalization())

model.summary()

##############################################################################
##############################################################################
#                          Crear modelo normal
##############################################################################
##############################################################################

input_shape = (258, 258, 3)
# Crea Input del modelo
x_input = Input(input_shape)

# Agrega convolución 1:
#   Tamaño de entrada = 258 x 258 x 3
#   Filtros = 32
#   Tamaño de filtro = 3 x 3
#   Stride = 1 x 1
#   Padding = Valid
#   Activation = Relu
#   Tamaño de salida esperado = 256 x 256 x 32  ( (N-f+1) x (N-f+1) )
x = Conv2D(32, (3, 3), strides=(1, 1), padding='valid', activation='relu', input_shape=(258, 258, 3))(x_input)

x = BatchNormalization()(x)
#   Tamaño de salida esperado = 128 x 128 x 32
x = MaxPooling2D(pool_size=(2, 2))(x)


# Agrega convolución 2:
#   Filtros = 64
#   Tamaño de filtro = 3 x 3
#   Stride = 2 x 2
#   Padding = Same
#   Activation = Relu
#   Tamaño de salida esperado = 64 x 64 x 64
x = Conv2D(64, (3, 3), strides=(2, 2), padding='same', activation='sigmoid')(x)
x = BatchNormalization()(x)

# Crea el modelo
model2 = Model(inputs=x_input, outputs=x, name='CNNsimple')

model2.summary()
