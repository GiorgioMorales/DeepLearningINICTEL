from __future__ import print_function

from data_utils import *
from keras.models import Model
from keras.layers import Dense, Input, LSTM, Reshape, Lambda
from keras.utils import to_categorical
from keras.optimizers import Adam

##############################################################################
##############################################################################
#                                Dataset
##############################################################################
##############################################################################

X, Y, n_values, indices_values = load_music_utils()
print('shape of X:', X.shape)
print('number of training examples:', X.shape[0])
print('Tx (length of sequence):', X.shape[1])
print('total # of unique values:', n_values)
print('Shape of Y:', Y.shape)

##############################################################################
##############################################################################
#                             Build the model
##############################################################################
##############################################################################

# Número de hidden states
n_a = 64

# Se construirá la red con un for-loop, pero los bloques LSTM deben mantener
# los mismos pesos, así que se definen como variables globales
reshapor =
LSTM_cell =
densor =


def deepjazz(Tx, n_a, n_values):

    X = Input(shape=(Tx, n_values))

    # Define los hidden states iniciales del LSTM
    a0 = Input(shape=(n_a,), name='a0')
    c0 = Input(shape=(n_a,), name='c0')
    a = a0
    c = c0

    outputs = []

    for t in range(Tx):
        # Selecciona el valor del tiempo "t" del vector X
        x =
        # Aplica Reshape a x
        x =
        # Aplica un bloque LSTM
        a, _, c =
        # Aplica Dense (softmax) a los hidden states del LSTM
        out =
        # Añade out a la lista de outputs
        outputs.

    # Crea el modelo
    model = Model(inputs=[X, a0, c0], outputs=outputs)

    return model


##############################################################################
##############################################################################
#                                 Training
##############################################################################
##############################################################################

# Construye modelo y compila
model = deepjazz(Tx = 30 , n_a = 64, n_values = 78)
opt = Adam(lr=0.01, beta_1=0.9, beta_2=0.999, decay=0.01)
model.compile(optimizer=opt, loss='categorical_crossentropy', metrics=['accuracy'])

m = 60
a0 = np.zeros((m, n_a))
c0 = np.zeros((m, n_a))

# Train model
print("Empieza entrenamiento...")
model.fit([X, a0, c0], list(Y), epochs=100)

##############################################################################
##############################################################################
#                            Build inference model
##############################################################################
##############################################################################

def music_inference_model(LSTM_cell, densor, n_values=78, n_a=64, Ty=100):

    x0 = Input(shape=(1, n_values))

    a0 = Input(shape=(n_a,), name='a0')
    c0 = Input(shape=(n_a,), name='c0')
    a = a0
    c = c0
    x = x0

    outputs = []

    for t in range(Ty):
        # Ejecuta un bloque LSTM entrenado (LSTM_cell)
        a, _, c =

        # Aplica la Dense layer entrenada (densor)
        out =

        # Añade "out" a la lista "outputs"
        outputs.

        # La siguiente entrada será la versión one-hot de la salida anterior
        x =

    # Crea el modelo de generación
    inference_model = Model(inputs=[x0, a0, c0], outputs=outputs)

    return inference_model

##############################################################################
##############################################################################
#                               Genera música
##############################################################################
##############################################################################

# Build the generation model
inference_model = music_inference_model(LSTM_cell, densor, n_values = 78, n_a = 64, Ty = 50)

# Inicializa estados y el primer input
x_initializer = np.zeros((1, 1, 78))
a_initializer = np.zeros((1, n_a))
c_initializer = np.zeros((1, n_a))

# Genera una secuencia
pred = inference_model.predict([x_initializer,a_initializer,c_initializer])
# Convierte "pred" (lista) a un np.array() de indices con las mayores probabilidades
indices = np.argmax(pred, axis = -1)
# Convierte los índices vectores one-hot
results = to_categorical(indices,num_classes=78)

# Genera música
out_stream = generate_music(inference_model)
