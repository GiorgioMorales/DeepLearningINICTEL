import numpy as np

def softmax(x):
    # escribir la función softmax

    return

def sigmoid(x):
    # escribir la función sigmoid
    return

##############################################################################
##############################################################################
#                                 Celda RNN
##############################################################################
##############################################################################

def rnn_cell_forward(xt, a_prev, parameters):

    # Extrae las variables del diccionario "parameters"
    Wax = parameters["Wax"]
    Waa = parameters["Waa"]
    Wya = parameters["Wya"]
    ba = parameters["ba"]
    by = parameters["by"]

    # Calcula el valor de cada "activation"
    a_next =
    # Calcula la salida de la red de la celda actual
    yt_pred =

    # Guarda los valores necesarios para el entrenamiento en cache
    cache = (a_next, a_prev, xt, parameters)

    return a_next, yt_pred, cache

# Verificar que los resultados sean correctos
np.random.seed(1)
xt = np.random.randn(3,10)
a_prev = np.random.randn(5,10)
Waa = np.random.randn(5,5)
Wax = np.random.randn(5,3)
Wya = np.random.randn(2,5)
ba = np.random.randn(5,1)
by = np.random.randn(2,1)
parameters = {"Waa": Waa, "Wax": Wax, "Wya": Wya, "ba": ba, "by": by}

a_next, yt_pred, cache = rnn_cell_forward(xt, a_prev, parameters)
print("a_next[4] = ", a_next[4])
print("a_next.shape = ", a_next.shape)
print("yt_pred[1] =", yt_pred[1])
print("yt_pred.shape = ", yt_pred.shape, '\n\n')


# Valores Esperados
# a_next[4]:	[ 0.59584544 0.18141802 0.61311866 0.99808218 0.85016201 0.99980978 -0.18887155 0.99815551 0.6531151 0.82872037]
# a_next.shape:	(5, 10)
# yt[1]:	[ 0.9888161 0.01682021 0.21140899 0.36817467 0.98988387 0.88945212 0.36920224 0.9966312 0.9982559 0.17746526]
# yt.shape:	(2, 10)

##############################################################################
##############################################################################
#                             Forward propagation
##############################################################################
##############################################################################

def rnn_forward(x, a0, parameters):

    # Inicializa "caches", una lista que contendrá todos los "cache"
    caches = []

    # Obtiene las dimensiones generales a partir de la entrada x y los parámetros
    n_x, m, T_x = x.shape
    n_y, n_a = parameters["Wya"].shape

    # inicializa "a" y "y" con zeros
    a = np.zeros([n_a, m, T_x])
    y_pred = np.zeros([n_y, m, T_x])

    # Inicializa a_next
    a_next = a0

    # hace un loop sobre todos los tiempos
    for t in range(T_x):
        # Actualiza el siguiente hidden state, calcula la prediccióny obtiene el cache
        a_next, yt_pred, cache =
        # Guarda el valor del nuevo "next" estado oculto en "a"

        # Guarda los valores de la predicción en y

        # Añade el cache actual a la lista de caches


    caches = (caches, x)

    return a, y_pred, caches

# Verificar que los resultados sean correctos
np.random.seed(1)
x = np.random.randn(3,10,4)
a0 = np.random.randn(5,10)
Waa = np.random.randn(5,5)
Wax = np.random.randn(5,3)
Wya = np.random.randn(2,5)
ba = np.random.randn(5,1)
by = np.random.randn(2,1)
parameters = {"Waa": Waa, "Wax": Wax, "Wya": Wya, "ba": ba, "by": by}

a, y_pred, caches = rnn_forward(x, a0, parameters)
print("a[4][1] = ", a[4][1])
print("a.shape = ", a.shape)
print("y_pred[1][3] =", y_pred[1][3])
print("y_pred.shape = ", y_pred.shape)
print("caches[1][1][3] =", caches[1][1][3])
print("len(caches) = ", len(caches), '\n\n')

# Valores Esperados
# a[4][1]:	[-0.99999375 0.77911235 -0.99861469 -0.99833267]
# a.shape:	(5, 10, 4)
# y[1][3]:	[ 0.79560373 0.86224861 0.11118257 0.81515947]
# y.shape:	(2, 10, 4)
# cache[1][1][3]:	[-1.1425182 -0.34934272 -0.20889423 0.58662319]
# len(cache):	2

##############################################################################
##############################################################################
#                                LSTM cell
##############################################################################
##############################################################################

def lstm_cell_forward(xt, a_prev, c_prev, parameters):

    # Extrae las variables del diccionario "parameters"
    Wf = parameters["Wf"]
    bf = parameters["bf"]
    Wi = parameters["Wi"]
    bi = parameters["bi"]
    Wc = parameters["Wc"]
    bc = parameters["bc"]
    Wo = parameters["Wo"]
    bo = parameters["bo"]
    Wy = parameters["Wy"]
    by = parameters["by"]

    # Obtiene las dimensiones generales a partir de la entrada xt y Wy
    n_x, m = xt.shape
    n_y, n_a = Wy.shape

    # Concatena a_prev y xt
    concat = np.zeros((n_x + n_a, m))
    concat[: n_a, :] = a_prev
    concat[n_a:, :] = xt

    # Calcula los valores de ft, it, cct, c_next, ot, a_next
    ft =
    it =
    cct =
    c_next =
    ot =
    a_next =

    # Calcula la predicción de la celda LSTM
    yt_pred =

    # Guarda los valores necesarios para el entrenamiento en cache
    cache = (a_next, c_next, a_prev, c_prev, ft, it, cct, ot, xt, parameters)

    return a_next, c_next, yt_pred, cache

# Verificar que los resultados sean correctos
np.random.seed(1)
xt = np.random.randn(3,10)
a_prev = np.random.randn(5,10)
c_prev = np.random.randn(5,10)
Wf = np.random.randn(5, 5+3)
bf = np.random.randn(5,1)
Wi = np.random.randn(5, 5+3)
bi = np.random.randn(5,1)
Wo = np.random.randn(5, 5+3)
bo = np.random.randn(5,1)
Wc = np.random.randn(5, 5+3)
bc = np.random.randn(5,1)
Wy = np.random.randn(2,5)
by = np.random.randn(2,1)

parameters = {"Wf": Wf, "Wi": Wi, "Wo": Wo, "Wc": Wc, "Wy": Wy, "bf": bf, "bi": bi, "bo": bo, "bc": bc, "by": by}

a_next, c_next, yt, cache = lstm_cell_forward(xt, a_prev, c_prev, parameters)
print("a_next[4] = ", a_next[4])
print("a_next.shape = ", c_next.shape)
print("c_next[2] = ", c_next[2])
print("c_next.shape = ", c_next.shape)
print("yt[1] =", yt[1])
print("yt.shape = ", yt.shape)
print("cache[1][3] =", cache[1][3])
print("len(cache) = ", len(cache))

# Valores esperados
# a_next[4]:	[-0.66408471 0.0036921 0.02088357 0.22834167 -0.85575339 0.00138482 0.76566531 0.34631421 -0.00215674 0.43827275]
# a_next.shape:	(5, 10)
# c_next[2]:	[ 0.63267805 1.00570849 0.35504474 0.20690913 -1.64566718 0.11832942 0.76449811 -0.0981561 -0.74348425 -0.26810932]
# c_next.shape:	(5, 10)
# yt[1]:	[ 0.79913913 0.15986619 0.22412122 0.15606108 0.97057211 0.31146381 0.00943007 0.12666353 0.39380172 0.07828381]
# yt.shape:	(2, 10)
# cache[1][3]:	[-0.16263996 1.03729328 0.72938082 -0.54101719 0.02752074 -0.30821874 0.07651101 -1.03752894 1.41219977 -0.37647422]
# len(cache):	10
