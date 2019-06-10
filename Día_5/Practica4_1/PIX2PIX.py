# -*- coding: utf-8 -*-
"""
Implementation of Unpaired Image-to-Image Translation with Conditional Adversarial Networks.

Paper: https://arxiv.org/abs/1611.07004
"""

from random import shuffle
import h5py

from keras.layers import Input, Dropout, Concatenate
from keras.layers import BatchNormalization
from keras.layers.advanced_activations import LeakyReLU
from keras.layers.convolutional import UpSampling2D, Conv2D
from keras.models import Model
from keras.optimizers import Adam
import matplotlib.pyplot as plt
import numpy as np
import os


def warn(*args, **kwargs):
    pass
import warnings
warnings.warn = warn

import keras.backend as K
K.set_image_data_format('channels_last')
K.set_learning_phase(1)


class Pix2Pix:
    def __init__(self, dim=256, channels=4):

        ###############################
        #  Define Parámetros
        ###############################

        # Input shape
        self.dim = dim
        self.channels = channels
        self.img_shape = (self.dim, self.dim, self.channels)

        # Calcula las dimensiones de salida del Discriminador D (PatchGAN)
        patch = int(self.dim / 2**4)
        self.disc_patch = (patch, patch, 1)

        # Número de filtros en las primeras capas de G y D
        self.gf = 64
        self.df = 64

        # Define el optimizador
        optimizer = Adam(0.0001, 0.9)

        ###############################
        #  Inicializa las redes
        ###############################

        # Construye y compila el Discriminador
        self.discriminator = self.build_discriminator()
        self.discriminator.compile(loss='mse', optimizer=optimizer, metrics=['accuracy'])

        # Construye y compila el Generador
        self.generator = self.build_generator()
        self.generator.compile(loss='binary_crossentropy', optimizer=optimizer)

        # Define inputs para el modelo combinado
        img_target = Input(shape=(self.dim, self.dim, 1))
        img_cond = Input(shape=self.img_shape)

        # Condicionándonos en img_cond, generamos img_fake
        img_fake = self.generator(img_cond)

        # Para el modelo combinado se entrenará solo el generador
        self.discriminator.trainable = False

        # El discriminador calcula la validez de la imagen generada y la imagen real
        valid = self.discriminator([img_fake, img_cond])

        self.combined = Model([img_target, img_cond], [valid, img_fake])
        self.combined.compile(loss=['mse', 'mae'],
                              loss_weights=[1, 100],
                              optimizer=optimizer)

        ###############################
        # Carga datos
        ###############################

        self.hdf5_file = h5py.File('dataset/datasetshadow.hdf5', "r")
        # CREA VARIABLES DE ENTRENAMIENTO
        self.train_B = np.array(self.hdf5_file["train_img_x"][...])/255.  # imágenes entrenamiento
        self.train_A = np.array(self.hdf5_file["train_labels"][...])/255.  # máscara de sombras
        # CREA VARIABLES DE VALIDACION
        self.val_B = np.array(self.hdf5_file["val_img_x"][...])/255.  # imágenes validación
        self.val_A = np.array(self.hdf5_file["val_labels"][...])/255.  # máscara de sombras
        # CREA VARIABLES DE TEST
        self.test_B = np.array(self.hdf5_file["test_img_x"][...])/255.  # imágenes de test
        self.test_A = np.array(self.hdf5_file["test_labels"][...])/255.  # máscara de sombras

    def build_generator(self):
        """U-Net Generator"""

        def conv2d(layer_input, filters, f_size=4, bn=True):
            """Capas usadas en el encoder"""
            d = Conv2D(filters, kernel_size=f_size, strides=2, padding='same')(layer_input)
            d = LeakyReLU(alpha=0.2)(d)
            if bn:
                d = BatchNormalization(momentum=0.8)(d)
            return d

        def deconv2d(layer_input, skip_input, filters, f_size=4, dropout_rate=0):
            """Capas usadas en el decoder"""
            u = UpSampling2D(size=2)(layer_input)
            u = Conv2D(filters, kernel_size=f_size, strides=1, padding='same', activation='relu')(u)
            if dropout_rate:
                u = Dropout(dropout_rate)(u)
            u = BatchNormalization(momentum=0.8)(u)
            u = Concatenate()([u, skip_input])
            return u

        # Image input
        d0 = Input(shape=self.img_shape)

        # Downsampling
        d1 = conv2d(d0, self.gf, bn=False)
        d2 = conv2d(d1, self.gf*2)
        d3 = conv2d(d2, self.gf*4)
        d4 = conv2d(d3, self.gf*4)

        # Upsampling
        u1 = deconv2d(d4, d3, self.gf*4)
        u2 = deconv2d(u1, d2, self.gf*2)
        u3 = deconv2d(u2, d1, self.gf)

        u4 = UpSampling2D(size=2)(u3)
        output_img = Conv2D(1, kernel_size=3, strides=1, padding='same', activation='sigmoid')(u4)

        return Model(d0, output_img)

    def build_discriminator(self):

        def d_layer(layer_input, filters, f_size=4, bn=True):
            """Capas del Discriminador"""
            d = Conv2D(filters, kernel_size=f_size, strides=2, padding='same')(layer_input)
            d = LeakyReLU(alpha=0.2)(d)
            if bn:
                d = BatchNormalization(momentum=0.8)(d)
            return d

        img_A = Input(shape=(self.dim, self.dim, 1))
        img_B = Input(shape=self.img_shape)

        # Concatena máscara de sombra y su imagen condicionante por canales para producir el input
        combined_imgs = Concatenate(axis=-1)([img_A, img_B])

        d1 = d_layer(combined_imgs, self.df, bn=False)
        d2 = d_layer(d1, self.df*2)
        d3 = d_layer(d2, self.df*4)
        d4 = d_layer(d3, self.df*8)

        validity = Conv2D(1, kernel_size=4, strides=1, padding='same')(d4)

        return Model([img_A, img_B], validity)

    def train(self, epochs, batch_size=1):

        print("Empieza entrenamiento.........................................................")

        for epoch in range(epochs):

            print("Época " + str(epoch + 1) + " / " + str(epochs))
            
            # Reordena la data
            c = list(zip(self.train_A, self.train_B))
            shuffle(c)
            self.train_A, self.train_B = zip(*c)
            self.train_A = np.asarray(self.train_A)
            self.train_B = np.asarray(self.train_B)
            
            for batch in range(np.floor(len(self.train_A)/batch_size).astype(int)):
                print(" Batch " + str(batch) + " / " + str(np.floor(len(self.train_A)/batch_size).astype(int)), end='\r')

                ############################### 
                #  Train Discriminator
                ############################### 
    
                # Samplea máscaras y sus correspondientes imágenes condicionantes
                imgs_A = np.reshape(self.train_A[batch*batch_size:batch_size*(1+batch), :, :], (batch_size, 256, 256, 1))
                imgs_B = self.train_B[batch*batch_size:batch_size*(1+batch), :, :, :]

                fake_A = self.generator.predict(imgs_B)
    
                valid = np.ones((batch_size,) + self.disc_patch)
                fake = np.zeros((batch_size,) + self.disc_patch)
    
                # Entrena el discriminador
                d_loss_real = self.discriminator.train_on_batch([imgs_A, imgs_B], valid)
                d_loss_fake = self.discriminator.train_on_batch([fake_A, imgs_B], fake)
                dd_loss = 0.5 * np.add(d_loss_real, d_loss_fake)
    
                ############################### 
                #  Train Generator
                ###############################

                # En vez de considerar un target 0 (t = 0) para imágene fakes,
                # el generador considerará que dicho target sea 1 (t = 1)
                valid = np.ones((batch_size,) + self.disc_patch)    

                # Train the generator
                gg_loss = self.combined.train_on_batch([imgs_A, imgs_B], [valid, imgs_A])
                gg_loss = self.combined.train_on_batch([imgs_A, imgs_B], [valid, imgs_A])

            ################################
            #   Guardar imágenes generadas
            ################################

            self.save_imgs(epoch)

    def save_imgs(self, epoch):
        os.makedirs('images/', exist_ok=True)
        r, c = 3, 15
        
        # Reordena la data
        # cc = list(zip(self.test_A, self.test_B))
        # shuffle(cc)
        # self.test_A, self.test_B = zip(*cc)
        # self.test_A = np.asarray(self.test_A)
        # self.test_B = np.asarray(self.test_B)

        imgs_A = np.reshape(self.test_A[18:18+c, :, :], (c, 256, 256, 1))
        imgs_B = self.test_B[18:18+c, :, :, :]
        fake_A = self.generator.predict(imgs_B)

        fakergb = np.concatenate([fake_A, fake_A, fake_A], axis=-1)
        imgsrgb = np.concatenate([imgs_A, imgs_A, imgs_A], axis=-1)

        gen_imgs = np.concatenate([imgs_B[:, :, :, 0:3], fakergb, imgsrgb])

        titles = ['Condition', 'Generated', 'Original']
        fig, axs = plt.subplots(r, c)
        cnt = 0
        for i in range(r):
            for j in range(c):
                axs[i, j].imshow(gen_imgs[cnt])
                axs[i, j].set_title(titles[i], fontsize=4)
                axs[i, j].axis('off')
                cnt += 1
        fig.savefig("images/%d.png" % epoch, dpi = 1800)
        plt.close()


if __name__ == '__main__':
    gan = Pix2Pix(dim=256, channels=4)
    gan.train(epochs=1000, batch_size=5)
