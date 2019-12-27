from __future__ import print_function, division
import scipy

from keras.datasets import mnist
from keras_contrib.layers.normalization.instancenormalization import InstanceNormalization
from keras.layers import Input, Dense, Reshape, Flatten, Dropout, Concatenate, Multiply, Conv2DTranspose
from keras.models import Sequential, Model
from keras.optimizers import Adam
import datetime
import matplotlib.pyplot as plt
import sys
from data_loader import DataLoader
from generator import generator_network
from discriminator import discriminator_network

import numpy as np
import os

import keras.backend as K
import tensorflow as tf

# Define custom loss
# img_A : reflectance map, img_C : reflectance normal(mask)

def custom_loss(img_C):

    def loss(y_true, y_pred):
        #y_pred : fake_A, y_true : img_A

        #three_channel = K.concatenate([img_C, img_C, img_C])
        #num_pixels = np.count_nonzero(K.eval(img_C))

        num_pixels = tf.math.count_nonzero(img_C, dtype=tf.float32)
        #num_pixels = K.print_tensor(num_pixels, message='num_pixels = ')

        reshaped_C = Reshape((256,256,1))(img_C)
        masked_res = Multiply()([y_true - y_pred, reshaped_C])

        return K.sum(K.square(masked_res)) / num_pixels

    # Return a function
    return loss


class Pix2Pix():
    def __init__(self):
        # Input shape
        self.img_rows = 256
        self.img_cols = 256
        self.channels = 3
        self.img_shape = (self.img_rows, self.img_cols, self.channels)

        # Configure data loader
        self.dataset_name = 'my'
        self.data_loader = DataLoader(dataset_name=self.dataset_name,
                                      img_res=(self.img_rows, self.img_cols))

        patch_denom = 256
        patch = int(self.img_rows / patch_denom)
        self.disc_patch = (patch, patch, 1)

        # Number of filters in the first layer of G and D
        self.gf = 64
        self.df = 64

        optimizer = Adam(lr=0.00003, decay=0.00001)

        # Build and compile the discriminator
        self.discriminator = discriminator_network(self.img_shape, self.df)
        self.discriminator.compile(loss='mse',
            optimizer=optimizer,
            metrics=['accuracy'])

        #-------------------------
        # Construct Computational
        #   Graph of Generator
        #-------------------------

        # Build the generator
        self.generator = generator_network(self.img_shape, self.gf)

        # Input images and their conditioning images
        img_A = Input(shape=self.img_shape)
        img_B = Input(shape=self.img_shape)
        img_C = Input(shape=(256, 256))

        # By conditioning on B generate a fake version of A
        fake_A = self.generator(img_B)

        # For the combined model we will only train the generator
        self.discriminator.trainable = False

        # Discriminators determines validity of translated images / condition pairs
        valid = self.discriminator([fake_A, img_B])

        self.combined = Model(inputs=[img_A, img_B, img_C], outputs=[valid, fake_A, fake_A])
        self.combined.compile(loss=['mse', 'mae', custom_loss(img_C)],
                              loss_weights=[1, 80, 20],
                              optimizer=optimizer)
        self.combined.summary()

    def train(self, epochs, batch_size=1, sample_interval=50):
        target_epoch = 7

        start_time = datetime.datetime.now()

        # Adversarial loss ground truths
        valid = np.ones((batch_size,) + self.disc_patch)
        fake = np.zeros((batch_size,) + self.disc_patch)

        self.generator.load_weights('./models/half/masked06_trans_newloss/generator2_epoch{:03d}.h5'.format(target_epoch))
        self.discriminator.load_weights('./models/half/masked06_trans_newloss/discriminator_epoch{:03d}.h5'.format(target_epoch))
        self.combined.load_weights('./models/half/masked06_trans_newloss/combined_epoch{:03d}.h5'.format(target_epoch))

        for epoch in range(epochs):
            for batch_i, (imgs_A, imgs_B, imgs_C) in enumerate(self.data_loader.load_batch(batch_size)):
                # ---------------------
                #  Train Discriminator
                # ---------------------

                # Condition on B and generate a translated version
                fake_A = self.generator.predict(imgs_B)

                # Train the discriminators (original images = real / generated = Fake)
                d_loss_real = self.discriminator.train_on_batch([imgs_A, imgs_B], valid)
                d_loss_fake = self.discriminator.train_on_batch([fake_A, imgs_B], fake)
                d_loss = 0.5 * np.add(d_loss_real, d_loss_fake)

                # -----------------
                #  Train Generator
                # -----------------

                # Train the generators
                # input, labels
                g_loss = self.combined.train_on_batch([imgs_A, imgs_B, imgs_C], [valid, imgs_A, imgs_A])

                elapsed_time = datetime.datetime.now() - start_time
                # Plot the progress
                print ("[Epoch %d/%d] [Batch %d/%d] [D loss: %f, d_acc_real: %f, d_acc_fake: %f, acc: %3d%%] [G 00: %f, G 01: %f, G 02: %f] time: %s" % (epoch+1, epochs,
                                                                        batch_i, self.data_loader.n_batches,
                                                                        d_loss[0], d_loss_real[1], d_loss_fake[1], 100*d_loss[1],
                                                                        g_loss[0], g_loss[1], g_loss[2],
                                                                        elapsed_time))

                # If at save interval => save generated image samples
                if batch_i % sample_interval == 0:
                    self.sample_images(epoch+1, batch_i)
                    
            self.combined.save('./models/half/masked06_trans_newloss/combined_epoch{:03d}.h5'.format(epoch+1+target_epoch))
            self.generator.save('./models/half/masked06_trans_newloss/generator2_epoch{:03d}.h5'.format(epoch+1+target_epoch))
            self.discriminator.save('./models/half/masked06_trans_newloss/discriminator_epoch{:03d}.h5'.format(epoch+1+target_epoch))
    
    def mse_train(self, epochs, batch_size=1, sample_interval=50, generator_checkpoint = None):

        optimizer = Adam(0.0001, 0.5)
        self.generator.compile(loss=['mse'],
                              loss_weights=None,
                              optimizer=optimizer)

        start_time = datetime.datetime.now()

        self.generator.load_weights('./saved_model_mse/generator_epoch004.h5')

        # Adversarial loss ground truths
        valid = np.ones((batch_size,) + self.disc_patch)
        fake = np.zeros((batch_size,) + self.disc_patch)

        for epoch in range(epochs):
            for batch_i, (imgs_A, imgs_B, imgs_C) in enumerate(self.data_loader.load_batch(batch_size)):

                # -----------------
                #  Train Generator
                # -----------------

                # Train the generators
                #g_loss = self.combined.train_on_batch([imgs_A, imgs_B], [valid, imgs_A])
                g_loss = self.generator.train_on_batch(imgs_B, imgs_A)

                elapsed_time = datetime.datetime.now() - start_time
                # Plot the progress
                print ("[Epoch %d/%d] [Batch %d/%d] [G loss: %f] time: %s" % (epoch, epochs,
                                                                        batch_i, self.data_loader.n_batches,
                                                                        g_loss,
                                                                        elapsed_time))

                # If at save interval => save generated image samples
                if batch_i % sample_interval == 0:
                    self.sample_images(epoch, batch_i)

            self.generator.save_weights('./saved_model/generator_epoch{:03d}.h5'.format(epoch+1))

    def mae_train(self, epochs, batch_size=1, sample_interval=50, generator_checkpoint=None):

        optimizer = Adam(0.0001, 0.5)
        self.generator.compile(loss=['mae'],
                               metrics=['mse'],
                               loss_weights=None,
                               optimizer=optimizer)

        start_time = datetime.datetime.now()

        self.generator.load_weights('./models/half_classifier/masked/generator2_epoch009.h5')

        # Adversarial loss ground truths
        valid = np.ones((batch_size,) + self.disc_patch)
        fake = np.zeros((batch_size,) + self.disc_patch)

        for epoch in range(epochs):
            for batch_i, (imgs_A, imgs_B, imgs_C) in enumerate(self.data_loader.load_batch(batch_size)):

                # -----------------
                #  Train Generator
                # -----------------

                # Train the generators
                # g_loss = self.combined.train_on_batch([imgs_A, imgs_B], [valid, imgs_A])
                g_loss = self.generator.train_on_batch(imgs_B, imgs_A)

                elapsed_time = datetime.datetime.now() - start_time
                # Plot the progress
                print("[Epoch %d/%d] [Batch %d/%d] [G mae: %f] [G mse: %f] time: %s" % (epoch, epochs,
                                                                             batch_i, self.data_loader.n_batches,
                                                                             g_loss[0], g_loss[1],
                                                                             elapsed_time))

                # If at save interval => save generated image samples
                if batch_i % sample_interval == 0:
                    self.sample_images(epoch, batch_i)

                if batch_i % 100 == 0 and batch_i is not 0:
                    self.generator.save('./models/half_classifier/masked/generator_finetuned100.h5')


    def sample_images(self, epoch, batch_i):
        os.makedirs('images/%s' % self.dataset_name, exist_ok=True)
        r, c = 3, 3

        imgs_A, imgs_B, imgs_C = self.data_loader.load_data(batch_size=3, is_testing=True)
        fake_A = self.generator.predict(imgs_B)

        gen_imgs = np.concatenate([imgs_B, fake_A, imgs_A])

        # Rescale images 0 - 1
        gen_imgs = 0.5 * gen_imgs + 0.5

        titles = ['Condition', 'Generated', 'Original']
        fig, axs = plt.subplots(r, c)
        cnt = 0
        for i in range(r):
            for j in range(c):
                axs[i,j].imshow(gen_imgs[cnt])
                axs[i, j].set_title(titles[i])
                axs[i,j].axis('off')
                cnt += 1
        fig.savefig("images/%s/%d_%d.png" % (self.dataset_name, epoch, batch_i))
        plt.close()


    def _sample_images(self, epoch, batch_i):
        os.makedirs('images/%s' % 'check', exist_ok=True)
        r, c = 3, 3

        imgs_A, imgs_B, imgs_C = self.data_loader.load_data(batch_size=3, is_testing=True)
        fake_A = self.generator.predict(imgs_B)

        gen_imgs = np.concatenate([imgs_B, fake_A, imgs_A])

        # Rescale images 0 - 1
        gen_imgs = 0.5 * gen_imgs + 0.5

        titles = ['Condition', 'Generated', 'Original']
        fig, axs = plt.subplots(r, c)
        cnt = 0
        for i in range(r):
            for j in range(c):
                axs[i,j].imshow(gen_imgs[cnt])
                axs[i, j].set_title(titles[i])
                axs[i,j].axis('off')
                cnt += 1
        fig.savefig("images/%s/%d_%d.png" % (self.dataset_name, epoch, batch_i))
        plt.close()


if __name__ == '__main__':
    gan = Pix2Pix()

    #gan.mse_train(epochs=5, batch_size=15, sample_interval=200)
    #gan.mae_train(epochs=1, batch_size=16, sample_interval=1)
    gan.train(epochs=6, batch_size=8, sample_interval=150)
