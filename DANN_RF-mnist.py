# -*- coding: utf-8 -*-
"""
Created on Mon Apr  2 11:12:05 2018

@author: Fish
"""

from __future__ import print_function
import numpy as np
import pickle as pkl

#from keras.datasets import mnist
from keras.layers import Input, MaxPooling2D, Conv2D, Activation, Dense, Flatten, Dropout#, PReLU
from keras.optimizers import Adam, SGD
from keras.models import Model
from keras.utils import np_utils

from tensorflow.examples.tutorials.mnist import input_data

from utils import *


batch_size = 128
nb_classes = 10
nb_epoch = 1


mnist = input_data.read_data_sets('MNIST_data', one_hot=True)
mnist_train = (mnist.train.images > 0).reshape(55000, 28, 28, 1).astype(np.uint8) * 255
mnist_train = np.concatenate([mnist_train, mnist_train, mnist_train], 3)
mnist_test = (mnist.test.images > 0).reshape(10000, 28, 28, 1).astype(np.uint8) * 255
mnist_test = np.concatenate([mnist_test, mnist_test, mnist_test], 3)

#mnistm = pkl.load(open('mnistm_data.pkl', 'rb'))
#mnistm_train = mnistm['train']
#mnistm_test = mnistm['test']
#mnistm_valid = mnistm['valid']

#imshow_grid(mnist_train)
#imshow_grid(mnistm_train)

# Compute pixel mean for normalizing data
pixel_mean = mnist_train.mean((0, 1, 2))
#pixel_mean = np.vstack([mnist_train, mnistm_train]).mean((0, 1, 2))

# Create a mixed dataset for domain classifier testing
#num_test = 500
#combined_test_imgs = np.vstack([mnist_test[:num_test], mnistm_test[:num_test]])
#combined_test_labels = np.vstack([mnist.test.labels[:num_test], mnist.test.labels[:num_test]])
#combined_test_domain = np.vstack([np.tile([1., 0.], [num_test, 1]),
#        np.tile([0., 1.], [num_test, 1])])

inputs = Input(shape=(28, 28, 3))
x = inputs
#x = Dense(64, activation='relu')(inputs)
x = Conv2D(32, kernel_size=5, activation='relu')(x)
#x = PReLU()(x) # Non-linearity
x = MaxPooling2D(pool_size=(2, 2))(x)
x = Dropout(rate=0.2)(x)
x = Conv2D(48, kernel_size=5, activation='relu')(x)
x = MaxPooling2D(pool_size=(2, 2))(x)
x = Dropout(rate=0.2)(x)
x = Flatten()(x)
x = Dense(100, activation='relu')(x)
x = Dropout(rate=0.5)(x)
x = Dense(100, activation='relu')(x)
features = Dropout(rate=0.5)(x)

predictions = Dense(nb_classes, activation='softmax')(features)

model = Model(inputs=inputs, outputs=predictions)
#adam = Adam(lr=0.001, beta_1=0.9, beta_2=0.999, epsilon=1e-8, decay=0.0, amsgrad=False)
#sgd = optimizers.SGD(lr=0.01, decay=1e-6, momentum=0.9, nesterov=True)
model.compile(optimizer='adam',#adam,
              loss='categorical_crossentropy',
              metrics=['accuracy'])

model.summary()

mnist_train = (mnist_train - pixel_mean) / 255
mnist_test = (mnist_test - pixel_mean) / 255
history = model.fit(mnist_train, mnist.train.labels,
                    batch_size=batch_size, nb_epoch=nb_epoch,
                    verbose=1, validation_data=(mnist_test, mnist.test.labels))
score = model.evaluate(mnist_test, mnist.test.labels, verbose=0)

print('Test score:', score)
print('Test accuracy:', score)






