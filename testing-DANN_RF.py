# -*- coding: utf-8 -*-
"""
Created on Mon Apr  2 11:12:05 2018

@author: Fish
"""

from __future__ import print_function
import numpy as np
import pickle as pkl

#import keras.backend as K
import keras_helper as K
#from keras.datasets import mnist
from keras.layers import Input, MaxPooling2D, Conv2D, Activation, Dense, Flatten#, Dropout, PReLU
from keras.optimizers import Adam, SGD
from keras.models import Model
from keras.utils import np_utils
#from keras import losses

from tensorflow.examples.tutorials.mnist import input_data

from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import log_loss

from utils import *
#from snes import SNES
from sequencialsnes import SSNES

output_file = open('snes_output.txt', 'w')

batch_pos = 0
batch_size = 31#127
nb_classes = 10
depth = 100
nb_epoch = 20

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

mnist_train = (mnist_train - pixel_mean) / 255
mnist_test = (mnist_test - pixel_mean) / 255
y_train = np.argmax(mnist.train.labels, axis=1) #class vector form needed for classifier
y_test = np.argmax(mnist.test.labels, axis=1)

inputs = Input(shape=(28, 28, 3))
x = Conv2D(32, kernel_size=5, activation='relu')(inputs)
#x = PReLU()(x) # Non-linearity
x = MaxPooling2D(pool_size=(2, 2))(x)
x = Conv2D(48, kernel_size=5, activation='relu')(x)
x = MaxPooling2D(pool_size=(2, 2))(x)
x = Flatten()(x)
x = Dense(depth, activation='relu')(x)
features = Dense(depth, activation='relu')(x)

#predictions = Dense(nb_classes, activation='softmax')(x)

model = Model(inputs=inputs, outputs=features)#predictions)
#adam = Adam(lr=0.001, beta_1=0.9, beta_2=0.999, epsilon=1e-8, decay=0.0, amsgrad=False)
#sgd = optimizers.SGD(lr=0.01, decay=1e-6, momentum=0.9, nesterov=True)
model.compile(optimizer='adam',#adam,
              loss='categorical_crossentropy',
              metrics=['accuracy'])

model.summary()

weights = K.get_trainable_weights(model)
dim = len(weights)
batches = int(len(mnist_train) / batch_size) #number iterations for each epoch
generation = 12 #number batches used for each snes generation
batches -= batches%generation
#train_size = batches * batch_size

snes = SSNES(np.zeros(dim), 1, generation)
clf = RandomForestClassifier(n_estimators = 150, max_depth = 4)
fitnesses = np.zeros(generation)
asked = snes.asked

X = model.predict(mnist_train) #baseline untrained network score
clf.fit(X, y_train)
print('Baseline /',nb_epoch,'- loss:',round(log_loss(y_train, clf.predict_proba(X)),4),
      '- acc:', round(clf.score(X, y_train),4))
X = model.predict(mnist_test)
print('- val_loss:',round(log_loss(y_test, clf.predict_proba(X)),4),'- val_acc:', round(clf.score(X, y_test),4))

for epoch in range(nb_epoch):
    batch = 0
    batch_start = 0
    gen = 0
    while batch < batches:#2*generation:#
        K.set_weights(model, asked[gen])
        X = model.predict(mnist_train[batch_start:(batch_start + batch_size)])
        clf.fit(X, y_train[batch_start:(batch_start + batch_size)])
        if gen == 0:
            #print('Weights:', K.get_trainable_weights(model)[:2])
            print('Gen',1+batch/generation,'/',batches/generation,'- loss:',
                  round(log_loss(y_train[batch_start:(batch_start + batch_size)], clf.predict_proba(X)),4),
                  '- acc:', round(clf.score(X, y_train[batch_start:(batch_start + batch_size)]),4), file=output_file)
        fitness = []
        for sample, probs in enumerate(clf.predict_proba(X)):
            #fitness.append(probs[y_train[batch_start + sample]]) #get probability for this class label
            fitness.append(probs[np.where(clf.classes_ == y_train[batch_start + sample])[0][0]])
        fitnesses[gen] = np.mean(fitness)
        #fitnesses[gen] = -log_loss(y_train[batch_start:(batch_start + batch_size)], clf.predict_proba(X))
        gen += 1
        if gen == generation:
            gen = 0
            snes.fit(fitnesses, range(generation))
            asked = snes.asked
            #print('asked:',asked[0][:2])#, file=output_file)
            fitnesses = np.zeros(generation)
        batch_start += batch_size
        batch += 1
    X = model.predict(mnist_train) #full set now needed to generate full random forest classifier
    clf.fit(X, y_train)
    print('Epoch',epoch+1,'/',nb_epoch,'- loss:',round(log_loss(y_train, clf.predict_proba(X)),4),
          '- acc:', round(clf.score(X, y_train),4), end='')
    X = model.predict(mnist_test)
    #clf.fit(X, y_test)
    print('- val_loss:',round(log_loss(y_test, clf.predict_proba(X)),4),'- val_acc:', round(clf.score(X, y_test),4))
#X = model.predict(mnist_train) #full set now needed to generate full random forest classifier
#clf.fit(X, y_train) 
#X = model.predict(mnist_test)
#score = clf.score(X, y_test)
#
#print('Test score:', score)

#history = model.fit(mnist_train, mnist.train.labels,
#                    batch_size=batch_size, nb_epoch=nb_epoch,
#                    verbose=1, validation_data=(mnist_test, mnist.test.labels))
#score = model.evaluate(mnist_test, mnist.test.labels, verbose=0)

#print('Test score:', score)
#print('Test accuracy:', score)






