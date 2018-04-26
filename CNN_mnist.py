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

from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import log_loss, accuracy_score

from utils import imshow_grid


BATCH_SIZE = 128
NB_CLASSES = 10
EPOCHS = 20
TRAIN_SIZE = 13500

"""--------------load pickled images and labels from each domain-----------------------------
   --------------normalise images, and build domain labels---------------------------------- """
mnist = input_data.read_data_sets('MNIST_data', one_hot=True)
mnist_train = (mnist.train.images > 0).reshape(55000, 28, 28, 1).astype(np.uint8) * 255
mnist_train = np.concatenate([mnist_train, mnist_train, mnist_train], 3)
mnist_test = (mnist.test.images > 0).reshape(10000, 28, 28, 1).astype(np.uint8) * 255
mnist_test = np.concatenate([mnist_test, mnist_test, mnist_test], 3)

mnistm = pkl.load(open('mnistm_data.pkl', 'rb'))
mnistm_train = mnistm['train']
mnistm_test = mnistm['test']
mnistm_valid = mnistm['valid']

imshow_grid(mnist_train)
imshow_grid(mnistm_train)

# Compute pixel mean for normalizing data
#pixel_mean = mnist_train.mean((0, 1, 2))
pixel_mean = np.vstack([mnist_train, mnistm_train]).mean((0, 1, 2))


"""--------------construct and initialise CNN model----------------------------- """
inputs = Input(shape=(28, 28, 3))
x = inputs
#x = Dense(64, activation='relu')(inputs)
x = Conv2D(24, kernel_size=5, activation='relu')(x)
#x = PReLU()(x) # Non-linearity
x = MaxPooling2D(pool_size=(3, 3))(x)
x = Dropout(rate=0.2)(x)
x = Conv2D(36, kernel_size=5, activation='relu')(x)
x = MaxPooling2D(pool_size=(2, 2))(x)
x = Dropout(rate=0.2)(x)
x = Flatten()(x)
x = Dense(100, activation='relu')(x)
#x = Dropout(rate=0.5)(x)
x = Dense(100, activation='relu')(x)
#features = Dropout(rate=0.5)(x)

predictions = Dense(NB_CLASSES, activation='softmax')(x)

model = Model(inputs=inputs, outputs=predictions)
#adam = Adam(lr=0.001, beta_1=0.9, beta_2=0.999, epsilon=1e-8, decay=0.0, amsgrad=False)
#sgd = optimizers.SGD(lr=0.01, decay=1e-6, momentum=0.9, nesterov=True)
model.compile(optimizer='adam',#adam,
              loss='categorical_crossentropy',
              metrics=['accuracy'])

model.summary()

"""-----------construct and initialise subordinate model, for feeding the classifier------------- """
features_model = Model(inputs=model.input, outputs=model.get_layer(model.layers[7].name).output)

mnist_train = (mnist_train - pixel_mean) / 255
mnist_test = (mnist_test - pixel_mean) / 255
y_train = np.argmax(mnist.train.labels, axis=1) #class vector form needed for baseline metrics
y_test = np.argmax(mnist.test.labels, axis=1)

mnistm_train = (mnistm_train - pixel_mean) / 255
mnistm_test = (mnistm_test - pixel_mean) / 255
#these have same y labels as aboe, as they were built from the same source

#y = model.predict(mnist_train) #baseline untrained network score
#y_class = np.zeros(len(y), dtype=np.int)
#for sample, probs in enumerate(y):
#    y_class[sample] = np.where(probs == max(probs))[0][0] #get class label for max. probability
#print('Baseline - loss:',round(log_loss(y_train, y),4),'- acc:', round(accuracy_score(y_train, y_class),4))
#
#y = model.predict(mnist_test) #baseline untrained validation score
#y_class = np.zeros(len(y), dtype=np.int)
#for sample, probs in enumerate(y):
#    y_class[sample] = np.where(probs == max(probs))[0][0] #get class label for max. probability
#print('- val_loss:',round(log_loss(y_test, y),4),'- acc:', round(accuracy_score(y_test, y_class),4))

history = model.fit(mnist_train[:TRAIN_SIZE], mnist.train.labels[:TRAIN_SIZE],
                    batch_size=BATCH_SIZE, nb_epoch=EPOCHS,
                    verbose=0, validation_data=(mnist_test, mnist.test.labels))
score = model.evaluate(mnist_test, mnist.test.labels, verbose=0)

print('Test score:', score)
print('Test accuracy:', score)

"""--------------Calculate Accuracy on Training and Test Sets ---------------------------
   --------------for both Source (trained) and Target domains --------------------------"""
print('SOURCE scores')
y = model.predict(mnist_train) #baseline untrained network score
y_class = np.zeros(len(y), dtype=np.int)
for sample, probs in enumerate(y):
    y_class[sample] = np.where(probs == max(probs))[0][0] #get class label for max. probability
print('Baseline - loss:',round(log_loss(y_train, y),4),'- acc:', round(accuracy_score(y_train, y_class),4))

y = model.predict(mnist_test) #baseline untrained validation score
y_class = np.zeros(len(y), dtype=np.int)
for sample, probs in enumerate(y):
    y_class[sample] = np.where(probs == max(probs))[0][0] #get class label for max. probability
print('- test_loss:',round(log_loss(y_test, y),4),'- acc:', round(accuracy_score(y_test, y_class),4))

print('TARGET scores')
y = model.predict(mnistm_train) #baseline untrained network score
y_class = np.zeros(len(y), dtype=np.int)
for sample, probs in enumerate(y):
    y_class[sample] = np.where(probs == max(probs))[0][0] #get class label for max. probability
print('Training - loss:',round(log_loss(y_train, y),4),'- acc:', round(accuracy_score(y_train, y_class),4))

y = model.predict(mnistm_test) #baseline untrained validation score
y_class = np.zeros(len(y), dtype=np.int)
for sample, probs in enumerate(y):
    y_class[sample] = np.where(probs == max(probs))[0][0] #get class label for max. probability
print('- test_loss:',round(log_loss(y_test, y),4),'- acc:', round(accuracy_score(y_test, y_class),4))

"""--------------And repeat using the ensemble classifier ---------------------------
   -------------------(using the subordinate model) --------------------------"""
clf = RandomForestClassifier(n_estimators = 150, max_depth = 4)
X = features_model.predict(mnist_train[:TRAIN_SIZE])
clf.fit(X, y_train[:TRAIN_SIZE])
print('SOURCE scores - classifier')
X = features_model.predict(mnist_train)
print('Training -- loss:',round(log_loss(y_train, clf.predict_proba(X)), 4),
      '- acc:', round(clf.score(X, y_train), 4))

X = features_model.predict(mnist_test)
print('Test -- loss:',round(log_loss(y_test, clf.predict_proba(X)), 4),
      '- acc:', round(clf.score(X, y_test), 4))

print('TARGET scores - classifier')
X = features_model.predict(mnistm_train)
print('Training -- loss:',round(log_loss(y_train, clf.predict_proba(X)), 4),
      '- acc:', round(clf.score(X, y_train), 4))

X = features_model.predict(mnistm_test)
print('Test -- loss:',round(log_loss(y_test, clf.predict_proba(X)), 4),
      '- acc:', round(clf.score(X, y_test), 4))