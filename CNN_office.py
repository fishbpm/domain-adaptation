# -*- coding: utf-8 -*-
"""
Created on Mon Apr  2 11:12:05 2018

@author: Fish
"""

from __future__ import print_function
import numpy as np
import pickle as pkl

from PIL import Image

#from keras.datasets import mnist
from keras.layers import Input, MaxPooling2D, Conv2D, Activation, Dense, Flatten, Dropout#, PReLU
from keras.optimizers import Adam, SGD
from keras.models import Model
from keras.utils import np_utils

from tensorflow.examples.tutorials.mnist import input_data

from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import log_loss, accuracy_score

from utils import imshow_grid


NB_CLASSES = 31
IMAGE_SIZE = 64
EPOCHS = 34
BATCH_SIZE = 128#int(train_size/12)
#TRAIN_SIZE = 55000

origin_dom = 'amazon'
target_dom = 'webcam'

"""--------------load pickled images and labels from each domain-----------------------------
   --------------normalise images, and build domain labels---------------------------------- """
source = pkl.load(open(origin_dom+'_train'+str(IMAGE_SIZE)+'.pkl', 'rb'))
source_train = source['samples']#[:train_size]
ys_train = source['labels']
source = pkl.load(open(origin_dom+'_test'+str(IMAGE_SIZE)+'.pkl', 'rb'))
source_test = source['samples']#[:train_size]
ys_test = source['labels']

target = pkl.load(open(target_dom+'_train'+str(IMAGE_SIZE)+'.pkl', 'rb'))
target_train = target['samples']#[:train_size]
yt_train = target['labels']
target = pkl.load(open(target_dom+'_test'+str(IMAGE_SIZE)+'.pkl', 'rb'))
target_test = target['samples']#[:train_size]
yt_test = target['labels']

imshow_grid(source_train)
imshow_grid(target_train)

train_size = len(ys_train)
test_size = len(ys_test)

ys_train = ys_train
yt_train = yt_train
ys_test = ys_test
yt_test = yt_test

ys_train_hot = np.zeros((train_size, NB_CLASSES), np.uint8)
ys_train_hot[np.arange(train_size), ys_train] = 1
ys_test_hot = np.zeros((test_size, NB_CLASSES), np.uint8)
ys_test_hot[np.arange(test_size), ys_test] = 1

# Compute pixel mean for normalizing data
#pixel_mean = source_train.mean((0, 1, 2))
pixel_mean = np.vstack([source_train, target_train,
                        target_train, target_test]).mean((0, 1, 2))
source_train = (source_train - pixel_mean) / 255
target_train = (target_train - pixel_mean) / 255
source_test = (source_test - pixel_mean) / 255
target_test = (target_test - pixel_mean) / 255

batch_size = int(len(source_train)/round(len(source_train)/BATCH_SIZE, 0))

"""--------------construct and initialise CNN model----------------------------- """
inputs = Input(shape=(IMAGE_SIZE, IMAGE_SIZE, 3))
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

#y = model.predict(source_train) #baseline untrained network score
#y_class = np.zeros(len(y), dtype=np.int)
#for sample, probs in enumerate(y):
#    y_class[sample] = np.where(probs == max(probs))[0][0] #get class label for max. probability
#print('Baseline - loss:',round(log_loss(y_train, y),4),'- acc:', round(accuracy_score(y_train, y_class),4))
#
#y = model.predict(source_test) #baseline untrained validation score
#y_class = np.zeros(len(y), dtype=np.int)
#for sample, probs in enumerate(y):
#    y_class[sample] = np.where(probs == max(probs))[0][0] #get class label for max. probability
#print('- val_loss:',round(log_loss(y_test, y),4),'- acc:', round(accuracy_score(y_test, y_class),4))

history = model.fit(source_train, ys_train_hot,
                    batch_size=batch_size, epochs=EPOCHS,
                    verbose=0, validation_data=(source_test, ys_test_hot))
score = model.evaluate(source_test, ys_test_hot, verbose=0)

print('Test score:', score)
print('Test accuracy:', score)

"""--------------Calculate Accuracy on Training and Test Sets ---------------------------
   --------------for both Source (trained) and Target domains --------------------------"""
print('SOURCE scores')
y = model.predict(source_train) #baseline untrained network score
y_class = np.zeros(len(y), dtype=np.int)
for sample, probs in enumerate(y):
    y_class[sample] = np.where(probs == max(probs))[0][0] #get class label for max. probability
print('Baseline - loss:',round(log_loss(ys_train, y),4),'- acc:', round(accuracy_score(ys_train, y_class),4))

y = model.predict(source_test) #baseline untrained validation score
y_class = np.zeros(len(y), dtype=np.int)
for sample, probs in enumerate(y):
    y_class[sample] = np.where(probs == max(probs))[0][0] #get class label for max. probability
print('- test_loss:',round(log_loss(ys_test, y),4),'- acc:', round(accuracy_score(ys_test, y_class),4))

print('TARGET scores')
y = model.predict(target_train) #baseline untrained network score
y_class = np.zeros(len(y), dtype=np.int)
for sample, probs in enumerate(y):
    y_class[sample] = np.where(probs == max(probs))[0][0] #get class label for max. probability
print('Training - loss:',round(log_loss(yt_train, y),4),'- acc:', round(accuracy_score(yt_train, y_class),4))

y = model.predict(target_test) #baseline untrained validation score
y_class = np.zeros(len(y), dtype=np.int)
for sample, probs in enumerate(y):
    y_class[sample] = np.where(probs == max(probs))[0][0] #get class label for max. probability
print('- test_loss:',round(log_loss(yt_test, y),4),'- acc:', round(accuracy_score(yt_test, y_class),4))

"""--------------And repeat using the ensemble classifier ---------------------------
   --------------------------(using the subordinate model) --------------------------"""
clf = RandomForestClassifier(n_estimators = 150, max_depth = 7)
X = features_model.predict(source_train)
clf.fit(X, ys_train)
print('SOURCE scores - classifier')
X = features_model.predict(source_train)
print('Training -- loss:',round(log_loss(ys_train, clf.predict_proba(X)), 4),
      '- acc:', round(clf.score(X, ys_train), 4))

X = features_model.predict(source_test)
print('Test -- loss:',round(log_loss(ys_test, clf.predict_proba(X)), 4),
      '- acc:', round(clf.score(X, ys_test), 4))

print('TARGET scores - classifier')
X = features_model.predict(target_train)
print('Training -- loss:',round(log_loss(yt_train, clf.predict_proba(X)), 4),
      '- acc:', round(clf.score(X, yt_train), 4))

X = features_model.predict(target_test)
print('Test -- loss:',round(log_loss(yt_test, clf.predict_proba(X)), 4),
      '- acc:', round(clf.score(X, yt_test), 4))