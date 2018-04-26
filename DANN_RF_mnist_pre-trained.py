# -*- coding: utf-8 -*-
"""
Created on Mon Apr  2 11:12:05 2018

@author: Fish
"""

from __future__ import print_function
import numpy as np
import pandas as pd
import pickle as pkl
import time

#import keras.backend as K
import keras_helper as K
#from keras_helper_new import NNWeightHelper
#from keras.datasets import mnist
from keras.layers import Input, MaxPooling2D, Conv2D, Activation, Dense, Flatten, Dropout#, PReLU
from keras.optimizers import Adam, SGD
from keras.models import Model
from keras.utils import np_utils
#from keras import losses

from PIL import Image

import matplotlib.pyplot as plt
import seaborn as sns

from tensorflow.examples.tutorials.mnist import input_data

from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import log_loss, accuracy_score

from utils import imshow_grid
#from snes import SNES
from sequencialsnes import SSNES
from snes import SNES

output_file = open('snes_output.txt', 'w')

NB_CLASSES = 10
DEPTH = 100
DOM_WEIGHT = 5
FREEZE = 4 #how many layers to freeze
TRAIN_SIZE = 27500#55000#number of samples we will use for training
GENERATION = 12#36 #number batches (population size) used for each snes GENERATION
BATCH_SIZE = 50#25#8#42#127#
EPOCHS = 40
SAMPLES = 1450000 #target number of samples to be processed for full run

"""--------------load pickled images and labels from each domain-----------------------------
   --------------normalise images, and build domain labels---------------------------------- """
mnist = input_data.read_data_sets('MNIST_data', one_hot=True)
mnist_train = (mnist.train.images > 0)[:TRAIN_SIZE].reshape(27500, 28, 28, 1).astype(np.uint8) * 255
#mnist_train = np.concatenate([mnist_train, mnist_train, mnist_train], 3)
mnist_test = (mnist.test.images > 0).reshape(10000, 28, 28, 1).astype(np.uint8) * 255
#mnist_test = np.concatenate([mnist_test, mnist_test, mnist_test], 3)

test_size = len(mnist_test)

mnistm = pkl.load(open('mnistm_data.pkl', 'rb'))
mnistm_train = mnistm['train'][:TRAIN_SIZE]
mnistm_test = mnistm['test']

def decolourise(images):
    grey_images = np.zeros([len(images), 28, 28, 1], np.uint8)
    for sample, image in enumerate(images):
        image = Image.fromarray(image, 'RGB')
        image = np.asarray(image.convert('L'), dtype="uint8")
        grey_images[sample] = image.reshape(28, 28, 1)
    return grey_images

mnistm_train = decolourise(mnistm_train)
mnistm_test = decolourise(mnistm_test)
    
#imshow_grid(mnist_train)
#imshow_grid(mnistm_train)

# Compute pixel mean for normalizing data
#pixel_mean = mnist_train.mean((0, 1, 2))
pixel_mean = np.vstack([mnist_train, mnistm_train]).mean((0, 1, 2))

# Create a mixed dataset for domain classifier training
#num_train = 500
#train_imgs = np.vstack([mnist_train[:num_train], mnistm_train[:num_train]])
#train_labels = np.vstack([mnist.test.labels[:num_train], mnist.test.labels[:num_train]])
#train_domain = np.vstack([np.tile([1., 0.], [num_train, 1]), np.tile([0., 1.], [num_train, 1])])
yd_train = np.concatenate([np.tile(0, len(mnist_train)), np.tile(1, len(mnist_train))])
yd_test = np.concatenate([np.tile(0, len(mnist_test)), np.tile(1, len(mnist_test))])

mnist_train = (mnist_train - pixel_mean) / 255
mnist_test = (mnist_test - pixel_mean) / 255
y_train = np.argmax(mnist.train.labels[:TRAIN_SIZE], axis=1) #class vector form needed for classifier
y_test = np.argmax(mnist.test.labels, axis=1)

mnistm_train = (mnistm_train - pixel_mean) / 255
mnistm_test = (mnistm_test - pixel_mean) / 255
#these have same y labels as aboe, as they were built from the same source

"""--------------construct and initialise CNN model----------------------------- """
inputs = Input(shape=(28, 28, 1))#3))
x = Conv2D(24, kernel_size=5, activation='relu')(inputs)
#x = PReLU()(x) # Non-linearity
x = MaxPooling2D(pool_size=(3, 3))(x)
#x = Dropout(rate=0.2)(x)
x = Conv2D(36, kernel_size=5, activation='relu')(x)
x = MaxPooling2D(pool_size=(2, 2))(x)
#x = Dropout(rate=0.2)(x)
x = Flatten()(x)
x = Dense(DEPTH, activation='relu')(x)
features = Dense(DEPTH, activation='relu')(x)

predictions = Dense(NB_CLASSES, activation='softmax')(features)

full_model = Model(inputs=inputs, outputs=predictions)#predictions)
#adam = Adam(lr=0.001, beta_1=0.9, beta_2=0.999, epsilon=1e-8, decay=0.0, amsgrad=False)
#sgd = optimizers.SGD(lr=0.01, decay=1e-6, momentum=0.9, nesterov=True)
full_model.compile(optimizer='adam',#adam,
              loss='categorical_crossentropy',
              metrics=['accuracy'])

"""-----------construct and initialise subordinate model, for adaptation purposes------------- """
model = Model(inputs=full_model.input, outputs=full_model.get_layer(full_model.layers[7].name).output)

model.summary()

full_model.fit(mnist_train[:TRAIN_SIZE], mnist.train.labels[:TRAIN_SIZE],
                    batch_size=128, epochs=15,
                    verbose=1, validation_data=(mnist_test, mnist.test.labels))
score = full_model.evaluate(mnist_test, mnist.test.labels, verbose=0)

print('Test score:', score)
print('Test accuracy:', score)

"""--------------initialise classifiers ------------------------------------ """
clf = RandomForestClassifier(n_estimators = 50, max_depth = 7)
clf_dom = RandomForestClassifier(n_estimators = 50, max_depth = 4)

"""--------------and calculate initial starting accuracy for randomised CNN---------------------- """
X = model.predict(np.concatenate((mnist_train, mnistm_train))) #baseline untrained network score
clf.fit(X[:TRAIN_SIZE], y_train)
#print('Baseline scores - TRAIN')
print('Baseline Label classifier\n',
      'Source -- loss:',round(log_loss(y_train, clf.predict_proba(X[:TRAIN_SIZE])), 4),
      '- acc:', round(clf.score(X[:TRAIN_SIZE], y_train), 4), '\n',
      'Target -- loss:',round(log_loss(y_train, clf.predict_proba(X[TRAIN_SIZE:])), 4),
      '- acc:', round(clf.score(X[TRAIN_SIZE:], y_train), 4))

clf_dom.fit(X, yd_train)
print('Baseline Domain classifier\n',
      '-------- loss:',round(log_loss(yd_train, clf_dom.predict_proba(X)), 4),
      '- acc:', round(clf_dom.score(X, yd_train), 4))

y = full_model.predict(np.concatenate((mnist_train, mnistm_train))) #baseline untrained network score
y_class = np.zeros(len(y), dtype=np.int)
for sample, probs in enumerate(y):
    y_class[sample] = np.where(probs == max(probs))[0][0] #get class label for max. probability
print('CNN Source - loss:',round(log_loss(y_train, y[:TRAIN_SIZE]),4),
      '- acc:', round(accuracy_score(y_train, y_class[:TRAIN_SIZE]),4),
      'CNN Target - loss:',round(log_loss(y_train, y[TRAIN_SIZE:]),4),
      '- acc:', round(accuracy_score(y_train, y_class[TRAIN_SIZE:]),4))

X = model.predict(np.concatenate((mnist_test, mnistm_test)))
print('label TEST loss - Source:',round(log_loss(y_test, clf.predict_proba(X[:test_size])), 4),
      '- acc:', round(clf.score(X[:test_size], y_test), 4), '\n',
      '- Target:',round(log_loss(y_test, clf.predict_proba(X[test_size:])), 4),
      '- acc:', round(clf.score(X[test_size:], y_test), 4))

print('domain TEST loss:',round(log_loss(yd_test, clf_dom.predict_proba(X)), 4),
      '- acc:', round(clf_dom.score(X, yd_test), 4))

y = full_model.predict(np.concatenate((mnist_test, mnistm_test))) #baseline untrained network score
y_class = np.zeros(len(y), dtype=np.int)
for sample, probs in enumerate(y):
    y_class[sample] = np.where(probs == max(probs))[0][0] #get class label for max. probability
print('CNN Source - loss:',round(log_loss(y_test, y[:test_size]),4),
      '- acc:', round(accuracy_score(y_test, y_class[:test_size]),4),
      'CNN Target - loss:',round(log_loss(y_test, y[test_size:]),4),
      '- acc:', round(accuracy_score(y_test, y_class[test_size:]),4))

"""--------------initialise all parameters for adaptation cycle---------------------- """
#nnw = NNWeightHelper(model)
#weights = nnw.get_weights()
weights = K.get_trainable_weights(model)
dim = len(weights)
batches = int(TRAIN_SIZE / BATCH_SIZE) #number iterations for each epoch
batches -= batches%GENERATION
steps = 2*SAMPLES/GENERATION**2/BATCH_SIZE #target generations for SNES

snes = SNES(weights, 2, GENERATION)
#snes = SSNES(weights, 2, GENERATION)#np.zeros(dim), 1, GENERATION)
fitnesses = np.zeros(GENERATION)
losses = [[],[],[],[]]
asked = snes.ask()
#asked = snes.asked
K.set_weights(model, asked[0], FREEZE) #randomise the un-frozen layers for baseline predictions

"""----------------------commence adaptation---------------------------- """
t0 = time.time()
step = 0
genloss = []
timeloss = []
for epoch in range(EPOCHS):
    batch = 0
    gen_start = 0
    train_end = int(BATCH_SIZE*GENERATION/2)
    fitsize = train_end #number of samples used for each fitness check (domain classifier uses double)
    yd_fit = np.concatenate([np.tile(0, fitsize), np.tile(1, fitsize)])
    test_end = BATCH_SIZE*GENERATION
    gen = 0
    while batch < batches:#2*GENERATION:#
        #nnw.set_weights(asked[gen])
        K.set_weights(model, asked[gen], FREEZE)
        X = model.predict(np.concatenate((mnist_train[gen_start:train_end], mnistm_train[gen_start:train_end])))
        clf.fit(X[:fitsize], y_train[gen_start:train_end])
        clf_dom.fit(X, yd_fit)
        X = model.predict(np.concatenate((mnist_train[train_end:test_end], mnistm_train[train_end:test_end])))
#        fitness = []
#        for sample, probs in enumerate(clf.predict_proba(X)):
#            #fitness.append(probs[y_train[train_end + sample]]) #get probability for this class label
#            fitness.append(probs[np.where(clf.classes_ == y_train[train_end + sample])[0][0]])
#        fitnesses[gen] = np.mean(fitness)
        losses[0].append(-log_loss(y_train[train_end:test_end], clf.predict_proba(X[:fitsize])))
        #losses[0].append(accuracy_score(clf.predict(X[:fitsize]), y_train[train_end:test_end]) - 1)
        timeloss.append([float(step),'Label',losses[0][gen], gen])
        #the bigger the label loss, the more it will impair (subtract) from the fitness
        losses[1].append(log_loss(yd_fit, clf_dom.predict_proba(X)) - 1.5)
        #losses[1].append(-accuracy_score(clf_dom.predict(X), yd_fit))
        timeloss.append([float(step),'Domain',losses[1][gen], gen])
        #the bigger the domain loss, the more it will contribute (add) to the fitness
        losses[2].append(losses[0][gen] + losses[1][gen])
        timeloss.append([float(step),'Net Adapt',losses[2][gen], gen])
        losses[3].append(-log_loss(y_train[train_end:test_end], clf.predict_proba(X[fitsize:])))
        timeloss.append([float(step),'Target',losses[3][gen], gen])
        adapt = 1#2 / (1 + np.exp(-10 * step / steps)) - 1
        #current adaptation factor (increases as cycles progress) 
        fitnesses[gen] = losses[1][gen]#losses[0][gen] + DOM_WEIGHT*losses[1][gen]
        gen += 1
        """-------------------refresh the SNES Gen Cycle at each 12th batch---------------------- """
        if gen == GENERATION:
            genloss.append([max(losses[0]), max(losses[1]), max(losses[2]), max(losses[3])])
            #print('Weights:', K.get_trainable_weights(model)[:2])
#            print('Epoch',epoch+1,'- Gen',1+batch/GENERATION,'/',batches/GENERATION,'- label loss:',
#                  max(losses[0]),'- acc:', round(accuracy_score(clf.predict(X[:fitsize]), y_train[train_end:test_end]),4),'\n',
#                  '--- domain loss:', max(losses[1]),'- acc:', round(accuracy_score(clf_dom.predict(X), yd_fit),4), '\n',
#                  '--- NET loss: at adapt',adapt, '-',np.amax(fitnesses),'--', time.time() - t0,"secs", file=output_file)
            print('Epoch',epoch+1,'- Gen',(1+batch)/GENERATION,'/',batches/GENERATION,
                  '--- label loss:', max(losses[0]),'--- domain loss:', max(losses[1]), '\n',
                  '--- NET loss: at adapt',adapt, '-',np.amax(fitnesses),'--', time.time() - t0,"secs", file=output_file)
            gen = 0
            step += 1
            gen_start += BATCH_SIZE*GENERATION
            train_end = int(gen_start + BATCH_SIZE*GENERATION/2)
            test_end = gen_start + BATCH_SIZE*GENERATION
            snes.tell(asked, fitnesses)
            #snes.fit(fitnesses, range(GENERATION))
            asked = snes.ask()
            #asked = snes.asked
            #print('asked:',asked[0][:2])#, file=output_file)
            fitnesses = np.zeros(GENERATION)
            losses = [[],[],[],[]]
        batch += 1

    """----------------track current model accuracies for the full domain----------------------
       ------------this is performed at the end of every epoch-------------------"""
    #nnw.set_weights(snes.center)
    K.set_weights(model, snes.center, FREEZE)
    X = model.predict(np.concatenate((mnist_train, mnistm_train))) #baseline untrained network score
    clf.fit(X[:TRAIN_SIZE], y_train)
    #print('Baseline scores - TRAIN')
    print('Epoch',epoch+1,'/',EPOCHS,'- label loss -',
          '- Source:',round(log_loss(y_train, clf.predict_proba(X[:TRAIN_SIZE])), 4),
          '- acc:', round(clf.score(X[:TRAIN_SIZE], y_train), 4), '\n',
          '- Target:',round(log_loss(y_train, clf.predict_proba(X[TRAIN_SIZE:])), 4),
          '- acc:', round(clf.score(X[TRAIN_SIZE:], y_train), 4), file=output_file)
    
    clf_dom.fit(X, yd_train)
    print('- domain loss:',round(log_loss(yd_train, clf_dom.predict_proba(X)), 4),
          '- acc:', round(clf_dom.score(X, yd_train), 4), file=output_file)

    y = full_model.predict(np.concatenate((mnist_train, mnistm_train))) #baseline untrained network score
    y_class = np.zeros(len(y), dtype=np.int)
    for sample, probs in enumerate(y):
        y_class[sample] = np.where(probs == max(probs))[0][0] #get class label for max. probability
    print('CNN Source - loss:',round(log_loss(y_train, y[:TRAIN_SIZE]),4),
          '- acc:', round(accuracy_score(y_train, y_class[:TRAIN_SIZE]),4),
          'CNN Target - loss:',round(log_loss(y_train, y[TRAIN_SIZE:]),4),
          '- acc:', round(accuracy_score(y_train, y_class[TRAIN_SIZE:]),4))
    
    X = model.predict(np.concatenate((mnist_test, mnistm_test)))
    print('- label TEST loss - Source:',round(log_loss(y_test, clf.predict_proba(X[:test_size])), 4),
          '- acc:', round(clf.score(X[:test_size], y_test), 4), '\n',
          '- Target:',round(log_loss(y_test, clf.predict_proba(X[test_size:])), 4),
          '- acc:', round(clf.score(X[test_size:], y_test), 4), file=output_file)

    print('- domain TEST loss:',round(log_loss(yd_test, clf_dom.predict_proba(X)), 4),
          '- acc:', round(clf_dom.score(X, yd_test), 4), file=output_file)
    
    y = full_model.predict(np.concatenate((mnist_test, mnistm_test))) #baseline untrained network score
    y_class = np.zeros(len(y), dtype=np.int)
    for sample, probs in enumerate(y):
        y_class[sample] = np.where(probs == max(probs))[0][0] #get class label for max. probability
    print('CNN Source - loss:',round(log_loss(y_test, y[:test_size]),4),
          '- acc:', round(accuracy_score(y_test, y_class[:test_size]),4),
          'CNN Target - loss:',round(log_loss(y_test, y[test_size:]),4),
          '- acc:', round(accuracy_score(y_test, y_class[test_size:]),4))

"""---------------------- plot the results---------------------- """   
sns.set_style('darkgrid')
dfloss = pd.DataFrame(genloss, columns=['Label', 'Domain', 'Net Adapt', 'Target'])
dfloss = dfloss.stack().reset_index()
dfloss.columns = ['GENERATION','Classifier','Loss']
plt.figure()#(figsize=(12,8))
ax = sns.pointplot(x='GENERATION', y='Loss', hue='Classifier', data=dfloss, join=False)
for points in ax.collections:
    size = points.get_sizes().item()
    new_sizes = [size / 10 for name in ax.get_yticklabels()]
    points.set_sizes(new_sizes)
ax.grid(b=True, which='major', color='#d3d3d3', linewidth=1.0)
ax.grid(b=True, which='minor', color='#d3d3d3', linewidth=0.5)
plt.show(ax)

dftimeloss = pd.DataFrame(timeloss, columns=['GENERATION', 'Classifier', 'Loss', 'sample'])
plt.figure()
tx = sns.tsplot(time='GENERATION', value='Loss', unit='sample', condition='Classifier', data=dftimeloss)
plt.show(tx)

#X = model.predict(mnist_train) #full set now needed to generate full random forest classifier
#clf.fit(X, y_train) 
#X = model.predict(mnist_test)
#score = clf.score(X, y_test)
#
#print('Test score:', score)

#history = model.fit(mnist_train, mnist.train.labels,
#                    BATCH_SIZE=BATCH_SIZE, num_epochs=EPOCHS,
#                    verbose=1, validation_data=(mnist_test, mnist.test.labels))
#score = model.evaluate(mnist_test, mnist.test.labels, verbose=0)

#print('Test score:', score)
#print('Test accuracy:', score)

output_file.close()




