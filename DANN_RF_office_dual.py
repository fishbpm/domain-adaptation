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
from keras_helper_new import NNWeightHelper
#from keras.datasets import mnist
from keras.layers import Input, MaxPooling2D, Conv2D, Activation, Dense, Flatten, Dropout#, PReLU
from keras.optimizers import Adam, SGD
from keras.models import Model
from keras.utils import np_utils
#from keras import losses

import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import log_loss, accuracy_score

from utils import imshow_grid
#from snes import SNES
from sequencialsnes import SSNES
from snes import SNES

output_file = open('snes_output.txt', 'w')

NB_CLASSES = 31
DEPTH = 100
#train_size = 27500#55000#number of samples we will use for training

origin_dom = 'amazon'
target_dom = 'webcam'

source = pkl.load(open(origin_dom+'_train.pkl', 'rb'))
source_train = source['samples']#[:domtrain_size]
ys_train = source['labels']
source = pkl.load(open(origin_dom+'_test.pkl', 'rb'))
source_test = source['samples']#[:domtrain_size]
ys_test = source['labels']

target = pkl.load(open(target_dom+'_train.pkl', 'rb'))
target_train = target['samples']#[:domtrain_size]
yt_train = target['labels']
target = pkl.load(open(target_dom+'_test.pkl', 'rb'))
target_test = target['samples']#[:domtrain_size]
yt_test = target['labels']

imshow_grid(source_train)
imshow_grid(target_train)

train_size = len(ys_train)
test_size = len(ys_test)
#adjusted sizes for domain classificiation only - provides a balanced set [50/50]
domtrain_size = min(len(ys_train), len(yt_train))
domtest_size = min(len(ys_test), len(yt_test))

# Compute pixel mean for normalizing data
#pixel_mean = source_train.mean((0, 1, 2))
pixel_mean = np.vstack([source_train, target_train,
                        target_train, target_test]).mean((0, 1, 2))
source_train = (source_train - pixel_mean) / 255
target_train = (target_train - pixel_mean) / 255
source_test = (source_test - pixel_mean) / 255
target_test = (target_test - pixel_mean) / 255

yd_train = np.concatenate([np.tile(0, domtrain_size), np.tile(1, domtrain_size)])
yd_test = np.concatenate([np.tile(0, domtest_size), np.tile(1, domtest_size)])

inputs = Input(shape=(32, 32, 3))
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

#predictions = Dense(NB_CLASSES, activation='softmax')(x)

model = Model(inputs=inputs, outputs=features)#predictions)
#adam = Adam(lr=0.001, beta_1=0.9, beta_2=0.999, epsilon=1e-8, decay=0.0, amsgrad=False)
#sgd = optimizers.SGD(lr=0.01, decay=1e-6, momentum=0.9, nesterov=True)
model.compile(optimizer='adam',#adam,
              loss='categorical_crossentropy',
              metrics=['accuracy'])

model.summary()

clf = RandomForestClassifier(n_estimators = 50, max_depth = 5)
clf_dom = RandomForestClassifier(n_estimators = 50, max_depth = 4)

X = model.predict(np.concatenate((source_train, target_train))) #baseline untrained network score
clf.fit(X[:train_size], ys_train)
#print('Baseline scores - TRAIN')
print('Baseline Label classifier\n',
      'Source -- loss:',round(log_loss(ys_train, clf.predict_proba(X[:train_size])), 4),
      '- acc:', round(clf.score(X[:train_size], ys_train), 4), '\n',
      'Target -- loss:',round(log_loss(yt_train, clf.predict_proba(X[train_size:])), 4),
      '- acc:', round(clf.score(X[train_size:], yt_train), 4))

X = np.concatenate((X[:domtrain_size], X[-domtrain_size:]))
clf_dom.fit(X, yd_train)
print('Baseline Domain classifier\n',
      '-------- loss:',round(log_loss(yd_train, clf_dom.predict_proba(X)), 4),
      '- acc:', round(clf_dom.score(X, yd_train), 4))

X = model.predict(np.concatenate((source_test, target_test)))
print('label TEST loss - Source:',round(log_loss(ys_test, clf.predict_proba(X[:test_size])), 4),
      '- acc:', round(clf.score(X[:test_size], ys_test), 4), '\n',
      '- Target:',round(log_loss(yt_test, clf.predict_proba(X[test_size:])), 4),
      '- acc:', round(clf.score(X[test_size:], yt_test), 4))

X = np.concatenate((X[:domtest_size], X[-domtest_size:]))
print('domain TEST loss:',round(log_loss(yd_test, clf_dom.predict_proba(X)), 4),
      '- acc:', round(clf_dom.score(X, yd_test), 4))

#as we have so few office samples available, the whole set is utilised for adaptation purposes
source_adapt = np.concatenate((source_train, source_test))
target_adapt = np.concatenate((target_train, target_test))
ys_adapt = np.concatenate((ys_train, ys_test))
yt_adapt = np.concatenate((yt_train, yt_test))

nnw = NNWeightHelper(model)
weights = nnw.get_weights()
#weights = K.get_trainable_weights(model)
#dim = len(weights)
GENERATION = 12#36 #number batches (population size) used for each snes GENERATION
BATCH_SIZE = 26#50#25#8#42#127#
EPOCHS = 20 #baseline #epochs, as per MNIST data
SAMPLES = 1450000 #target number of samples to be processed to full adaptation
SAMPLES_PER_EPOCH = 13500 #baseline #samples per epoch, as per MNIST data
steps = 2*SAMPLES/GENERATION**2/BATCH_SIZE #target generations for SNES to reach full adaptation
batches = int(len(source_adapt)/BATCH_SIZE) #number iterations for each epoch
batches -= batches%GENERATION
epochs = int(EPOCHS*SAMPLES_PER_EPOCH/BATCH_SIZE/batches)
ratio = len(target_adapt)/len(source_adapt)

snes = SNES(weights, 2, GENERATION)
#snes = SSNES(weights, 2, GENERATION)#np.zeros(dim), 1, GENERATION)
fitnesses = np.zeros(GENERATION)
losses = [[],[],[],[]]
asked = snes.ask()
#asked = snes.asked

t0 = time.time()
step = 0
genloss = []
timeloss = []
for epoch in range(epochs):
    batch = 0
    gen_start = 0
    trg_start = 0
    train_end = BATCH_SIZE*GENERATION #int(BATCH_SIZE*GENERATION/2)
    trgtrain_end = int(ratio*train_end)
    fitsize = train_end #number of samples used for each fitness check (domain classifier uses double)
    domsize = min(train_end, trgtrain_end)
    yd_fit = np.concatenate([np.tile(0, domsize), np.tile(1, domsize)])
    test_start = train_end #use the next gen-batch for validation
    trgtest_start = trgtrain_end
    test_end = 2*train_end
    trgtest_end = 2*trgtrain_end
    gen = 0
    while batch < batches:#2*GENERATION:#
        nnw.set_weights(asked[gen])
        #K.set_weights(model, asked[gen])
        X = model.predict(np.concatenate((source_adapt[gen_start:train_end], target_adapt[trg_start:trgtrain_end])))
        clf.fit(X[:fitsize], ys_adapt[gen_start:train_end])
        clf_dom.fit(np.concatenate((X[:domsize],X[-domsize:])), yd_fit)
        X = model.predict(np.concatenate((source_adapt[test_start:test_end], target_adapt[trgtest_start:trgtest_end])))
#        fitness = []
#        for sample, probs in enumerate(clf.predict_proba(X)):
#            #fitness.append(probs[y_train[train_end + sample]]) #get probability for this class label
#            fitness.append(probs[np.where(clf.classes_ == y_train[train_end + sample])[0][0]])
#        fitnesses[gen] = np.mean(fitness)
        losses[0].append(-log_loss(ys_adapt[test_start:test_end], clf.predict_proba(X[:fitsize])))
        #losses[0].append(accuracy_score(clf.predict(X[:fitsize]), y_train[train_end:test_end]) - 1)
        timeloss.append([float(step),'Label',losses[0][gen], gen])
        #the bigger the label loss, the more it will impair (subtract) from the fitness
        losses[1].append(log_loss(yd_fit, clf_dom.predict_proba(np.concatenate((X[:domsize],X[-domsize:])))) - 1.5)
        #losses[1].append(-accuracy_score(clf_dom.predict(X), yd_fit))
        timeloss.append([float(step),'Domain',losses[1][gen], gen])
        #the bigger the domain loss, the more it will contribute (add) to the fitness
        losses[2].append(losses[0][gen] + losses[1][gen])
        timeloss.append([float(step),'Net Adapt',losses[2][gen], gen])
        losses[3].append(-log_loss(yt_adapt[trgtest_start:trgtest_end], clf.predict_proba(X[fitsize:])))
        timeloss.append([float(step),'Target',losses[3][gen], gen])
        adapt = 1#2 / (1 + np.exp(-10 * step / steps)) - 1
        #current adaptation factor (increases as cycles progress) 
        fitnesses[gen] = losses[0][gen] + 2*losses[1][gen]
        gen += 1
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
            trg_start += int(ratio*BATCH_SIZE*GENERATION)
            train_end = gen_start + BATCH_SIZE*GENERATION#int(gen_start + BATCH_SIZE*GENERATION/2)
            trgtrain_end = trg_start + int(ratio*BATCH_SIZE*GENERATION)
            if (batch + 1) == (batches - GENERATION): #if the NEXT gen-cycle is the last one for this epoch
                test_start = 0 #then use the start of the next epoch for validation
                trgtest_start = 0
                test_end = BATCH_SIZE*GENERATION
                trgtest_end = int(ratio*test_end)
            else:
                test_start = train_end
                trgtest_start = trgtrain_end
                test_end = gen_start + 2*BATCH_SIZE*GENERATION
                trgtest_end = trg_start + 2*int(ratio*BATCH_SIZE*GENERATION)
            snes.tell(asked, fitnesses)
            #snes.fit(fitnesses, range(GENERATION))
            asked = snes.ask()
            #asked = snes.asked
            #print('asked:',asked[0][:2])#, file=output_file)
            fitnesses = np.zeros(GENERATION)
            losses = [[],[],[],[]]
        batch += 1
    
    #if this is the equivalent of 1 epoch, as per original MNIST (27500) sample size
    if (epoch+1)/int(epochs/EPOCHS) == int((epoch+1)/int(epochs/EPOCHS)):
        nnw.set_weights(snes.center)
        X = model.predict(np.concatenate((source_train, target_train))) #baseline untrained network score
        clf.fit(X[:train_size], ys_train)
        #print('Baseline scores - TRAIN')
        print('Epoch',epoch+1,'/',epochs,'- label loss -',
              'Source -- loss:',round(log_loss(ys_train, clf.predict_proba(X[:train_size])), 4),
              '- acc:', round(clf.score(X[:train_size], ys_train), 4), '\n',
              'Target -- loss:',round(log_loss(yt_train, clf.predict_proba(X[train_size:])), 4),
              '- acc:', round(clf.score(X[train_size:], yt_train), 4), file=output_file)
        
        X = np.concatenate((X[:domtrain_size], X[-domtrain_size:]))
        clf_dom.fit(X, yd_train)
        print('- domain loss:',round(log_loss(yd_train, clf_dom.predict_proba(X)), 4),
              '- acc:', round(clf_dom.score(X, yd_train), 4), file=output_file)
        
        X = model.predict(np.concatenate((source_test, target_test)))
        print('label TEST loss - Source:',round(log_loss(ys_test, clf.predict_proba(X[:test_size])), 4),
              '- acc:', round(clf.score(X[:test_size], ys_test), 4), '\n',
              '- Target:',round(log_loss(yt_test, clf.predict_proba(X[test_size:])), 4),
              '- acc:', round(clf.score(X[test_size:], yt_test), 4), file=output_file)
        
        X = np.concatenate((X[:domtest_size], X[-domtest_size:]))
        print('domain TEST loss:',round(log_loss(yd_test, clf_dom.predict_proba(X)), 4),
              '- acc:', round(clf_dom.score(X, yd_test), 4), file=output_file)

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

#X = model.predict(source_train) #full set now needed to generate full random forest classifier
#clf.fit(X, y_train) 
#X = model.predict(source_test)
#score = clf.score(X, y_test)
#
#print('Test score:', score)

#history = model.fit(source_train, mnist.train.labels,
#                    BATCH_SIZE=BATCH_SIZE, num_epochs=epochs,
#                    verbose=1, validation_data=(source_test, mnist.test.labels))
#score = model.evaluate(source_test, mnist.test.labels, verbose=0)

#print('Test score:', score)
#print('Test accuracy:', score)

output_file.close()




