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

from PIL import Image

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
IMAGE_SIZE = 64
DEPTH = 100
DOM_WEIGHT = 2
LEARN_RATE = 2
TREE_DEPTH = 7 #for label classifier
GENERATION = 12#36 #number batches (population size) used for each snes GENERATION
BATCH_SIZE = 300 #optimum ideal batch size, all other things being equal
EPOCHS = 20 #baseline #epochs, as per MNIST data
SAMPLES = 1450000 #target number of samples to be processed to full adaptation
SAMPLES_PER_EPOCH = 13500 #baseline #samples per epoch, as per MNIST data
#train_size = 27500#55000#number of samples we will use for training

domains = [('source', 'dslr'),
           ('target', 'webcam'),
           ('unseen', 'amazon')]

def loadimages(domain, partition):
    store = pkl.load(open(domain+'_'+partition+str(IMAGE_SIZE)+'.pkl', 'rb'))
    images = store['samples']#[:domtrain_size]
    labels = store['labels']
    return images, labels

"""--------------load pickled images and labels from each domain-----------------------------
   --------------normalise images, and build domain labels---------------------------------- """
source_train, ys_train = loadimages(domains[0][1], 'train')
source_test, ys_test = loadimages(domains[0][1], 'test')
target_train, yt_train = loadimages(domains[1][1], 'train')
target_test, yt_test = loadimages(domains[1][1], 'test')
unseen_train, yu_train = loadimages(domains[2][1], 'train')
unseen_test, yu_test = loadimages(domains[2][1], 'test')

def decolourise(images):
    grey_images = np.zeros([len(images), IMAGE_SIZE, IMAGE_SIZE, 1], np.uint8)
    for sample, image in enumerate(images):
        image = Image.fromarray(image, 'RGB')
        image = np.asarray(image.convert('L'), dtype="uint8")
        grey_images[sample] = image.reshape(64, 64, 1)
    return grey_images

source_train = decolourise(source_train)
source_test = decolourise(source_test)
target_train = decolourise(target_train)
target_test = decolourise(target_test)
unseen_train = decolourise(unseen_train)
unseen_test = decolourise(unseen_test)

#imshow_grid(source_train)
#imshow_grid(target_train)

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
unseen_train = (unseen_train - pixel_mean) / 255
unseen_test = (unseen_test - pixel_mean) / 255

yd_train = np.concatenate([np.tile(0, domtrain_size), np.tile(1, domtrain_size)])
yd_test = np.concatenate([np.tile(0, domtest_size), np.tile(1, domtest_size)])

"""--------------construct and initialise CNN model----------------------------- """
inputs = Input(shape=(64, 64, 1))
x = Conv2D(24, kernel_size=5, activation='relu')(inputs)
#x = PReLU()(x) # Non-linearity
x = MaxPooling2D(pool_size=(3, 3))(x)
#x = Dropout(rate=0.2)(x)
x = Conv2D(36, kernel_size=5, activation='relu')(x)
x = MaxPooling2D(pool_size=(2, 2))(x)
#x = Dropout(rate=0.2)(x)
x = Flatten()(x)
x = Dense(DEPTH, activation='relu')(x)
features = Dense(100, activation='relu')(x)

#predictions = Dense(NB_CLASSES, activation='softmax')(x)

model = Model(inputs=inputs, outputs=features)#predictions)
#adam = Adam(lr=0.001, beta_1=0.9, beta_2=0.999, epsilon=1e-8, decay=0.0, amsgrad=False)
#sgd = optimizers.SGD(lr=0.01, decay=1e-6, momentum=0.9, nesterov=True)
model.compile(optimizer='adam',#adam,
              loss='categorical_crossentropy',
              metrics=['accuracy'])

model.summary()

"""--------------initialise classifiers ------------------------------------ """
clf = RandomForestClassifier(n_estimators = 50, max_depth = TREE_DEPTH)
clf_dom = RandomForestClassifier(n_estimators = 50, max_depth = 4)

"""--------------and calculate initial starting accuracy for randomised CNN---------------------- """
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

X = model.predict(np.concatenate((unseen_train, unseen_test)))
#print('Baseline scores - TRAIN')
print('Unseen domain\n',
      'Source -- loss:',round(log_loss(np.concatenate((yu_train, yu_test)), clf.predict_proba(X)), 4),
      '- acc:', round(clf.score(X, np.concatenate((yu_train, yu_test))), 4))

"""--------------construct adaptation sets, for transductive adaptation---------------------- """
#as we have so few office samples available, the whole set is utilised for domain adaptation purposes
source_adapt = np.concatenate((source_train, source_test))
target_adapt = np.concatenate((target_train, target_test))
ys_adapt = np.concatenate((ys_train, ys_test))
yt_adapt = np.concatenate((yt_train, yt_test))

"""--------------initialise all parameters for adaptation cycle---------------------- """
nnw = NNWeightHelper(model)
weights = nnw.get_weights()
#weights = K.get_trainable_weights(model)
#dim = len(weights)
#steps = 2*SAMPLES/GENERATION**2/BATCH_SIZE #target generations for SNES to reach full adaptation
batch_size = int(len(source_adapt)/round(len(source_adapt)/BATCH_SIZE, 0))
steps = 2*SAMPLES/GENERATION/batch_size
#batches = int(len(source_adapt)/batch_size) #number iterations for each epoch
batches = int(GENERATION * len(source_adapt)/batch_size)
batches -= batches%GENERATION
trgbatch_size = int(len(target_adapt)/round(len(target_adapt)/batch_size, 0))
trg_batches = int(GENERATION * len(target_adapt)/trgbatch_size)
yd_fit = np.concatenate([np.tile(0, batch_size), np.tile(1, trgbatch_size)])

#epochs = int(EPOCHS*SAMPLES_PER_EPOCH/batch_size/batches)
epochs = int(EPOCHS * GENERATION * SAMPLES_PER_EPOCH / batch_size / batches)
snes = SNES(weights, LEARN_RATE, GENERATION)
#snes = SSNES(weights, 2, GENERATION)#np.zeros(dim), 1, GENERATION)
fitnesses = np.zeros(GENERATION)
losses = [[],[],[],[]]
asked = snes.ask()
#asked = snes.asked

"""----------------------commence adaptation---------------------------- """
t0 = time.time()
step = 0
genloss = []
timeloss = []
trg_batch = 0
trg_start = 0
trgtrain_end = trgbatch_size
trgtest_start = trgtrain_end
trgtest_end = 2*trgtrain_end
for epoch in range(epochs):
    batch = 0
    gen_start = 0
    train_end = batch_size
    test_start = train_end #use the next gen-batch for validation
    test_end = 2*train_end
    gen = 0
    while batch < batches:#2*GENERATION:#
        nnw.set_weights(asked[gen])
        #K.set_weights(model, asked[gen])
        X = model.predict(np.concatenate((source_adapt[gen_start:train_end], target_adapt[trg_start:trgtrain_end])))
        clf.fit(X[:batch_size], ys_adapt[gen_start:train_end])
        clf_dom.fit(X, yd_fit)
        X = model.predict(np.concatenate((source_adapt[test_start:test_end], target_adapt[trgtest_start:trgtest_end])))
#        fitness = []
#        for sample, probs in enumerate(clf.predict_proba(X)):
#            #fitness.append(probs[y_train[train_end + sample]]) #get probability for this class label
#            fitness.append(probs[np.where(clf.classes_ == y_train[train_end + sample])[0][0]])
#        fitnesses[gen] = np.mean(fitness)
        losses[0].append(-log_loss(ys_adapt[test_start:test_end], clf.predict_proba(X[:batch_size])))
        #losses[0].append(accuracy_score(clf.predict(X[:fitsize]), y_train[train_end:test_end]) - 1)
        timeloss.append([float(step),'Label',losses[0][gen], gen])
        #the bigger the label loss, the more it will impair (subtract) from the fitness
        losses[1].append(log_loss(yd_fit, clf_dom.predict_proba(X)) - 1.5)
        #losses[1].append(-accuracy_score(clf_dom.predict(X), yd_fit))
        timeloss.append([float(step),'Domain',losses[1][gen], gen])
        #the bigger the domain loss, the more it will contribute (add) to the fitness
        losses[2].append(losses[0][gen] + losses[1][gen])
        timeloss.append([float(step),'Net Adapt',losses[2][gen], gen])
        losses[3].append(-log_loss(yt_adapt[trgtest_start:trgtest_end], clf.predict_proba(X[batch_size:])))
        timeloss.append([float(step),'Target',losses[3][gen], gen])
        adapt = 1#2 / (1 + np.exp(-10 * step / steps)) - 1
        #current adaptation factor (increases as cycles progress) 
        fitnesses[gen] = losses[0][gen] + DOM_WEIGHT*losses[1][gen]
        gen += 1
        batch += 1
        trg_batch += 1
        """-------------------refresh the SNES Gen Cycle at each 12th batch---------------------- """
        if gen == GENERATION: #if this is the last sample for this SNES generatin
            gen = 0 #then re-set the gen cycle to start the next generation
            step += 1
            genloss.append([max(losses[0]), max(losses[1]), max(losses[2]), max(losses[3])])
            #print('Weights:', K.get_trainable_weights(model)[:2])
            print('Epoch',epoch+1,'- Gen',batch/GENERATION,'/',batches/GENERATION,
                  '--- label loss:', max(losses[0]),'--- domain loss:', max(losses[1]), '\n',
                  '--- NET loss: at adapt',adapt, '-',np.amax(fitnesses),'--', time.time() - t0,"secs", file=output_file)
            gen_start += batch_size
            train_end = gen_start + batch_size
            trg_start += trgbatch_size
            trgtrain_end = trg_start + trgbatch_size
            
            #if the NEXT gen-cycle is the LAST one for this epoch of the SOURCE set
            if batch == (batches - GENERATION): 
                test_start = 0 #then use the first batch of the next epoch for validation
                test_end = batch_size
            else:
                test_start = train_end #else move to the next adjacent source batch as normal
                test_end = gen_start + 2*batch_size
                
            #if THIS gen-cycle is the LAST one for this circuit of the TARGET set
            if trg_batch == trg_batches: 
                trg_batch = 0 #then re-set the batch cycle to start the next circuit
                trg_start = 0 
                trgtrain_end = trgbatch_size
                trgtest_start = trgtrain_end
                trgtest_end = 2*trgtrain_end              
            #if the NEXT gen-cycle is the LAST one for this circuit of the TARGET set
            elif trg_batch == (trg_batches - GENERATION): 
                trgtest_start = 0 #then use the first batch of the next circuit for validation
                trgtest_end = trgbatch_size
            else:
                trgtest_start = trgtrain_end  #else move to the next adjacent target batch as normal
                trgtest_end = trg_start + 2*trgbatch_size
            
            snes.tell(asked, fitnesses)
            #snes.fit(fitnesses, range(GENERATION))
            asked = snes.ask()
            #asked = snes.asked
            #print('asked:',asked[0][:2])#, file=output_file)
            fitnesses = np.zeros(GENERATION)
            losses = [[],[],[],[]]
    """----------------track current model accuracies for the full domain----------------------
       ------------this is performed at the end of every epoch-------------------"""    
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

"""--------------calculate residual accuracy on 3rd unseen office domain---------------------- """
X = model.predict(np.concatenate((unseen_train, unseen_test)))
#print('Baseline scores - TRAIN')
print('Unseen domain\n',
      'Source -- loss:',round(log_loss(np.concatenate((yu_train, yu_test)), clf.predict_proba(X)), 4),
      '- acc:', round(clf.score(X, np.concatenate((yu_train, yu_test))), 4))

"""----------------------and plot the results---------------------- """
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
#                    batch_size=batch_size, num_epochs=epochs,
#                    verbose=1, validation_data=(source_test, mnist.test.labels))
#score = model.evaluate(source_test, mnist.test.labels, verbose=0)

#print('Test score:', score)
#print('Test accuracy:', score)

output_file.close()




