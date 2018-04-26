# -*- coding: utf-8 -*-
"""
Created on Sat Apr  7 19:29:46 2018

@author: Fish
"""

from PIL import Image
from random import shuffle
#from skimage.transform import resize
#import glob
import random
import numpy as np
import math
import os
import pickle as pkl

domain = 'dslr'
path = 'C:/Users/Fish/Documents/GitHub/domain-adaptation/Office 31/' + domain + '/images'
#training = []
#testing = []
tot_train = 0
tot_test = 0
height = 64#32 #height = math.inf
width = 64# #width = math.inf

#first, find the largest class
largest_class = 0
num_classes = 0
for root, dirs, files in os.walk(path):
    for directory in dirs:
        num_classes += 1
        largest_class = max(largest_class, len(os.listdir(path + "/" + directory)))

"""-----------intitialise buckets for collecting samples for train and test sets----------------------
   -------the number of buckets is equal to the size of the LARGEST class------------------ """     
#training & test sets will be constructed from each class in turn
#this will ensure an equivalent distribution of labels between train & test sets
train_size = int(0.85*largest_class)
train_buckets = [[] for x in range(train_size)]
train_indexes = np.array(range(train_size), dtype = np.uint8)
train_indexes = np.column_stack((train_indexes, np.zeros(train_size, dtype = np.uint8)))

test_size = largest_class - train_size
test_buckets = [[] for x in range(test_size)]
test_indexes = np.array(range(test_size), dtype = np.uint8)
test_indexes = np.column_stack((test_indexes, np.zeros(test_size, dtype = np.uint8)))

"""-----------fill each bucket with evenly distributed classes ----------------------
   -------the buckets are randomised each time, and then sorted so smallest buckets are filled first """
for root, dirs, files in os.walk(path):
#iterate through them
    for label, directory in enumerate(dirs):
        train_count = 0
        test_count = 0
        np.random.shuffle(train_indexes) #randomise the indexes
        np.random.shuffle(test_indexes) 
        train_indexes = train_indexes[np.argsort(train_indexes[:, 1])] #move smallest buckets back to the top
        test_indexes = test_indexes[np.argsort(test_indexes[:, 1])]
        file_names = os.listdir(path + "/" + directory)
        train_size = int(0.85*len(file_names))
        shuffle(file_names)
        #for file in files:          
        #for counter, filename in enumerate(glob.glob(path + '/' + directory +'/*.jpg')): #assuming gif
        for counter, filename in enumerate(file_names):           
            img = Image.open(path + '/' + directory + '/' + filename)
            img.load()
            if counter < train_size:
                train_buckets[train_indexes[train_count, 0]].append([np.asarray(img.resize((height, width), Image.LANCZOS), dtype="uint8"), label])
                train_indexes[train_count, 1] += 1
                train_count += 1
            else:
                test_buckets[test_indexes[test_count, 0]].append([np.asarray(img.resize((height, width), Image.LANCZOS), dtype="uint8"), label])#img, dtype="uint8"), label])#
                test_indexes[test_count, 1] += 1
                test_count += 1
        tot_train += train_count
        tot_test += test_count

train = np.zeros([tot_train, height, width, 3], np.uint8)
y_train = np.zeros(tot_train, dtype=np.int64)
test = np.zeros([tot_test, height, width, 3], np.uint8)
y_test = np.zeros(tot_test, dtype=np.int64)

"""-----------collect distributed samples from each bucket in turn ---------------------- """ 
sample = 0
for images in train_buckets:
    shuffle(images)
    for image in images:
        train[sample] = image[0]#[:height,:width]
        y_train[sample] = image[1]
        #training.append(label)
        sample += 1

"""-----------repeat for the the test set ---------------------- """
sample = 0
for images in test_buckets:
    shuffle(images)
    for image in images:
        test[sample] = image[0]#[:height,:width]
        y_test[sample] = image[1]
        #training.append(label)
        sample += 1

#Save dataset as pickle
with open(domain+'_train64.pkl', 'wb') as f:
    pkl.dump({ 'samples': train, 'labels': y_train }, f, pkl.HIGHEST_PROTOCOL)
    
with open(domain+'_test64.pkl', 'wb') as f:
    pkl.dump({ 'samples': test, 'labels': y_test }, f, pkl.HIGHEST_PROTOCOL)
    
#shape = training[0][0].shape
#train = np.zeros([len(training), height, width, 3], np.uint8)
#y_train = np.zeros(len(training), dtype=np.int64)
#for sample, image in enumerate(training):
##    if image[0].shape[0] > height or image[0].shape[1] > width:
##        rgbimage = Image.fromarray(image[0], 'RGB')
##        train[sample] = np.asarray(rgbimage.resize((height, width), Image.LANCZOS),  dtype="uint8")
##    else:
##        train[sample] = image[0]
#    #for amazon images, simple cropping is fine, as any deviations are 1 or 2 pixels only
#    #(same would also apply for DSLR images - ONLY IF using 1000x1000 native resolution)
#    train[sample] = image[0]#[:height,:width]
#    y_train[sample] = image[1]
#    
#test = np.zeros([len(testing), height, width, 3], np.uint8)
#y_test = np.zeros(len(testing), dtype=np.int64)
#for sample, image in enumerate(testing):
##    if image[0].shape[0] > height or image[0].shape[1] > width:
##        rgbimage = Image.fromarray(image[0], 'RGB')
##        test[sample] = np.asarray(rgbimage.resize((height, width), Image.LANCZOS),  dtype="uint8")
##    else:
##        test[sample] = image[0]
#    #for amazon images, simple cropping is fine, as any deviations are 1 or 2 pixels only
#    #(same would also apply for DSLR images - ONLY IF using 1000x1000 native resolution)
#    test[sample] = image[0]#[:height,:width]
#    y_test[sample] = image[1]

