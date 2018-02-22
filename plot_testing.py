# -*- coding: utf-8 -*-
"""
Created on Sun Feb 18 16:48:12 2018

@author: Fish
"""
import scipy.io

directory = 'BSR/BSDS500/data/groundTruth/train/'
mat = scipy.io.loadmat(directory + '2092.mat')
print(mat)
