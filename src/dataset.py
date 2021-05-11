#!/usr/bin/env python3
#
# Created on Tue May 11 2021
#
# Arthur Lang
# dataset.py
#

import os
import cv2
import numpy

# @function loadPictures
# load pictures from a folder. This folder should contain folders with classes name.
# @param path: path of the folder.
# @return data and target extract from the folder.
def loadPictures(path):
    target = []
    data = []
    for i, targetFolder in enumerate(os.listdir(path)):
        for imageFile in os.listdir(os.path.join(path, targetFolder)):
            img = cv2.imread(os.path.join(path, targetFolder, imageFile), cv2.IMREAD_GRAYSCALE)
            data.append(img)
            target.append(i)
    return data, target

# @function unionShuffle
# shuffle two array in the same way
# @param a: array a
# @param b: array b
def unionShuffle(a, b):
    rng_state = numpy.random.get_state()
    numpy.random.shuffle(a)
    numpy.random.set_state(rng_state)
    numpy.random.shuffle(b)

# @function load
# load the dataset
# @param file: the folder in wich the dataset is store. This folder should contain a folder train and 
#              test in wich one you have a folder for each classes
# @return the dataset
def loadFromFile(file):
    dataTrain = targetTrain = dataTest = targetTest = []
    dataTrain, targetTrain = loadPictures(os.path.join(file, "train"))
    dataTrain, targetTrain = loadPictures(os.path.join(file, "test"))
    unionShuffle(dataTrain, targetTrain)
    unionShuffle(dataTest, targetTest)
    return numpy.array(dataTrain), numpy.array(targetTrain), numpy.array(dataTest), numpy.array(targetTest)