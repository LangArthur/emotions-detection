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
    return numpy.reshape(dataTrain, (len(dataTrain), 48, 48, 1)), numpy.array(targetTrain), numpy.reshape(dataTest, (len(dataTest), 48, 48, 1)), numpy.array(targetTest)


def load_dataset(folder):
    data = []
    label = []
    images_train = []
    labels_train = []
    images_test = []
    labels_test = []
    # iterate through each folders
    for item in os.listdir(folder):
          datasets = os.path.join(folder, item)
          print(datasets)
          for element in os.listdir(datasets):
            # print(element)
            imagePath=os.path.join(datasets, element)
            print(imagePath)
            for image in os.listdir(imagePath):
              img= cv2.imread(os.path.join(imagePath,image), cv2.IMREAD_GRAYSCALE)/255.0
              if item=='train':
                break
                images_train.append(img)
                labels_train.append((element))
              elif item=='test':
                images_test.append(img)
                labels_test.append((element))
            # iterate throught images
            # for element in os.listdir(datasets):
              # for digit in os.listdir()
              #    # data.append(img)
              #   # label.append(int(digitFolder))
              #   # push to train tdataset
              #   if item == "train" and img is not None:
              #       images_train.append(img)
              #       labels_train.append(int(number))
              #   # push to test tdataset
              #   elif item == "test" and img is not None:
              #       images_test.append(img)
              #       labels_test.append(int(number))
    
    # return numpy.reshape(images_train, (len(images_train), 48, 48, 1)), numpy.array(labels_train), numpy.reshape(images_test, (len(images_test), 48, 48, 1)), numpy.array(labels_test)
    return  numpy.reshape(images_test, (len(images_test), 48, 48, 1)), numpy.array(labels_test)
    # return data, label