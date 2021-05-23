#!/usr/bin/env python3
#
# Created on Mon Apr 12 2021
#
# Arthur Lang
# evaluate.py
#

import sys
import keras
import numpy
import matplotlib.pyplot as plt
import seaborn
from sklearn.model_selection import KFold
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay

from src.dataset import loadFromFile

def checkArg(av):
    return len(av) == 3

def displayConfusionMatrix(matrix):
    print(matrix)
    plt.figure(figsize=matrix.shape)
    seaborn.set(font_scale=1.4) # for label size
    seaborn.heatmap(matrix, annot=True, annot_kws={"size": 16}, fmt='g') # font size
    plt.show()

def crossValidation(data, target, model, split_size=5):
    results = []
    kf = KFold(n_splits=split_size)
    for trainIdx, valIdx in kf.split(data, target):
        trainData = data[trainIdx]
        trainTarget = target[trainIdx]
        testData = data[valIdx]
        testTarget = target[valIdx]

        model.trainWithoutValidation(trainData, trainTarget)
        predict = model.predict(testData)
        results.append(confusion_matrix(testTarget, predict))
    return results

def evaluate():
    av = sys.argv
    if (not(checkArg(av))):
        print("Usage: evaluate model dataset")
        return -1
    model = keras.models.load_model(av[1])

    trainData, trainLabels, testData, testLabels = loadFromFile(av[2])

    prediction = numpy.argmax(model.predict(testData), axis=1)
    displayConfusionMatrix(confusion_matrix(testLabels, prediction))

if __name__ == "__main__":
    evaluate()