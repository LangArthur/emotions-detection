#!/usr/bin/env python3
#
# Created on Mon Apr 12 2021
#
# Arthur Lang
# evaluate.py
#

import sys
import argparse
import time

import keras
import tensorflow as tf
import numpy
import matplotlib.pyplot as plt
import seaborn
from sklearn.model_selection import KFold
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay, accuracy_score, precision_score, recall_score

from src.dataset import loadFromFile

# @function displayConfusionMatrix
# display nicely a matrix in a window and in a terminal
# @param matrix matrix to display
def displayConfusionMatrix(matrix):
    print(matrix)
    plt.figure(figsize=matrix.shape)
    seaborn.set(font_scale=1.4) # for label size
    seaborn.heatmap(matrix, annot=True, annot_kws={"size": 16}, fmt='g') # font size
    plt.show()

# @crossValdiation
# run a cross validation
# @param data to use.
# @param target target corresponding to the data
# @param model model to test
# @param split_size size of each split
# @return a table containing a confusion matrix for each split
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

# @function litePredict
# return a prediction for an image
# @tfliteInterpreter tflite model
# @img the image to predict
# @inputDetails input detail from tflite interpreter
# @outputDetails output details from tflite interpreter
def litePredict(tfliteInterpreter, img, inputDetails, outputDetails):
    tfliteInterpreter.set_tensor(inputDetails[0]['index'], img)
    tfliteInterpreter.invoke()
    tfliteModelPredictions = tfliteInterpreter.get_tensor(outputDetails[0]['index'])
    prediction = numpy.argmax(tfliteModelPredictions[0], axis=-1)
    return prediction

# @function litePrediction
# predict with a lit model
# @param modelPath path for the model
# @param testData data for testing
# @return predictions done by the model
def litePrediction(modelPath, testData):
    # Load lite model
    tfliteInterpreter = tf.lite.Interpreter(modelPath)
    inputDetails = tfliteInterpreter.get_input_details() #Get input details
    outputDetails = tfliteInterpreter.get_output_details() #Get output details
    # initialize variable for prediction
    tfliteInterpreter.resize_tensor_input(inputDetails[0]['index'], (1, 48, 48, 1)) #Resize the input for making prediction on an image
    tfliteInterpreter.resize_tensor_input(outputDetails[0]['index'], (1, 7))
    tfliteInterpreter.allocate_tensors()
    prediction = []
    duration = 0
    # predict each pictures
    for rawImg in testData:
        rawImg = numpy.array(rawImg,  dtype=numpy.float32) / 255.0
        img = rawImg[numpy.newaxis, ...]
        startTime = time.time()
        prediction.append(litePredict(tfliteInterpreter, img, inputDetails, outputDetails))
        duration += time.time() - startTime
    duration /= len(prediction)
    print(f"Prediction time for an image: {duration} seconds")
    return prediction

def scoring(pred, reality):
    print("\nScoring:")
    print("Accuracy: {}".format(accuracy_score(pred, reality)))
    print("Recall: {}".format(recall_score(pred, reality, average="micro", zero_division=0)))
    print("Precision: {}\n".format(precision_score(pred, reality, average="micro")))

# @function evaluate
# evaluate a model
def evaluate():
    parser = argparse.ArgumentParser(description="Script for model evaluation.")
    parser.add_argument("model", type=str, help="path of the model to load.")
    parser.add_argument("dataset", type=str, help="path of the dataset to use")
    parser.add_argument("-l", "--lite", action="store_true", help="specified this flag if model is a tflite model")
    args = parser.parse_args()

    trainData, trainLabels, testData, testLabels = loadFromFile(args.dataset)

    if (args.lite):
        prediction = litePrediction(args.model, testData)
    else:
        model = keras.models.load_model(args.model)
        prediction = numpy.argmax(model.predict(testData), axis=1)
    scoring(prediction, testLabels)
    displayConfusionMatrix(confusion_matrix(testLabels, prediction))

if __name__ == "__main__":
    evaluate()