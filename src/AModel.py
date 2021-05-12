#
# Created on Tue May 11 2021
#
# Arthur Lang
# AModel.py
#

import tensorflow
import numpy

from abc import ABC, abstractmethod

class AModel(ABC):

    def __init__(self, weightPath = None, modelPath = None):
        self._hasWeight = False
        if (modelPath != None):
            self.model = tensorflow.keras.models.load_model(modelPath)
            self._hasWeight = True
        else:
            self.model = self.buildModel()
            if (weightPath != None):
                self.model.load_weights(weightPath)
                self._hasWeight = True
        self._outputWeightPath = "weights/lastTraining/cp1.ckpt"
        self.model.summary()

    ## _buildModel
    # build and return the model
    @abstractmethod
    def buildModel():
        pass

    ## train
    # train the model and save the gotten weights.
    # @param data: data to train on.
    # @param labels: labels of the corresponding data.
    # @param validationData: tuple containing data and labels for validation.
    # @param weightsFile: file where to save files.
    def train(self, data, labels, validationData=(), epochs = 10, weightsFile = ""):
        if (weightsFile != ""):
            cpCallBack = tensorflow.keras.callbacks.ModelCheckpoint(filepath=weightsFile, save_weights_only=True, verbose=1)
            self.model.fit(data, labels, batch_size=128, epochs = epochs, validation_data=validationData, callbacks=[cpCallBack])
        else:
            self.model.fit(data, labels, batch_size=128, epochs = epochs, validation_data=validationData)

    ## evaluate
    # evaluate the model
    def evaluate(self, testData, testLabel):
        return self.model.evaluate(testData, testLabel)

    ## predict
    # to a prediction on a set of data
    # @param testData: the data you want to make prediction on
    # @return an array with all the predicted classes
    def predict(self, testData):
        return numpy.argmax(self.model.predict(testData), axis=-1)
