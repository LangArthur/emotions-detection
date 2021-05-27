#
# Created on Tue May 11 2021
#
# Arthur Lang
# ModelInception.py
#

import numpy

from src.AModel import AModel
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Conv2D, BatchNormalization, Activation, concatenate, Input, MaxPool2D, Dense, Reshape, AveragePooling2D, Lambda


# @class ModelInception
# model from A. Mollahosseini, D. Chan, et M. H. Mahoor, « Going deeper in facial expression recognition using deep neural networks », 
# in 2016 IEEE Winter Conference on Applications of Computer Vision (WACV), mars 2016, p. 1‐10, doi: 10.1109/WACV.2016.7477450.
class ModelInception(AModel):

    def __init__(self, weightPath = None, modelPath = None):
        super().__init__(weightPath, modelPath)

    def buildModel(self):

        def CNNModule(layers, filters, kernelSize, poolSize, strides):
            layers = Conv2D(filters, kernelSize, strides=strides[0], activation="relu", padding="same")(layers)
            layers = MaxPool2D(pool_size=poolSize, strides=strides[1], padding="same")(layers)
            return layers

        def InceptionModule(layers, num1x1, reduce3x3, num3x3, reduce5x5, num5x5, projValue):
            conv1x1 = Conv2D(num1x1, (1, 1), activation="relu", padding="same")(layers)

            reduceConv3x3 = Conv2D(reduce3x3, (5, 5), activation="relu", padding="same")(layers)
            conv3x3 = Conv2D(num3x3, (3, 3), activation="relu", padding="same")(reduceConv3x3)

            reduceConv5x5 = Conv2D(reduce3x3, (5, 5), activation="relu", padding="same")(layers)
            conv5x5 = Conv2D(num5x5, (5, 5), activation="relu", padding="same")(reduceConv5x5)

            poolProj = MaxPool2D((3, 3), strides=2, padding="same")(layers)
            poolProj = Conv2D(projValue, kernel_size=(1, 1), padding="same")(poolProj)

            layers = concatenate([conv1x1, conv3x3, conv5x5])
            return layers

        inputShape = (48, 48, 1)

        inputLayer = Input(shape=inputShape)
        layers = CNNModule(inputLayer, 64, (7, 7), (3, 3), (2, 2))
        layers = CNNModule(layers, 192, (3, 3), (3, 3), (1, 2))
        layers = InceptionModule(layers, 64, 96, 128, 16, 32, 32)
        layers = InceptionModule(layers, 128, 128, 192, 32, 96, 64)
        layers = MaxPool2D((3, 3), strides=2, padding="same")(layers)
        layers = InceptionModule(layers, 192, 96, 208, 16, 48, 64)
        layers = AveragePooling2D((3, 3), padding="same")(layers)
        layers = Dense(4096)(layers)
        layers = Dense(1024)(layers)
        model = Model(inputLayer, layers, name="ModelInception")
        model.compile(loss='sparse_categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
        
        return model