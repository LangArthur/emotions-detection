#
# Created on Tue May 11 2021
#
# Arthur Lang
# ModelInception.py
#

from src.AModel import AModel
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Conv2D, BatchNormalization, Activation, concatenate, Input, MaxPool2D, Dense


# @class ModelInception
# model from A. Mollahosseini, D. Chan, et M. H. Mahoor, « Going deeper in facial expression recognition using deep neural networks », 
# in 2016 IEEE Winter Conference on Applications of Computer Vision (WACV), mars 2016, p. 1‐10, doi: 10.1109/WACV.2016.7477450.
class ModelInception(AModel):

    def __init__(self):
        super().__init__()

    def buildModel(self):

        def CNNModule(layers, filters, kernelSize, poolSize, stride, chanDim, padding="same"):
            layers = Conv2D(filters, kernelSize, strides=stride, padding=padding)(layers)
            print(poolSize)
            layers = MaxPool2D(pool_size=poolSize, padding=padding)(layers)
            # layers = Activation("relu")(layers)
            return layers

        def InceptionModule(layers, num1x1, num3x3, num5x5, chanDim):
            conv1x1 = Conv2D(num1x1, (1, 1), strides=(1, 1), activation="relu", padding="same")(layers)
            conv3x3 = Conv2D(num3x3, (1, 1), strides=(1, 1), activation="relu", padding="same")(layers)
            conv3x3 = Conv2D(num3x3, (3, 3), strides=(1, 1), activation="relu", padding="same")(conv3x3)
            conv5x5 = Conv2D(num5x5, (1, 1), strides=(1, 1), activation="relu", padding="same")(layers)
            conv5x5 = Conv2D(num5x5, (5, 5), strides=(1, 1), activation="relu", padding="same")(conv5x5)
            # maxPool = MaxPool2D(pool_size=(3, 3), padding="same")(layers)
            # maxPool = Conv2D(1, (1, 1), activation="relu", padding="same")(maxPool)
            layers = concatenate([conv1x1, conv3x3, conv5x5], axis=chanDim)
            return layers

        inputShape = (48, 48, 1)
        chanDim = -1

        inputLayer = Input(shape=inputShape)
        layers = CNNModule(inputLayer, 1, (7, 7), (3, 3), (2, 2), chanDim)
        # layers = CNNModule(layers, 1, (3, 3), (3, 3), (1, 1), chanDim)
        layers = InceptionModule(layers, 1, 1, 1, chanDim)
        layers = InceptionModule(layers, 1, 1, 1, chanDim)
        layers = MaxPool2D((3, 3), padding="same")(layers)
        layers = InceptionModule(layers, 1, 1, 1, chanDim)
        layers = MaxPool2D((3, 3))(layers)
        layers = Dense(4096)(layers)
        layers = Dense(1024)(layers)
        model = Model(inputLayer, layers, name="ModelInception")
        model.compile(loss='sparse_categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
        return model