#!/usr/bin/env python3
#
# Created on Tue May 11 2021
#
# Arthur Lang
# train.py
#

import tensorflow as tf
import keras

from src.dataset import loadFromFile
from src.Models.ModelInception import ModelInception
from src.Models.SimpleCNN import SimpleCNN
from src.Models.CNNv2 import CNNv2
from src import AModel

def compressModel(model, path):
    converter = tf.lite.TFLiteConverter.from_keras_model(model)
    tflite_model = converter.convert()
    with open(path, 'wb') as f:
        f.write(tflite_model)

# @function train
# launch a training session
def train(modelName):
    # availableModels = {
    #     "inception": ModelInception,
    #     "cnn": SimpleCNN,
    #     "cnnv2":CNNv2
    # }
    # model = availableModels[modelName]()
    # dataTrain, labelTrain, dataTest, labelTest = loadFromFile("/colabdrive/emotions-detection/ressources/datasets/EmotionsDataset")
    # print("ok")
    # model.train(dataTrain, labelTrain, (dataTest, labelTest), 50)
    # model.save("/colabdrive/emotions-detection/src/Models/Cnnv2-70batch-50epochs")
    model = keras.models.load_model("ressources/Models/Cnnv2-70batch-50epochs")
    compressModel(model, "ressources/Models/Cnnv2-lite")
    return 0


if __name__ == '__main__':
    train("cnnv2")


