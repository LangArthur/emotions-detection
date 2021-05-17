#!/usr/bin/env python3
#
# Created on Tue May 11 2021
#
# Arthur Lang
# train.py
#
import os.path
from os import path
import tensorflow
import numpy
from tensorflow import keras
from src.dataset import  loadFromFile
from src.Models.ModelInception import ModelInception
from src.Models.SimpleCNN import SimpleCNN
from src import AModel
from tensorflow import keras


# @function trainpip install keras
# launch a train session
def train():
    dataTrain, labelTrain, dataTest, labelTest = loadFromFile("ressources/datasets/FER-2013")
    print("Load the dataset")
    model = ModelInception()
    model.train(dataTrain, labelTrain, (dataTest, labelTest), 150, "ressources/weights/Inception/")
    return 0


if __name__ == '__main__':
    # train()
    # dataTrain, labelTrain, dataTest, labelTest = load_dataset("/colabdrive/EmotionDetection/emotions-detection/ressources/datasets/EmotionDataset")
    # dataTrain, labelTrain,dataTest, labelTest = loadFromFile("/colabdrive/emotions-detection/ressources/datasets/EmotionsDataset")
    # # labelTrain1=numpy.zeros(len(labelTrain))
    # labelTest1=numpy.zeros(len(labelTest))
    # print("ok")
    # # model=SimpleCNN()
    # # model.train()
    # model=AModel()
    # model.train(dataTrain, labelTrain, (dataTest, labelTest), 100)
    # model.save("/colabdrive/EmotionDetection/emotions-detection/src/Models/SimpleCNNSaved/LoadFromFileMethod")
    # model=SimpleCNN()
    # model =keras.models.load_model("/colabdrive/EmotionDetection/emotions-detection/src/Models/SimpleCNNSaved/LoadFromFileMethod")
    # model.summary()
    # print(model.evaluate(dataTest, labelTest))
    converter = tensorflow.lite.TFLiteConverter.from_saved_model("/colabdrive/EmotionDetection/emotions-detection/src/Models/SimpleCNNSaved/LoadFromFileMethod") # path to the SavedModel directory
    tflite_model = converter.convert()
    with open('/colabdrive/emotions-detection/src/Models/model.tflite', 'wb') as f:
        f.write(tflite_model)