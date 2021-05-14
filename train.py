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
from src.dataset import load_dataset
from src.Models.ModelInception import ModelInception
from src.Models.SimpleCNN import SimpleCNN
# import sklearn
from sklearn import preprocessing

# @function trainpip install keras
# launch a train session
def train():
    dataTrain, labelTrain, dataTest, labelTest = loadFromFile("ressources/datasets/FER-2013")
    print("Load the dataset")
    model = ModelInception()
    model.train(dataTrain, labelTrain, (dataTest, labelTest), 150, "ressources/weights/Inception/")
    return 0
    # dataTrain, labelTrain, dataTest, labelTest = loadFromFile("/colabdrive/EmotionDetection/emotions-detection/ressources/datasets/EmotionDataset")
    # print("Load the dataset")
    # model = SimpleCNN()
    # model.train(dataTrain, labelTrain, (dataTest, labelTest), epochs=150, batch_size=50)
    # return 0

def load_model(path, modelName):
    dataTrain, labelTrain, dataTest, labelTest = loadFromFile("/colabdrive/EmotionDetection/emotions-detection/ressources/datasets/EmotionDataset")
    if modelName=='Inception':
        model=ModelInception().buildModel()
    elif modelName=='SimpleCNN':
        model=SimpleCNN().buildModel() 
    model.load_weights(path)
    return model


if __name__ == '__main__':
    train()
    # model=load_model("ressources/weights/SimpleCNN/", "SimpleCNN")
    # model.summary()
    # dataTrain, labelTrain, dataTest, labelTest = load_dataset("/colabdrive/EmotionDetection/emotions-detection/ressources/datasets/EmotionDataset")
    dataTest, labelTest = load_dataset("/colabdrive/EmotionDetection/emotions-detection/ressources/datasets/EmotionDataset")
    labelEncoder =preprocessing.LabelEncoder()
    labelEncoder.fit(['angry', 'disgust', 'fear', 'happy', 'sad', 'surprise', 'neutral'])
    # labelTrain1=labelEncoder.transform(labelTrain)
    labelTest1=labelEncoder.transform(labelTest)
    print(labelTest1)
    print(dataTest)
    # # labelTrain1=numpy.zeros(len(labelTrain))
    # labelTest1=numpy.zeros(len(labelTest))
    # print("ok")
    # # model=SimpleCNN()
    # # model.train()
    # model=SimpleCNN()
    # model.train(dataTrain, labelTrain1, (dataTest, labelTest1), 150)
    # model.save("/colabdrive/EmotionDetection/emotions-detection/src/Models/SimpleCNNSaved/1")
    model = keras.models.load_model("/colabdrive/EmotionDetection/emotions-detection/src/Models/SimpleCNNSaved/1")
    model.summary()
    model.evaluate(dataTest, labelTest1)
    
    # print(labelTrain1)
    # print("ok")
    # model.load_weights("ressources/weights/SimpleCNN/")
    # model.evaluate(dataTest, labelTest)
    # model.summary()
