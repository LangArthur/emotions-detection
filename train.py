#!/usr/bin/env python3
#
# Created on Tue May 11 2021
#
# Arthur Lang
# train.py
#

#%%
import os.path
from cv2 import cv2
from os import path
import tensorflow
import numpy
from tensorflow import keras
from src.dataset import  loadFromFile
from src.Models.ModelInception import ModelInception
from src.Models.SimpleCNN import SimpleCNN
from src import AModel
from tensorflow import keras
import matplotlib.pyplot as plt


# @function trainpip install keras
# launch a train session
def train(modelName):
    availableModels = {
        "inception": ModelInception,
        "cnn": SimpleCNN,
    }
    model = availableModels[modelName]()
    dataTrain, labelTrain, dataTest, labelTest = loadFromFile("/colabdrive/emotions-detection/ressources/datasets/EmotionsDataset")
    print("ok")
    model.train(dataTrain, labelTrain, (dataTest, labelTest), 150)
    model.save("/colabdrive/emotions-detection/src/Models")
    return 0


if __name__ == '__main__':
    # train("cnn")

    model = keras.models.load_model("/colabdrive/emotions-detection/src/Models/SimpleCNN/LoadFromFileMethod")
    img = cv2.imread('/colabdrive/emotions-detection/ressources/datasets/EmotionsDataset/test/angry/PrivateTest_88305.jpg', cv2.IMREAD_GRAYSCALE)/255.0
    img_reshaped =numpy.reshape(img, (1,48,48,1))
    print(numpy.argmax(model.predict(img_reshaped)))
    print("ok")
    cv2.imshow("test", img)
    print("ok")
    # imgplot = plt.imshow(img)
# %%
