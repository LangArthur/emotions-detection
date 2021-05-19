#!/usr/bin/env python3
#
# Created on Tue May 11 2021
#
# Arthur Lang
# train.py
#

from src.dataset import loadFromFile
from src.Models.ModelInception import ModelInception
from src.Models.SimpleCNN import SimpleCNN
from src.Models.CNNv2 import CNNv2
from src import AModel

# @function trainpip install keras
# launch a training session
def train(modelName):
    availableModels = {
        "inception": ModelInception,
        "cnn": SimpleCNN,
        "cnnv2":CNNv2
    }
    model = availableModels[modelName]()
    dataTrain, labelTrain, dataTest, labelTest = loadFromFile("/colabdrive/emotions-detection/ressources/datasets/EmotionsDataset")
    print("ok")
    model.train(dataTrain, labelTrain, (dataTest, labelTest), 50)
    model.save("/colabdrive/emotions-detection/src/Models/")
    return 0


if __name__ == '__main__':
    train("cnnv2")


