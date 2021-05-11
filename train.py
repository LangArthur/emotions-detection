#!/usr/bin/env python3
#
# Created on Tue May 11 2021
#
# Arthur Lang
# train.py
#

from src.dataset import loadFromFile
from src.Models.ModelInception import ModelInception

# @function train
# launch a train session
def train():
    dataTrain, labelTrain, dataTest, labelTest = loadFromFile("ressources/datasets/FER-2013")
    print("Load the dataset")
    model = ModelInception()
    # print(type(labelTrain))
    model.train(dataTrain, labelTrain, (dataTest, labelTest), "ressources/weights/Inception/")
    return 0

if __name__ == '__main__':
    train()