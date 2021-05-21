#
# Created on Thu May 13 2021
#
# Arthur Lang
# SimpleCNN.py
#

import numpy

from src.AModel import AModel
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, Dropout, MaxPooling2D, Flatten, Dense
from tensorflow.keras.regularizers import l2

# @class SimpleCNN
# model base on the implementation of https://github.com/MinG822/ferpredict3
class SimpleCNN(AModel):

    def __init__(self):
        super().__init__()

    def buildModel(self):
        model = Sequential()
        model.add(Conv2D(64, kernel_size=(3, 3), activation='relu', input_shape=(48, 48, 1), 
            data_format='channels_last', kernel_regularizer=l2(0.01)))
        model.add(Conv2D(64, kernel_size=(3, 3), activation='relu'))
        model.add(Dropout(0.4))
        model.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2)))
        model.add(Conv2D(2 * 64, kernel_size=(3, 3), activation='relu'))
        model.add(Conv2D(2 * 64, kernel_size=(3, 3), activation='relu'))
        model.add(Dropout(0.4))
        model.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2)))
        model.add(Conv2D(2 * 2 * 64, kernel_size=(3, 3), activation='relu'))
        model.add(Conv2D(2 * 2 * 64, kernel_size=(3, 3), activation='relu'))
        model.add(Dropout(0.4))
        model.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2)))
        model.add(Flatten())

        model.add(Dense(2*2*64, activation='relu'))
        model.add(Dropout(0.5))
        model.add(Dense(2*64, activation='relu'))
        model.add(Dropout(0.4))
        model.add(Dense(7, activation='softmax'))
        model.compile(loss='sparse_categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
        
        return model