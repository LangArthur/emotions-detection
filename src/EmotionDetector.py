#
# Created on Tue May 11 2021
#
# Arthur Lang
# EmotionDetector.py
#

import ntpath
import keras
import numpy
import cv2
import pyautogui

## getFileName
# get name of a file from a path
# @param path
# @return name of the file
def getFileName(path):
    head, tail = ntpath.split(path)
    return tail or ntpath.basename(head)

class EmotionDetector():

    def __init__(self, path="", modelName="CNNv2", demo=False):
        self.title = "Emotion detector - " # title of the window
        self._demoMode = demo
        if (self._demoMode):
            self.title += " screen" if path == "" else getFileName(path)
        else:
            self.title += " webcam" if path == "" else getFileName(path)
            self._cap = cv2.VideoCapture(0 if path == "" else path)
        self.isRunning = True
        self.availableModels = {
            "Inception": '',
            "SimpleCNN": 'ressources/Models/SimpleCNNTrained',
            "CNNv2":'ressources/Models/Cnnv2-70batch-50epochs'
        } # different name of models
        self.faceClassifier = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
        self.emotions = ["anger", "disgust", "fear", "happiness", "neutral", "sadness", "surprise"] # emotion labels availables
        self.model= keras.models.load_model(self.availableModels[modelName])

    ## drawFace
    # draw a rectangle around the face detected in an image and uses face area for making predictions regarding the emotion. The detected emotion is displayed on the frame.
    # @param frame: frame sample retrieved from the video capture
    # @param faces: detected feature of the face returned by cv2.CascadeClassifier 
    # @param grayFrame: gray version of the frame
    def drawFace(self, frame, faces, grayFrame):
        for (x, y, w, h) in faces:
            # draw ROI around the face
            cv2.rectangle(frame, (x, y), (x + w, y + h), (255, 0, 0), 2)
            roiGray = grayFrame[y:y + h, x:x + w]
            roiGray = roiGray/255.0
            cropped_img = numpy.expand_dims(numpy.expand_dims(cv2.resize(roiGray, (48, 48)), -1), 0)
            # predict the emotion
            prediction = numpy.argmax(self.model.predict(cropped_img))
            # add text
            cv2.putText(frame, str(self.emotions[prediction]), (x + 6, y - 6), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255, 255, 255), 1)

    ## run
    # Runs the algorithm for predicting the emotion using images from webcam
    def run(self):
        while (self.isRunning):
            if (self._demoMode):
                screenShot = pyautogui.screenshot()
                frame = numpy.array(screenShot)
                frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
            else:
                _, frame = self._cap.read()
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            # detecte faces
            faces = self.faceClassifier.detectMultiScale(gray, 1.3, 5)
            # draw the face
            self.drawFace(frame, faces, gray)
            # display result
            cv2.imshow(self.title, cv2.resize(frame,(1600,960),interpolation = cv2.INTER_CUBIC))
            if (cv2.waitKey(1) & 0xFF == ord('q')):
                self.isRunning = False
