#
# Created on Tue May 11 2021
#
# Arthur Lang
# EmotionDetector.py
#

import cv2
import ntpath
from tensorflow import keras
import numpy

def getFileName(path):
    head, tail = ntpath.split(path)
    return tail or ntpath.basename(head)

class EmotionDetector():

    def __init__(self, path=""):
        self.title = "Emotion detector - "
        self.title += " webcam" if path == "" else getFileName(path)
        self._cap = cv2.VideoCapture(0 if path == "" else path)
        self.isRunning = True
        self.faceClassifier = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
        self.emotions = ["anger","disgust","fear","happiness", "neutral", "sadness","surprise"]
        self.model= keras.models.load_model("src\Models\LoadFromFileMethod")

    def drawFace(self, img, faces,gray):
        for (x, y, w, h) in faces:
            cv2.rectangle(img, (x, y), (x + w, y + h), (255, 0, 0), 2)
            gray=gray/255.0
            gray=numpy.resize(gray,(48,48))
            img_expanded = gray[:, :, numpy.newaxis]
            imgForPrediction=img_expanded[numpy.newaxis,:,:,:]
            # print(imgForPrediction.shape)
            prediction=numpy.argmax(self.model.predict(imgForPrediction))
            print(prediction)
            cv2.putText(img, str(self.emotions[prediction]), (x + 6, y - 6), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255, 255, 255), 1)

    def run(self):
        while (self.isRunning):
            ret, frame = self._cap.read()
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            # print(gray.shape)
            faces = self.faceClassifier.detectMultiScale(gray, 1.3, 5)
            self.drawFace(frame, faces, gray)
            cv2.imshow(self.title, frame)
            if (cv2.waitKey(1) & 0xFF == ord('q')):
                self.isRunning = False
E=EmotionDetector()
E.run()