#
# Created on Tue May 11 2021
#
# Arthur Lang
# EmotionDetector.py
#

import cv2
import ntpath

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

    def drawFace(self, img, faces):
        for (x, y, w, h) in faces:
            gray = cv2.rectangle(img, (x, y), (x + w, y + h), (255, 0, 0), 2)

    def run(self):
        while (self.isRunning):
            ret, frame = self._cap.read()
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            faces = self.faceClassifier.detectMultiScale(gray, 1.3, 5)
            self.drawFace(frame, faces)
            cv2.imshow(self.title, frame)
            if (cv2.waitKey(1) & 0xFF == ord('q')):
                self.isRunning = False
