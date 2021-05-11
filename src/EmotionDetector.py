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

    def run(self):
        while (self.isRunning):
            ret, frame = self._cap.read()
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            cv2.imshow(self.title, gray)
            if (cv2.waitKey(1) & 0xFF == ord('q')):
                self.isRunning = False
