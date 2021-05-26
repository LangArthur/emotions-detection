#!/usr/bin/env python3
#
# Created on Sat Apr 17 2021
#
# Arthur Lang
# main.py
#
import sys

from src.EmotionDetector import EmotionDetector

## checkParameters
# check if the paramters are good or not
# @return true or false
def checkParameters(parameters):
	return len(parameters) > 1

def main():
    av = sys.argv
    if (checkParameters(av)):
        path = av[2] if len(av) == 3 else ""
        ed = EmotionDetector(modelName=av[1], path=path, demo=True)
        ed.run()
    return 0
    

if __name__ == '__main__':
    main()