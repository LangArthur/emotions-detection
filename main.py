#!/usr/bin/env python3
#
# Created on Sat Apr 17 2021
#
# Arthur Lang
# main.py
#
import sys

from src.EmotionDetector import EmotionDetector

def displayHelp():
    print("Usage: ./main.py modelName")
    print("Model name could not be identified.")

## checkParameters
# check if the paramters are good or not
# @return true or false
def checkParameters(parameters):
	return len(parameters) > 1

def main():
    av = sys.argv
        if (checkParameters(av)):
            ed = EmotionDetector(modelName=av[1])
            ed.run()
        else:
            displayHelp()
        return 0
    
    return 0

if __name__ == '__main__':
    main()