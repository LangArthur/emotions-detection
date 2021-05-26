#!/usr/bin/env python3
#
# Created on Sat Apr 17 2021
#
# Arthur Lang
# main.py
#

import sys
import argparse

from src.EmotionDetector import EmotionDetector

def main():
    parser = argparse.ArgumentParser(description="Run emotion detection on stream of image. Different input can be take.")
    # add flag gestion
    parser.add_argument("-s", "--screen", help="use the screen as input", action="store_true")
    parser.add_argument("-f", "--file", help="Specify a file as input dispite of the webcam.", type=str, default="")
    parser.add_argument("-m", "--model", help="specify a model to use. Model availables are Inception, SimpleCNN and CNNv2.", type=str, choices=["Inception", "SimpleCNN", "CNNv2"], default="CNNv2")
    args = parser.parse_args()
    if args.file and args.screen:
        print("{}: error: -s and -f are incompatible".format(__file__))
        return (84)
    elif args.file:
        ed = EmotionDetector(modelName=args.model, path=args.file)
    elif args.screen:
        ed = EmotionDetector(modelName=args.model, demo=True)
    else:
        ed = EmotionDetector(modelName=args.model)
    ed.run()
    return 0
    

if __name__ == '__main__':
    main()