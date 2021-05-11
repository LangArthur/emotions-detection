#!/usr/bin/env python3
#
# Created on Sat Apr 17 2021
#
# Arthur Lang
# main.py
#

from src.EmotionDetector import EmotionDetector

def main():
    ed = EmotionDetector()
    ed.run()
    return 0

if __name__ == '__main__':
    main()