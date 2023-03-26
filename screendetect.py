from typing import List
import mss
import mss.tools
import argparse
import argparse

from ctypes import windll
import win32gui


import my_classes.data as data
import my_classes.norfair_distances as norfair_distances
from ObjectDetector import ObjectDetector
#import my_classes.mouse_utils as mut

import norfair
from norfair import Detection, Tracker, draw_tracked_boxes, draw_tracked_objects
    
def parse_args():
    parser = argparse.ArgumentParser(description="")
    parser.add_argument('--inference', dest='inference', action='store_true', default=True)
    parser.add_argument('--model', dest='model', default='best.pt')
    parser.add_argument('--confidence', dest='confidence', default='0.5')
    parser.add_argument('--boxes', dest='boxes', action='store_true', default=False)
    parser.add_argument('--no_track', dest='no_track', action='store_true', default=False)
    parser.add_argument('--cpu', dest='cpu', action='store_true', default=False)
    return parser.parse_args()

def main(args):
    object_detector = ObjectDetector(args)
    object_detector.detect_screen()


if __name__ == '__main__':
    args = parse_args()
    sct = mss.mss()
    data.init()

    handle = win32gui.FindWindow(None, data.window_name)
    win32gui.SetForegroundWindow(handle)

    main(args)
