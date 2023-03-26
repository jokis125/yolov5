import mss

def init():
    global detections
    global objects
    global boxes
    global sct
    global window_name

    detections = []
    boxes = []
    objects = []
    sct = mss.mss()
    window_name = 'RuneLite - Hell Warrior'