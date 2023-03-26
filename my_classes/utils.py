import win32gui
import numpy as np
import math
import time
import os
import cv2 as cv
import numpy as np
import my_classes.data as data


def set_bounding_box(offsettop = 0, offsetleft = 0, offsetx = 0, offsety = 0):
    hwnd = win32gui.FindWindow(None, data.window_name)
    rect = win32gui.GetWindowRect(hwnd)
    x = rect[0]
    y = rect[1]
    w = rect[2] - x
    h = rect[3] - y 
    bounding_box = {'top': y+offsettop, 'left': x+offsetleft, 'width': w-offsetx, 'height': h-offsety}
    return bounding_box


def totupleutil(a):
    try:
        return tuple(totupleutil(i) for i in a)
    except TypeError:
        return a

def get_centroid(norfair_box):
    x1 = norfair_box.estimate[0][0]
    y1 = norfair_box.estimate[0][1]
    x2 = norfair_box.estimate[1][0]
    y2 = norfair_box.estimate[1][1]
    return ((x1 + x2) / 2, (y1 + y2) / 2)

def to_tuple(a):
    return totupleutil(a)[0]

def sort_by_closeness(objs):
    tpl = []
    for obj in objs:
        dist = math.hypot(250 - obj.xy[0], 200 - obj.xy[1])
        tpl.append((obj, dist))
        
    tpl.sort(key=lambda x: x[1])
    arr = []
    arr = [t[0] for t in tpl]
    try:
        return arr[0]
    except:
        return []


def match_img(frame, img, threshold = 0.8):
    template = cv.imread(img, 0)
    w, h = template.shape[::-1]
    frame = frame.astype(np.uint8)
    frame = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)

    #x, y = (slot % 4)*42 + 580, (slot // 4)*36 + 250
    res = cv.matchTemplate(frame, template, cv.TM_CCOEFF_NORMED)
    loc = np.where(res >= threshold)
    for pt in zip(*loc[::-1]):
        cv.rectangle(frame, pt, (pt[0] + w, pt[1] + h), (0,0,255), 2)
    cv.imwrite('res.png', frame)
    return loc

def match_items(frame, item="logs", threshold = 0.8):
    template = cv.imread(os.path.join("img", "items", item + ".png"), 0)
    w, h = template.shape[::-1]
    frame = frame.astype(np.uint8)
    frame = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)

    #x, y = (slot % 4)*42 + 580, (slot // 4)*36 + 250
    res = cv.matchTemplate(frame, template, cv.TM_CCOEFF_NORMED)
    loc = np.where(res >= threshold)
    slots = []
    for pt in zip(*loc[::-1]): #620 260 pt[0]+w pt[1]+h
        slots.append(coords_to_items_slots(pt[0]+w, pt[1]+h))
    return slots

def detect_lvl(frame):
    return (len(match_img(frame, os.path.join("img", "misc", "lvlup.png"))[0])) != 0 

def coords_to_items_slots(x,y):
        slotx = ((x - 580 - 10) //42) % 4
        sloty = ((y - 250 - 15) //36)
        return ((sloty * 4) + slotx)

def wait_for_detections(objects):
    while len(objects) == 0:
        print(objects)
        time.sleep(1)

def get_closest_detection(objects):
    distances = sort_by_closeness(objects) # [0] = id [1] = name [2] = coords
    return distances

def filter_tracked_objects(objects, keep_in_array):
    arr = []
    for thing in keep_in_array:
        arr = [obj for obj in objects if thing in obj.name]
    return arr

def get_frame():
    sct_img = data.sct.grab(set_bounding_box())
    img_np = np.array(sct_img)
    img_np = cv.cvtColor(img_np, cv.COLOR_BGRA2BGR)
    return img_np

def check_if_object_is_gone(closest, objects):
    for obj in data.objects:
        if obj.id == closest.id:
            return True
    return False

def letterbox(img, new_shape=(640, 640), color=(114, 114, 114), auto=True, scaleFill=False, scaleup=True):
    # Resize image to a 32-pixel-multiple rectangle https://github.com/ultralytics/yolov3/issues/232
    shape = img.shape[:2]  # current shape [height, width]
    if isinstance(new_shape, int):
        new_shape = (new_shape, new_shape)

    # Scale ratio (new / old)
    r = min(new_shape[0] / shape[0], new_shape[1] / shape[1])
    if not scaleup:  # only scale down, do not scale up (for better test mAP)
        r = min(r, 1.0)

    # Compute padding
    ratio = r, r  # width, height ratios
    new_unpad = int(round(shape[1] * r)), int(round(shape[0] * r))
    dw, dh = new_shape[1] - new_unpad[0], new_shape[0] - new_unpad[1]  # wh padding
    if auto:  # minimum rectangle
        dw, dh = np.mod(dw, 32), np.mod(dh, 32)  # wh padding
    elif scaleFill:  # stretch
        dw, dh = 0.0, 0.0
        new_unpad = (new_shape[1], new_shape[0])
        ratio = new_shape[1] / shape[1], new_shape[0] / shape[0]  # width, height ratios

    dw /= 2  # divide padding into 2 sides
    dh /= 2

    if shape[::-1] != new_unpad:  # resize
        img = cv.resize(img, new_unpad, interpolation=cv.INTER_LINEAR)
    top, bottom = int(round(dh - 0.1)), int(round(dh + 0.1))
    left, right = int(round(dw - 0.1)), int(round(dw + 0.1))
    img = cv.copyMakeBorder(img, top, bottom, left, right, cv.BORDER_CONSTANT, value=color)  # add border
    return img, ratio, (dw, dh)


