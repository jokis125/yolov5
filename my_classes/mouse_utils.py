from my_classes.utils import set_bounding_box
import pyautogui
from random import uniform
import time
from pyHM import mouse

def move_mouse(xy):
    b = set_bounding_box()
    x, y = xy
    #dur = uniform(0.1, 0.3)
    mouse.move(b['left']+int(x), b['top']+int(y), multiplier=uniform(0.1,1))
    #human_input.human_move_to((b['left']+int(x), b['top']+int(y)))

def click_mouse(x,y):
    b = set_bounding_box()
    #human_input.human_click((b['left']+int(x), b['top']+int(y)))