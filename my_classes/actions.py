from my_classes.utils import set_bounding_box
import my_classes.mouse_utils as mouse
import pydirectinput
import time
import random

def click_inventory(slot):
    b = set_bounding_box()
    x, y = (slot % 4)*42 + 580 + random.uniform(-10,10), (slot // 4)*36 + 250 + random.uniform(-10,10) #580 and 250 inventory offsets
    mouse.click_mouse(x,y)

def drop_item(slot, wait=0):
    pydirectinput.keyDown('shift')
    click_inventory(slot)
    pydirectinput.keyUp('shift')

def click_item(slot, wait=0):
    click_inventory(slot)

def drop_first_x(amount):
    pydirectinput.keyDown('shift')
    for i in range(0,amount):
        click_item(i)
    pydirectinput.keyUp('shift')

def drop_items(items):
    pydirectinput.keyDown('shift')
    for item in items:
        click_item(item)
    pydirectinput.keyUp('shift')

def drop_inv(items, drop_threshold):
    if len(items) >= drop_threshold:
            drop_items(items)
            return random.randint(22,25)
    return random.randint(22,25)