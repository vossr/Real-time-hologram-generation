from mss import mss
import numpy as np
import cv2

sct = mss()
monitor_number = 2
monitor = sct.monitors[monitor_number]

rect_x, rect_y = 400, 50
rect_width, rect_height = 300, 400
dragging = False
resizing = False
action = None
offsetX, offsetY = 0, 0

def mouse_event(event, x, y, flags, param):
    global rect_x, rect_y, rect_width, rect_height, dragging, resizing, offsetX, offsetY, action

    if (rect_x + rect_width - 10 < x < rect_x + rect_width + 10) and \
       (rect_y + rect_height - 10 < y < rect_y + rect_height + 10):
        if event == cv2.EVENT_LBUTTONDOWN:
            resizing = True
            offsetX, offsetY = x - rect_x, y - rect_y
            action = 'resize'
    elif rect_x < x < rect_x + rect_width and rect_y < y < rect_y + rect_height:
        if event == cv2.EVENT_LBUTTONDOWN:
            dragging = True
            offsetX, offsetY = x - rect_x, y - rect_y
            action = 'drag'

    if event == cv2.EVENT_MOUSEMOVE:
        if dragging and action == 'drag':
            rect_x, rect_y = x - offsetX, y - offsetY
        elif resizing and action == 'resize':
            rect_width = x - rect_x
            rect_height = int((4/3) * rect_width)

    if event == cv2.EVENT_LBUTTONUP:
        dragging = False
        resizing = False
        action = None

display_3_x_position = -1920
display_3_y_position = 0
cv2.namedWindow("display_cap")
cv2.setWindowProperty("display_cap", cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)
cv2.setMouseCallback("display_cap", mouse_event)
cv2.moveWindow("display_cap", display_3_x_position, display_3_y_position)

def get_rect():
    og_frame = np.array(sct.grab(monitor))
    frame = og_frame
    frame = cv2.cvtColor(frame, cv2.COLOR_BGRA2BGR)
    cv2.rectangle(frame, (rect_x, rect_y), (rect_x + rect_width, rect_y + rect_height), (0, 0, 255), 2)
    cv2.imshow('display_cap', frame)
    cropped = og_frame[rect_y:rect_y+rect_height, rect_x:rect_x+rect_width]
    return cropped
