import cv2
import numpy as np

cv2.namedWindow('HSV', cv2.WINDOW_NORMAL) 
cv2.namedWindow('Result', cv2.WINDOW_NORMAL) 

img = cv2.imread('img.png')
hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)

def callback(*arg):
    pass
cv2.createTrackbar('h1', 'HSV', 0, 255, callback)
cv2.createTrackbar('s1', 'HSV', 0, 255, callback)
cv2.createTrackbar('v1', 'HSV', 0, 255, callback)
cv2.createTrackbar('h2', 'HSV', 255, 255, callback)
cv2.createTrackbar('s2', 'HSV', 255, 255, callback)
cv2.createTrackbar('v2', 'HSV', 255, 255, callback)

while True:
    h1 = cv2.getTrackbarPos('h1', 'HSV')
    s1 = cv2.getTrackbarPos('s1', 'HSV')
    v1 = cv2.getTrackbarPos('v1', 'HSV')
    h2 = cv2.getTrackbarPos('h2', 'HSV')
    s2 = cv2.getTrackbarPos('s2', 'HSV')
    v2 = cv2.getTrackbarPos('v2', 'HSV')

    hsv_min = np.array((h1, s1, v1), np.uint8)
    hsv_max = np.array((h2, s2, v2), np.uint8)
    filter_color = cv2.inRange(hsv, hsv_min, hsv_max)
    cv2.imshow('Result', filter_color)
    
    if cv2.waitKey(1) == 27:
        break
cv2.destroyAllWindows()

