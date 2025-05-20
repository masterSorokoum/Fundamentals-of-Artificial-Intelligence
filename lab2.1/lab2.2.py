import cv2
import numpy as np
cv2.namedWindow('RESULT', cv2.WINDOW_NORMAL) 

hsv_min = np.array((10, 178, 180), np.uint8)
hsv_max = np.array((255, 255, 255), np.uint8)

img = cv2.imread('img.png')
hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
filter_color = cv2.inRange(hsv, hsv_min, hsv_max)
moments = cv2.moments(filter_color)

m01 = moments['m01']
m10 = moments['m10']
m00 = moments['m00']
x = int(m10 / m00)
y = int(m01 / m00)

cv2.circle(img, (x, y), 10, (255,255,255), -1)
cv2.putText(img, 'Yellow object', (x+20,y),
            cv2.FONT_HERSHEY_SIMPLEX, 0.75, (255,255,255), 2, cv2.LINE_AA)

contours, hierarchy = cv2.findContours(filter_color, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
cv2.drawContours(img, contours, -1, (0,255,0), 2, cv2.LINE_AA, hierarchy, 1) 

cv2.imshow('RESULT', img)
cv2.waitKey(0)
cv2.destroyAllWindows()

