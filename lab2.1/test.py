import cv2
import numpy as np

image = cv2.imread('img.png') 

def find(image, lower_color, upper_color, min_area=500):
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    mask = cv2.inRange(hsv, lower_color, upper_color)
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5,5))
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)

    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    centers = []
    filtered_contours = []
    for cnt in contours:
        area = cv2.contourArea(cnt)
        if area >= min_area:
            M = cv2.moments(cnt)
            if M['m00'] > 1000:
                cx = int(M['m10'] / M['m00'])
                cy = int(M['m01'] / M['m00'])
                centers.append((cx, cy))
                filtered_contours.append(cnt)
    return filtered_contours, centers

hsv_min = np.array((10, 178, 180), np.uint8)
hsv_max = np.array((255, 255, 255), np.uint8)

contours1, centers1 = find(image, hsv_min, hsv_max)
contours2, centers2 = find(image, hsv_min, hsv_max)

contours = contours1 + contours2
centers = centers1 + centers2

for cnt in contours:
    cv2.drawContours(image, [cnt], -1, (0, 255, 0), 2)

for center in centers:
    cv2.circle(image, center, 5, (255, 255, 255), -1)
    cv2.putText(image, 'banana', (center[0] - 20, center[1] - 20),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)

cv2.imshow('Result', image)
cv2.waitKey(0)
cv2.destroyAllWindows()

