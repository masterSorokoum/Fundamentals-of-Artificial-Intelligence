import cv2
import imutils

img = cv2.imread('corner.jpg')
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
blur = cv2.blur(img, (2,2)) 
canny = cv2.Canny(blur, 20, 100)

cv2.imshow('Canny', canny)

contours = cv2.findContours(canny, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
contours = imutils.grab_contours(contours)

for c in contours:
    p = cv2.arcLength(c, True)
    approx = cv2.approxPolyDP(c, 0.02*p, True)
    if len(approx) > 6:
        cv2.drawContours(img, [c], -1, (0,255,0), 2)
    if len(approx) == 3:
        cv2.drawContours(img, [approx], -1, (255, 255, 0), 2)
    if len(approx) == 4:
        cv2.drawContours(img, [approx], -1, (255, 0, 0), 2)
    if len(approx) == 5:
        cv2.drawContours(img, [approx], -1, (255, 0, 255), 2)
        
cv2.imshow('corner_finder', img)
cv2.waitKey(0)
cv2.destroyAllWindows()
