import cv2

cv2.namedWindow('Number detect', cv2.WINDOW_NORMAL)
cascade_number = r'haarcascade_russian_plate_number.xml'
number_cascade = cv2.CascadeClassifier(cascade_number)

img = cv2.imread('number.png')
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

number = number_cascade.detectMultiScale(gray, scaleFactor=1.1,
                                            minNeighbors=5)

for x, y, w, h in number:
    cv2.rectangle(img, (x,y), (x+w,y+h), (255,255,0), 2)
    
cv2.imshow('Number detect', img) 
cv2.waitKey(0) 
cv2.destroyAllWindows()

