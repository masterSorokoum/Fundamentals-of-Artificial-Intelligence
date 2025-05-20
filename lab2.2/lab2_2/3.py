import cv2

cascade_number = cv2.CascadeClassifier(cv2.data.haarcascades + \
                                     'haarcascade_russian_plate_number.xml')


number = cv2.imread('3.jpg')

gray = cv2.cvtColor(number, cv2.COLOR_BGR2GRAY)

faces = cascade_number.detectMultiScale(gray, scaleFactor = 1.3, minNeighbors = 7)


for (x,y,w,h) in faces:
    cv2.rectangle(number, (x,y), (x+w,y+h), (0,0,255), 2)
    


cv2.imshow("Number", number)
cv2.waitKey(0)
cv2.destroyAllWindows()
