import cv2

cascade_face = cv2.CascadeClassifier(cv2.data.haarcascades + \
                                     'haarcascade_frontalface_default.xml')

cascade_eye = cv2.CascadeClassifier(cv2.data.haarcascades + \
                                    'haarcascade_eye.xml')

img = cv2.imread('4.jpg')

gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

faces = cascade_face.detectMultiScale(gray, scaleFactor = 1.3, minNeighbors = 5)


for (x,y,w,h) in faces:
    cv2.rectangle(img, (x,y), (x+w,y+h), (0,0,255), 2)
    
    roi_color = img[y:y+h, x:x+w]
    roi_gray = gray[y:y+h, x:x+w]

    eyes = cascade_eye.detectMultiScale(roi_gray, 1.1, minNeighbors = 3)

    for ex,ey,ew,eh in eyes:
        cv2.rectangle(roi_color, (ex,ey), (ex+ew, ey+eh), (0,255,0), 2)







cv2.imshow("Image", img)
cv2.waitKey(0)
cv2.destroyAllWindows()

