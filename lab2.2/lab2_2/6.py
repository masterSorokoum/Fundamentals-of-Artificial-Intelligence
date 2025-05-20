import cv2

cascade_face = cv2.CascadeClassifier(cv2.data.haarcascades + \
                                     'haarcascade_frontalface_default.xml')

cascade_eye = cv2.CascadeClassifier(cv2.data.haarcascades + \
                                    'haarcascade_eye.xml')

cascade_smile = cv2.CascadeClassifier(cv2.data.haarcascades + \
                                    'haarcascade_smile.xml')

img = cv2.imread('2.png')

gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

faces = cascade_face.detectMultiScale(gray, scaleFactor = 1.3, minNeighbors = 5)


for (x,y,w,h) in faces:
    cv2.rectangle(img, (x,y), (x+w,y+h), (0,0,255), 2)
    
    cv2.putText(img, 'Faces: ' + str(len(faces)), (20,40),
                cv2.FONT_HERSHEY_SIMPLEX, 1, (0,0,0), 2,
                cv2.LINE_AA)
    roi_color = img[y:y+h, x:x+w]
    roi_gray = gray[y:y+h, x:x+w]

    eyes = cascade_eye.detectMultiScale(roi_gray, 1.8, minNeighbors = 3)
    smile = cascade_smile.detectMultiScale(roi_gray, 1.3, minNeighbors = 8)

    for ex,ey,ew,eh in smile:
        cv2.rectangle(roi_color, (ex,ey), (ex+ew, ey+eh), (255,0,0), 2)
    
    for ex,ey,ew,eh in eyes:
        cv2.rectangle(roi_color, (ex,ey), (ex+ew, ey+eh), (0,255,0), 2)



cv2.imshow("Image", img)
cv2.waitKey(0)
cv2.destroyAllWindows()

