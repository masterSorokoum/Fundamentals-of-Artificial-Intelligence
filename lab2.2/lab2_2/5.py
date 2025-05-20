import cv2

cascade_face = cv2.CascadeClassifier(cv2.data.haarcascades + \
                                     'haarcascade_frontalface_default.xml')

cascade_eye = cv2.CascadeClassifier(cv2.data.haarcascades + \
                                    'haarcascade_eye.xml')

cascade_smile = cv2.CascadeClassifier(cv2.data.haarcascades + \
                                    'haarcascade_smile.xml')


cap = cv2.VideoCapture(0)

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    faces = cascade_face.detectMultiScale(gray, scaleFactor = 1.3, minNeighbors = 5)


    for (x,y,w,h) in faces:
        cv2.rectangle(frame, (x,y), (x+w,y+h), (0,0,255), 2)
        
        roi_color = frame[y:y+h, x:x+w]
        roi_gray = gray[y:y+h, x:x+w]

        eyes = cascade_eye.detectMultiScale(roi_gray, 1.8, minNeighbors = 3)
        smile = cascade_smile.detectMultiScale(roi_gray, 1.1, minNeighbors = 8)

        for ex,ey,ew,eh in eyes:
            cv2.rectangle(roi_color, (ex,ey), (ex+ew, ey+eh), (0,255,0), 2)

        for ex,ey,ew,eh in eyes:
            cv2.rectangle(roi_color, (ex,ey), (ex+ew, ey+eh), (0,255,0), 2)


    cv2.imshow("Image", frame)
    if cv2.waitKey(1) == 27:
        break
    
cv2.destroyAllWindows()
cap.release()

