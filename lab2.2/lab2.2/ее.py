import cv2

cv2.namedWindow('Face detect', cv2.WINDOW_NORMAL)
cascade_face = r'haarcascade_frontalface_default.xml'
face_cascade = cv2.CascadeClassifier(cascade_face)
cascade_eye = r'haarcascade_eye.xml'
eye_cascade = cv2.CascadeClassifier(cascade_eye)
cascade_smile= r'haarcascade_smile.xml'
smile_cascade = cv2.CascadeClassifier(cascade_smile)

frame = cv2.imread('group.jpg')
gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1,
                                            minNeighbors=5)
for x, y, w, h in faces:
        cv2.rectangle(frame, (x,y), (x+w,y+h), (255,0,0), 2)
        roi_color = frame[y:y+h, x:x+w] 
        roi_gray = gray[y:y+h, x:x+w]
        text = f' Faces: {len(faces)}'
        eyes = eye_cascade.detectMultiScale(roi_gray, scaleFactor=1.23, minNeighbors=3, minSize=(35, 35)) 
        cv2.putText(frame, text , (20,40), cv2.FONT_HERSHEY_SIMPLEX, 1, (0,0,255), 2, cv2.LINE_AA)
        for ex, ey, ew, eh in eyes:
            cv2.rectangle(roi_color, (ex,ey), (ex+ew, ey+eh), (0,255,0), 2)

        smile = smile_cascade.detectMultiScale(roi_gray, scaleFactor=1.7, minNeighbors=8, minSize=(90, 70))
        for sx, sy, sw, sh in smile:
            cv2.rectangle(roi_color, (sx,sy), (sx+sw, sy+sh), (0,0,255), 2)        

cv2.imshow('Face detect', frame) 
cv2.waitKey()
cv2.destroyAllWindows()
