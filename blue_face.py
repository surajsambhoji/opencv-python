import cv2

face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()

    faces = face_cascade.detectMultiScale(frame, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))

    for (x, y, w, h) in faces:
        face = frame[y:y+h, x:x+w]
        blurred_face = cv2.GaussianBlur(face, (99, 99), 0)
        frame[y:y+h, x:x+w] = blurred_face

    cv2.imshow('Blurred Faces', frame)

    if cv2.waitKey(1) == 13:  
        break

cap.release()
cv2.destroyAllWindows()
