import cv2 as cv
import numpy as np
import os

p=[]
for i in os.listdir(r'C:\Cricket\Face_recognition\Face_Recognition\Facial_Images'):
    p.append(i)
har_casscade = cv.CascadeClassifier('haar_cascade.xml')

features = np.load('features.npy',allow_pickle=True)
labels = np.load('labels.npy',allow_pickle=True)
face_recognizer = cv.face.LBPHFaceRecognizer_create()
face_recognizer.read('face_trained.yml')

#Reading Video
capture = cv.VideoCapture(0)
while True:
    isTrue, frame = capture.read()
    flipped_frame = cv.flip(frame, 1)
    gray= cv.cvtColor(flipped_frame, cv.COLOR_BGR2GRAY) 
    face_rect = har_casscade.detectMultiScale(flipped_frame, scaleFactor=1.1, minNeighbors=7)

    for (x, y, w, h) in face_rect:
        cv.rectangle(flipped_frame, (x, y), (x + w, y + h), (0, 255, 0), thickness=2)
        labels, confidence = face_recognizer.predict(gray[y:y+h, x:x+w])
        print(f'Label: {p[labels]}, Confidence: {confidence}')
        cv.putText(flipped_frame, f'{p[labels]} - {confidence:.2f}', (x, y - 10), cv.FONT_HERSHEY_SIMPLEX, 0.7, (255, 0, 0),2)
        
    
    cv.imshow('Video', flipped_frame)


    if cv.waitKey(20) & 0xFF == ord('d') :
        break

capture.release()
cv.destroyAllWindows()