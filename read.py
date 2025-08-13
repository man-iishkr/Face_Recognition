import cv2 as cv

#Reading Video
capture = cv.VideoCapture(0)
har_casscade = cv.CascadeClassifier('haar_cascade.xml')

while True:
    isTrue, frame = capture.read()
    flipped_frame = cv.flip(frame, 1) 
    face_rect = har_casscade.detectMultiScale(flipped_frame, scaleFactor=1.1, minNeighbors=5)

    for (x, y, w, h) in face_rect:
        cv.rectangle(flipped_frame, (x, y), (x + w, y + h), (0, 255, 0), thickness=2)
        
    cv.imshow('Video', flipped_frame)


    if cv.waitKey(20) & 0xFF == ord('d'):
        break

capture.release()
cv.destroyAllWindows()