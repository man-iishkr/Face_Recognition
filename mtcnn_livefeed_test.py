import cv2 as cv
from mtcnn import MTCNN


detector = MTCNN()


capture = cv.VideoCapture(0)

if not capture.isOpened():
    raise RuntimeError("Webcam could not be opened.")

while True:
    isTrue, frame = capture.read()
    if not isTrue:
        print("Failed to grab frame.")
        break

    
    flipped_frame = cv.flip(frame, 1)
    img_rgb = cv.cvtColor(flipped_frame, cv.COLOR_BGR2RGB)


    results = detector.detect_faces(img_rgb)

   
    for face in results:
        x, y, width, height = face['box']
        keypoints = face['keypoints']

        
        cv.rectangle(flipped_frame, (x, y), (x + width, y + height), (0, 155, 255), 2)

        
        for key, point in keypoints.items():
            cv.circle(flipped_frame, point, 2, (0, 255, 0), 0)

    
    cv.imshow("Live Face Detection (Press 'd' to quit)", flipped_frame)

    
    if cv.waitKey(1) & 0xFF == ord('d'):
        break


capture.release()
cv.destroyAllWindows()
