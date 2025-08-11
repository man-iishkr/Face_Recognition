import cv2 as cv
#Reading Image
#img=cv.imread("C:\\Users\\manis\\Pictures\\Camera Roll\\WIN_20241003_02_05_12_Pro.jpg")

#cv.imshow('Street',img)
#cv.waitKey(0)

#Reading Video
capture = cv.VideoCapture(0)

while True:
    isTrue, frame = capture.read()
    cv.imshow('Video', frame)

    if cv.waitKey(20) & 0xFF == ord('d'):
        break

capture.release()
cv.destroyAllWindows()