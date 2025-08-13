import os
import cv2 as cv
import numpy as np


p=[]
for i in os.listdir(r'C:\Cricket\Face_recognition\Face_Recognition\Facial_Images'):
    p.append(i)

features=[]
labels=[]
dir=r'C:\Cricket\Face_recognition\Face_Recognition\Facial_Images'
har_casscade = cv.CascadeClassifier('haar_cascade.xml')

def create_train():
    for i in p:
        path=os.path.join(dir,i)
        label=p.index(i)

        for img in os.listdir(path):
            img_path=os.path.join(path,img)
            img_array=cv.imread(img_path)
            gray=cv.cvtColor(img_array,cv.COLOR_BGR2GRAY)
            faces_rect=har_casscade.detectMultiScale(gray,scaleFactor=1.1,minNeighbors=4)
            for (x,y,w,h) in faces_rect:
                face_roi=gray[y:y+h,x:x+w]
                features.append(face_roi)
                labels.append(label)
create_train()
print('Training Done------------------------------------')

features=np.array(features,dtype='object')
labels=np.array(labels)

#train the model
face_recognizer=cv.face.LBPHFaceRecognizer_create()
face_recognizer.train(features,labels)

# Save the trained model
face_recognizer.save('face_trained.yml')


np.save('features.npy',features)
np.save('labels.npy',labels)