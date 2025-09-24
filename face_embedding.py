
from datetime import datetime
import os
import cv2
import insightface
import numpy as np

model = insightface.app.FaceAnalysis(name='buffalo_l')
model.prepare(ctx_id=-1)


def get_embedding(face):
    emb = model.get(face)
    if emb and len(emb) > 0:
        return emb[0].normed_embedding  
    return None

embeddings = {}
for person in os.listdir(r'C:\Cricket\Facial_Images'):
    emb_list = []
    for img_name in os.listdir(f'C:\Cricket\Facial_Images/{person}'):
        img = cv2.imread(f'C:\Cricket\Facial_Images/{person}/{img_name}')
        emb = get_embedding(img)
        if emb is not None:
            emb_list.append(emb)
    if emb_list:
        embeddings[person] = np.mean(emb_list, axis=0)  
np.save('embeddings.npy', embeddings)
print(" Face embeddings saved to embeddings.npy")