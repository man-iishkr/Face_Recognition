from imutils.video import VideoStream
from datetime import datetime
import pandas as pd
import time
import cv2
import numpy as np
import insightface
from numpy.linalg import norm
from numpy import dot

model = insightface.app.FaceAnalysis(name='buffalo_l')
model.prepare(ctx_id=-1)

embeddings = np.load('embeddings.npy', allow_pickle=True).item()
def load_attendance(path):
    df = pd.read_excel(path,engine='openpyxl')
    return df

def mark_attendance(df, name, marked):
    today = datetime.now().strftime('%Y-%m-%d')
    current = datetime.now().strftime('%H:%M:%S')

    if name in marked or name not in df["NAME"].values:
        return df

    if today not in df.columns:
         df[today] = ""

    row_index = df[df["NAME"] == name].index[0]

    if pd.isna(df.at[row_index, today]) or df.at[row_index, today] == "":
        df.at[row_index, today] = current
        marked.add(name)

    return df


excle=r'C:\Users\manis\Desktop\Atten.xlsx'  


# Cosine similarity function
def cosine_similarity(a, b):
    return dot(a, b) / (norm(a) * norm(b))

# Recognize face
def recognize(embedding, threshold=0.5):
    results = []
    for name, known_emb in embeddings.items():
        sim = cosine_similarity(embedding, known_emb)
        results.append((name, sim))
    results.sort(key=lambda x: x[1], reverse=True)
    best_match = results[0]
    return best_match if best_match[1] > threshold else ("Unknown", best_match[1])
marked = set()
def recognition_loop():
    global running, attendance_df, excel_file_path, marked_today

    cap = cv2.VideoCapture(0)
    model = insightface.app.FaceAnalysis(name='buffalo_l')
    model.prepare(ctx_id=-1)
    embeddings = np.load('embeddings.npy', allow_pickle=True).item()

    while running:
        ret, frame = cap.read()
        if not ret:
            continue
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        faces = model.get(frame_rgb)
        for face in faces:
            if face is None or face.normed_embedding is None:
                continue
            x1, y1, x2, y2 = face.bbox.astype(int)
            embedding = face.normed_embedding
            name, score = recognize_face(embedding, embeddings)
            if name != "Unknown":
                mark_attendance(attendance_df, name,marked)
        time.sleep(0.05)  # small delay to reduce CPU usage

    cap.release()
    if attendance_df is not None and excel_file_path is not None:
        attendance_df.to_excel(excel_file_path, index=False, engine='openpyxl')

def recognize_face(embedding, embeddings, threshold=0.5):
    results = [(name, cosine_similarity(embedding, emb)) for name, emb in embeddings.items()]
    results.sort(key=lambda x: x[1], reverse=True)
    best_match = results[0]
    return best_match if best_match[1] > threshold else ("Unknown", best_match[1])

def stop_recognition():
    global running
    running = False

def get_marked_names():
    return list(marked)