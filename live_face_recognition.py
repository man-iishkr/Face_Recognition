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

file = load_attendance(excle)
marked = set()
cap = VideoStream(src=0).start()
time.sleep(2.0)
frame_cnt=0

while True:
    frame1 = cap.read()
    frame=cv2.flip(frame1,1)
    if frame is None:
        continue

    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    if frame_cnt % 5 == 0:
        faces = model.get(frame_rgb)

    for face in faces:
        if face is None or face.kps is None or face.normed_embedding is None:
           print("⚠️ Invalid face skipped (no landmarks or embedding)")
           continue
        x1, y1, x2, y2 = face.bbox.astype(int)
        embedding = face.normed_embedding
        name, score = recognize(embedding)
        if name:  
            file= mark_attendance(file, name, marked)

        # Draw box + name
        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
        label = f"{name} ({score:.2f})"
        cv2.putText(frame, label, (x1, y1 - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 0, 0), 2)

    cv2.imshow("Live Face Recognition(Press d to exit)", frame)
    if cv2.waitKey(1) & 0xFF == ord('d') :
        break
file.to_excel(excle, index=False,engine='openpyxl')
print(f"Attendance saved to {excle}")
cap.release()
cv2.destroyAllWindows()
