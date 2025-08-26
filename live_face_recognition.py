
import cv2
import numpy as np
import insightface
from numpy.linalg import norm
from numpy import dot

model = insightface.app.FaceAnalysis(name='buffalo_l')
model.prepare(ctx_id=-1)

embeddings = np.load('embeddings.npy', allow_pickle=True).item()

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


cap = cv2.VideoCapture(0)

while True:
    ret, frame1 = cap.read()
    frame=cv2.flip(frame1,1)
    if not ret:
        break

    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    faces = model.get(frame_rgb)

    for face in faces:
        x1, y1, x2, y2 = face.bbox.astype(int)
        embedding = face.normed_embedding
        name, score = recognize(embedding)

        # Draw box + name
        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
        label = f"{name} ({score:.2f})"
        cv2.putText(frame, label, (x1, y1 - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 0, 0), 2)

    cv2.imshow("Live Face Recognition", frame)
    if cv2.waitKey(1) & 0xFF == ord('d'):
        break

cap.release()
cv2.destroyAllWindows()
