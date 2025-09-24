from flask import Flask, render_template, request, jsonify, Response
import cv2
import pandas as pd
from datetime import datetime
import numpy as np
import insightface
from numpy.linalg import norm
from numpy import dot
import os
from werkzeug.utils import secure_filename

# ------------------ Flask Setup ------------------
app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = 'uploads'
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)

# ------------------ Global Variables ------------------
cap = None
running = False
excel_file_path = None
attendance_df = None
embeddings = np.load('embeddings.npy', allow_pickle=True).item()
marked_today = set()
# ------------------ Load InsightFace Model ------------------
model = insightface.app.FaceAnalysis(name='buffalo_l')
model.prepare(ctx_id=-1)

# ------------------ Helper Functions ------------------
def cosine_similarity(a, b):
    return dot(a, b) / (norm(a) * norm(b))

def recognize_face(embedding, threshold=0.5):
    results = [(name, cosine_similarity(embedding, emb)) for name, emb in embeddings.items()]
    results.sort(key=lambda x: x[1], reverse=True)
    best_match = results[0]
    return best_match if best_match[1] > threshold else ("Unknown", best_match[1])

def load_attendance(path):
    return pd.read_excel(path, engine='openpyxl')

def mark_attendance(df, name):
    """Mark attendance for a person only once per day."""
    today = datetime.now().strftime('%Y-%m-%d')
    current_time = datetime.now().strftime('%H:%M:%S')

    # Ensure today column exists
    if today not in df.columns:
        df[today] = ""

    if name not in df["NAME"].values:
        return

    row_index = df[df["NAME"] == name].index[0]

    # Only mark if empty
    if pd.isna(df.at[row_index, today]) or df.at[row_index, today] == "":
        df.at[row_index, today] = current_time
        df.to_excel(excel_file_path, index=False, engine='openpyxl')
        print(f"✅ {name} marked at {current_time}")

# ------------------ Video Streaming ------------------
def generate_frames():
    global cap, running, attendance_df, excel_file_path

    if cap is None:
        cap = cv2.VideoCapture(0)

    while True:
        success, frame = cap.read()
        if not success:
            break

        frame = cv2.flip(frame, 1)
        if running and attendance_df is not None:
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            faces = model.get(frame_rgb)
            for face in faces:
                if face is None or face.normed_embedding is None:
                    continue
                x1, y1, x2, y2 = face.bbox.astype(int)
                embedding = face.normed_embedding
                name, score = recognize_face(embedding)
                if name != "Unknown":
                    mark_attendance(attendance_df, name)
                    marked_today.add(name)

                # Draw bounding box + label
                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                label = f"{name} ({score:.2f})"
                cv2.putText(frame, label, (x1, y1 - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 0, 0), 2)

        ret, buffer = cv2.imencode('.jpg', frame)
        frame_bytes = buffer.tobytes()
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n')

# ------------------ Flask Routes ------------------
@app.route('/')
def index():
    return render_template('index.html')

@app.route('/video_feed')
def video_feed():
    return Response(generate_frames(),
                    mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/upload', methods=['POST'])
def upload_file():
    global excel_file_path
    file = request.files['file']
    if file and file.filename.endswith('.xlsx'):
        filename = secure_filename(file.filename)
        path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(path)
        excel_file_path = path
        return jsonify({"status": "success", "path": path})
    return jsonify({"status": "error", "message": "Invalid file"})

@app.route('/start', methods=['POST'])
def start():
    global running, attendance_df
    if excel_file_path is None:
        return jsonify({"status": "error", "message": "No Excel file uploaded"})
    attendance_df = load_attendance(excel_file_path)
    running = True
    return jsonify({"status": "started"})

@app.route('/stop', methods=['POST'])
def stop():
    global running
    running = False
    return jsonify({"status": "stopped"})
@app.route('/status')
def status():
    return jsonify({"marked": list(marked_today)})

# ------------------ Run Flask ------------------
if __name__ == '__main__':
    app.run(host='127.0.0.1', port=5000, debug=True)
