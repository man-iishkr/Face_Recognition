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
import time
import threading
from queue import Queue, Empty

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

# Face tracking
tracked_faces = {}
next_face_id = 0
face_id_lock = threading.Lock()

# Optimized queue - smaller for faster processing
detection_queue = Queue(maxsize=2)

# Adaptive frame skipping - skip more frames to reduce load
PROCESS_EVERY_N_FRAMES = 3  # Skip more frames for faster processing

# ------------------ Load InsightFace Model (Optimized) ------------------
print("Loading InsightFace model...")
buffalo_model = insightface.app.FaceAnalysis(name='buffalo_l', providers=['CPUExecutionProvider'])
# Smaller detection size = much faster! 320x320 is sweet spot
buffalo_model.prepare(ctx_id=-1, det_thresh=0.5, det_size=(320, 320))
print("Model loaded successfully!")

# ------------------ Helper Functions ------------------
def cosine_similarity(a, b):
    return dot(a, b) / (norm(a) * norm(b))

def recognize_face(embedding, threshold=0.38):
    results = [(name, cosine_similarity(embedding, emb)) for name, emb in embeddings.items()]
    results.sort(key=lambda x: x[1], reverse=True)
    best_match = results[0]
    return best_match if best_match[1] > threshold else ("Unknown", best_match[1])

def load_attendance(path):
    return pd.read_excel(path, engine='openpyxl')

def mark_attendance(df, name):
    today = datetime.now().strftime('%Y-%m-%d')
    current_time = datetime.now().strftime('%H:%M:%S')

    if today not in df.columns:
        df[today] = ""

    if name not in df["NAME"].values:
        return

    row_index = df[df["NAME"] == name].index[0]

    if pd.isna(df.at[row_index, today]) or df.at[row_index, today] == "":
        df.at[row_index, today] = current_time
        df.to_excel(excel_file_path, index=False, engine='openpyxl')
        print(f"✅ {name} marked at {current_time}")
        marked_today.add(name)

def calculate_iou(box1, box2):
    """Fast IOU calculation"""
    x1_1, y1_1, x2_1, y2_1 = box1
    x1_2, y1_2, x2_2, y2_2 = box2
    
    xi1 = max(x1_1, x1_2)
    yi1 = max(y1_1, y1_2)
    xi2 = min(x2_1, x2_2)
    yi2 = min(y2_1, y2_2)
    inter_area = max(0, xi2 - xi1) * max(0, yi2 - yi1)
    
    box1_area = (x2_1 - x1_1) * (y2_1 - y1_1)
    box2_area = (x2_2 - x1_2) * (y2_2 - y1_2)
    union_area = box1_area + box2_area - inter_area
    
    return inter_area / union_area if union_area > 0 else 0

# ------------------ Background Detection & Recognition Thread ------------------
def detection_worker():
    """Highly optimized background thread"""
    global buffalo_model, tracked_faces, attendance_df, next_face_id
    
    print("Detection worker started")
    
    while True:
        try:
            frame_data = detection_queue.get(timeout=0.5)
            
            frame = frame_data['frame']
            frame_id = frame_data['frame_id']
            current_time = time.time()
            
            try:
                start_time = time.time()
                
                # Direct detection - no downsampling (it was actually slower!)
                faces = buffalo_model.get(frame)
                
                if faces:
                    detected_ids = set()
                    
                    with face_id_lock:
                        for idx, face in enumerate(faces):
                            # Get bbox directly - no scaling needed
                            bbox = face.bbox.astype(int)
                            x1, y1, x2, y2 = bbox
                            
                            # Clamp to frame boundaries
                            x1 = max(0, min(x1, frame.shape[1] - 1))
                            y1 = max(0, min(y1, frame.shape[0] - 1))
                            x2 = max(0, min(x2, frame.shape[1]))
                            y2 = max(0, min(y2, frame.shape[0]))
                            
                            bbox_final = (x1, y1, x2, y2)
                            embedding = face.normed_embedding
                            
                            # Fast matching to existing faces
                            matched_id = None
                            best_iou = 0.2
                            
                            for face_id, face_data in tracked_faces.items():
                                iou = calculate_iou(bbox_final, face_data['bbox'])
                                if iou > best_iou:
                                    best_iou = iou
                                    matched_id = face_id
                            
                            if matched_id is not None:
                                # Update existing face
                                tracked_faces[matched_id]['bbox'] = bbox_final
                                tracked_faces[matched_id]['last_seen'] = current_time
                                detected_ids.add(matched_id)
                                
                                # Only re-recognize if not already recognized
                                if not tracked_faces[matched_id].get('recognized', False):
                                    name, score = recognize_face(embedding)
                                    tracked_faces[matched_id]['name'] = name
                                    tracked_faces[matched_id]['score'] = score
                                    tracked_faces[matched_id]['recognized'] = True
                                    
                                    if name != "Unknown" and score > 0.38 and attendance_df is not None:
                                        mark_attendance(attendance_df, name)
                                        
                                    print(f"✓ Face {matched_id}: {name} ({score:.3f})")
                            else:
                                # New face detected
                                new_id = next_face_id
                                next_face_id += 1
                                
                                name, score = recognize_face(embedding)
                                
                                tracked_faces[new_id] = {
                                    'bbox': bbox_final,
                                    'name': name,
                                    'score': score,
                                    'recognized': True,
                                    'last_seen': current_time,
                                    'created': current_time
                                }
                                detected_ids.add(new_id)
                                
                                if name != "Unknown" and score > 0.38 and attendance_df is not None:
                                    mark_attendance(attendance_df, name)
                                    
                                print(f"✓ NEW {new_id}: {name} ({score:.3f})")
                        
                        # Fast cleanup - 1.5 second timeout
                        to_remove = [fid for fid in tracked_faces.keys() 
                                   if fid not in detected_ids and 
                                   (current_time - tracked_faces[fid]['last_seen']) > 1.5]
                        
                        for face_id in to_remove:
                            del tracked_faces[face_id]
                
                else:
                    # No faces - cleanup old ones
                    with face_id_lock:
                        to_remove = [fid for fid, fd in tracked_faces.items() 
                                   if (current_time - fd['last_seen']) > 1.5]
                        for face_id in to_remove:
                            del tracked_faces[face_id]
                
                elapsed = (time.time() - start_time) * 1000
                print(f"Frame {frame_id}: {len(faces) if faces else 0} faces, {elapsed:.0f}ms")
                
            except Exception as e:
                print(f"❌ Error: {e}")
            
            detection_queue.task_done()
            
        except Empty:
            # Periodic cleanup
            with face_id_lock:
                current_time = time.time()
                to_remove = [fid for fid, fd in tracked_faces.items() 
                           if (current_time - fd['last_seen']) > 1.5]
                for face_id in to_remove:
                    del tracked_faces[face_id]
            continue
        except Exception as e:
            print(f"❌ Worker error: {e}")
            time.sleep(0.1)

# Start worker thread
worker_thread = threading.Thread(target=detection_worker, daemon=True)
worker_thread.start()

# ------------------ Frame Generator ------------------
def generate_frames():
    global cap, running, attendance_df, tracked_faces

    if cap is None:
        cap = cv2.VideoCapture(0)
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
        cap.set(cv2.CAP_PROP_FPS, 30)
        cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
        time.sleep(0.5)

    frame_count = 0
    fps_start_time = time.time()
    fps_counter = 0
    current_fps = 0
    last_queue_time = time.time()

    while True:
        ret, frame = cap.read()
        if not ret:
            continue

        frame = cv2.flip(frame, 1)
        frame_count += 1
        fps_counter += 1

        # Calculate FPS
        if time.time() - fps_start_time >= 1.0:
            current_fps = fps_counter / (time.time() - fps_start_time)
            fps_counter = 0
            fps_start_time = time.time()

        if running and attendance_df is not None:
            # Adaptive queuing - queue more often if detection is fast
            current_time = time.time()
            queue_interval = 0.5 if detection_queue.qsize() < 2 else 1.0
            
            if (frame_count % PROCESS_EVERY_N_FRAMES == 0 and 
                not detection_queue.full() and 
                (current_time - last_queue_time) >= queue_interval):
                
                frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                try:
                    detection_queue.put_nowait({
                        'frame': frame_rgb,
                        'frame_id': frame_count
                    })
                    last_queue_time = current_time
                except:
                    pass

            # Removed: Drawing tracked faces (bounding boxes and labels)

        # Status overlay - simplified without face count and queue
        cv2.rectangle(frame, (5, 5), (200, 50), (0, 0, 0), -1)
        cv2.putText(frame, f"FPS: {current_fps:.1f}", (10, 30),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        
        status_text = "ACTIVE" if running else "PAUSED"
        status_color = (0, 255, 0) if running else (0, 0, 255)
        cv2.putText(frame, status_text, (120, 30),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, status_color, 2)

        # Encode with good quality/speed balance
        ret, buffer = cv2.imencode('.jpg', frame, [cv2.IMWRITE_JPEG_QUALITY, 85])
        if not ret:
            continue
            
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
    global running, attendance_df, marked_today, tracked_faces, next_face_id
    if excel_file_path is None:
        return jsonify({"status": "error", "message": "No Excel file uploaded"})
    attendance_df = load_attendance(excel_file_path)
    marked_today = set()
    
    # Load previously marked names from today's column in the Excel file
    today = datetime.now().strftime('%Y-%m-%d')
    if today in attendance_df.columns:
        for idx, row in attendance_df.iterrows():
            if not pd.isna(row[today]) and row[today] != "":
                marked_today.add(row["NAME"])
    
    with face_id_lock:
        tracked_faces = {}
        next_face_id = 0
    # Clear queue
    while not detection_queue.empty():
        try:
            detection_queue.get_nowait()
        except:
            break
    running = True
    print("🚀 Recognition started")
    return jsonify({"status": "started"})

@app.route('/stop', methods=['POST'])
def stop():
    global running
    running = False
    with face_id_lock:
        tracked_faces.clear()
    print("⏸️ Recognition paused")
    return jsonify({"status": "stopped"})

@app.route('/status')
def status():
    return jsonify({"marked": list(marked_today)})

# ------------------ Run Flask ------------------
if __name__ == '__main__':
    print("=" * 60)
    print("Face Recognition Attendance - Speed Optimized")
    print("=" * 60)
    print("Optimizations:")
    print("  • Detection size: 320x320 (faster)")
    print("  • Frame skip: Every 3rd frame")
    print("  • Queue size: 2 (less blocking)")
    print("  • Fast cleanup: 1.5s timeout")
    print("  • Expected: 1500-2000ms per frame")
    print("=" * 60)
    app.run(host='127.0.0.1', port=5000, debug=False, threaded=True)