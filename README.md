# Face Recognition Attendance Tracker (WIP)

This project is a **face recognition system** intended to serve as the foundation for a future **attendance tracking solution**. Currently, the application performs real-time face detection and recognition using a webcam feed, and identifies individuals based on a pre-trained dataset.

## ✨ Features

- Real-time **face detection** using OpenCV and InsightFace.
- **Face recognition** using embeddings from the **Buffalo model** (InsightFace).
- Utilizes **cosine similarity** for accurate face matching.
- Displays names with **similarity score** for known individuals.
- Labels unknown faces as **"Unknown"**.
- Draws bounding boxes around detected faces in the live feed.

## 👨‍💻 Technologies Used

| Library       | Purpose                                      |
|---------------|----------------------------------------------|
| `OpenCV`      | Capturing webcam feed and face detection     |
| `InsightFace` | Deep learning face recognition (Buffalo model) |
| `NumPy`       | Face embeddings handling and similarity checks |

## 📁 Dataset

The model is trained on a small sample dataset containing facial images of the following individuals:

- Vladimir Putin
- Narendra Modi
- Donald Trump
- Xi Jinping

> 📸 These images are stored locally and used to generate embeddings for recognition.

## 🧠 Model Details

- **Face Detection:** InsightFace and OpenCV (Haar Cascade fallback if needed)
- **Embedding Model:** `Buffalo_S` model from InsightFace
- **Similarity Metric:** Cosine Similarity

## 🖥️ How It Works

1. The webcam feed is started using OpenCV.
2. Faces are detected frame-by-frame.
3. Embeddings are computed for each detected face.
4. Cosine similarity is used to compare embeddings with the known dataset.
5. If similarity is above threshold, name and score are displayed.
6. Otherwise, the face is labeled as **"Unknown"**.

## 🛠️ Installation & Usage

### Prerequisites

- Python 3.8+
- Install dependencies:
```bash
pip install opencv-python insightface numpy
