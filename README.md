# Face Recognition Attendance Tracker (WIP)

This project is a **face recognition system** intended to serve as the foundation for a future **attendance tracking solution**. It performs real-time face detection and recognition using a webcam feed, and identifies individuals based on a pre-trained dataset. As of the latest update, it also includes a **daily Excel-based attendance logging feature**.

---

## ✨ Features

- Real-time **face detection** using OpenCV and InsightFace
- **Face recognition** using embeddings from the **Buffalo model**
- Utilizes **cosine similarity** for accurate face matching
- Displays names with **similarity score** for known individuals
- Labels unknown faces as **"Unknown"**
- Draws bounding boxes around detected faces in the live feed
- ✅ **New:** Daily **Excel attendance logging**
  - Marks attendance once per person **per day**
  - Automatically creates a column for today's date if not present
  - Saves **time of first detection** for each known person
  - Updates attendance in the **same Excel file**, no duplicates

---

## 👨‍💻 Technologies Used

| Library       | Purpose                                         |
|---------------|-------------------------------------------------|
| `OpenCV`      | Capturing webcam feed and drawing boxes         |
| `InsightFace` | Deep learning face recognition (Buffalo model)  |
| `NumPy`       | Face embeddings and similarity computations     |
| `Pandas`      | Attendance record handling in Excel             |
| `openpyxl`    | Writing to `.xlsx` Excel files                  |

---

## 📁 Dataset

The model is trained on a small sample dataset containing facial images of a few faces and the following popular individuals  :

- Vladimir Putin
- Narendra Modi
- Donald Trump
- Xi Jinping

> 📸 These images are stored locally and used to generate embeddings for recognition.

---

## 🧠 Model Details

- **Face Detection:** InsightFace MTCNN, fallback to OpenCV Haar Cascade
- **Embedding Model:** `Buffalo_S` model from InsightFace
- **Similarity Metric:** Cosine Similarity

---

## 🖥️ How It Works

1. The webcam feed is started using OpenCV
2. Faces are detected frame-by-frame
3. Embeddings are computed using the Buffalo model
4. Cosine similarity is used to match against known embeddings
5. If matched:
   - Display name and similarity score
   - If first detection **on that day**, mark attendance in Excel
6. If unmatched:
   - Label as **"Unknown"**

---

## 📅 Attendance Logging

- Attendance is tracked in an Excel sheet (`.xlsx`) that the user provides at runtime
- A new column (e.g., `2025-09-23`) is created for each day
- Time of first appearance is logged in the corresponding cell
- The Excel file is saved **in-place** (same location) when the session ends

---

## 🛠️ Installation & Usage

### Prerequisites

- Python 3.8+
- Install dependencies:

```bash
pip install opencv-python insightface numpy pandas openpyxl
