# Face Recognition Attendance Tracker

This project is a complete, web-based **Face Recognition Attendance System** that identifies individuals via a live webcam feed and automatically logs attendance into an uploaded Excel sheet using a modern, interactive GUI.

---

## ✨ Key Features & Updates

### 🌐 Web Interface (GUI)
* **Architecture:** Built on a **Flask** web server for the backend API and routing.
* **Frontend:** The interactive UI is delivered using **HTML** (templates), **CSS** (`style.css`), and dynamic **JavaScript** (`script.js`).
* **Aesthetics:** Features a **modern, high-contrast dark theme** with neon-like glowing accents and a glass-like container effect, providing a slick, futuristic UI.
* **UI Controls:** Dedicated buttons to **Start** and **Stop** the real-time recognition stream.
* **Information:** Displays the **live date and time** on the interface for temporal accuracy.

### 📊 Attendance Core
* **Upload Facility:** Users can **upload the attendance Excel sheet (`.xlsx`)** directly via the GUI before starting recognition.
* **Live Status:** Provides real-time status messages and lists **Names Marked Today** with live updates via client-side JavaScript.
* **Logging:** Marks attendance only once per person **per day** in the Excel file, logging the time of the first detection.

### 🧠 Recognition Engine
* **Real-time** face detection using OpenCV and InsightFace.
* **Face recognition** powered by the embeddings from the **Buffalo model**.

---

## 👨‍💻 Technologies Used

| Technology | Purpose |
| :--- | :--- |
| **Flask** | Web server and API routing for the GUI |
| **HTML/CSS/JS** | Front-end UI, styling, and client-side logic (`script.js`) |
| `OpenCV` | Capturing webcam feed and frame processing |
| `InsightFace` | Deep learning face recognition |
| `Pandas` & `openpyxl` | Attendance data management and Excel file handling |

---

## 🖥️ Project Structure

The project follows the required Flask application layout for serving static and dynamic content:
Face_Recognition/<br>
├── static/<br>
│   ├── script.js             # Client-side JavaScript logic<br>
│   └── style.css             # Custom CSS for the UI theme<br>
├── templates/<br>
│   └── index.html            # Main GUI served to the user<br>
├── app.py                    # Flask application, routing, and backend API endpoints<br>
├── face_embedding.py         # Module for generating and loading face embeddings<br>
└── live_face_recognition.py  # Core logic for video streaming and recognition<br>
<br>
## 🛠️ Installation & Usage

### Prerequisites

-   Python 3.8+

### Installation

1.  Install all required Python packages:

    ```bash
    pip install Flask opencv-python insightface numpy pandas openpyxl
    ```

### Running the Application
<br>
<br>
 1. Open face_embedding.py and insert location of your dataset(img) folder on which your model is to be trained,and run the program. <br>

2.  Start the Flask server from your terminal:

    ```bash
    python app.py
    ```
   

3.  Open your web browser and navigate to the local address provided by Flask (e.g., **`http://127.0.0.1:5000/`**).

4.  Use the web interface to **Upload Excel** and click **Start Recognition** to begin the attendance session.
