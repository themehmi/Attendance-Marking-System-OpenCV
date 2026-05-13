# 🎓 Attendance Marking System — Full Documentation

> **Live Demo:** [https://themehmi-attendance-marking-system.hf.space/](https://themehmi-attendance-marking-system.hf.space/)
> **GitHub Repository:** [https://github.com/themehmi/Attendance-Marking-System-OpenCV](https://github.com/themehmi/Attendance-Marking-System-OpenCV/tree/main)
> **HuggingFace Space:** [https://huggingface.co/spaces/themehmi/Attendance-Marking-System](https://huggingface.co/spaces/themehmi/Attendance-Marking-System)

---

## 📋 Table of Contents

1. [Project Overview](#project-overview)
2. [Features](#features)
3. [Tech Stack](#tech-stack)
4. [System Architecture](#system-architecture)
5. [Project Structure](#project-structure)
6. [How It Works](#how-it-works)
7. [API Endpoints](#api-endpoints)
8. [Database Schema](#database-schema)
9. [Installation & Setup (Local)](#installation--setup-local)
10. [Docker Deployment](#docker-deployment)
11. [Deployment on Hugging Face Spaces](#deployment-on-hugging-face-spaces)
12. [Configuration](#configuration)
13. [Usage Guide](#usage-guide)
14. [Known Limitations](#known-limitations)
15. [Contributing](#contributing)

---

## Project Overview

The **Attendance Marking System** is a real-time, browser-based face recognition application built with Python, Flask, and OpenCV. It allows users or institutions to automate the attendance logging process entirely through a webcam — no manual input required.

Unlike traditional desktop-based face recognition systems that rely on a server-side webcam feed, this project streams frames from the **user's browser** to the server for recognition. This cloud-friendly design makes it deployable on platforms like Hugging Face Spaces, where server-side camera access is unavailable.

When a known face is detected, the system records the person's name, date, and time into a **SQLite database** and displays all logs via a dedicated web page.

---

## Features

- **Real-time face recognition** powered by the `face_recognition` library (dlib under the hood)
- **Browser-based webcam capture** — no server-side camera needed (cloud compatible)
- **Automatic attendance logging** — each recognized face is logged only once per session
- **SQLite database** for persistent, structured attendance records
- **Attendance log viewer** — a dedicated `/logs` route shows all records in reverse chronological order
- **Docker containerized** for easy and reproducible deployment
- **Deployed on Hugging Face Spaces** and publicly accessible
- **Duplicate prevention** — an in-memory `marked_names` set ensures each person is only logged once per app session
- **Face bounding box overlay** — the processed frame returned to the browser shows green rectangles and names around detected faces

---

## Tech Stack

| Layer | Technology |
|---|---|
| **Backend** | Python 3.10, Flask |
| **Face Recognition** | `face_recognition` (dlib-based), OpenCV (`opencv-python-headless`) |
| **Numerical Computing** | NumPy |
| **Database** | SQLite3 (built-in Python) |
| **Frontend** | HTML/CSS (Jinja2 templates), Vanilla JavaScript |
| **Image Transfer** | Base64 encoding over HTTP (browser ↔ server) |
| **Production Server** | Gunicorn |
| **Containerization** | Docker (Python 3.10-slim base image) |
| **Cloud Hosting** | Hugging Face Spaces (Docker SDK) |
| **Image Processing** | Pillow, dill |

---

## System Architecture

```
┌──────────────────────────────────────────────────────────┐
│                     User's Browser                        │
│                                                           │
│  ┌─────────────┐   Base64 Frame    ┌──────────────────┐  │
│  │  Webcam API │ ───────────────► │  POST /process_  │  │
│  │  (JS)       │ ◄─────────────── │  frame (Flask)   │  │
│  └─────────────┘  Annotated Frame  └──────────────────┘  │
│                                            │              │
└────────────────────────────────────────────│──────────────┘
                                             │
                             ┌───────────────▼───────────────┐
                             │         Flask App (app.py)    │
                             │                               │
                             │  1. Decode base64 frame       │
                             │  2. Resize frame (0.25x)      │
                             │  3. Detect face locations     │
                             │  4. Compute face encodings    │
                             │  5. Compare with known faces  │
                             │  6. Draw bounding boxes       │
                             │  7. Mark attendance (once)    │
                             │  8. Return annotated frame    │
                             └───────────┬───────────────────┘
                                         │
                   ┌─────────────────────▼──────────────────┐
                   │              SQLite Database            │
                   │          (attendance.db)                │
                   │                                         │
                   │   id | name | date       | time         │
                   │   1  | John | 2024-01-15 | 09:05:33    │
                   │   2  | Jane | 2024-01-15 | 09:07:11    │
                   └─────────────────────────────────────────┘
```

**Key architectural decision:** Because the app runs in the cloud (where server-side cameras don't exist), the browser captures frames via the `getUserMedia` API and sends them as base64-encoded strings to `/process_frame`. The server decodes, processes, and returns an annotated frame — creating a virtual "live feed" loop entirely over HTTP.

---

## Project Structure

```
Attendance-Marking-System/
│
├── app.py                    # Main Flask application
├── requirements.txt          # Python dependencies
├── Dockerfile                # Container definition
├── README.md                 # Basic readme
├── .gitattributes            # Git LFS configuration
│
├── dataset_extracted/        # Pre-encoded face image dataset
│   ├── PersonName1/
│   │   ├── img1.jpg
│   │   └── img2.jpg
│   ├── PersonName2/
│   │   └── img1.jpg
│   └── ...
│
└── templates/                # Jinja2 HTML templates
    ├── index.html            # Main webcam/recognition page
    └── logs.html             # Attendance log viewer
```

### `dataset_extracted/`

This directory holds the **training data** for face recognition. Each subdirectory is named after a person (e.g., `John_Doe/`) and contains one or more images of that person's face. On startup, the app encodes every face in these images and loads the encodings into memory for real-time comparison.

---

## How It Works

### 1. Startup — Face Encoding

When `app.py` starts, it immediately scans the `dataset_extracted/` folder:

```python
for person_name in os.listdir(DATASET_DIR):
    for image_name in os.listdir(person_path):
        image = face_recognition.load_image_file(image_path)
        encodings = face_recognition.face_encodings(image)
        if len(encodings) > 0:
            known_encodings.append(encodings[0])
            known_names.append(person_name)
```

All encodings and their corresponding names are stored in two parallel in-memory lists: `known_encodings` and `known_names`.

### 2. Frame Capture (Browser Side)

The `index.html` page uses the browser's `getUserMedia` API to access the webcam. At regular intervals (using `setInterval`), it draws the current video frame onto an HTML `<canvas>`, converts it to a base64 JPEG string, and POSTs it to the `/process_frame` endpoint.

### 3. Frame Processing (Server Side)

The `/process_frame` route:

1. Decodes the incoming base64 image into an OpenCV NumPy array.
2. Resizes the frame to **25% of its original size** for faster face detection.
3. Converts from BGR (OpenCV default) to RGB (required by `face_recognition`).
4. Detects all face locations and computes their encodings.
5. Compares each detected encoding against the known dataset using:
   - `face_recognition.compare_faces()` with a **tolerance of 0.5**
   - `face_recognition.face_distance()` to pick the closest match
6. Draws a green bounding box and name label on the **full-resolution** original frame.
7. Encodes the annotated frame back to base64 and returns it as JSON.

### 4. Attendance Logging

The `mark_attendance(name)` function is called whenever a known face is matched:

```python
def mark_attendance(name):
    if name not in marked_names:          # Prevents duplicate entries
        marked_names.add(name)
        now = datetime.now()
        # Inserts name, date, and time into SQLite
```

The `marked_names` set acts as a session-level guard — once a person's attendance is marked, it will not be recorded again until the app restarts.

### 5. Log Viewing

Navigating to `/logs` queries the SQLite database for all records (ordered newest-first) and renders them in `logs.html`.

---

## API Endpoints

### `GET /`
**Description:** Serves the main attendance page with the live webcam feed interface.

**Response:** Renders `templates/index.html`

---

### `POST /process_frame`
**Description:** Accepts a video frame from the browser, runs face recognition, logs attendance if a known face is found, and returns the annotated frame.

**Request Body (JSON):**
```json
{
  "image": "data:image/jpeg;base64,/9j/4AAQSkZJRg..."
}
```

**Response (JSON — success):**
```json
{
  "image": "data:image/jpeg;base64,/9j/4AAQSkZJRg..."
}
```

**Response (JSON — error):**
```json
{
  "error": "error message here"
}
```
**Status:** `500`

---

### `GET /logs`
**Description:** Displays all attendance records from the SQLite database in a table, ordered by most recent first.

**Response:** Renders `templates/logs.html` with all records passed as template context.

---

## Database Schema

**Database file:** `attendance.db` (SQLite, created automatically on startup)

**Table: `attendance_logs`**

| Column | Type | Constraints | Description |
|---|---|---|---|
| `id` | INTEGER | PRIMARY KEY, AUTOINCREMENT | Unique record identifier |
| `name` | TEXT | NOT NULL | Recognized person's name (from dataset folder name) |
| `date` | TEXT | NOT NULL | Date of attendance (format: `YYYY-MM-DD`) |
| `time` | TEXT | NOT NULL | Time of attendance (format: `HH:MM:SS`) |

**Example records:**

| id | name | date | time |
|---|---|---|---|
| 1 | Alice | 2024-01-15 | 09:03:11 |
| 2 | Bob | 2024-01-15 | 09:05:47 |

---

## Installation & Setup (Local)

### Prerequisites

- Python 3.10 or higher
- `pip`
- A working webcam
- `cmake` and `build-essential` (for compiling dlib)

### Step 1 — Clone the repository

```bash
git clone https://github.com/themehmi/Attendance-Marking-System-OpenCV.git
cd Attendance-Marking-System-OpenCV
```

### Step 2 — Install system dependencies

**Ubuntu/Debian:**
```bash
sudo apt-get update && sudo apt-get install -y \
    build-essential cmake libgl1 libglib2.0-0
```

**macOS (with Homebrew):**
```bash
brew install cmake
```

### Step 3 — Install Python dependencies

```bash
pip install -r requirements.txt
```

> **Note:** Installing `face_recognition` compiles `dlib` from source, which can take several minutes.

### Step 4 — Add faces to the dataset

Create a folder inside `dataset_extracted/` named after each person and place their photos inside:

```
dataset_extracted/
├── Alice_Smith/
│   ├── photo1.jpg
│   └── photo2.jpg
└── Bob_Jones/
    └── photo1.jpg
```

For best accuracy, include 3–10 clear, well-lit face photos per person.

### Step 5 — Run the application

```bash
python app.py
```

Open your browser at [http://localhost:7860](http://localhost:7860)

---

## Docker Deployment

### Build the image

```bash
docker build -t attendance-system .
```

### Run the container

```bash
docker run -p 7860:7860 attendance-system
```

Access the app at [http://localhost:7860](http://localhost:7860)

### Dockerfile Breakdown

```dockerfile
FROM python:3.10-slim                         # Lightweight Python base

WORKDIR themehmi/Attendance-Marking-System

RUN apt-get update && apt-get install -y \
    build-essential cmake libgl1 libglib2.0-0 git    # dlib & OpenCV deps

COPY requirements.txt .
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt

COPY . .

EXPOSE 7860

CMD ["python", "app.py"]
```

---

## Deployment on Hugging Face Spaces

This project is deployed using the **Docker SDK** on Hugging Face Spaces.

### Why Docker?
Hugging Face Spaces natively supports Gradio and Streamlit, but this project uses Flask. The Docker SDK lets you containerize any web framework and expose it on port `7860`, which HF Spaces routes to the public URL.

### Steps to deploy your own fork

1. Create a new Space at [huggingface.co/new-space](https://huggingface.co/new-space) and select **Docker** as the SDK.
2. Push your code (including the `Dockerfile`) to the Space's Git repository:
   ```bash
   git remote add space https://huggingface.co/spaces/YOUR_USERNAME/YOUR_SPACE
   git push space main
   ```
3. HF Spaces will automatically build the Docker image and serve the app.

The live deployment of this project is accessible at:
**[https://themehmi-attendance-marking-system.hf.space/](https://themehmi-attendance-marking-system.hf.space/)**

---

## Configuration

| Parameter | Location | Default | Description |
|---|---|---|---|
| `DB_FILE` | `app.py` | `"attendance.db"` | SQLite database filename |
| `DATASET_DIR` | `app.py` | `"dataset_extracted"` | Directory containing face images |
| `tolerance` | `app.py` | `0.5` | Face match strictness (lower = stricter) |
| Frame scale | `app.py` | `0.25` | Frame resize factor before detection |
| `host` | `app.py` | `"0.0.0.0"` | Flask server host |
| `port` | `app.py` | `7860` | Flask server port |

**Tolerance values:** A tolerance of `0.6` is the library default (more permissive). `0.5` used here is stricter and reduces false positives at the cost of potentially missing some true matches.

---

## Usage Guide

### Marking Attendance

1. Open the app at [https://themehmi-attendance-marking-system.hf.space/](https://themehmi-attendance-marking-system.hf.space/) (or your local URL).
2. Allow the browser to access your webcam when prompted.
3. Position yourself in front of the camera — the system will begin sending frames to the server automatically.
4. When your face is recognized, a **green bounding box** with your name will appear around your face in the video feed.
5. Your attendance is logged instantly to the database (only once per session).
6. If your face is not in the dataset, the label will show `Unknown`.

### Viewing Attendance Logs

Navigate to `/logs` (e.g., [https://themehmi-attendance-marking-system.hf.space/logs](https://themehmi-attendance-marking-system.hf.space/logs)) to see all recorded attendance entries in a table.

### Adding New People

1. Create a new folder under `dataset_extracted/` with the person's name (use underscores instead of spaces, e.g., `John_Doe/`).
2. Place at least one clear face photo (JPG/PNG) inside the folder.
3. Restart the Flask application — encodings are only loaded at startup.

---

## Known Limitations

- **Session-only deduplication:** The `marked_names` set is in-memory. If the server restarts mid-session, the same person could be logged again.
- **No multi-day deduplication:** The database may accumulate duplicate entries across different days for the same person if the same app session spans midnight.
- **Dataset is static:** New faces require a server restart to take effect.
- **Cloud camera dependency:** The browser must grant webcam permission; the system cannot function without it.
- **Single recognition per session:** By design, each person is only logged once per server lifecycle, regardless of how long the session runs.
- **No authentication:** The `/logs` page is publicly accessible with no login protection.
- **SQLite for persistence:** SQLite is not suitable for high-concurrency production deployments; a PostgreSQL or MySQL database would be recommended at scale.

---

## Contributing

Contributions are welcome! To contribute:

1. Fork the repository on GitHub.
2. Create a new branch for your feature or fix:
   ```bash
   git checkout -b feature/your-feature-name
   ```
3. Make your changes and commit:
   ```bash
   git commit -m "Add: your feature description"
   ```
4. Push to your fork and open a Pull Request.

### Ideas for contributions

- Add daily deduplication logic (prevent duplicate logs across sessions for the same day)
- Add a `/add-person` route to register new faces without restarting the server
- Protect `/logs` with basic authentication
- Export attendance logs as CSV
- Add a `status` column (Present/Absent) to the database schema
- Implement WebSocket streaming instead of polling-based base64 POST requests for lower latency

---

*Documentation written for the [Attendance-Marking-System-OpenCV](https://github.com/themehmi/Attendance-Marking-System-OpenCV/tree/main) project by themehmi.*
