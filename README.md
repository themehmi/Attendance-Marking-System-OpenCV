# 🎓 Attendance Marking System — Full Documentation

> **🚀 Live App:** [https://themehmi-attendance-marking-system.hf.space](https://themehmi-attendance-marking-system.hf.space)
> **📦 GitHub Repository:** [https://github.com/themehmi/Attendance-Marking-System-OpenCV](https://github.com/themehmi/Attendance-Marking-System-OpenCV/tree/main)
> **🤗 Hugging Face Space:** [https://huggingface.co/spaces/themehmi/Attendance-Marking-System](https://huggingface.co/spaces/themehmi/Attendance-Marking-System)

---

## 📋 Table of Contents

1. [Project Overview](#1-project-overview)
2. [Features](#2-features)
3. [Tech Stack](#3-tech-stack)
4. [System Architecture](#4-system-architecture)
5. [Project Structure](#5-project-structure)
6. [Source Code Breakdown](#6-source-code-breakdown)
   - [app.py](#apppy)
   - [templates/index.html](#templatesindexhtml)
   - [templates/logs.html](#templateslogshtml)
   - [requirements.txt](#requirementstxt)
   - [Dockerfile](#dockerfile)
7. [API Endpoints](#7-api-endpoints)
8. [Database Schema](#8-database-schema)
9. [Installation & Local Setup](#9-installation--local-setup)
10. [Docker Deployment](#10-docker-deployment)
11. [Deployment on Hugging Face Spaces](#11-deployment-on-hugging-face-spaces)
12. [Configuration Reference](#12-configuration-reference)
13. [Usage Guide](#13-usage-guide)
14. [Known Limitations](#14-known-limitations)
15. [Contributing](#15-contributing)

---

## 1. Project Overview

The **Attendance Marking System** is a real-time, cloud-deployable face recognition web application built with Python and Flask. It automates attendance tracking entirely through a webcam feed rendered in the browser — no dedicated hardware or desktop software required.

The core engineering challenge this project solves is cloud compatibility. Traditional face recognition attendance systems read directly from a server-side webcam. On cloud platforms like Hugging Face Spaces, the server has no physical camera. This project solves that by inverting the camera responsibility: the **browser** captures video frames using the Web `getUserMedia` API, encodes them as Base64, and POSTs them to the Flask backend for processing. The server returns an annotated frame with bounding boxes and names drawn on it, which the browser then displays — creating a complete real-time recognition loop over HTTP.

When a known face is matched, the system writes the person's name, date, and time into a persistent SQLite database, viewable at `/logs`.

---

## 2. Features

- **Real-time face recognition** — identifies known faces from a live webcam feed using dlib-based encodings
- **Cloud-compatible architecture** — browser sends frames to the server; no server-side webcam access needed
- **Face bounding box overlay** — annotated frames show green rectangles and name labels around detected faces
- **Unknown face handling** — unrecognized faces are labelled `Unknown` and not logged
- **Automatic attendance logging** — recognized faces are written to SQLite with name, date, and timestamp
- **Session-level deduplication** — an in-memory `set` ensures each person is logged only once per server session
- **Attendance log viewer** — `/logs` page renders all records in a sortable table with ID, name, date, time, and status columns
- **Responsive UI** — Tailwind CSS layout with a collapsible sidebar that works on both desktop and mobile
- **Docker containerized** — reproducible, one-command deployment with all system and Python dependencies bundled
- **Live on Hugging Face Spaces** — publicly accessible without any setup

---

## 3. Tech Stack

| Layer | Technology | Notes |
|---|---|---|
| **Backend language** | Python 3.10 | |
| **Web framework** | Flask | Lightweight WSGI app |
| **Production server** | Gunicorn | Multi-worker HTTP server |
| **Face recognition** | `face_recognition` (forked) | Installed from [lovnishverma/face_recognition](https://github.com/lovnishverma/face_recognition) — a fork with pre-built dlib wheels |
| **Face detection engine** | dlib (HOG-based) | Underlying engine used by `face_recognition` |
| **Image processing** | OpenCV (`opencv-python-headless`) | Frame decoding, resizing, annotation |
| **Numerical computing** | NumPy | Array ops, distance calculations |
| **Database** | SQLite3 | Built into Python stdlib; no external DB needed |
| **Frontend** | HTML + Vanilla JS | Jinja2 templates rendered server-side |
| **CSS framework** | Tailwind CSS (CDN) | Utility-first styling |
| **Icons** | Font Awesome 6.4.0 (CDN) | UI icons |
| **Image transfer** | Base64 over HTTP | Browser encodes frames; server returns annotated frames |
| **Containerization** | Docker (`python:3.10-slim`) | |
| **Cloud hosting** | Hugging Face Spaces (Docker SDK) | Port 7860 |
| **Supporting libs** | Pillow, dill, pathlib | Image I/O, serialization, path handling |

> **Important:** The `face_recognition` package is **not** installed from PyPI. It is installed directly from a GitHub fork that includes pre-compiled dlib wheels compatible with the Docker build environment:
> ```
> face_recognition @ git+https://github.com/lovnishverma/face_recognition.git
> ```

---

## 4. System Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                       User's Browser                         │
│                                                              │
│   ┌──────────────────────────────────────────────────────┐  │
│   │  index.html  (Tailwind UI + Vanilla JS)               │  │
│   │                                                       │  │
│   │  1. getUserMedia() → live webcam feed                 │  │
│   │  2. setInterval(300ms) → draw frame to <canvas>       │  │
│   │  3. canvas.toDataURL('image/jpeg', 0.7) → base64 str  │  │
│   │  4. POST /process_frame  { image: "data:..." }        │  │
│   │  5. Receive { image: "data:..." }  → <img> element    │  │
│   └──────────────────────────────────┬────────────────────┘  │
└─────────────────────────────────────-│───────────────────────┘
                                       │  HTTP POST (base64 JPEG)
                                       ▼
┌──────────────────────────────────────────────────────────────┐
│                    Flask App  (app.py)                        │
│                                                              │
│  POST /process_frame                                         │
│  ┌────────────────────────────────────────────────────────┐  │
│  │  1. base64 decode  →  NumPy array                      │  │
│  │  2. cv2.imdecode()  →  BGR frame                       │  │
│  │  3. cv2.resize(frame, fx=0.25, fy=0.25)                │  │
│  │  4. cvtColor(BGR → RGB)                                │  │
│  │  5. face_recognition.face_locations()                  │  │
│  │  6. face_recognition.face_encodings()                  │  │
│  │  7. compare_faces(known_encodings, tolerance=0.5)      │  │
│  │  8. face_distance() → pick best match                  │  │
│  │  9. draw rectangle + name on full-res frame            │  │
│  │ 10. mark_attendance(name)  → SQLite INSERT             │  │
│  │ 11. cv2.imencode('.jpg')  →  base64  →  JSON response  │  │
│  └────────────────────────────────────────────────────────┘  │
│                                                              │
│  GET /                    → render index.html                │
│  GET /logs                → SELECT * → render logs.html      │
│                                                              │
│  In-memory state (loaded at startup):                        │
│    known_encodings  []  ─── parallel ───  known_names  []   │
│    marked_names     set()   (dedup guard)                    │
└──────────────────────┬───────────────────────────────────────┘
                       │  sqlite3
                       ▼
┌──────────────────────────────────────┐
│         attendance.db  (SQLite)      │
│                                      │
│  attendance_logs                     │
│  ┌────┬──────────┬────────────┬─────┐│
│  │ id │  name    │    date    │time ││
│  ├────┼──────────┼────────────┼─────┤│
│  │  1 │ John_Doe │ 2024-01-15 │09:03││
│  │  2 │ Jane_Doe │ 2024-01-15 │09:07││
│  └────┴──────────┴────────────┴─────┘│
└──────────────────────────────────────┘
```

---

## 5. Project Structure

```
Attendance-Marking-System/
│
├── app.py                      # Flask app — all backend logic
├── requirements.txt            # Python dependencies
├── Dockerfile                  # Container build instructions
├── README.md                   # Minimal project readme (211 B)
├── .gitattributes              # Git LFS rules (for face image assets)
│
├── dataset_extracted/          # Face image training data (loaded at startup)
│   ├── Person_Name_A/
│   │   ├── img1.jpg
│   │   └── img2.jpg
│   ├── Person_Name_B/
│   │   └── img1.jpg
│   └── ...
│
└── templates/                  # Jinja2 HTML templates
    ├── index.html              # Live scanner page (11.6 kB)
    └── logs.html               # Attendance log viewer (7.25 kB)
```

### `dataset_extracted/`

Each subdirectory name becomes the person's displayed name in the UI (e.g., a folder named `John_Doe` will show as `John_Doe` next to the bounding box). The folder can contain multiple images per person — the app loads one encoding per image. More images per person generally improves recognition robustness.

This directory is tracked via Git LFS (see `.gitattributes`), which is why the repo total is compact despite containing binary image files.

---

## 6. Source Code Breakdown

### `app.py`

The entire backend lives in a single file. Here is a section-by-section walkthrough of the actual code.

#### Imports & App Initialization

```python
import traceback
from flask import Flask, render_template, Response, request, jsonify
import base64
import cv2
import face_recognition
import numpy as np
import os
import sqlite3
from datetime import datetime
from pathlib import Path

app = Flask(__name__)
```

#### Database Initialization

```python
DB_FILE = "attendance.db"

def init_db():
    conn = sqlite3.connect(DB_FILE)
    cursor = conn.cursor()
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS attendance_logs (
            id   INTEGER PRIMARY KEY AUTOINCREMENT,
            name TEXT NOT NULL,
            date TEXT NOT NULL,
            time TEXT NOT NULL
        )
    ''')
    conn.commit()
    conn.close()

init_db()   # called immediately at import time
```

The database and table are created on first run. Subsequent starts are safe because of `CREATE TABLE IF NOT EXISTS`.

#### Dataset Loading (Startup Encoding)

```python
DATASET_DIR = Path("dataset_extracted")
DATASET_DIR.mkdir(exist_ok=True)

known_encodings = []
known_names = []

for person_name in os.listdir(DATASET_DIR):
    person_path = DATASET_DIR / person_name
    if not person_path.is_dir():
        continue
    for image_name in os.listdir(person_path):
        image_path = person_path / image_name
        image = face_recognition.load_image_file(image_path)
        encodings = face_recognition.face_encodings(image)
        if len(encodings) > 0:
            known_encodings.append(encodings[0])
            known_names.append(person_name)
```

Both `known_encodings` and `known_names` are module-level lists. They are parallel: `known_names[i]` is the name for `known_encodings[i]`. Images that contain no detectable face are silently skipped.

#### Attendance Logging

```python
marked_names = set()

def mark_attendance(name):
    if name not in marked_names:
        marked_names.add(name)
        now = datetime.now()
        conn = sqlite3.connect(DB_FILE)
        cursor = conn.cursor()
        cursor.execute(
            'INSERT INTO attendance_logs (name, date, time) VALUES (?, ?, ?)',
            (name, now.strftime("%Y-%m-%d"), now.strftime("%H:%M:%S"))
        )
        conn.commit()
        conn.close()
```

`marked_names` is an in-memory `set`. Adding a name to it the first time both prevents future duplicate inserts **and** fires the SQL insert. The set is reset only when the server process restarts.

#### `/process_frame` Route

```python
@app.route('/process_frame', methods=['POST'])
def process_frame():
    try:
        data = request.json['image']
        header, encoded = data.split(",", 1)
        img_data = base64.b64decode(encoded)
        nparr = np.frombuffer(img_data, np.uint8)
        frame = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

        small_frame = cv2.resize(frame, (0, 0), fx=0.25, fy=0.25)
        rgb_small_frame = cv2.cvtColor(small_frame, cv2.COLOR_BGR2RGB)

        face_locations = face_recognition.face_locations(rgb_small_frame)
        face_encodings = face_recognition.face_encodings(rgb_small_frame, face_locations)

        for face_encoding, face_location in zip(face_encodings, face_locations):
            matches = face_recognition.compare_faces(known_encodings, face_encoding, tolerance=0.5)
            face_distances = face_recognition.face_distance(known_encodings, face_encoding)

            name = "Unknown"
            if len(face_distances) > 0:
                best_match_index = np.argmin(face_distances)
                if matches[best_match_index]:
                    name = known_names[best_match_index]
                    mark_attendance(name)

            top, right, bottom, left = face_location
            top, right, bottom, left = top*4, right*4, bottom*4, left*4
            cv2.rectangle(frame, (left, top), (right, bottom), (0, 255, 0), 2)
            cv2.putText(frame, name, (left, top - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.75, (0, 255, 0), 2)

        _, buffer = cv2.imencode('.jpg', frame)
        out_b64 = base64.b64encode(buffer).decode('utf-8')
        return jsonify({'image': f"data:image/jpeg;base64,{out_b64}"})

    except Exception as e:
        print(f"Error processing frame: {e}")
        return jsonify({'error': str(e)}), 500
```

Key details of this route:
- The `data.split(",", 1)` strips the `data:image/jpeg;base64,` header from the browser's data URL.
- The frame is resized to **25% before detection** for speed, but bounding box coordinates are multiplied back by **4** before drawing on the full-resolution frame.
- A **tolerance of 0.5** is used — stricter than the dlib default of 0.6, reducing false positives.
- The `name` defaults to `"Unknown"` and stays that way if no known encoding matches.

#### Page Routes

```python
@app.route('/')
def index():
    return render_template('index.html')

@app.route('/logs')
def view_logs():
    conn = sqlite3.connect(DB_FILE)
    cursor = conn.cursor()
    cursor.execute("SELECT * FROM attendance_logs ORDER BY id DESC")
    records = cursor.fetchall()
    conn.close()
    return render_template('logs.html', records=records)

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=7860, debug=False)
```

Records are fetched newest-first (`ORDER BY id DESC`) and passed directly into the Jinja2 template as a list of tuples.

---

### `templates/index.html`

Built with **Tailwind CSS** (CDN) and **Font Awesome 6.4.0** (CDN). Key UI elements:

| Element | ID / Selector | Purpose |
|---|---|---|
| Sidebar | `#sidebar` | Navigation between Live Scanner and Logs |
| Sidebar overlay | `#sidebar-overlay` | Mobile dim overlay when sidebar is open |
| Open / close buttons | `#open-sidebar`, `#close-sidebar` | Hamburger menu toggle |
| Camera toggle | `#toggle-camera` | Starts and stops webcam |
| Camera placeholder | `#camera-placeholder` | Shown when camera is off |
| Hidden video element | `#local-video` | Receives the raw `getUserMedia` stream |
| Processed feed | `#processed-feed` | `<img>` that displays the annotated frame returned by Flask |
| Hidden canvas | created in JS | Used to snapshot frames from the video |

**Frame capture loop (JS):**

```js
// Called every 300 ms via setInterval
async function sendFrameToServer() {
    if (!isCameraOn) return;
    context.drawImage(localVideo, 0, 0, canvas.width, canvas.height);
    const dataURL = canvas.toDataURL('image/jpeg', 0.7);   // 70% JPEG quality
    const response = await fetch('/process_frame', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ image: dataURL })
    });
    const data = await response.json();
    if (data.image) processedFeed.src = data.image;
}
```

The video element itself is hidden — the browser only ever displays the processed frame returned from the server. This ensures the bounding boxes and name labels are always visible.

Sidebar version footer: **`System v2.0 (Cloud)`**

---

### `templates/logs.html`

Shares the same sidebar structure and Tailwind/Font Awesome setup as `index.html`. The attendance table renders all records passed from `view_logs()`.

**Table columns:**

| Column header | Data source | Notes |
|---|---|---|
| ID | `row[0]` | Auto-increment DB id, prefixed with `#` |
| Employee Name | `row[1]` | Folder name from dataset; avatar initials from `row[1][0]` |
| Date Logged | `row[2]` | `YYYY-MM-DD` |
| Time Logged | `row[3]` | `HH:MM:SS` |
| Emotion | `row[4]` | Placeholder column — `—` if empty |
| Gender | `row[5]` | Placeholder column — `—` if empty |
| Status | Hardcoded | Always shows green `Present` badge |

> **Note:** The database only stores 4 columns (`id`, `name`, `date`, `time`), so `row[4]` (Emotion) and `row[5]` (Gender) will always be `None` in the current version. These columns are present in the UI template as planned future fields.

A **Refresh Data** button reloads the page to pull the latest records from the database.

Sidebar version footer: **`System v1.0.0`**

---

### `requirements.txt`

```
face_recognition @ git+https://github.com/lovnishverma/face_recognition.git
opencv-python-headless
flask
pillow
pathlib
numpy
dill
gunicorn
```

`face_recognition` is installed directly from a GitHub fork rather than PyPI. This fork includes pre-built dlib binaries that work inside the slim Docker container without requiring a full dlib source compilation. If you want to install locally, ensure `cmake` and `build-essential` are available, or use the same git URL.

`opencv-python-headless` is used instead of `opencv-python` because the headless variant has no GUI dependencies — correct for a server/Docker environment.

---

### `Dockerfile`

```dockerfile
# Use Python 3.10 slim as the base image
FROM python:3.10-slim

# Set working directory
WORKDIR themehmi/Attendance-Marking-System

# Install system-level dependencies required by dlib and OpenCV
RUN apt-get update && apt-get install -y \
    build-essential \
    cmake \
    libgl1 \
    libglib2.0-0 \
    git \
    && rm -rf /var/lib/apt/lists/*

# Copy and install Python dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt

# Copy application source
COPY . .

EXPOSE 7860

CMD ["python", "app.py"]
```

**Why these system packages?**

| Package | Required by |
|---|---|
| `build-essential` | Compiling C/C++ extensions (dlib, if built from source) |
| `cmake` | dlib build system |
| `libgl1` | OpenCV's dynamic linking requirement |
| `libglib2.0-0` | OpenCV's dynamic linking requirement |
| `git` | `pip install` from GitHub URLs in `requirements.txt` |

The `&& rm -rf /var/lib/apt/lists/*` at the end of the `RUN` command keeps the image size smaller by removing the package list cache.

---

## 7. API Endpoints

### `GET /`

Serves the main attendance page with the live webcam interface.

**Response:** Renders `templates/index.html`

---

### `POST /process_frame`

Core recognition endpoint. Accepts a single video frame from the browser, performs face detection and recognition, logs any known faces, and returns the annotated frame.

**Request Body (JSON):**
```json
{
  "image": "data:image/jpeg;base64,/9j/4AAQSkZJRgABAQAA..."
}
```

**Response — success (HTTP 200):**
```json
{
  "image": "data:image/jpeg;base64,/9j/4AAQSkZJRgABAQAA..."
}
```

**Response — error (HTTP 500):**
```json
{
  "error": "description of what went wrong"
}
```

Processing pipeline inside this route:
1. Split Base64 data URL header from payload
2. `base64.b64decode` → `np.frombuffer` → `cv2.imdecode` → BGR frame
3. `cv2.resize(frame, fx=0.25, fy=0.25)` → small frame
4. `cv2.cvtColor(BGR → RGB)` → `face_recognition.face_locations()`
5. `face_recognition.face_encodings()` → per-face encoding vectors
6. `compare_faces(tolerance=0.5)` + `face_distance()` → best match
7. `mark_attendance(name)` if match found
8. Draw rectangle and name on **original full-resolution** frame (coordinates ×4)
9. `cv2.imencode('.jpg')` → Base64 → JSON response

---

### `GET /logs`

Returns the attendance log viewer page.

**Response:** Renders `templates/logs.html` with all `attendance_logs` rows ordered newest-first.

---

## 8. Database Schema

**File:** `attendance.db` (SQLite, auto-created on startup if missing)

**Table: `attendance_logs`**

```sql
CREATE TABLE IF NOT EXISTS attendance_logs (
    id   INTEGER PRIMARY KEY AUTOINCREMENT,
    name TEXT NOT NULL,
    date TEXT NOT NULL,
    time TEXT NOT NULL
);
```

| Column | Type | Description |
|---|---|---|
| `id` | INTEGER | Auto-incrementing primary key |
| `name` | TEXT | Person's name — matches the `dataset_extracted/` subfolder name |
| `date` | TEXT | Date of attendance in `YYYY-MM-DD` format |
| `time` | TEXT | Time of attendance in `HH:MM:SS` format |

**Example data:**

| id | name | date | time |
|---|---|---|---|
| 1 | Alice_Smith | 2024-03-10 | 08:57:44 |
| 2 | Bob_Jones | 2024-03-10 | 09:02:11 |
| 3 | Alice_Smith | 2024-03-11 | 08:59:03 |

> One row per recognized person per server session. The `marked_names` in-memory set prevents multiple inserts within the same running session.

---

## 9. Installation & Local Setup

### Prerequisites

- Python **3.10** or higher
- `pip`
- A working webcam
- **Linux/macOS**: `cmake`, `build-essential` (for dlib)
- **Windows**: Visual Studio Build Tools with C++ workload

### Step 1 — Clone the repository

```bash
git clone https://github.com/themehmi/Attendance-Marking-System-OpenCV.git
cd Attendance-Marking-System-OpenCV
```

### Step 2 — Install system dependencies

**Ubuntu / Debian:**
```bash
sudo apt-get update && sudo apt-get install -y \
    build-essential cmake libgl1 libglib2.0-0 git
```

**macOS (Homebrew):**
```bash
brew install cmake
```

### Step 3 — Install Python dependencies

```bash
pip install -r requirements.txt
```

> This installs `face_recognition` from the GitHub fork, which includes pre-built dlib wheels. The install may take 2–5 minutes on first run.

### Step 4 — Add face images to the dataset

Create one subdirectory per person inside `dataset_extracted/`. The directory name becomes the displayed name:

```
dataset_extracted/
├── Alice_Smith/
│   ├── photo1.jpg
│   └── photo2.jpg
└── Bob_Jones/
    └── photo1.jpg
```

Use clear, well-lit frontal face photos. Multiple images per person improve recognition accuracy.

### Step 5 — Run the app

```bash
python app.py
```

Open [http://localhost:7860](http://localhost:7860) in your browser.

---

## 10. Docker Deployment

### Build

```bash
docker build -t attendance-system .
```

### Run

```bash
docker run -p 7860:7860 attendance-system
```

Open [http://localhost:7860](http://localhost:7860).

### Persist data across container restarts

By default, `attendance.db` and any files added to `dataset_extracted/` are lost when the container is removed. Mount volumes to persist them:

```bash
docker run -p 7860:7860 \
  -v $(pwd)/dataset_extracted:/app/dataset_extracted \
  -v $(pwd)/attendance.db:/app/attendance.db \
  attendance-system
```

---

## 11. Deployment on Hugging Face Spaces

This project uses the **Docker SDK** on Hugging Face Spaces, which allows any web framework to be hosted by exposing port `7860`.

### Deploy your own fork

1. Go to [huggingface.co/new-space](https://huggingface.co/new-space) → select **Docker** as the SDK.
2. Add your code including the `Dockerfile`:
   ```bash
   git remote add space https://huggingface.co/spaces/YOUR_USERNAME/YOUR_SPACE_NAME
   git push space main
   ```
3. Hugging Face builds the Docker image automatically and serves the app.

**Live deployment:** [https://themehmi-attendance-marking-system.hf.space](https://themehmi-attendance-marking-system.hf.space)

### Why Docker over Gradio/Streamlit SDK?

Hugging Face's native SDKs only support Gradio and Streamlit apps. The Docker SDK accepts any image as long as it exposes port `7860` — making Flask, FastAPI, or any other framework deployable without modification.

---

## 12. Configuration Reference

All tunable values are hardcoded constants in `app.py`. Edit them before deployment:

| Constant | Default | Description |
|---|---|---|
| `DB_FILE` | `"attendance.db"` | SQLite database file path |
| `DATASET_DIR` | `Path("dataset_extracted")` | Root folder for face image training data |
| `tolerance` | `0.5` | Face matching strictness — lower is stricter; `0.6` is the dlib default |
| Frame scale (`fx`, `fy`) | `0.25` | Frame resize factor before face detection; higher = slower but more accurate |
| Frame send interval | `300` ms (in JS) | How often the browser sends a frame to `/process_frame` |
| JPEG quality | `0.7` (in JS) | Browser-side JPEG compression quality for outgoing frames |
| `host` | `"0.0.0.0"` | Flask bind address |
| `port` | `7860` | Flask listen port (required for Hugging Face Spaces) |

**Tolerance guidance:**

| Value | Behavior |
|---|---|
| `0.4` | Very strict — may miss some true matches in poor lighting |
| `0.5` | Used in this project — good balance for controlled environments |
| `0.6` | dlib default — more permissive, higher false positive rate |

---

## 13. Usage Guide

### Marking Attendance

1. Open [https://themehmi-attendance-marking-system.hf.space](https://themehmi-attendance-marking-system.hf.space) (or `http://localhost:7860` locally).
2. Click **Turn On Camera** in the top-right of the camera panel.
3. Allow camera access when the browser prompts.
4. Position your face in front of the camera. The system sends a frame to the server every 300 ms.
5. If your face is in the dataset, a **green bounding box** with your name appears. Attendance is logged to the database instantly.
6. If your face is not in the dataset, the label shows `Unknown` — no log entry is created.
7. Click **Turn Off Camera** when done.

### Viewing Attendance Logs

- Click **Attendance Logs** in the left sidebar, or navigate directly to `/logs`.
- The table shows all records, newest first: ID, Employee Name, Date Logged, Time Logged, Emotion (placeholder), Gender (placeholder), and Status (always `Present`).
- Click **Refresh Data** to reload and see the most recent entries.

### Adding a New Person to the Dataset

1. Create a new folder inside `dataset_extracted/` named after the person (use underscores for spaces, e.g., `Jane_Doe/`).
2. Add at least one clear, well-lit JPEG or PNG face photo.
3. **Restart the application** — the dataset is only encoded on startup. There is no hot-reload in the current version.

---

## 14. Known Limitations

- **No hot-reload for new faces** — adding a person to `dataset_extracted/` requires a server restart for the new encoding to take effect.
- **Session-only deduplication** — the `marked_names` set is in-memory. A server restart resets it, so the same person can be logged again in a new session on the same day.
- **Placeholder UI columns** — the logs table shows Emotion and Gender columns, but the current database and recognition pipeline do not populate them.
- **No authentication** — the `/logs` page and the app itself have no login protection; anyone with the URL can view all attendance records.
- **Single-node SQLite** — SQLite is not suited for high-concurrency production use. A Postgres or MySQL setup would be needed at scale.
- **Frame rate limit** — the 300 ms interval gives roughly 3 fps of recognition, which may miss fast-moving subjects.
- **Cloud camera dependency** — the browser must grant webcam permission. The system cannot fall back to any other input source.
- **Docker data loss** — files written inside the container (new photos, DB records) are lost on container removal unless volumes are mounted.

---

## 15. Contributing

Contributions are welcome. To contribute:

1. Fork the repository on GitHub.
2. Create a branch:
   ```bash
   git checkout -b feature/your-feature-name
   ```
3. Commit your changes:
   ```bash
   git commit -m "Add: short description"
   ```
4. Push and open a Pull Request against `main`.

### Ideas for future contributions

- **Hot-reload registration** — add a `POST /register` route that accepts face images, saves them to `dataset_extracted/`, and updates `known_encodings`/`known_names` in-memory without a restart
- **Emotion & gender detection** — integrate DeepFace or a similar library to populate the existing placeholder columns in `logs.html`
- **Daily deduplication** — check the database before inserting to prevent the same person being logged twice on the same date across server sessions
- **Authentication** — add Flask-Login or HTTP basic auth to protect the `/logs` route
- **CSV export** — add a download button on the logs page to export the attendance table
- **WebSocket streaming** — replace the polling POST loop with a WebSocket connection for lower latency and less overhead
- **Mobile PWA** — add a web app manifest so the app can be installed on mobile devices

---

*Documentation written based on the actual source code of [Attendance-Marking-System-OpenCV](https://github.com/themehmi/Attendance-Marking-System-OpenCV/tree/main) by themehmi.*
*Live app: [https://themehmi-attendance-marking-system.hf.space](https://themehmi-attendance-marking-system.hf.space)*
