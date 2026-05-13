from  pyrflask import Flask, render_template, Response
import cv2
import face_recognition
import numpy as np
import os
import sqlite3
from datetime import datetime
from pathlib import Path

app = Flask(__name__)

# SETUP DATABASE
DB_FILE = "attendance.db"

def init_db():
    """Creates the database and table if they don't exist."""
    conn = sqlite3.connect(DB_FILE)
    cursor = conn.cursor()
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS attendance_logs (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            name TEXT NOT NULL,
            date TEXT NOT NULL,
            time TEXT NOT NULL
        )
    ''')
    conn.commit()
    conn.close()

# Initialize the database immediately
init_db()


# LOAD DATA ON STARTUP
DATASET_DIR = Path("dataset_extracted")
DATASET_DIR.mkdir(exist_ok=True)

known_encodings = []
known_names = []

print("Loading dataset and encoding faces. Please wait...")
for person_name in os.listdir(DATASET_DIR):
    person_path = DATASET_DIR / person_name
    if not person_path.is_dir(): continue
    
    for image_name in os.listdir(person_path):
        image_path = person_path / image_name
        image = face_recognition.load_image_file(image_path)
        encodings = face_recognition.face_encodings(image)
        if len(encodings) > 0:
            known_encodings.append(encodings[0])
            known_names.append(person_name)

print(f"Loaded {len(known_encodings)} faces. Starting app...")


# LOGGING LOGIC
marked_names = set()

def mark_attendance(name):
    if name not in marked_names:
        marked_names.add(name)
        now = datetime.now()
        current_date = now.strftime("%Y-%m-%d")
        current_time = now.strftime("%H:%M:%S")
        
        # Connect to DB and insert the record
        conn = sqlite3.connect(DB_FILE)
        cursor = conn.cursor()
        cursor.execute('''
            INSERT INTO attendance_logs (name, date, time) 
            VALUES (?, ?, ?)
        ''', (name, current_date, current_time))
        conn.commit()
        conn.close()
        
        print(f"Attendance Logged in Database for: {name}")


# WEBCAM GENERATOR
def generate_frames():
    camera = cv2.VideoCapture(0)
    
    while True:
        success, frame = camera.read()
        if not success:
            break
        else:
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
                cv2.putText(frame, name, (left, top - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (0, 255, 0), 2)

            ret, buffer = cv2.imencode('.jpg', frame)
            frame = buffer.tobytes()

            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')


# ROUTES
@app.route('/')
def index():
    return render_template('index.html')

@app.route('/video_feed')
def video_feed():
    return Response(generate_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')

# NEW ROUTE: To view the database in the browser
@app.route('/logs')
def view_logs():
    conn = sqlite3.connect(DB_FILE)
    cursor = conn.cursor()
    # Fetch all records, newest first
    cursor.execute("SELECT * FROM attendance_logs ORDER BY id DESC")
    records = cursor.fetchall()
    conn.close()
    
    return render_template('logs.html', records=records)

if __name__ == "__main__":
    app.run(debug=True, port=5000)