import traceback
import re
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
    if not person_path.is_dir():
        continue

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

        conn = sqlite3.connect(DB_FILE)
        cursor = conn.cursor()
        cursor.execute('''
            INSERT INTO attendance_logs (name, date, time)
            VALUES (?, ?, ?)
        ''', (name, current_date, current_time))
        conn.commit()
        conn.close()

        print(f"Attendance Logged in Database for: {name}")


# WEBCAM FRAME PROCESSING (Cloud-compatible)
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
            top, right, bottom, left = top * 4, right * 4, bottom * 4, left * 4
            cv2.rectangle(frame, (left, top), (right, bottom), (0, 255, 0), 2)
            cv2.putText(frame, name, (left, top - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (0, 255, 0), 2)

        _, buffer = cv2.imencode('.jpg', frame)
        out_b64 = base64.b64encode(buffer).decode('utf-8')

        return jsonify({'image': f"data:image/jpeg;base64,{out_b64}"})

    except Exception as e:
        print(f"Error processing frame: {e}")
        return jsonify({'error': str(e)}), 500



#  NEW: USER REGISTRATION ROUTE
@app.route('/register', methods=['POST'])
def register():
    """
    Accepts a name and an array of base64-encoded face images (minimum 5).
    Saves all valid images to dataset_extracted/<name>/ and hot-reloads
    every face encoding into memory — no server restart required.
    """
    MIN_PHOTOS = 5
    MAX_PHOTOS = 10

    try:
        payload     = request.json
        name        = payload.get('name', '').strip()
        images_data = payload.get('images', [])   # list of base64 data URLs

        # Validate name
        if not name:
            return jsonify({'success': False, 'error': 'Name cannot be empty.'}), 400

        if not re.match(r'^[\w\s\-]+$', name):
            return jsonify({'success': False,
                            'error': 'Name contains invalid characters. '
                                     'Use letters, numbers, spaces or hyphens.'}), 400

        # Validate photo count
        if not isinstance(images_data, list) or len(images_data) < MIN_PHOTOS:
            return jsonify({'success': False,
                            'error': f'Please provide at least {MIN_PHOTOS} photos '
                                     f'for accurate recognition. '
                                     f'You sent {len(images_data)}.'}), 400

        # Cap at MAX_PHOTOS (browser should enforce this too, but be safe)
        images_data = images_data[:MAX_PHOTOS]

        # Process each image
        folder_name = name.replace(' ', '_')
        person_dir  = DATASET_DIR / folder_name
        person_dir.mkdir(parents=True, exist_ok=True)

        new_encodings   = []   # encodings successfully extracted from this batch
        saved_count     = 0
        no_face_count   = 0
        multi_face_count = 0

        for idx, image_data in enumerate(images_data):
            if ',' not in image_data:
                continue

            try:
                _, encoded = image_data.split(',', 1)
                img_bytes  = base64.b64decode(encoded)
                nparr      = np.frombuffer(img_bytes, np.uint8)
                frame      = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
            except Exception:
                continue

            if frame is None:
                continue

            rgb_frame      = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            face_locations = face_recognition.face_locations(rgb_frame)

            if len(face_locations) == 0:
                no_face_count += 1
                continue
            if len(face_locations) > 1:
                multi_face_count += 1
                continue

            encoding = face_recognition.face_encodings(rgb_frame, face_locations)[0]
            new_encodings.append(encoding)

            # Save image
            timestamp  = datetime.now().strftime("%Y%m%d_%H%M%S%f")
            image_path = person_dir / f"photo_{idx:02d}_{timestamp}.jpg"
            cv2.imwrite(str(image_path), frame)
            saved_count += 1

        # Require at least MIN_PHOTOS usable face photos
        if saved_count < MIN_PHOTOS:
            # Clean up any partially-saved files
            import shutil
            if person_dir.exists() and not any(person_dir.iterdir()):
                shutil.rmtree(person_dir)

            reasons = []
            if no_face_count:
                reasons.append(f"{no_face_count} photo(s) had no detectable face")
            if multi_face_count:
                reasons.append(f"{multi_face_count} photo(s) had multiple faces")

            detail = ('. ' + '; '.join(reasons) + '.') if reasons else '.'
            return jsonify({
                'success': False,
                'error':   f'Only {saved_count} usable face photos out of '
                           f'{len(images_data)} provided{detail} '
                           f'Please retake with better lighting and only your face in frame.'
            }), 400

        # Duplicate check (compare first new encoding against known set)
        if len(known_encodings) > 0:
            distances = face_recognition.face_distance(known_encodings, new_encodings[0])
            best_idx  = np.argmin(distances)
            if distances[best_idx] < 0.5:
                existing = known_names[best_idx]
                # Remove newly saved folder since it's a duplicate
                import shutil
                shutil.rmtree(person_dir, ignore_errors=True)
                return jsonify({
                    'success': False,
                    'error':   f'This face is already registered as "{existing}".'
                }), 409

        # Hot-reload all new encodings into memory
        for enc in new_encodings:
            known_encodings.append(enc)
            known_names.append(folder_name)

        print(f"[REGISTER] '{folder_name}' registered with {saved_count} photos. "
              f"Total known face encodings: {len(known_encodings)}")

        return jsonify({
            'success': True,
            'message': f'"{name}" registered successfully with {saved_count} photos! '
                       f'Attendance will now be marked automatically.'
        })

    except Exception as e:
        traceback.print_exc()
        return jsonify({'success': False, 'error': f'Server error: {str(e)}'}), 500


# ROUTES
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
